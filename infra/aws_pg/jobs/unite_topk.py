"""UNITE J5, F2 top-k against tail (studies/unite/PREREG.md §6). Hops H5 and H5p in full, H4 'both' only.

    python -m infra.aws_pg.jobs.unite_topk --items <s3> --out <dir>

Before any forward: the mass table. Arms: top k and tail, each on input rows, output column, both; the singular-value control at matched mass on both sides; all-MLP and all-attention ceilings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

from infra.aws_pg.jobs.unite_swap import bank_arm

MODULE = "infra.aws_pg.jobs.unite_topk"


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hops", default="H5,H5p,H4")
    return ap.parse_args(argv)


def run_hop(bank, hop: str, P: dict, C: dict, text: str, ks, full: bool, window_args=None):
    from infra.pg import unite_swap as SW
    tok, an, n_layers = C["tok"], C["an"], len(C["an"]["layers"])
    pp, pc = SW.text_params(P), SW.text_params(C)
    C0, P0 = SW.clone(pc), SW.clone(pp)
    total = SW.mass(C0, P0, list(pc))
    m_in, m_out = SW.neuron_mass(C0, P0, an, n_layers)
    W = m_in.shape[1]
    table = SW.mass_table(m_in, m_out, ks)
    report = {"n_neurons": table["n_neurons"], "mlp_mass": table["total_mass"], "text_mass": total,
              "k": {str(k): {x: v[x] for x in v if x not in ("top", "tail")} for k, v in table["k"].items()}}
    bank.put("controls", f"{hop}__mass_table", report)
    wins = SW.prepared_windows(C, text, **(window_args or {}))
    r_c, r_p = SW.score(C, wins), SW.score(P, wins)
    bank_arm(bank, hop, "child", r_c, wins, tok, r_c, r_p, 0.0, total, "base")
    bank_arm(bank, hop, "parent", r_p, wins, tok, r_c, r_p, total, total, "base")

    def arm(name, names=None, apply=None, m=None, kind="f2"):
        if names is not None:
            m = SW.mass(C0, P0, names)
            SW.copy_in(pc, P0, names)
        if apply is not None:
            m = apply()
        realized = SW.mass(pc, C0, list(pc))
        bank_arm(bank, hop, name, SW.score(C, wins), wins, tok, r_c, r_p, realized, total, kind)
        bank.metrics["readouts"][hop][name]["mass_intended"] = m
        bank.write()
        SW.copy_in(pc, C0, list(pc))
        bank.event("arm", hop=hop, arm=name)

    arm("all_mlp", names=SW.mlp_names(pc), kind="ceiling")
    arm("all_attention", names=SW.attn_names(pc), kind="ceiling")
    sides = ("input", "output", "both") if full else ("both",)
    plan = SW.sv_plan(C0, P0, an, n_layers) if full else None
    for k in ks:
        rec = table["k"][k]
        if not rec["tail_exists"]:
            bank.event("k_skipped", hop=hop, k=k, reason="no least-changed set within 5% of the top-k mass")
        for side in sides:
            def top(side=side, rec=rec):
                SW.restore_neurons(pc, P0, an, rec["top"], W, side)
            arm(f"top{k}_{side}", apply=top)
            if rec["tail_exists"]:
                def tail(side=side, rec=rec):
                    SW.restore_neurons(pc, P0, an, rec["tail"], W, side)
                arm(f"tail{k}_{side}", apply=tail)
        if full:
            out = {}

            def sv(rec=rec, out=out):
                out.update(SW.restore_sv(pc, plan, rec["top_mass"]))
                return out["realized_mass"]
            arm(f"sv{k}_both", apply=sv, kind="sv_control")
            bank.metrics["readouts"][hop][f"sv{k}_both"].update(out)
            bank.write()
    del plan


def main(argv=None):
    a = parse(argv)
    from infra.aws_pg.jobs._stagediff_shared import boot
    from infra.aws_pg.jobs.neurons_harvest import fetch
    from infra.pg import sinkpatch as SP, stagediff_bank as SB, stagediff_models as SM
    if boot(MODULE, "C4", sys.argv[1:]):
        return
    raw = pathlib.Path(fetch(a.items, "j5_items.json")).read_bytes()
    items = json.loads(raw)
    bank = SB.Bank(a.out, "unite_topk", vars(a))
    bank.manifest({"items_sha256": hashlib.sha256(raw).hexdigest(), "k": items["k"], "hops": a.hops})
    text = SP.wikitext_text()
    for hop in a.hops.split(","):
        P_id, C_id = SM.HOPS[hop]
        P, C = SM.load_ckpt(P_id, "cuda", batch_size=1), SM.load_ckpt(C_id, "cuda", batch_size=1)
        assert P["hashes"]["prefix_vocab_sha256"] == C["hashes"]["prefix_vocab_sha256"], "shared prefix differs"
        run_hop(bank, hop, P, C, text, items["k"], hop != "H4")
        SM.free(P), SM.free(C)
    bank.top("status", "done")
    bank.event("done", out=a.out)


if __name__ == "__main__":
    main()
