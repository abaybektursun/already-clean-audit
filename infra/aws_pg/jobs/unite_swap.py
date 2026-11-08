"""UNITE J2, F1 block swap (studies/unite/PREREG.md §5): 16 forward arms and 7 reverse arms per hop, H4, H5, H5p, on the 54 scored wikitext windows.

    python -m infra.aws_pg.jobs.unite_swap --items <s3> --out <dir>

Parent and child resident, in place swaps of text parameters, restored after each arm. Gates: parent and child NLL within 0.002 of SINKPATCH's banked values, k = 0 reproduces the child within 1e-4,
all text parameters reproduce the parent within 1e-3, and the failing input: blocks from Molmo2-4B must not give R near 1. Per-window values go to per_item/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

MODULE = "infra.aws_pg.jobs.unite_swap"
BANKED = {"H4": (2.589747866575713, 2.7149177336553385), "H5": (2.7149177336553385, 3.0020457427171703), "H5p": (2.7149177336553385, 3.1100349621466026)}
R_NEAR_1 = 0.9


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hops", default="H4,H5,H5p")
    return ap.parse_args(argv)


def bank_arm(bank, hop: str, arm: str, rows: list, wins: list, tok, child_rows: list, parent_rows: list, m: float, total: float, kind: str = "forward"):
    """Per-window NLLs and samples for one arm. The summary R here serves the sample check; the interval is computed offline from per_item."""
    from infra.pg import unite_stats as US
    from infra.pg import unite_swap as SW
    nll = sum(r[0] for r in rows) / len(rows)
    c, p = (sum(r[0] for r in x) / len(x) for x in (child_rows, parent_rows))
    bank.per_item(f"{hop}__{arm}", {"nll_per_window": [r[0] for r in rows], "n_targets": [r[1] for r in rows], "top": [r[2] for r in rows], "top_p": [r[3] for r in rows], "mass_restored": m})
    bank.metrics["readouts"].setdefault(hop, {})[arm] = {"nll": nll, "R": US.recovery(c, p, nll), "mass_restored": m, "mass_share": m / total, "kind": kind}
    bank.write()
    for s in SW.window_samples(rows, wins, tok):
        bank.sample(hop, arm, s)


def run_hop(bank, hop: str, P: dict, C: dict, text: str, prompts: dict, blocks_foreign: dict, v_shared: int, banked=None, tol=None, window_args=None):
    from infra.pg import unite_stats as US
    from infra.pg import unite_swap as SW
    tol = tol or {"known": 0.002, "k0": 1e-4, "all": 1e-3}
    tok, an_c, n_layers = C["tok"], C["an"], len(C["an"]["layers"])
    pp, pc = SW.text_params(P), SW.text_params(C)
    assert set(pp) == set(pc), set(pp) ^ set(pc)
    C0, P0 = SW.clone(pc), SW.clone(pp)
    total = SW.mass(C0, P0, list(pc))
    wins = SW.prepared_windows(C, text, **(window_args or {}))
    r_c, r_p = SW.score(C, wins), SW.score(P, wins)
    n_c, n_p = sum(r[0] for r in r_c) / len(r_c), sum(r[0] for r in r_p) / len(r_p)
    bank_arm(bank, hop, "child", r_c, wins, tok, r_c, r_p, 0.0, total, "base")
    bank_arm(bank, hop, "parent", r_p, wins, tok, r_c, r_p, total, total, "base")
    gates = {"parent_nll": n_p, "child_nll": n_c, "text_mass_total": total, "n_windows": len(wins), "tol": tol}
    if banked is not None:
        gates.update({"banked": banked, "known_pass": abs(n_p - banked[0]) <= tol["known"] and abs(n_c - banked[1]) <= tol["known"]})

    # k = 0: the swap path with no names, and a restore after a real swap
    SW.copy_in(pc, P0, SW.names_of(pc, an_c, blocks=[n_layers - 1]))
    SW.copy_in(pc, C0, list(pc))
    r0 = SW.score(C, wins)
    gates["k0_nll"] = sum(r[0] for r in r0) / len(r0)
    gates["k0_pass"] = abs(gates["k0_nll"] - n_c) <= tol["k0"]
    SW.copy_in(pc, P0, list(pc))
    ra = SW.score(C, wins)
    gates["all_text_nll"] = sum(r[0] for r in ra) / len(ra)
    gates["all_text_pass"] = abs(gates["all_text_nll"] - n_p) <= tol["all"]
    SW.copy_in(pc, C0, list(pc))
    bank.put("controls", f"{hop}__gates", gates)

    # the failing input: every block from another model must not give R near 1
    names = SW.names_of(pc, an_c, blocks=range(n_layers))
    missing = [n for n in names if n not in blocks_foreign]
    assert not missing, missing[:3]
    SW.copy_in(pc, blocks_foreign, names)
    rf = SW.score(C, wins)
    SW.copy_in(pc, C0, list(pc))
    Rf = US.recovery(n_c, n_p, sum(r[0] for r in rf) / len(rf))
    bank.per_item(f"{hop}__failing_input", {"nll_per_window": [r[0] for r in rf]})
    bank.put("controls", f"{hop}__failing_input", {"nll": sum(r[0] for r in rf) / len(rf), "R": Rf, "line": R_NEAR_1, "failing_input_failed": Rf < R_NEAR_1})

    # forward arms
    for arm, names in SW.forward_arms(pc, an_c, n_layers).items():
        m = SW.mass(C0, P0, names)
        SW.copy_in(pc, P0, names)
        bank_arm(bank, hop, arm, SW.score(C, wins), wins, tok, r_c, r_p, m, total)
        SW.copy_in(pc, C0, names)
        bank.event("arm", hop=hop, arm=arm)

    # reverse arms: the parent with the child's parts, top-1 agreement with the child (text agreement, not robot success)
    base = {k: SW.top1(C, v, v_shared) for k, v in prompts.items()}
    rev = {}
    for arm, names in SW.reverse_arms(pp, P["an"], n_layers).items():
        SW.copy_in(pp, C0, names)
        rev[arm] = {}
        for k, v in prompts.items():
            t = SW.top1(P, v, v_shared)
            rev[arm][k] = sum(a[0] == b[0] for a, b in zip(t, base[k])) / len(v)
            for i in (0, len(v) // 2):
                bank.sample(f"{hop}_reverse_{k}", arm, {"prompt": v[i], "top_token": tok.decode([t[i][0]]), "top_p": t[i][1], "child_top_token": tok.decode([base[k][i][0]]), "child_top_p": base[k][i][1], "agrees": t[i][0] == base[k][i][0]})
        SW.copy_in(pp, P0, names)
    t = {k: SW.top1(P, v, v_shared) for k, v in prompts.items()}
    bank.metrics["readouts"].setdefault(hop, {})["reverse_top1_agreement_with_child"] = {"arms": rev, "parent_itself": {k: sum(a[0] == b[0] for a, b in zip(t[k], base[k])) / len(t[k]) for k in prompts}}
    bank.write()


def main(argv=None):
    a = parse(argv)
    from infra.aws_pg.jobs import _stagediff_shared as SH
    from infra.aws_pg.jobs._stagediff_shared import boot
    from infra.aws_pg.jobs.neurons_harvest import fetch
    from infra.pg import sinkpatch as SP, stagediff_bank as SB, stagediff_items as SI, stagediff_models as SM
    if boot(MODULE, "C4", sys.argv[1:]):
        return
    p = fetch(a.items, "j2_items.json")
    raw = pathlib.Path(p).read_bytes()
    items = json.loads(raw)
    bank = SB.Bank(a.out, "unite_swap", vars(a))
    bank.manifest({"items_sha256": hashlib.sha256(raw).hexdigest(), "libero_source": items["libero_source"], "n_libero": len(items["libero"]), "rosters": {h: SM.HOPS[h] for h in a.hops.split(",")}, "sets_sha256": SI.FILE_SHA256})
    from safetensors import safe_open
    local = SM.snapshot("C1")
    idx = json.load(open(pathlib.Path(local) / "model.safetensors.index.json"))["weight_map"]
    foreign = {}
    for name, shard in idx.items():
        if name.startswith("model.transformer.blocks."):
            with safe_open(pathlib.Path(local) / shard, "pt") as f:
                foreign[name] = f.get_tensor(name)
    bank.put("controls", "foreign_blocks", {"repo": SM.ROSTER["C1"]["repo"], "sha": SM.ROSTER["C1"]["sha"], "n_tensors": len(foreign)})
    text = SP.wikitext_text()
    for hop in a.hops.split(","):
        P_id, C_id = SM.HOPS[hop]
        P, C = SM.load_ckpt(P_id, "cuda", batch_size=1), SM.load_ckpt(C_id, "cuda", batch_size=1)
        assert P["hashes"]["prefix_vocab_sha256"] == C["hashes"]["prefix_vocab_sha256"], "shared prefix differs"
        beh = SI.beh_items(C["tok"], SI.load_sets())
        prompts = {"B1": [it["prompt"] for it in beh], "LIBERO": list(items["libero"])}
        bank.put("controls", f"{hop}__counts", {k: len(v) for k, v in prompts.items()})
        with SH.Timer(bank, hop):
            run_hop(bank, hop, P, C, text, prompts, foreign, SM.V_SHARED, banked=BANKED[hop])
        SM.free(P), SM.free(C)
    bank.top("status", "done")
    bank.event("done", out=a.out)


if __name__ == "__main__":
    main()
