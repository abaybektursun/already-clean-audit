"""STAGEDIFF preflight — the known-number gate (spec §9.1): fit our lens on `Qwen/Qwen3-4B` and compare it
with the released Neuronpedia lens for the same model.

    python -m infra.aws_pg.jobs.stagediff_preflight --out /work/out --n-seqs 64 --seq-len 128 --dim-batch 32

Gate (frozen): median over layers 8..34 of the per-layer cosine between our J_ℓ and theirs ≥ 0.80 AND top-1
readout agreement on 200 held-out wikitext positions ≥ 0.70 averaged over layers 18..34. Ceiling: our lens
must agree with theirs MORE than the identity lens does at layers 18..30 (else the comparison is void: both
near identity). VOID when the released config names another base model. Exit 0 PASS, 3 FAIL, 4 VOID.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

MODULE = "infra.aws_pg.jobs.stagediff_preflight"
GATE = {"cos_median_min": 0.80, "cos_layers": (8, 34), "agree_min": 0.70, "agree_layers": (18, 34), "ceiling_layers": (18, 30)}


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-seqs", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--dim-batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dryrun", action="store_true")
    return ap.parse_args(argv)


def released_lens(out_dir: str):
    from huggingface_hub import snapshot_download
    from infra.pg.stagediff_models import RELEASED_LENS as R
    from infra.pg.stagediff_lens import load_lens
    local = snapshot_download(R["repo"], revision=R["sha"], allow_patterns=[f"{R['dir']}/*"], repo_type="model")
    cfg = open(os.path.join(local, R["dir"], "config.yaml")).read()
    m = re.search(r'^hf_model_name:\s*"?([^"\n]+)"?', cfg, re.M)
    assert m, "released config.yaml has no hf_model_name"
    lens = load_lens(os.path.join(local, R["dir"], R["file"]))
    return lens, {"hf_model_name": m.group(1).strip(), "config_yaml": cfg, "path": os.path.join(local, R["dir"], R["file"])}


def compare(lm, ours, theirs, held_ids: list, gate=GATE, layers_all=None) -> dict:
    import torch
    from infra.pg.stagediff_lens import capture, lens_logits
    n_layers = lm.n_layers
    layers_all = layers_all or list(range(n_layers - 1))
    cos = {}
    for L in layers_all:
        if L in ours.jacobians and L in theirs.jacobians:
            a, b = ours.jacobians[L].flatten(), theirs.jacobians[L].flatten()
            cos[L] = float(torch.dot(a, b) / (a.norm() * b.norm()))
    lo, hi = gate["cos_layers"]
    cos_sel = [cos[L] for L in cos if lo <= L <= hi]
    cos_median = float(torch.tensor(cos_sel).median()) if cos_sel else None
    # top-1 agreement on held-out positions
    agree, agree_id, n_pos = {}, {}, 0
    acc_o, acc_t, acc_i = {}, {}, {}
    for ids in held_ids:
        acts = capture(lm, ids)                                   # [L, S, d]
        S = acts.shape[1]
        pos = list(range(max(1, S // 2), S - 1))                  # second half of the window, not the last position
        n_pos += len(pos)
        for L in cos:
            h = acts[L, pos]
            to = lens_logits(lm, h, L, ours).argmax(-1)
            tt = lens_logits(lm, h, L, theirs).argmax(-1)
            ti = lens_logits(lm, h, L, None).argmax(-1)
            agree[L] = agree.get(L, 0) + int((to == tt).sum())
            agree_id[L] = agree_id.get(L, 0) + int((ti == tt).sum())
    agree = {L: v / n_pos for L, v in agree.items()}
    agree_id = {L: v / n_pos for L, v in agree_id.items()}
    lo, hi = gate["agree_layers"]
    sel = [agree[L] for L in agree if lo <= L <= hi]
    agree_mean = sum(sel) / len(sel) if sel else None
    lo, hi = gate["ceiling_layers"]
    ceil_ok = all(agree[L] > agree_id[L] for L in agree if lo <= L <= hi) if agree else False
    return {"cos_by_layer": cos, "cos_median_8_34": cos_median, "agree_by_layer": agree, "agree_identity_by_layer": agree_id,
            "agree_mean_18_34": agree_mean, "n_positions": n_pos, "ceiling_ours_above_identity_18_30": ceil_ok,
            "floor_pass": (cos_median is not None and cos_median >= gate["cos_median_min"] and agree_mean is not None and agree_mean >= gate["agree_min"]),
            "gate": gate}


def main(argv=None):
    a = parse(argv)
    if not a.dryrun:
        from infra.aws_pg.jobs._stagediff_shared import boot
        if boot(MODULE, "P0", sys.argv[1:]):
            return
    import torch
    from infra.pg import stagediff_bank as SB, stagediff_lens as SL, stagediff_models as SM
    from infra.aws_pg.jobs import _stagediff_shared as SH
    bank = SB.Bank(a.out, "stagediff_preflight", vars(a))
    with SH.Timer(bank, "load"):
        member = SM.stub_member(bos_id=None) if a.dryrun else SM.load_ckpt("P0", a.device)
    lm = SL.AnatomyLensModel(member)
    bank.manifest({"ckpt": "P0", "repo": member["repo"], "repo_sha": member["sha"], "hashes": member["hashes"], "released": SM.RELEASED_LENS})
    with SH.Timer(bank, "corpora"):
        if a.dryrun:
            fc = SL.fit_corpus(lm.tokenizer, n_seqs=4, seq_len=48, seed=0, docs=SH.fake_docs(40))
            held = SL.trace_corpus(lm.tokenizer, n_seqs=4, seq_len=32, seed=1, docs=SH.fake_docs(30))["seqs"]
        else:
            fc = SL.fit_corpus(lm.tokenizer, n_seqs=a.n_seqs, seq_len=a.seq_len, seed=SL.CORPUS_SEED)
            held = SL.trace_corpus(lm.tokenizer, n_seqs=20, seq_len=32, seed=SL.CORPUS_SEED)["seqs"]
    seqs = fc["seqs"][:2] if a.smoke else fc["seqs"]
    bank.put("controls", "fit_corpus", {k: v for k, v in fc.items() if k != "seqs"} | {"n_used": len(seqs)})
    with SH.Timer(bank, "fit"):
        ours, info = SL.fit_lens(lm, seqs, dim_batch=a.dim_batch, ckpt_path=os.path.join(a.out, "fit_ckpt.pt"), save_dir=None)
    ours.save(os.path.join(a.out, "lens.pt"), dtype=torch.float16)
    os.remove(os.path.join(a.out, "fit_ckpt.pt"))
    total_s = sum(s["seconds"] for s in info["stages"])
    bank.put("readouts", "fit", {**info, "seconds_total": total_s, "seconds_per_prompt": total_s / max(1, len(seqs)),
                                 "stats": SL.lens_stats(ours)})
    if a.dryrun:
        # the "released" lens is a second fit on the same stub: the harness must agree with itself
        theirs, meta = SL.fit_lens(lm, fc["seqs"][:3], dim_batch=4, ckpt_path=os.path.join(a.out, "ck2.pt"))[0], {"hf_model_name": member["repo"]}
        os.remove(os.path.join(a.out, "ck2.pt"))
        gate = dict(GATE, cos_layers=(0, 2), agree_layers=(1, 2), ceiling_layers=(1, 1))
    else:
        theirs, meta = released_lens(a.out)
        gate = GATE
    bank.put("controls", "released_lens", {"hf_model_name": meta["hf_model_name"], "n_prompts": theirs.n_prompts,
                                           "source_layers": [theirs.source_layers[0], theirs.source_layers[-1]], "d_model": theirs.d_model})
    if meta["hf_model_name"] != member["repo"]:
        bank.stamp_void("preflight", f"released lens is for {meta['hf_model_name']}, not {member['repo']}")
        bank.top("gate", {"verdict": "VOID"}); bank.top("status", "done")
        sys.exit(4)
    SL.lens_to(ours, lm.device); SL.lens_to(theirs, lm.device)
    with SH.Timer(bank, "compare"):
        cmp = compare(lm, ours, theirs, held, gate=gate, layers_all=list(range(lm.n_layers - 1)))
    verdict = "PASS" if (cmp["floor_pass"] and cmp["ceiling_ours_above_identity_18_30"]) else "FAIL"
    bank.top("gate", {**cmp, "verdict": verdict})
    bank.top("status", "done")
    bank.event("gate", verdict=verdict, cos_median=cmp["cos_median_8_34"], agree=cmp["agree_mean_18_34"], ceiling=cmp["ceiling_ours_above_identity_18_30"])
    if verdict != "PASS":
        sys.exit(3)


if __name__ == "__main__":
    main()
