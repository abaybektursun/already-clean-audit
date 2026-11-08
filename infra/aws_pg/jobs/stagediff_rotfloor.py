"""STAGEDIFF E2 — ROT instrument floor (remediation plan 2026-09-05 §3 E2; prefix results/stagediff/w6/rot-floor-C0/<n>).

    python -m infra.aws_pg.jobs.stagediff_rotfloor --ckpt C0 --out /work/out \
        --fit-a s3://…/w2/fit-C0a/1/out --fit-b s3://…/w2/fit-C0b/1/out
    python -m infra.aws_pg.jobs.stagediff_rotfloor --dryrun --out <dir>

The frozen ROT statistic (`stagediff_lens.rot_by_layer`: per layer, median over the fit corpus's top tokens of the
cosine between the two lenses' read-out vectors v_t = J_ℓᵀ(γ ⊙ W_U[t]); band median over layers 12..31) between the
two C0 half-lenses (32 disjoint fit prompts each) on the SAME weights and head. An identity pair should give 1; what
it gives instead is the fit noise of the instrument, and 1 − that value is the floor under which a hop's ROT reading
means nothing. Exploratory, post-unblind.
"""
from __future__ import annotations

import argparse
import os
import sys

MODULE = "infra.aws_pg.jobs.stagediff_rotfloor"


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="C0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fit-a", default=None, help="S3 prefix (…/out) of the first half-lens")
    ap.add_argument("--fit-b", default=None, help="S3 prefix (…/out) of the second half-lens")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dryrun", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    if not a.dryrun:
        from infra.aws_pg.jobs._stagediff_shared import boot
        if boot(MODULE, a.ckpt, sys.argv[1:]):
            return
    import json
    import torch
    from infra.pg import stagediff_bank as SB, stagediff_lens as SL, stagediff_models as SM
    from infra.aws_pg.jobs import _stagediff_shared as SH
    bank = SB.Bank(a.out, "stagediff_rotfloor", vars(a))
    with SH.Timer(bank, "load"):
        member = SM.stub_member(bos_id=1, seed=0) if a.dryrun else SM.load_ckpt(a.ckpt, a.device)
    lm = SL.AnatomyLensModel(member)
    band = SM.BAND if not a.dryrun else (1, 2)
    bank.manifest({"ckpt": member["id"], "repo": member["repo"], "repo_sha": member["sha"], "hashes": member["hashes"]})
    if a.dryrun:      # two lenses of the SAME stub fitted on disjoint windows — the identity pair the real cell measures
        from infra.pg.stagediff_lens import fit_corpus, fit_lens, top_tokens
        seqs = fit_corpus(member["tok"], n_seqs=6, seq_len=40, seed=0, docs=SH.fake_docs(30))["seqs"]
        lens_a, _ = fit_lens(lm, seqs[:3], dim_batch=4, ckpt_path=os.path.join(a.out, "cka.pt"))
        lens_b, _ = fit_lens(lm, seqs[3:], dim_batch=4, ckpt_path=os.path.join(a.out, "ckb.pt"))
        toks = top_tokens(seqs, k=20)
        meta = {"a": {"n_prompts": 3}, "b": {"n_prompts": 3}}
    else:
        assert a.fit_a and a.fit_b, "--fit-a and --fit-b are required"
        for side, prefix in (("a", a.fit_a), ("b", a.fit_b)):
            SH.s3_fetch(prefix, os.path.join(a.out, "inputs", side), ["lens.pt", "metrics.json"])
        lens_a = SL.load_lens(os.path.join(a.out, "inputs", "a", "lens.pt"))
        lens_b = SL.load_lens(os.path.join(a.out, "inputs", "b", "lens.pt"))
        metA = json.load(open(os.path.join(a.out, "inputs", "a", "metrics.json")))
        metB = json.load(open(os.path.join(a.out, "inputs", "b", "metrics.json")))
        toks = metA["controls"]["top_tokens"]
        assert toks == metB["controls"]["top_tokens"], "the two halves banked different top-token lists"
        meta = {"a": {"n_prompts": lens_a.n_prompts, "prefix": a.fit_a}, "b": {"n_prompts": lens_b.n_prompts, "prefix": a.fit_b}}
    SL.lens_to(lens_a, lm.device); SL.lens_to(lens_b, lm.device)
    g, W = lm.head()
    with SH.Timer(bank, "ROT_identity"):
        rot = SL.rot_by_layer(lens_a, lens_b, (g, W), (g, W), toks, layers=range(lm.n_layers))
        self_rot = SL.rot_by_layer(lens_a, lens_a, (g, W), (g, W), toks[:50], layers=list(band)[:2])   # the instrument's own ceiling
    bandv = [rot[L]["own_heads"]["point"] for L in band]
    med = sorted(bandv)[len(bandv) // 2]
    assert all(abs(v["own_heads"]["point"] - 1) < 1e-4 for v in self_rot.values()), "a lens against itself must give cosine 1"
    bank.put("readouts", "ROT_identity", {"by_layer": rot, "n_tokens": len(toks), "band": [band[0], band[-1]],
                                          "band_median": med, "band_min": min(bandv), "band_max": max(bandv),
                                          "floor_one_minus_median": 1 - med, "self_pair_ceiling_ok": True, "lenses": meta,
                                          "statistic": "stagediff_lens.rot_by_layer between fit-C0a and fit-C0b on C0's own head (identity pair)"})
    bank.top("status", "done")
    bank.event("done", out=a.out, band_median=med)
    if a.dryrun:
        assert 0 < med <= 1 + 1e-6 and len(rot) == lm.n_layers, (med, len(rot))
        print(f"PASS stagediff_rotfloor dry-run (identity-pair ROT on the stub {med:.3f}; self pair 1.000)")


if __name__ == "__main__":
    main()
