"""DARWIN W2-F — the interaction matrix (§4/W2-F, §5 COMPOSITION). GPU.

Why this arm gates everything: §5 makes GRADUATED require EFFECT and ENVELOPE and COMPOSITION,
and COMPOSITION is "no interaction term in W2-F worse than -1 CI on any stack it must join".
Until W2-F runs, every Register-B arm that passes EFFECT and ENVELOPE is stuck at
NOT-GRADUATED-pending-composition — which is where W1-N's six passing cells currently sit.

SCOPE, stated rather than implied. §4/W2-F names two factorials:
  {anchor x restriction x breathing dose}                 <- W1-U x W1-T x W1-S
  {quantization codebook x highway delta x burned vector} <- W1-N x W1-D x W1-A
W1-T, W1-S and W1-A are not executed (W1-T's incumbents are unimplemented; W1-S and W1-A are not
built), so the full three-factor designs are not available. This arm therefore runs the
2x2 factorial over the two factors that ARE implemented and that W1-N's graduation actually needs:

    A  core-column fp16 protection      (W1-N arm c)      on / off
    B  highway-truncated delta applied  (W1-D, rank r)    on / off

and records the interaction with a paired bootstrap CI. The missing factors are recorded as
`factors_unavailable` on every row, so the COMPOSITION criterion this arm discharges is scoped to
the stack it actually tested — not to stacks it never saw.

Interaction, on the ppl scale (lower is better):

    I = (ppl[A+B] - ppl[A]) - (ppl[B] - ppl[none])

I ~ 0 means the two interventions compose additively. I > 0 means the stack is WORSE than the sum
of its parts, which is what §5's "-1 CI" clause is guarding against.

  pg run infra.aws_pg.jobs.darwin_w2_f --args "--arm W2-F" \\
      --type g6e.2xlarge,g6e.4xlarge,g6e.8xlarge,g6e.xlarge --no-weights ...
"""
import argparse
import json
import pathlib

from infra.pg import darwin_ceq as C
from infra.pg import darwin_core as D
from infra.pg import darwin_w0 as DW
from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as W
from infra.aws_pg.jobs.darwin_w1_n import outlier_columns_from_w0_1b
from infra.aws_pg.jobs.darwin_w1_surgery import highway_basis, truncate_to_basis

FT_REPO = "google/paligemma-3b-ft-vqav2-224"
SPEC_EXPECTATIONS = {
    "W2-F": ("no interaction term is distinguishable from zero",
             "at least one pairwise interaction term is non-zero beyond its CI"),
}
# The 2x2 design, written out. Deriving (use_A, use_B) by parsing the cell name would be one
# typo away from silently running the wrong factorial.
CELLS = {
    "none": (False, False),
    "A_core_cols": (True, False),
    "B_highway_delta": (False, True),
    "AB_both": (True, True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="W2-F", choices=("W2-F",))
    ap.add_argument("--run-id", default="darwin_w2_f")
    ap.add_argument("--s3-prefix", default="darwin/darwin_w2_f")
    ap.add_argument("--types", default="q_proj,o_proj,gate_proj,up_proj,v_proj,k_proj")
    ap.add_argument("--layers", default="")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--k", type=int, default=H.BANK_K)
    ap.add_argument("--n-seq", type=int, default=40)
    ap.add_argument("--w0-1b-prefix", default="darwin/darwin_w0_1b")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    types = [t.strip() for t in a.types.split(",") if t.strip()]
    layers = [int(x) for x in a.layers.split(",") if x.strip()] or list(range(H.N_LAYERS))
    if a.smoke:
        types, layers, a.n_seq = types[:1], layers[:2], 4

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    H.hf_setup()

    import numpy as np
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoProcessor
    from infra.aws_pg.jobs.invariant_core_func import (PG1_REPO, _decoder_layers,
                                                       _load_full_pg1, _paired_contrast_ci,
                                                       _ppl_blankimg)
    from infra.aws_pg.jobs.darwin_w1_n import _ppl_of

    utc = D.utc_now()
    exp = D.Expectations(out.parent / "expectations.jsonl")
    if not exp.by_arm(a.arm):
        refuter, direction = SPEC_EXPECTATIONS[a.arm]
        exp.log(a.arm, refuter=refuter, direction=direction, utc=utc)
    ref = exp.require(a.arm)

    api = HfApi()
    base_sha = api.model_info(PG1_REPO).sha
    ft_sha = api.model_info(FT_REPO).sha
    bank_dir = D.fetch_all_banks(H.cache_dir(out, "banks"))
    b_dir = D.fetch_prior_arm(H.cache_dir(out, "w0_1b"), a.w0_1b_prefix)
    have_wide = len(list(pathlib.Path(bank_dir).glob("*__highway_top256.npy"))) > 0

    gates = H.Gates(out)
    atlas = D.Atlas(out, arm=a.arm, run_id=a.run_id, register="GRAD", wave="W2",
                    expected_ref=ref)
    unavailable = ["W1-T restriction", "W1-S breathing", "W1-A burned vector"]
    manifest = {"program": D.PROGRAM, "arm": a.arm, "run_id": a.run_id, "register": "GRAD",
                "wave": "W2", "commit_sha": D.commit_sha(), "provenance": D.provenance(),
                "base_repo": PG1_REPO, "base_sha": base_sha, "ft_repo": FT_REPO, "ft_sha": ft_sha,
                "types": types, "layers": layers, "bits": a.bits, "rank": a.rank,
                "factors": {"A": "core-column fp16 protection (W1-N arm c)",
                            "B": f"highway-truncated delta at rank {a.rank} (W1-D)"},
                "factors_unavailable": unavailable,
                "design": "2x2 full factorial, one dose each, paired bootstrap on the interaction",
                "expected_ref": ref, "started_utc": utc, "seed": a.seed, "n_seq": a.n_seq}
    D.write_json(out, "manifest.json", manifest)

    model = _load_full_pg1(base_sha, "cuda")
    processor = AutoProcessor.from_pretrained(PG1_REPO, revision=base_sha)
    dl = _decoder_layers(model)
    mods = {}
    for t in types:
        for L in layers:
            parent = dl[L].self_attn if t.startswith(("q", "k", "v", "o")) else dl[L].mlp
            mods[(t, L)] = getattr(parent, t)
    originals = {k: m.weight.detach().clone() for k, m in mods.items()}
    w_ft = W._load_decoder(FT_REPO, ft_sha)

    def effective(t, L, use_A, use_B):
        """One cell's effective matrix. Both factors act on the SAME base weight, so the
        factorial is genuinely crossed rather than two separate experiments."""
        W0 = originals[(t, L)].to(torch.float64).cpu().numpy()
        if use_B:
            dW = w_ft[W.matrix_key(t, L)].to(torch.float64).numpy() - W0
            W0 = W0 + truncate_to_basis(dW, highway_basis(bank_dir, t, L, a.rank,
                                                          have_wide=have_wide))
        if use_A:
            oc = outlier_columns_from_w0_1b(b_dir, "pg1-pt224", L, H.SITE[t],
                                            C.matched_k_for_vault(m=W0.shape[0], n=W0.shape[1],
                                                                  k_vault=a.k))
            core_cols = DW.top_k_coords(H.leverage(W.load_bank(bank_dir, t, L, "core32")),
                                        min(len(oc), W0.shape[1]))
            return C.quantize_with_fp16_columns(W0, core_cols, a.bits)
        return C.quantize_uniform(W0, a.bits)

    per_window, ppl = {}, {}
    for cell, (use_A, use_B) in CELLS.items():
        for (t, L), mod in mods.items():
            We = effective(t, L, use_A, use_B)
            mod.weight.data.copy_(torch.as_tensor(We, dtype=mod.weight.dtype,
                                                  device=mod.weight.device))
        res = _ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)
        per_window[cell] = res["per_window"]
        ppl[cell] = float(res["ppl"])
        atlas.add("factorial_ppl", value=ppl[cell], bank_id=cell, dose=float(a.bits),
                  incumbent_id="none", model_id=PG1_REPO, model_sha=base_sha,
                  flags={"cell": cell, "factor_A_core_cols": use_A,
                         "factor_B_highway_delta": use_B, "bits": a.bits, "rank": a.rank,
                         "factors_unavailable": unavailable})
        print(f"[W2-F] {cell}: ppl={ppl[cell]:.5f}", flush=True)
        for k, mod in mods.items():
            mod.weight.data.copy_(originals[k])
        D.s3_sync(out, a.s3_prefix)

    # Interaction on the ppl scale, with a paired bootstrap over the SAME windows. Resampling the
    # window index set once and applying it to all four cells is what makes this a paired estimate
    # of the interaction rather than four independent noisy numbers subtracted.
    rng = np.random.default_rng(a.seed)
    Wn = len(per_window["none"])
    def _p(cell, idx):
        blk = [per_window[cell][i] for i in idx]
        return _ppl_of(blk)
    full = list(range(Wn))
    point = ((_p("AB_both", full) - _p("A_core_cols", full))
             - (_p("B_highway_delta", full) - _p("none", full)))
    boots = np.empty(1000)
    for i in range(1000):
        idx = rng.integers(0, Wn, Wn).tolist()
        boots[i] = ((_p("AB_both", idx) - _p("A_core_cols", idx))
                    - (_p("B_highway_delta", idx) - _p("none", idx)))
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    hw = (hi - lo) / 2.0
    atlas.add("interaction_AxB", value=float(point), bank_id="AB_both", dose=float(a.bits),
              incumbent_id="additive", ci_lo=lo, ci_hi=hi, n=Wn,
              model_id=PG1_REPO, model_sha=base_sha,
              flags={"cells": dict(ppl), "ci_halfwidth": hw,
                     "worse_than_additive": bool(point > 0),
                     # SIGN (defect D20). §5 reads "no interaction term worse than -1 CI", which
                     # is written for a HIGHER-IS-BETTER effect scale. This interaction is on the
                     # PPL scale, where lower is better, so "worse" is POSITIVE. The check is
                     # therefore I <= +hw, not I >= -hw. The first version had it backwards and
                     # passed a stack that is significantly worse than additive.
                     "composition_passes": bool(point <= hw),
                     "scale": "ppl (lower is better); worse-than-additive is POSITIVE",
                     "effect_scale_equivalent": -float(point),
                     "factors_unavailable": unavailable,
                     "note": "ppl scale, lower better; I>0 means the stack is worse than "
                             "the sum of its parts"})
    gates.record("W2F_interaction_measured", True, interaction=float(point), ci_lo=lo, ci_hi=hi,
                 ci_halfwidth=hw, cells=dict(ppl),
                 composition_passes=bool(point <= hw),
                 scale="ppl (lower better); worse-than-additive is positive",
                 factors_unavailable=unavailable)
    print(f"[W2-F] interaction AxB = {point:+.5f}  CI[{lo:+.5f},{hi:+.5f}]", flush=True)

    atlas.close()
    rep = atlas.to_parquet()
    D.write_json(out, "w2_f_summary.json",
                 {"manifest": manifest, "atlas": rep, "gates": gates.gates,
                  "ppl": ppl, "interaction": {"point": float(point), "ci_lo": lo, "ci_hi": hi},
                  "finished_utc": D.utc_now()})
    D.s3_sync(out, a.s3_prefix)
    print("=" * 78)
    print(f"DARWIN_W2_F_OK rows={rep['n_rows']} interaction={point:+.5f}")


if __name__ == "__main__":
    main()
