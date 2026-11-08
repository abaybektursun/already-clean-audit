"""INHERIT closeout compute (CPU, weight-free) — the Arm P GEOMETRIC leg + a consolidated
synthesis JSON. overlap(S_action, U_unt) at the bite layers vs a matched-random null band with
rejection room (the CTRLIFACE Axis-2 lesson). Raw frame (the RMSNorm-pullback frame needs γ and is
documented-not-run — the Arm-P CAUSAL leg already settles H_separate). Uses the frozen bank
(S_action) + Gate-0's banked U_unt (uunt_selected.pt) — no model, no scratch download.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.inherit_closeout --type r7i.2xlarge --no-weights \
      --s3-prefix results/inherit_closeout --budget-min 20 \
      --args "--gate0-s3 s3://.../inherit_gate0/out/uunt_selected.pt"
"""
import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate0-uunt-s3", required=True)
    ap.add_argument("--k", type=int, default=16)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    import torch
    from infra.aws_pg.jobs import _inherit_shared as SH
    from infra.pg import ctrliface_core as CC
    from infra.pg import inherit_core as IC

    bases, _h, _t = SH.load_bank()
    uunt = torch.load(SH.fetch(a.gate0_uunt_s3, "/tmp/inherit/uunt_selected.pt"),
                      map_location="cpu", weights_only=False)      # {L: U (2048,128)}
    d = bases[0].shape[0]
    rep = {"k": a.k, "arm_p_geometric": {}}
    for L in sorted(int(x) for x in uunt):
        Sa = IC.slice_k(bases[L], a.k)
        U = torch.as_tensor(uunt[L], dtype=torch.float64)[:, :a.k]
        ov, cos = CC.principal_angle_overlap(CC.orthonormalize(Sa), CC.orthonormalize(U))
        band = IC.overlap_null_band(CC.orthonormalize(Sa), d, a.k, n_draws=200, seed=13)
        rep["arm_p_geometric"][str(L)] = {
            "overlap_raw": float(ov), "null_mean": band["mean"], "null_p95": band["p95"],
            "null_spread": band["spread"], "beyond_band": IC.beyond_band(ov, band),
            "top_cos": [float(x) for x in cos[:4]]}
    (out / "closeout.json").write_text(json.dumps(rep, indent=2, default=str))
    print("CLOSEOUT_OK", json.dumps({L: {"ov": round(v["overlap_raw"], 4),
                                         "null_p95": round(v["null_p95"], 4),
                                         "beyond": v["beyond_band"]}
                                     for L, v in rep["arm_p_geometric"].items()}))


if __name__ == "__main__":
    main()
