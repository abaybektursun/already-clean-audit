"""RELAW offline gate (prereg §9.2) — the real-data, NO-model analysis-pipeline check.

The RELAW-specific offline gate proves the LAW-FIT pipeline reproduces a KNOWN banked number on
real data (CPU, `pg --no-weights`, NO model forward): it re-runs the SAV-C within-family ΔLOO-R²
of the activation gauge R_ovl vs the weight-distance baseline l2_base on S16 from the FROZEN
SAV-C pool, and asserts it matches the banked SAV-C `within_family` analysis (the
H_between-regime-only fingerprint) within tolerance — the "recompute a known number" gate for the
analysis wiring R_prin will ride on. Also checks the generation-firewall layer counter and the
pool sanity (G/overlap/l2 present + in range). The MMLU DATA wiring for Arm-1/Arm-3 (degenerate
filter, split, ladder, cloze boundary) is the REUSED `latmech_offline` gate, run alongside.

    python -m infra.aws_pg.jobs.relaw_offline --out <dir> \
        --pool studies/lineage/paligemma1-pi0/results/savc/savc_metrics.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", default="studies/lineage/paligemma1-pi0/results/savc/savc_metrics.json")
    ap.add_argument("--n-perm", type=int, default=2000)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from infra.pg import relaw as RL
    from infra.pg import savb as SB

    pool_path = a.pool
    if pool_path.startswith("s3://"):
        pool_path = str(out / "savc_metrics.json")
        subprocess.run(["aws", "s3", "cp", a.pool, pool_path, "--region", "us-east-1"], check=True)
    savc = json.loads(pathlib.Path(pool_path).read_text())
    pool = savc["pool"]
    rep = {"n_pool": len(pool)}

    # 1. pool sanity — real ground-truth present + in range.
    assert len(pool) >= 16, f"SAV-C pool too small: {len(pool)}"
    for u in pool:
        assert 0.0 <= u["overlap"] <= 1.0, f"{u['unit']} overlap out of [0,1]: {u['overlap']}"
        assert np.isfinite(u["S_16k"]) and u["l2_base"] > 0, f"{u['unit']} bad G/l2"
    rep["pool_sanity_ok"] = True

    # 2. reproduce the banked SAV-C within-family ΔLOO-R²(overlap | l2,lmc) on S16 (known-number gate).
    cut = savc["within_cut"]
    wf = [u for u in pool if u["overlap"] > cut]
    y = np.array([u["S_16k"] for u in wf], float)
    ov = np.array([u["overlap"] for u in wf], float)
    Xb = np.column_stack([[u["l2_base"] for u in wf], [u["lmc_barrier"] for u in wf]])
    recomputed = SB.delta_loo_perm(Xb, ov, y, n_perm=a.n_perm, seed=1337)
    banked = savc["analysis"]["within_family"]["S_16k"]["loo_overlap"]
    rep["within_family_recompute"] = {"n": len(wf), "recomputed_delta": recomputed["delta"],
                                      "banked_delta": banked["delta"],
                                      "recomputed_p": recomputed["p_perm"], "banked_p": banked["p_perm"]}
    assert abs(recomputed["delta"] - banked["delta"]) < 1e-6, \
        f"within-family ΔLOO-R² drift: recomputed {recomputed['delta']} vs banked {banked['delta']}"
    rep["known_number_reproduced"] = True
    assert savc["verdict"]["named"] == "H_between-regime-only", \
        f"SAV-C pool fingerprint changed: {savc['verdict']['named']}"

    # 3. generation firewall — the layer counter discriminates 18 (G1) vs 26 (G2).
    g1 = [f"model.layers.{i}.self_attn.q_proj.weight" for i in range(18)]
    g2 = [f"model.layers.{i}.mlp.gate_proj.weight" for i in range(26)]
    assert RL.n_decoder_layers(g1) == 18 and RL.n_decoder_layers(g2) == 26
    rep["firewall_ok"] = True

    rep["OFFLINE_OK"] = True
    (out / "relaw_offline.json").write_text(json.dumps(rep, indent=1, default=str))
    print(f"[RELAW-OFFLINE] OK — pool n={len(pool)}, within-family ΔLOO-R² reproduced "
          f"({recomputed['delta']:+.4f} == banked {banked['delta']:+.4f}), firewall 18/26 ok")
    print("RELAW_OFFLINE_OK")


if __name__ == "__main__":
    main()
