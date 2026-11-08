"""SPECTER Arm M unblind (CPU, ZERO forwards) — completes the R-M1 statistics from the sealed
battery's banked measurements. The battery itself ran once and touched the sealed half once
(20 scored + 1 void, all rows durable in results/specter_m_sealed/out/metrics.json); its
in-job unblinding crashed on a REACHABLE estimator corner the parent never hit: M0's vqa link
is FLAT over the selected cells' energy range, so its prediction vector is all-tied and
`atlas_fit.spearman` (fail-loud) asserts. Frozen completion of the estimator's domain,
disclosed (journal E13): ρ(x, y) := 0 when either vector is all-tied — a constant predictor
carries zero ordering information; Δρ and the pair bootstrap use the same convention.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.specter_m_unblind --type r7i.2xlarge --no-weights \
      --s3-prefix results/specter_m_unblind --budget-min 30
"""
import argparse
import json
import os
import pathlib


def rho_or_zero(a, b):
    import numpy as np

    from infra.pg.atlas_fit import spearman
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if np.unique(a).size < 2 or np.unique(b).size < 2:
        return 0.0
    return spearman(a, b)


def drho_bootstrap(pred_a, pred_b, y, n_boot=10000, seed=1337):
    import numpy as np
    pa, pb, yy = map(lambda v: __import__("numpy").asarray(v, float), (pred_a, pred_b, y))
    rng = np.random.default_rng(seed)
    obs = rho_or_zero(pa, yy) - rho_or_zero(pb, yy)
    boots = []
    n = len(yy)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.unique(yy[idx]).size < 3:
            continue
        boots.append(rho_or_zero(pa[idx], yy[idx]) - rho_or_zero(pb[idx], yy[idx]))
    b = np.asarray(boots)
    return (float(obs), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)),
            rho_or_zero(pa, yy), rho_or_zero(pb, yy))


def perm_p_rho(a, y, n_perm=10000, seed=1337):
    import numpy as np
    a = np.asarray(a, float)
    y = np.asarray(y, float)
    obs = rho_or_zero(a, y)
    rng = np.random.default_rng(seed)
    cnt = sum(rho_or_zero(a, y[rng.permutation(len(y))]) >= obs - 1e-12
              for _ in range(n_perm))
    return obs, (cnt + 1) / (n_perm + 1)


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np

    from infra.aws_pg.jobs import _specter_shared as SS
    from infra.pg import specter_core as SC

    freeze = SS.assert_frozen("m_sealed")          # same arm contract as the battery
    rep = {"object": "specter_m_unblind_v1", "freeze": freeze,
           "battery_source": "results/specter_m_sealed/out/metrics.json (touched-once rows)"}
    tm = SS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    with tm.stage("unblind"):
        ms = json.loads(SS.fetch(f"{SS.S3_BUCKET}/results/specter_m_sealed/out/metrics.json",
                                 "/tmp/specter/ms.json").read_text())
        scored = ms["scored"]
        voids = ms.get("voids", [])
        assert len(scored) == 20, f"expected 20 scored sealed cells, got {len(scored)}"
        rep["scored_n"] = len(scored)
        rep["n_voids"] = len(voids)
        res, pvals, labels = {}, [], []
        for ro, mkey in (("ppl", "measured_log1p_dppl"), ("vqa", "measured_vqa_effect")):
            y = np.array([s[mkey] for s in scored], float)
            m4 = np.array([s["pred"][f"M4_{ro}"] for s in scored], float)
            m0 = np.array([s["pred"][f"M0_{ro}"] for s in scored], float)
            drho, lo, hi, r4, r0 = drho_bootstrap(m4, m0, y)
            _, p4 = perm_p_rho(m4, y)
            pvals.append(p4)
            labels.append(ro)
            res[ro] = {"drho": drho, "ci": [lo, hi], "rho_m4": r4, "rho_m0": r0,
                       "m0_degenerate_all_tied": bool(np.unique(m0).size < 2),
                       "p_m4_orders": p4,
                       "adjudication": SC.adjudicate_rm1(
                           drho=drho, drho_cilo=lo, drho_cihi=hi, rho_m4=r4, rho_m0=r0,
                           n_void_backups=len(voids))}
            print(f"[m] {ro}: drho={drho:+.3f} [{lo:+.3f},{hi:+.3f}] rho_m4={r4:.3f} "
                  f"rho_m0={r0:.3f} -> {res[ro]['adjudication']['verdict']}", flush=True)
        SS.emit_family(rep, "rm1", pvals, labels)
        rep["rm1"] = res
    _w()

    SS.assert_gpu_seconds(rep)
    print("SPECTER_M_UNBLIND_OK", json.dumps({ro: res[ro]["adjudication"]["verdict"]
                                              for ro in res}))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
