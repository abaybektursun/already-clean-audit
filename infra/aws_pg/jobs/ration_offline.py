"""RATION offline gate (CPU, no model forward) — spec §9 step 2 extended (feedback_offline_gate_
before_gpu): run the FULL non-forward pipeline on the REAL banked Q operator before any GPU minute.
Validates: every job module imports; carry_q_bank.pt loads with the right shapes; banks assemble from
a synthetic-but-structured Σ (planted sink at the massive coords); all 7 policy specs build on the
REAL 2048-dim Q eigenbasis at every budget and hold b̄; sink detection fires; adjudicate integrates;
the split is balanced. GPU then only validates what only GPU can (the forward).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.ration_offline --type r7i.2xlarge \
      --no-weights --s3-prefix results/ration_offline --budget-min 30
"""
import argparse
import json
import os
import pathlib
import time


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import torch

    from infra.aws_pg.jobs import _ration_shared as RS
    from infra.pg import ration_core as RC
    rep = {"object": "ration_offline_v1", "checks": []}
    t0 = time.time()

    def ok(name, **extra):
        rep["checks"].append({"name": name, "ok": True, **extra})
        (out / "offline.json").write_text(json.dumps(rep, indent=2, default=float))
        print(f"[ration-offline] {name} OK {extra}", flush=True)

    # 1. every job module imports (catches import errors + module-level NameErrors)
    import importlib
    for m in ("ration_disc", "ration_conf", "ration_layer", "ration_kv", "ration_dryrun"):
        importlib.import_module(f"infra.aws_pg.jobs.{m}")
    ok("job_modules_import", n=5)

    # 2. the REAL Q bank loads with the right shapes
    qb = RS.load_q_bank("/tmp/ration")
    d = int(qb["evecs"].shape[0])
    assert qb["evecs"].shape == (d, d), qb["evecs"].shape
    assert int(qb["evals"].shape[0]) == d, qb["evals"].shape
    k_stiff = int(qb["k_stiff"])
    # eigenvalues descending, non-negative
    ev = torch.as_tensor(qb["evals"], dtype=torch.float64)
    assert bool((ev[:-1] >= ev[1:] - 1e-6).all()), "Q evals not descending"
    assert float(ev.min()) >= -1e-6, "Q evals negative"
    ok("q_bank_load", d=d, k_stiff=k_stiff, eval_max=float(ev.max()), eval_min=float(ev.min()))

    # 3. banks from a synthetic Σ with a planted sink at the real massive coords
    gram = torch.eye(d, dtype=torch.float64)
    for c in RC.MASSIVE_COORDS:
        gram[c, c] = 5000.0                              # huge variance at the sink coords
    sig = {"gram": gram, "sum": torch.zeros(d, dtype=torch.float64),
           "abs": torch.ones(d, dtype=torch.float64), "n": 100000}
    banks = RS.build_banks(qb, sig)
    banks["sink_dirs"] = RS.detect_sink(banks)
    banks["oracle_w"] = RC.oracle_importance(np.linspace(1.0, 0.0, RC.N_ORACLE_BANDS), d)
    assert len(banks["sink_dirs"]) >= 1, "sink detection found nothing on a planted sink"
    ok("banks_assemble", n_sink=len(banks["sink_dirs"]))

    # 4. all 7 policy specs build on the REAL Q eigenbasis and hold b̄ at every budget
    per_policy = {}
    for policy in RC.POLICIES:
        for b in RC.BUDGET_GRID:
            spec = RC.make_spec(policy, b, banks)
            assert int(spec["bits"].sum()) == round(b * d), (policy, b, int(spec["bits"].sum()))
            assert spec["R"].shape == (d, d)
        per_policy[policy] = int(RC.make_spec(policy, 4, banks)["bits"].max())
    ok("policy_specs_build", policies=len(RC.POLICIES), budgets=len(RC.BUDGET_GRID))

    # 5. adjudicate integrates on a synthetic confirmation (each verdict reachable)
    pe = lambda lo, hi: {"E": (lo + hi) / 2, "ci_lo": lo, "ci_hi": hi, "se": 0.01, "n": 500}
    uni = {"E": 0.1, "se": 0.006, "ci_lo": 0.08, "ci_hi": 0.12}
    v = RC.adjudicate_ration(E_vs_sinkaware=pe(0.02, 0.05), E_vs_naive=pe(0.03, 0.06),
                             uniform_loss=uni, oracle_headroom={"E": 0.25}, delta_range=0.02,
                             random_saturated=False, mirror_mse_excess={"val": 0.4})
    assert v["verdict"] == "H_prescriptive", v
    ok("adjudicate_integration", verdict=v["verdict"])

    # 5b. the harness-math functions the JOBS call by name resolve + run (catches module-attribution
    #     bugs that py_compile + import cannot see — e.g. RC.ppl_goodness must exist on ration_core).
    pw = [[3.0 * i, 10] for i in range(1, 40)]              # synthetic per-window [Σnll, n_tok]
    g = RC.ppl_goodness(pw)
    assert len(g) == len(pw) and g[0] > g[-1], "ppl_goodness wrong"
    lc = RC.loss_ci(g, [x - 0.1 for x in g])
    assert set(lc) >= {"E", "ci_lo", "ci_hi", "se", "n"}, lc
    ok("harness_math_functions", ppl_goodness=True, loss_ci=True)

    # 6. the split is balanced on synthetic hashes
    import hashlib
    hashes = [hashlib.sha1(f"item{i}".encode()).hexdigest() for i in range(2000)]
    held = RS.ration_split(hashes, name="ration_vqa")
    frac = float(held.mean())
    assert 0.4 < frac < 0.6, f"split imbalanced: {frac}"
    ok("split_balanced", conf_frac=frac)

    rep["green"] = True
    rep["wall_seconds"] = round(time.time() - t0, 1)
    (out / "offline.json").write_text(json.dumps(rep, indent=2, default=float))
    print("RATION_OFFLINE_OK", json.dumps({"n_checks": len(rep["checks"])}))


if __name__ == "__main__":
    main()
