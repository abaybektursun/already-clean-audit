"""CARRY scope (CPU) — §0.5 executor items 1–3's artifact half, run BEFORE the freeze:
(1) read the SPECTER banks and record exactly which metric the stiffness completion covers;
materialize Q at the frozen key (L0/dc/ppl), verify the dense materialization against the
closed-form functional on probes, eigendecompose, record k_stiff, and bank the split
(carry_q_bank.pt) every arm consumes; (2) record that per-position Rayleigh terms are absent
from the bank (the Arm-P recompute battery is thereby REQUIRED — amendment A3); (3) verify
the COMWRITE roster/SHAs/excluded pairs and the banked donor cells Gate-0 must reproduce.

CPU job (r7i): eigh of a 2048² fp64 matrix + S3 reads; no model forward.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.carry_scope --type r7i.2xlarge \
      --no-weights --s3-prefix results/carry_scope --budget-min 60
"""
import argparse
import hashlib
import json
import os
import pathlib


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch

    from infra.aws_pg.jobs import _carry_shared as CS
    from infra.pg import carry_core as CC
    from infra.pg import specter_core as SC

    rep = {"object": "carry_scope_v1", "gpu_seconds": {"total": 0.0, "note": "CPU job"}}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    wd = "/tmp/carry_scope"

    # ---- (1) the stiffness metric: coverage + the dense Q + the split -----------------
    U, lam, bulk, basis, sb, entry = CS.load_completion_env(wd, key=CS.STIFF_METRIC_KEY)
    rep["metric"] = {
        "key": CS.STIFF_METRIC_KEY,
        "n_spikes": 0 if U is None else int(U.shape[1]),
        "spike_evals": [float(x) for x in lam],
        "band_bulk": [float(b) for b in bulk],
        "n_subbands": len(sb),
        "trace_total_banked": float(entry["trace_total"]),
        "single_metric_scope": "A2: one site (L0), one mode (dc), one loss (ppl); the "
                               "residual stream is the shared coordinate system all 126 "
                               "matrices face; NO extra operator pass (journal E2)",
        "per_position_terms_banked": False,
        "arm_p_recompute_required": True,
    }
    Q = CC.materialize_q(U, lam, bulk, basis, sb)

    # materialization fidelity: dense vᵀQv must equal the closed-form functional exactly.
    gen = torch.Generator(device="cpu").manual_seed(4242)
    rels = []
    for _ in range(32):
        v = torch.randn(2048, generator=gen, dtype=torch.float64)
        v = v / v.norm()
        q_dense = float(v @ Q @ v)
        q_form = SC.spectral_completion_quadform(v, U, lam, bulk, basis, sb)
        rels.append(abs(q_dense - q_form) / max(abs(q_form), 1e-300))
    rep["materialization_max_rel_err"] = max(rels)
    assert max(rels) < 1e-8, f"dense Q disagrees with the closed form: {max(rels):.2e}"

    evals, evecs, k_stiff = CC.stiff_split(Q)
    tr = float(evals.sum())
    rep["split"] = {
        "k_stiff": int(k_stiff), "d": int(evals.numel()),
        "trace": tr,
        "trace_share_at_k": float(evals[:k_stiff].sum() / tr),
        "eval_top8": [float(x) for x in evals[:8]],
        "eval_at_k": float(evals[k_stiff - 1]),
        "eval_median": float(evals.median()),
        "chance_energy_floor_kd": k_stiff / evals.numel(),
    }
    _w()

    # the banked split (fp32 evecs; fp64 evals) + content hash.
    qb = {"metric_key": CS.STIFF_METRIC_KEY, "evals": evals, "evecs": evecs.float(),
          "k_stiff": int(k_stiff),
          "sha_ingredients": {
              "spike_evals": [float(x) for x in lam],
              "trace_total_banked": float(entry["trace_total"]),
              "band_bulk_sha": hashlib.sha256(
                  json.dumps([float(b) for b in bulk]).encode()).hexdigest()}}
    torch.save(qb, out / "carry_q_bank.pt")
    rep["q_bank_sha256"] = hashlib.sha256((out / "carry_q_bank.pt").read_bytes()).hexdigest()

    # the L14 overlay diagnostic (F-C2 only, never the split) — record availability.
    U14, lam14, bulk14, _, _, _ = CS.load_completion_env(wd, key="L14/dc/ppl")
    rep["l14_overlay"] = {"n_spikes": 0 if U14 is None else int(U14.shape[1]),
                          "spike_evals": [float(x) for x in lam14]}
    _w()

    # ---- (2) promoted spikes present + resolvable --------------------------------------
    spikes = CS.load_promoted_spikes(wd)
    rep["promoted_spikes"] = [{"key": s["key"], "idx": s["idx"], "eval": s["eval"]}
                              for s in spikes]

    # ---- (3) roster / SHAs / excluded + the Gate-0 reproduction target ------------------
    anat = CS.load_roster(wd)
    rep["roster"] = {"n": len(anat["roster"]), "k_star": anat["k_star"],
                     "base_sha": anat["base_sha"],
                     "excluded_lt": anat["excluded_lt"],
                     "slugs": [r.split("-ft-")[1].rsplit("-", 1)[0]
                               for r in anat["roster"]]}
    b_disc = json.loads(CS.fetch(f"{CS.COMWRITE_B_DISC}/vqav2/metrics.json",
                                 pathlib.Path(wd) / "b_vqav2.json").read_text())
    ident = b_disc["readouts"]["identity@0"]["donor"]["score"]
    ftc = b_disc["readouts"]["ft@0"]["donor"]["score"]
    rep["gate0_reproduction_target"] = {
        "donor": "vqav2", "half": "disc", "cell": "ft@0",
        "identity_score": ident, "ft_score": ftc, "banked_gain": ftc - ident}
    _w()

    # ---- witnesses + noise rates for Arm N(a) (recorded here so N(a) is pure-CPU) ------
    h = json.loads(CS.fetch(f"{CS.SPECTER_H}/metrics.json",
                            pathlib.Path(wd) / "specter_h.json").read_text())
    rep["arm_h_rows"] = {k: {kk: v[kk] for kk in
                             ("battery_log1p_dppl", "null_typical", "null_draws", "G",
                              "quad_pred_log1p", "Q_ratio")}
                         for k, v in h["remeasure"].items()}
    o = json.loads(CS.fetch(f"{CS.SPECTER_O}/metrics.json",
                            pathlib.Path(wd) / "specter_o.json").read_text())
    rep["ggn_gap_L0"] = o["ggn_gap"]["L0"]

    print("CARRY_SCOPE_OK", json.dumps({"k_stiff": int(k_stiff),
                                        "trace_share": rep["split"]["trace_share_at_k"],
                                        "n_spikes": rep["metric"]["n_spikes"]}))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
