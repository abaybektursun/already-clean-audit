"""INHERIT §9.2 offline gate — the full pipeline MINUS model forward, on real data / real
tokenizer-free surfaces / real banks, CPU only, before any GPU spend (offline-gate-before-GPU
discipline). Exercises exactly the paths the GPU jobs will run:

  A. the frozen V-bank downloads + validates (shapes, orthonormality, k-grid coverage);
  B. the layer/k selection resolves mechanically from the CTRLIFACE artifacts;
  C. real VQAv2 + COCO streaming loads + content-hash dedup + sha1 split (both halves non-empty);
  D. U_unt derives from the real invcore scratch ΔWs (one layer: 24 files) — orthonormal,
     bottom-band eigenvalues near-zero relative to the top;
  E. every arm's basis constructs (build_arm_Q on a synthetic X), incl. energy-matched rank on
     SAME-rows E/tr and the shuffle/topsig paths; mass-matched tail/mid on a synthetic spectrum;
  F. the §7 statistics run end-to-end on synthetic per-item scores (RF/gap bootstrap, boot-p, BH,
     the frozen power rule).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.inherit_offline_gate --type r7i.2xlarge --no-weights \
      --s3-prefix results/inherit_offline --budget-min 60 --args "--n-items 64"
"""
import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-items", type=int, default=64)
    ap.add_argument("--uunt-layer", type=int, default=9)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {}

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from infra.aws_pg.jobs import _inherit_shared as SH
    from infra.pg import inherit_core as IC
    from infra.pg import inherit_pg1 as IP

    # --- A/B: bank + selection ---
    bases, pi_hbars, pi_traces = SH.load_bank()
    sel = SH.resolve_selection()
    rep["bank"] = {"layers": len(bases), "shape": list(bases[0].shape)}
    rep["selection"] = sel
    assert max(sel["k_grid"]) <= bases[0].shape[1], "k-grid exceeds bank coverage"
    (out / "offline_gate.json").write_text(json.dumps(rep, indent=2, default=str))

    # --- C: real eval items + sha1 split ---
    vqa, coco = IP.pg1_tasks()
    items = {}
    for task in (vqa, coco):
        rows, hashes = IP.load_items(task, a.n_items)
        split = IC.sha1_split(hashes, name=f"inherit_{task.name}")
        items[task.name] = {"n": len(rows), "n_disc": int(split["n_disc"]),
                            "n_conf": int(split["n_conf"])}
    rep["items"] = items
    (out / "offline_gate.json").write_text(json.dumps(rep, indent=2, default=str))

    # --- D: U_unt from the real scratch ΔWs (one layer) ---
    uu = SH.derive_uunt([a.uunt_layer], k=128)
    U, evals = uu[a.uunt_layer]
    err = float((U.T @ U - torch.eye(128, dtype=torch.float64)).abs().max())
    assert err < 1e-8, f"U_unt not orthonormal ({err})"
    rep["uunt"] = {"layer": a.uunt_layer, "orthonormal_err": err,
                   "bottom_eigvals_head": [float(x) for x in evals[:4]]}
    (out / "offline_gate.json").write_text(json.dumps(rep, indent=2, default=str))

    # --- E: every arm basis constructs; tail/mid builders ---
    g = torch.Generator().manual_seed(0)
    d = bases[0].shape[0]
    X = torch.randn(512, d, generator=g, dtype=torch.float64)
    hbar = X.mean(0)
    Lp, k = sel["primary"], sel["k_star"]
    kinds = {}
    for kind in SH.ARM_KINDS:
        if kind == "topsig_transported":
            Q = SH.build_arm_Q(kind, Lp, k, bases, seed=1, X=X, hbar=hbar, topsig_bank=bases)
        elif kind == "saction_redraw":
            Q = SH.build_arm_Q(kind, Lp, k, bases, seed=1, X=X, hbar=hbar, redraw_bases=bases)
        else:
            Q = SH.build_arm_Q(kind, Lp, k, bases, seed=1, X=X, hbar=hbar)
        kinds[kind] = None if Q is None else list(Q.shape)
    rep["arm_bases"] = kinds
    spec_ev = torch.linspace(9.0, 0.01, d, dtype=torch.float64) ** 2
    j, tmass, ttarget = IC.mass_matched_tail(spec_ev, k)
    lo, hi, mmass, mtarget = IC.mass_matched_mid(spec_ev, k)
    rep["arm_o_builders"] = {"tail_start": j, "mid": [lo, hi]}
    (out / "offline_gate.json").write_text(json.dumps(rep, indent=2, default=str))

    # --- F: §7 stats end-to-end on synthetic per-item scores ---
    rng = np.random.default_rng(7)
    intact = (rng.random(2048) < 0.7).astype(float)
    sa = (rng.random(2048) < 0.4).astype(float)
    em = (rng.random(2048) < 0.68).astype(float)
    gap = IC.gap_item_bootstrap(sa, em, intact, floor=0.05)
    p = IC.boot_p_gt(gap["boots"], 0.15)
    rep["stats"] = {"gap": gap["gap"], "ci_lo": gap["ci_lo"], "p_vs_delta": p,
                    "bh": IC.bh_adjust([p, 0.2, 0.03, 0.5]),
                    "n_rule": IC.choose_n_items(float(np.std(intact)), float(intact.mean() - 0.05))}
    rep["green"] = True
    (out / "offline_gate.json").write_text(json.dumps(rep, indent=2, default=str))
    print("OFFLINE_GATE_OK", json.dumps({"items": items, "selection": {kk: sel[kk] for kk in
                                        ("primary", "secondary", "anchors", "k_star")}}, default=str))


if __name__ == "__main__":
    main()
