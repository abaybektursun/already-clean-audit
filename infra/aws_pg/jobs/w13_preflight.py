"""W13 §9-2 PREFLIGHT — synthetic-control validation of the two decomposition arms.

The blind-protocol step that MUST pass before any real-ΔW decomposition or behavioral graft
(prereg §9 step 2 "develop on control data"; §5 controls; §9-1c identity). It validates the
INSTRUMENT on planted structure — it does NOT touch real weights or the GPU behavioral graft
(those are Tier-3, a separate `--type g6e` job after the e4 graft path is ported). CPU-only,
numpy/torch, no sklearn, no torch.cuda → route to the r-family Standard quota:

    ./infra/aws_pg/pg run infra.aws_pg.jobs.w13_preflight --type r7i.2xlarge --no-weights \
        --s3-prefix results/w13_preflight --budget-min 20

Gates (HALT — raise, STATUS exit=1 — on any failure):
  G1  fp64 SVD reconstruction identity  < 1e-10                              (§9-1c)
  G2  low-rank arm recovers a planted rank-r0 (N(τ)=r0)                      (pos-control #1a)
  G3  sparse arm beats SVD on the adversarial plant (full SVD-rank but       (pos-control #1b/#2)
      s0-sparse over an overcomplete dict): SVD needs ~full rank, sparse
      arm recovers it at ~s0 per-row sparsity AND recovers the planted atoms
      (atom-recovery ≥ thr — the "method can detect a basis that IS present"
      HALT criterion that makes a null N≈eff-rank evidence, not impotence)
  G4  shuffle collapses the sparse structure (residual blows up at matched   (§5 shuffle)
      sparsity) — the sparse signal is not a value-distribution artifact

The gate is STRUCTURE-RECOVERY (does the arm recover the planted basis? is the sparse/SVD
component-count separation real?), NOT Frobenius reconstruction — gating a count on Frobenius is
the exact trap the prereg forbids (§6, §8: faithfulness is behavioral/structural, never
‖ΔW−Σcomp‖, because rank-8 closes ~58% KL at ~0% capability). Frobenius residual is REPORTED as a
diagnostic only. The binding behavioral capability gate (`R_cap`, §6A) is the Tier-3 GPU step.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import torch

from infra.pg import w13_decomp as W


# --- frozen control config (the synthetic regime; thresholds are the gate's, not the verdict's) ---
CFG = dict(
    seed=0,
    identity_shape=(96, 64),       # rectangular MLP-like matrix for the fp64 identity
    lowrank=dict(m=64, n=64, r0=8, energy_target=0.9999),
    sparse=dict(n_rows=2048, d=64, n_atoms0=96, s0=3,  # n_atoms0 > d, n_rows > d ⇒ SVD-rank = d (full)
                lam_frac=0.10, n_outer=60, n_iter=200,
                svd_energy_target=0.999),              # SVD-full read at high fidelity (energy concentrates)
)
THR = dict(
    identity_max=1e-10,
    sparse_rowsparsity_max=6.0,         # ≤ 2·s0 active atoms per row (s0=3) — sparse code, ≪ d
    svd_full_frac_min=0.90,             # at 0.999 fidelity the full-SVD-rank plant needs ≥0.9·d comps
    atom_recovery_min=0.80,             # ≥80% planted atoms recovered |cos|≥0.9 — the #2 HALT criterion
    shuffle_rowsparsity_ratio_min=2.0,  # shuffling must force a ≥2× denser code (structure destroyed)
)


def _gate(name, ok, detail):
    print(f"[W13-PRE] {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    return {"gate": name, "ok": bool(ok), "detail": detail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny/fast sizes for the wiring smoke test")
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)

    assert not torch.cuda.is_available() or True  # CPU job; never call .cuda below
    torch.set_default_dtype(torch.float64)
    cfg = json.loads(json.dumps(CFG))
    if args.smoke:
        cfg["sparse"].update(n_rows=512, n_outer=30, n_iter=120)
    print(f"[W13-PRE] torch={torch.__version__} smoke={args.smoke} cfg={json.dumps(cfg)}")

    gates = []

    # ---- G1: fp64 reconstruction identity (§9-1c) -------------------------------------------
    g = torch.Generator().manual_seed(cfg["seed"])
    Wid = torch.randn(*cfg["identity_shape"], generator=g, dtype=torch.float64)
    resid = W.recon_identity(Wid)
    gates.append(_gate("G1_fp64_identity", resid < THR["identity_max"],
                       f"recon_residual={resid:.2e} (<{THR['identity_max']:.0e})"))

    # ---- G2: low-rank arm recovers planted rank-r0 (pos-control #1a) -------------------------
    lr = cfg["lowrank"]
    Wlr = W.plant_lowrank(lr["m"], lr["n"], lr["r0"], cfg["seed"])
    n_lr, _ = W.lowrank_n(Wlr, lr["energy_target"])
    gates.append(_gate("G2_lowrank_recovery", n_lr == lr["r0"],
                       f"N(τ={lr['energy_target']})={n_lr}  planted r0={lr['r0']}"))

    # ---- G3: sparse arm beats SVD on the adversarial full-rank-but-sparse plant (#1b/#2) -----
    sp = cfg["sparse"]
    Xsp, D0, A0 = W.plant_sparse_overcomplete(sp["n_rows"], sp["d"], sp["n_atoms0"], sp["s0"], cfg["seed"])
    n_svd, _ = W.lowrank_n(Xsp, sp["svd_energy_target"])       # SVD view of the SAME matrix (high fidelity)
    svd_full_frac = n_svd / sp["d"]
    D, A = W.dict_learn(Xsp, sp["n_atoms0"], sp["lam_frac"], sp["n_outer"], sp["n_iter"], cfg["seed"])
    sp_resid = W.recon_residual(Xsp, A, D)             # diagnostic only (recovery-limited), NOT gated
    sp_rowsp = W.row_sparsity(A)
    sp_nact = W.active_atoms(A)
    recov = W.atom_recovery(D, D0)
    g3_ok = (svd_full_frac >= THR["svd_full_frac_min"]
             and sp_rowsp <= THR["sparse_rowsparsity_max"]
             and recov >= THR["atom_recovery_min"])
    gates.append(_gate("G3_sparse_vs_svd", g3_ok,
                       f"SVD N={n_svd}/{sp['d']} (frac={svd_full_frac:.2f}≥{THR['svd_full_frac_min']}) vs "
                       f"sparse atom_recovery={recov:.2f}≥{THR['atom_recovery_min']} (the #2 HALT), "
                       f"rowsparsity={sp_rowsp:.1f}≤{THR['sparse_rowsparsity_max']} (s0={sp['s0']}), "
                       f"n_active={sp_nact}; recon_resid={sp_resid:.3f} (diagnostic — Frobenius NOT gated, §6)"))

    # ---- G4: shuffle collapses the sparse structure (§5) — code must turn DENSE -------------
    Xsh = W.shuffle_entries(Xsp, cfg["seed"] + 1)
    Dsh, Ash = W.dict_learn(Xsh, sp["n_atoms0"], sp["lam_frac"], sp["n_outer"], sp["n_iter"], cfg["seed"])
    sh_resid = W.recon_residual(Xsh, Ash, Dsh)
    sh_rowsp = W.row_sparsity(Ash)
    rowsp_ratio = sh_rowsp / max(sp_rowsp, 1e-9)
    g4_ok = rowsp_ratio >= THR["shuffle_rowsparsity_ratio_min"] and sh_resid > sp_resid
    gates.append(_gate("G4_shuffle_collapse", g4_ok,
                       f"shuffle rowsparsity={sh_rowsp:.1f} vs real {sp_rowsp:.1f} → {rowsp_ratio:.1f}×"
                       f"≥{THR['shuffle_rowsparsity_ratio_min']}; shuffle resid={sh_resid:.3f} > real "
                       f"{sp_resid:.3f} (no sparse structure survives permutation)"))

    all_ok = all(x["ok"] for x in gates)
    result = {
        "torch": torch.__version__, "smoke": args.smoke, "cfg": cfg, "thresholds": THR,
        "gates": gates,
        "measurements": {
            "g1_recon_residual": resid,
            "g2_lowrank_N": n_lr,
            "g3_svd_N": n_svd, "g3_svd_full_frac": svd_full_frac,
            "g3_sparse_resid": sp_resid, "g3_sparse_rowsparsity": sp_rowsp,
            "g3_sparse_n_active": sp_nact, "g3_atom_recovery": recov,
            "g4_shuffle_resid": sh_resid, "g4_shuffle_rowsparsity": sh_rowsp,
            "g4_rowsparsity_ratio": rowsp_ratio,
        },
        "PREFLIGHT_OK": bool(all_ok),
    }
    (out / "controls.json").write_text(json.dumps(result, indent=2))
    print(f"[W13-PRE] PREFLIGHT_OK={all_ok}  ({sum(x['ok'] for x in gates)}/{len(gates)} gates passed)")
    if not all_ok:
        failed = [x["gate"] for x in gates if not x["ok"]]
        raise RuntimeError(
            f"PREFLIGHT HALT: gate(s) {failed} failed. The decomposition instrument cannot detect "
            f"the planted structure in the regime the verdict turns on — a null N≈eff-rank would be "
            f"method impotence, not evidence (prereg §5). Fix the arm before touching real ΔW.")
    print("[W13-PRE] gate passed → instrument validated; safe to build the Tier-3 behavioral graft.")


if __name__ == "__main__":
    main()
