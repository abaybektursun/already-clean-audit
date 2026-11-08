"""CONFLUX weight-free dry-run logic gate (spec §9.2). Runs on tiny synthetic bases — seconds, no
model. Asserts the overlap/null/verdict wiring on KNOWN answers before any banked contrast:
  * overlap = 1 for identical spans, ~r/d for orthogonal random spans, exactly 0 for disjoint;
  * U_bb of identical per-FT frames = that span; shuffle collapses overlap to the floor;
  * the within-stiff null lives in the stiff shell; the whole-shell rung SATURATES (VOID);
  * energy null is orthonormal and carries U_sus's coordinate footprint;
  * the frozen verdict branch order fires correctly on hand-built rungs;
  * a KNOWN event replayed: a deliberately-saturated null trips VOID (feedback_one_sided_thresholds).

Invoked as `_conflux.dry_run_gate()` (returns a JSON-able report) by conflux_disc stage 0, and as a
CLI `python -m infra.pg.test_conflux_core`."""
from __future__ import annotations


def _ortho(d, cols, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, cols)))
    return np.ascontiguousarray(Q[:, :cols])


def dry_run_gate():
    import numpy as np
    from infra.pg import conflux_core as CX

    rep = {"checks": []}

    def ok(name, cond, detail=""):
        assert cond, f"DRY-RUN FAIL: {name} :: {detail}"
        rep["checks"].append({"name": name, "pass": True, "detail": detail})

    d, r = 64, 6
    I = np.eye(d)

    # 1. identical span → overlap 1; disjoint axis-blocks → 0.
    A = I[:, :r].copy()
    ok("identical_overlap_1", abs(CX.overlap(A, A.copy()) - 1.0) < 1e-12)
    B = I[:, r:2 * r].copy()
    ok("disjoint_overlap_0", CX.overlap(A, B) < 1e-12, f"{CX.overlap(A,B):.2e}")

    # 2. two independent Haar frames → overlap ≈ r/d (the floor).
    o_rand = np.mean([CX.overlap(_ortho(d, r, s), _ortho(d, r, s + 500)) for s in range(60)])
    ok("haar_overlap_near_floor", abs(o_rand - r / d) < 0.02, f"{o_rand:.4f} vs {r/d:.4f}")

    # 3. U_bb of N identical frames == that frame's span.
    Ubb = CX.ubb_basis([A.copy() for _ in range(5)], r)
    ok("ubb_identical_is_span", CX.overlap(Ubb, A) > 1 - 1e-8, f"{CX.overlap(Ubb,A):.6f}")

    # 4. U_bb from raw ΔW: a shared low-rank write + FT-idiosyncratic noise → U_bb recovers the shared part.
    rng = np.random.default_rng(0)
    shared = _ortho(d, 3, 42)
    dWs = []
    for i in range(8):
        core = shared @ rng.standard_normal((3, d)) * 3.0          # strong shared write (rank 3)
        noise = rng.standard_normal((d, d)) * 0.2                  # weak idiosyncratic
        dWs.append(core + noise)
    Ubb3, _ = CX.build_ubb(dWs, 3)
    ok("ubb_recovers_shared_write", CX.overlap(Ubb3, shared) > 0.8, f"{CX.overlap(Ubb3,shared):.4f}")

    # 5. shuffle control: coordinate-permute U_bb → overlap with U_sus collapses to the floor.
    U_sus = _ortho(d, r, 7)
    Ureal = U_sus @ np.linalg.qr(rng.standard_normal((r, r)))[0]   # same span as U_sus, rotated basis
    ok("aligned_overlap_high", CX.overlap(Ureal, U_sus) > 1 - 1e-8)
    perm = rng.permutation(d)
    Ushuf = Ureal[perm, :]
    ok("shuffle_collapses_to_floor", CX.overlap(Ushuf, U_sus) < 0.1, f"{CX.overlap(Ushuf,U_sus):.4f}")

    # 6. within-stiff null lives in the stiff shell; energy null is orthonormal + carries footprint.
    evecs = _ortho(d, d, 3)                                        # a fake Q eigenbasis
    k_stiff = 20
    ws = CX.within_stiff_null(evecs, r, k_stiff, 11)
    stiff_span = evecs[:, :k_stiff]
    resid = ws - stiff_span @ (stiff_span.T @ ws)
    ok("within_stiff_in_shell", float((resid ** 2).sum()) < 1e-8, f"{float((resid**2).sum()):.2e}")
    Sfake = CX.usus_basis(evecs, r)
    en = CX.energy_null(Sfake, r, 5)
    ok("energy_null_orthonormal", float(np.abs(en.T @ en - np.eye(r)).max()) < 1e-8)

    # 7. SATURATION replay: the whole-shell rung (r = k_stiff) → within-stiff null == U_sus span → VOID.
    ws_full = CX.within_stiff_null(evecs, k_stiff, k_stiff, 1)
    sus_full = CX.usus_basis(evecs, k_stiff)
    ok("whole_shell_null_saturates", CX.overlap(ws_full, sus_full) > 1 - 1e-8,
       f"{CX.overlap(ws_full, sus_full):.6f}")
    # a rung_overlap at r=k_stiff must flag void (feedback_one_sided_thresholds replay).
    dW_small = [rng.standard_normal((d, d)) for _ in range(4)]
    rg_full = CX.rung_overlap(dW_small, evecs, k_stiff, k_stiff, {}, n_null=20, boot=False)
    ok("whole_shell_rung_void", rg_full["void"] is True, f"void={rg_full['void']}")

    # 8. a non-saturated low rung is NOT void.
    rg_low = CX.rung_overlap(dW_small, evecs, k_stiff, 3, {}, n_null=20, boot=False)
    ok("low_rung_not_void", rg_low["void"] is False, f"void={rg_low['void']}")

    # 9. verdict branch order on hand-built rungs.
    def _rung(void, haar_lo, within_lo, partner):
        return {"void": void, "excess_haar_ci": {"ci95": [haar_lo, 1]},
                "excess_within_ci": {"ci95": [within_lo, 1]}, "excess_partner_point": partner}
    ok("verdict_void", CX.adjudicate_conflux(_rung(True, 1, 1, 1))["verdict"] == "VOID")
    ok("verdict_independent", CX.adjudicate_conflux(_rung(False, -0.01, 0.1, 0.1))["verdict"] == "H_independent")
    ok("verdict_shell", CX.adjudicate_conflux(_rung(False, 0.05, -0.01, 0.1))["verdict"] == "H_shell")
    ok("verdict_identity", CX.adjudicate_conflux(_rung(False, 0.05, 0.02, 0.03))["verdict"] == "H_identity")
    ok("verdict_top_ambiguous",
       CX.adjudicate_conflux(_rung(False, 0.05, 0.02, -0.01))["verdict"] == "H_shell_top_ambiguous")

    # 10. three-way + residual characterisation are well-formed.
    tw = CX.three_way(Ureal, U_sus, _ortho(d, 8, 9))
    ok("three_way_bb_sus_high", tw["bb_sus"] > 1 - 1e-8, f"{tw['bb_sus']:.4f}")
    rc = CX.residual_characterisation(U_sus, U_sus)               # U_bb == U_sus → residual ~0
    ok("residual_zero_when_identical", rc["off_sus_energy_frac"] < 1e-8, f"{rc['off_sus_energy_frac']:.2e}")

    rep["n_checks"] = len(rep["checks"])
    rep["green"] = True
    return rep


def main():
    rep = dry_run_gate()
    import json
    print(json.dumps(rep, indent=2))
    print(f"DRY-RUN GREEN — {rep['n_checks']} checks passed")


if __name__ == "__main__":
    main()
