"""TESSERA §9 step-2 offline gate — the weight-free dry-run (principle #13).

Proves the LOGIC before any compute is spent. Distinct from the runtime preflights (P1–P5), which
prove the WIRING on real weights. Everything here runs on tiny tensors in seconds, numpy only, so
it also runs on the control plane.

Every check REPLAYS A KNOWN EVENT — a planted structure whose answer is known in advance — and
every check has a partner that must come back NEGATIVE. A gate that only ever fires one way is
the failure mode this program has shipped twice, so each test below names both outcomes.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.tessera_offline --type r7i.2xlarge --no-weights \
      --s3-prefix results/tessera_offline --budget-min 30
  # or locally:  python -m infra.aws_pg.jobs.tessera_offline --out /tmp/tessera_offline

No try/except (CLAUDE.md): a failed check must raise, not be recorded as a soft warning.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from infra.pg import tessera_core as TC


def check(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    assert ok, f"OFFLINE GATE FAILED — {name}: {detail}"
    return {"check": name, "passed": True, "detail": detail}


# ===========================================================================
# S — subspace overlap: a planted shared channel vs an independent pair
# ===========================================================================

def gate_overlap(rng, n=256, r=16):
    rows = []
    null = TC.haar_overlap_null(n, r, r, n_draws=400, seed=7)

    # KNOWN EVENT 1 — identical subspaces must read SHARED-CHANNEL at overlap exactly 1.
    Q, _ = np.linalg.qr(rng.standard_normal((n, r)))
    o_same = TC.subspace_overlap(Q, Q)
    v_same = TC.overlap_verdict(o_same, null)
    rows.append(check("S/overlap-identical", abs(o_same - 1.0) < 1e-10
                      and v_same["verdict"] == "SHARED-CHANNEL",
                      f"overlap={o_same:.6f} verdict={v_same['verdict']}"))

    # KNOWN EVENT 2 — two INDEPENDENT random subspaces must read HAAR-DISJOINT. This is the leg
    # that can fail: if the null band were miscomputed, independent subspaces would read as a
    # shared channel and every overlap claim in Arm S would be manufactured.
    A, _ = np.linalg.qr(rng.standard_normal((n, r)))
    B, _ = np.linalg.qr(rng.standard_normal((n, r)))
    o_ind = TC.subspace_overlap(A, B)
    v_ind = TC.overlap_verdict(o_ind, null)
    rows.append(check("S/overlap-independent", v_ind["verdict"] == "HAAR-DISJOINT",
                      f"overlap={o_ind:.6f} (analytic mean r/n={r/n:.6f}) "
                      f"verdict={v_ind['verdict']} z={v_ind['z']:.2f}"))

    # KNOWN EVENT 3 — a HALF-shared subspace (r/2 columns common) must clear the band.
    C = np.concatenate([A[:, :r // 2], B[:, :r // 2]], axis=1)
    C, _ = np.linalg.qr(C)
    o_half = TC.subspace_overlap(A, C)
    v_half = TC.overlap_verdict(o_half, null)
    rows.append(check("S/overlap-half-shared", v_half["verdict"] == "SHARED-CHANNEL",
                      f"overlap={o_half:.6f} verdict={v_half['verdict']} z={v_half['z']:.2f}"))

    # the null's own mean must equal the analytic r/n — asserted inside haar_overlap_null, recorded
    rows.append(check("S/haar-null-analytic",
                      abs(null["mean"] - null["analytic_mean"]) < 5 * null["std"] / np.sqrt(400),
                      f"null mean={null['mean']:.6f} analytic={null['analytic_mean']:.6f}"))
    return rows


# ===========================================================================
# S — the consensus core: a planted untouched subspace vs an iid control
# ===========================================================================

def gate_core(rng, n=96, n_units=26, k=8, n_null=200):
    rows = []

    def null_band(spectra):
        vals = []
        for _ in range(n_null):
            G0 = np.zeros((n, n))
            for p in spectra:
                Z = rng.standard_normal((n, p.size))
                Q, _ = np.linalg.qr(Z)
                G0 += (Q * p[None, :]) @ Q.T
            w = np.clip(np.linalg.eigvalsh(G0), 0.0, None)
            vals.append(w[:k].mean())
        v = np.asarray(vals)
        return {"mean": float(v.mean()), "std": float(v.std()),
                "p2.5": float(np.percentile(v, 2.5))}

    # KNOWN EVENT 1 — PLANTED CORE: every unit writes only in the first n-k directions, so the
    # last k are jointly untouched. Must read CORE-BEYOND-NULL.
    dws_planted, dws_control = [], []
    for _ in range(n_units):
        A = rng.standard_normal((n, n))
        A[:, n - k:] = 0.0                      # the planted untouched subspace
        dws_planted.append(A)
        dws_control.append(rng.standard_normal((n, n)))     # iid — no shared untouched subspace

    for label, dws, want in (("planted", dws_planted, "CORE-BEYOND-NULL"),
                             ("iid-control", dws_control, "INSIDE-NULL")):
        G = TC.consensus_gram(dws)
        real = TC.bottom_k_mean(G, k)
        band = null_band([TC.spectrum_normsq(A) for A in dws])
        v = TC.core_verdict(real, band["mean"], band["std"], band["p2.5"])
        rows.append(check(f"S/core-{label}", v["verdict"] == want,
                          f"real={real:.6g} null={band['mean']:.6g}±{band['std']:.2g} "
                          f"z={v['z']:.1f} verdict={v['verdict']} (want {want})"))
    return rows


# ===========================================================================
# K1 — the footprint: a write inside the subspace vs a curvature-indifferent one
# ===========================================================================

def gate_footprint(rng, n=256, m=128, r=16):
    rows = []
    U, _ = np.linalg.qr(rng.standard_normal((n, r)))

    # KNOWN EVENT 1 — a write built entirely inside the curvature subspace reads footprint 1.
    W_in = rng.standard_normal((m, r)) @ U.T
    f_in = TC.footprint(W_in, U)
    rows.append(check("K1/footprint-inside", abs(f_in - 1.0) < 1e-10,
                      f"footprint={f_in:.10f} (want 1)"))

    # KNOWN EVENT 2 — a write built entirely OUTSIDE reads 0. Without this leg the statistic
    # could be a constant near 1 and the inside-check would still pass.
    Uc = np.linalg.qr(rng.standard_normal((n, n)))[0]
    comp = Uc - U @ (U.T @ Uc)
    comp = np.linalg.qr(comp[:, :n - r])[0]
    W_out = rng.standard_normal((m, n - r)) @ comp.T
    f_out = TC.footprint(W_out, U)
    rows.append(check("K1/footprint-outside", f_out < 1e-8,
                      f"footprint={f_out:.3e} (want 0)"))

    # KNOWN EVENT 3 — an isotropic write reads the matched-random reference r/n, NOT zero. This
    # is what makes "a footprint must be read against r/n" checkable rather than asserted.
    f_iso = np.mean([TC.footprint(rng.standard_normal((m, n)), U) for _ in range(64)])
    rows.append(check("K1/footprint-isotropic",
                      abs(f_iso - TC.footprint_null(n, r)) < 0.02,
                      f"mean footprint={f_iso:.4f} vs r/n={TC.footprint_null(n, r):.4f}"))
    return rows


# ===========================================================================
# K1 — the fit pipeline: a planted predictor vs a pure-noise predictor (N3)
# ===========================================================================

def gate_fit(rng, n_units=24, n_fac=3):
    """Replays PLENUM's own instrument check: the law fit recovers a PLANTED effect and does NOT
    fire on noise. Uses the same grouped-LOO shape K1 will run through the REUSED pipeline."""
    rows = []
    units = [f"u{i}" for i in range(n_units)]

    def loo_r2(X, y, groups):
        """Leave-one-UNIT-out R² — PLENUM defect #2: dropping a ROW lets a unit's other faculties
        predict its own mean, which is the claim under test."""
        pred = np.empty_like(y)
        for g in np.unique(groups):
            tr, te = groups != g, groups == g
            A = np.c_[np.ones(tr.sum()), X[tr]]
            beta, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
            pred[te] = np.c_[np.ones(te.sum()), X[te]] @ beta
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        return 1.0 - ss_res / ss_tot

    groups = np.repeat(np.arange(n_units), n_fac)
    base = rng.standard_normal((n_units * n_fac, 2))

    # KNOWN EVENT 1 — a predictor that genuinely drives y must show ΔLOO > 0.
    signal = rng.standard_normal(n_units * n_fac)
    y = 2.0 * signal + 0.3 * rng.standard_normal(n_units * n_fac)
    d_signal = loo_r2(np.c_[base, signal], y, groups) - loo_r2(base, y, groups)
    rows.append(check("K1/fit-planted-predictor", d_signal > 0.2,
                      f"ΔLOO(planted) = {d_signal:+.4f} (want > 0.2)"))

    # KNOWN EVENT 2 — a pure-noise predictor must NOT. This is the leg that makes a positive K1
    # mean something: a pipeline that rewards any added column would report the footprint live
    # whatever the footprint was.
    noise = rng.standard_normal(n_units * n_fac)
    d_noise = loo_r2(np.c_[base, noise], y, groups) - loo_r2(base, y, groups)
    rows.append(check("K1/fit-noise-predictor", d_noise < 0.1,
                      f"ΔLOO(noise) = {d_noise:+.4f} (want < 0.1)"))

    # KNOWN EVENT 3 — label permutation (N3) must destroy the planted effect.
    perm = rng.permutation(y.size)
    d_perm = loo_r2(np.c_[base, signal], y[perm], groups) - loo_r2(base, y[perm], groups)
    rows.append(check("K1/fit-permutation-N3", d_perm < 0.1,
                      f"ΔLOO(planted | permuted labels) = {d_perm:+.4f} (want < 0.1)"))
    return rows


# ===========================================================================
# K3 — the n* ladder logic (P3 replayed weight-free)
# ===========================================================================

def gate_nstar():
    """QUORUM's necessary-rank readout on synthetic retention curves.

    KNOWN EVENT: a capability destroyed by removing ONE direction must read n* ≈ 1-2; a capability
    that survives until a large rank is removed must read a large n*. Both directions are checked,
    so a readout stuck at either end fails.
    """
    from infra.pg import quorum_core as QC
    rows = []
    ranks = list(QC.RANK_LADDER)
    rho_break = 0.5

    # rank-1 capability: retention collapses immediately (P3's planted positive control)
    ret_r1 = [0.05 if r >= 1 else 1.0 for r in ranks]
    n1 = QC.necessary_rank(ret_r1, ranks, rho_break)
    rows.append(check("K3/nstar-planted-rank1",
                      (not n1["censored"]) and n1["n_star"] <= 2,
                      f"n* = {n1['n_star']} censored={n1['censored']} (want <= 2, uncensored)"))

    # a distributed capability: retention holds until rank ~350
    ret_hi = [1.0 if r < 350 else 0.05 for r in ranks]
    nhi = QC.necessary_rank(ret_hi, ranks, rho_break)
    rows.append(check("K3/nstar-distributed",
                      (not nhi["censored"]) and nhi["n_star"] >= 150,
                      f"n* = {nhi['n_star']} censored={nhi['censored']} (want >= 150)"))

    # a capability that never breaks must be RIGHT-CENSORED, never silently assigned a finite n*.
    # This is the ceiling leg: a readout that reports a number here would manufacture an n* for a
    # capability no removal ever touched.
    ret_never = [1.0 for _ in ranks]
    nnever = QC.necessary_rank(ret_never, ranks, rho_break)
    rows.append(check("K3/nstar-never-breaks-is-censored",
                      nnever["censored"] and nnever["n_star"] >= max(ranks),
                      f"n* = {nnever['n_star']} censored={nnever['censored']} "
                      f"(want censored, >= {max(ranks)})"))
    return rows


# ===========================================================================
# X — the identity-pair band logic
# ===========================================================================

def gate_fractions():
    rows = []
    band = {"p2.5": 0.001, "p50": 0.004, "p97.5": 0.010}
    for obs, want in ((0.05, "ABOVE-IDENTITY-BAND"), (0.004, "INSIDE-IDENTITY-BAND"),
                      (0.0001, "BELOW-IDENTITY-BAND")):
        v = TC.fraction_verdict(obs, band)
        rows.append(check(f"X/fraction-{want.lower()}", v["verdict"] == want,
                          f"observed={obs} verdict={v['verdict']}"))
    return rows


# ===========================================================================
# X — the REUSED estimator, in isolation from the NEW port (P1 attribution)
# ===========================================================================

def gate_estimator(rng, dict_size=4096, d=256, n_emergent=456):
    """Feed `classify_features` a decoder with a KNOWN after-only block. No training, no port.

    PRE-REGISTERED 2026-08-01, before the run. P1 returned PORT: the pg crosscoder path cannot
    recover a planted 11% emergent fraction at any width, with or without noise. The path has two
    halves and they carry very different consequences:

      NEW port    `tessera_crosscoder.{train_crosscoder, code_stats, fvu}`  -> TESSERA-local defect
      REUSED      `delta_crosscoder.{decoder_norms, CrosscoderFeatureStats, classify_features}`
                  -> also bears on the PG2 W4 "no new features" verdict, which rides this classifier

    This test exercises ONLY the reused half, by CONSTRUCTING the decoder rather than training one.

    FROZEN RULE:
      ESTIMATOR-OK      recovered emergent fraction is within 1e-6 of the planted one
                        => the classifier is sound; P1's failure is in the NEW port.
      ESTIMATOR-BROKEN  otherwise => the defect is in the REUSED estimator and reaches past TESSERA.
    """
    import numpy as np
    from infra.pg import delta_crosscoder as DC
    rows = []
    # decoder weight is (2, dict_size, d): layer 0 = BEFORE, layer 1 = AFTER (the stacking contract)
    w = rng.standard_normal((2, dict_size, d))
    # the planted after-only block: zero the BEFORE decoder for the last n_emergent latents, so
    # their relative norm is exactly 1.0 and they MUST classify as emergent
    w[0, dict_size - n_emergent:, :] = 0.0
    nb, na = DC.decoder_norms(w)
    fire = np.full(dict_size, 0.5)                # every latent live, so none is filtered as dead
    stats = DC.CrosscoderFeatureStats(norm_before=nb, norm_after=na,
                                      fire_before=fire, fire_after=fire)
    cl = DC.classify_features(stats)
    planted = n_emergent / dict_size
    rec = cl["emergent_fraction"]
    rows.append(check("X/estimator-planted-emergent", abs(rec - planted) < 1e-6,
                      f"recovered {rec:.6f} vs planted {planted:.6f} "
                      f"(counts {cl['counts']}) -> "
                      f"{'ESTIMATOR-OK, P1 failure is in the NEW port' if abs(rec - planted) < 1e-6 else 'ESTIMATOR-BROKEN, reaches past TESSERA'}"))
    # the mirror: zero the AFTER decoder instead, which must classify as LOST at the same fraction
    w2 = rng.standard_normal((2, dict_size, d))
    w2[1, dict_size - n_emergent:, :] = 0.0
    nb2, na2 = DC.decoder_norms(w2)
    cl2 = DC.classify_features(DC.CrosscoderFeatureStats(norm_before=nb2, norm_after=na2,
                                                         fire_before=fire, fire_after=fire))
    rows.append(check("X/estimator-planted-lost", abs(cl2["lost_fraction"] - planted) < 1e-6,
                      f"recovered lost {cl2['lost_fraction']:.6f} vs planted {planted:.6f}"))
    return rows


# ===========================================================================
# SCHEMA — every quantity a claim rests on must carry a matched null IN THE SAME CELL.
#
# Added 2026-08-02, after `tessera_x.py` v1 trained the shuffle-paired crosscoder, computed its
# full fractions, and persisted exactly one field of them. `lost_fraction` -- the only non-zero
# quantity Arm X measures and the one the π0 erasure claim rests on -- was therefore left with no
# null, while `emergent_fraction`, which came back ~0 nearly everywhere, had one.
#
# None of the other 20 checks could have caught this. They test TRANSFORMS: given inputs, does the
# logic return the right number? Storing a subset of a correct dict is not a transform error. This
# gate tests the ARTIFACT SCHEMA instead: for each quantity a claim will be read off, is its null
# actually written down next to it? The negative partner replays the v1 record verbatim.
# ===========================================================================

# quantity -> the key its matched null must appear under, within one site cell
CLAIM_NULLS = {"emergent_fraction": "shuffle_null", "lost_fraction": "shuffle_null"}


def cell_nulls_present(cell, claim_nulls=CLAIM_NULLS):
    """{quantity: bool} — is this quantity's null persisted alongside it in this cell?

    A null exists for a quantity only if the null record carries THAT FIELD. A null record that
    exists but holds a different quantity is worse than none: it reads as coverage.
    """
    out = {}
    for q, nk in claim_nulls.items():
        measured = [s for s in cell.get("seeds", []) if q in s]
        nulls = [s for s in cell.get(nk, []) if q in s]
        out[q] = bool(measured) and bool(nulls)
    return out


def gate_schema():
    rows = []
    good = {"seeds": [{"seed": 0, "emergent_fraction": 0.0, "lost_fraction": 0.02}],
            "shuffle_null": [{"seed": 0, "emergent_fraction": 0.13, "lost_fraction": 0.11}]}
    v1_bug = {"seeds": [{"seed": 0, "emergent_fraction": 0.0, "lost_fraction": 0.02}],
              "shuffle_null": [{"seed": 0, "emergent_fraction": 0.13}]}       # the shipped record
    g, b = cell_nulls_present(good), cell_nulls_present(v1_bug)
    rows.append(check("SCHEMA/full-cell-has-both-nulls", all(g.values()),
                      f"complete cell reports {g}"))
    # The negative partner: the gate must FLAG the record that actually shipped, and must flag it
    # on lost_fraction specifically while still seeing emergent_fraction's null as present.
    rows.append(check("SCHEMA/v1-record-is-flagged",
                      b["emergent_fraction"] and not b["lost_fraction"],
                      f"the as-shipped Arm X cell reports {b} — lost_fraction correctly flagged"))
    # A null record that exists but carries the wrong quantity must not read as coverage.
    wrong = {"seeds": [{"seed": 0, "lost_fraction": 0.02}],
             "shuffle_null": [{"seed": 0, "survival_fraction": 0.98}]}
    rows.append(check("SCHEMA/wrong-field-null-not-counted",
                      not cell_nulls_present(wrong)["lost_fraction"],
                      "a null record holding a different quantity does not count as a null"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    t0 = time.time()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    results = []
    results += gate_overlap(rng)
    results += gate_core(rng)
    results += gate_footprint(rng)
    results += gate_fit(rng)
    results += gate_nstar()
    results += gate_fractions()
    results += gate_estimator(rng)
    results += gate_schema()

    controls = {
        "object": "TESSERA offline gate — weight-free dry-run of the pure logic (§9 step 2)",
        "n_checks": len(results),
        "all_passed": all(r["passed"] for r in results),
        "checks": results,
        "note": "every check replays a KNOWN event and every check has a partner that must come "
                "back negative; a one-sided gate is the failure mode this program has shipped "
                "twice",
        "elapsed_s": time.time() - t0,
    }
    (out / "controls.json").write_text(json.dumps(controls, indent=1))
    print(f"\nOFFLINE GATE: {len(results)} checks, all passed.", flush=True)


if __name__ == "__main__":
    main()
