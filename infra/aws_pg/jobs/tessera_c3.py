"""TESSERA-FIX C3 — re-base Result 3's D1 denominator on a SAME-BASE control. Spec §2, §3 row 8.

FINDING 8. `tessera_derived`'s D1 compared

    real increase   = bottom8(G26_real) − bottom8(G24_real)
    "random" increase = bottom8_nullmean(26 units) − bottom8_nullmean(24 units)

The second is a **between-null difference**: two independently-drawn null distributions differenced
against each other. It is not "what two random units would have added to THIS base", because its
first term rotates all 26 spectra and its second rotates all 24 — the 24 shared units are re-drawn
between numerator and denominator, so the difference carries their re-randomisation too.

The same-base control holds the real 24-FT Gram FIXED and randomises only the two units under test:

    control increase = mean_draws[ bottom8( G24_real + Q₁diag(p_π0)Q₁ᵀ + Q₂diag(p_π0.5)Q₂ᵀ ) ]
                       − bottom8(G24_real)

with Q Haar. Same base, same two spectra, same energy — only the ORIENTATION of the two VLA writes
is randomised. That is the control the claim "the VLA hops avoid the core" actually needs.

FINDING 16 is fixed here too: `_null_band` draws every matrix's null from ONE seed, correlating the
per-matrix nulls. C3 seeds per matrix (`seed ⊕ hash(key)`), matching `invariant_core`'s per-matrix
scheme.

The spec says "CPU from banked spectra". **Spectra are not banked** — `tessera_s_full2` carries
`bottom_k` / `null_bottom` per matrix and no spectra (amendment AF-3, recorded before the run). So
this job re-streams the 26 units to rebuild `G24_real` and the two VLA spectra, over q/k ONLY (36
matrices), skipping the overlap sweep, `ft_V` and `ft_gram_all` that made Arm S an 18-hour job. The
REAL increase is read from the banked artifact, so nothing measured is recomputed.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.tessera_c3 --type r7i.2xlarge --no-weights \
      --args "--s s3://.../tessera_s_full2/out/metrics.json" \
      --s3-prefix results/tessera_c3 --budget-min 600

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import time

PG1_REPO = "google/paligemma-3b-pt-224"
K_CORE = 8
N_DRAWS = 100


def _read(p):
    if str(p).startswith("s3://"):
        return json.loads(subprocess.run(["aws", "s3", "cp", str(p), "-"], check=True,
                                         capture_output=True, text=True).stdout)
    return json.loads(pathlib.Path(p).read_text())


def _mseed(base, key):
    """Per-MATRIX seed (finding 16) — a shared seed correlates the 36 nulls."""
    return (base ^ int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)) % (2 ** 31 - 1)


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--s", required=True, help="banked tessera_s_full2 metrics.json")
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    t0 = time.time()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    from infra.aws_pg.jobs.tessera_s_core import (CORE_TYPES, _delete_cache, _ftype, _hf_lm_sd,
                                                  _policy_lm_sd, unit_sources)
    from infra.pg import tessera_core as TC

    s = _read(a.s)
    n24 = {r["key"]: r for r in s["core"]["n24"]}
    n26 = {r["key"]: r for r in s["core"]["n26"]}
    ft_units = [u for u in s["roster26"] if u.startswith("ft-")]
    assert len(ft_units) == 24, f"expected 24 FT units, got {len(ft_units)}"

    w_pg1 = _hf_lm_sd(PG1_REPO)
    core_keys = sorted(k for k in w_pg1 if _ftype(k) in CORE_TYPES)
    assert len(core_keys) == 36, f"{len(core_keys)} core keys, want 36"

    rep = {"object": "TESSERA-FIX C3 — same-base control for Result 3's avoidance magnitude",
           "finding": [8, 16], "k": K_CORE, "n_draws": a.n_draws, "seed": a.seed,
           "amendment": "AF-3 — spectra are NOT banked in tessera_s_full2, so the spec's "
                        "'CPU from banked spectra' form is not runnable; the 26 units are "
                        "re-streamed over q/k only. The REAL increase is read from the banked "
                        "artifact and is not recomputed.",
           "denominator_note": "D1's denominator was a BETWEEN-NULL difference (26-unit null mean "
                               "minus 24-unit null mean), which re-randomises the 24 shared units "
                               "between its two terms. C3 holds G24_real fixed and randomises only "
                               "the two VLA orientations."}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    _w()

    gram24 = {k: None for k in core_keys}
    vla_spectra = {k: {} for k in core_keys}
    # Incremental progress: the unit stream is ~24 HF downloads and dominates wall-clock. Without a
    # per-unit write, a budget timeout during streaming loses the whole run with no record of how
    # far it got (the house fleet-job rule: bind the accumulator BEFORE the loop, write inside it).
    rep["units_streamed"] = []
    _w()
    todo = unit_sources(ft_units + ["pi0", "pi05"])
    for i, (uid, (kind, ref)) in enumerate(todo, 1):
        print(f"[C3 unit {i}/{len(todo)}] {uid} ({kind}:{ref})", flush=True)
        w = _hf_lm_sd(ref) if kind == "hf" else _policy_lm_sd(ref)
        is_ft = uid.startswith("ft-")
        for k in core_keys:
            A = (w[k].to(torch.float64) - w_pg1[k].to(torch.float64)).numpy()
            fro2 = float((A ** 2).sum())
            assert fro2 > 0, f"{uid}/{k}: zero ΔW"
            if is_ft:
                G = (A.T @ A) / fro2
                gram24[k] = G if gram24[k] is None else gram24[k] + G
                del G
            else:
                vla_spectra[k][uid] = TC.spectrum_normsq(A)
            del A
        del w
        if kind == "hf":
            _delete_cache(ref)
        rep["units_streamed"].append({"i": i, "unit": uid, "elapsed_s": time.time() - t0})
        _w()

    rows = []
    for k in core_keys:
        G24 = gram24[k]
        n = G24.shape[0]
        b24_real = float(np.clip(np.linalg.eigvalsh(G24), 0.0, None)[:K_CORE].mean())
        # Sanity: the rebuilt G24 must reproduce the banked 24-unit bottom-8, else this control is
        # not on the same base the result used.
        b24_banked = n24[k]["bottom_k"][str(K_CORE)]
        rel = abs(b24_real - b24_banked) / max(abs(b24_banked), 1e-300)
        assert rel < 1e-6, f"{k}: rebuilt G24 bottom-8 {b24_real:.6e} != banked {b24_banked:.6e}"

        b26_banked = n26[k]["bottom_k"][str(K_CORE)]
        real_inc = b26_banked - b24_banked

        rng = np.random.default_rng(_mseed(a.seed, k))
        ctrl = []
        for _ in range(a.n_draws):
            G = G24.copy()
            for uid in ("pi0", "pi05"):
                p = np.asarray(vla_spectra[k][uid], dtype=np.float64)
                Q, _ = np.linalg.qr(rng.standard_normal((n, p.size)))
                G += (Q * p[None, :]) @ Q.T
            ctrl.append(float(np.clip(np.linalg.eigvalsh(G), 0.0, None)[:K_CORE].mean()))
            del G
        ctrl = np.asarray(ctrl)
        ctrl_inc = float(ctrl.mean() - b24_real)
        rows.append({"key": k, "type": _ftype(k),
                     "b24_banked": b24_banked, "b26_banked": b26_banked,
                     "real_increase": real_inc,
                     "control_increase_samebase": ctrl_inc,
                     "control_increase_p2.5": float(np.percentile(ctrl, 2.5) - b24_real),
                     "control_increase_p97.5": float(np.percentile(ctrl, 97.5) - b24_real),
                     "ratio": (real_inc / ctrl_inc) if ctrl_inc > 0 else None,
                     "below_one": bool(ctrl_inc > 0 and real_inc < ctrl_inc)})
        print(f"[C3] {k} real {real_inc:.4e} ctrl {ctrl_inc:.4e} ratio "
              f"{rows[-1]['ratio'] if rows[-1]['ratio'] is None else round(rows[-1]['ratio'], 4)}",
              flush=True)
        rep["per_matrix"] = rows
        _w()

    rat = [r["ratio"] for r in rows if r["ratio"] is not None]
    n_below = sum(1 for r in rows if r["below_one"])
    med_of_ratios = float(np.median(rat))
    ratio_of_meds = float(np.median([r["real_increase"] for r in rows])
                          / np.median([r["control_increase_samebase"] for r in rows]))
    # H_avoidance-magnitude (spec §1.2): <1.0 on >=30/36, registered point expectation 0.13-0.26.
    verdict = ("SURVIVES" if (med_of_ratios < 1.0 and n_below >= 30) else
               "RETRACTED — the magnitude claim does not survive the same-base control")
    rep["summary"] = {
        "median_of_ratios": med_of_ratios, "ratio_of_medians": ratio_of_meds,
        "n_below_one": n_below, "n_matrices": len(rows),
        "registered_point_expectation": [0.13, 0.26],
        "inside_registered_expectation": bool(0.13 <= med_of_ratios <= 0.26),
        "H_avoidance_magnitude": verdict,
        "superseded": {"between_null_median_of_ratios": 0.388,
                       "note": "the D1 number Result 3 shipped; kept visible, not overwritten"}}
    rep["elapsed_s"] = time.time() - t0
    _w()
    print(f"\nmedian-of-ratios {med_of_ratios:.4f} · ratio-of-medians {ratio_of_meds:.4f} · "
          f"below 1.0 on {n_below}/{len(rows)}")
    print(f"H_avoidance-magnitude: {verdict}")
    print("TESSERA_C3_OK", flush=True)


if __name__ == "__main__":
    main()
