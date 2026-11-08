"""TESSERA derived quantities — the MAGNITUDES behind the pass/fail counts, recomputed from banked
artifacts so the result doc cites a file instead of transcribing a number.

Everything here is DESCRIPTIVE elaboration of an already-frozen row. No verdict is issued, no
threshold is introduced, and no row of the §11 grid changes. It exists because three of TESSERA's
rows report a COUNT ("36/36", "3/3", "694/702") whose null is loose enough or whose members are
dependent enough that the count alone overstates what was learned. The house rule is to report the
verdict WITH its magnitude; these are the magnitudes.

  D1  Arm S core — how much did adding π0/π0.5 actually move the consensus, relative to what two
      MATCHED-RANDOM units of the same energy move it? The 36/36 count rides on a null whose std is
      ~0.5% of its mean, so |z| ≥ 3 is cleared by any systematic structure at all. The informative
      number is the ratio.
  D2  K2 seal — are the three per-faculty rankings three tests or one? If the footprint ordering and
      the survival ordering are both uniform in sign across faculties, "3/3" is one comparison
      restated, and the effective evidence is ~1 bit, not ~3.
  D3  K4 census — excess of each mechanism fraction over its own analytic chance floor, and the
      internal validity check that the `random` component lands ON that floor.

  python -m infra.aws_pg.jobs.tessera_derived --out d --s <arm_s.json> --k24 <k24.json> --k1 <k1.json>

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
import subprocess

K_CORE = "8"          # the registered core rank Arm S reports at
R_FOOT = "679"        # QUORUM's validated footprint rank (amendment A6b)


def _read(p):
    if p is None:
        return None
    if str(p).startswith("s3://"):
        return json.loads(subprocess.run(["aws", "s3", "cp", str(p), "-"], check=True,
                                         capture_output=True, text=True).stdout)
    return json.loads(pathlib.Path(p).read_text())


def d1_core_magnitude(s):
    """How much of the core did the VLA units actually consume, in units of a random write?"""
    n24 = {r["key"]: r for r in s["core"]["n24"]}
    n26 = {r["key"]: r for r in s["core"]["n26"]}
    assert set(n24) == set(n26), "the two rosters are not over the same matrices"
    per = []
    for k in sorted(n26):
        a, b = n24[k], n26[k]
        ba, bb = a["bottom_k"][K_CORE], b["bottom_k"][K_CORE]
        na, nb = a["null_bottom"][K_CORE]["mean"], b["null_bottom"][K_CORE]["mean"]
        # Adding ANY two units raises a PSD Gram's bottom eigenvalues; the null tells us by how much
        # two matched-random units of the same energy would raise them. The ratio is the answer to
        # "did the VLA writes avoid these directions", on a scale where 1.0 = wrote like noise and
        # 0.0 = did not touch them at all.
        per.append({"key": k, "n24_bottom": ba, "n26_bottom": bb,
                    "n24_null_mean": na, "n26_null_mean": nb,
                    "ratio_n24": ba / na, "ratio_n26": bb / nb,
                    "vla_increase": bb - ba, "random_increase": nb - na,
                    "vla_over_random": ((bb - ba) / (nb - na)) if (nb - na) > 0 else None,
                    "z": b["verdict"][K_CORE].get("z"),
                    "null_rel_std": (b["null_bottom"][K_CORE]["std"]
                                     / b["null_bottom"][K_CORE]["mean"])})
    vr = [p["vla_over_random"] for p in per if p["vla_over_random"] is not None]
    return {"k": int(K_CORE), "n_matrices": len(per),
            "ratio_real_over_null_24": {"median": st.median(p["ratio_n24"] for p in per),
                                        "min": min(p["ratio_n24"] for p in per),
                                        "max": max(p["ratio_n24"] for p in per)},
            "ratio_real_over_null_26": {"median": st.median(p["ratio_n26"] for p in per),
                                        "min": min(p["ratio_n26"] for p in per),
                                        "max": max(p["ratio_n26"] for p in per)},
            "vla_over_random_increase": {"median": st.median(vr), "min": min(vr), "max": max(vr),
                                         "n": len(vr)},
            "null_rel_std": {"median": st.median(p["null_rel_std"] for p in per)},
            "z_at_k": {"median": st.median(p["z"] for p in per if p["z"] is not None)},
            "reading": "vla_over_random < 1 ⇒ the VLA writes put LESS energy into the consensus-core "
                       "directions than matched-random writes of the same Frobenius norm would. "
                       "This is the magnitude behind the 36/36 count, which by itself only says the "
                       "real value cleared a null whose std is a fraction of a percent of its mean.",
            "per_matrix": per}


def d2_seal_independence(k24, k1):
    """Are K2's three per-faculty rankings three tests, or one restated three times?"""
    sp, sc = k24["K2_sealed"]["per_faculty"], k24["K2_score"]["per_faculty"]
    facs = sorted(sp)
    fp0 = [sp[f]["footprint_pi0"] for f in facs]
    fp1 = [sp[f]["footprint_pi05"] for f in facs]
    scored = [f for f in facs if "hit" in sc[f]]
    dfp = [sp[f]["footprint_pi0"] - sp[f]["footprint_pi05"] for f in scored]
    dS = [sc[f]["S_pi0"] - sc[f]["S_pi05"] for f in scored]
    uniform_pred = len({d > 0 for d in dfp}) == 1
    uniform_obs = len({d > 0 for d in dS}) == 1
    # Between-faculty spread of the predictor, inside each unit, vs the gap the prediction turns on.
    spread = max(max(fp0) - min(fp0), max(fp1) - min(fp1))
    gap = st.median(abs(d) for d in dfp)
    k1law = k1["law"] if k1 else None
    return {"faculties_scored": scored,
            "footprint_pi0": dict(zip(facs, fp0)), "footprint_pi05": dict(zip(facs, fp1)),
            "delta_footprint": dict(zip(scored, dfp)), "delta_survival": dict(zip(scored, dS)),
            "prediction_sign_uniform": uniform_pred, "observation_sign_uniform": uniform_obs,
            "within_unit_between_faculty_spread": spread,
            "median_between_unit_gap": gap,
            "spread_over_gap": spread / gap if gap else None,
            "effective_independent_tests": (1 if (uniform_pred and uniform_obs) else len(scored)),
            "n_reported_hits": k24["K2_score"]["n_hit"],
            "n_reported_scored": k24["K2_score"]["n_scored"],
            "sign_source": {"k1_delta": k1law["delta_loo_footprint"]["delta"] if k1law else None,
                            "k1_ci": k1law["delta_loo_footprint_ci"] if k1law else None,
                            "k1_p_perm": k1law["delta_loo_footprint"]["p_perm"] if k1law else None,
                            "k1_verdict": k1law["verdict"]["verdict"] if k1law else None},
            "reading": "when BOTH the predicted ordering and the observed ordering are uniform in "
                       "sign across faculties, the three rankings are one comparison restated three "
                       "times: effective n = 1, chance 0.5, ~1 bit. The sign that generated the "
                       "seal came from K1, whose own ΔLOO verdict must be read alongside it — a "
                       "coin-flip that lands heads off an unestablished direction is consistent "
                       "with that direction, and is not evidence for it."}


def d3_census_excess(k24):
    """Mechanism fractions minus their own analytic chance floor, plus the `random` sanity check."""
    out = {}
    for u, c in k24.get("K4_census", {}).items():
        fr, fl = c["fractions"], c["chance_floor"]
        ranks = sorted(fl, key=int)
        out[u] = {"basis_faculty": c["basis_faculty"], "total_fro2": c["total_fro2"],
                  "excess_over_floor": {p: {r: fr[p][r] - float(fl[r]) for r in ranks} for p in fr},
                  "random_minus_floor": {r: fr["random"][r] - float(fl[r]) for r in ranks},
                  "validity_check": "the `random` component is a control that should land ON the "
                                    "analytic floor r/d; a drift away from it would mean the census "
                                    "arithmetic is wrong, and the energy excess uninterpretable"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    for k in ("s", "k24", "k1"):
        ap.add_argument(f"--{k}", default=None)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    s, k24, k1 = _read(a.s), _read(a.k24), _read(a.k1)
    rep = {"object": "TESSERA derived magnitudes — descriptive, no verdict, no new threshold",
           "label": "DESCRIPTIVE elaboration of frozen rows; changes no §11 verdict"}
    if s:
        rep["D1_core_magnitude"] = d1_core_magnitude(s)
        m = rep["D1_core_magnitude"]
        print("[D1] core: 24-unit real/null median %.4f -> 26-unit %.4f"
              % (m["ratio_real_over_null_24"]["median"], m["ratio_real_over_null_26"]["median"]))
        print("[D1] VLA increase / matched-random increase: median %.3f (range %.3f-%.3f, n=%d)"
              % (m["vla_over_random_increase"]["median"], m["vla_over_random_increase"]["min"],
                 m["vla_over_random_increase"]["max"], m["vla_over_random_increase"]["n"]))
        print("[D1] null rel-std median %.5f, z median %.1f — the band is TIGHT, so the count is "
              "cheap and the ratio is the number"
              % (m["null_rel_std"]["median"], m["z_at_k"]["median"]))
    if k24:
        rep["D2_seal_independence"] = d2_seal_independence(k24, k1)
        e = rep["D2_seal_independence"]
        print("[D2] reported %d/%d hits; prediction-sign uniform=%s observation-sign uniform=%s "
              "=> effective independent tests = %d"
              % (e["n_reported_hits"], e["n_reported_scored"], e["prediction_sign_uniform"],
                 e["observation_sign_uniform"], e["effective_independent_tests"]))
        print("[D2] within-unit between-faculty spread %.4f vs between-unit gap %.4f (ratio %.2f)"
              % (e["within_unit_between_faculty_spread"], e["median_between_unit_gap"],
                 e["spread_over_gap"]))
        rep["D3_census_excess"] = d3_census_excess(k24)
        for u, c in rep["D3_census_excess"].items():
            print("[D3] %s excess@r%s energy %+.4f susceptibility %+.4f random %+.4f (control)"
                  % (u, R_FOOT, c["excess_over_floor"]["energy"][R_FOOT],
                     c["excess_over_floor"]["susceptibility"][R_FOOT],
                     c["random_minus_floor"][R_FOOT]))
    (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    print("TESSERA_DERIVED_OK", flush=True)


if __name__ == "__main__":
    main()
