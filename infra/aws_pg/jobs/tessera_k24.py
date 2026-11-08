"""TESSERA K2 + K4 — the VLA writes' footprints, sealed predictions, and stiff/energy census.

K2 (CONFIRMATORY, out-of-family). π0 and π0.5 are NOT in K1's fit population — `plenum_agg_fc`
lists them under `excluded`, which is exactly why amendment A2 pinned K1 there. So a prediction of
their survival pattern emitted from the K1-fitted direction is a genuine out-of-sample test, and
it is the leg that carries K1's evidential weight (K1 itself is confirmatory only through the
frozen pipeline, since its outcome labels are already known to us).

THE SEAL. The prediction is emitted MECHANICALLY from K1's fitted coefficient sign and the two
units' footprints, and written to `sealed_predictions.json` BEFORE this job reads any π0/π0.5
survival label. The scoring pass then reads the labels. Both artifacts are separate files with
separate hashes, so "predicted before seen" is checkable rather than asserted.

  known pattern (from PLENUM, verdict-level): π0 all-dead · π0.5 faculty-split (VQA retained)
  the prediction under test is the RANKING the footprint direction implies, per faculty.

K4 (EXPLORATORY CENSUS, labelled). The stiff-band / Q⊥energy decomposition of the π0 and π0.5
writes — `cleave_core.mechanism_census` over the same 126 matrices, reported against its own Haar
chance floor r/d. Census only: no verdict rides on it, and it is never in a BH family (§7).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.tessera_k24 --type r7i.2xlarge --no-weights \
      --args "--banks vqa=...,ocr=...,caption=... --k1 s3://.../tessera_k1_fit/out/metrics.json \
              --agg-fc s3://.../plenum_agg_fc/out/metrics.json" \
      --s3-prefix results/tessera_k24 --budget-min 300

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
FACULTIES = ("vqa", "ocr", "caption")
CENSUS_RANKS = (20, 50, 350, 679)
REGISTERED_RANK = 679
VLA_UNITS = ("pi0", "pi05")


def _read_json(p):
    if str(p).startswith("s3://"):
        return json.loads(subprocess.run(["aws", "s3", "cp", str(p), "-"], check=True,
                                         capture_output=True, text=True).stdout)
    return json.loads(pathlib.Path(p).read_text())


def _fetch(uri, dest):
    subprocess.run(["aws", "s3", "cp", uri, dest, "--only-show-errors"], check=True)
    return dest


def _policy_decoder_sd(name):
    """{(layer, type): tensor} of a published policy's PaliGemma-half LM decoder."""
    import re
    from infra.pg import lacuna_shell as LS
    pat = re.compile(r"^(?:language_model\.)?layers\.(\d+)\.(?:self_attn|mlp)\.([a-z_]+)\.weight$")
    full, _sha = LS.policy_full_sd(name)
    vlm, _h, _c = LS.policy_vlm_sd(name, full_sd=full)
    del full
    out = {}
    for k, v in vlm.items():
        m = pat.match(k)
        if m and v.ndim == 2:
            out[(int(m.group(1)), m.group(2))] = v
    assert len(out) == 126, f"{name}: {len(out)} decoder matrices, want 126"
    return out


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--banks", required=True, help="comma list faculty=uri")
    ap.add_argument("--k1", required=True, help="tessera_k1_fit metrics.json")
    ap.add_argument("--agg-fc", required=True,
                    help="the artifact K1 FIT on (plenum_agg_fc)")
    ap.add_argument("--labels", default=None,
                    help="artifact holding the π0/π0.5 SURVIVAL LABELS. Amendment A2 pinned K1 to "
                         "`plenum_agg_fc` precisely BECAUSE it excludes π0/π0.5 — which is what "
                         "makes K2 out-of-family — and that same exclusion means agg_fc has no "
                         "labels to score K2 against. They live in `plenum_agg_final2` (the "
                         "26-unit fit). Scoring there cannot contaminate K1, which never saw those "
                         "rows. Defaults to --agg-fc, which VOIDs if the units are absent.")
    a = ap.parse_args()
    t0 = time.time()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    from infra.aws_pg.jobs.tessera_k1_fit import _decoder_sd, footprints_for_unit
    from infra.pg import cleave_core as CL

    banks = {}
    for spec in a.banks.split(","):
        fac, uri = spec.split("=", 1)
        banks[fac] = torch.load(_fetch(uri, f"/tmp/k24_{fac}.pt"), map_location="cpu",
                                weights_only=False)
    k1 = _read_json(a.k1)
    sign = 1.0 if k1["law"]["delta_loo_footprint"]["delta"] >= 0 else -1.0
    k1_verdict = k1["law"]["verdict"]["verdict"]

    rep = {"object": "TESSERA K2 (sealed, out-of-family) + K4 (census)",
           "k1_verdict": k1_verdict, "k1_delta": k1["law"]["delta_loo_footprint"]["delta"],
           "registered_rank": REGISTERED_RANK, "census_ranks": list(CENSUS_RANKS)}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    _w()

    # ---- footprints of the VLA writes ---------------------------------------------------------
    base_sd = _decoder_sd(PG1_REPO)
    fps, census = {}, {}
    for uid in VLA_UNITS:
        sd = _policy_decoder_sd(uid)
        fps[uid] = footprints_for_unit(base_sd, sd, banks, CENSUS_RANKS)
        print(f"[K2] {uid} " + " ".join(
            f"{f}={fps[uid][f]['footprint'][REGISTERED_RANK]:.4f}" for f in banks), flush=True)
        # ---- K4: the stiff/energy census on the SAME write (exploratory, labelled) ------------
        acc = {p: {r: 0.0 for r in CENSUS_RANKS} for p in ("energy", "susceptibility", "random")}
        floor, wsum = {}, 0.0
        anyfac = sorted(banks)[0]
        for (L, t), W in base_sd.items():
            dW = (sd[(L, t)].float() - W.float()).numpy()
            fro2 = float((dW ** 2).sum())
            if fro2 <= 0:
                continue
            for r in CENSUS_RANKS:
                cen = CL.mechanism_census(dW, banks[anyfac][L], r, t)
                for p in acc:
                    acc[p][r] += cen[p] * fro2
                floor[r] = cen["chance_floor"]
            wsum += fro2
        census[uid] = {"label": "EXPLORATORY census (§7) — never in a BH family",
                       "basis_faculty": anyfac,
                       "fractions": {p: {r: acc[p][r] / wsum for r in CENSUS_RANKS} for p in acc},
                       "chance_floor": floor, "total_fro2": wsum}
        del sd
        rep["K2_footprints"], rep["K4_census"] = fps, census
        _w()

    # ---- THE SEAL: emit the prediction before reading any label ------------------------------
    # Higher footprint ⇒ (sign>0) more death. The prediction is the per-faculty RANKING between the
    # two units, which is what a direction can predict without an intercept.
    pred = {}
    for f in banks:
        v0 = fps["pi0"][f]["footprint"][REGISTERED_RANK]
        v1 = fps["pi05"][f]["footprint"][REGISTERED_RANK]
        more_damaged = "pi0" if sign * (v0 - v1) > 0 else "pi05"
        pred[f] = {"footprint_pi0": v0, "footprint_pi05": v1, "sign": sign,
                   "predicted_more_damaged": more_damaged}
    sealed = {"object": "TESSERA K2 sealed predictions — emitted from the K1-fitted direction "
                        "BEFORE any pi0/pi05 survival label was read by this job",
              "k1_sign": sign, "k1_verdict": k1_verdict, "per_faculty": pred}
    sp = out / "sealed_predictions.json"
    sp.write_text(json.dumps(sealed, indent=1))
    rep["sealed_sha256"] = hashlib.sha256(sp.read_bytes()).hexdigest()
    rep["K2_sealed"] = sealed
    _w()
    print("[K2] SEALED " + json.dumps({f: pred[f]["predicted_more_damaged"] for f in pred}),
          flush=True)

    # ---- score the seal against the banked labels ---------------------------------------------
    agg = _read_json(a.labels or a.agg_fc)
    rep["labels_source"] = a.labels or a.agg_fc
    law_readout = agg["law_readout_per_faculty"]
    S = {}
    for c in agg["cells"]:
        if c["unit"] in VLA_UNITS and law_readout.get(c["faculty"]) == c["readout"]:
            S.setdefault(c["unit"], {})[c["faculty"]] = float(c["S"])
    scored, n_hit, n_scored = {}, 0, 0
    for f in pred:
        s0, s1 = S.get("pi0", {}).get(f), S.get("pi05", {}).get(f)
        if s0 is None or s1 is None:
            scored[f] = {"VOID": "no measured grid row for one of the units on this faculty",
                         "S_pi0": s0, "S_pi05": s1}
            continue
        observed = "pi0" if s0 < s1 else "pi05"          # lower survival = more damaged
        hit = bool(observed == pred[f]["predicted_more_damaged"])
        scored[f] = {"S_pi0": s0, "S_pi05": s1, "observed_more_damaged": observed,
                     "predicted": pred[f]["predicted_more_damaged"], "hit": hit,
                     "tied": bool(s0 == s1)}
        n_scored += 1
        n_hit += int(hit)
    rep["K2_score"] = {"per_faculty": scored, "n_scored": n_scored, "n_hit": n_hit,
                       "note": "a 2-way ranking per faculty: chance is 0.5, so with m=3 faculties "
                               "this is a weak but genuinely out-of-family test — reported with "
                               "its magnitude, never as a bare hit count"}
    rep["elapsed_s"] = time.time() - t0
    _w()
    print("TESSERA_K24_OK", json.dumps({"n_hit": n_hit, "n_scored": n_scored}), flush=True)


if __name__ == "__main__":
    main()
