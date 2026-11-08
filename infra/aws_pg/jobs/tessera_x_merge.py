"""TESSERA Arm X merge — join the per-job Arm X artifacts into ONE artifact the aggregator reads.

Arm X ran as several `pg` jobs (a GPU job is one instance, and the 10 site-cells did not fit one
budget). Merging is a bookkeeping step, so it must not become a place where a number can quietly
change. Three properties are enforced, each failing loud:

  1. PROVENANCE EQUALITY. Every input must agree on `pinned_sha`, `base_ppl`, the harvest size, and
     the per-pair `input_hash`. Different inputs are not mergeable cells of one measurement, and a
     merge that papers over that would be inventing an experiment nobody ran.
  2. DUPLICATE CELLS ARE A REPRODUCIBILITY CHECK, NOT A CHOICE. `identity` was measured in two
     separate jobs on two separate instances. Where the same (pair, site, seed) appears twice, the
     two values are COMPARED and the comparison is reported. A merge that silently picked one would
     throw away a free cross-instance replication.
  3. DENSITY IS CARRIED, NOT AVERAGED AWAY. A cell keeps its own realised seed count, and the
     merged `density_cut` reports the per-cell floor and ceiling rather than one global claim.

The identity band is recomputed from the POOLED identity seeds, so the band the other pairs are read
against widens correctly when identity was measured more times.

  python -m infra.aws_pg.jobs.tessera_x_merge --out merged \\
      --inputs s3://.../tessera_x_c/out/metrics.json,s3://.../tessera_x_d/out/metrics.json \\
      --steps-actual 20000 --lr-actual 1e-3 --hp-evidence "..."

`--steps-actual`/`--lr-actual` exist because the Arm X job's v1 recorded `XC.CC_TRAIN_STEPS` (the
A9 port constant, 4000) into `hyperparams.steps` while training with the `--steps` override, and
omitted `lr` entirely. The banked artifacts therefore under-report their own training setting. The
fix is in `tessera_x.py`, but the ALREADY-BANKED artifacts cannot be rewritten, so the true setting
is passed here WITH its evidence and recorded next to the wrong value rather than replacing it
silently. Both readings stay on the record.

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import time

# Fields every input must agree on for the cells to be one measurement.
SHARED_KEYS = ("pinned_sha", "base_ppl", "n_items_discovery", "n_kept_tokens_estimate",
               "n_image_tokens", "sites", "ft_pair_rule")


def _read(p):
    if str(p).startswith("s3://"):
        return json.loads(subprocess.run(["aws", "s3", "cp", str(p), "-"], check=True,
                                         capture_output=True, text=True).stdout)
    return json.loads(pathlib.Path(p).read_text())


def _cell_seeds(site_rec):
    """{seed -> the seed's record} for a completed site cell."""
    return {int(s["seed"]): s for s in site_rec.get("seeds", [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", required=True, help="comma list of Arm X metrics.json (s3 or local)")
    ap.add_argument("--steps-actual", type=int, required=True)
    ap.add_argument("--lr-actual", type=float, required=True)
    ap.add_argument("--hp-evidence", required=True,
                    help="how the true training setting was recovered, independent of the artifact")
    a = ap.parse_args()
    t0 = time.time()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    uris = [u.strip() for u in a.inputs.split(",") if u.strip()]
    arts = [(u, _read(u)) for u in uris]
    assert len(arts) >= 2, "merge of a single artifact is a copy, not a merge"
    # Shuffle-only artifacts ATTACH to measured cells, so they must be processed after every
    # artifact that could supply one. Ordering the inputs by hand would make the merge's result
    # depend on the caller's argument order, which is not a property a merge should have.
    arts.sort(key=lambda kv: any(s.get("shuffle_null_only")
                                 for p in kv[1]["pairs"].values()
                                 for s in p.get("sites", {}).values()))

    # ---- 1. provenance equality ------------------------------------------------------------------
    ref_u, ref = arts[0]
    for u, d in arts[1:]:
        for k in SHARED_KEYS:
            assert d.get(k) == ref.get(k), \
                f"provenance mismatch on '{k}': {ref_u} has {ref.get(k)!r}, {u} has {d.get(k)!r}"
    hashes = {}
    for u, d in arts:
        for pn, pr in d["pairs"].items():
            h = pr.get("input_hash")
            if pn in hashes:
                assert hashes[pn][0] == h, \
                    f"pair {pn}: input_hash {hashes[pn][0]} ({hashes[pn][1]}) != {h} ({u})"
            else:
                hashes[pn] = (h, u)

    # ---- 2. merge cells; duplicates become a reported replication --------------------------------
    pairs, dup = {}, []
    for u, d in arts:
        for pn, pr in d["pairs"].items():
            tgt = pairs.setdefault(pn, {"units": pr["units"], "input_hash": pr["input_hash"],
                                        "sites": {}})
            assert tgt["units"] == pr["units"], f"pair {pn}: unit disagreement across inputs"
            for L, s in pr.get("sites", {}).items():
                have = tgt["sites"].get(L)
                # A `--shuffle-only` re-run carries no real leg; it exists to supply the LOST-side
                # null that v1 discarded. It ATTACHES to the measured cell under its own key and
                # never replaces the banked `shuffle_null`, so the recovered numbers can always be
                # told apart from the original ones.
                if s.get("shuffle_null_only"):
                    assert have is not None and "seeds" in have, \
                        f"{pn}@L{L}: shuffle-only record with no measured cell to attach to"
                    have["shuffle_null_full"] = s["shuffle_null"]
                    have["shuffle_null_full_source"] = u
                    continue
                if have is None:
                    tgt["sites"][L] = dict(s, _source=u)
                    continue
                # Same (pair, site) measured twice. Compare every shared seed, then keep the
                # denser cell. Both provenances are recorded on the kept cell.
                a_s, b_s = _cell_seeds(have), _cell_seeds(s)
                shared = sorted(set(a_s) & set(b_s))
                for sd in shared:
                    for fld in ("emergent_fraction", "lost_fraction", "n_live"):
                        va, vb = a_s[sd].get(fld), b_s[sd].get(fld)
                        dup.append({"pair": pn, "site": L, "seed": sd, "field": fld,
                                    "a": va, "b": vb, "a_src": have.get("_source"), "b_src": u,
                                    "equal": bool(va == vb),
                                    "abs_diff": (abs(va - vb) if isinstance(va, (int, float))
                                                 and isinstance(vb, (int, float)) else None)})
                if len(b_s) > len(a_s):
                    tgt["sites"][L] = dict(s, _source=u, _also_measured_in=have.get("_source"))
                else:
                    tgt["sites"][L]["_also_measured_in"] = u

    # ---- 3. identity band from the POOLED identity seeds -----------------------------------------
    idv = [q["emergent_fraction"] for L, s in pairs.get("identity", {}).get("sites", {}).items()
           for q in s.get("seeds", [])]
    band = None
    if idv:
        import numpy as np
        band = {"p2.5": float(np.percentile(idv, 2.5)), "p50": float(np.percentile(idv, 50)),
                "p97.5": float(np.percentile(idv, 97.5)), "n": len(idv),
                "note": "pooled over every identity seed in every merged input"}

    # ---- 3b. the LOST identity band, and the frozen rule applied to LOST -------------------------
    # `tessera_x.py` v1 built a band for `emergent_fraction` only and scored only that, so LOST was
    # never put through `fraction_verdict` at all. This is not a new rule: it is THE frozen rule
    # (`tessera_core.fraction_verdict` — "a real fraction must clear that band") applied to the
    # quantity the erasure question turns on, using the identity pair the same way.
    from infra.pg import tessera_core as TC
    idl = [q["lost_fraction"] for s in pairs.get("identity", {}).get("sites", {}).values()
           for q in s.get("seeds", [])]
    band_lost = None
    if idl:
        b = TC.boot_ci(idl)
        band_lost = {"p2.5": b["lo95"], "p50": b["point"], "p97.5": b["hi95"], "n": len(idl)}
        for pn, pr in pairs.items():
            if pn == "identity":
                continue
            for L, s in pr["sites"].items():
                if "seeds" not in s:
                    continue
                m = sum(q["lost_fraction"] for q in s["seeds"]) / len(s["seeds"])
                s["mean_lost"] = m
                s["lost_verdict"] = TC.fraction_verdict(m, band_lost)

    # ---- 4. the LOST side, against its recovered null ---------------------------------------------
    # The spec's frozen gate is on `emergent_fraction` (the shuffle null was banked for it). LOST had
    # no null banked at all, so this comparison is a REPAIR of that gap and is reported DESCRIPTIVELY:
    # the numbers and their ratio, with no threshold invented after the fact. Cells with no recovered
    # null say so rather than being scored against nothing.
    lost_tab = []
    for pn, pr in pairs.items():
        for L, s in pr["sites"].items():
            if "seeds" not in s:
                continue
            m = sum(q["lost_fraction"] for q in s["seeds"]) / len(s["seeds"])
            full = s.get("shuffle_null_full")
            row = {"pair": pn, "site": L, "measured_lost": m,
                   "n_seeds": len(s["seeds"]),
                   "lost_verdict": (s.get("lost_verdict") or {}).get("verdict"),
                   "shuffle_lost": ([q["lost_fraction"] for q in full] if full else None),
                   "shuffle_emergent_banked": [q["emergent_fraction"]
                                               for q in s.get("shuffle_null", [])]}
            if full:
                sl = [q["lost_fraction"] for q in full]
                row["shuffle_lost_max"] = max(sl)
                row["ratio_measured_over_shuffle_max"] = (m / max(sl)) if max(sl) > 0 else None
                row["above_shuffle_max"] = bool(m > max(sl))
            else:
                row["note"] = ("no LOST null recovered for this cell — v1 discarded the shuffle "
                               "leg's lost_fraction; only cells re-run with --shuffle-only have one")
            lost_tab.append(row)

    n_cells = sum(1 for p in pairs.values() for s in p["sites"].values() if "seeds" in s)
    seed_counts = sorted({len(s["seeds"]) for p in pairs.values()
                          for s in p["sites"].values() if "seeds" in s})
    rep = {"object": "TESSERA Arm X — merged across jobs",
           "inputs": uris, "n_pairs": len(pairs), "n_site_cells": n_cells,
           "sites": ref["sites"], "ft_pair_rule": ref["ft_pair_rule"],
           "pinned_sha": ref["pinned_sha"], "base_ppl": ref["base_ppl"],
           "n_items_discovery": ref["n_items_discovery"],
           "n_kept_tokens_estimate": ref["n_kept_tokens_estimate"],
           "hyperparams": dict(ref["hyperparams"], steps=a.steps_actual, lr=a.lr_actual),
           "hyperparams_correction": {
               "as_banked_steps": ref["hyperparams"].get("steps"),
               "as_banked_lr": ref["hyperparams"].get("lr"),
               "true_steps": a.steps_actual, "true_lr": a.lr_actual,
               "evidence": a.hp_evidence,
               "note": "the banked Arm X artifacts recorded the A9 port CONSTANT rather than the "
                       "`--steps`/`--lr` the trainer was given. The measurement was unaffected — "
                       "the override reaches `train_crosscoder` — but the artifact under-reported "
                       "its own setting. Corrected here with independent evidence; the banked "
                       "value is kept beside it, not overwritten."},
           "identity_band": band, "identity_band_lost": band_lost,
           "lost_vs_shuffle": {
               "label": "DESCRIPTIVE — repairs a banked-null gap, not a frozen gate",
               "rows": lost_tab,
               "n_with_recovered_null": sum(1 for r in lost_tab if r.get("shuffle_lost")),
               "why": "Arm X's frozen shuffle gate was banked for `emergent_fraction` only. "
                      "`lost_fraction` — the one non-zero quantity the arm measures — was left "
                      "with no matched null because `tessera_x.py` v1 computed the shuffle leg's "
                      "full fractions and stored a single field of them."},
           "density_cut": {"seeds_per_cell_observed": seed_counts,
                           "full_density": {"seeds": 2, "sites": 2},
                           "cut_applied": bool(min(seed_counts) < 2) if seed_counts else None,
                           "rule": "§8/§12 pre-registered cut ladder. Cells differ in realised "
                                   "seed count; the per-cell count is on each cell, and the range "
                                   "is here. No cell is upgraded by the merge."},
           "duplicate_replication": {
               "n_comparisons": len(dup), "n_equal": sum(1 for x in dup if x["equal"]),
               "max_abs_diff": max([x["abs_diff"] for x in dup if x["abs_diff"] is not None],
                                   default=None),
               "comparisons": dup,
               "note": "cells measured in two separate jobs on two separate instances. Reported as "
                       "a cross-instance replication check, never used to pick a winner."},
           "pairs": pairs, "elapsed_s": time.time() - t0}
    (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))

    for pn, pr in pairs.items():
        for L, s in sorted(pr["sites"].items(), key=lambda kv: int(kv[0])):
            if "seeds" not in s:
                print(f"[merge] {pn:12s} L{L:<3s} INCOMPLETE {sorted(s)}", flush=True)
                continue
            em = [q["emergent_fraction"] for q in s["seeds"]]
            lo = [q["lost_fraction"] for q in s["seeds"]]
            sh = [q["emergent_fraction"] for q in s.get("shuffle_null", [])]
            print(f"[merge] {pn:12s} L{L:<3s} em={[round(x,4) for x in em]} "
                  f"lost={[round(x,4) for x in lo]} shuf={[round(x,4) for x in sh]}", flush=True)
    print(f"\nTESSERA_X_MERGE_OK cells={n_cells} dup_comparisons={len(dup)} "
          f"dup_equal={sum(1 for x in dup if x['equal'])}", flush=True)


if __name__ == "__main__":
    main()
