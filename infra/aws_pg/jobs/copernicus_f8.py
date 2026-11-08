"""CO-F8 (item 62; curiosity reserve) — the biography of coordinate 1967. CHARTER, no verdict.

QUESTION. Coordinate 1967 is the rank-1 leverage coordinate of the down_proj WRITE bank at both L0
and L9, and the reader picks it up only by L9. It has been named repeatedly and never traced. This
job reads every banked bottom-subspace frame that lives on the 2048 residual axes and asks where
1967 sits in each.

THE CONTROL IS THE POINT, and it is the reason this arm is worth its (small) cost rather than a
victory lap. "1967 is rank-1 in cell X" is unreadable on its own: some coordinate is rank-1 in every
cell. So the job computes, for EVERY coordinate, the number of cells in which it takes rank 1, and
reports the leaderboard. If 1967 tops it by a wide margin it is a distinguished axis of this model;
if a dozen coordinates score alike, then "the 1967 story" is one draw from a population of equally
loaded axes and should stop being told as a singular one. The house failure mode is premature
unification around an elegant object, and a lone coordinate with a memorable number is exactly the
shape that failure takes.

ONLY 2048-DIM FRAMES ARE ADMITTED. A coordinate index means "residual axis 1967" only in a frame
whose ambient space IS the residual. HUMBOLDT's down_proj column frame is 16384-dim (the MLP
intermediate) and MERCATOR's gate/up row frames likewise; index 1967 there names a different thing
entirely. CO-F4 reported a clean 0/32 agreement at every depth by comparing across exactly this
gap (journal D102), so admission is by measured `d`, and every excluded cell is recorded with its
dimension rather than dropped.

    python3 -m infra.aws_pg.jobs.copernicus_f8 --out out
"""
import argparse
import collections
import io
import json
import pathlib
import subprocess

import numpy as np

from infra.pg import copernicus_core as C

ARM = "CO-F8"
QUEUE_ITEM = "62"
WAVE = "WF"
REGISTER = "CHARTER (described artifact, no verdict)"

BUCKET = "s3://pg-lineage-769063704696"
COL_BANKS = f"{BUCKET}/humboldt/banks"
ROW_BANKS = f"{BUCKET}/mercator/banks"
TYPES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LAYERS = tuple(range(18))
RESIDUAL_D = 2048
SUBJECT = 1967
K = 32


def s3_npy(uri):
    r = subprocess.run(["aws", "s3", "cp", uri, "-", "--region", "us-east-1"],
                       capture_output=True)
    if r.returncode != 0 or not r.stdout:
        return None                      # a (type, layer, side) combination that was never banked
    return np.load(io.BytesIO(r.stdout))


def leverage(A):
    """Per-coordinate leverage of an orthonormal k-frame: squared row norms, summing to k."""
    return (np.asarray(A, dtype=np.float64) ** 2).sum(axis=1)


def cell_record(A, coord=SUBJECT):
    """Where `coord` sits in this frame: its leverage, its rank, and what the top of the frame
    looks like — so a rank can be read against the leverage that earned it."""
    assert 0 <= coord < A.shape[0], (
        f"coordinate {coord} does not exist in a {A.shape[0]}-dim frame")
    lev = leverage(A)
    # stable sort: ties are measure-zero in real frames, but an unstable tie-break would make the
    # rank (and the leaderboard's argmax) vary between runs on identical input.
    order = np.argsort(-lev, kind="mergesort")
    rank = int(np.where(order == coord)[0][0]) + 1
    return {"d": int(A.shape[0]), "k": int(A.shape[1]),
            "coord_leverage": float(lev[coord]), "coord_rank": rank,
            "max_leverage": float(lev.max()), "argmax_coord": int(lev.argmax()),
            "haar_expectation": float(A.shape[1] / A.shape[0]),
            "leverage_over_haar": float(lev[coord] / (A.shape[1] / A.shape[0]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--coord", type=int, default=SUBJECT)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    rep = {"object": "copernicus_f8_v1", "arm": ARM, "program": "copernicus",
           "queue_item": QUEUE_ITEM, "register": REGISTER, "subject_coord": a.coord,
           "residual_d": RESIDUAL_D, "k": K,
           "sources": {"col_banks": COL_BANKS, "row_banks": ROW_BANKS},
           "provenance": C.provenance(), "utc": C.utc_now(),
           "cells": [], "excluded": []}

    def save():
        (out / "copernicus_f8.json").write_text(json.dumps(rep, indent=2, default=float))
    save()

    rank1_counts = collections.Counter()
    for side in ("col", "row"):
        for t in TYPES:
            for L in LAYERS:
                uri = (f"{ROW_BANKS}/{t}__L{L}__row__core32.npy" if side == "row"
                       else f"{COL_BANKS}/{t}__L{L}__core32.npy")
                A = s3_npy(uri)
                if A is None:
                    rep["excluded"].append({"side": side, "type": t, "layer": L,
                                            "reason": "not banked"})
                    continue
                if A.shape[0] != RESIDUAL_D:
                    rep["excluded"].append({"side": side, "type": t, "layer": L,
                                            "reason": "not a residual-space frame",
                                            "d": int(A.shape[0])})
                    continue
                err = float(np.abs(A.T @ A - np.eye(A.shape[1])).max())
                assert err < 1e-10, f"{side} {t} L{L} not orthonormal ({err:.2e})"
                rec = cell_record(A, a.coord)
                rec.update({"side": side, "type": t, "layer": L})
                rep["cells"].append(rec)
                rank1_counts[rec["argmax_coord"]] += 1
            save()
        print(f"[f8] {side}: {len(rep['cells'])} residual cells so far", flush=True)

    cells = rep["cells"]
    assert cells, "no residual-space frames were admitted"
    ranks = np.array([c["coord_rank"] for c in cells])
    rep["summary"] = {
        "n_cells": len(cells),
        "n_excluded": len(rep["excluded"]),
        "n_rank1": int((ranks == 1).sum()),
        "n_top32": int((ranks <= K).sum()),
        "median_rank": float(np.median(ranks)),
        "mean_leverage_over_haar": float(np.mean([c["leverage_over_haar"] for c in cells])),
        "by_side": {s: {"n": int(sum(1 for c in cells if c["side"] == s)),
                        "n_rank1": int(sum(1 for c in cells
                                           if c["side"] == s and c["coord_rank"] == 1))}
                    for s in ("col", "row")},
    }
    # THE CONTROL: some coordinate is rank-1 in every cell, so 1967's count is only readable
    # against the whole leaderboard.
    rep["rank1_leaderboard"] = [{"coord": int(c), "n_cells_rank1": int(n)}
                                for c, n in rank1_counts.most_common(10)]
    rep["subject_rank1_share"] = (rank1_counts[a.coord] / len(cells)) if cells else 0.0
    rep["n_distinct_rank1_coords"] = len(rank1_counts)
    save()
    print(f"[f8] {json.dumps(rep['summary'], default=float)}", flush=True)
    print(f"[f8] leaderboard {json.dumps(rep['rank1_leaderboard'])}", flush=True)


if __name__ == "__main__":
    main()
