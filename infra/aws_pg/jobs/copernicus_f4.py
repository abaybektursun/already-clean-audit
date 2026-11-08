"""CO-F4 (item 49, scoped) — the L9 dossier. CHARTER: a described artifact, no verdict.

QUESTION. Item 49 began as "characterize the lone L9 ultra-avoided direction — the single bottom
eigenvector separated from the rest of the spectrum, the most-avoided direction in the model, never
individually examined" and was UPGRADED per section K of the queue addendum to "what is L9", because
five instruments converge there. This job assembles the dossier from banked artifacts only.

EVERY NUMBER IS RECOMPUTED FROM AN ARTIFACT. Item 49's prose carries figures (a 4.4x bottom
separation, a 7.6x position-0 spike) that are NOT reproduced here and must not be read off this
job's output: the program's baseline-discipline rule is that a number enters a report from
`metrics.json`/weights/git or not at all. Where a quantity is reachable it is recomputed; where it
is not, the gap is named as a gap. That is why the dossier reports no eigenvalue separation — the
HUMBOLDT banks publish the bottom EIGENVECTORS (`humboldt/banks/<type>__L<k>__core32.npy`) and not
the spectrum they came from, so the "4.4x" that names this direction "lone" cannot be re-derived
from what is banked. Recomputing it needs the 24-delta Gram, which is a job, not a dossier.

WHAT L9 IS COMPARED AGAINST. A dossier that reports L9 alone cannot show that L9 is special. Every
statistic is computed at L0, L9 and L17 — the depths HUMBOLDT banked and the mining note contrasts
— so a reader can see the depth profile instead of taking "L9 is where things converge" on trust.

SOURCES (all immutable outputs of finished arms):
  * `humboldt/banks/{q_proj,down_proj}__L{0,9,17}__{core32,anti32,haar32_s0}.npy` — the read-side
    (column) and write-side (row) bottom subspaces.
  * `humboldt/humboldt_b1/atlas.parquet` — `e_U_mean`, the activation energy each bank carries.
  * `results/darwin_w0_1c/out/atlas.jsonl` — `outlier_overlap_jaccard`, overlap with the
    massive-activation set.

    python3 -m infra.aws_pg.jobs.copernicus_f4 --out out
"""
import argparse
import io
import json
import pathlib
import subprocess

import numpy as np

from infra.pg import copernicus_core as C

ARM = "CO-F4"
QUEUE_ITEM = "49"
WAVE = "WF"
REGISTER = "CHARTER (described artifact, no verdict)"

BUCKET = "s3://pg-lineage-769063704696"
BANKS = f"{BUCKET}/humboldt/banks"                    # column/input-side frames
ROW_BANKS = f"{BUCKET}/mercator/banks"                # row/output-side frames
HB1_ATLAS = f"{BUCKET}/humboldt/humboldt_b1/atlas.parquet"
W01C_ATLAS = f"{BUCKET}/results/darwin_w0_1c/out/atlas.jsonl"

LAYERS = (0, 9, 17)
READ, WRITE = "q_proj", "down_proj"
TOPK = 32


def s3_bytes(uri):
    r = subprocess.run(["aws", "s3", "cp", uri, "-", "--region", "us-east-1"], capture_output=True)
    assert r.returncode == 0 and r.stdout, f"unreadable: {uri}"
    return r.stdout


def bank(type_, layer, name="core32", side="col"):
    """A banked orthonormal frame. `side` picks the SPACE, and picking it wrong is not a detail:
    HUMBOLDT's `down_proj__L9__core32` is a 16384-dim INPUT-side frame (the MLP intermediate),
    while the residual axes q_proj's column frame lives on are 2048-dim. Comparing coordinate
    indices across the two is comparing labels from different spaces — it returns 0 shared at every
    layer, which looks like a measurement and is an arithmetic guarantee. The output-side frames
    live in mercator/banks under a `__row__` infix."""
    uri = (f"{ROW_BANKS}/{type_}__L{layer}__row__{name}.npy" if side == "row"
           else f"{BANKS}/{type_}__L{layer}__{name}.npy")
    A = np.load(io.BytesIO(s3_bytes(uri)))
    err = float(np.abs(A.T @ A - np.eye(A.shape[1])).max())
    assert err < 1e-10, f"{type_} L{layer} {name} not orthonormal ({err:.2e})"
    return A


def direction_profile(u):
    """How concentrated a single unit direction is on the coordinate basis.

    participation_ratio = 1/sum(u^4) for a unit u: 1 = the direction IS one coordinate, 2048 = it
    is spread evenly over all of them. Reported alongside the max component because a large max can
    coexist with a long tail, and the two together say which."""
    u = np.asarray(u, dtype=np.float64)
    u = u / np.linalg.norm(u)
    w = u ** 2
    order = np.argsort(-w)
    return {"max_component": float(np.abs(u).max()),
            "argmax_coord": int(np.abs(u).argmax()),
            "participation_ratio": float(1.0 / (w ** 2).sum()),
            "top5_coords": [int(c) for c in order[:5]],
            "top5_weight": [float(w[c]) for c in order[:5]],
            "mass_in_top5": float(w[order[:5]].sum())}


def bank_leverage(A):
    """Per-coordinate leverage of a k-frame: the squared row norms, which sum to k. A coordinate
    with leverage near 1 is almost entirely inside the subspace."""
    lev = (np.asarray(A, dtype=np.float64) ** 2).sum(axis=1)
    order = np.argsort(-lev)
    return {"max": float(lev.max()), "argmax_coord": int(lev.argmax()),
            "topk_coords": [int(c) for c in order[:TOPK]], "d": int(A.shape[0]),
            "haar_expectation": float(A.shape[1] / A.shape[0])}


def coord_agreement(a, b):
    """Shared top-k coordinates between the read-side and write-side banks, against the count
    expected by chance for two independent k-subsets of d coordinates (k^2/d).

    Refuses frames of different ambient dimension. A coordinate INDEX only means the same thing in
    both frames if both index the same space, and the first cut of this arm compared a 2048-dim
    column frame against a 16384-dim input frame and reported 0/32 shared at every depth."""
    assert a["d"] == b["d"], (
        f"coordinate agreement across different spaces: d={a['d']} vs d={b['d']}")
    sa, sb = set(a["topk_coords"]), set(b["topk_coords"])
    return {"shared": len(sa & sb), "k": TOPK, "d": a["d"],
            "jaccard": len(sa & sb) / len(sa | sb),
            "chance_shared": TOPK * TOPK / float(a["d"])}


def energy_rows(pq_bytes):
    import pandas as pd
    df = pd.read_parquet(io.BytesIO(pq_bytes))
    return df[df.statistic == "e_U_mean"]


def energy_at(df, layer, type_, model="pg1-pt224"):
    """Activation energy the bank carries, and its enrichment over the Haar banks measured on the
    SAME rows. The enrichment is the point: a raw e_U is not interpretable without the null that
    HUMBOLDT banked next to it."""
    sel = df[(df.layer == layer) & (df.type == type_) & (df.model_id == model)]
    assert len(sel), f"no e_U rows for L{layer} {type_} {model}"
    got = {}
    for b in ("core32", "anti32", "haar32_s0", "haar32_s1", "haar32_s2"):
        v = sel[sel.bank_id == b]["value"]
        assert len(v), f"no {b} rows at L{layer} {type_}"
        got[b] = float(v.mean())          # mean over the corpora HUMBOLDT harvested
    haar = float(np.mean([got[f"haar32_s{i}"] for i in range(3)]))
    return {"core32": got["core32"], "anti32": got["anti32"], "haar_mean": haar,
            "core_over_haar": got["core32"] / haar, "anti_over_haar": got["anti32"] / haar,
            "n_corpora": int(len(sel[sel.bank_id == "core32"]))}


def outlier_overlap(jsonl_bytes, layer, type_, bank_id="core32"):
    """Jaccard against the massive-activation set, per model, from W0-1c."""
    out = {}
    for line in jsonl_bytes.decode().splitlines():
        d = json.loads(line)
        if (d.get("statistic") == "outlier_overlap_jaccard" and d.get("bank_id") == bank_id
                and d.get("layer") == layer and d.get("type") == type_):
            out[d["model_id"]] = float(d["value"])
    assert out, f"no W0-1c overlap rows at L{layer} {type_}"
    return out


CONVERGENCE = [
    {"instrument": "GESTALT shape content", "claim_lives_in": "project_gestalt_verdict",
     "artifact": "studies/lineage/paligemma1-pi0/research (GESTALT result)",
     "recomputed_here": False},
    {"instrument": "the lone ultra-avoided direction", "claim_lives_in": "item 49",
     "artifact": f"{BANKS}/{READ}__L9__core32.npy", "recomputed_here": True},
    {"instrument": "position-decodability peak", "claim_lives_in": "H3 / DARWIN",
     "artifact": "darwin position-decodability arm", "recomputed_here": False},
    {"instrument": "read-column arrival", "claim_lives_in": "2026-08-14 director mining note (R1d)",
     "artifact": f"{BANKS}/{READ}__L*__core32.npy + {ROW_BANKS}/{WRITE}__L*__row__core32.npy",
     "recomputed_here": True},
    {"instrument": "the position-0 activation spike", "claim_lives_in": "H3 / S2",
     "artifact": "humboldt e_U_mean_by_class", "recomputed_here": False},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    rep = {"object": "copernicus_f4_v1", "arm": ARM, "program": "copernicus",
           "queue_item": QUEUE_ITEM, "register": REGISTER, "layers": list(LAYERS),
           "read_type": READ, "write_type": WRITE, "topk": TOPK,
           "sources": {"col_banks": BANKS, "row_banks": ROW_BANKS,
                       "humboldt_b1": HB1_ATLAS, "w0_1c": W01C_ATLAS},
           "not_recomputed": {
               "bottom_eigenvalue_separation": (
                   "item 49's 4.4x is NOT reproduced: the banks publish the bottom eigenvectors, "
                   "not the spectrum. Re-deriving it needs the 24-delta Gram — a job, not a "
                   "dossier."),
               "position_0_spike": (
                   "item 49's 7.6x is NOT reproduced: it is an e_U_mean_by_class quantity whose "
                   "token-class definition is not bound to a frozen label set here (see CO-F3's "
                   "park for the same missing object).")},
           "convergence_record": CONVERGENCE,
           "provenance": C.provenance(), "utc": C.utc_now(), "cells": {}}

    def save():
        (out / "copernicus_f4.json").write_text(json.dumps(rep, indent=2, default=float))
    save()

    print("[f4] fetching humboldt b1 atlas", flush=True)
    edf = energy_rows(s3_bytes(HB1_ATLAS))
    print("[f4] fetching w0-1c atlas", flush=True)
    w01c = s3_bytes(W01C_ATLAS)

    for L in LAYERS:
        print(f"[f4] L{L}", flush=True)
        read_bank, write_bank = bank(READ, L), bank(WRITE, L, side="row")
        rl, wl = bank_leverage(read_bank), bank_leverage(write_bank)
        rep["cells"][f"L{L}"] = {
            "read": {"lone_direction": direction_profile(read_bank[:, 0]), "bank_leverage": rl,
                     "activation_energy": energy_at(edf, L, READ),
                     "massive_activation_overlap": outlier_overlap(w01c, L, READ)},
            "write": {"lone_direction": direction_profile(write_bank[:, 0]), "bank_leverage": wl,
                      "activation_energy": None,      # down_proj_in is not a Phase-B site (W0-1c)
                      "massive_activation_overlap": None},
            "read_write_coordinate_agreement": coord_agreement(rl, wl),
        }
        save()

    # depth profile, so "L9 is special" is a reading of the table rather than an assertion over it
    rep["depth_profile"] = {
        "read_lone_max_component": {f"L{L}": rep["cells"][f"L{L}"]["read"]["lone_direction"]["max_component"] for L in LAYERS},
        "read_bank_leverage_max": {f"L{L}": rep["cells"][f"L{L}"]["read"]["bank_leverage"]["max"] for L in LAYERS},
        "write_bank_leverage_max": {f"L{L}": rep["cells"][f"L{L}"]["write"]["bank_leverage"]["max"] for L in LAYERS},
        "read_write_shared_coords": {f"L{L}": rep["cells"][f"L{L}"]["read_write_coordinate_agreement"]["shared"] for L in LAYERS},
        "core_energy_over_haar": {f"L{L}": rep["cells"][f"L{L}"]["read"]["activation_energy"]["core_over_haar"] for L in LAYERS},
    }
    save()
    print(f"[f4] depth profile: {json.dumps(rep['depth_profile'], default=float)}", flush=True)


if __name__ == "__main__":
    main()
