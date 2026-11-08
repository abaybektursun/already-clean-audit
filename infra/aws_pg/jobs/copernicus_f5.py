"""CO-F5 (item 50) — massive-activation-set drift vs fine-tuning intensity, per layer.

QUESTION (item 50, verbatim): "The massive-activation set MOVES under fine-tuning while the sink
stays fixed (k_proj overlap: base 0.385 -> refcoco 0.260, ordered by aggressiveness). Correlate
per-model overlap drop vs per-layer ||dW||; per-layer resolution." The statistic is frozen at §6.3:
Spearman(per-model outlier-overlap drop, per-layer ||dW||) with a bootstrap CI over models x layers
plus a shuffled-pairing null. MONOTONE fires iff the CI excludes 0 AND exceeds the shuffle band.
§11 carries it at mid confidence.

THE OUTLIER SET IS THE MASSIVE-ACTIVATION SET, and it is already banked. An earlier draft of this
job bound "outlier set" to the SPECTRAL sense -- `gauge_rmt.outlier_subspaces` -- on the reasoning
that it is what this repo's code calls an outlier set. That was wrong, and item 50's own text says
so: the numbers it cites (0.385, 0.260) are DARWIN W0-1c's `outlier_overlap_jaccard`, the Jaccard
between a bank and the massive-activation set. Reading the source item rather than inferring the
object from a function name is the difference; the numbers were in the item all along.

INPUT: `results/darwin_w0_1c/out/atlas.jsonl`, statistic `outlier_overlap_jaccard`, bank `core32`,
18 layers x 6 types per model. Drop is measured against the PG1 base cell-by-cell:

    drop(m, L, t) = jaccard(pg1-pt224, L, t) - jaccard(m, L, t)

and paired with ||W_m - W_base||_F at the SAME (L, t). No activation harvest happens here -- W0-1c
already paid for it.

WHAT THIS ARM CANNOT DO, STATED BEFORE THE RESULT. W0-1c harvested five models, of which exactly
TWO are PG1 fine-tunes (`pg1-ft-vqav2`, `pg1-ft-refcoco-seg`). `gemma-2b` is the LM ancestor and
`pi0-reshell` is a re-shelled VLA, neither of which is a fine-tune of the base whose ||dW|| the
item asks to correlate against. So the cell count is 2 x 18 x 6 = 216 and the MODEL dimension is
n=2. The Spearman is a cell-level statistic and is well-posed at that resolution; a claim about
fine-tunes IN GENERAL is not, and the per-model rho is reported separately so a reader can see
whether the two agree rather than having to trust that they do. The unlock is named and costed in
the artifact: harvest the massive-activation set for the other 22 manifest fine-tunes.

    python3 -m infra.aws_pg.jobs.copernicus_f5 --out out
"""
import argparse
import json
import pathlib
import subprocess

import numpy as np

from infra.pg import copernicus_core as C
from infra.pg import humboldt_core as H
from infra.aws_pg.jobs.invariant_core import (
    PG1_REPO, _load_decoder, _decoder_matrices, _ftype, _delete_cache,
)
from infra.aws_pg.jobs.copernicus_a1_banks import load_manifest, MANIFEST

ARM = "CO-F5"
QUEUE_ITEM = "50"
WAVE = "WF"
W01C_ATLAS = ("s3://pg-lineage-769063704696/results/darwin_w0_1c/out/atlas.jsonl")
STATISTIC = "outlier_overlap_jaccard"
BANK = "core32"
BASE_MODEL = "pg1-pt224"
# W0-1c's model_id -> the manifest repo whose delta against the base gives ||dW||. Only genuine
# PG1 fine-tunes appear: gemma-2b is the LM ancestor and pi0-reshell is a re-shelled VLA, so
# neither has a fine-tuning ||dW|| against this base to correlate against.
FT_MODELS = {"pg1-ft-vqav2": "google/paligemma-3b-ft-vqav2-224",
             "pg1-ft-refcoco-seg": "google/paligemma-3b-ft-refcoco-seg-224"}
MIN_CELLS = 100
N_BOOT = 2000
N_SHUFFLE = 2000
SEED = 1337


def spearman(x, y):
    """Spearman rho via Pearson on ranks. Ties get average ranks."""
    def rank(v):
        v = np.asarray(v, dtype=np.float64)
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=np.float64)
        r[order] = np.arange(len(v), dtype=np.float64)
        # average ranks within tied groups, or ties bias the correlation
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt)); np.add.at(sums, inv, r)
        return (sums / cnt)[inv]
    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    assert den > 0, "a constant series has no rank correlation"
    return float((rx * ry).sum() / den)


def bootstrap_ci(x, y, n=N_BOOT, seed=SEED, lo=2.5, hi=97.5):
    """Percentile CI over resampled (drop, ‖ΔW‖) PAIRS -- pairs, so the association is what is
    resampled and not the two margins independently."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    draws = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            continue                      # a degenerate resample has no rank correlation
        draws.append(spearman(x[idx], y[idx]))
    d = np.array(draws)
    return {"n_draws": len(d), "mean": float(d.mean()),
            "ci_lo": float(np.percentile(d, lo)), "ci_hi": float(np.percentile(d, hi))}


def shuffle_band(x, y, n=N_SHUFFLE, seed=SEED, lo=2.5, hi=97.5):
    """The null the frozen rule names: break the PAIRING, keep both margins. This controls for the
    marginal distributions of drop and ‖ΔW‖ -- including that both are layer-structured -- which a
    normal-theory p-value would not."""
    rng = np.random.default_rng(seed + 1)
    x, y = np.asarray(x), np.asarray(y)
    d = np.array([spearman(x, y[rng.permutation(len(y))]) for _ in range(n)])
    return {"n_draws": int(n), "mean": float(d.mean()),
            "lo": float(np.percentile(d, lo)), "hi": float(np.percentile(d, hi))}


def load_atlas(uri=W01C_ATLAS):
    """{model_id: {(layer, type): jaccard}} from W0-1c's atlas — the banked massive-activation
    overlap, read by hash-free S3 path because it is a FINISHED arm's immutable output."""
    r = subprocess.run(["aws", "s3", "cp", uri, "-", "--region", "us-east-1"], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout, f"W0-1c atlas unreadable: {r.stderr[:300]}"
    out = {}
    for line in r.stdout.splitlines():
        d = json.loads(line)
        if d.get("statistic") != STATISTIC or d.get("bank_id") != BANK:
            continue
        key = (d["layer"], d["type"])
        assert key not in out.setdefault(d["model_id"], {}), f"duplicate cell {d['model_id']} {key}"
        out[d["model_id"]][key] = float(d["value"])
    return out


def verdict(ci, shuf, drops):
    """§6.3: MONOTONE iff the CI excludes 0 AND exceeds the shuffle band.

    Guarded first by a degeneracy branch. A drop series with no spread cannot rank anything, and a
    Spearman computed on it would be reporting the ORDER OF ||dW|| against noise — the ceiling
    §6.5's one-sided-threshold rule requires, in the form this arm can actually hit."""
    spread = float(np.max(drops) - np.min(drops))
    if spread <= 0.0 or len(np.unique(drops)) < 3:
        return "VOID-INSTRUMENT", {"reason": "the overlap-drop series has no usable spread",
                                   "spread": spread, "n_unique": int(len(np.unique(drops)))}
    excludes_zero = ci["ci_lo"] > 0.0 or ci["ci_hi"] < 0.0
    exceeds_null = ci["ci_lo"] > shuf["hi"]
    if excludes_zero and exceeds_null:
        return "MONOTONE", {"ci": ci, "shuffle_hi": shuf["hi"]}
    return "REFUTER-FIRED", {"excludes_zero": excludes_zero, "exceeds_null": exceeds_null,
                             "ci": ci, "shuffle": shuf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--manifest-sha256", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    H.hf_setup()

    man = load_manifest(a.manifest, a.manifest_sha256)
    base_sha, shas = man["base_sha"], man["shas"]
    for repo in FT_MODELS.values():
        assert repo in shas, f"{repo} is not in the frozen delta manifest"

    atlas = load_atlas()
    assert BASE_MODEL in atlas, f"W0-1c atlas has no {BASE_MODEL} cells"
    base_ov = atlas[BASE_MODEL]
    print(f"[f5] atlas: {len(atlas)} models, {len(base_ov)} base cells", flush=True)

    rep = {"object": "copernicus_f5_v1", "arm": ARM, "program": "copernicus",
           "queue_item": QUEUE_ITEM, "bank": BANK, "statistic": STATISTIC,
           "base_model": BASE_MODEL, "ft_models": FT_MODELS,
           "base_repo": PG1_REPO, "base_sha": base_sha, "manifest_sha256": man["sha256"],
           "input_artifact": W01C_ATLAS,
           "outlier_binding": ("MASSIVE-ACTIVATION: DARWIN W0-1c outlier_overlap_jaccard, the "
                               "Jaccard between the core32 bank and the massive-activation set. "
                               "This is the object item 50 names; its cited 0.385/0.260 are cells "
                               "of this artifact."),
           "power_bound": ("W0-1c harvested 5 models, only 2 of them PG1 fine-tunes, so the model "
                           "dimension is n=2 and the Spearman is a CELL-level statistic. Unlock: "
                           "harvest the massive-activation set for the other 22 manifest "
                           "fine-tunes."),
           "provenance": C.provenance(), "utc": C.utc_now(), "rows": []}

    def save():
        (out / "copernicus_f5.json").write_text(json.dumps(rep, indent=2, default=float))
    save()

    import torch
    w_base = _load_decoder(PG1_REPO, base_sha)
    for mid, repo in FT_MODELS.items():
        assert mid in atlas, f"W0-1c atlas has no {mid} cells"
        print(f"[f5] {mid} <- {repo}", flush=True)
        w_ft = _load_decoder(repo, shas[repo])
        matched = _decoder_matrices(w_base, w_ft)
        for key, (wb, wf) in sorted(matched.items()):
            t = _ftype(key)
            L = int(key.split("layers.")[1].split(".")[0])
            cell = (L, t)
            if cell not in base_ov or cell not in atlas[mid]:
                continue                      # a site W0-1c did not harvest (e.g. down_proj_in)
            dw = float(torch.linalg.norm((wf.to(torch.float64) - wb.to(torch.float64))).item())
            assert dw > 0, f"zero ΔW {key} in {repo}"
            rep["rows"].append({"model": mid, "layer": L, "type": t, "dW_fro": dw,
                                "jaccard_base": base_ov[cell], "jaccard_ft": atlas[mid][cell],
                                "drop": base_ov[cell] - atlas[mid][cell]})
        del w_ft, matched
        _delete_cache(repo)
        save()

    drops = [r["drop"] for r in rep["rows"]]
    norms = [r["dW_fro"] for r in rep["rows"]]
    rep["n_cells"] = len(drops)
    rep["n_models"] = len(FT_MODELS)
    assert len(drops) >= MIN_CELLS, f"only {len(drops)} paired cells; expected >= {MIN_CELLS}"
    rep["rho"] = spearman(drops, norms)
    rep["bootstrap"] = bootstrap_ci(drops, norms)
    rep["shuffle_null"] = shuffle_band(drops, norms)
    # per-model rho, so a two-model agreement (or disagreement) is visible rather than assumed
    rep["per_model_rho"] = {
        m: spearman([r["drop"] for r in rep["rows"] if r["model"] == m],
                    [r["dW_fro"] for r in rep["rows"] if r["model"] == m])
        for m in FT_MODELS}
    rep["verdict"], rep["evidence"] = verdict(rep["bootstrap"], rep["shuffle_null"], drops)
    save()
    print(f"[f5] n={rep['n_cells']} rho={rep['rho']:.4f} per_model={rep['per_model_rho']} "
          f"verdict={rep['verdict']}", flush=True)


if __name__ == "__main__":
    main()
