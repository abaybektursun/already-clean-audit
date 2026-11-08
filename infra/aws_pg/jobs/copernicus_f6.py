"""CO-F6 (item 80) — PLENUM's survival question, asked with M-C1's feature set.

QUESTION. PLENUM's §4.1d recorded THE LAW: across 24 fine-tunes of one base, which faculties
survive is not predicted by recipe membership, by ||dW||_LM, or by task similarity — all three
leave-one-unit-out deltas negative. M-C1 later built a feature set that predicts DAMAGE 1.6x
better than no model. Item 80 aims that working feature set at the old open question. §6.3's
frozen rule: LOO dMAE vs predict-the-mean; BEATS iff the improvement's CI excludes 0 — the
null gate M-C2's lesson made mandatory.

THE TWO ARMS RAN ON DISJOINT POPULATIONS, which is why this is a job and not a join. M-C1's roster
came from `darwin_census.select_derivatives`, a Hub census that returned 23 COMMUNITY fine-tunes
(DepositorOP, NYUAD-ComNets, eagle0504, ...). PLENUM's 24 units are the OFFICIAL google
`paligemma-3b-ft-*-224` family. The intersection of the two rosters is EMPTY — verified, not
assumed: 0 of 23 M-C1 repos appear in the frozen delta manifest, while 24 of PLENUM's units do. So
M-C1's fitted coefficients cannot be transferred; its FEATURE DEFINITIONS can, and this job
recomputes them on PLENUM's own roster.

The features are imported from `mercator_c1`, not reimplemented — `features()` and its `BANKS`
tuple verbatim, so "M-C1's feature set" means the same eleven numbers it meant there.

NO GPU AND NO PERPLEXITY PASS. M-C1 needed a forward pass because its target was log_damage.
F6's target is PLENUM's banked survival score S, so the only work is 24 deltas against the base.

LEAVE-ONE-UNIT-OUT, NOT LEAVE-ONE-ROW-OUT. Each unit contributes 3 faculty rows; holding out a
single row would leave the same fine-tune's other two rows in the training set and let the model
see the unit it is being scored on. `loo_ridge_by_unit` mirrors `mercator_c1.loo_ridge` in every
respect (same standardisation, same unpenalised intercept, no-model baseline refit per fold) and
differs only in grouping the held-out fold by unit, which is also how PLENUM's own law was scored.

    python3 -m infra.aws_pg.jobs.copernicus_f6 --out out --manifest-sha256 <sha>
"""
import argparse
import json
import pathlib
import subprocess

import numpy as np
import torch

from infra.pg import copernicus_core as C
from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as WT
from infra.pg import mercator_core as M
from infra.aws_pg.jobs.mercator_c1 import BANKS, features
from infra.aws_pg.jobs.invariant_core import _delete_cache
from infra.aws_pg.jobs.copernicus_a1_banks import load_manifest, MANIFEST

ARM = "CO-F6"
QUEUE_ITEM = "80"
WAVE = "WF"
PLENUM_METRICS = ("s3://pg-lineage-769063704696/results/plenum_agg_fc/out/metrics.json")
TYPES = ("q_proj", "o_proj")            # M-C1's scope, kept for a like-for-like feature set
FACULTIES = ("vqa", "ocr", "caption")   # PLENUM's law readout
READOUT = "forced_choice"
ALPHA = 1.0
N_BOOT = 2000
SEED = 1337
FEAT_NAMES = [f"share_{b}" for b in BANKS] + ["log_motion", "eff_rank", "n_cells_moved"]


def s3_json(uri):
    r = subprocess.run(["aws", "s3", "cp", uri, "-", "--region", "us-east-1"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout, f"unreadable: {uri}"
    return json.loads(r.stdout)


def unit_of(repo):
    """`google/paligemma-3b-ft-ai2d-224` -> `ft-ai2d`, PLENUM's unit key."""
    tail = repo.split("/")[-1]
    assert tail.startswith("paligemma-3b-") and tail.endswith("-224"), f"unexpected repo {repo}"
    return tail[len("paligemma-3b-"):-len("-224")]


def survival_targets(metrics):
    """{(unit, faculty): S} from PLENUM's banked cells — the continuous survival score its own law
    was scored on, not the categorical label."""
    out = {}
    for c in metrics["cells"]:
        if c.get("readout") != READOUT or c.get("faculty") not in FACULTIES:
            continue
        key = (c["unit"], c["faculty"])
        assert key not in out, f"duplicate PLENUM cell {key}"
        out[key] = float(c["S"])
    assert out, "no PLENUM forced-choice cells matched"
    return out


def loo_ridge_by_unit(X, y, units, alpha=ALPHA):
    """Leave-one-UNIT-out ridge, and the no-model (training-mean) prediction beside it.

    Mirrors mercator_c1.loo_ridge — same standardisation, same unpenalised intercept, baseline
    refit inside each fold so the comparison is cross-validated on both sides. The only difference
    is the fold: every row belonging to the held-out unit leaves together, or a model scored on one
    faculty of a fine-tune would have trained on that fine-tune's other two."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    units = np.asarray(units)
    uniq = sorted(set(units.tolist()))
    assert len(uniq) > X.shape[1] + 1, (
        f"{len(uniq)} units against {X.shape[1]} features; leave-one-unit-out ridge would "
        f"interpolate rather than generalise")
    pred, base = np.zeros(len(y)), np.zeros(len(y))
    for u in uniq:
        te = units == u
        tr = ~te
        Xt, yt = X[tr], y[tr]
        mu, sd = Xt.mean(0), Xt.std(0)
        sd[sd == 0] = 1.0
        Z = (Xt - mu) / sd
        A = np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=1)
        R = alpha * np.eye(A.shape[1])
        R[-1, -1] = 0.0
        w = np.linalg.solve(A.T @ A + R, A.T @ yt)
        Zi = np.concatenate([(X[te] - mu) / sd, np.ones((te.sum(), 1))], axis=1)
        pred[te] = Zi @ w
        base[te] = yt.mean()
    return pred, base


def delta_mae(y, pred, base, units, n=N_BOOT, seed=SEED):
    """Improvement in MAE over predict-the-mean, with a CI bootstrapped over UNITS — the
    independent sampling unit here is the fine-tune, not the row."""
    y, pred, base, units = map(np.asarray, (y, pred, base, units))
    mae_m = float(np.abs(y - pred).mean())
    mae_0 = float(np.abs(y - base).mean())
    uniq = sorted(set(units.tolist()))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        pick = rng.choice(len(uniq), len(uniq))
        idx = np.concatenate([np.flatnonzero(units == uniq[i]) for i in pick])
        draws.append(float(np.abs(y[idx] - base[idx]).mean() - np.abs(y[idx] - pred[idx]).mean()))
    d = np.array(draws)
    return {"mae_model": mae_m, "mae_no_model": mae_0, "improvement": mae_0 - mae_m,
            "ci_lo": float(np.percentile(d, 2.5)), "ci_hi": float(np.percentile(d, 97.5)),
            "n_draws": n, "n_units": len(uniq), "n_rows": len(y)}


def verdict(dm):
    """§6.3: BEATS iff the improvement's CI excludes 0. A negative improvement whose CI also
    excludes 0 is WORSE-THAN-MEAN — recorded as its own outcome rather than folded into 'does not
    beat', because a feature set that actively hurts is a different finding from one that adds
    nothing."""
    if dm["ci_lo"] > 0.0:
        return "BEATS", {"improvement": dm["improvement"], "ci": [dm["ci_lo"], dm["ci_hi"]]}
    if dm["ci_hi"] < 0.0:
        return "WORSE-THAN-MEAN", {"improvement": dm["improvement"],
                                   "ci": [dm["ci_lo"], dm["ci_hi"]]}
    return "REFUTER-FIRED", {"improvement": dm["improvement"],
                             "ci": [dm["ci_lo"], dm["ci_hi"]],
                             "reason": "the improvement's CI includes zero"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--manifest-sha256", required=True)
    ap.add_argument("--layers", default="")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    H.hf_setup()
    layers = [int(x) for x in a.layers.split(",") if x.strip()] or list(range(H.N_LAYERS))

    man = load_manifest(a.manifest, a.manifest_sha256)
    repos, shas, base_sha = man["repos"], man["shas"], man["base_sha"]
    targets = survival_targets(s3_json(PLENUM_METRICS))

    rep = {"object": "copernicus_f6_v1", "arm": ARM, "program": "copernicus",
           "queue_item": QUEUE_ITEM, "types": list(TYPES), "layers": layers,
           "faculties": list(FACULTIES), "readout": READOUT, "alpha": ALPHA,
           "features": FEAT_NAMES, "manifest_sha256": man["sha256"],
           "input_artifact": PLENUM_METRICS,
           "roster_note": ("M-C1's fitted coefficients are NOT transferred: its roster was 23 "
                           "community fine-tunes with an EMPTY intersection against this "
                           "manifest. Its feature DEFINITIONS are imported verbatim and "
                           "recomputed on PLENUM's own 24 units."),
           "provenance": C.provenance(), "utc": C.utc_now(), "rows": []}

    def save():
        (out / "copernicus_f6.json").write_text(json.dumps(rep, indent=2, default=float))
    save()

    bank_dir = M.fetch_all_banks(H.cache_dir(out, "banks"))
    keys = [(t, L) for t in TYPES for L in layers]
    banks = {k: {b: WT.load_bank(bank_dir, k[0], k[1], b) for b in BANKS} for k in keys}
    base_mats = M.load_decoder_matrices(WT.PG1_REPO, base_sha, TYPES, layers)
    print(f"[f6] banks for {len(keys)} cells; {len(targets)} PLENUM targets", flush=True)

    feats = {}
    for i, repo in enumerate(repos):
        unit = unit_of(repo)
        if not any((unit, f) in targets for f in FACULTIES):
            rep["rows"].append({"repo": repo, "unit": unit, "skipped": "no PLENUM cell"})
            save()
            continue
        print(f"[f6] {repo} ({i + 1}/{len(repos)})", flush=True)
        ft_mats = M.load_decoder_matrices(repo, shas[repo], TYPES, layers)
        # load_decoder_matrices returns torch Tensors; the .to(float64).numpy() conversion is
        # M-C1's own (mercator_c1 line ~180) and is kept identical so `features` sees the same
        # dtype and precision it saw there.
        dW = {k: (ft_mats[k].to(torch.float64) - base_mats[k].to(torch.float64)).numpy()
              for k in keys}
        feats[unit] = features(dW, banks)
        rep["rows"].append({"repo": repo, "unit": unit, **feats[unit]})
        del ft_mats, dW
        _delete_cache(repo)
        save()

    X, y, units, cells = [], [], [], []
    for unit, f in sorted(feats.items()):
        for fac in FACULTIES:
            if (unit, fac) not in targets:
                continue
            X.append([f[n] for n in FEAT_NAMES])
            y.append(targets[(unit, fac)])
            units.append(unit)
            cells.append({"unit": unit, "faculty": fac, "S": targets[(unit, fac)]})
    rep["n_units"] = len(feats)
    rep["n_rows_fit"] = len(y)
    rep["cells"] = cells
    save()

    pred, base = loo_ridge_by_unit(X, y, units)
    dm = delta_mae(y, pred, base, units)
    rep["delta_mae"] = dm
    rep["per_faculty"] = {}
    for fac in FACULTIES:
        m = np.array([c["faculty"] == fac for c in cells])
        if m.sum() < 3:
            continue
        rep["per_faculty"][fac] = {
            "n": int(m.sum()),
            "mae_model": float(np.abs(np.array(y)[m] - pred[m]).mean()),
            "mae_no_model": float(np.abs(np.array(y)[m] - base[m]).mean())}
    rep["predictions"] = [{**c, "forecast": float(p), "no_model": float(b)}
                          for c, p, b in zip(cells, pred, base)]
    rep["verdict"], rep["evidence"] = verdict(dm)
    save()
    print(f"[f6] {json.dumps(dm, default=float)}", flush=True)
    print(f"[f6] verdict={rep['verdict']}", flush=True)


if __name__ == "__main__":
    main()
