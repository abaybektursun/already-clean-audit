"""CO-F10 (item 51 carry-forward) — M-9a's two eigenvector universals. CHARTER-plus-carry.

M-9a recorded BOTH of these cells as COVERAGE-IMPOSSIBLE, each with a named unlock:

  cross_layer_verticality        unlock: "re-run the census for the foreign families with
                                 eigenvector banking"           -> DONE (mercator/recensus/banks)
  published_outlier_index_       unlock: "same re-census, PLUS a cite-verified index list from
  comparison                     the source paper"              -> NOT DONE

§3 orders this arm explicitly: FIRST verify the recensus banked eigenvectors — the unlock
condition, never assume — THEN compute both cells on banked data. This job does exactly that and
reports the second cell as still COVERAGE-IMPOSSIBLE, because no cite-verified published index list
exists in this repo and inventing one would be a fabricated citation, not a result.

VERIFICATION IS BY RECOMPUTATION, NOT BY PRESENCE. "The banks exist" is not the unlock condition;
"the banks are the object M-9a needed" is. The re-census banked both the eigenvector frames
(`core32`) and the derived spike-coordinate INDEX sets (`spike_coords`) — the exact quantity whose
absence made this cell impossible. So the check recomputes `spike_coords` from each `core32` frame
with `mercator_recensus.spike_coords` and asserts it reproduces the banked index set exactly. A
frame that is not orthonormal, or whose recomputed indices disagree, fails the arm loudly rather
than quietly producing a verticality number from the wrong object.

BOTH STATISTICS ARE IMPORTED, not reimplemented: `spike_coords` and `verticality` come from
`mercator_recensus`, so "M-9a's verticality" is M-9a's.

SCOPE, DISCLOSED: the re-census banked `q_proj` only, at 8-10 sampled layers per base. This is a
cross-layer statistic on one type, not the per-type account CO-X2/X3 carry.

    python3 -m infra.aws_pg.jobs.copernicus_f10 --out out
"""
import argparse
import collections
import io
import json
import pathlib
import re
import subprocess

import numpy as np

from infra.pg import copernicus_core as C
from infra.aws_pg.jobs.mercator_recensus import spike_coords, verticality, SPIKE_MULT

ARM = "CO-F10"
QUEUE_ITEM = "51"
WAVE = "WF"
REGISTER = "CHARTER (described artifact) + one carried COVERAGE-IMPOSSIBLE cell"
BANKS = "s3://pg-lineage-769063704696/mercator/recensus/banks"
NAME_RE = re.compile(r"^(.*?)__([a-z_]+_proj)__L(\d+)__(\w+)\.npy$")


def s3_ls(prefix):
    r = subprocess.run(["aws", "s3", "ls", prefix + "/", "--region", "us-east-1"],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"cannot list {prefix}: {r.stderr[:300]}"
    return [line.split()[-1] for line in r.stdout.splitlines() if line.strip().endswith(".npy")]


def s3_npy(uri):
    r = subprocess.run(["aws", "s3", "cp", uri, "-", "--region", "us-east-1"],
                       capture_output=True)
    assert r.returncode == 0 and r.stdout, f"unreadable: {uri}"
    return np.load(io.BytesIO(r.stdout))


def chance_verticality(coords_by_layer, d):
    """Expected shared-coordinate fraction for spike sets of the OBSERVED sizes placed at random.

    Verticality is not readable without this. The raw statistic tracks spike COUNT: a base whose
    layers each flag 80 coordinates has far more opportunity to share than one flagging 2, so an
    ordering of bases by raw verticality can inherit an ordering by spike count and say nothing
    about universality. For two random subsets of sizes a and b drawn from d coordinates,
    E|A n B| = ab/d, and the shared fraction is that over |A u B| = a + b - ab/d.
    """
    layers = sorted(coords_by_layer)
    vals = []
    for i, x in enumerate(layers):
        for y in layers[i + 1:]:
            a, b = len(coords_by_layer[x]), len(coords_by_layer[y])
            if a == 0 or b == 0:
                continue                      # the same pairs verticality() calls undefined
            inter = a * b / float(d)
            vals.append(inter / (a + b - inter))
    return {"mean": (float(np.mean(vals)) if vals else None), "n_pairs": len(vals),
            "d": int(d),
            "note": "expected shared fraction for random sets of the observed per-layer sizes"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    names = s3_ls(BANKS)
    parsed = []
    for n in names:
        m = NAME_RE.match(n)
        assert m, f"unparseable bank name {n}"
        base, t, L, bank_id = m.groups()
        parsed.append((base, t, int(L), bank_id, n))

    rep = {"object": "copernicus_f10_v1", "arm": ARM, "program": "copernicus",
           "queue_item": QUEUE_ITEM, "register": REGISTER, "banks": BANKS,
           "spike_mult": SPIKE_MULT, "n_bank_objects": len(parsed),
           "provenance": C.provenance(), "utc": C.utc_now(),
           "unlock_verification": {}, "verticality": {}}

    def save():
        (out / "copernicus_f10.json").write_text(json.dumps(rep, indent=2, default=float))
    save()

    cores = {(b, t, L): n for b, t, L, bid, n in parsed if bid == "core32"}
    banked_idx = {(b, t, L): n for b, t, L, bid, n in parsed if bid == "spike_coords"}
    assert cores, "no core32 frames in the recensus bank library"
    print(f"[f10] {len(cores)} core32 frames, {len(banked_idx)} banked index sets", flush=True)

    coords, checks, dims = collections.defaultdict(dict), [], {}
    for key in sorted(cores):
        base, t, L = key
        U = s3_npy(f"{BANKS}/{cores[key]}")
        err = float(np.abs(U.T @ U - np.eye(U.shape[1])).max())
        recomputed = spike_coords(U)
        row = {"base": base, "type": t, "layer": L, "d": int(U.shape[0]), "k": int(U.shape[1]),
               "orthonormality_err": err, "n_spikes": len(recomputed)}
        assert err < 1e-8, f"{cores[key]} is not orthonormal ({err:.2e})"
        if key in banked_idx:
            banked = [int(x) for x in np.asarray(s3_npy(f"{BANKS}/{banked_idx[key]}")).ravel()]
            row["banked_n_spikes"] = len(banked)
            row["reproduces_banked_indices"] = (sorted(banked) == list(recomputed))
            assert row["reproduces_banked_indices"], (
                f"{key}: recomputed spike indices differ from the banked set — the banks are not "
                f"the object M-9a's unlock names")
        else:
            row["reproduces_banked_indices"] = None
        checks.append(row)
        coords[(base, t)][L] = recomputed
        dims[(base, t)] = int(U.shape[0])
    rep["unlock_verification"] = {
        "n_frames": len(checks),
        "n_index_sets_reproduced": sum(1 for r in checks if r["reproduces_banked_indices"]),
        "n_without_banked_index_set": sum(1 for r in checks
                                          if r["reproduces_banked_indices"] is None),
        "max_orthonormality_err": max(r["orthonormality_err"] for r in checks),
        "rows": checks,
        "verdict": "UNLOCK-VERIFIED"}
    save()

    for (base, t), by_layer in sorted(coords.items()):
        if len(by_layer) < 2:
            rep["verticality"][f"{base}|{t}"] = {"skipped": "fewer than two banked layers"}
            continue
        v = verticality(by_layer)
        d = dims[(base, t)]
        ch = chance_verticality(by_layer, d)
        v["chance"] = ch
        v["enrichment_over_chance"] = (
            (v["mean_shared_fraction"] / ch["mean"])
            if (v["mean_shared_fraction"] is not None and ch["mean"]) else None)
        v["n_spikes_total"] = int(sum(len(c) for c in by_layer.values()))
        v["n_spikes_mean_per_layer"] = float(np.mean([len(c) for c in by_layer.values()]))
        rep["verticality"][f"{base}|{t}"] = v
        save()

    rep["published_outlier_index_comparison"] = {
        "outcome": "COVERAGE-IMPOSSIBLE",
        "carried_from": "M-9a",
        "evidence": ("the re-census unlock is satisfied — spike INDICES are banked and reproduce "
                     "from the frames — but the comparison also needs a cite-verified published "
                     "index list, and no such list exists in this repo"),
        "unlock": "a cite-verified outlier-coordinate index list from the source paper",
        "note": ("not run rather than run-and-failed: inventing the published indices would be a "
                 "fabricated citation, not a result")}
    save()
    for k, v in rep["verticality"].items():
        if "skipped" in v:
            continue
        print(f"[f10] {k}: mean_shared={v['mean_shared_fraction']} "
              f"chance={v['chance']['mean']} enrich={v['enrichment_over_chance']} "
              f"spikes/layer={v['n_spikes_mean_per_layer']:.1f} "
              f"({v['n_defined_pairs']}/{v['n_pairs']} pairs defined)", flush=True)


if __name__ == "__main__":
    main()
