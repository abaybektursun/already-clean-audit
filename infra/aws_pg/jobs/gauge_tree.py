"""GAUGE-TREE (P4) — ultrametricity of the 24-FT PG1 ΔW family (CPU, --no-weights).

Streams the 24 fine-tunes exactly as invariant_core.py / gauge_cs.py do (enumerate -224 live, pin
SHAs, load base once, difference per decoder matrix in fp64, fp16 scratch, delete cache). PASS 2
assembles the 24×24 raw inner-product Gram(s) one matrix at a time — pooled (all 126 matrices) and
per-matrix-type — from which `gauge_tree.gram_to_distance` builds the per-FT-normalised Frobenius
distance matrix (§7.2). Then, per distance matrix: real CCC + triple statistic + both null CCC
distributions (matched-Gaussian-cloud, label-permutation) + δ_UM cuts, and the within-largest-task-
cluster re-test (labels = `task_distance` from 2026-06-02_n5c_cloud.json::members, frozen at Gate-0;
UNTESTABLE if the largest cluster < 8). SINGLE-SHOT at n=24 (§7.2): the rule + both nulls + the
within-cluster control are frozen and null-calibrated before the real distance matrix is read; the
statistic is computed once. Disclosed exploratory-strong at best.

The optional S_action/S_vl read-family rows (§7.2, banked on branch `science/inherit`) are attempted
ONLY when --draw-bank points at a banked .npz of orthonormal draws; missing ⇒ that path asserts loud,
the primary ΔW-family rows are unaffected (frozen pivot: read-family descriptive-only).

  # smoke (2 ft, q_proj only — Gram/CCC/nulls on a tiny family):
  pg run infra.aws_pg.jobs.gauge_tree --type r7i.4xlarge --no-weights \
      --args "--ft-tasks vqav2,gqa,coco,ai2d --types q_proj --n-null 100" --budget-min 60
  # full (24 ft, all 7 types, pooled + per-type):
  pg run infra.aws_pg.jobs.gauge_tree --type r7i.8xlarge --no-weights \
      --args "--n-null 200" --budget-min 600
"""
import argparse
import json
import pathlib
import shutil

from infra.pg import gauge_common as GC
from infra.pg import gauge_tree as TREE
# REUSED streaming spine (invariant_core.py): base/ft decoder load, matrix matching, cache delete.
from infra.aws_pg.jobs.invariant_core import (PG1_REPO, PROJ, _load_decoder, _decoder_matrices,
                                              _ftype, _delete_cache)

N5C_META = ("studies/lineage/paligemma1-pi0/research/2026-06-02_n5c_cloud.json")


def _task_labels(repos, meta_path):
    """Per-FT task-type cluster labels (the P4 within-cluster control, §7.2), read from the frozen
    N5c roster metadata `members[*].task_distance` and aligned to the live `repos` order. `meta_path`
    is a LOCAL path the caller has already materialised (the pg code package ships only infra/, so a
    studies/ file must be fetched from S3 first — done in main). RAISES if a live-enumerated repo is
    absent from the frozen roster (family drift — must fail loud, never silently drop the control)."""
    p = pathlib.Path(meta_path)
    assert p.exists(), f"N5c roster metadata not found at {p}"
    meta = json.loads(p.read_text())
    m = {d["repo_id"]: d["task_distance"] for d in meta["members"]}
    for r in repos:
        assert r in m, f"repo {r} missing from frozen N5c roster — task labels stale, fix Gate-0"
    return [m[r] for r in repos]


def _analyze_matrix(name, G, labels, n_null, seed, min_cluster=8):
    """One distance matrix (pooled or a per-matrix-type Gram) → the full P4 record: full-family
    CCC-with-nulls, the within-largest-cluster control (or UNTESTABLE), and the detection verdict."""
    import numpy as np
    D = TREE.gram_to_distance(G)
    full = TREE.ccc_with_nulls(D, n_null=n_null, seed=seed)
    lc = TREE.largest_cluster(labels, min_size=min_cluster)
    if lc["testable"]:
        idx = np.asarray(lc["indices"])
        sub = TREE.ccc_with_nulls(D[np.ix_(idx, idx)], n_null=n_null, seed=seed + 101)
        within = TREE.beats_both(sub)
        control = {"testable": True, "cluster_label": lc["label"], "cluster_size": lc["size"],
                   "within_cluster": sub, "within_cluster_excess": bool(within)}
    else:
        within = None
        control = {"testable": False, "cluster_label": lc["label"], "cluster_size": lc["size"],
                   "reason": f"largest task cluster {lc['size']} < {min_cluster} (UNTESTABLE, frozen rule)"}
    det = TREE.detection(full, within)
    return {"matrix": name, "n": int(D.shape[0]), "full": full, "cluster_control": control,
            "detection_grade": det}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ft-tasks", default="", help="comma slugs; empty ⇒ enumerate all -224 (24-FT)")
    ap.add_argument("--types", default=",".join(PROJ), help="matrix types entering the pooled/type Grams")
    ap.add_argument("--n-null", type=int, default=200, help="null CCC draws per matrix (both nulls)")
    ap.add_argument("--min-cluster", type=int, default=8, help="min members for a TESTABLE cluster control")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--draw-bank", default="", help="optional .npz of banked S_action/S_vl orthonormal "
                    "draws (read-family rows, §7.2); missing ⇒ asserts loud, ΔW rows unaffected")
    # The pg code package ships only infra/, so the studies/ roster metadata (task_distance cluster
    # labels) must come from S3 at run time. Default = the uploaded copy; local path also accepted.
    ap.add_argument("--cluster-meta",
                    default="s3://pg-lineage-769063704696/meta/2026-06-02_n5c_cloud.json",
                    help="n5c_cloud.json (s3:// or local) — members[*].task_distance cluster labels")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    want_types = tuple(t.strip() for t in a.types.split(",") if t.strip())
    assert all(t in PROJ for t in want_types), f"unknown type in {want_types}"

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()                       # FIRST — gated repos 401 otherwise
    import numpy as np
    import torch
    from huggingface_hub import HfApi
    api = HfApi()

    if a.ft_tasks.strip():
        repos = [f"google/paligemma-3b-ft-{t.strip()}-224" for t in a.ft_tasks.split(",") if t.strip()]
    else:
        repos = sorted({m.id for m in api.list_models(author="google", search="paligemma-3b-ft")
                        if (m.id.split("/")[-1].endswith("-224") and "-jax" not in m.id
                            and m.id.split("/")[-1].startswith("paligemma-3b-ft-"))})
    shas = {r: api.model_info(r).sha for r in repos}
    base_sha = api.model_info(PG1_REPO).sha
    # materialise the cluster-label metadata locally (fetch from S3 if needed — studies/ is not in
    # the code package), then read the Gate-0-frozen task-type labels aligned to `repos`.
    meta_local = a.cluster_meta
    if a.cluster_meta.startswith("s3://"):
        import subprocess
        meta_local = str(out / "n5c_cloud.json")
        subprocess.run(["aws", "s3", "cp", a.cluster_meta, meta_local, "--region", "us-east-1"],
                       check=True)
    labels = _task_labels(repos, meta_local)   # Gate-0 frozen task-type labels, aligned to repos
    n_ft = len(repos)
    print(f"[gauge-tree] base={PG1_REPO}@{base_sha[:8]} n_ft={n_ft} types={want_types} "
          f"labels={sorted(set(labels))}", flush=True)

    rep = {"object": "gauge_tree_v1", "generation": "gemma1_18L", "base_repo": PG1_REPO,
           "base_sha": base_sha, "cloud_repos": repos, "cloud_shas": shas, "n_ft": n_ft,
           "task_labels": {r: l for r, l in zip(repos, labels)}, "types": list(want_types),
           "n_null": a.n_null, "min_cluster": a.min_cluster, "seed": a.seed, "design": "single_shot",
           "estimand": "cophenetic correlation (average-linkage) of the per-FT-normalised Frobenius "
                       "distance matrix of the 24-FT ΔW family vs matched-cloud + label-permutation nulls",
           "disclosure": "n=24 single-shot; exploratory-strong at best; promotion = external FT family"}

    w_base = _load_decoder(PG1_REPO, base_sha)

    # ---- PASS 1: difference every FT → fp16 scratch per (type,layer,ridx) (invariant_core pattern).
    scratch = out / "scratch"
    scratch.mkdir(exist_ok=True)
    for ridx, repo_id in enumerate(repos):
        print(f"[gauge-tree] diff {repo_id}@{shas[repo_id][:8]} ({ridx+1}/{n_ft})", flush=True)
        w_ft = _load_decoder(repo_id, shas[repo_id])
        matched = _decoder_matrices(w_base, w_ft)
        if ridx == 0:
            GC.assert_gemma1(list(matched.keys()))          # firewall (18-layer)
        for key, (wb, wf) in matched.items():
            t = _ftype(key)
            if t not in want_types:
                continue
            layer = GC.layer_index(key)
            dW = (wf.to(torch.float64) - wb.to(torch.float64)).cpu().numpy()
            np.save(scratch / f"{t}__L{layer}__{ridx}.npy", dW.astype(np.float16))
        del w_ft, matched
        _delete_cache(repo_id)

    # ---- PASS 2: assemble the raw inner-product Grams one matrix at a time.
    # G_ij = Σ_matrices ⟨ΔW_i^m, ΔW_j^m⟩ ; pooled sums all wanted matrices, per-type sums that type.
    pooled_G = np.zeros((n_ft, n_ft), dtype=np.float64)
    type_G = {t: np.zeros((n_ft, n_ft), dtype=np.float64) for t in want_types}
    for t in want_types:
        layers = sorted({int(f.stem.split("__L")[1].split("__")[0])
                         for f in scratch.glob(f"{t}__L*__0.npy")})
        for layer in layers:
            rows = []
            for ridx in range(n_ft):
                f = scratch / f"{t}__L{layer}__{ridx}.npy"
                assert f.exists(), f"missing scratch {f} (a dropped FT breaks the Gram)"
                rows.append(np.load(f).astype(np.float64).ravel())
            X = np.stack(rows, axis=0)                       # (n_ft, m*n) fp64
            Gc = X @ X.T                                     # (n_ft, n_ft) inner-product contribution
            pooled_G += Gc
            type_G[t] += Gc
            del X, rows
            print(f"[gauge-tree] gram {t} L{layer}", flush=True)

    shutil.rmtree(scratch)

    # ---- adjudicate: pooled first (primary), then per matrix-type.
    records = [_analyze_matrix("pooled", pooled_G, labels, a.n_null, a.seed, a.min_cluster)]
    for t in want_types:
        records.append(_analyze_matrix(t, type_G[t], labels, a.n_null, a.seed + 7, a.min_cluster))
    for r in records:
        f = r["full"]
        c = r["cluster_control"]
        wc = c.get("within_cluster_excess", "UNTESTABLE") if c["testable"] else "UNTESTABLE"
        print(f"[gauge-tree] {r['matrix']:10s} CCC={f['ccc']:+.4f} "
              f"δ_cloud={f['cloud_delta_um']:+.4f} δ_perm={f['perm_delta_um']:+.4f} "
              f"triple={f['triple_statistic']:.4f} within={wc} → detect={r['detection_grade']}", flush=True)
    rep["records"] = records

    # ---- OPTIONAL exploratory read-family rows (S_action/S_vl), only if a draw bank is supplied.
    if a.draw_bank.strip():
        bank = pathlib.Path(a.draw_bank)
        assert bank.exists(), f"--draw-bank {bank} not found (read-family rows require the banked draws)"
        npz = np.load(bank, allow_pickle=True)
        read_rows = []
        for fam in npz.files:                                # each entry: (n_draws, d, k) orthonormal draws
            draws = npz[fam]
            assert draws.ndim == 3, f"{fam}: expected (n_draws,d,k) orthonormal draws, got {draws.shape}"
            n_draws = draws.shape[0]
            if n_draws < 6:                                  # frozen pivot: <6 draws ⇒ descriptive-only
                read_rows.append({"family": fam, "n_draws": int(n_draws), "status": "descriptive_only"})
                continue
            D = TREE.chordal_distance_matrix([draws[i] for i in range(n_draws)])
            read_rows.append({"family": fam, "n_draws": int(n_draws), "status": "analyzed",
                              "full": TREE.ccc_with_nulls(D, n_null=a.n_null, seed=a.seed + 13)})
        rep["read_family_rows"] = read_rows
        rep["read_family_note"] = "exploratory/descriptive-only (§7.2); no detection verdict issued"

    (out / "gauge_tree.json").write_text(json.dumps(rep, indent=2))
    n_detect = sum(1 for r in records if r["detection_grade"])
    print("=" * 78)
    print(f"GAUGE_TREE_OK n_ft={n_ft} n_matrices={len(records)} detection_grade={n_detect} "
          f"(pooled CCC={records[0]['full']['ccc']:+.4f})")


if __name__ == "__main__":
    main()
