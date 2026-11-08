"""GUTENBERG GB-A2 (the same-object CEILING) + GB-B (the λ-knob spectrum) — one pass, one job.

Spec AM-0.6.1. Two numbers the corpus does not contain, from the same 24 delta loads:

  GB-A2  GB-A found the 24-FT invariant core and the g2b census core DISTINCT (median subspace
         Jaccard 0.0586) against a matched-rotation FLOOR (0.0080). It has no CEILING: nothing
         says what two independent estimates of the SAME object score. Split the frozen 24-delta
         manifest into two disjoint halves of 12 (repos sorted, even index -> A, odd -> B), build
         each half's q_proj consensus Gram, take each bottom-32 eigenspace, and measure the SAME
         statistic. That is the ceiling at family size 12 (a lower bound on the ceiling at 24).
         Frozen rule: gutenberg_core.gba2_ceiling_verdict (registered before this datum existed).

  GB-B   the draft's "no eigengap cliff (λ32/λ31 ≈ 1.00–1.01)" sentence has NO column-side
         artifact anywhere (verified 2026-08-19: only a row-side λ33/λ32 = 1.001–1.023 exists).
         copernicus_a1_banks computed these eigenvalues and printed only min/max. Bank them.

G_full = G_A + G_B, so one pass over the deltas yields all three Grams and the FAIL-FAST
PREFLIGHT (#12) comes free: G_full's bottom-32 must reproduce the BANKED a1_banks core/32/L at
J >= 0.99 on all 18 layers before any new number is trusted. Eigenvector sign/order conventions
make byte-hashes brittle across BLAS builds, so the invariant (the subspace) is the gate; the
banked hash is recorded beside it either way.

    pg run infra.aws_pg.jobs.gutenberg_gbb_spectrum --no-weights --type r7i.2xlarge,m7i.2xlarge \
        --args "--manifest-sha256 <sha>" --s3-prefix results/gutenberg_gba2 --budget-min 240
"""
import argparse
import json
import pathlib
import subprocess

import numpy as np

from infra.pg import copernicus_core as C
from infra.pg import gutenberg_core as G
from infra.pg.humboldt_core import hf_setup

from infra.aws_pg.jobs.copernicus_a1_banks import MANIFEST, load_manifest
from infra.aws_pg.jobs.invariant_core import (
    N_LAYERS, PG1_REPO, _decoder_matrices, _delete_cache, _ftype, _load_decoder)

A1_NPZ = "s3://pg-lineage-769063704696/results/copernicus_a1_banks/out/a1_banks.npz"
A1_MANIFEST = "s3://pg-lineage-769063704696/results/copernicus_a1_banks/out/a1_banks_manifest.json"
# The frozen rule's other input is GB-A's banked median. It is READ FROM ITS ARTIFACT at run time,
# never transcribed — a hand-copied value here was wrong in the 4th decimal on the first attempt,
# which is precisely the failure the house provenance rule exists to prevent.
GBA_VERDICT_S3 = ("s3://pg-lineage-769063704696/results/gutenberg_gba_v2/out/"
                  "gutenberg_gba_overlap.json")
N_IN, K = 2048, 32
REPRO_BAR = 0.99


def _fetch(s3_uri, dst):
    subprocess.run(["aws", "s3", "cp", s3_uri, str(dst), "--region", "us-east-1",
                    "--only-show-errors"], check=True)
    return dst


def half_grams(repos, shas, w_base, layers, half_of):
    """ONE pass over the deltas; accumulate the two disjoint halves' Grams separately.
    Accumulation is VERBATIM copernicus_a1_banks.consensus_gram's (ΔWᵀΔW / ‖ΔW‖²_F, fp64)."""
    import torch
    GA = {L: np.zeros((N_IN, N_IN), dtype=np.float64) for L in layers}
    GB = {L: np.zeros((N_IN, N_IN), dtype=np.float64) for L in layers}
    n = {"A": 0, "B": 0}
    for ridx, repo_id in enumerate(repos):
        sha = shas[repo_id]
        h = half_of[repo_id]
        print(f"[gba2] diff {repo_id}@{sha[:8]} ({ridx + 1}/{len(repos)}) -> half {h}", flush=True)
        w_ft = _load_decoder(repo_id, sha)
        matched = _decoder_matrices(w_base, w_ft)
        tgt = GA if h == "A" else GB
        for L in layers:
            key = f"layers.{L}.self_attn.q_proj.weight"
            wb, wf = matched[key]
            assert _ftype(key) == "q_proj"
            A = wf.to(torch.float64) - wb.to(torch.float64)
            fro2 = float((A ** 2).sum())
            assert fro2 > 0, f"zero ΔW {key} in {repo_id}"
            tgt[L] += (A.transpose(0, 1) @ A).cpu().numpy().astype(np.float64) / fro2
        n[h] += 1
        del w_ft, matched
        _delete_cache(repo_id)
    assert n["A"] == n["B"] == len(repos) // 2, n
    return GA, GB, n


def bottom_k(Gm, k):
    evals, V = np.linalg.eigh(Gm)                 # ascending
    return np.ascontiguousarray(V[:, :k]), evals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--manifest-sha256", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wk = pathlib.Path("/tmp/gutenberg_gba2")
    wk.mkdir(parents=True, exist_ok=True)

    hf_setup()
    man = load_manifest(a.manifest, a.manifest_sha256)
    repos, shas, base_sha = man["repos"], man["shas"], man["base_sha"]
    layers = list(range(N_LAYERS))

    # frozen split (AM-0.6.1): sorted repos, even index -> A, odd -> B
    ordered = sorted(repos)
    half_of = {r: ("A" if i % 2 == 0 else "B") for i, r in enumerate(ordered)}

    # banked reference for the preflight
    a1man = json.loads(_fetch(A1_MANIFEST, wk / "a1_manifest.json").read_text())
    a1npz = np.load(_fetch(A1_NPZ, wk / "a1_banks.npz"))
    banked_hash = {b["layer"]: b["sha256"] for b in a1man["banks"]
                   if b["bank"] == "core" and b["k"] == K}
    gba = json.loads(_fetch(GBA_VERDICT_S3, wk / "gba_overlap.json").read_text())
    gba_median_j = float(gba["median_jaccard"])
    assert gba["verdict"] in ("DISTINCT", "AMBIGUOUS", "SAME-OBJECT"), gba["verdict"]
    print(f"[gba2] GB-A banked: verdict={gba['verdict']} median_J={gba_median_j:.6f}", flush=True)

    w_base = _load_decoder(PG1_REPO, base_sha)
    GA, GB, n_half = half_grams(repos, shas, w_base, layers, half_of)

    per_layer, J_AB, J_Afull, J_Bfull, repro = {}, [], [], [], []
    for L in layers:
        Gf = GA[L] + GB[L]
        UA, _ = bottom_k(GA[L], K)
        UB, _ = bottom_k(GB[L], K)
        Uf, lam = bottom_k(Gf, K)          # lam ascending, full spectrum
        Ubank = np.asarray(a1npz[f"core/{K}/{L}"], dtype=np.float64)

        j_repro = G.subspace_jaccard(Uf, Ubank)
        repro.append({"layer": L, "jaccard_vs_banked": j_repro,
                      "hash_match": C.bank_hash(Uf) == banked_hash[L],
                      "banked_sha256": banked_hash[L]})
        j_ab = G.subspace_jaccard(UA, UB)
        J_AB.append(j_ab)
        J_Afull.append(G.subspace_jaccard(UA, Uf))
        J_Bfull.append(G.subspace_jaccard(UB, Uf))
        per_layer[str(L)] = {
            "jaccard_halves": j_ab,
            "jaccard_A_vs_full": J_Afull[-1], "jaccard_B_vs_full": J_Bfull[-1],
            "jaccard_full_vs_banked": j_repro,
            "lam_bottom64": lam[:64].tolist(), "lam_top8": lam[-8:].tolist(),
            "lam32_over_lam31": float(lam[31] / lam[30]),
            "lam33_over_lam32": float(lam[32] / lam[31]),
        }
        print(f"[gba2] L{L}: repro J={j_repro:.5f} | ceiling J(A,B)={j_ab:.4f} | "
              f"λ32/λ31={lam[31] / lam[30]:.4f} λ33/λ32={lam[32] / lam[31]:.4f}", flush=True)
        GA[L] = GB[L] = None

    # ---- FAIL-FAST PREFLIGHT: reproduce the banked object before trusting anything new ----
    worst = min(r["jaccard_vs_banked"] for r in repro)
    assert worst >= REPRO_BAR, (
        f"PREFLIGHT FAILED: full-24 bottom-32 does not reproduce the banked a1 core "
        f"(worst layer J={worst:.4f} < {REPRO_BAR}) — the Gram path or the manifest moved")
    print(f"[gba2] preflight PASSED: worst reproduction J={worst:.5f}, "
          f"{sum(r['hash_match'] for r in repro)}/18 exact hash matches", flush=True)

    null_med, null_per = G.rotation_null_median_jaccard(N_IN, K, N_LAYERS, 1000, seed=20260819)
    verdict = G.gba2_ceiling_verdict(J_AB, gba_median_j)
    ratios32 = [per_layer[str(L)]["lam32_over_lam31"] for L in layers]
    ratios33 = [per_layer[str(L)]["lam33_over_lam32"] for L in layers]

    payload = {
        "object": "gutenberg_gba2_ceiling_and_spectrum_v1",
        "estimand": "GB-A2: subspace Jaccard (k=32) between the bottom-32 q_proj consensus-Gram "
                    "eigenspaces of two DISJOINT 12-fine-tune halves of the frozen 24-delta "
                    "manifest, per layer — the same-object ceiling at family size 12. "
                    "GB-B: eigenvalues of the full 24-FT consensus Gram, fp64.",
        "ceiling_verdict": verdict,
        "rotation_null_floor": {"median_p95": float(np.percentile(null_med, 95)),
                                "per_layer_p95": float(np.percentile(null_per, 95))},
        "ceiling_summary": {"median": float(np.median(J_AB)), "min": float(min(J_AB)),
                            "max": float(max(J_AB)),
                            "median_A_vs_full": float(np.median(J_Afull)),
                            "median_B_vs_full": float(np.median(J_Bfull))},
        "preflight_reproduction": {"bar": REPRO_BAR, "worst_jaccard": worst, "per_layer": repro},
        "lam32_over_lam31": {"min": min(ratios32), "median": float(np.median(ratios32)),
                             "max": max(ratios32)},
        "lam33_over_lam32": {"min": min(ratios33), "median": float(np.median(ratios33)),
                             "max": max(ratios33)},
        "split": {"rule": "repos sorted lexicographically; even index -> A, odd -> B",
                  "half_of": half_of, "n_per_half": n_half},
        "base_repo": PG1_REPO, "base_sha": base_sha,
        "manifest_sha256": man["sha256"], "n_ft": len(repos),
        "gba_banked_median_jaccard": gba_median_j,
        "gba_source": GBA_VERDICT_S3,
        "gba_banked_verdict": gba["verdict"],
        "per_layer": per_layer, "code": G.provenance(), "utc": C.utc_now(),
    }
    (out / "gutenberg_gba2_ceiling.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=float))
    print(f"GUTENBERG_GBA2_OK ceiling={verdict['verdict']} "
          f"median_J_halves={verdict['median_jaccard_halves']:.4f} "
          f"gba_over_ceiling={verdict['ratio_gba_over_ceiling']:.3f} "
          f"lam33/lam32_median={float(np.median(ratios33)):.4f}", flush=True)


if __name__ == "__main__":
    main()
