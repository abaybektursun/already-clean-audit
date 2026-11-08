"""GUTENBERG GB-A [THE MDE] — object identity: 24-FT invariant core vs g2b census core.

Spec §3 GB-A + AM-0.5.7. Computes per-layer subspace Jaccard between
  U_unt(24-FT, k=32)  = copernicus_a1_banks a1_banks.npz key "core/32/L" (hash-verified
                        against its manifest — the M-D99 consumer rule), and
  census core32(g2b)  = shackleton_census_g2b banks q_proj__L{L}__core32.npy (QR'd, the
                        v1_ablation convention),
against the matched-rotation null (Haar 32-frames in R^2048; the null of the DECISION statistic
= median over layers). Frozen verdict in gutenberg_core.gba_verdict. CPU only, no weights.

    pg run infra.aws_pg.jobs.gutenberg_gba --no-weights --type r7i.2xlarge,m7i.2xlarge \
        --s3-prefix results/gutenberg_gba --budget-min 60
"""
import argparse
import hashlib
import json
import pathlib
import subprocess

import numpy as np

from infra.pg import copernicus_core as C
from infra.pg import gutenberg_core as G

A1_NPZ = "s3://pg-lineage-769063704696/results/copernicus_a1_banks/out/a1_banks.npz"
A1_MANIFEST = "s3://pg-lineage-769063704696/results/copernicus_a1_banks/out/a1_banks_manifest.json"
CENSUS = "s3://pg-lineage-769063704696/results/shackleton_census_g2b/out/banks/"
N_LAYERS, K, N_IN = 18, 32, 2048


def _fetch(s3_uri, dst):
    subprocess.run(["aws", "s3", "cp", s3_uri, str(dst), "--region", "us-east-1",
                    "--only-show-errors"], check=True)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-null-draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260819)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wk = pathlib.Path("/tmp/gutenberg_gba")
    wk.mkdir(parents=True, exist_ok=True)

    man = json.loads(_fetch(A1_MANIFEST, wk / "a1_manifest.json").read_text())
    assert man["n_ft"] == 24 and man["n_in"] == N_IN, (man["n_ft"], man["n_in"])
    want_hash = {b["layer"]: b["sha256"] for b in man["banks"]
                 if b["bank"] == "core" and b["k"] == K}
    assert len(want_hash) == N_LAYERS, len(want_hash)

    npz = np.load(_fetch(A1_NPZ, wk / "a1_banks.npz"))
    subprocess.run(["aws", "s3", "cp", "--recursive", CENSUS, str(wk / "census"),
                    "--region", "us-east-1", "--only-show-errors"], check=True)

    J = {}
    for L in range(N_LAYERS):
        U = np.asarray(npz[f"core/{K}/{L}"], dtype=np.float64)
        assert C.bank_hash(U) == want_hash[L], f"a1 core/32/{L} hash mismatch"
        C.orthonormal_or_die(U, f"a1core32 L{L}")
        raw = np.load(wk / "census" / f"q_proj__L{L}__core32.npy")
        assert raw.shape == (N_IN, K), raw.shape
        V, _ = np.linalg.qr(np.asarray(raw, dtype=np.float64))
        C.orthonormal_or_die(V, f"census core32 L{L}")
        J[L] = G.subspace_jaccard(U, V)
        print(f"[gba] L{L}: J={J[L]:.4f}", flush=True)

    med_draws, per_draws = G.rotation_null_median_jaccard(
        N_IN, K, N_LAYERS, a.n_null_draws, a.seed)
    prov = {
        "a1_banks": A1_NPZ, "a1_manifest_sha256": man["manifest_sha256"],
        "a1_bank_hashes_verified": True, "census_banks": CENSUS,
        "census_sha256": {L: hashlib.sha256(
            (wk / "census" / f"q_proj__L{L}__core32.npy").read_bytes()).hexdigest()
            for L in range(N_LAYERS)},
        "n_null_draws": a.n_null_draws, "seed": a.seed,
        "code": G.provenance(), "utc": C.utc_now(),
    }
    metrics = G.build_gba_metrics(J, med_draws, per_draws, prov)
    (out / "gutenberg_gba_overlap.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=float))
    print(f"GUTENBERG_GBA_OK verdict={metrics['verdict']} "
          f"medianJ={metrics['median_jaccard']:.4f} "
          f"null_p95={metrics['rotation_null_median_p95']:.5f}", flush=True)


if __name__ == "__main__":
    main()
