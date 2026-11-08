"""CO-A1 stage-1 — build and HASH every grid bank, once, from the frozen 24-delta manifest.

Spec §2.1 / §4 / §9.3. This is the `--bank-dir` PRODUCER the spec requires: the anchor's decisive
job rebuilt the consensus Gram in-process from ~300 GB of live Hub downloads on every invocation,
which made re-scoring unaffordable and put a live roster query inside the estimand. Here the
expensive half runs ONCE and publishes hashed banks; `copernicus_a1.py` consumes them by prefix
and asserts the hashes (the M-D99 consumer rule).

WHAT IS BUILT (per layer, per k in the grid), all from the SAME generating objects the banked
k=32 battery came from, so the extension is a widening and not a new object:

    core-k   bottom-k eigenvectors of the q_proj consensus Gram G_q(L) = Σ_i ΔWᵢᵀΔWᵢ/‖ΔWᵢ‖²_F
    anti-k   top-k eigenvectors of the same Gram (the other extreme of one spectrum)
    topsigma-k / bottomsigma-k   largest/smallest-σ right-singular vectors of the BASE q_proj
    complement32                 a random k-subspace inside core's orthogonal complement

NOT built here: topenergy-k and the stage-2 banks (k_b, the set differences). Those are functions
of the READOUT corpus D-half activation energy, which does not exist until the scoring job has run
its ACT pass — building them here would either leak the confirmation half or invent an energy
accounting from a different corpus (§2.1's "never divide a wikitext numerator by an MMLU
denominator").

fp32 THROUGHOUT. Every derivation reads `_load_decoder`'s genuine fp32 checkpoint tensors. The
anchor's σ banks were cut from the *bf16* live model and cast up (`layers[L]...weight.float()`),
which is a bf16-precision basis wearing an fp32 dtype; §0.2 forbids the mixed path.

    pg run infra.aws_pg.jobs.copernicus_a1_banks \
        --args "--k-grid 2,4,8,16,32,64,128" --no-weights \
        --type r7i.8xlarge --s3-prefix results/copernicus_a1_banks --budget-min 240
"""
import argparse
import hashlib
import json
import pathlib

import numpy as np

from infra.pg import copernicus_core as C
from infra.pg import humboldt_core as H

# Reuse the load + difference path VERBATIM (imported, not re-implemented) — the same functions
# the anchor used to define the band.
from infra.aws_pg.jobs.invariant_core import (
    PG1_REPO, N_LAYERS, _load_decoder, _decoder_matrices, _ftype, _delete_cache,
)

MANIFEST = pathlib.Path(__file__).resolve().parents[2] / "pg" / "copernicus_delta_manifest.json"
ARM = "CO-A1"
N_IN = 2048                      # q_proj input dim = the residual stream


def load_manifest(path, expect_sha256: str) -> dict:
    """The frozen (repo, sha) cloud, asserted by content hash. A live Hub roster query at run time
    is forbidden (§4): the anchor resolved 24 repos at a moment in time, and a cloud that silently
    gained a member would redefine the population mid-program."""
    m = json.loads(pathlib.Path(path).read_text())
    body = json.dumps({"repos": m["repos"], "shas": m["shas"], "base_sha": m["base_sha"]},
                      sort_keys=True, separators=(",", ":")).encode()
    got = hashlib.sha256(body).hexdigest()
    assert got == m["sha256"], f"manifest self-hash mismatch: {got} != {m['sha256']}"
    assert got == expect_sha256, (
        f"manifest hash {got} != --manifest-sha256 {expect_sha256} — the delta cloud moved")
    assert len(m["repos"]) == m["n_ft"] == 24, f"expected 24 deltas, manifest has {len(m['repos'])}"
    assert set(m["repos"]) == set(m["shas"]), "manifest repo/sha key mismatch"
    return m


def consensus_gram(repos, shas, w_base, layers):
    """Per-layer q_proj consensus Gram, fp64, VERBATIM the anchor's accumulation (invariant_core_func
    `_consensus_gram_qproj`), re-implemented here only to take the FROZEN roster as an argument
    instead of discovering it from the Hub."""
    import torch
    G = {L: np.zeros((N_IN, N_IN), dtype=np.float64) for L in layers}
    n_terms = {L: 0 for L in layers}
    for ridx, repo_id in enumerate(repos):
        sha = shas[repo_id]
        print(f"[a1-banks] diff {repo_id}@{sha[:8]} ({ridx + 1}/{len(repos)})", flush=True)
        w_ft = _load_decoder(repo_id, sha)
        matched = _decoder_matrices(w_base, w_ft)
        for L in layers:
            key = f"layers.{L}.self_attn.q_proj.weight"
            wb, wf = matched[key]
            assert _ftype(key) == "q_proj"
            A = wf.to(torch.float64) - wb.to(torch.float64)
            fro2 = float((A ** 2).sum())
            assert fro2 > 0, f"zero ΔW {key} in {repo_id}"
            G[L] += (A.transpose(0, 1) @ A).cpu().numpy().astype(np.float64) / fro2
            n_terms[L] += 1
        del w_ft, matched
        _delete_cache(repo_id)
    for L, nt in n_terms.items():
        assert nt == len(repos), f"L{L} got {nt} terms, expected {len(repos)}"
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k-grid", default="2,4,8,16,32,64,128")
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--manifest-sha256", required=True)
    ap.add_argument("--layers", default="", help="comma idxs; empty => all 18")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    k_grid = sorted(int(x) for x in a.k_grid.split(","))
    layers = ([int(x) for x in a.layers.split(",") if x.strip()] if a.layers.strip()
              else list(range(N_LAYERS)))
    k_max = max(k_grid)

    H.hf_setup()                     # HF_HUB_DISABLE_XET=1 + gated-repo auth, FIRST
    import torch

    man = load_manifest(a.manifest, a.manifest_sha256)
    repos, shas, base_sha = man["repos"], man["shas"], man["base_sha"]
    print(f"[a1-banks] base={PG1_REPO}@{base_sha[:8]} cloud={len(repos)} layers={len(layers)} "
          f"k_grid={k_grid}", flush=True)

    w_base = _load_decoder(PG1_REPO, base_sha)
    for L in layers:
        W0 = w_base[f"layers.{L}.self_attn.q_proj.weight"]
        assert W0.dtype == torch.float32, f"L{L} base q_proj is {W0.dtype}, not fp32 (§0.2)"
        assert W0.shape[1] == N_IN, f"L{L} q_proj input dim {W0.shape[1]} != {N_IN}"

    G = consensus_gram(repos, shas, w_base, layers)

    banks, meta = {}, []
    for L in layers:
        # ONE eigendecomposition per layer, ascending — every core-k and anti-k is a slice of it,
        # which is what makes the grid NESTED by construction (core-2 ⊂ core-4 ⊂ … ⊂ core-128).
        evals, V = np.linalg.eigh(G[L])
        W0 = w_base[f"layers.{L}.self_attn.q_proj.weight"].to(torch.float64).cpu().numpy()
        _, s, Vh = np.linalg.svd(W0, full_matrices=False)
        Vsig = np.ascontiguousarray(Vh.T)                      # n×r, σ DESCENDING
        r = Vsig.shape[1]
        assert k_max <= r, f"k_max={k_max} exceeds economy rank r={r}"
        for k in k_grid:
            cell = {
                "core": np.ascontiguousarray(V[:, :k]),                 # bottom-k eigvecs
                "anti": np.ascontiguousarray(V[:, -k:]),                # top-k eigvecs
                "bottomsigma": np.ascontiguousarray(Vsig[:, r - k:]),
                "topsigma": np.ascontiguousarray(Vsig[:, :k]),
            }
            if k == H.BANK_K:                                           # complement is a k=32 row
                cell["complement"] = H.complement_subspace(cell["core"],
                                                           H.cell_seed(a.seed, ARM, L, k))
            for bid, Q in cell.items():
                C.orthonormal_or_die(Q, f"{bid}-{k} L{L}")
                banks[f"{bid}/{k}/{L}"] = Q
                meta.append({"bank": bid, "k": k, "layer": L, "n": int(Q.shape[0]),
                             "dim": int(Q.shape[1]), "sha256": C.bank_hash(Q)})
        print(f"[a1-banks] L{L}: eig[min,max]=[{evals.min():.3e},{evals.max():.3e}] "
              f"sigma[min,max]=[{s.min():.3e},{s.max():.3e}] banks={len(k_grid) * 4}", flush=True)

    np.savez_compressed(out / "a1_banks.npz", **banks)
    payload = {
        "object": "copernicus_a1_banks_v1", "arm": ARM, "program": C.PROGRAM,
        "base_repo": PG1_REPO, "base_sha": base_sha, "manifest_sha256": man["sha256"],
        "n_ft": len(repos), "repos": repos, "layers": layers, "k_grid": k_grid, "seed": a.seed,
        "n_in": N_IN, "provenance": C.provenance(), "utc": C.utc_now(),
        "generating_objects": {
            "core/anti": "eigenvectors of the q_proj consensus Gram, ascending; core=bottom-k, anti=top-k",
            "topsigma/bottomsigma": "right-singular vectors of the fp32 BASE q_proj, desc/asc sigma",
            "complement": "random k-subspace inside core's orthogonal complement (k=32 only)"},
        "not_built_here": ("topenergy-k and the stage-2 banks (k_b, core-minus-topenergy, "
                           "topenergy-minus-core) are functions of the readout D-half activation "
                           "energy and are built in the scoring job"),
        "banks": meta,
    }
    (out / "a1_banks_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"COPERNICUS_A1_BANKS_OK n_banks={len(banks)} layers={len(layers)} k_grid={k_grid} "
          f"manifest_sha={man['sha256'][:12]}", flush=True)


if __name__ == "__main__":
    main()
