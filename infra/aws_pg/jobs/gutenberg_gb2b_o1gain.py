"""GUTENBERG GB-2b — reader gain and singular-subspace overlap ON THE PRIMARY OBJECT (O1).

Post-critic arm (2026-08-19). The blind reviewer found that the paper's §2 asserted, OF O1 (the
24-fine-tune invariant core), an overlap-with-the-weight-null-space number that the results table
assigns to O2 (the gemma-2b census core) — and paired an O2 reader gain with an O1 deletion cost
inside one parenthetical. That is a substrate crossing inside the paper's own declared firewall,
and it is the f_perp-class conflation this whole program exists to remove. It also means the
mechanism (§5) was never measured on the object where the phenomenon (§3) was measured.

Both are fixable by measurement rather than by re-wording, and cheaply: the banks are banked and
the weights are a CPU load. This job computes, per layer, on O1's own banks against PaliGemma-1's
own decoder weights:

  reader gain  ||W U||_F^2 / k at q/k/v_proj and gate_proj (the off-path reference), for
               core32, anti32, three Haar draws, bottomsigma32 and topsigma32
  overlaps     of core32 with q_proj's bottom-32, bottom-256 and top-32 right-singular
               subspaces (the A13 analogue on O1; chance = k_sing / n)

Same instrument as GB-2 (gutenberg_core.reader_gain / overlap_with_singular_subspace), so the
O1 and O2 tables are comparable as ratios. Weights come through invariant_core's LM-SCOPED
loader, never a suffix resolver: SigLIP shares the layer suffix in a PaliGemma checkpoint and a
naive key match reads vision weights (house trap, cost six launches once).

    pg run infra.aws_pg.jobs.gutenberg_gb2b_o1gain --no-weights --type r7i.2xlarge,m7i.2xlarge \
        --s3-prefix results/gutenberg_gb2b --budget-min 120
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
from infra.aws_pg.jobs.invariant_core import N_LAYERS, PG1_REPO, _load_decoder

A1_NPZ = "s3://pg-lineage-769063704696/results/copernicus_a1_banks/out/a1_banks.npz"
A1_MANIFEST = "s3://pg-lineage-769063704696/results/copernicus_a1_banks/out/a1_banks_manifest.json"
SITES = ("q_proj", "k_proj", "v_proj", "gate_proj")
N_IN, K = 2048, 32


def _fetch(s3_uri, dst):
    subprocess.run(["aws", "s3", "cp", s3_uri, str(dst), "--region", "us-east-1",
                    "--only-show-errors"], check=True)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--manifest-sha256", required=True)
    ap.add_argument("--seed", type=int, default=20260819)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wk = pathlib.Path("/tmp/gutenberg_gb2b")
    wk.mkdir(parents=True, exist_ok=True)

    hf_setup()
    import torch

    man = load_manifest(a.manifest, a.manifest_sha256)
    base_sha = man["base_sha"]
    a1man = json.loads(_fetch(A1_MANIFEST, wk / "a1_manifest.json").read_text())
    want = {(b["bank"], b["layer"]): b["sha256"] for b in a1man["banks"] if b["k"] == K}
    npz = np.load(_fetch(A1_NPZ, wk / "a1_banks.npz"))

    w = _load_decoder(PG1_REPO, base_sha)
    rng = np.random.default_rng(a.seed)
    rows = []
    for L in range(N_LAYERS):
        banks = {}
        for bid in ("core", "anti"):
            Q = np.asarray(npz[f"{bid}/{K}/{L}"], dtype=np.float64)
            assert C.bank_hash(Q) == want[(bid, L)], f"{bid}/32/{L} hash mismatch"
            C.orthonormal_or_die(Q, f"{bid}32 L{L}")
            banks[f"{bid}32"] = Q
        for s in range(3):
            banks[f"haar32_s{s}"] = G.haar_frame(N_IN, K, rng)

        Wq = w[f"layers.{L}.self_attn.q_proj.weight"].to(torch.float64).cpu().numpy()
        assert Wq.shape[1] == N_IN, (L, Wq.shape)
        _, _, Vh = np.linalg.svd(Wq, full_matrices=False)
        Vs = Vh.T                                    # columns: sigma DESCENDING
        r = Vs.shape[1]
        banks["bottomsigma32"] = np.ascontiguousarray(Vs[:, r - K:])
        banks["topsigma32"] = np.ascontiguousarray(Vs[:, :K])

        row = {"layer": L}
        for site in SITES:
            key = (f"layers.{L}.self_attn.{site}.weight" if site != "gate_proj"
                   else f"layers.{L}.mlp.gate_proj.weight")
            W = w[key].to(torch.float64).cpu().numpy()
            assert W.shape[1] == N_IN, (site, L, W.shape)
            cell = {b: G.reader_gain(W, U) for b, U in banks.items()}
            cell["haar32"] = float(np.median([cell[f"haar32_s{s}"] for s in range(3)]))
            cell["allmean"] = G.allmean_gain(W)
            row[site] = cell
        core = banks["core32"]
        row["overlaps_vs_q_proj_singular"] = {
            "core_bottom32": G.overlap_with_singular_subspace(core, Vs[:, ::-1], 32),
            "core_bottom256": G.overlap_with_singular_subspace(core, Vs[:, ::-1], 256),
            "core_top32": G.overlap_with_singular_subspace(core, Vs, 32),
            "chance_bottom32": K / N_IN, "chance_bottom256": 256 / N_IN,
        }
        rows.append(row)
        print(f"[gb2b] L{L}: q core={row['q_proj']['core32']:.4g} "
              f"haar={row['q_proj']['haar32']:.4g} "
              f"ovl_bot32={row['overlaps_vs_q_proj_singular']['core_bottom32']:.4g}",
              flush=True)

    med = {site: {stat: float(np.median([rw[site][stat] for rw in rows]))
                  for stat in ("core32", "anti32", "haar32", "allmean",
                               *(("bottomsigma32", "topsigma32") if site == "q_proj" else ()))}
           for site in SITES}
    for site in SITES:
        med[site]["core_over_haar"] = med[site]["core32"] / med[site]["haar32"]
        med[site]["anti_over_haar"] = med[site]["anti32"] / med[site]["haar32"]
    ovl = {k: float(np.median([rw["overlaps_vs_q_proj_singular"][k] for rw in rows]))
           for k in ("core_bottom32", "core_bottom256", "core_top32")}

    payload = {
        "object": "gutenberg_gb2b_o1_reader_gain_v1",
        "which_object": "O1 — the 24-fine-tune invariant core (copernicus_a1_banks core/32/L)",
        "estimand": "median over 18 layers of ||W U||_F^2 / k for each 32-dim bank at "
                    "q/k/v_proj and gate_proj of google/paligemma-3b-pt-224's LM decoder "
                    "(fp32 checkpoint tensors, fp64 arithmetic), and the overlap of core32 with "
                    "q_proj's singular subspaces; same instrument as GB-2's O2/O3 tables",
        "why": "the paper's §2 quoted an O2 overlap and an O2 reader gain as if they were O1's — "
               "a crossing inside its own substrate firewall, found by the blind reviewer",
        "per_layer": rows, "median": med, "overlap_median": ovl,
        "base_repo": PG1_REPO, "base_sha": base_sha,
        "a1_manifest_sha256": a1man["manifest_sha256"], "seed": a.seed,
        "code": G.provenance(), "utc": C.utc_now(),
    }
    (out / "gutenberg_gb2b_o1gain.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=float))
    print(f"GUTENBERG_GB2B_OK q_core/haar={med['q_proj']['core_over_haar']:.4f} "
          f"gate={med['gate_proj']['core_over_haar']:.4f} "
          f"ovl_bot32={ovl['core_bottom32']:.5f} (chance {K / N_IN:.5f})", flush=True)


if __name__ == "__main__":
    main()
