"""GUTENBERG GB-2c — the mechanism test ON THE PRIMARY OBJECT: does gain explain O1's inversion?

Post-critic arm (2026-08-19), and the arm GB-2b's own result demanded. GB-2b measured reader gain
on O1 and found the band only mildly attenuated (q_proj core/Haar 0.716, not O2's 0.131). That
alone cannot explain O1's inversion, because the inversion is a comparison of the band against its
ENERGY-MATCHED TWIN, and the twin's gain was never measured on any object. If "high energy x low
reader gain" is the mechanism, then on O1 the twin must read HIGHER than the band by roughly the
cost ratio: cost(twin)/cost(band) = 2.902 [2.738, 3.084].

That is a real, falsifiable, cheap prediction, and it is registered here BEFORE the arm runs:

  MECHANISM-EXPLAINS   gain(twin)/gain(band) at q_proj lands inside the cost ratio's 95% CI
                       [2.738, 3.084]
  MECHANISM-DIRECTIONAL gain ratio > 1 (twin reads more) but outside that interval — the sign is
                       right, the magnitude is not fully accounted for
  MECHANISM-FAILS      gain ratio <= 1 — the twin reads no more than the band, so reader gain
                       does not account for O1's inversion and §5's claim is scoped to O2/O3

The banks are GB-1's own constructed arms, banked by that run (gb1_banks.npz): the SAME matrices
that produced the costs, so this is not a re-derivation. Reproduction gate first: recompute
core32/anti32/haar/sigma with GB-2b's seed and assert every median matches GB-2b to rel <= 1e-9,
since identical inputs and instrument must give identical output before any new number is read.

    pg run infra.aws_pg.jobs.gutenberg_gb2c_o1twin --no-weights --type r7i.2xlarge,m7i.2xlarge \
        --args " --manifest-sha256 <sha>" --s3-prefix results/gutenberg_gb2c --budget-min 120
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
GB1_BANKS = "s3://pg-lineage-769063704696/results/gutenberg_gb1_v2/out/gb1_banks.npz"
GB1_VERDICT = ("s3://pg-lineage-769063704696/results/gutenberg_gb1_v2/out/"
               "gutenberg_gb1_verdict.json")
GB2B = "s3://pg-lineage-769063704696/results/gutenberg_gb2b/out/gutenberg_gb2b_o1gain.json"
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
    ap.add_argument("--seed", type=int, default=20260819)   # GB-2b's seed: same Haar draws
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wk = pathlib.Path("/tmp/gutenberg_gb2c")
    wk.mkdir(parents=True, exist_ok=True)

    hf_setup()
    import torch

    man = load_manifest(a.manifest, a.manifest_sha256)
    base_sha = man["base_sha"]
    a1man = json.loads(_fetch(A1_MANIFEST, wk / "a1_manifest.json").read_text())
    want = {(b["bank"], b["layer"]): b["sha256"] for b in a1man["banks"] if b["k"] == K}
    a1 = np.load(_fetch(A1_NPZ, wk / "a1_banks.npz"))
    gb1b = np.load(_fetch(GB1_BANKS, wk / "gb1_banks.npz"))
    gb1v = json.loads(_fetch(GB1_VERDICT, wk / "gb1_verdict.json").read_text())
    prev = json.loads(_fetch(GB2B, wk / "gb2b.json").read_text())

    cost_ratio = gb1v["sites"]["q"]["E_over_C"]          # [ratio, lo95, hi95], paired bootstrap
    twin_arms = sorted({k.rsplit("/", 1)[0] for k in gb1b.files
                        if k.startswith(("energymatched32", "specmatched32"))})
    assert twin_arms, f"no twin banks in {GB1_BANKS}: {gb1b.files[:6]}"
    print(f"[gb2c] twin banks: {twin_arms}", flush=True)

    w = _load_decoder(PG1_REPO, base_sha)
    rng = np.random.default_rng(a.seed)
    rows = []
    for L in range(N_LAYERS):
        banks = {}
        for bid in ("core", "anti"):
            Q = np.asarray(a1[f"{bid}/{K}/{L}"], dtype=np.float64)
            assert C.bank_hash(Q) == want[(bid, L)], f"{bid}/32/{L} hash mismatch"
            banks[f"{bid}32"] = Q
        for s in range(3):
            banks[f"haar32_s{s}"] = G.haar_frame(N_IN, K, rng)   # same seed => same frames
        for arm in twin_arms:
            Q = np.asarray(gb1b[f"{arm}/{L}"], dtype=np.float64)
            C.orthonormal_or_die(Q, f"{arm} L{L}", tol=1e-5)     # banked as fp32
            banks[arm] = Q

        Wq = w[f"layers.{L}.self_attn.q_proj.weight"].to(torch.float64).cpu().numpy()
        _, _, Vh = np.linalg.svd(Wq, full_matrices=False)
        Vs = Vh.T
        r = Vs.shape[1]
        banks["bottomsigma32"] = np.ascontiguousarray(Vs[:, r - K:])
        banks["topsigma32"] = np.ascontiguousarray(Vs[:, :K])

        row = {"layer": L}
        for site in SITES:
            key = (f"layers.{L}.self_attn.{site}.weight" if site != "gate_proj"
                   else f"layers.{L}.mlp.gate_proj.weight")
            W = w[key].to(torch.float64).cpu().numpy()
            cell = {b: G.reader_gain(W, U) for b, U in banks.items()}
            cell["haar32"] = float(np.median([cell[f"haar32_s{s}"] for s in range(3)]))
            cell["allmean"] = G.allmean_gain(W)
            row[site] = cell
        rows.append(row)
        q = row["q_proj"]
        em = float(np.median([q[t] for t in twin_arms if t.startswith("energymatched32")]))
        print(f"[gb2c] L{L}: core={q['core32']:.4g} ematch={em:.4g} haar={q['haar32']:.4g}",
              flush=True)

    def med(site, stat):
        return float(np.median([rw[site][stat] for rw in rows]))

    # ---- reproduction gate: identical inputs + instrument must reproduce GB-2b exactly ----
    checked = 0
    for site in SITES:
        for stat in ("core32", "anti32", "haar32", "allmean"):
            got, exp = med(site, stat), prev["median"][site][stat]
            assert abs(got - exp) <= 1e-9 * max(abs(exp), 1e-30), \
                f"GB-2b reproduction FAILED {site}/{stat}: {got} vs {exp}"
            checked += 1
    print(f"[gb2c] GB-2b reproduction gate PASSED ({checked} medians)", flush=True)

    # ---- the frozen mechanism test, at the definitional site ----
    def pool(prefix, site="q_proj"):
        arms = [t for t in twin_arms if t.startswith(prefix)]
        return float(np.median([np.median([rw[site][t] for t in arms]) for rw in rows])), arms

    g_core = med("q_proj", "core32")
    g_em, em_arms = pool("energymatched32")
    g_sp, sp_arms = pool("specmatched32")
    ratio_em = g_em / g_core
    ratio_sp = g_sp / g_core
    lo, hi = cost_ratio[1], cost_ratio[2]
    if ratio_em <= 1.0:
        verdict = "MECHANISM-FAILS"
    elif lo <= ratio_em <= hi:
        verdict = "MECHANISM-EXPLAINS"
    else:
        verdict = "MECHANISM-DIRECTIONAL"

    payload = {
        "object": "gutenberg_gb2c_o1_twin_gain_v1",
        "which_object": "O1 — the 24-fine-tune invariant core, and GB-1's OWN banked twin arms",
        "estimand": "median over 18 layers of ||W U||_F^2 / k at q/k/v_proj and gate_proj of "
                    "PaliGemma-1's LM decoder, for the band and for the energy-matched and "
                    "spectrum-matched twins GB-1 actually ablated",
        "frozen_rule": "MECHANISM-EXPLAINS iff gain(twin)/gain(band) at q_proj lies inside the "
                       "cost ratio's 95% paired-bootstrap CI; MECHANISM-DIRECTIONAL iff >1 but "
                       "outside; MECHANISM-FAILS iff <=1. Registered before the arm ran.",
        "verdict": verdict,
        "gain_ratio_energymatched_over_core": ratio_em,
        "gain_ratio_specmatched_over_core": ratio_sp,
        "cost_ratio_energymatched_over_core": cost_ratio,
        "gain_medians_q_proj": {"core32": g_core, "energymatched32": g_em,
                                "specmatched32": g_sp, "haar32": med("q_proj", "haar32"),
                                "anti32": med("q_proj", "anti32")},
        "pooled_arms": {"energymatched32": em_arms, "specmatched32": sp_arms},
        "median": {site: {stat: med(site, stat) for stat in rows[0][site]} for site in SITES},
        "per_layer": rows,
        "reproduction_gate": {"source": GB2B, "n_medians": checked, "passed": True},
        "gb1_banks": GB1_BANKS, "base_repo": PG1_REPO, "base_sha": base_sha,
        "seed": a.seed, "code": G.provenance(), "utc": C.utc_now(),
    }
    (out / "gutenberg_gb2c_o1twin.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=float))
    print(f"GUTENBERG_GB2C_OK verdict={verdict} gain_ratio={ratio_em:.3f} "
          f"cost_ratio={cost_ratio[0]:.3f} [{lo:.3f}, {hi:.3f}] spec_gain_ratio={ratio_sp:.3f}",
          flush=True)


if __name__ == "__main__":
    main()
