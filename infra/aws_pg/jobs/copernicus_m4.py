"""CO-M4 (item 54) — is the core's motion under SGD DIFFUSIVE or DIRECTED?

Spec §3 Tier M / §6.3. Runs entirely from TENANT's banked per-step lag banks: no GPU, no training.
The freeze sweep found `results/tenant_x4/out/sgd_s0/` and `sgd_s1/` each carrying `lag_bank.npz`
AND its matched `lag_bank_rand.npz`, which is exactly what the gate needs, so the ≤1 fallback run
the spec allowed is not required (journal 2026-08-17).

THE STATISTIC. Each banked row is the per-step in-core component v_t = (ΔW_t · Q). Over T steps,

    D = ‖Σ_t v_t‖ / Σ_t ‖v_t‖        (net displacement over path length)

D → 1 is a straight walk (directed), D → 0 is a random walk that returns (diffusive). This is
item 54's distinction: net ΔW cancels the back-and-forth that the path keeps, so a bank can look
barely-moved in net terms while having travelled a long way.

THE BAND IS THE SGD ARM'S OWN. §6.3 is explicit that the reference is that arm's
`lag_bank_rand.npz`, never the adamw arm's — in-run bands are never imported across runs. The
adamw arm is read too, but only as a CONTEXT row.

THE HIGHWAY LEG IS NOT FILLABLE. §6.3 makes it a context row conditional on the fallback run
happening; no banked highway projection exists in the lag banks, and the fallback is not needed,
so the row ships empty and says so. That is the pre-registered outcome, not an omission.

    pg run infra.aws_pg.jobs.copernicus_m4 \
        --args "--tenant-s3 s3://.../results/tenant_x4/out" --no-weights \
        --type r7i.2xlarge --s3-prefix results/copernicus_m4 --budget-min 60
"""
import argparse
import json
import pathlib
import subprocess

import numpy as np

from infra.pg import copernicus_core as C

ARM = "CO-M4"
QUEUE_ITEM = "54"
WAVE = "WF"
SGD_ARMS = ("sgd_s0", "sgd_s1")
CONTEXT_ARMS = ("adamw_s0",)
N_BOOT = 2000
BLOCK = 25                    # step-window length for the block bootstrap


def diffusivity(V: np.ndarray) -> float:
    """D = ‖Σ_t v_t‖ / Σ_t ‖v_t‖ for a (T, d) stack of per-step components."""
    V = np.asarray(V, dtype=np.float64)
    assert V.ndim == 2 and V.shape[0] >= 2, V.shape
    path = float(np.linalg.norm(V, axis=1).sum())
    assert path > 0, "zero path length — nothing moved in this bank"
    return float(np.linalg.norm(V.sum(axis=0)) / path)


def block_stats(V: np.ndarray, block: int):
    """Per-block-start sufficient statistics: S[s] = Σ rows in the block, P[s] = Σ row norms.

    D over ANY concatenation of blocks is exactly ‖Σ S‖ / Σ P, so the bootstrap never has to
    materialise a resampled array. That is not a micro-optimisation at this width: each banked
    cell is 500 steps × 65536 components, so the naive `V[idx]` copies 262 MB PER DRAW, and the
    first version of this job was killed after 900 CPU-seconds without finishing a single cell.
    Cumulative sums make the precompute one pass.
    """
    V = np.asarray(V, dtype=np.float64)
    T, d = V.shape
    starts = np.arange(0, max(1, T - block + 1))
    csum = np.concatenate([np.zeros((1, d)), np.cumsum(V, axis=0)], axis=0)
    cn = np.concatenate([[0.0], np.cumsum(np.linalg.norm(V, axis=1))])
    ends = np.minimum(starts + block, T)
    S = csum[ends] - csum[starts]
    P = cn[ends] - cn[starts]
    return S, P, starts


def _D_from_blocks(S, P, picks) -> float:
    tot = float(P[picks].sum())
    assert tot > 0, "zero path length in the resample"
    return float(np.linalg.norm(S[picks].sum(axis=0)) / tot)


def diffusivity_ci(V: np.ndarray, n_boot: int = N_BOOT, block: int = BLOCK,
                   seed: int = 1337) -> dict:
    """Block bootstrap over STEP-WINDOWS (§7), reported as CONTEXT.

    Steps are serially correlated — that correlation is the very thing D measures — so an iid
    bootstrap over single steps would return a CI for a quantity nobody asked about.

    KNOWN UPWARD BIAS, measured in the dry-run and stated here rather than hidden: resampling
    blocks WITH replacement lets one block appear twice, and its contribution to Σ_t v_t then adds
    coherently, inflating ‖Σ_t v_t‖ for a diffusive series. The bootstrap distribution therefore
    sits ABOVE the point estimate. That is why the SCORED gate is `paired_diffusivity_contrast`
    below and not this marginal interval — the band the spec compares against is built from point
    estimates, so scoring a biased interval against an unbiased band would tilt the gate.
    """
    V = np.asarray(V, dtype=np.float64)
    T = V.shape[0]
    S, P, starts = block_stats(V, block)
    nb = max(1, T // block)
    rng = np.random.default_rng(seed)
    boots = np.array([_D_from_blocks(S, P, rng.choice(starts.size, nb)) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"D": diffusivity(V), "lo": float(lo), "hi": float(hi), "n_steps": int(T),
            "block": block, "n_blocks": nb, "n_boot": n_boot,
            "bootstrap_median": float(np.median(boots)),
            "construction": (f"block bootstrap, {block}-step windows, {n_boot} draws; CONTEXT only "
                             f"— carries a known upward bias for diffusive series")}


def paired_diffusivity_contrast(V_core: np.ndarray, V_rand: np.ndarray, n_boot: int = N_BOOT,
                                block: int = BLOCK, seed: int = 1337) -> dict:
    """THE SCORED STATISTIC: D(core) − D(random reference), on a SHARED block resample.

    Both arms are the same cell's per-step components projected onto different subspaces, so they
    share their step index exactly. Resampling them together makes the with-replacement inflation
    common to both and it cancels in the difference — the same pairing logic every CO-A1 contrast
    uses. Negative means the core returns on itself MORE than a matched-random subspace does,
    which is §6.3's "sits below the SGD arm's own matched-random reference".
    """
    A = np.asarray(V_core, dtype=np.float64)
    B = np.asarray(V_rand, dtype=np.float64)
    assert A.shape[0] == B.shape[0], (A.shape, B.shape)
    T = A.shape[0]
    SA, PA, starts = block_stats(A, block)
    SB, PB, _ = block_stats(B, block)
    nb = max(1, T // block)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        picks = rng.choice(starts.size, nb)        # SHARED across both arms — that is the pairing
        boots[b] = _D_from_blocks(SA, PA, picks) - _D_from_blocks(SB, PB, picks)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"contrast": diffusivity(A) - diffusivity(B), "lo": float(lo), "hi": float(hi),
            "D_core": diffusivity(A), "D_random": diffusivity(B), "n_steps": int(T),
            "block": block, "n_boot": n_boot,
            "construction": (f"paired block bootstrap of D(core)-D(random) on a SHARED {block}-step "
                             f"resample, {n_boot} draws")}


def fetch(tenant_s3: str, dest: pathlib.Path, arms):
    assert tenant_s3.startswith("s3://"), f"--tenant-s3 must be an s3 uri, got {tenant_s3!r}"
    dest.mkdir(parents=True, exist_ok=True)
    got = {}
    for arm in arms:
        d = dest / arm
        d.mkdir(parents=True, exist_ok=True)
        for fn in ("lag_bank.npz", "lag_bank_rand.npz"):
            p = d / fn
            if not p.exists():
                subprocess.run(["aws", "s3", "cp", f"{tenant_s3}/{arm}/{fn}", str(p),
                                "--region", "us-east-1", "--quiet"], check=True)
            assert p.stat().st_size > 0, f"empty dependency {tenant_s3}/{arm}/{fn}"
        got[arm] = {fn: np.load(d / fn) for fn in ("lag_bank.npz", "lag_bank_rand.npz")}
    assert got, f"no lag banks found under {tenant_s3}"
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tenant-s3",
                    default="s3://pg-lineage-769063704696/results/tenant_x4/out")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    banks = fetch(a.tenant_s3, out / "_lag", SGD_ARMS + CONTEXT_ARMS)
    exp = C.Expectations(out / "copernicus_expectations.jsonl")
    exp.log(ARM, "diffusive_core",
            "core motion under SGD is substantially diffusive (net << path), i.e. D(core) below "
            "the SGD arm's own matched-random band", confidence="mid",
            basis="spec §11 row CO-M4", family="mechanism")
    at = C.Atlas(out, arm=ARM, run_id="copernicus_m4", wave=WAVE, queue_item=QUEUE_ITEM,
                 expected_ref=exp.require(ARM, "diffusive_core"))
    gates = C.Gates(out)

    rows, core_D, rand_D, contrasts = [], [], [], []
    for arm in SGD_ARMS + CONTEXT_ARMS:
        z, zr = banks[arm]["lag_bank.npz"], banks[arm]["lag_bank_rand.npz"]
        for key in sorted(z.files):
            layer = int(key.split("layers.")[1].split(".")[0])
            ci = diffusivity_ci(z[key], seed=a.seed + layer)
            cir = diffusivity_ci(zr[key], seed=a.seed + layer)
            pc = paired_diffusivity_contrast(z[key], zr[key], seed=a.seed + layer)
            is_sgd = arm in SGD_ARMS
            rows.append({"arm": arm, "layer": layer, "key": key, "is_sgd": is_sgd,
                         "core": ci, "random": cir, "paired_contrast": pc})
            if is_sgd:
                core_D.append(ci["D"])
                rand_D.append(cir["D"])
                contrasts.append(pc)
            at.add("m4_diffusivity", value=ci["D"], bank_id="core32", layer=layer, type="q_proj",
                   n=ci["n_steps"], ci_lo=ci["lo"], ci_hi=ci["hi"],
                   flags={"arm_cell": arm, "optimizer": "sgd" if is_sgd else "adamw",
                          "random_reference_D": cir["D"],
                          "paired_contrast": pc["contrast"], "paired_ci": [pc["lo"], pc["hi"]],
                          "band": pc["construction"],
                          "role": "scored" if is_sgd else "CONTEXT row (never a band source)"})
            print(f"[m4] {arm} L{layer}: D(core)={ci['D']:.4f} D(rand)={cir['D']:.4f} "
                  f"paired={pc['contrast']:+.4f} [{pc['lo']:+.4f},{pc['hi']:+.4f}]", flush=True)

    # CONTEXT: the marginal band over the SGD arm's own random draws, as §6.3 describes it.
    band = C.band(rand_D)
    # SCORED: every SGD cell's PAIRED contrast must sit wholly below zero. Paired because the
    # marginal bootstrap carries a known upward bias for diffusive series (see diffusivity_ci);
    # pairing makes that bias common to both arms so it cancels instead of tilting the gate.
    # §6.3's frozen partition, applied per cell: DIFFUSIVE-CORE requires every SGD cell's paired
    # interval wholly BELOW zero; the refuter fires as soon as any cell's core sits INSIDE or above
    # its own random reference — "inside" counts as refutation, because a core indistinguishable
    # from a matched-random subspace is not a diffusive CORE, it is a diffusive subspace.
    diffusive = bool(contrasts and all(c["hi"] < 0.0 for c in contrasts))
    refuted = bool(contrasts and any(c["hi"] >= 0.0 for c in contrasts))
    gates.record("m4_diffusive_core", diffusive,
                 core_D=core_D, random_D=rand_D,
                 paired_contrast=[c["contrast"] for c in contrasts],
                 paired_ci_hi=[c["hi"] for c in contrasts],
                 marginal_random_band=[band["lo"], band["hi"]], n_random_draws=band["n_draws"],
                 band=contrasts[0]["construction"] if contrasts else "n/a",
                 rule=("DIFFUSIVE-CORE fires iff EVERY SGD cell's paired D(core)-D(random) 95% "
                       "interval lies wholly below 0, against that arm's OWN random reference; "
                       "the refuter fires as soon as any cell's interval reaches or crosses 0 "
                       "(§6.3: 'refuted iff D(core) sits inside/above that band')"),
                 prediction="core net << path under SGD (§11, mid)")

    rep = {"object": "copernicus_m4_v1", "arm": ARM, "program": C.PROGRAM,
           "queue_item": QUEUE_ITEM, "tenant_s3": a.tenant_s3,
           "sgd_arms": list(SGD_ARMS), "context_arms": list(CONTEXT_ARMS),
           "random_band": band, "core_D_sgd": core_D,
           "verdict": (C.REFUTER_DID_NOT_FIRE if diffusive else
                       (C.REFUTER_FIRED if refuted else C.AMBIGUOUS)),
           "diffusive_core": diffusive,
           "highway_context_row": {
               "filled": False,
               "reason": ("§6.3 makes the highway comparison a CONTEXT row conditional on the "
                          "fallback SGD run; no banked highway projection exists in the lag banks "
                          "and the fallback was not needed, so the row ships empty by design")},
           "rows": rows, "provenance": C.provenance(), "utc": C.utc_now()}
    (out / "copernicus_m4.json").write_text(json.dumps(rep, indent=2, default=float))
    at.close()
    at.to_parquet()
    print("=" * 78, flush=True)
    print(f"COPERNICUS_M4_OK diffusive_core={diffusive} verdict={rep['verdict']} "
          f"core_D={[round(x, 4) for x in core_D]} rand_band=[{band['lo']:.4f},{band['hi']:.4f}]",
          flush=True)


if __name__ == "__main__":
    main()
