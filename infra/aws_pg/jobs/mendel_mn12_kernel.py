"""MENDEL MN-12 — the width-8 kernel's stability across consensus re-draws (queue item 3).

THE ARM HAS TWO KILL CLAUSES AND THIS JOB ANSWERS THE SECOND ONE COMPLETELY.

    "KILL: k-profile fails a third split, OR kernel coordinates unstable across consensus
     re-draws."

  (a) the k-profile re-score needs the A1r v0.9 kernel instrument and up to two inference jobs.
      NOT attempted here; named as a residual.
  (b) **kernel coordinates unstable across consensus re-draws** — fully computable from banked
      artifacts, and answered here.

WHAT THE KERNEL IS, and a simplification worth stating. The width-8 kernel is the bottom-8
eigenspace of the consensus Gram. The banked `core32` is the bottom-32 eigenvectors in ASCENDING
eigenvalue order, so **core8 is exactly `core32[:, :8]`** — no separate k=8 bank is needed and the
296 MB `copernicus_a1_banks` k-grid does not have to be fetched to read the kernel.

THE RE-DRAW. A "consensus re-draw" means: build the consensus from a different sample of the 24
fine-tunes and ask whether the kernel's coordinates move. This job bootstraps the 24 fine-tunes and
rebuilds a COORDINATE-LEVEL consensus from the per-fine-tune column write energies that
`mendel_ft_family` banks — the coordinates a resampled cloud says it wrote into least.

The disclosure that goes with it: the banked kernel is read from an EIGENBASIS of the consensus
Gram, while the re-draw is over COLUMN ENERGY, because per-fine-tune Grams are not banked (43 GB)
and column energies are. These are different objects, and the honest reading of the result is
therefore about coordinate-level consensus stability, which is the level the "kernel coordinates"
claim lives at, and not about eigenvector stability. Both the agreement with the banked kernel and
the re-draw stability are reported, so a reader can see how much the two constructions even agree
before asking whether one of them is stable.

FROZEN RULE. KERNEL-REPLICATES iff the bootstrap re-draws agree with each other at Jaccard >= 0.5
(median over draws, per cell) on >= half the cells; FLUKE iff median agreement is at or below the
haar floor; else AMBIGUOUS.

CPU only, banked artifacts.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import collections
import json
import pathlib
import sys

import numpy as np

from infra.pg import darwin_w0 as W0
from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as WT
from infra.pg import mendel_core as MC
from infra.pg import mendel_program as MP
from infra.pg import mendel_transport as MT

ARM = "MN-12"
FTFAM_DEFAULT = f"{MC.RESULTS_S3}/mendel_ftfam_v2/out"
TYPES = H.TYPES_RESID
K_KERNEL = 8
N_DRAW = 200
AGREE_BAR = 0.5


def consensus_kernel(colenergy_by_ft, units, k: int):
    """The k coordinates a cloud of fine-tunes wrote into LEAST, on average.

    Each fine-tune's column energy already sums to 1 (it is normalised by that fine-tune's own
    ‖ΔW‖²_F), so averaging them weights every fine-tune equally rather than letting the largest
    write dominate — the same normalisation the consensus Gram uses.
    """
    e = np.mean([colenergy_by_ft[u] for u in units], axis=0)
    return W0.top_k_coords(-e, k)


def redraw_stability(colenergy_by_ft, units, k: int, *, n_draw: int, seed: int) -> dict:
    """Bootstrap the fine-tune cloud; how much do the kernels of two independent re-draws agree?

    Pairs of INDEPENDENT draws rather than each draw against the full-cloud kernel: agreement with
    the full cloud is inflated because every draw is a subsample of it, so it measures overlap of
    a set with its own superset. Two independent resamples is the honest question — would a
    different cloud have named the same coordinates?
    """
    rng = np.random.default_rng(seed)
    n = len(units)
    js = []
    for _ in range(n_draw):
        a = [units[i] for i in rng.integers(0, n, n)]
        b = [units[i] for i in rng.integers(0, n, n)]
        js.append(W0.jaccard(consensus_kernel(colenergy_by_ft, a, k),
                             consensus_kernel(colenergy_by_ft, b, k)))
    return {"median": float(np.median(js)), "p2.5": float(np.percentile(js, 2.5)),
            "p97.5": float(np.percentile(js, 97.5)), "n_draw": int(n_draw),
            "values": [float(x) for x in js[:20]]}


def main():
    ap = argparse.ArgumentParser(description="MENDEL MN-12 — kernel stability across re-draws")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--ftfam", default=FTFAM_DEFAULT,
                    help="the FT-family run to read; either the bounded one or a reaped longer one")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = MP.arm(ARM)
    at = MC.Atlas(out, arm=ARM, run_id="mendel_mn12", wave=meta["wave"], queue_item=meta["item"])
    gates = H.Gates(out)
    man = MC.Manifest(H.cache_dir(out, "mendel_mn12"), out)

    # The roster comes from the VECTORS THAT EXIST, not from a metrics.json. A download-dominated
    # family run can be reaped before it aggregates, and its per-cell vectors are durable while its
    # metrics.json never appears — so keying off metrics would discard a run that holds more
    # fine-tunes than the one that finished. A fine-tune counts only if it has a vector for EVERY
    # cell, so a run interrupted mid-fine-tune contributes its complete units and not its partial one.
    vec = man.require("ftfam_vectors", f"{a.ftfam}/vectors", recursive=True)
    banks = WT.fetch_banks(H.cache_dir(out, "banks"))
    cells_needed = [(t, L) for t in TYPES for L in range(H.N_LAYERS)]
    seen = collections.Counter()
    for p in vec.glob("colenergy__*.npy"):
        seen[p.name.split("__")[1]] += 1
    units = sorted(u for u, n in seen.items() if n == len(cells_needed))
    incomplete = sorted(u for u, n in seen.items() if n != len(cells_needed))
    assert len(units) >= 8, (
        f"only {len(units)} complete fine-tunes in {a.ftfam} (incomplete: {incomplete}) — "
        f"a bootstrap needs a cloud")
    print(f"[mn12] {len(units)} complete fine-tunes"
          + (f", {len(incomplete)} partial and excluded: {incomplete}" if incomplete else ""),
          flush=True)

    cells, agreements = [], []
    for t in TYPES:
        for L in range(H.N_LAYERS):
            col = {}
            for u in units:
                p = vec / f"colenergy__{u}__{t}__L{L}.npy"
                assert p.exists(), f"missing per-fine-tune column energy: {p.name}"
                col[u] = np.load(p).astype(np.float64)
            n = int(next(iter(col.values())).size)

            # core8 == core32[:, :8]: both are the bottom eigenvectors in ascending order.
            U = WT.load_bank(banks, t, L, "core32")
            banked_kernel = MT.bank_address(U[:, :K_KERNEL], K_KERNEL)
            full_kernel = consensus_kernel(col, units, K_KERNEL)
            agree_with_banked = W0.jaccard(banked_kernel, full_kernel)

            st = redraw_stability(col, units, K_KERNEL, n_draw=N_DRAW,
                                  seed=H.cell_seed(a.seed, ARM, t, L))
            haar = float(np.median([
                W0.jaccard(MT.haar_address(n, K_KERNEL, H.cell_seed(a.seed, ARM, t, L, i)),
                           MT.haar_address(n, K_KERNEL, H.cell_seed(a.seed, ARM, t, L, i + 5000)))
                for i in range(64)]))
            cells.append({"type": t, "layer": L, "n": n,
                          "redraw_median_jaccard": st["median"],
                          "redraw_p2.5": st["p2.5"], "redraw_p97.5": st["p97.5"],
                          "haar_floor": haar,
                          "agreement_with_banked_eigen_kernel": agree_with_banked,
                          "stable": bool(st["median"] >= AGREE_BAR)})
            agreements.append(st["median"])
            at.add("kernel_redraw_agreement", value=st["median"], layer=L, type=t,
                   bank_id="core8", control_id=MC.TREATMENT, n=N_DRAW,
                   ci_lo=st["p2.5"], ci_hi=st["p97.5"],
                   flags={"k": K_KERNEL, "n_fine_tunes": len(units),
                          "agreement_with_banked_eigen_kernel": agree_with_banked})
            at.add("kernel_redraw_agreement", value=haar, layer=L, type=t, bank_id="core8",
                   control_id=MC.HAAR,
                   flags={"what": "two independent random 8-subsets of the same space"})
        print(f"[mn12] {t}: median re-draw agreement "
              f"{np.median([c['redraw_median_jaccard'] for c in cells if c['type'] == t]):.3f}",
              flush=True)

    med = float(np.median(agreements))
    haar_med = float(np.median([c["haar_floor"] for c in cells]))
    n_stable = sum(1 for c in cells if c["stable"])
    banked_agree = float(np.median([c["agreement_with_banked_eigen_kernel"] for c in cells]))
    if n_stable >= len(cells) / 2 and med >= AGREE_BAR:
        verdict = "KERNEL-REPLICATES"
    elif med <= haar_med:
        verdict = "FLUKE"
    else:
        verdict = "AMBIGUOUS"
    verdict = MP.check_verdict(ARM, verdict)
    gates.record("mn12_kernel_stable_across_redraws", n_stable >= len(cells) / 2,
                 median_agreement=med, haar_floor=haar_med, n_stable=n_stable,
                 n_cells=len(cells), bar=AGREE_BAR)
    gates.record("mn12_colenergy_consensus_agrees_with_the_eigen_kernel", banked_agree >= 0.5,
                 median_agreement=banked_agree,
                 note=("RECORDED. The re-draw is over COLUMN ENERGY because per-fine-tune Grams "
                       "are not banked; the banked kernel is an EIGENBASIS. How much the two "
                       "constructions agree is reported before asking whether one is stable"))
    info = at.to_parquet()
    man.write()
    MC.write_metrics(out, {
        "arm": ARM, "verdict": verdict, "queue_item": meta["item"],
        "k_kernel": K_KERNEL, "n_fine_tunes": len(units), "n_cells": len(cells),
        "ftfam_source": a.ftfam, "fine_tunes": units,
        "incomplete_fine_tunes_excluded": incomplete,
        "n_draws": N_DRAW, "bar": AGREE_BAR,
        "median_redraw_agreement": med, "median_haar_floor": haar_med,
        "n_cells_stable": n_stable,
        "median_agreement_with_banked_eigen_kernel": banked_agree,
        "by_type": {t: float(np.median([c["redraw_median_jaccard"] for c in cells
                                        if c["type"] == t])) for t in TYPES},
        "not_attempted": {"k_profile_third_split":
                          "needs the A1r v0.9 kernel instrument and up to 2 inference jobs"},
        "construction_note": ("core8 == core32[:, :8] (both are bottom eigenvectors in ascending "
                              "order), so no k=8 bank and no 296 MB k-grid fetch is required"),
        "cells": cells, "atlas": info})
    print("MN12_OK " + json.dumps({"verdict": verdict, "median": round(med, 4),
                                   "haar": round(haar_med, 4), "n_stable": n_stable,
                                   "n_cells": len(cells)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
