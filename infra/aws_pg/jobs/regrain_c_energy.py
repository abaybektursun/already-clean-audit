"""REGRAIN Arm C — the energy diagnostic (GPU, DISCOVERY half only, EXPLORATORY).

WHY THIS EXISTS. The sealed pass returned `H_no-atom-handle`, and the verdict fired on ONE clause:
`em_reproduces = effect_em >= effect_bstar - δ_C`. The em-span control is the energy-matched RANDOM
SUBSPACE. Matching B*'s removed energy (25.0% of the residual trace, at rank 25) requires rank 512,
which drives sealed VQA to exactly 0.0000 and Δppl to +18,579 — a dead model. The clause therefore
cannot distinguish "a random subspace of the same energy reproduces B*" from "a random subspace of
the same energy destroys the network". Both satisfy the inequality.

That is an instrument defect. **The verdict is NOT re-adjudicated** (no post-unblind rule swap; the
sealed half is touched once and is not touched again). This job runs the control the situation
actually calls for, on the DISCOVERY half, labelled EXPLORATORY, to find the MECHANISM:

  energy_core        the top-|B*| atoms by REMOVED ENERGY alone (no alignment, no causal screen)
  energy_matched_k   random |B*|-atom unions whose per-atom removed-energy MARGINAL is matched to
                     B*'s, stratum by stratum — rank-matched AND energy-matched, which neither the
                     em-span (rank 512) nor the uniform/usage unions (low energy) are.
                     (Rejection-sampling the union's total energy does not converge: B*'s atoms sit
                     at a median energy percentile of 0.97, and 0 of 4000 random unions landed
                     within ±5% of E_b. Measured, not assumed.)

The reading it adjudicates: is B* a task-specific control surface, or simply the high-energy core
of the embedding residual — a set of directions selected, implicitly, for the energy they carry?

  ./infra/aws_pg/pg run infra.aws_pg.jobs.regrain_c_energy --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/regrain_c_energy --budget-min 150 \
      --args "--cdisc-s3 s3://…/regrain_c_disc/out"
"""
import argparse
import json
import os
import pathlib

N_STRATA = 20       # deciles are too coarse for a 25-atom match at the top of the energy tail


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cdisc-s3", required=True)
    ap.add_argument("--n-disc-items", type=int, default=None)
    ap.add_argument("--n-energy-rows", type=int, default=20000)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from infra.aws_pg.jobs import _regrain_shared as RS
    from infra.pg import grain_bsf as GB
    from infra.pg import inherit_pg1 as IP
    from infra.pg import regrain_core as RC
    from infra.pg.ctrliface_core import removed_energy
    from infra.pg.inherit_core import slice_k

    RS.install_pg()
    cd = json.loads(RS.fetch(f"{a.cdisc_s3}/c_disc.json", "/tmp/regrain/c_disc.json").read_text())
    bstar = [int(x) for x in cd["bstar"]]
    med_s = int(cd["median_seed"])
    L = int(cd["L"])
    k = len(bstar)
    n_disc = a.n_disc_items or RS.N_DISC
    rep = {"object": "regrain_c_energy_v1", "label": "EXPLORATORY / DISCOVERY-ONLY",
           "L": L, "bstar": bstar, "k": k, "median_seed": med_s,
           "why": ("the sealed em-span control saturated (rank 512, RF=0.0); this job runs the "
                   "rank-AND-energy-matched null on discovery. No verdict is re-adjudicated.")}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    banks = RS.load_read_banks("/tmp/regrain")
    V = banks["saction"]["V"]
    meta = json.loads(RS.fetch(f"{RS.GRAIN_HARVEST}/harvest_meta.json",
                               "/tmp/regrain/harvest_meta.json").read_text())
    X, labs = RS.load_shards(RS.GRAIN_HARVEST, meta, f"L{L}", workdir="/tmp/regrain/shards")
    stats = torch.load(RS.fetch(f"{RS.GRAIN_HARVEST}/site_stats.pt",
                                "/tmp/regrain/site_stats.pt"), map_location="cpu",
                       weights_only=False)
    hbar = stats[L]["hbar"]
    nm = RS.dict_bank_name("primary", 1, med_s)
    src = f"{RS.GRAIN_DICTS_L0}/{nm}" if med_s == 0 else f"{a.cdisc_s3}/{nm}"
    ae, _c, _q, _m = GB.load_dict_bank(RS.fetch(src, f"/tmp/regrain/{nm}"))
    Wd = RC.decoder_matrix(ae)
    U = RC.unit_decoder(Wd)                                        # (d, W) fp64

    # ---- per-atom removed energy on a fixed activation sample ----
    Xe = X[: a.n_energy_rows].double()
    Zc = Xe - hbar.double()                                        # (n, d)
    proj = Zc @ U                                                  # (n, W)
    atom_energy = (proj ** 2).sum(0)                               # (W,)
    trace = float((Zc ** 2).sum())
    del proj

    usage = GB.block_usage(ae, X[:40000], labs[:40000].long(), "cuda")
    active = torch.nonzero(usage["freq"] > 0).flatten()
    Qb = GB.union_Q(Wd, 1, bstar)
    E_b = removed_energy(Xe, Qb, hbar)
    rep["energy"] = {
        "trace": trace, "E_bstar_union": E_b, "E_bstar_frac_of_trace": E_b / trace,
        "sum_atom_energy_bstar": float(atom_energy[bstar].sum()),
        "mean_atom_energy_bstar": float(atom_energy[bstar].mean()),
        "mean_atom_energy_active": float(atom_energy[active].mean()),
        "bstar_energy_rank_percentiles": [
            float((atom_energy[active] < atom_energy[b]).double().mean()) for b in bstar]}
    _w()

    # ---- the ENERGY CORE: top-k atoms by removed energy, ignoring alignment entirely ----
    order_e = active[torch.argsort(atom_energy[active], descending=True)].tolist()
    energy_core = order_e[:k]
    rep["energy_core"] = {"atoms": energy_core,
                          "overlap_with_bstar": sorted(set(energy_core) & set(bstar)),
                          "n_overlap": len(set(energy_core) & set(bstar)),
                          "E_union": removed_energy(Xe, GB.union_Q(Wd, 1, energy_core), hbar)}
    _w()

    # ---- an ENERGY LADDER of rank-matched nulls (the control the em-span could not be) ----
    #
    # Two facts, both measured, shape this:
    #   (i)  rejection-sampling a 25-atom union on its TOTAL removed energy does not converge —
    #        B*'s atoms sit at a median energy percentile of 0.97 and 0/4000 random unions landed
    #        within ±5% of E_b;
    #   (ii) matching each atom's MARGINAL energy (the A2 construction, applied to the right
    #        nuisance) still lands at E/E_b ≈ 0.55, because B*'s atoms are mutually CORRELATED in a
    #        way that inflates their span's captured energy: E_b / Σ_j e_j = 1.139, whereas a set of
    #        near-orthogonal atoms of the same marginal energies captures ≈ Σ_j e_j.
    #
    # So the ladder carries three rank-25 nulls, each reported with its realized E/E_b:
    #   marginal_matched   per-atom energy marginal matched, correlation structure NOT matched
    #   union_matched      the union's removed energy matched to ±5% (random subset of the top-M
    #                      energy atoms, M searched per seed) — the control the em-span should be
    #   energy_core        the top-25 atoms by energy outright (no alignment, no causal screen)
    active_mask = torch.zeros(atom_energy.numel(), dtype=torch.bool)
    active_mask[active] = True
    strata_e = RC.usage_strata(atom_energy, active_mask, n_strata=N_STRATA)
    ladder = {}

    def _E(atoms):
        return removed_energy(Xe, GB.union_Q(Wd, 1, list(atoms)), hbar)

    marg = []
    for i in range(3):
        pick = RC.usage_matched_union(bstar, strata_e, seed=4242 + i)
        E = _E(pick)
        marg.append({"atoms": pick, "E_union": E, "E_ratio": E / E_b,
                     "union_over_sum": E / float(atom_energy[pick].sum())})
    ladder["marginal_matched"] = marg

    def draw_from_topM(M, seed):
        """25 atoms drawn from the M highest-energy active atoms (B* excluded)."""
        pool = [x for x in order_e[:M] if x not in set(bstar)]
        if len(pool) < k:
            return None
        g2 = torch.Generator().manual_seed(seed)
        return [pool[i] for i in torch.randperm(len(pool), generator=g2)[:k].tolist()]

    uni = []
    for i in range(3):
        seed = 909 + i
        grid = [25, 30, 40, 60, 100, 200, 400, 800, 1600, 3200, len(order_e)]
        grid = [m for m in grid if m >= k + len(bstar)]
        pts = [(m, _E(draw_from_topM(m, seed))) for m in grid]
        pts = [(m, e) for m, e in pts if e is not None]
        lo_m = max([m for m, e in pts if e >= E_b] or [grid[0]])
        hi_m = min([m for m, e in pts if e < E_b] or [grid[-1]])
        best = min(pts, key=lambda p: abs(p[1] - E_b))
        for _ in range(10):                                   # bisect on M
            mid = (lo_m + hi_m) // 2
            if mid in (lo_m, hi_m):
                break
            pick = draw_from_topM(mid, seed)
            E = _E(pick)
            if abs(E - E_b) < abs(best[1] - E_b):
                best = (mid, E)
            if E >= E_b:
                lo_m = mid
            else:
                hi_m = mid
        M, E = best
        pick = draw_from_topM(M, seed)
        uni.append({"atoms": pick, "M": M, "E_union": E, "E_ratio": E / E_b,
                    "union_over_sum": E / float(atom_energy[pick].sum())})
    ladder["union_matched"] = uni
    rep["energy_ladder"] = {
        "E_bstar_union": E_b,
        "bstar_union_over_sum": E_b / float(atom_energy[bstar].sum()),
        "marginal_matched_E_ratios": [m["E_ratio"] for m in marg],
        "union_matched_E_ratios": [u["E_ratio"] for u in uni],
        "union_matched_M": [u["M"] for u in uni],
        "energy_core_E_ratio": rep["energy_core"]["E_union"] / E_b,
        "draws": ladder}
    _w()
    ur = [u["E_ratio"] for u in uni]
    assert all(0.85 <= r <= 1.15 for r in ur), \
        f"union-matched search failed to bracket E_b (ratios {ur}) — report and halt"

    # ---- behaviour on the DISCOVERY half ----
    sha, _ = RS.pinned_pg1_sha("/tmp/regrain")
    model, proc = IP.load_pg1(sha=sha)
    layers = IP.decoder_layers(model)
    vqa, _coco = IP.pg1_tasks()
    it = json.loads(RS.fetch(f"{RS.GRAIN_GATE0}/items_vqa.json",
                             "/tmp/regrain/items_vqa.json").read_text())
    rows, hashes = IP.load_items(vqa, it["stream_window"])
    assert hashes == it["hashes"], "item-stream drift — halt"
    split = RS.grain_split(hashes, name="grain_vqa")
    disc_rows = [rows[i] for i in split["disc_idx"]][:n_disc]
    base_ppl = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)["ppl"]
    intact = IP.eval_pg1(model, proc, vqa, disc_rows, "cuda")
    ii = np.asarray(intact["per_item"], float)
    fl = IP.eval_pg1(model, proc, vqa, disc_rows, "cuda", ctx=IP.replace_at(layers, L, hbar))
    floor = float(np.asarray(fl["per_item"], float).mean())

    def cell(tag, Q):
        r = IP.eval_pg1(model, proc, vqa, disc_rows, "cuda", ctx=IP.ablate_at(layers, L, Q, hbar))
        with IP.ablate_at(layers, L, Q, hbar):
            p = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)["ppl"]
        e = RS.effect_from(np.asarray(r["per_item"], float), ii, floor)
        rep.setdefault("cells", {})[tag] = {"vqa": r["score"], "effect": e, "ppl": p,
                                            "dppl": p - base_ppl}
        _w()
        print(f"[energy] {tag}: vqa={r['score']:.4f} effect={e:.4f} dppl={p-base_ppl:+.2f}",
              flush=True)
        return e

    rep["reference"] = {"intact": intact["score"], "floor": floor, "base_ppl": base_ppl}
    e_full = cell("full_read", slice_k(V, RS.K_READ))
    e_b = cell("bstar", Qb)
    e_core = cell("energy_core", GB.union_Q(Wd, 1, energy_core))
    e_marg = [cell(f"marginal_matched_{i}", GB.union_Q(Wd, 1, m["atoms"]))
              for i, m in enumerate(marg)]
    e_uni = [cell(f"union_matched_{i}", GB.union_Q(Wd, 1, u["atoms"]))
             for i, u in enumerate(uni)]

    dC = float(cd["delta_C"]["value"])
    m_mean, u_mean = float(np.mean(e_marg)), float(np.mean(e_uni))
    rep["headline"] = {
        "effect_full_read": e_full, "effect_bstar": e_b,
        "effect_energy_core": e_core,
        "effect_marginal_matched_mean": m_mean, "effect_marginal_matched": e_marg,
        "effect_union_matched_mean": u_mean, "effect_union_matched": e_uni,
        "E_ratio_union_matched": [u["E_ratio"] for u in uni],
        "E_ratio_marginal_matched": [m["E_ratio"] for m in marg],
        "E_ratio_energy_core": rep["energy_ladder"]["energy_core_E_ratio"],
        "bstar_over_union_matched": (e_b / u_mean) if u_mean > 1e-6 else None,
        "union_matched_reproduces_bstar": bool(u_mean >= e_b - dC),
        "energy_core_reproduces_bstar": bool(e_core >= e_b - dC),
        "delta_C_from_c_disc": dC,
        "reading": ("If the UNION-energy-matched random unions reproduce B*'s effect, then B* is "
                    "the high-energy core of the embedding residual and its causal effect is an "
                    "energy effect. If they do not, B* carries direction-specific causal load that "
                    "neither the saturated em-span (rank 512) nor the low-energy rank-matched "
                    "unions could reveal — and the compact-handle question is OPEN, not closed. "
                    "This job adjudicates nothing: it is DISCOVERY-only and EXPLORATORY.")}
    print("C_ENERGY_OK", json.dumps(rep["headline"], default=str))
    # green written LAST: a crash after this point must not leave a green
    # artifact -- the drivers gate on `green` (regrain_drive*.sh wait_for_green).
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
