"""BISPEC Arm K (GPU) — the coherence trio and the cancellation law. The MDE.

One question, asked three ways through three different corruption geometries:

    **does the SEQUENCE axis carry behavioural susceptibility?**

Every cell is a matched pair. The two members inject the SAME per-token perturbation energy — hence
the same per-token fidelity, hence (in expectation) the same realized R² — and differ ONLY in how
that energy is arranged across token positions. REGRAIN R10 proved susceptibility is not a function
of R²; if these pairs separate, the arrangement across positions is one of the coordinates it IS a
function of.

  P1   shared-draw Gaussian  vs  iid Gaussian            (coherence, in the crudest possible form)
  P1g  the cancellation-law sweep over group size g      (measurement, not gated)
  P2   quantization error    vs  the SAME error, permuted across positions   (identical marginals)
  P3   one shared random subspace removed at every position
                             vs  an INDEPENDENT random subspace per position (matched rank + energy)

**Why P1 is identified regardless of the direct-path floor (A4).** The pre-hook site puts δ_t on the
skip connection into the readout at its own position, so damage carries a g-independent direct term.
That term is identical in distribution for shared and iid draws at matched per-token energy. The
whole shared-minus-iid gap is therefore ROUTED — attention-mediated — and P1 needs no knowledge of
the floor. The floor only matters for the *shape* of damage(g), which is why P1g fits
`a + b·g^(−p)` rather than a bare power law.

**The saturation guard.** REGRAIN R1's `em_reproduces` scored "a random subspace reproduced B*" and
"a random subspace obliterated the network" as the same event, because both arms sat at the dead
floor. P3 at high rank is in exactly that danger zone (rank 512 returns RF = 0.0000 at L0). Every
pair here passes through `saturation_guard` and a pinned pair is reported **VOID**, never as a null.

Discovery pass. The promoted trio takes ONE sealed pass (`bispec_k_sealed`) on the `bispec|` sealed
half (A6: REGRAIN's `grain|` sealed half was touched by its R1, and REGRAIN's own §(iv) freezes
re-salting on exposure).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.bispec_k --type g6e.8xlarge,p5.4xlarge --no-weights \
      --s3-prefix results/bispec_k --budget-min 600 \
      --args "--sigma-s3 s3://…/bispec_sigma/out --harvest-s3 s3://…/regrain_pipeA/out/harvest \
              --frontier-s3 s3://…/regrain_pipeA/out/i_frontier/frontier.json"
"""
import argparse
import json
import os
import pathlib

N_DRAWS = 3
G_SWEEP = (1, 4, 16, 64, 256, None)      # None = fully coherent (one draw, all positions)
QUANT_LEVELS = (0.4, 0.8)                # × activation rms
SUBSPACE_RANKS = (64, 128)               # kept below the rank-512 dead-model regime (REGRAIN R1)
N_POOL = 512                             # per-token subspace pool; >= max window length used here
N_ITEMS = 200


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma-s3", required=True)
    ap.add_argument("--harvest-s3", required=True)
    ap.add_argument("--frontier-s3", required=True)
    ap.add_argument("--site", default="L0")
    ap.add_argument("--n-items", type=int, default=N_ITEMS)
    ap.add_argument("--sealed", action="store_true", help="the ONE sealed pass over the promoted trio")
    ap.add_argument("--promoted", default="", help="sealed only: comma list of cells + the level")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch

    from infra.aws_pg.jobs import _regrain_shared as RS
    from infra.aws_pg.jobs.bispec_gate0 import bispec_split
    from infra.pg import bispec_core as BC
    from infra.pg import inherit_pg1 as IP
    from infra.pg import regrain_core as RC
    from infra.pg.ctrliface_core import apply_mean_ablation, random_basis

    RS.install_pg()
    L = int(a.site[1:])
    half = "sealed" if a.sealed else "discovery"
    rep = {"object": "bispec_k_v1", "site": a.site, "half": half, "n_draws": N_DRAWS,
           "n_ppl_seq": RS.N_PPL_SEQ, "n_items": a.n_items,
           "question": "does the SEQUENCE axis carry behavioural susceptibility?",
           "identification": "matched per-token energy ⇒ the direct (skip-connection) term is matched "
                             "across arms; the entire coherent-minus-independent gap is routed",
           "cells": []}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    # ---- Σ bank, frontier anchors --------------------------------------------
    bank = torch.load(RS.fetch(f"{a.sigma_s3}/sigma_bank.pt", "/tmp/bispec/sigma_bank.pt"),
                      map_location="cpu", weights_only=False)[a.site]
    hbar = bank["hbar"]
    front = json.loads(RS.fetch(a.frontier_s3, "/tmp/bispec/frontier.json").read_text())
    gauss_pts = [p for p in front["points"] if p["family"] == "gauss"]
    assert len(gauss_pts) >= 2, "frontier has no gauss family — cannot read the matched levels"
    # The two frozen levels, chosen by TARGET R² on the REGRAIN frontier — not by rank order.
    # `gauss_pts[0]` (the highest-R² point, σ=0.01·rms, R²=0.9999) is not a benign perturbation, it
    # is no perturbation: a first launch measured ΔVQA +0.0017 / Δppl +0.02 there, and the pair would
    # have been VOIDed by the saturation guard as "both arms indistinguishable from intact".
    # BENIGN := the frontier's R² ≈ 0.94 point (REGRAIN R10's σ=0.2·rms, where VQA has NOT yet moved
    #           but the perturbation is real);  MID := its R² ≈ 0.78 point (σ=0.4·rms).
    benign = min(gauss_pts, key=lambda p: abs(p["r2"] - 0.94))
    mid = min(gauss_pts, key=lambda p: abs(p["r2"] - 0.78))
    assert abs(benign["r2"] - mid["r2"]) > 0.05, "the two frozen levels collapsed onto one point"
    levels = {"benign": benign["level"], "mid": mid["level"]}
    rep["frozen_levels"] = {"source": a.frontier_s3, "benign": benign, "mid": mid,
                            "targets": {"benign_r2": 0.94, "mid_r2": 0.78},
                            "note": "σ in units of the activation rms; the levels are selected by "
                                    "TARGET R² from the REGRAIN frontier artifact, never from its "
                                    "prose and never by rank order"}
    _w()

    # ---- model, items, floors -------------------------------------------------
    sha, _ = RS.pinned_pg1_sha("/tmp/bispec")
    model, proc = IP.load_pg1(sha=sha)
    layers = IP.decoder_layers(model)
    vqa, _c = IP.pg1_tasks()
    it = json.loads(RS.fetch(f"{RS.GRAIN_GATE0}/items_vqa.json", "/tmp/bispec/items_vqa.json").read_text())
    rows, hashes = IP.load_items(vqa, it["stream_window"])
    assert hashes == it["hashes"], "item-stream drift — halt"
    sp = bispec_split(hashes, name="bispec_vqa")
    idx = sp["conf_idx"] if a.sealed else sp["disc_idx"]
    items = [rows[i] for i in idx][: a.n_items]
    intact = IP.eval_pg1(model, proc, vqa, items, "cuda")
    base_ppl = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)["ppl"]
    floor = IP.eval_pg1(model, proc, vqa, items, "cuda", scramble_seed=7)["score"]
    rep["reference"] = {"intact_vqa": intact["score"], "base_ppl": base_ppl,
                        "dead_floor_vqa": floor, "n_items": len(items), "salt": "bispec|"}
    _w()

    meta = json.loads(RS.fetch(f"{a.harvest_s3}/harvest_meta.json",
                               "/tmp/bispec/harvest_meta.json").read_text())
    shard0 = meta["shards"][a.site][0]
    X = torch.load(RS.fetch(f"{a.harvest_s3}/{shard0}", f"/tmp/bispec/shards/{shard0}"),
                   map_location="cpu", weights_only=False)["X"][:8192].float().cuda()
    xbar, rms, d = X.mean(0), float(X.pow(2).mean().sqrt()), X.shape[1]
    rep["site_rms"] = rms

    def measure(name, family, arm, hook_fn, draw, r2_rows=None, **extra):
        with torch.inference_mode():
            r2 = RC.r2_of(X, r2_rows(), xbar) if r2_rows is not None else None
        r = IP.eval_pg1(model, proc, vqa, items, "cuda", ctx=BC.hook_at(layers, L, hook_fn()))
        with BC.hook_at(layers, L, hook_fn()):
            p = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)["ppl"]
        eff = RS.effect_from(r["per_item"], intact["per_item"], floor)
        se = RS.effect_se(r["per_item"], intact["per_item"], floor)
        rec = {"cell": name, "family": family, "arm": arm, "draw": draw,
               "realized_r2": r2, "vqa": r["score"], "dvqa": r["score"] - intact["score"],
               "effect": eff, "effect_se": se, "ppl": p, "dppl": p - base_ppl, **extra}
        rep["cells"].append(rec)
        _w()
        print(f"[k] {name} draw{draw}: R2={'' if r2 is None else round(r2,4)} "
              f"dVQA={rec['dvqa']:+.4f} dppl={rec['dppl']:+.3f}", flush=True)
        return rec

    # ---- anchors (F-K3 continuity) -------------------------------------------
    measure("identity", "anchor", "identity", lambda: BC.make_hook_gaussian_group(0.0, 1, None, d), 0,
            lambda: X.clone())

    # =====================================================================================
    # SEALED PASS (the ONE exposure). §4.2 promotes "the trio at the level where discovery
    # shows the larger separation (<= 6 cells)". From the discovery half:
    #   P1 -> mid level   (the larger separation; benign clears on ppl, mid does not, but its
    #                      separation is the larger and the rule names separation, not clearance)
    #   P3 -> rank 128    (gap +7.60 ppl vs +1.61 at rank 64; both cleared)
    #   P2 -> delta 0.8   (H_alignment-matters; inert at 0.4)
    # Six cell-types, three draws each. BH q=0.10 over the promoted family. Then STOP.
    # =====================================================================================
    if a.sealed:
        rep["promoted_family"] = {
            "P1": {"level": "mid", "arms": ["coherent(g=all)", "independent(g=1)"]},
            "P3": {"rank": 128, "arms": ["shared", "per-token"]},
            "P2": {"level": 0.8, "arms": ["aligned", "permuted"]},
            "why": "the trio at the level where the discovery half showed the larger separation; "
                   "6 cell-types, 3 draws each; BH q=0.10; the sealed half is touched ONCE"}
        _w()
        sig_mid = levels["mid"] * rms
        for s_ in range(N_DRAWS):
            measure("P1_mid_gall", "P1", "coherent",
                    lambda ss=s_: BC.make_hook_gaussian_group(sig_mid, 11 + 1000 * ss, None, d), s_,
                    None, level="mid", g=-1, sigma_over_rms=levels["mid"])
            measure("P1_mid_g1", "P1", "independent",
                    lambda ss=s_: BC.make_hook_gaussian_group(sig_mid, 11 + 1000 * ss, 1, d), s_,
                    None, level="mid", g=1, sigma_over_rms=levels["mid"])
        r_ = 128
        for s_ in range(N_DRAWS):
            Qsh = random_basis(d, r_, 777 + s_, "cpu", torch.float64)
            measure(f"P3_r{r_}_shared", "P3", "coherent",
                    lambda Q=Qsh: BC.make_hook_subspace_removal(Q, hbar, False, 0, d, r_), s_,
                    None, rank=r_)
            pool = BC.pertoken_subspace_pool(d, r_, N_POOL, 9 + s_, device="cuda")
            measure(f"P3_r{r_}_pertoken", "P3", "independent",
                    lambda pl=pool: BC.make_hook_subspace_removal(None, hbar, True, 0, d, r_, pool=pl),
                    s_, None, rank=r_, n_pool=N_POOL)
            del pool
            torch.cuda.empty_cache()
        step = 0.8 * rms
        measure("P2_q0.8_true", "P2", "coherent",
                lambda st=step: BC.make_hook_quant_permuted(st, 5, permute=False), 0, None, level=0.8)
        for s_ in range(N_DRAWS):
            measure("P2_q0.8_permuted", "P2", "independent",
                    lambda st=step, ss=s_: BC.make_hook_quant_permuted(st, 5 + ss, permute=True), s_,
                    None, level=0.8)

        # ---- sealed adjudication, per readout, BH over the promoted family ----
        def pair(fam, **sel):
            co = [c for c in rep["cells"] if c["family"] == fam and c["arm"] == "coherent"
                  and all(c.get(k2) == v2 for k2, v2 in sel.items())]
            ino = [c for c in rep["cells"] if c["family"] == fam and c["arm"] == "independent"
                   and all(c.get(k2) == v2 for k2, v2 in sel.items())]
            return co, ino

        from infra.pg.diff_metrics import bootstrap_ci  # noqa: F401  (house surface, kept in ledger)
        sealed = {}
        for name, fam, sel in (("P1@mid", "P1", {"level": "mid"}),
                               ("P3@r128", "P3", {"rank": 128}),
                               ("P2@0.8", "P2", {"level": 0.8})):
            co, ino = pair(fam, **sel)
            assert co and ino, f"sealed pair missing an arm: {name}"
            for ro in ("effect", "dppl"):
                cv = np.array([c[ro] for c in co], float)
                iv = np.array([c[ro] for c in ino], float)
                gap = float(cv.mean() - iv.mean())
                se = float(np.hypot(cv.std() / len(cv) ** 0.5, iv.std() / len(iv) ** 0.5))
                dK = float(iv.std())
                sat = BC.saturation_guard(
                    damage_treat=intact["score"] - (cv.mean() if ro == "effect" else 0) * (intact["score"] - floor),
                    damage_null=intact["score"] - (iv.mean() if ro == "effect" else 0) * (intact["score"] - floor),
                    floor=floor, intact=intact["score"], name=f"sealed/{name}/{ro}") if ro == "effect" else {"void": False}
                sealed[f"{name}/{ro}"] = {
                    "coherent": float(cv.mean()), "independent": float(iv.mean()),
                    "gap": gap, "gap_cilo": gap - 1.96 * se, "gap_cihi": gap + 1.96 * se,
                    "delta_K": dK, "void": sat["void"],
                    "clears": bool((not sat["void"]) and abs(gap) - 1.96 * se > dK),
                    "sign": "coherent_more_damaging" if gap > 0 else "independent_more_damaging"}
        rep["sealed_adjudication"] = sealed
        rep["sealed_note"] = ("two-sided by construction: P2's discovery landing was "
                              "aligned_LESS_damaging, a branch the spec's one-sided trio rule could "
                              "not express (A9). The sealed half is touched ONCE and is not touched "
                              "again.")
        _w()
        print("BISPEC_K_SEALED_OK", json.dumps(
            {k2: {"gap": round(v2["gap"], 4), "clears": v2["clears"], "sign": v2["sign"]}
             for k2, v2 in sealed.items()}, default=str))
        rep["green"] = True
        _w()
        return

    Qrot = RC.haar_rotation(d, RS.ROT_SEED, device="cuda", dtype=torch.float32)

    # ---- P1 / P1g : Gaussian coherence ---------------------------------------
    for lname, lev in levels.items():
        sigma = lev * rms
        for g in ((1, None) if lname == "benign" else G_SWEEP):
            arm = "coherent" if g is None else ("independent" if g == 1 else f"group{g}")
            for s in range(N_DRAWS):
                def rr(sg=sigma, gg=g, ss=s):
                    # realized R² of the SAME construction, on the harvest rows
                    gen = torch.Generator(device="cpu").manual_seed(11 + 1000 * ss)
                    if gg is None:
                        dl = (sg * torch.randn(d, generator=gen, dtype=torch.float64)).float().cuda()
                        return X + dl.unsqueeze(0)
                    return X + sg * torch.randn(X.shape, generator=torch.Generator(device="cuda")
                                                .manual_seed(11 + 1000 * ss), device="cuda")
                measure(f"P1_{lname}_g{g}", "P1g" if lname == "mid" else "P1", arm,
                        lambda sg=sigma, gg=g, ss=s: BC.make_hook_gaussian_group(sg, 11 + 1000 * ss, gg, d),
                        s, rr, level=lname, sigma_over_rms=lev, g=(-1 if g is None else g))

    # ---- P2 : quantization, decohered ----------------------------------------
    for q in QUANT_LEVELS:
        step = q * rms
        measure(f"P2_q{q}_true", "P2", "coherent",
                lambda st=step: BC.make_hook_quant_permuted(st, 5, permute=False), 0,
                lambda st=step: RC.quantize_step(X, st), level=q)
        for s in range(N_DRAWS):
            measure(f"P2_q{q}_permuted", "P2", "independent",
                    lambda st=step, ss=s: BC.make_hook_quant_permuted(st, 5 + ss, permute=True), s,
                    None, level=q,
                    note="identical error marginals; only the error's alignment to its own token moves")

    # ---- P3 : subspace removal, decohered ------------------------------------
    # The per-token arm draws its frames from a precomputed pool (one batched QR on device). A fresh
    # QR per token per forward costs ~60k QRs of (2048×k) on the ppl pass alone. The pool is indexed
    # by absolute position, so frames are independent across any window shorter than N_POOL and
    # repeat with period N_POOL beyond it — reported per cell.
    hb64 = hbar.double().cuda()
    for r_ in SUBSPACE_RANKS:
        for s in range(N_DRAWS):
            Qsh = random_basis(d, r_, 777 + s, "cpu", torch.float64)
            measure(f"P3_r{r_}_shared", "P3", "coherent",
                    lambda Q=Qsh: BC.make_hook_subspace_removal(Q, hbar, False, 0, d, r_), s,
                    lambda Q=Qsh.cuda(): apply_mean_ablation(X.double(), Q, hb64).float(),
                    rank=r_)
            pool = BC.pertoken_subspace_pool(d, r_, N_POOL, 9 + s, device="cuda")
            measure(f"P3_r{r_}_pertoken", "P3", "independent",
                    lambda p=pool: BC.make_hook_subspace_removal(None, hbar, True, 0, d, r_, pool=p), s,
                    None, rank=r_, n_pool=N_POOL,
                    note="an independent random rank-k subspace per token, drawn from a pool of "
                         f"{N_POOL} frames indexed by absolute position; matched rank, matched "
                         "per-token removed energy in expectation (realized value measured)")
            del pool
            torch.cuda.empty_cache()

    # ---- rotated-error anchor -------------------------------------------------
    measure("rot_err_a1", "anchor", "rotated", lambda: RC.make_pre_hook_gaussian(0.0, 1), 0, None,
            note="placeholder anchor: the rotated-error family is reproduced in bispec_figures F-K3 "
                 "directly from the REGRAIN frontier artifact, not re-run here")

    # ---- adjudication ---------------------------------------------------------
    def band(family, arm, **sel):
        v = [c["effect"] for c in rep["cells"] if c["family"] == family and c["arm"] == arm
             and all(c.get(kk) == vv for kk, vv in sel.items())]
        return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (float("nan"), float("nan"), 0)

    def ppl_band(family, arm, **sel):
        v = [c["dppl"] for c in rep["cells"] if c["family"] == family and c["arm"] == arm
             and all(c.get(kk) == vv for kk, vv in sel.items())]
        return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (float("nan"), float("nan"), 0)

    cells_adj, adj = {}, {}
    for name, fam, sel_c, sel_i in (
            ("P1", "P1", {"arm": "coherent", "level": "benign"}, {"arm": "independent", "level": "benign"}),
            ("P1g", "P1g", {"arm": "coherent"}, {"arm": "independent"}),
            ("P2", "P2", {"arm": "coherent"}, {"arm": "independent"}),
            ("P3", "P3", {"arm": "coherent"}, {"arm": "independent"})):
        levels_cleared, per_level = 0, {}
        keys = sorted({c.get("level") for c in rep["cells"] if c["family"] == fam})
        for lev in keys:
            mc, sc, nc = band(fam, sel_c["arm"], level=lev)
            mi, si, ni = band(fam, sel_i["arm"], level=lev)
            if nc == 0 or ni == 0:
                continue
            delta_K = si if si == si else 0.0                      # half-width of the iid band
            sat = BC.saturation_guard(damage_treat=intact["score"] - mc * (intact["score"] - floor),
                                      damage_null=intact["score"] - mi * (intact["score"] - floor),
                                      floor=floor, intact=intact["score"], name=f"{name}@{lev}")
            gap = mc - mi
            gap_cilo = gap - 1.96 * float(np.hypot(sc / max(nc, 1) ** 0.5, si / max(ni, 1) ** 0.5))
            halfw = 1.96 * float(np.hypot(sc / max(nc, 1) ** 0.5, si / max(ni, 1) ** 0.5))
            per_level[str(lev)] = {"coherent_effect": mc, "independent_effect": mi, "gap": gap,
                                   "gap_cilo": gap_cilo, "gap_cihi": gap + halfw,
                                   "delta_K": delta_K, "void": sat["void"], "saturation": sat,
                                   "coherent_dppl": ppl_band(fam, sel_c["arm"], level=lev)[0],
                                   "independent_dppl": ppl_band(fam, sel_i["arm"], level=lev)[0]}
            if (not sat["void"]) and gap_cilo > delta_K:
                levels_cleared += 1
        any_void = all(v["void"] for v in per_level.values()) if per_level else True
        cells_adj[name] = {"gap_cilo": max((v["gap_cilo"] for v in per_level.values()), default=-1.0),
                           "gap_cihi": max((v["gap_cihi"] for v in per_level.values()), default=-1.0),
                           "gap": max((v["gap"] for v in per_level.values()), key=abs, default=0.0),
                           "levels_cleared": levels_cleared, "void": any_void}
        adj[name] = per_level
    rep["per_cell"] = adj
    delta_K_global = float(np.nanmean([v["delta_K"] for c in adj.values() for v in c.values()])) \
        if adj else 0.0
    rep["delta_K"] = delta_K_global

    # A9: P1/P3 adjudicate the SEQUENCE axis; P2 adjudicates the perturbation–token ALIGNMENT axis,
    # two-sided. Gate-0b measured permuted quantization error to be 4.6× MORE VQA-damaging than the
    # true error at identical marginals — a branch the spec's §1.2 table did not contain, and one the
    # frozen one-sided coherence rule would have recorded as "no coherence effect".
    rep["adjudication"] = BC.adjudicate_coherence(cells_adj, delta_K_global)
    p2 = cells_adj.get("P2")
    if p2:
        rep["alignment_axis"] = BC.adjudicate_alignment(
            gap=p2["gap"], gap_cilo=p2["gap_cilo"], gap_cihi=p2["gap_cihi"],
            delta_K=delta_K_global, void=p2["void"])
        rep["alignment_axis"]["note"] = (
            "P2 varies whether the perturbation is a FUNCTION OF THE TOKEN IT LANDS ON, holding the "
            "across-position arrangement fixed (every token still gets exactly one error row). That "
            "is a third coordinate, distinct from the feature band and the sequence mode. Arm S's ω "
            "axis does not span it.")
    _w()

    # ---- the cancellation law (A4) -------------------------------------------
    gs, dmg = [], []
    for g in G_SWEEP:
        gg = 1 if g is None else g
        v = [c["effect"] for c in rep["cells"] if c["family"] == "P1g" and c.get("g") == (-1 if g is None else g)]
        if v and g is not None:
            gs.append(float(g))
            dmg.append(v)
    if len(gs) >= 3:
        rep["cancellation_law"] = BC.fit_cancellation_law(gs, dmg)
        rep["cancellation_law"]["reading"] = (
            "p is the AMPLITUDE exponent; the ideal attention-pool value is 0.5. `a` is the "
            "direct-path floor that the skip connection guarantees and no decoherence can average "
            "away. p_bare_powerlaw is what the spec's own fit would have reported (A4).")
    _w()

    print("BISPEC_K_OK", json.dumps({"verdict": rep["adjudication"]["verdict"],
                                     "cleared": rep["adjudication"]["cleared"],
                                     "delta_K": delta_K_global,
                                     "law": rep.get("cancellation_law", {}).get("p")}, default=str))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
