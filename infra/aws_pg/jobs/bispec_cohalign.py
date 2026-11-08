"""BISPEC — Arm K re-adjudication + the mechanism test. Weight-free, zero new forwards.

Two jobs in one, both operating on Arm K's banked `metrics.json`.

--------------------------------------------------------------------------------------------
PART 1 — re-adjudicate Arm K under its OWN frozen rule (§4.2), which the job failed to implement
--------------------------------------------------------------------------------------------
§4.2: *H_coherence(cell) iff the coherent cell's damage exceeds its matched independent twin's band
by CI-lo > δ_K, **at both frozen levels***.

`bispec_k.py` filed the benign level under family `P1` and the mid level under family `P1g`. The
adjudicator then evaluated `P1` against a single level and required `levels_cleared >= 2`, so the
clause **could never be satisfied** — and it returned `H_spectrum-suffices` with `cleared = []`
against a measured 174× ppl gap (iid +5.16 vs coherent +900.1 at matched R² = 0.776).

That is a **rule that could not fire**: the fourth in this lineage (REGRAIN R1's `em_reproduces`,
R12's em-ppl band, R18's monotone statistic) and the second committed by this program's own author
(after A7's chordal diameter). It is an IMPLEMENTATION defect, not a rule change — the frozen rule's
data are banked, the coherent and independent arms exist at both levels, and nothing about the
decision rule is altered here. The verdict of record is recomputed from the artifact, and the
original job's `adjudication` block is preserved in the output for audit.

**This is not a post-unblind rule swap.** The rule is §4.2 verbatim. What changes is that both
frozen levels are now handed to it.

--------------------------------------------------------------------------------------------
PART 2 — is the coherence effect the mean-shift mechanism?
--------------------------------------------------------------------------------------------
The spec assigns the g-sweep to **attention pooling**: iid noise attenuates like 1/√n_eff through
the weighted average, coherent noise does not. Gate-0b measured `n_eff = 3.93`. That law therefore
predicts the sweep **SATURATES once g ≳ n_eff ≈ 4**: past that point every position an attention row
reads already shares a draw, and further grouping can add nothing.

Measured (L0, mid level, Δppl):  g=1 → +5.16 · g=4 → +6.54 · g=16 → +10.96 · g=64 → +19.86 ·
g=256 → +99.6 · g=all → +900.1.  It does not saturate; it climbs another ~140× past g = 4.

The competing mechanism is Arm A's, and it scales correctly. Averaging `T/g` independent draws
leaves a residual-stream MEAN SHIFT of norm ∝ σ·√(d·g/T) — growing like **√g**, without bound, long
after attention pooling has saturated. And a mean shift along a direction the data already occupies
(at L0 one eigendirection carries 70.22 % of the centered trace) makes ⟨x_t, δ⟩ systematically large,
which RMSNorm converts into uniform content attenuation.

Two independent tests, both free:

  T1 (scaling)   regress log(damage) on log(g). Pooling predicts a plateau past g ≈ n_eff; the
                 mean-shift predicts continued growth tracking the mean-shift norm ∝ √g.
  T2 (direction) every draw is generated from a recorded seed by `make_hook_gaussian_group`, so each
                 δ is exactly reproducible. Correlate the coherent cells' damage with
                 |cos(δ̄, e₁)| — pooling is direction-BLIND and predicts ρ ≈ 0; the mean-shift
                 mechanism predicts ρ > 0.

Pre-declared, symmetric kills. `H_coherence-is-pooling` is a reachable observation: a plateau past
g ≈ 4 with a direction-blind correlation would fire it.

If the mean-shift branch lands, **Arm K's sequence axis and Arm A's mechanism are not independent
evidence**, P1's shared arm requires a norm-coherence control, and the surface's ω axis at DC is
partly a restatement of its feature axis.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.bispec_cohalign --type r7i.2xlarge --no-weights \
      --s3-prefix results/bispec_cohalign --budget-min 30 \
      --args "--k-s3 s3://…/bispec_k/out/metrics.json --sigma-s3 s3://…/bispec_sigma/out"
"""
import argparse
import json
import pathlib

N_EFF_GATE0B = 3.93          # measured, Gate-0b (pooled median inverse participation ratio)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k-s3", required=True)
    ap.add_argument("--sigma-s3", required=True)
    ap.add_argument("--site", default="L0")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seq-len", type=int, default=512, help="ppl window; sets the mean-shift scale")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from scipy.stats import spearmanr

    from infra.aws_pg.jobs import _regrain_shared as RS
    from infra.pg import bispec_core as BC

    K = json.loads(RS.fetch(a.k_s3, "/tmp/bispec/k_metrics.json").read_text())
    bank = torch.load(RS.fetch(f"{a.sigma_s3}/sigma_bank.pt", "/tmp/bispec/sigma_bank.pt"),
                      map_location="cpu", weights_only=False)[a.site]
    e1 = bank["basis"][:, 0].double()
    hbar = bank["hbar"].double()
    hhat = hbar / hbar.norm()
    d = e1.shape[0]
    rms = K["site_rms"]
    intact, floor = K["reference"]["intact_vqa"], K["reference"]["dead_floor_vqa"]

    rep = {"object": "bispec_cohalign_v1", "site": a.site, "n_eff_measured_gate0b": N_EFF_GATE0B,
           "source_k_artifact": a.k_s3,
           "original_k_adjudication_PRESERVED": K.get("adjudication"),
           "defect": "bispec_k filed benign under family P1 and mid under family P1g, so §4.2's "
                     "'at both frozen levels' clause could never be satisfied for P1. The rule is "
                     "unchanged; both levels are now handed to it. Implementation defect, disclosed."}

    def _w():
        (out / "cohalign.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    C = K["cells"]

    def sel(**kw):
        return [c for c in C if all(c.get(k) == v for k, v in kw.items())]

    def eff(cells):
        return np.array([c["effect"] for c in cells], dtype=float)

    def dppl(cells):
        return np.array([c["dppl"] for c in cells], dtype=float)

    # ---------------- PART 1: re-adjudicate §4.2 with BOTH levels ----------------
    # coherent = g == -1 (all positions);  independent = g == 1.
    pairs = {
        "P1@benign": (sel(family="P1", arm="coherent", level="benign"),
                      sel(family="P1", arm="independent", level="benign")),
        "P1@mid": (sel(family="P1g", arm="coherent", level="mid"),
                   sel(family="P1g", arm="independent", level="mid")),
    }
    for r_ in sorted({c.get("rank") for c in C if c["family"] == "P3" and c.get("rank")}):
        pairs[f"P3@rank{r_}"] = (sel(family="P3", arm="coherent", rank=r_),
                                 sel(family="P3", arm="independent", rank=r_))

    per_level, delta_Ks = {}, []
    for name, (coh, ind) in pairs.items():
        if not coh or not ind:
            per_level[name] = {"status": "MISSING_ARM"}
            continue
        ec, ei = eff(coh), eff(ind)
        delta_K = float(ei.std())                 # §4.2: the half-width of the iid ensemble band
        delta_Ks.append(delta_K)
        gap = float(ec.mean() - ei.mean())
        se = float(np.hypot(ec.std() / max(len(ec), 1) ** 0.5, ei.std() / max(len(ei), 1) ** 0.5))
        sat = BC.saturation_guard(
            damage_treat=intact - ec.mean() * (intact - floor),
            damage_null=intact - ei.mean() * (intact - floor),
            floor=floor, intact=intact, name=name)
        per_level[name] = {
            "coherent_effect": float(ec.mean()), "independent_effect": float(ei.mean()),
            "gap": gap, "gap_cilo": gap - 1.96 * se, "gap_cihi": gap + 1.96 * se,
            "delta_K": delta_K, "void": sat["void"], "saturation": sat,
            "coherent_dppl": float(dppl(coh).mean()), "independent_dppl": float(dppl(ind).mean()),
            "dppl_ratio": float(dppl(coh).mean() / max(dppl(ind).mean(), 1e-9)),
            "clears": bool((not sat["void"]) and (gap - 1.96 * se) > delta_K)}
    rep["per_level"] = per_level
    delta_K_global = float(np.mean(delta_Ks)) if delta_Ks else 0.0
    rep["delta_K"] = delta_K_global

    cells_adj = {
        "P1": {"gap_cilo": min(per_level["P1@benign"]["gap_cilo"], per_level["P1@mid"]["gap_cilo"]),
               "levels_cleared": int(per_level["P1@benign"]["clears"]) + int(per_level["P1@mid"]["clears"]),
               "void": per_level["P1@benign"]["void"] and per_level["P1@mid"]["void"]},
    }
    p3 = [v for k, v in per_level.items() if k.startswith("P3@") and v.get("status") != "MISSING_ARM"]
    if p3:
        cells_adj["P3"] = {"gap_cilo": min(v["gap_cilo"] for v in p3),
                           "levels_cleared": sum(int(v["clears"]) for v in p3),
                           "void": all(v["void"] for v in p3)}
    rep["readjudication"] = BC.adjudicate_coherence(cells_adj, delta_K_global)
    rep["readjudication"]["cells_adj"] = cells_adj

    # §4.2 says "δ_K **per readout**". The job scored only the VQA-effect scale. The ppl readout has
    # its own band and its own test, and the rule is applied to it here, unchanged.
    ppl_levels, ppl_dKs = {}, []
    for name, (coh, ind) in pairs.items():
        if not coh or not ind:
            continue
        pc, pi = dppl(coh), dppl(ind)
        dK = float(pi.std())
        ppl_dKs.append(dK)
        gap = float(pc.mean() - pi.mean())
        se = float(np.hypot(pc.std() / max(len(pc), 1) ** 0.5, pi.std() / max(len(pi), 1) ** 0.5))
        ppl_levels[name] = {"coherent_dppl": float(pc.mean()), "independent_dppl": float(pi.mean()),
                            "gap": gap, "gap_cilo": gap - 1.96 * se, "delta_K_ppl": dK,
                            "clears": bool((gap - 1.96 * se) > dK)}
    dK_ppl = float(np.mean(ppl_dKs)) if ppl_dKs else 0.0
    cells_ppl = {
        "P1": {"gap_cilo": min(ppl_levels["P1@benign"]["gap_cilo"], ppl_levels["P1@mid"]["gap_cilo"]),
               "levels_cleared": int(ppl_levels["P1@benign"]["clears"]) + int(ppl_levels["P1@mid"]["clears"]),
               "void": False}}
    p3p = [v for kk, v in ppl_levels.items() if kk.startswith("P3@")]
    if p3p:
        cells_ppl["P3"] = {"gap_cilo": min(v["gap_cilo"] for v in p3p),
                           "levels_cleared": sum(int(v["clears"]) for v in p3p), "void": False}
    rep["readjudication_ppl"] = BC.adjudicate_coherence(cells_ppl, dK_ppl)
    rep["readjudication_ppl"]["per_level"] = ppl_levels
    rep["readjudication_ppl"]["delta_K_ppl"] = dK_ppl

    # ---- T3: the mean-shift-free coherence control, already inside Arm K ----------
    # P1's shared arm ADDS the same vector at every position -> it induces a stream MEAN SHIFT.
    # P3's shared arm REMOVES the same subspace at every position -> equally coherent across the
    # sequence, but a projection toward h̄ induces NO mean shift. If sequence coherence per se were
    # the driver, both should be large. Their ratio isolates the mean shift.
    p1_ratio = per_level["P1@mid"]["dppl_ratio"]
    p3_ratios = [v["dppl_ratio"] for kk, v in per_level.items() if kk.startswith("P3@")]
    # The P1 coherent arm's MEAN is driven by one catastrophic draw (2624 vs 24 and 52), so the
    # isolation factor is also reported CONSERVATIVELY, from the weakest coherent draw against the
    # strongest independent one. The conclusion must not depend on the unstable mean.
    _cp = [c["dppl"] for c in sel(family="P1g", arm="coherent", level="mid")]
    _ip = [c["dppl"] for c in sel(family="P1g", arm="independent", level="mid")]
    p1_ratio_conservative = (min(_cp) / max(_ip)) if (_cp and _ip) else None
    rep["T3_meanshift_free_control"] = {
        "P1_shared_add_dppl_ratio": p1_ratio,
        "P1_shared_add_dppl_ratio_conservative_min_over_max": p1_ratio_conservative,
        "P3_shared_removal_dppl_ratio": p3_ratios,
        "P3_mean": float(np.mean(p3_ratios)) if p3_ratios else None,
        "isolation": float(p1_ratio / np.mean(p3_ratios)) if p3_ratios else None,
        "isolation_conservative": (float(p1_ratio_conservative / np.mean(p3_ratios))
                                   if (p3_ratios and p1_ratio_conservative) else None),
        "logic": "P1-shared and P3-shared are BOTH fully coherent across the sequence. P1-shared "
                 "adds a constant vector (a mean shift); P3-shared removes a fixed subspace toward "
                 "h̄ (no mean shift). Sequence coherence per se buys the P3 ratio; the excess is the "
                 "mean shift.",
        "reading": ("coherence per se is worth ~%.1fx on ppl; adding a coherent constant is worth "
                    "~%.0fx. The gap is the mean-shift mechanism." % (float(np.mean(p3_ratios)), p1_ratio))
        if p3_ratios else None}

    # the A9 alignment axis, on the full discovery set
    p2 = {}
    for lev in sorted({c.get("level") for c in C if c["family"] == "P2"}):
        al, pe = sel(family="P2", arm="coherent", level=lev), sel(family="P2", arm="independent", level=lev)
        if not al or not pe:
            continue
        ea, ep = eff(al), eff(pe)
        gap = float(ea.mean() - ep.mean())
        se = float(np.hypot(ea.std() / max(len(ea), 1) ** 0.5, ep.std() / max(len(ep), 1) ** 0.5))
        p2[str(lev)] = BC.adjudicate_alignment(gap=gap, gap_cilo=gap - 1.96 * se,
                                               gap_cihi=gap + 1.96 * se, delta_K=delta_K_global)
        p2[str(lev)].update({"aligned_dvqa": float(np.mean([c["dvqa"] for c in al])),
                             "permuted_dvqa": float(np.mean([c["dvqa"] for c in pe])),
                             "aligned_dppl": float(dppl(al).mean()),
                             "permuted_dppl": float(dppl(pe).mean())})
    rep["alignment_axis"] = p2
    _w()

    # ---------------- PART 2, T1: does the sweep saturate past g ≈ n_eff? ----------------
    gs = sorted({c["g"] for c in C if c["family"] == "P1g" and c["g"] > 0})
    sweep = []
    for g in gs:
        cs = [c for c in C if c["family"] == "P1g" and c["g"] == g]
        sweep.append({"g": g, "dppl": float(dppl(cs).mean()), "effect": float(eff(cs).mean()),
                      "n": len(cs)})
    coh_cells = sel(family="P1g", arm="coherent", level="mid")
    sweep.append({"g": a.seq_len, "g_label": "all", "dppl": float(dppl(coh_cells).mean()),
                  "effect": float(eff(coh_cells).mean()), "n": len(coh_cells)})
    # pooling predicts a plateau for g >= n_eff; measure the growth AFTER the plateau should begin
    post = [s for s in sweep if s["g"] >= 4]
    growth_post_neff = post[-1]["dppl"] / max(post[0]["dppl"], 1e-9)
    rep["T1_scaling"] = {
        "sweep": sweep, "n_eff": N_EFF_GATE0B,
        "dppl_growth_from_g4_to_gall": growth_post_neff,
        "pooling_prediction": "saturation for g >= n_eff ≈ 4 (every position an attention row reads "
                              "already shares a draw); growth ratio ≈ 1",
        "meanshift_prediction": "mean-shift norm ∝ √g, so damage keeps growing without bound",
        "verdict": ("POOLING_REFUTED" if growth_post_neff > 3.0 else "POOLING_CONSISTENT")}
    _w()

    # ---------------- PART 2, T2: is coherent damage direction-dependent? ----------------
    def delta_of(seed, block, sigma):
        gen = torch.Generator(device="cpu").manual_seed(int(seed) * 1_000_003 + int(block))
        return float(sigma) * torch.randn(d, generator=gen, dtype=torch.float32)

    draws = []
    for c in C:
        if c["family"] not in ("P1", "P1g") or c["arm"] == "independent":
            continue
        seed = 11 + 1000 * c["draw"]
        sigma = c["sigma_over_rms"] * rms
        g = c["g"]
        n_blocks = 1 if g == -1 else max(1, a.seq_len // g)
        ds = torch.stack([delta_of(seed, b, sigma) for b in range(min(n_blocks, 16))]).double()
        dbar = ds.mean(0)                              # the stream-level mean shift this cell induces
        dhat = dbar / dbar.norm().clamp_min(1e-12)
        draws.append({"cell": c["cell"], "g": g, "draw": c["draw"], "level": c.get("level"),
                      "dppl": c["dppl"], "effect": c["effect"],
                      "abs_cos_e1": float(abs(dhat @ e1)), "abs_cos_hbar": float(abs(dhat @ hhat)),
                      "mean_shift_norm_over_rms": float(dbar.norm() / rms)})
    rep["draws"] = draws
    _w()

    def corr(xs, ys):
        x, y = np.asarray(xs, float), np.asarray(ys, float)
        if len(x) < 4 or len(set(x)) < 3:
            return None
        rho = float(spearmanr(x, y).statistic)
        rng = np.random.default_rng(1337)
        b = []
        for _ in range(a.n_boot):
            i = rng.integers(0, len(x), len(x))
            if len(set(x[i])) < 3:
                continue
            v = spearmanr(x[i], y[i]).statistic
            if np.isfinite(v):
                b.append(v)
        b = np.asarray(b)
        return {"rho": rho, "n": int(len(x)),
                "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}

    # the fully-coherent + strongly-grouped cells at the mid level: the regime where the two
    # mechanisms make different predictions
    coh = [r for r in draws if r["g"] in (-1, 256, 64) and r["level"] == "mid"]
    rep["T2_direction"] = {
        "n_coherent_draws": len(coh),
        "abs_cos_e1_vs_dppl": corr([r["abs_cos_e1"] for r in coh], [r["dppl"] for r in coh]),
        "abs_cos_hbar_vs_dppl": corr([r["abs_cos_hbar"] for r in coh], [r["dppl"] for r in coh]),
        "mean_shift_norm_vs_dppl": corr([r["mean_shift_norm_over_rms"] for r in coh],
                                        [r["dppl"] for r in coh]),
        "pooling_prediction": "direction-blind: rho ≈ 0",
        "meanshift_prediction": "rho > 0 with |cos(δ̄, e₁)|"}
    _w()

    # ---- T2b: the direction-blindness test that does NOT need a correlation ----------
    # Attention pooling is direction-BLIND: at fixed per-token energy it predicts the same damage
    # for every coherent draw, so the coherent arm's draw-to-draw spread should be sampling noise —
    # the same as the independent arm's. The mean-shift mechanism predicts the opposite: a coherent
    # draw's damage depends on where δ̄ points relative to the data, so the spread should be large
    # AND should grow with g (because the mean-shift norm grows like √g).
    #
    # This is a much stronger instrument than T2's rank correlation at n = 9, and it needs no
    # eigenbasis at all: it is a statement about the damage distribution alone.
    def spread(vals):
        v = np.asarray(vals, float)
        return float(v.max() / max(v.min(), 1e-9))

    by_g = {}
    for g in sorted({c["g"] for c in C if c["family"] == "P1g"}):
        cs = [c for c in C if c["family"] == "P1g" and c["g"] == g]
        if len(cs) >= 2:
            by_g[("all" if g == -1 else str(g))] = {
                "dppl": [c["dppl"] for c in cs], "spread": spread([c["dppl"] for c in cs])}
    ind_spread = by_g.get("1", {}).get("spread")
    coh_spread = by_g.get("all", {}).get("spread")
    # the nonparametric separation: does EVERY coherent draw beat EVERY independent draw?
    coh_p = [c["dppl"] for c in sel(family="P1g", arm="coherent", level="mid")]
    ind_p = [c["dppl"] for c in sel(family="P1g", arm="independent", level="mid")]
    from math import comb
    strict_sep = bool(coh_p and ind_p and min(coh_p) > max(ind_p))
    exact_p = 1.0 / comb(len(coh_p) + len(ind_p), len(coh_p)) if strict_sep else None
    rep["T2b_direction_blindness"] = {
        "by_g": by_g,
        "independent_draw_spread": ind_spread, "coherent_draw_spread": coh_spread,
        "spread_ratio": (coh_spread / ind_spread) if (coh_spread and ind_spread) else None,
        "strict_separation_every_coherent_beats_every_independent": strict_sep,
        "exact_one_sided_p": exact_p,
        "min_coherent_over_max_independent": (min(coh_p) / max(ind_p)) if (coh_p and ind_p) else None,
        "pooling_prediction": "direction-blind ⇒ coherent spread ≈ independent spread (sampling noise)",
        "meanshift_prediction": "coherent spread ≫ independent spread, and grows with g",
        "verdict": ("POOLING_REFUTED_DIRECTION_DEPENDENT"
                    if (coh_spread and ind_spread and coh_spread / ind_spread > 5.0)
                    else "DIRECTION_BLIND_CONSISTENT_WITH_POOLING")}
    _w()

    ce = rep["T2_direction"]["abs_cos_e1_vs_dppl"]
    t1_refutes = rep["T1_scaling"]["verdict"] == "POOLING_REFUTED"
    t2_directional = (bool(ce and (ce["ci95"][0] > 0))
                      or rep["T2b_direction_blindness"]["verdict"].startswith("POOLING_REFUTED"))
    t3 = rep["T3_meanshift_free_control"]
    # T3 fires when the mean-shift-free coherent arm (P3) buys an order of magnitude less than the
    # mean-shift-inducing coherent arm (P1) at matched coherence.
    # fire on the CONSERVATIVE isolation, so one catastrophic draw cannot carry the verdict
    t3_isolates = bool(t3.get("isolation_conservative") and t3["isolation_conservative"] > 2.0)
    if t1_refutes and (t2_directional or t3_isolates):
        label = "H_coherence-is-meanshift"
    elif t1_refutes:
        label = "H_coherence-not-pooling-mechanism-open"
    elif not t2_directional and not t3_isolates:
        label = "H_coherence-is-pooling"
    else:
        label = "H_mixed"
    rep["mechanism_verdict"] = {
        "label": label, "T1": rep["T1_scaling"]["verdict"],
        "T2_rho": (ce or {}).get("rho"), "T2_ci95": (ce or {}).get("ci95"),
        "T3_isolation": t3.get("isolation"), "T3_fires": t3_isolates,
        "reading": {
            "H_coherence-is-meanshift":
                "Arm K's sequence axis and Arm A's mean-shift mechanism are NOT independent "
                "evidence. P1's shared arm needs a norm-coherence control; the surface's ω=DC "
                "column is partly a restatement of its feature axis.",
            "H_coherence-not-pooling-mechanism-open":
                "the pooling law is refuted by its own saturation prediction, but coherent damage is "
                "not demonstrably direction-dependent at this draw count. The mechanism is open and "
                "the sweep is underpowered for T2.",
            "H_coherence-is-pooling":
                "the sweep plateaus past g ≈ n_eff and is direction-blind: the cancellation law holds.",
            "H_mixed": "the two tests disagree; report both and do not unify."}[label]}
    _w()

    print("BISPEC_COHALIGN_OK", json.dumps(
        {"readjudication_vqa": rep["readjudication"]["verdict"],
         "readjudication_ppl": rep["readjudication_ppl"]["verdict"],
         "original": (rep["original_k_adjudication_PRESERVED"] or {}).get("verdict"),
         "mechanism": label, "T1": rep["T1_scaling"]["verdict"],
         "T3_isolation": t3.get("isolation"), "T3_isolation_conservative": t3.get("isolation_conservative"),
         "T2b": rep["T2b_direction_blindness"]["verdict"],
         "coherent_draw_spread": rep["T2b_direction_blindness"]["coherent_draw_spread"],
         "independent_draw_spread": rep["T2b_direction_blindness"]["independent_draw_spread"],
         "growth_g4_to_gall": round(growth_post_neff, 2)}, default=str))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
