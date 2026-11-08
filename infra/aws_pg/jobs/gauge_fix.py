"""GAUGE-FIX (P2) — gauge-fixed differencing over the 24-FT PG1 ΔW family (CPU, --no-weights).

Streams the 24 fine-tunes exactly as invariant_core.py does (enumerate -224 live, pin SHAs, load base
once, difference per decoder matrix in fp64, delete cache). Per FT, per decoder LAYER, aligns the
fine-tune to the base over the frozen symmetry catalogue (infra/pg/gauge_fix.py §5.2): the MLP
hidden-unit permutation (Hungarian on gate/up rows + down cols), the GQA-consistent attention-head
permutation (within-group Hungarian on q/k/v/o blocks), and the RMSNorm-gain ↔ adjacent-matrix
positive-diagonal scale transfer. It reports the gauge fraction

    γ = 1 − ‖W_ft − g*·W_base‖²_F / ‖W_ft − W_base‖²_F   per (attn, mlp, pooled), per FT

against the OVERFIT-NULL band δ_GF — the same aligner run on (W_base + spectrum-matched rotation-null
Δ) pairs, ≥3 draws — plus a planted-gauge positive control (aligner must recover γ≈1 on a known
permutation) and an identity check (align base to itself → γ=0). Detection (§5.3): family-median γ
excess CI-lo > δ_GF on discovery AND on the sealed 12.

FROZEN pre-flight: gauge_fix.catalogue_dry_run runs at startup — every generator computes the identical
function on a tiny numpy layer (RoPE included) before any weight touches it.

Head geometry is PARAMETERIZED (--n-heads/--n-kv/--head-dim) and ASSERTED against the real q/k/v shapes
at load — Gemma-1 / gemma-2b is MQA (8 query heads, 1 kv head, head_dim 256, d_model 2048); a mismatch
fails loud rather than guessing the GQA grouping. The MLP Hungarian is O(hidden³) at hidden=16384; the
frozen mitigation (§5.4) is a budget cap: layers with hidden > --max-hidden report the MLP (and pooled)
subsystem UNALIGNED — never silently identity (§5.2 intercurrent-events rule).

  # smoke (2 ft, one layer, MQA geometry, 3 null draws):
  pg run infra.aws_pg.jobs.gauge_fix --type r7i.4xlarge --no-weights \
      --args "--ft-tasks vqav2,gqa --n-heads 8 --n-kv 1 --head-dim 256 --layers 0 --n-null 3" --budget-min 90
  # full (24 ft, all 18 layers, budget the 16384-hidden MLP Hungarian):
  pg run infra.aws_pg.jobs.gauge_fix --type r7i.8xlarge --no-weights \
      --args "--n-heads 8 --n-kv 1 --head-dim 256 --n-null 5 --max-hidden 16384" --budget-min 600
"""
import argparse
import json
import pathlib

from infra.pg import gauge_common as GC
from infra.pg import gauge_fix as GF
# REUSED streaming spine (invariant_core.py): base/ft decoder load (full decoder sd incl. norm gains),
# matrix matching, projection-type lookup, HF cache delete.
from infra.aws_pg.jobs.invariant_core import (PG1_REPO, PROJ, _load_decoder, _decoder_matrices,
                                              _delete_cache)

SUBSYS = ("attn", "mlp", "pooled")          # P2 promoted family: pooled + per-subsystem rows (≤3)


def _layer_bundle(sd, layer, np, keep_types=PROJ):
    """Pull one decoder layer's weight bundle: the `keep_types` projection matrices (fp64) + the two
    RMSNorm EFFECTIVE gains (Gemma RMSNorm gain = 1 + weight). `keep_types` defaults to all 7 but the
    job restricts it to the ALIGNED types — when the 16384-wide MLP is UNALIGNED (§5.4) its matrices
    are never read into a bundle, cutting per-layer fp64 memory ~13× (a 16384×2048 matrix is 268 MB;
    holding all 18 layers' gate/up/down was ~16 GB and OOM'd a 128 GB box). Fail loud on any missing
    key — a missing norm means a Gemma-2 (4-norm) layer slipped the firewall."""
    pre = f"layers.{layer}."
    b = {}
    for t in keep_types:
        key = f"{pre}{'self_attn' if t[0] in 'qkvo' else 'mlp'}.{t}.weight"
        assert key in sd, f"missing {key}"
        b[t] = sd[key].to("cpu").double().numpy()
    for g, k in (("in_gain", "input_layernorm.weight"), ("post_gain", "post_attention_layernorm.weight")):
        key = pre + k
        assert key in sd, f"missing {key} (Gemma-1 has exactly input+post_attention norms per layer)"
        b[g] = 1.0 + sd[key].to("cpu").double().numpy()
    return b


def _scale_choice(gain_b, gain_f, mats_b, mats_f, np):
    """Best of {closed-form positive scale transfer, identity} for a (norm gain, consuming matrices)
    group — the per-generator identity guard (§5.3, identity ∈ catalogue ⇒ γ≥0). `mats_b` are the base
    matrices AFTER any discrete permutation. Returns min aligned residual over {gain, mats}."""
    D = np.clip(gain_f / gain_b, 1e-3, 1e3)
    g2, m2 = GF.norm_scale_transfer(gain_b, mats_b, D)
    scaled = GF.frob2(gain_f - g2) + sum(GF.frob2(Wf - Wb) for Wf, Wb in zip(mats_f, m2))
    ident = GF.frob2(gain_f - gain_b) + sum(GF.frob2(Wf - Wb) for Wf, Wb in zip(mats_f, mats_b))
    return float(min(scaled, ident))


def _attn_gamma(base, ft, n_heads, n_kv, head_dim, np):
    """ATTN subsystem {q,k,v,o + input norm}: head permutation, then input-norm scale transfer on q/k/v
    columns. Returns {aligned2, raw2}."""
    ar = GF.attn_align(base, ft, n_heads, n_kv, head_dim)
    q_hp, k_hp, v_hp, o_hp = GF.attn_permute(base["q_proj"], base["k_proj"], base["v_proj"],
                                             base["o_proj"], ar["q_head_src"], ar["kv_src"], head_dim)
    qkvg = _scale_choice(base["in_gain"], ft["in_gain"], [q_hp, k_hp, v_hp],
                         [ft["q_proj"], ft["k_proj"], ft["v_proj"]], np)
    aligned = qkvg + GF.frob2(ft["o_proj"] - o_hp)
    raw = (GF.frob2(ft["q_proj"] - base["q_proj"]) + GF.frob2(ft["k_proj"] - base["k_proj"])
           + GF.frob2(ft["v_proj"] - base["v_proj"]) + GF.frob2(ft["o_proj"] - base["o_proj"])
           + GF.frob2(ft["in_gain"] - base["in_gain"]))
    return {"aligned2": aligned, "raw2": raw}


def _mlp_gamma(base, ft, np):
    """MLP subsystem {gate,up,down + post-attn norm}: hidden permutation (Hungarian), then post-norm
    scale transfer on gate/up columns. Returns {aligned2, raw2}."""
    mr = GF.mlp_align(base, ft)
    g_hp, u_hp, d_hp = GF.mlp_permute(base["gate_proj"], base["up_proj"], base["down_proj"], mr["hidden_src"])
    gug = _scale_choice(base["post_gain"], ft["post_gain"], [g_hp, u_hp],
                        [ft["gate_proj"], ft["up_proj"]], np)
    aligned = gug + GF.frob2(ft["down_proj"] - d_hp)
    raw = (GF.frob2(ft["gate_proj"] - base["gate_proj"]) + GF.frob2(ft["up_proj"] - base["up_proj"])
           + GF.frob2(ft["down_proj"] - base["down_proj"]) + GF.frob2(ft["post_gain"] - base["post_gain"]))
    return {"aligned2": aligned, "raw2": raw}


def _layer_gammas(base, ft, n_heads, n_kv, head_dim, max_hidden, np):
    """Per-subsystem γ for one layer. ATTN always runs (8 heads). MLP runs iff hidden ≤ max_hidden,
    else UNALIGNED (§5.4 budget cap; never silent identity) ⇒ mlp/pooled γ = None. Returns
    {subsys: γ or None}."""
    attn = _attn_gamma(base, ft, n_heads, n_kv, head_dim, np)
    g_attn = GF.gauge_fraction(attn["aligned2"], attn["raw2"])
    # MLP is UNALIGNED unless its matrices are present in the bundle (kept only when hidden ≤
    # max_hidden) — the bundle omits them otherwise, so a missing key IS the UNALIGNED signal.
    if "gate_proj" not in base or base["gate_proj"].shape[0] > max_hidden:
        return {"attn": g_attn, "mlp": None, "pooled": None}
    mlp = _mlp_gamma(base, ft, np)
    pooled = GF.gauge_fraction(attn["aligned2"] + mlp["aligned2"], attn["raw2"] + mlp["raw2"])
    return {"attn": g_attn, "mlp": GF.gauge_fraction(mlp["aligned2"], mlp["raw2"]), "pooled": pooled}


def _null_ft(base, ft, seed, np):
    """A structureless ft matched to THIS fine-tune's per-matrix ΔW spectrum: base + spectrum-matched
    rotation-null Δ (Haar singular vectors at the real σ) on every projection matrix; norm gains fixed at
    base (the overfit floor lives in the discrete permutations, not the near-1 gains). The optimiser run
    on this pair measures how much γ pure assignment over-fitting extracts from a structureless change of
    the same magnitude (§5.3 overfit null)."""
    nft = {}
    keep = [t for t in PROJ if t in base]          # only the ALIGNED types are in the bundle
    for i, t in enumerate(keep):
        nft[t] = base[t] + GF.spectrum_matched_null(ft[t] - base[t], seed + i)
    nft["in_gain"] = base["in_gain"].copy()
    nft["post_gain"] = base["post_gain"].copy()
    return nft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ft-tasks", default="", help="comma slugs; empty ⇒ enumerate all -224 (24-FT)")
    ap.add_argument("--n-heads", type=int, required=True, help="Gemma-1 query heads (assert vs q_proj)")
    ap.add_argument("--n-kv", type=int, required=True, help="Gemma-1 kv heads/groups (assert vs k_proj)")
    ap.add_argument("--head-dim", type=int, required=True, help="Gemma-1 head_dim (assert vs shapes)")
    ap.add_argument("--n-null", type=int, default=5, help="overfit-null draws (≥3)")
    # Default keeps Gemma-1's 16384-hidden MLP UNALIGNED: an EXACT Hungarian on 16384 units is a
    # 2 GB cost matrix + O(n³) per (layer,FT,null-draw) — an autonomous-run footgun. Attention + norm
    # gauge (well-defined, cheap) always run; MLP-hidden gauge is an explicit opt-in
    # (--max-hidden 16384) or a successor's approximate re-basin matcher. §5.4 budget-cap mitigation.
    ap.add_argument("--max-hidden", type=int, default=4096,
                    help="MLP Hungarian budget cap; hidden>cap ⇒ mlp/pooled UNALIGNED (§5.4). "
                         "Default 4096 ⇒ Gemma-1 MLP UNALIGNED by default; pass 16384 to opt in.")
    ap.add_argument("--layers", default="", help="comma layer subset; empty ⇒ all 18")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    assert a.n_null >= 3, "≥3 overfit-null draws (frozen seed rule §5.3)"
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()                       # FIRST — gated repos 401 otherwise
    import numpy as np
    from huggingface_hub import HfApi

    # FROZEN pre-flight: the whole symmetry catalogue computes the identical function (weight-free).
    dry = GF.catalogue_dry_run()
    print(f"[gauge-fix] catalogue dry-run (max|Δ| per generator): {dry}", flush=True)
    assert all(v < 1e-6 for v in dry.values()), f"catalogue not function-preserving: {dry}"

    api = HfApi()
    if a.ft_tasks.strip():
        repos = [f"google/paligemma-3b-ft-{t.strip()}-224" for t in a.ft_tasks.split(",") if t.strip()]
    else:
        repos = sorted({m.id for m in api.list_models(author="google", search="paligemma-3b-ft")
                        if (m.id.split("/")[-1].endswith("-224") and "-jax" not in m.id
                            and m.id.split("/")[-1].startswith("paligemma-3b-ft-"))})
    shas = {r: api.model_info(r).sha for r in repos}
    base_sha = api.model_info(PG1_REPO).sha
    discovery, confirmation = GC.sha1_split(repos) if len(repos) >= GC.N_DISCOVERY else (repos, [])
    print(f"[gauge-fix] base={PG1_REPO}@{base_sha[:8]} n_ft={len(repos)} "
          f"discovery={len(discovery)} sealed={len(confirmation)} "
          f"heads=({a.n_heads},{a.n_kv},{a.head_dim}) max_hidden={a.max_hidden}", flush=True)

    w_base = _load_decoder(PG1_REPO, base_sha)
    n_layers = GC.assert_gemma1(list(w_base.keys()))
    layers = ([int(x) for x in a.layers.split(",") if x.strip()] if a.layers.strip()
              else list(range(n_layers)))
    # Only bundle the ALIGNED types: attention always; MLP (gate/up/down) only when hidden ≤
    # max_hidden (else UNALIGNED, §5.4). Skipping the 16384-wide MLP keeps base_bundles ~1.2 GB
    # instead of ~16 GB (which OOM'd a 128 GB box), and _null_ft then never builds MLP nulls.
    hidden = w_base["layers.0.mlp.gate_proj.weight"].shape[0]
    align_mlp = hidden <= a.max_hidden
    keep_types = PROJ if align_mlp else ("q_proj", "k_proj", "v_proj", "o_proj")
    print(f"[gauge-fix] hidden={hidden} align_mlp={align_mlp} keep_types={keep_types}", flush=True)
    base_bundles = {L: _layer_bundle(w_base, L, np, keep_types) for L in layers}
    # geometry assertion (fail loud, never guess the GQA grouping).
    q0, k0 = base_bundles[layers[0]]["q_proj"], base_bundles[layers[0]]["k_proj"]
    assert q0.shape[0] == a.n_heads * a.head_dim, f"q rows {q0.shape[0]} != n_heads*head_dim"
    assert k0.shape[0] == a.n_kv * a.head_dim, f"k rows {k0.shape[0]} != n_kv*head_dim"
    assert a.n_heads % a.n_kv == 0, "n_heads not a multiple of n_kv"

    rep = {"object": "gauge_fix_v1", "generation": "gemma1_18L", "base_repo": PG1_REPO,
           "base_sha": base_sha, "cloud_repos": repos, "cloud_shas": shas,
           "discovery": discovery, "confirmation": confirmation, "layers": layers,
           "n_heads": a.n_heads, "n_kv": a.n_kv, "head_dim": a.head_dim, "n_null": a.n_null,
           "max_hidden": a.max_hidden, "seed": a.seed,
           "catalogue": [g["name"] for g in GF.SYMMETRY_CATALOGUE], "catalogue_dry_run": dry,
           "subsystems": list(SUBSYS),
           "estimand": "family-median γ excess over the overfit-null median, per subsystem; "
                       "γ = 1 − ‖W_ft − g*·W_base‖²/‖W_ft − W_base‖² over the frozen symmetry catalogue"}

    # ---- controls (once): identity check + planted-gauge positive on the first layer. ----
    L0 = layers[0]
    ident = _layer_gammas(base_bundles[L0], base_bundles[L0], a.n_heads, a.n_kv, a.head_dim, a.max_hidden, np)
    rep["identity_check"] = ident
    # planted-gauge positive: only run the MLP Hungarian control when MLP is actually ALIGNED (hidden
    # ≤ max_hidden). At the default cap Gemma-1's 16384 MLP is UNALIGNED everywhere else, so aligning
    # a 16384 planted permutation here is inconsistent AND a memory/time hog (2 GB cost matrix +
    # O(n³)); the aligner's teeth are already proven weight-free (test_gauge_fix::test_mlp/attn_align_
    # recovers_planted_*). When UNALIGNED, record that the positive control is deferred to the suite.
    if align_mlp:
        planted_ft = GF.planted_gauge_mlp(base_bundles[L0], seed=a.seed, noise=0.0)
        pr = GF.mlp_align(base_bundles[L0], planted_ft)
        gmlp = GF.gauge_fraction(pr["aligned2"], pr["raw2"])
        rep["planted_gauge_positive"] = {"gamma_mlp": gmlp, "ran": True,
                                         "note": "known hidden-unit permutation, noise=0 ⇒ γ≈1"}
        assert gmlp > 0.99, "planted positive failed: aligner has no teeth"
    else:
        rep["planted_gauge_positive"] = {"gamma_mlp": None, "ran": False,
                                         "note": f"MLP UNALIGNED (hidden {hidden} > max_hidden "
                                                 f"{a.max_hidden}); aligner teeth proven weight-free"}
    print(f"[gauge-fix] identity γ={ident} planted={rep['planted_gauge_positive']}", flush=True)

    # ---- per-FT real γ (per layer) + overfit-null band (matched to the first FT's ΔW spectrum, once). ----
    real = {r: {s: [] for s in SUBSYS} for r in repos}       # per-FT per-subsystem per-layer γ (None-filtered)
    null_pool = {s: [] for s in SUBSYS}                       # all overfit-null γ draws (for δ_GF)
    null_layer_med = {s: {} for s in SUBSYS}                  # per-layer null median (for excess)

    for ridx, repo_id in enumerate(repos):
        print(f"[gauge-fix] align {repo_id}@{shas[repo_id][:8]} ({ridx+1}/{len(repos)})", flush=True)
        w_ft = _load_decoder(repo_id, shas[repo_id])
        if ridx == 0:
            GC.assert_gemma1(list(_decoder_matrices(w_base, w_ft).keys()))     # firewall
        for L in layers:
            ftb = _layer_bundle(w_ft, L, np, keep_types)
            g = _layer_gammas(base_bundles[L], ftb, a.n_heads, a.n_kv, a.head_dim, a.max_hidden, np)
            for s in SUBSYS:
                if g[s] is not None:
                    real[repo_id][s].append(g[s])
            if ridx == 0:                                      # overfit null computed once (base-anchored)
                draws = {s: [] for s in SUBSYS}
                for d in range(a.n_null):
                    seed_d = (a.seed * 7919 + L * 101 + d * 13) % (2 ** 31)
                    nft = _null_ft(base_bundles[L], ftb, seed_d, np)
                    ng = _layer_gammas(base_bundles[L], nft, a.n_heads, a.n_kv, a.head_dim, a.max_hidden, np)
                    for s in SUBSYS:
                        if ng[s] is not None:
                            draws[s].append(ng[s])
                for s in SUBSYS:
                    if draws[s]:
                        null_pool[s].extend(draws[s])
                        null_layer_med[s][L] = float(np.median(draws[s]))
                print(f"[gauge-fix]   L{L} real γ={ {s: (round(g[s],4) if g[s] is not None else None) for s in SUBSYS} } "
                      f"null-med={ {s: (round(null_layer_med[s].get(L),4) if L in null_layer_med[s] else None) for s in SUBSYS} }",
                      flush=True)
        del w_ft
        _delete_cache(repo_id)
        (out / "gauge_fix.json").write_text(json.dumps({**rep, "partial": True}, indent=2))

    # per-FT γ = median over layers; null median = median of per-layer null medians; δ_GF = null half-width.
    null_med = {s: (float(np.median(list(null_layer_med[s].values()))) if null_layer_med[s] else None)
                for s in SUBSYS}
    delta = {s: (GF.delta_gf(null_pool[s]) if len(null_pool[s]) >= 3 else None) for s in SUBSYS}
    rep["overfit_null"] = {s: {"median": null_med[s], "delta_gf": delta[s], "n_draws": len(null_pool[s])}
                           for s in SUBSYS}

    def _rows(repo_set):
        rows = {}
        for s in SUBSYS:
            per_ft = [float(np.median(real[r][s])) for r in repo_set if real[r][s]]
            if not per_ft or null_med[s] is None:
                rows[s] = {"n_ft": len(per_ft), "gamma_median": None, "excess_ci": None}
                continue
            excesses = [g - null_med[s] for g in per_ft]
            rows[s] = {"n_ft": len(per_ft), "gamma_median": float(np.median(per_ft)),
                       "excess_ci": GF.median_excess_ci(excesses)}
        return rows

    disc_rows = _rows(discovery)
    seal_rows = _rows(confirmation) if confirmation else {}
    for s in SUBSYS:
        ci = disc_rows[s]["excess_ci"]
        disc_rows[s]["discovery_clear"] = bool(ci and delta[s] is not None
                                               and GF.discovery_clear(ci["ci_lo"], delta[s]))
    rep["discovery"] = disc_rows
    rep["confirmation"] = seal_rows

    # detection on the sealed 12 over the promoted (≤3) discovery-cleared rows (§5.3). The decision rule
    # is the hard threshold "sealed CI-lo > δ_GF", so BH over ≤3 rows is degenerate (each row all-or-
    # nothing): bh_ok := (sealed CI-lo > δ_GF). Detection = discovery-cleared ∧ that sealed rule.
    promoted = [s for s in SUBSYS if disc_rows[s]["discovery_clear"]][:3]
    rep["detection"] = {}
    for s in promoted:
        sci = seal_rows.get(s, {}).get("excess_ci")
        if sci and delta[s] is not None:
            rep["detection"][s] = GF.detection(sci["ci_lo"], delta[s], bh_ok=(sci["ci_lo"] > delta[s]))
    rep["promoted"] = promoted

    rep.pop("partial", None)
    rep["real_per_ft"] = {r: {s: real[r][s] for s in SUBSYS} for r in repos}
    (out / "gauge_fix.json").write_text(json.dumps(rep, indent=2))
    print("=" * 78)
    print(f"GAUGE_FIX_OK n_ft={len(repos)} promoted={promoted} "
          f"delta_gf={ {s: (round(delta[s], 4) if delta[s] is not None else None) for s in SUBSYS} }")


if __name__ == "__main__":
    main()
