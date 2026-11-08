"""HANDLE Arm N — naming round 2 (spec §3.1 Arm N, §6 R-N-a/R-N-b, §5 controls, §11 P7/P8).

The two oldest unnamed objects of the lineage, each on a near-zero-forward budget:

  --phase carrier  (CPU, ZERO forwards; R-N-a)
      The 5-candidate carrier regression on the banked 9 coherent draws. Each draw's mean-shift
      DIRECTION is reconstructed deterministically from its seed via the make_hook_gaussian_group
      block-average (no forward), and cross-checked bit-close against the banked abs_cos_hbar /
      abs_cos_e1 covariates (fail loud on mismatch). Five candidate features are scored per draw;
      τ = carry_core.kendall0(feature, damage) with an EXACT n=9 permutation p; BH over 5 →
      handle_core.adjudicate_rna.

  --phase spikes   (GPU; R-N-b)
      Census the 8 promoted L0 spike eigvecs against the SigLIP-side library — patch-position
      2-D DCT harmonics + image-mean (built from captured L0 image-token residuals), projector
      rowspace, embedding-PCA axes, and massive-coordinate axes — vs a matched-rank Haar p95
      (the specter_c C-4 census form). If any element identifies (≥3× p95), the ONE causal cell
      mean-ablates the best-matching structure from the L0 residual (a projector hook) and
      remeasures the best-matching spike's operator Rayleigh quotient vᵀH_ω v (must drop ≥50%).
      → handle_core.adjudicate_rnb.

Feeds the ALREADY-WRITTEN adjudicators handle_core.adjudicate_rna / adjudicate_rnb (never
redefined). Swap-shell / operator machinery REUSED (specter_core operator, inherit_pg1.ablate_at,
bispec_core / projector_ablation library builders). Q eigvectors, spikes, Σ/h̄, γ, massive coords,
the 9 draws — every load-bearing number RECOMPUTED from its named artifact at run time (baseline
discipline). No try/except hiding — a wrong assumption RAISES (CLAUDE.md).

§0.5 amendments the lead must ratify (see the delivery report; dated 2026-07-13):
  HN-a1  R-N-a canonical draw set = the bispec_cohalign mid-level g-sweep (9 draws, g∈{all,256,64}
         × seeds 11/1011/2011) — the set with the 110× damage SPREAD the carrier exists to
         explain and the ONLY banked set on which all 5 candidates (incl. the g-dependent
         n_eff interaction) have range. specter_c's c1_rows (all g=all) is reported for P7
         continuity only.
  HN-a2  candidate `spike_p8` is RECOMPUTED as the L0/dc/ppl top-8-spike-subspace energy of each
         draw's mean-shift dbar (‖P8ᵀ dbar‖²), not reused from specter_c's banked reg_p8 (a
         different draw set); the P8 basis is the same one specter_c banked.
  HN-a3  candidate `n_eff_interaction` is AUTHOR-DEFINED (the spec left it to "define cleanly"):
         the mean-shift-mechanism composite  mean_shift_norm_over_rms · |cos(dbar, e₁)|  —
         g-driven mean-shift magnitude (∝√g) modulated by alignment to the top data direction.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.handle_n --type r7i.2xlarge --no-weights \
      --s3-prefix results/handle_n_carrier --budget-min 60 --args "--phase carrier"
  ./infra/aws_pg/pg run infra.aws_pg.jobs.handle_n --type g6e.8xlarge,p5.4xlarge --no-weights \
      --s3-prefix results/handle_n_spikes --budget-min 150 --args "--phase spikes"
"""
import argparse
import itertools
import json
import math
import os
import pathlib

# ---- frozen constants (spec §3.1/§6; reconstruction conventions from bispec_cohalign) --------
SEQ_LEN = 512             # bispec_cohalign mean-shift window (its --seq-len default)
N_BLOCKS_CAP = 16         # bispec_cohalign caps the mean-shift average at 16 blocks
CROSSCHECK_TOL = 0.02     # reconstructed |cos| vs the banked covariate (fp32-draw agreement)
SINK_TOP = 4              # per-layer top sink coords for the persistent-channel union
R_LIB = 16               # matched census rank for the SUBSPACE library elements (handle_t R_CENSUS)
N_HAAR = 4000            # Haar null draws (specter_c C-4 convention)
CAND_NAMES = ("massive_coord_frac", "hbar_overlap", "spike_p8", "gamma_overlap", "n_eff_interaction")


# ===========================================================================
# Shared: the persistent massive-coordinate set (recomputed, never transcribed).
# ===========================================================================
def _recompute_massive(carry_gate0):
    """Massive-activation coords = coords in the top-SINK_TOP of ≥2 of the L0/L9/L17 γ-sink
    readouts (the 'union of persistent channels' rule; canonical set 738/674/604/2015 emerges,
    but the recomputed set is what is USED). Returns (sorted list, per-layer top table)."""
    from collections import Counter
    gsr = carry_gate0["gamma_sink_readout"]
    cnt, per_layer = Counter(), {}
    for lk, row in gsr.items():
        top = [int(c) for c in row["coords"][:SINK_TOP]]
        per_layer[lk] = top
        for c in set(top):
            cnt[c] += 1
    massive = sorted(c for c, n in cnt.items() if n >= 2)
    assert massive, "no persistent massive coords recomputed from the γ-sink readout"
    return massive, per_layer


def _exact_perm_tau(score, damage):
    """τ = carry_core.kendall0(score, damage) + EXACT one-sided permutation p (n≤10 enumerate;
    else a large random-perm null). A fully-tied candidate ⇒ τ:=0, p:=1 (a signal that cannot
    masquerade as a hit — the degenerate completion, §2)."""
    import numpy as np
    from infra.pg import carry_core as CC
    s = list(map(float, score))
    y = list(map(float, damage))
    n = len(s)
    if len(set(s)) < 2 or len(set(y)) < 2:
        return 0.0, 1.0
    tau_obs = CC.kendall0(s, y)
    if n <= 10:
        cnt = tot = 0
        for perm in itertools.permutations(range(n)):
            t = CC.kendall0(s, [y[i] for i in perm])
            cnt += int(t >= tau_obs - 1e-12)
            tot += 1
        return float(tau_obs), cnt / tot
    rng = np.random.default_rng(1337)
    yy = np.asarray(y, float)
    n_perm, cnt = 200_000, 0
    for _ in range(n_perm):
        cnt += int(CC.kendall0(s, list(yy[rng.permutation(n)])) >= tau_obs - 1e-12)
    return float(tau_obs), (cnt + 1) / (n_perm + 1)


# ===========================================================================
# R-N-a — the carrier regression (CPU, zero forwards).
# ===========================================================================
def _run_carrier(rep, wd, _w):
    import torch

    from infra.aws_pg.jobs import _handle_shared as HS
    from infra.pg import handle_core as HC
    d = 2048

    # ---- parent artifacts -------------------------------------------------------------
    coh = json.loads(HS.fetch(f"{HS.BISPEC_COHALIGN}/cohalign.json",
                              wd / "cohalign.json").read_text())
    assert coh.get("green"), "bispec_cohalign not green — R-N-a refuses"
    # the 9 coherent mid-level g-sweep draws (bispec_cohalign T2 `coh`; the 110× spread object)
    draws = [r for r in coh["draws"] if r["g"] in (-1, 256, 64) and r.get("level") == "mid"]
    assert len(draws) == 9, f"expected 9 coherent mid g-sweep draws, got {len(draws)}"

    sig = torch.load(HS.fetch(f"{HS.BISPEC_SIGMA}/sigma_bank.pt", wd / "sigma_bank.pt"),
                     map_location="cpu", weights_only=False)["L0"]
    hhat = sig["hbar"].double(); hhat = hhat / hhat.norm()
    e1 = sig["basis"][:, 0].double(); e1 = e1 / e1.norm()

    lib = torch.load(HS.fetch(f"{HS.SPECTER_O}/library_bank.pt", wd / "library_bank.pt"),
                     map_location="cpu", weights_only=False)
    ghat = lib["gamma_dir"].double(); ghat = ghat / ghat.norm()

    bank = torch.load(HS.fetch(f"{HS.SPECTER_O}/operator_bank.pt", wd / "operator_bank.pt"),
                      map_location="cpu", weights_only=False)
    sp = bank["L0/dc/ppl"]["spikes"][:8]
    assert sp, "no L0/dc/ppl spikes for the spike-p8 regressor"
    P8 = bank["L0/dc/ppl"]["evecs_A1"][:, [s["idx"] for s in sp]].double()      # (d, 8)

    cg0 = json.loads(HS.fetch(f"{HS.CARRY_GATE0}/gate0.json", wd / "carry_gate0.json").read_text())
    massive, per_layer = _recompute_massive(cg0)
    rep["massive_coords"] = {"recomputed": massive, "per_layer_top": per_layer}

    # ---- reconstruct each draw's mean-shift DIRECTION from its seed --------------------
    def mean_shift_dir(g, draw):
        """dbar direction (bispec_cohalign form): mean over min(T/g,16) block draws of
        randn(d, seed·1_000_003 + block). σ cancels in the unit direction — no σ needed."""
        seed = 11 + 1000 * int(draw)
        n_blocks = 1 if int(g) == -1 else max(1, SEQ_LEN // int(g))
        blocks = []
        for b in range(min(n_blocks, N_BLOCKS_CAP)):
            gen = torch.Generator(device="cpu").manual_seed(seed * 1_000_003 + b)
            blocks.append(torch.randn(d, generator=gen, dtype=torch.float32))
        dbar = torch.stack(blocks).double().mean(0)
        return dbar / dbar.norm()

    rows = []
    for r in draws:
        dhat = mean_shift_dir(r["g"], r["draw"])
        cc_hbar, cc_e1 = float(abs(dhat @ hhat)), float(abs(dhat @ e1))
        assert abs(cc_hbar - float(r["abs_cos_hbar"])) < CROSSCHECK_TOL, \
            f"reconstruction hbar mismatch g={r['g']} draw={r['draw']}: " \
            f"{cc_hbar:.4f} vs banked {r['abs_cos_hbar']:.4f}"
        assert abs(cc_e1 - float(r["abs_cos_e1"])) < CROSSCHECK_TOL, \
            f"reconstruction e1 mismatch g={r['g']} draw={r['draw']}: " \
            f"{cc_e1:.4f} vs banked {r['abs_cos_e1']:.4f}"
        ms = float(r["mean_shift_norm_over_rms"])           # banked g-driven mean-shift norm (∝√g)
        feats = {
            "massive_coord_frac": float(sum(float(dhat[c]) ** 2 for c in massive)),  # σ-invariant frac
            "hbar_overlap": cc_hbar,                                                  # banked, reused
            "spike_p8": float(((dhat @ P8) ** 2).sum()) * (ms ** 2),                  # HN-a2 recompute
            "gamma_overlap": float(abs(dhat @ ghat)),
            "n_eff_interaction": ms * cc_e1,                                          # HN-a3 author-def
        }
        rows.append({"g": int(r["g"]), "draw": int(r["draw"]), "dppl": float(r["dppl"]),
                     "abs_cos_hbar_banked": float(r["abs_cos_hbar"]),
                     "abs_cos_e1_banked": float(r["abs_cos_e1"]),
                     "mean_shift_norm_over_rms": ms,
                     **{f"feat_{k}": v for k, v in feats.items()}})
    rep["carrier_rows"] = rows

    # ---- τ per candidate (exact n=9 permutation) + BH over 5 → adjudicate_rna ----------
    damage = [r["dppl"] for r in rows]
    candidate_stats = {}
    for name in CAND_NAMES:
        tau, p = _exact_perm_tau([r[f"feat_{name}"] for r in rows], damage)
        candidate_stats[name] = {"tau": tau, "p": p}
    rep["candidate_stats"] = candidate_stats
    rep["banked_vs_computed"] = {
        "hbar_overlap": "banked (bispec_cohalign abs_cos_hbar; recomputed + cross-checked)",
        "spike_p8": "recomputed (HN-a2: L0/dc/ppl top-8-spike energy on dbar; P8 basis banked)",
        "massive_coord_frac": "computed (dhat energy frac on the recomputed massive set)",
        "gamma_overlap": "computed (|cos(dbar, specter_o gamma_dir)|)",
        "n_eff_interaction": "author-defined (HN-a3: mean_shift_norm · |cos(dbar, e1)|)"}
    rep["rna"] = HC.adjudicate_rna(candidate_stats, q=0.10)

    # ---- P7 continuity: the literal specter_c banked reg_p8 on its own c1_rows ---------
    sc = json.loads(HS.fetch(f"{HS.S3_BUCKET}/results/specter_c/out/metrics.json",
                             wd / "specter_c.json").read_text())
    c1 = sc.get("c1_rows", [])
    if c1 and all("reg_p8" in r for r in c1):
        disc = [r for r in c1 if r.get("src") == "disc"]
        tau_p7, p_p7 = _exact_perm_tau([r["reg_p8"] for r in disc],
                                       [r["log1p_dppl"] for r in disc])
        rep["p7_continuity"] = {"set": "specter_c c1_rows disc (6 unique, g=all)",
                                "regressor": "banked reg_p8", "n": len(disc),
                                "tau": tau_p7, "p": p_p7,
                                "note": "reported for P7 continuity; NOT in the R-N-a BH family"}
    else:
        rep["p7_continuity"] = {"note": "specter_c c1_rows/reg_p8 unavailable — P7 continuity skipped"}

    rep["gpu_seconds"] = {"total": 0.0, "note": "CPU zero-forward carrier phase"}
    print("HANDLE_N_OK", json.dumps({
        "phase": "carrier", "rna": rep["rna"]["verdict"], "carrier": rep["rna"].get("carrier"),
        "taus": {k: round(v["tau"], 3) for k, v in candidate_stats.items()}}, default=str))


# ===========================================================================
# R-N-b — the SigLIP library builders (vision-side; one capture set).
# ===========================================================================
def _dct_harmonics(grid, r_lib):
    """The r_lib lowest-frequency 2-D DCT-II harmonics over a `grid`×`grid` patch grid (DC
    excluded — it equals the image-mean element), each a (grid²,) function over raster-ordered
    image-token positions p → (row=p//grid, col=p%grid). Returns (H (r_lib, grid²) fp64, freqs)."""
    import torch
    P = grid * grid
    freqs = sorted([(ku, kv) for ku in range(grid) for kv in range(grid) if (ku, kv) != (0, 0)],
                   key=lambda t: (t[0] + t[1], t[0], t[1]))[:r_lib]
    rows = (torch.arange(P) // grid).double()
    cols = (torch.arange(P) % grid).double()
    H = torch.zeros(len(freqs), P, dtype=torch.float64)
    for i, (ku, kv) in enumerate(freqs):
        cu = torch.cos(math.pi * (2.0 * rows + 1.0) * ku / (2.0 * grid))
        cv = torch.cos(math.pi * (2.0 * cols + 1.0) * kv / (2.0 * grid))
        H[i] = cu * cv
    return H, freqs


def _capture_image_residuals(model, proc, layers, rows, task, img_tok, dev, d, n_img):
    """Mean L0 residual profile over the 256 image-token positions, averaged across n_img images.
    Returns Rbar (256, d) fp64. §0.5-5: asserts exactly 256 image tokens per image (fail loud)."""
    import torch
    store = {}

    def cap(_m, args, kwargs):
        hs = args[0] if len(args) >= 1 else kwargs["hidden_states"]
        store["h"] = hs.detach()
        return None

    Rsum, n = None, 0
    for r in rows[:n_img]:
        img = task.image_fn(r).convert("RGB")
        inp = proc(text=[task.prompt_fn(r)], images=[img], return_tensors="pt",
                   padding=False).to(dev)
        inp["pixel_values"] = inp["pixel_values"].to(next(model.parameters()).dtype)
        h = layers[0].register_forward_pre_hook(cap, with_kwargs=True)
        with torch.no_grad():
            model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                  pixel_values=inp["pixel_values"])
        h.remove()
        mask = (inp["input_ids"][0] == img_tok)
        n_img_tok = int(mask.sum())
        assert n_img_tok == 256, f"expected 256 image tokens (16×16), got {n_img_tok}"
        R = store["h"][0][mask].double().cpu()                     # (256, d), raster order
        Rsum = R if Rsum is None else Rsum + R
        n += 1
    assert n >= 1, "no image-token residuals captured"
    return Rsum / n


def _causal_cell(model, proc, layers, spike, struct, hbar_L0, n_op, dev, d):
    """Ablate the matched structure from the L0 residual (mean-ablate span(Q)→h̄) and remeasure the
    spike's operator Rayleigh quotient vᵀH_ω v (ω = the spike's own mode). Manual double-backprop
    path (robust to the injector+ablate two-hook composition). Returns (drop, cell)."""
    import torch

    from infra.aws_pg.jobs import _carry_shared as CS
    from infra.pg import inherit_pg1 as IP
    from infra.pg import specter_core as SC
    Q = (struct.reshape(d, 1) if struct.dim() == 1 else struct).float()
    hbar = hbar_L0.float()
    v = (spike["evec"].float() / spike["evec"].float().norm()).to(dev)
    mode = spike["key"].split("/")[1]
    sign_seed = SC.WHITE_SIGN_SEEDS[0] if mode == "white" else None

    install, holder = SC.make_residual_injector(layers, 0)
    items = [CS.build_ppl_item(model, proc, dev, w, install, holder)
             for w in CS.wikitext_windows(proc, n_op)]
    before = SC.mode_quadform(items, v, mode, sign_seed=sign_seed, path="manual")
    with IP.ablate_at(layers, 0, Q, hbar):
        after = SC.mode_quadform(items, v, mode, sign_seed=sign_seed, path="manual")
    drop = (before - after) / before if abs(before) > 1e-30 else 0.0
    return float(drop), {"before": float(before), "after": float(after), "mode": mode,
                         "n_op": n_op, "struct_rank": int(Q.shape[1])}


# ===========================================================================
# R-N-b driver (GPU): census + (conditionally) the ONE causal cell.
# ===========================================================================
def _run_spikes(rep, wd, tm, _w, args):
    import torch

    from infra.aws_pg.jobs import _carry_shared as CS
    from infra.aws_pg.jobs import _handle_shared as HS
    from infra.pg import handle_core as HC
    from infra.pg import inherit_pg1 as IP
    from infra.pg import projector_ablation as PA
    from infra.pg.entities import resolve_submodule_paths
    from infra.pg.projector_ablation import build_tasks, load_eval_items
    d = IP.D

    # ---- the 8 promoted spikes --------------------------------------------------------
    spikes = CS.load_promoted_spikes(str(wd))                    # [{key, idx, eval, evec (d,) fp32}]
    U_spk = [s["evec"].double() / s["evec"].double().norm() for s in spikes]
    rep["spikes"] = [{"key": s["key"], "idx": s["idx"], "eval": s["eval"]} for s in spikes]

    with tm.stage("load_model"):
        sha, _ = HS.pinned_pg1_sha(str(wd))
        model, proc = IP.load_pg1(sha=sha)
        layers = IP.decoder_layers(model)
        dev = next(model.parameters()).device
        paths = resolve_submodule_paths(model)
        g0 = json.loads(HS.fetch(f"{HS.HANDLE_GATE0}/gate0.json", wd / "g0.json").read_text())
        assert g0.get("green"), "handle gate0 not green"
        pp = IP.ppl_gate(model, proc, "cuda", n_seq=args.n_ppl_seq)
        assert abs(pp["ppl"] - g0["ppl_anchor"]["ppl"]) <= 0.5, "base ppl drifted vs gate0"
        rep["base_ppl"] = pp["ppl"]
    _w()

    # ---- weight-side library elements (zero-forward) ----------------------------------
    with tm.stage("weight_library"):
        _pm, W_proj, _b = PA.get_projector(model, paths)                       # (2048, 1152)
        U_proj = PA.svd_U(W_proj.double())[0][:, :R_LIB].contiguous()          # projector rowspace
        emb = model.get_submodule(paths["language_model"]).get_input_embeddings() \
            .weight.detach().float()
        embc = emb - emb.mean(0, keepdim=True)
        cov = (embc.transpose(0, 1) @ embc).double() / float(embc.shape[0])    # (d, d)
        ev, evec = torch.linalg.eigh(0.5 * (cov + cov.transpose(0, 1)))
        U_emb = evec[:, torch.argsort(ev, descending=True)[:R_LIB]].contiguous().cpu()
        del emb, embc, cov, ev, evec
        cg0 = json.loads(HS.fetch(f"{HS.CARRY_GATE0}/gate0.json",
                                  wd / "carry_gate0.json").read_text())
        massive, per_layer = _recompute_massive(cg0)
        Q_mass = torch.zeros(d, len(massive), dtype=torch.float64)
        for j, c in enumerate(massive):
            Q_mass[c, j] = 1.0
        sig = torch.load(HS.fetch(f"{HS.BISPEC_SIGMA}/sigma_bank.pt", wd / "sigma_bank.pt"),
                         map_location="cpu", weights_only=False)["L0"]
        hbar_L0 = sig["hbar"].float()
        U_proj, U_emb, Q_mass = U_proj.cpu(), U_emb.cpu(), Q_mass
        rep["massive_coords"] = {"recomputed": massive, "per_layer_top": per_layer}
    _w()

    # ---- vision-side library: L0 image-token residual profile (ONE capture set) --------
    with tm.stage("image_residuals"):
        img_tok = getattr(model.config, "image_token_index", None)
        assert img_tok is not None, "PG1 config has no image_token_index (§0.5-5)"
        task = {t.name: t for t in build_tasks()}["vqav2"]
        rows_img = load_eval_items(task, max(args.n_img, 16))
        Rbar = _capture_image_residuals(model, proc, layers, rows_img, task, int(img_tok),
                                        dev, d, args.n_img)                    # (256, d) fp64
        image_mean = Rbar.mean(0); image_mean = (image_mean / image_mean.norm())
        H, freqs = _dct_harmonics(16, R_LIB)                                   # (R_LIB, 256)
        Q_harm, _ = torch.linalg.qr((H @ Rbar).transpose(0, 1))                # (d, R_LIB)
        Q_harm = Q_harm[:, :R_LIB].contiguous()
        rep["vision_library"] = {"n_img": args.n_img, "n_harmonics": len(freqs),
                                 "image_token_index": int(img_tok)}
    _w()

    # ---- the census (specter_c C-4 form): 8 spikes × library vs matched-rank Haar p95 --
    with tm.stage("census"):
        vec_lib = {"image_mean": image_mean}
        sub_lib = {"patch_harmonics": Q_harm, "projector_rowspace": U_proj,
                   "embedding_pca": U_emb, "massive_coords": Q_mass}
        gen = torch.Generator(device="cpu").manual_seed(909)
        haar = torch.randn(N_HAAR, d, generator=gen, dtype=torch.float64)
        haar = haar / haar.norm(dim=1, keepdim=True)
        per_spike = {}
        for i, u in enumerate(U_spk):
            row = {nm: float(abs(u @ v)) for nm, v in vec_lib.items()}
            row.update({nm: float((Q.transpose(0, 1) @ u).norm()) for nm, Q in sub_lib.items()})
            per_spike[f"spike{i}"] = {"key": spikes[i]["key"], "idx": spikes[i]["idx"], **row}
        haar_p95 = {nm: float((haar @ v).abs().quantile(0.95)) for nm, v in vec_lib.items()}
        haar_p95.update({nm: float((haar @ Q).norm(dim=1).quantile(0.95))
                         for nm, Q in sub_lib.items()})
        elems = list(vec_lib) + list(sub_lib)
        library_overlaps = {e: max(per_spike[f"spike{i}"][e] for i in range(len(U_spk)))
                            for e in elems}
        argmax_spike = {e: int(max(range(len(U_spk)),
                                   key=lambda i: per_spike[f"spike{i}"][e])) for e in elems}
        ranks = {"image_mean": 1, "patch_harmonics": R_LIB, "projector_rowspace": R_LIB,
                 "embedding_pca": R_LIB, "massive_coords": len(massive)}
        rep["census"] = {"per_spike": per_spike, "haar_p95": haar_p95,
                         "library_overlaps": library_overlaps, "argmax_spike": argmax_spike,
                         "ranks": ranks, "elements_built": elems, "elements_dropped": []}
    _w()

    # ---- identification + (only if identified) the ONE causal cell --------------------
    ratios = {e: (library_overlaps[e] / haar_p95[e] if haar_p95[e] > 0 else 0.0) for e in elems}
    rep["census"]["ratios"] = ratios
    identified = [e for e in elems if ratios[e] >= HC.RNB_IDENT_MULT]
    rayleigh_drop = None
    if identified:
        best_elem = max(identified, key=lambda e: ratios[e])
        best_spike = argmax_spike[best_elem]
        struct = {**vec_lib, **sub_lib}[best_elem]
        with tm.stage("causal_cell"):
            rayleigh_drop, cell = _causal_cell(model, proc, layers, spikes[best_spike],
                                               struct, hbar_L0, args.n_op, dev, d)
        rep["causal_cell"] = {"best_elem": best_elem, "best_spike": best_spike,
                              "best_spike_key": spikes[best_spike]["key"],
                              "identification_ratio": ratios[best_elem], **cell,
                              "rayleigh_drop": rayleigh_drop}
        _w()

    rep["rnb"] = HC.adjudicate_rnb(library_overlaps, haar_p95, rayleigh_drop=rayleigh_drop)
    HS.assert_gpu_seconds(rep)
    print("HANDLE_N_OK", json.dumps({"phase": "spikes", "rnb": rep["rnb"]["verdict"],
                                     "identified": identified, "rayleigh_drop": rayleigh_drop},
                                    default=str))


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", required=True, choices=("carrier", "spikes"))
    ap.add_argument("--n-img", type=int, default=16, help="images for the L0 residual profile")
    ap.add_argument("--n-op", type=int, default=8, help="ppl operator windows for the causal cell")
    ap.add_argument("--n-ppl-seq", type=int, default=120)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.aws_pg.jobs import _handle_shared as HS

    freeze = HS.assert_frozen("n")
    HS.install_pg()
    rep = {"object": f"handle_n_{a.phase}_v1", "freeze": freeze, "phase": a.phase}
    tm = HS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    wd = pathlib.Path("/tmp/handle_n")
    wd.mkdir(parents=True, exist_ok=True)

    if a.phase == "carrier":
        _run_carrier(rep, wd, _w)
    else:
        from infra.pg.diff_harvest import hf_online_setup
        hf_online_setup()
        _run_spikes(rep, wd, tm, _w, a)

    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
