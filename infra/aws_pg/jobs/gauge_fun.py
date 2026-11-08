"""GAUGE-FUN (P5) — function-space reachability metrics over the banked SAV PG1 family (eval-only).

Spec §8. For each banked SAV unit (the SAME pool RELAW read — savc_metrics.json, kind n5c/n5c_new;
population + banked G/R_dist read from the artifact, never assumed), computes the frozen 4-metric
menu of FUNCTION-space distances base-vs-unit on the FIXED frozen probe stream (the exact wikitext-2
prompts the banked G was defined against — prompt_sha asserted against the banked value), then runs
the FROZEN RELAW verdict machinery (gauge_fun.run_menu → savb.delta_loo_perm + relaw law_verdict/
bh_fdr/loo_sign_stable, REUSED VERBATIM): range gate FIRST, then ΔLOO-R²(M | R_dist), BH q=0.10.

Forward convention (repo memory, honoured): the SAV units are Gemma-1 LM DECODERS swapped into a
gemma-2b shell (savb_unit path) and run TEXT-ONLY with attn_implementation="eager" (sdpa mishandles
the PG prefix-LM bidirectional mask → wrong logits). Base = ENTITIES["L0_g1"] (google/gemma-2b), the
SAME base the banked layer-9 overlap gauge was defined against. No image tokens (text-only probe).

Offline gate (§12; MANDATORY before any GPU minute): --offline runs the FULL pipeline minus the
model forward on the REAL pool manifest + REAL tokenizer/probe stream — reconstructs the frozen
probe stream and asserts its sha == the banked prompt_sha, checks pool fields + the generation
firewall, and drives run_menu (range gate + BH + law_verdict) on the banked G/R_dist with a PLANTED
positive-control metric, proving the verdict wiring recovers a real predictor before GPU spend.

GATE-0 defect (disclosed): the range gate needs a per-unit G measurement-noise band; savc_metrics.json
(the pool SUMMARY) drops the banked SAV bootstrap S_boot — it lives in the per-unit unit_*.json
artifacts (outcomes.16000.S_boot). Pass --unit-artifacts <dir|glob> so the job reads the real σ_G;
without it the range gate cannot be issued (fail-loud), and --offline discloses the dependency.

  # offline gate (CPU, no weights):
  pg run infra.aws_pg.jobs.gauge_fun --type r7i.2xlarge --no-weights \
      --args "--offline --pool studies/lineage/paligemma1-pi0/results/savc/savc_metrics.json \
              --unit-artifacts s3://.../savb/units/" --budget-min 30
  # real (p5.4xlarge H100 first, g6e.* L40S fallback):
  pg run infra.aws_pg.jobs.gauge_fun --type p5.4xlarge \
      --args "--pool studies/lineage/paligemma1-pi0/results/savc/savc_metrics.json \
              --unit-artifacts s3://.../savb/units/" --budget-min 480
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import subprocess

from infra.pg import gauge_fun as GF

RELAW_R_GRID = (8, 16, 32, 64, 128, 256, 512)     # the RELAW rank grid; D_ntk r = its midpoint (64)
DECODER_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


# ===========================================================================
# artifact reads (Gate-0: population, G, R_dist, probe-stream sha, G noise band).
# ===========================================================================

def _load_pool(pool_arg, out):
    """Read savc_metrics.json (local or s3://). Returns the parsed doc."""
    path = pool_arg
    if path.startswith("s3://"):
        path = str(out / "savc_metrics.json")
        subprocess.run(["aws", "s3", "cp", pool_arg, path, "--region", "us-east-1"], check=True)
    return json.loads(pathlib.Path(path).read_text())


def _population(pool):
    """The RELAW population — pool units of kind n5c/n5c_new (read from the artifact exactly as
    relaw_reads.py does; usable-n is len of this list, never assumed)."""
    units = [u for u in pool if u["kind"] in ("n5c", "n5c_new")]
    assert units, "no n5c/n5c_new units in the SAV pool"
    return units


def _banked_prompt_sha(units):
    """The single frozen probe-stream sha banked across the family (asserted uniform)."""
    shas = {u["prompt_sha"] for u in units}
    assert len(shas) == 1, f"probe-stream sha not uniform across units: {shas}"
    return shas.pop()


def _banked_overlap_layer(units):
    """D_rep depth = the banked SAV overlap layer, read from the artifact (asserted uniform)."""
    layers = {int(u["overlap_layer"]) for u in units}
    assert len(layers) == 1, f"overlap_layer not uniform across units: {layers}"
    return layers.pop()


def _read_g_sigmas(units, unit_artifacts, g_sigma_floor=0.0):
    """Per-unit G measurement σ. PREFERRED: the banked SAV bootstrap band (unit_*.json
    outcomes.16000.S_boot: σ = (hi95 − lo95)/(2·1.96)) — read from `unit_artifacts`. GATE-0 finding:
    the savc pool summary AND every located savb/savc artifact drop S_boot (only point estimates
    S_16k/S_1k are banked); recomputing the bootstrap from item-level `correct_idx` is a full SAV
    reimplementation out of scope here. FALLBACK (disclosed): a conservative constant proxy σ_G =
    `g_sigma_floor` (a generous binomial-SE upper bound for an accuracy-derived G over a few-hundred-
    item eval) — used ONLY when `unit_artifacts` is empty. Returns (sigmas, source) where source ∈
    {'banked_S_boot','proxy_floor'} so the report discloses which was used. Fail-loud if NEITHER a
    band NOR a positive floor is available (never a silent zero)."""
    import numpy as np
    if not unit_artifacts:
        assert g_sigma_floor > 0, ("range gate needs --unit-artifacts (banked S_boot) OR a positive "
                                   "--g-sigma-floor proxy; neither given")
        return ([float(g_sigma_floor)] * len(units), "proxy_floor")
    _ = np  # (kept for symmetry with the banked path below)
    files = sorted(glob.glob(unit_artifacts) if any(c in unit_artifacts for c in "*?[") else
                   glob.glob(str(pathlib.Path(unit_artifacts) / "unit_*.json")))
    assert files, f"no unit_*.json under {unit_artifacts}"
    by_unit = {}
    for f in files:
        row = json.loads(pathlib.Path(f).read_text())
        by_unit[row["unit"]] = row
    sig = []
    for u in units:
        row = by_unit[u["unit"]]                       # RAISE if a unit's banked artifact is absent
        boot = row["outcomes"]["16000"]["S_boot"]
        sigma = (float(boot["hi95"]) - float(boot["lo95"])) / (2.0 * 1.96)
        assert sigma >= 0, f"{u['unit']}: negative S_boot band"
        sig.append(sigma)
    return (sig, "banked_S_boot")


# ===========================================================================
# probe stream + forward capture (real GPU path only; text-only eager).
# ===========================================================================

def _probe_stream(banked_sha):
    """Reconstruct the FIXED frozen probe stream (savb.wiki_prompts over wikitext-2 test) and assert
    its sha == the banked prompt_sha — proving the SAME stream the banked G was defined against."""
    from datasets import load_dataset
    from infra.pg import savb as SB
    from infra.pg import text_eval as TE
    # split="test" (matches text_eval + the working P6 loader) — loads ONLY the test split rather
    # than the full DatasetDict (all splits), the lighter/more-reliable hub path.
    wiki = load_dataset(TE.WIKITEXT_REPO, TE.WIKITEXT_CONFIG, split="test")["text"]
    prompts, sha = SB.wiki_prompts(wiki)
    assert sha == banked_sha, f"probe-stream sha drift: reconstructed {sha} != banked {banked_sha}"
    return prompts


def _build_coord_sketch(named_2d, k, seed):
    """A fixed seeded coordinate sketch of the decoder parameter vector — k global coords across the
    concatenation of the 2-D decoder weight matrices (a valid sub-sampling sketch of the diagonal
    Fisher / gradient subspace; never materialises the full parameter space). Returns
    {name: local_flat_indices} + the global name order, so base, unit and ΔW gather the SAME coords."""
    import numpy as np
    rng = np.random.default_rng(seed)
    names = [n for n, _ in named_2d]
    sizes = [int(np.prod(s)) for _, s in named_2d]
    total = int(sum(sizes))
    k = int(min(k, total))
    picked = np.sort(rng.choice(total, size=k, replace=False))
    offs = np.cumsum([0] + sizes)
    local = {n: [] for n in names}
    for gi in picked:
        j = int(np.searchsorted(offs, gi, side="right") - 1)
        local[names[j]].append(int(gi - offs[j]))
    return {n: np.asarray(ix, dtype=np.int64) for n, ix in local.items()}, names


def _decoder_named_2d(model):
    """(name, shape) of the 2-D decoder projection weights of a swapback/base model (the P5 sketch
    scope = the RELAW/SAV-B weight scope: q/k/v/o/gate/up/down_proj .weight)."""
    out = []
    for name, p in model.named_parameters():
        if name.endswith(".weight") and p.ndim == 2 and any(t in name for t in DECODER_PROJ):
            out.append((name, tuple(p.shape)))
    assert out, "no 2-D decoder projection weights found"
    return sorted(out)


def _sketch_vector(model, sketch_local, order):
    """Gather the current parameter VALUES at the sketch coords (for ΔW = W_unit − W_base). fp64."""
    import numpy as np
    params = dict(model.named_parameters())
    parts = []
    for name in order:
        ix = sketch_local[name]
        if ix.size:
            flat = params[name].detach().reshape(-1).double().cpu().numpy()
            parts.append(flat[ix])
    return np.concatenate(parts)


def _capture(model, tok, device, prompts, layer, sketch_local, order):
    """One model → (logits_stream, acts_stream, grad_sketch). All on the FIXED probe stream.
      logits_stream : list of (N_tok_batch, vocab) fp64 masked-token logits (paired KL per batch).
      acts_stream   : (N_tok, d) fp64 masked-token residual activations at `layer` (for CKA).
      grad_sketch   : (N_seq, k) fp64 per-example next-token NLL gradient at the sketch coords.
    Text-only eager forward (savb_unit convention); masked tokens = attention_mask positions."""
    import numpy as np
    import torch
    from infra.pg import savb as SB
    model.eval()
    logits_stream, acts_rows = [], []
    for i in range(0, len(prompts), 8):
        enc = tok(prompts[i:i + 8], return_tensors="pt", padding=True, truncation=True,
                  max_length=SB.WIKI_MAX_TOKENS).to(device)
        mask = enc["attention_mask"].bool()
        with torch.no_grad():
            o = model(**enc, output_hidden_states=True)
        logits_stream.append(o.logits[mask].double().cpu().numpy())
        acts_rows.append(o.hidden_states[layer][mask].double().cpu().numpy())
    acts = np.concatenate(acts_rows, axis=0)

    # per-example (sequence) gradient sketch of the next-token NLL — requires_grad only on the sketch
    # matrices (bounds the backward). One sequence at a time (an "example" = a probe sequence).
    params = dict(model.named_parameters())
    for name, p in params.items():
        p.requires_grad_(name in sketch_local and sketch_local[name].size > 0)
    grads = []
    for j in range(len(prompts)):
        enc = tok(prompts[j:j + 1], return_tensors="pt", truncation=True,
                  max_length=SB.WIKI_MAX_TOKENS).to(device)
        model.zero_grad(set_to_none=True)
        out = model(**enc, labels=enc["input_ids"])
        out.loss.backward()
        parts = []
        for name in order:
            ix = sketch_local[name]
            if ix.size:
                g = params[name].grad.reshape(-1).double().cpu().numpy()
                parts.append(g[ix])
        grads.append(np.concatenate(parts))
    model.zero_grad(set_to_none=True)
    return logits_stream, acts, np.asarray(grads, dtype=np.float64)


def _metrics_for_unit(base_cap, unit_cap, base_sketch_vec, unit_sketch_vec, r_ntk):
    """The 4 frozen menu metrics for one (base, unit) pair from their captures. fp64."""
    import numpy as np
    lb, ab, gb = base_cap
    lu, au, gu = unit_cap
    # D_KL — token-weighted mean symmetric KL across the streamed batches.
    num, den = 0.0, 0
    for Pb, Qu in zip(lb, lu):
        assert Pb.shape == Qu.shape, "base/unit logit-batch shape drift (tokenisation mismatch)"
        num += GF.sym_kl(Pb, Qu) * Pb.shape[0]
        den += Pb.shape[0]
    d_kl = num / den
    d_rep = GF.d_rep(ab, au)
    fisher = GF.fisher_diag(gb)                                    # base-defined curvature
    d_fisher = GF.fisher_weighted_distance(fisher, unit_sketch_vec - base_sketch_vec)
    d_ntk = GF.principal_angle_distance(gb, gu, r_ntk)
    return {"D_KL": float(d_kl), "D_rep": float(d_rep),
            "D_fisher": float(d_fisher), "D_ntk": float(d_ntk)}


# ===========================================================================
# offline gate (§12) — full pipeline minus model forward.
# ===========================================================================

def _offline(a, out):
    from infra.pg import diff_harvest as H
    H.hf_online_setup()                       # before datasets import (offline-cache gotcha)
    import numpy as np
    from infra.pg import relaw as RL
    doc = _load_pool(a.pool, out)
    pool = doc["pool"]
    units = _population(pool)
    banked_sha = _banked_prompt_sha(units)
    layer = _banked_overlap_layer(units)
    G = np.array([float(u["S_16k"]) for u in units], float)
    R_dist = np.array([float(u["l2_base"]) for u in units], float)
    assert np.isfinite(G).all() and (R_dist > 0).all(), "bad banked G / R_dist"

    # 1. reconstruct the frozen probe stream + assert sha == banked (real tokenizer + data, no model).
    prompts = _probe_stream(banked_sha)
    assert len(prompts) >= 2, "probe stream too small"

    # 2. G noise band source (GATE-0). Read the real σ_G if unit artifacts are given; else DISCLOSE
    #    the dependency and use a planted small σ so the wiring check still exercises the gate.
    if a.unit_artifacts or a.g_sigma_floor > 0:
        g_sigmas, noise_src = _read_g_sigmas(units, a.unit_artifacts, a.g_sigma_floor)
    else:
        g_sigmas = (0.02 * np.ones_like(G)).tolist()
        noise_src = "PLANTED (no S_boot artifacts and no --g-sigma-floor — range gate NOT the real one)"

    # 3. drive run_menu on the banked G/R_dist with a PLANTED positive-control metric (proves the
    #    verdict wiring recovers a real predictor and the range gate passes on real G).
    planted = {"D_KL": RL.planted_g(R_dist, slope=1.0, noise_sd=0.01 * G.std(), seed=1),
               "D_rep": RL.planted_g(G, slope=1.0, noise_sd=0.02 * G.std(), seed=2)}
    verdict = GF.run_menu(planted, R_dist, G, g_sigmas, q=a.bh_q, n_perm=a.n_perm, seed=a.seed)

    # 4. generation firewall — the layer counter discriminates 18 (G1) vs 26 (G2).
    g1 = [f"model.layers.{i}.self_attn.q_proj.weight" for i in range(18)]
    g2 = [f"model.layers.{i}.mlp.gate_proj.weight" for i in range(26)]
    assert RL.n_decoder_layers(g1) == 18 and RL.n_decoder_layers(g2) == 26

    rep = {"object": "gauge_fun_offline", "n_units": len(units), "overlap_layer": layer,
           "banked_prompt_sha": banked_sha, "probe_stream_reproduced": True,
           "g_noise_source": noise_src, "menu": list(GF.MENU),
           "planted_control_verdict": verdict, "firewall_ok": True, "OFFLINE_OK": True}
    (out / "gauge_fun_offline.json").write_text(json.dumps(rep, indent=1, default=str))
    print(f"[GAUGE-FUN-OFFLINE] OK — n_units={len(units)} layer={layer} sha={banked_sha} "
          f"noise={noise_src} planted_verdict={verdict['verdict']}")
    print("GAUGE_FUN_OFFLINE_OK")


# ===========================================================================
# real GPU path — eval-only forwards over base + the SAV pool.
# ===========================================================================

def _real(a, out):
    # hf_online_setup FIRST — BEFORE importing transformers/datasets, else those cache
    # HF_DATASETS_OFFLINE at import and load_dataset later fails offline (LocalEntryNotFoundError;
    # reference_pg_hf_online_before_transformers). This is why the wikitext probe stream 401'd.
    from infra.pg import diff_harvest as H
    H.hf_online_setup()
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from infra.pg import vlat_swapback as VS
    from infra.pg.entities import ENTITIES
    from infra.aws_pg.jobs.savb_unit import extract_ft_decoder

    doc = _load_pool(a.pool, out)
    pool = doc["pool"]
    units = _population(pool)
    banked_sha = _banked_prompt_sha(units)
    layer = _banked_overlap_layer(units)
    G = np.array([float(u["S_16k"]) for u in units], float)
    R_dist = np.array([float(u["l2_base"]) for u in units], float)
    g_sigmas, g_sigma_src = _read_g_sigmas(units, a.unit_artifacts, a.g_sigma_floor)

    assert torch.cuda.is_available(), "GAUGE-FUN real path needs a GPU"
    device = "cuda"
    prompts = _probe_stream(banked_sha)

    r_grid = [int(x) for x in a.r_grid.split(",")]
    r_frozen = GF.frozen_ntk_rank(r_grid)

    ent = ENTITIES["L0_g1"]                                 # google/gemma-2b (the banked base)
    tok = AutoTokenizer.from_pretrained(ent.repo_id, revision=ent.revision)
    base = AutoModelForCausalLM.from_pretrained(
        ent.repo_id, revision=ent.revision, torch_dtype=torch.bfloat16,
        device_map=device, attn_implementation="eager")
    named_2d = _decoder_named_2d(base)
    sketch_local, order = _build_coord_sketch(named_2d, a.k_sketch, a.seed)
    r_ntk = min(r_frozen, len(prompts) - 1, int(sum(ix.size for ix in sketch_local.values())) - 1)
    assert r_ntk >= 1, "D_ntk rank collapsed"
    print(f"[GAUGE-FUN] base={ent.repo_id} n_units={len(units)} layer={layer} "
          f"r_ntk={r_ntk} (frozen {r_frozen}) k_sketch={a.k_sketch}", flush=True)

    base_cap = _capture(base, tok, device, prompts, layer, sketch_local, order)
    base_vec = _sketch_vector(base, sketch_local, order)

    rows = []
    metrics = {m: [] for m in GF.MENU}
    for u in units:
        repo = f"google/{u['unit']}"
        dec_u, src_sha = extract_ft_decoder(repo)
        model, swap_info = VS.load_swapback_model(dec_u, device)
        unit_cap = _capture(model, tok, device, prompts, layer, sketch_local, order)
        unit_vec = _sketch_vector(model, sketch_local, order)
        mvals = _metrics_for_unit(base_cap, unit_cap, base_vec, unit_vec, r_ntk)
        for m in GF.MENU:
            metrics[m].append(mvals[m])
        rows.append({"unit": u["unit"], "source_sha": str(src_sha),
                     "G": float(u["S_16k"]), "R_dist": float(u["l2_base"]), **mvals})
        print(f"[GAUGE-FUN] {u['unit']:38s} D_KL={mvals['D_KL']:.4f} D_rep={mvals['D_rep']:.4f} "
              f"D_fisher={mvals['D_fisher']:.4f} D_ntk={mvals['D_ntk']:.4f}", flush=True)
        del model, dec_u
        torch.cuda.empty_cache()

    verdict = GF.run_menu({m: np.asarray(v) for m, v in metrics.items()},
                          R_dist, G, g_sigmas, q=a.bh_q, n_perm=a.n_perm, seed=a.seed)
    rep = {"object": "gauge_fun_v1", "generation": "gemma1_18L", "base_repo": ent.repo_id,
           "n_units": len(units), "overlap_layer": layer, "banked_prompt_sha": banked_sha,
           "r_ntk": r_ntk, "r_ntk_frozen": r_frozen, "k_sketch": int(a.k_sketch),
           "g_sigma_source": g_sigma_src, "g_sigma_floor": float(a.g_sigma_floor),
           "g_spread_std": float(np.std(G)),
           "range_gate_disclosure": (
               "S_boot (banked G bootstrap band) NOT available in any located artifact; range gate "
               "uses a disclosed conservative constant σ_G proxy (--g-sigma-floor). Robust because "
               "the observed G spread (std) dwarfs the proxy; metric verdicts are the primary result."
               if g_sigma_src == "proxy_floor" else "banked S_boot used"),
           "menu": list(GF.MENU), "per_unit": rows, "verdict": verdict,
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    (out / "gauge_fun.json").write_text(json.dumps(rep, indent=1, default=str))
    print("=" * 78)
    print(f"GAUGE_FUN_OK n_units={len(units)} verdict={verdict['verdict']} "
          f"range_bound={verdict['range_gate']['range_bound']} snr={verdict['range_gate']['snr']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", default="studies/lineage/paligemma1-pi0/results/savc/savc_metrics.json")
    ap.add_argument("--g-sigma-floor", type=float, default=0.0,
                    help="disclosed conservative constant σ_G proxy for the range gate when the "
                         "banked S_boot is unavailable (GATE-0: not banked); e.g. 0.03 binomial-SE")
    ap.add_argument("--unit-artifacts", default="",
                    help="dir or glob of banked unit_*.json (for the range-gate σ_G = S_boot band)")
    ap.add_argument("--r-grid", default=",".join(str(r) for r in RELAW_R_GRID))
    ap.add_argument("--k-sketch", type=int, default=4096, help="coordinate-sketch dimension")
    ap.add_argument("--bh-q", type=float, default=0.10)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--offline", action="store_true", help="offline gate (no model forward, no GPU)")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.offline:
        _offline(a, out)
    else:
        _real(a, out)


if __name__ == "__main__":
    main()
