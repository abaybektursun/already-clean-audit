"""conserved-core Tier 1 — NAME what U_unt computes (position / sink / induction / other).

Spec: studies/lineage/paligemma1-pi0/research/2026-06-19_conserved_core_prereg.md (§3.1 MDE).
Builds on invariant_core-FUNC: U_unt(L,k) = bottom-k eigvecs of the q_proj consensus Gram over the 24
PG1 fine-tunes (FUNC found it LIVE, +49 ppl ablated, q/k untouched 18/18). THIS job asks WHICH function
that conserved q/k input subspace carries, by ABLATING it (FUNC's x'=x-QQᵀx on q+k, every layer) and
reading three function-probes + two synthetic tasks on the SAME forward.

THE ENERGY CONFOUND (the load-bearing control, added after adversarial review). FUNC: U_unt removes
~28% of the q/k-input activation energy at k=128 vs a random k-subspace's ~6%. So ablating U_unt strips
~4-5x the variance a baseline-trained probe reads — for EVERY probe, generically. A bare random null is
NOT a fair control. The PRIMARY null here is an ENERGY-MATCHED random subspace (same removed-energy
fraction as U_unt, built from the activation-Gram eigenbasis, NOT aligned to any function). A function X
is NAMED only if U_unt degrades probe X past the ENERGY-MATCHED null (not merely past a low-energy
random/bottom-σ null), with a discovery-calibrated θ_X, monotone dose-response over k, the matching
synthetic task collapsing, BH-FDR control, and single-dominance over the other probes.

FAIL LOUD: no try/except, no synthetic fallback. Reuses the FUNC harness VERBATIM (imported).

  # smoke (wiring + probe construct-validity; tiny ft set => U_unt meaningless, plumbing only):
  pg run infra.aws_pg.jobs.conserved_core --args="--smoke" --no-weights \
      --type g6e.8xlarge,p5.4xlarge --s3-prefix results/conserved_core_smoke --budget-min 90
  # full (24 ft, the MDE):
  pg run infra.aws_pg.jobs.conserved_core --args="--k-grid 8,32,128 --n-ppl-seq 120 --n-synth 256" \
      --no-weights --type g6e.8xlarge,p5.4xlarge --s3-prefix results/conserved_core --budget-min 300
"""
import argparse
import json
import math
import pathlib

# Reuse the FUNC harness VERBATIM (imported) — §13 reuse ledger.
from infra.aws_pg.jobs.invariant_core import PG1_REPO, N_LAYERS, _load_decoder
from infra.aws_pg.jobs.invariant_core_func import (
    _consensus_gram_qproj, _bottom_eigvecs, _base_top_sigma, _base_bottom_sigma,
    _random_subspace, _load_full_pg1, _decoder_layers, _InputProject, _InputCapture,
    _blank_image, _locate_text_block,
)
from infra.pg import conserved_core_probes as P

FUNC_BASE_PPL = 22.4258          # FUNC anchor (recomputed in-run; wiring gate)
FUNC_UNT_DPPL_K128 = 49.14       # FUNC anchor: U_unt k=128 Δppl (replication gate, ±10)
N_RAND_DRAWS = 2                 # low-energy random null (context)
N_EM_DRAWS = 3                   # energy-matched null draws (ematch[0]=headline null; pairs calibrate θ)
SINK_FLOOR = 0.05                # §8: baseline BOS-sink mass below this => H_sink low-power


# ===========================================================================
# Per-token q/k INPUT capture (post any active _InputProject ablation) at the probe layers, on the
# ABLATED side (q_proj for q+k / q-only; k_proj for k-only — at baseline q_proj input == k_proj input,
# so the baseline-trained probe transfers). Ordering: enter _InputProject FIRST so its hook runs before
# this capture. One window per forward (b=1).
# ===========================================================================
class _TokenCapture:
    def __init__(self, layers, probe_layers, capture_proj="q_proj"):
        self.layers = layers
        self.probe_layers = probe_layers
        self.capture_proj = capture_proj
        self.buf = {L: None for L in probe_layers}
        self._handles = []

    def _mk(self, L):
        def pre_hook(_m, args):
            x = args[0]                                   # (1, T, n) projection input = post-LN residual
            assert x.shape[0] == 1, "token capture assumes one window per forward (b=1)"
            self.buf[L] = x.detach()[0].float().cpu().numpy()    # (T, n) fp32
        return pre_hook

    def __enter__(self):
        for L in self.probe_layers:
            proj = getattr(self.layers[L].self_attn, self.capture_proj)
            self._handles.append(proj.register_forward_pre_hook(self._mk(L)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


# Sided q/k input projection (k-only / q-only sensitivity arm). q+k headline uses FUNC's _InputProject.
class _InputProjectSided:
    def __init__(self, layers, Q_by_layer, device, dtype, sides):
        import torch
        assert sides in ("q", "k"), sides
        self.layers = layers
        self.sides = sides
        self.Q = {L: torch.as_tensor(Q_by_layer[L], device=device, dtype=dtype) for L in Q_by_layer}
        self._handles = []

    def _mk(self, L):
        Q = self.Q[L]
        def pre_hook(_m, args):
            x = args[0]
            return (x - (x @ Q) @ Q.transpose(0, 1),) + tuple(args[1:])
        return pre_hook

    def __enter__(self):
        for L in self.Q:
            self._handles.append(getattr(self.layers[L].self_attn, f"{self.sides}_proj")
                                 .register_forward_pre_hook(self._mk(L)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


# ===========================================================================
# Wikitext windows — mirror _ppl_blankimg's windowing EXACTLY so the ppl gate (base≈22.43, U_unt k128
# Δppl≈+49.1 over the SAME full window set) validates THIS forward path.
# ===========================================================================
def _wikitext_windows(processor, n_seq, window):
    from datasets import load_dataset
    tok = processor.tokenizer
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    windows = []
    for start in range(0, ids.size(0), window):
        if n_seq is not None and len(windows) >= n_seq:
            break
        chunk_ids = ids[start:start + window]
        if chunk_ids.size(0) < 8:
            break
        window_text = tok.decode(chunk_ids.tolist())
        win_ids = tok(window_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if int(win_ids.shape[0]) < 8:
            continue
        windows.append((window_text, win_ids))
    assert windows, "no wikitext windows built"
    return windows


def _probe_forward(model, processor, device, layers, windows, probe_layers,
                   capture_proj="q_proj", gray=128):
    """ONE forward per window capturing, together: per-window ppl [nll_sum,n_tok] (the gate readout),
    per-probe-layer text-region q/k INPUT features (n_text x n), per-probe-layer BOS-sink attention mass
    (mean attention from text query positions to BOS = position pos-1) + image-prefix mass. The
    loss/window code mirrors _ppl_blankimg verbatim."""
    import torch
    import numpy as np
    tok = processor.tokenizer
    blank = _blank_image(gray)
    out = []
    for window_text, win_ids in windows:
        n_text = int(win_ids.shape[0])
        inp = processor(text=[window_text], images=[blank], return_tensors="pt", padding=False).to(device)
        in_ids = inp["input_ids"][0]
        assert in_ids.shape[0] > n_text, "text span >= prompt — image prefix missing"
        pos = _locate_text_block(in_ids, win_ids)
        assert pos >= 1, "no BOS before text block (sink index undefined)"
        bos_idx = pos - 1                                 # PaliGemma: [256 image][BOS][text]; BOS = pos-1
        cap = _TokenCapture(layers, probe_layers, capture_proj)
        with cap, torch.inference_mode():
            mo = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                       pixel_values=inp["pixel_values"].to(torch.bfloat16), output_attentions=True)
        logits = mo.logits[0]
        text_logits = logits[pos:pos + n_text - 1, :].float()
        targets = win_ids[1:].to(device)
        assert text_logits.shape[0] == targets.shape[0], (text_logits.shape, targets.shape)
        loss = torch.nn.functional.cross_entropy(text_logits, targets, reduction="sum")
        feats, sink, imgmass = {}, {}, {}
        attns = mo.attentions
        assert attns is not None and len(attns) == N_LAYERS, \
            f"output_attentions returned {None if attns is None else len(attns)} layers (want {N_LAYERS})"
        for L in probe_layers:
            buf = cap.buf[L]
            assert buf is not None, f"no capture at L{L}"
            feats[L] = buf[pos:pos + n_text, :].astype(np.float32)
            a = attns[L][0]                                               # (heads, T, T) softmax weights
            qslice = a[:, pos:pos + n_text, :].float()                    # text query rows
            sink[L] = float(qslice[:, :, bos_idx].mean().item())         # mass to BOS (the sink)
            imgmass[L] = float(qslice[:, :, 0:bos_idx].mean().item())    # mass to image prefix (context)
        out.append({"text": window_text, "is_conf": P.is_confirmation(window_text),
                    "n_text": n_text, "ppl_pw": [float(loss.item()), int(targets.numel())],
                    "pos_labels": P.position_labels(n_text),
                    "indu_labels": P.induction_labels(win_ids.cpu().numpy()),
                    "feats": feats, "sink": sink, "imgmass": imgmass})
    return out


def _ppl_from_pw(per_window):
    import numpy as np
    a = np.asarray([w["ppl_pw"] for w in per_window], dtype=np.float64)
    return math.exp(a[:, 0].sum() / a[:, 1].sum())


def _activation_grams(model, processor, device, layers, windows, want_layers, gray=128):
    """One blank-image forward over the windows capturing the q_proj INPUT Gram per layer (FUNC
    _InputCapture). Returns {L: (n×n) np.float64} — the activation second-moment used to (a) measure
    U_unt's removed-energy fraction and (b) build the energy-matched null."""
    import torch
    tok = processor.tokenizer
    blank = _blank_image(gray)
    cap = _InputCapture(layers)
    with cap:
        for window_text, win_ids in windows:
            inp = processor(text=[window_text], images=[blank], return_tensors="pt", padding=False).to(device)
            with torch.inference_mode():
                model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                      pixel_values=inp["pixel_values"].to(torch.bfloat16))
    return {L: cap.gram[L] for L in want_layers}


def _synth_acc(model, processor, device, items, gray=128):
    """Accuracy = fraction where next-token argmax (decoded, stripped) == gold. Caller wraps in the arm's
    projection context. Per-item 0/1 banked for a paired item bootstrap."""
    import torch
    import numpy as np
    tok = processor.tokenizer
    blank = _blank_image(gray)
    per_item = []
    for prompt, gold in items:
        win_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
        n_text = int(win_ids.shape[0])
        inp = processor(text=[prompt], images=[blank], return_tensors="pt", padding=False).to(device)
        in_ids = inp["input_ids"][0]
        pos = _locate_text_block(in_ids, win_ids)
        with torch.inference_mode():
            logits = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                           pixel_values=inp["pixel_values"].to(torch.bfloat16)).logits[0]
        pred = tok.decode([int(logits[pos + n_text - 1].float().argmax().item())]).strip()
        per_item.append(1 if pred == gold else 0)
    return float(np.mean(per_item)), per_item


def _build_u_unt(api, k_grid, want_layers, ft_tasks):
    """Rebuild U_unt(L,k) = bottom-k eigvecs of the q_proj consensus Gram over the 24 PG1 fine-tunes
    (NOT cached). Reuses FUNC's _consensus_gram_qproj verbatim."""
    import numpy as np
    if ft_tasks.strip():
        repos = [f"google/paligemma-3b-ft-{t.strip()}-224" for t in ft_tasks.split(",") if t.strip()]
    else:
        repos = sorted({m.id for m in api.list_models(author="google", search="paligemma-3b-ft")
                        if (m.id.split("/")[-1].endswith("-224") and "-jax" not in m.id
                            and m.id.split("/")[-1].startswith("paligemma-3b-ft-"))})
    shas = {r: api.model_info(r).sha for r in repos}
    base_sha = api.model_info(PG1_REPO).sha
    w_base = _load_decoder(PG1_REPO, base_sha)
    G_q = _consensus_gram_qproj(repos, shas, w_base, want_layers)
    del w_base
    U_unt = {}
    for L in want_layers:
        for k in k_grid:
            Uk, _ = _bottom_eigvecs(G_q[L], k)
            assert abs((Uk.T @ Uk) - np.eye(k)).max() < 1e-8, f"U_unt not orthonormal L{L} k{k}"
            U_unt[(L, k)] = Uk
    return U_unt, repos, shas, base_sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ft-tasks", default="", help="comma slugs; empty => all -224 (the 24-ft cloud)")
    ap.add_argument("--layers", default="", help="ablation layers; empty => all 18")
    ap.add_argument("--probe-layers", default="4,9,13", help="layers to capture probe features at")
    ap.add_argument("--headline-layer", type=int, default=9, help="frozen headline probe layer (§ pre-reg)")
    ap.add_argument("--k-grid", default="8,32,128")
    ap.add_argument("--n-ppl-seq", type=int, default=120, help="#wikitext windows (probe + ppl)")
    ap.add_argument("--ppl-window", type=int, default=512)
    ap.add_argument("--n-synth", type=int, default=256, help="#items per synthetic task")
    ap.add_argument("--ridge-lam", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    k_grid = [int(x) for x in a.k_grid.split(",")]
    headline_k = max(k_grid)
    headline_layer = a.headline_layer
    if a.smoke:
        k_grid = [8, 32]
        headline_k = 32
        a.n_ppl_seq = 24
        a.n_synth = 48
        want_layers = [0, 9, 17]
        probe_layers = [9]
        headline_layer = 9
        if not a.ft_tasks.strip():
            a.ft_tasks = "vqav2,gqa,ai2d"
    else:
        want_layers = ([int(x) for x in a.layers.split(",") if x.strip()]
                       if a.layers.strip() else list(range(N_LAYERS)))
        probe_layers = [int(x) for x in a.probe_layers.split(",") if x.strip()]
    assert all(L in want_layers for L in probe_layers), "probe layers must be among ablated layers"
    assert headline_layer in probe_layers, "headline layer must be a probe layer"

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()                       # FIRST — gated repos 401 otherwise
    import numpy as np
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoProcessor
    assert torch.cuda.is_available(), "GPU required for the pt-224 forward"
    device = "cuda"
    api = HfApi()
    n = 2048

    rep = {"object": "conserved_core_tier1_v2_energymatched", "base_repo": PG1_REPO, "k_grid": k_grid,
           "headline_k": headline_k, "headline_layer": headline_layer, "layers": want_layers,
           "probe_layers": probe_layers, "n_ppl_seq": a.n_ppl_seq, "n_synth": a.n_synth,
           "ridge_lam": a.ridge_lam, "seed": a.seed, "smoke": a.smoke,
           "func_anchors": {"base_ppl": FUNC_BASE_PPL, "unt_dppl_k128": FUNC_UNT_DPPL_K128}}

    print("[cc] building U_unt (rebuild from consensus Gram over the ft cloud)...", flush=True)
    U_unt, repos, shas, base_sha = _build_u_unt(api, k_grid, want_layers, a.ft_tasks)
    rep.update({"base_sha": base_sha, "cloud_repos": repos, "n_ft": len(repos)})
    print(f"[cc] U_unt built: {len(repos)} ft, layers={len(want_layers)}, k={k_grid}", flush=True)

    processor = AutoProcessor.from_pretrained(PG1_REPO, revision=base_sha)
    model = _load_full_pg1(base_sha, device)
    layers = _decoder_layers(model)
    dtype = next(model.parameters()).dtype
    windows = _wikitext_windows(processor, a.n_ppl_seq, a.ppl_window)
    print(f"[cc] {len(windows)} wikitext windows", flush=True)

    # ---- activation Gram per layer -> U_unt removed-energy fraction + energy-matched nulls.
    print("[cc] activation Gram (energy-matched-null construction)...", flush=True)
    G_act = _activation_grams(model, processor, device, layers, windows, want_layers)
    e_unt = {}                          # e_unt[(L,k)] = U_unt's removed-energy fraction
    Q_em = {}                           # Q_em[(L,k,d)] = energy-matched null draw d
    e_em = {}
    for L in want_layers:
        w_ev, V_ev = np.linalg.eigh(G_act[L])           # ascending
        V_desc = V_ev[:, ::-1]
        mu_desc = w_ev[::-1]
        tot = float(mu_desc.sum())
        for k in k_grid:
            Uk = U_unt[(L, k)]
            e_unt[(L, k)] = float(np.trace(Uk.T @ G_act[L] @ Uk)) / tot
            for d in range(N_EM_DRAWS):
                Q, ee = P.energy_matched_subspace(V_desc, mu_desc, e_unt[(L, k)], k,
                                                  seed=a.seed * 7919 + 1000 * L + 100 * k + d)
                Q_em[(L, k, d)] = Q
                e_em[(L, k, d)] = ee
    rep["energy"] = {f"{L},{k}": {"e_unt": e_unt[(L, k)],
                                  "e_em": [e_em[(L, k, d)] for d in range(N_EM_DRAWS)]}
                     for L in probe_layers for k in k_grid}
    eu = e_unt[(headline_layer, headline_k)]
    print(f"[cc] L{headline_layer} k{headline_k}: e_unt={eu:.3f} "
          f"e_em={[round(e_em[(headline_layer, headline_k, d)], 3) for d in range(N_EM_DRAWS)]}", flush=True)

    def _Q(kind, k):
        if kind == "unt":
            return {L: U_unt[(L, k)] for L in want_layers}
        if kind == "bots":
            return {L: _base_bottom_sigma(layers[L].self_attn.q_proj.weight.detach().float().cpu(), k)
                    for L in want_layers}
        if kind == "tops":
            return {L: _base_top_sigma(layers[L].self_attn.q_proj.weight.detach().float().cpu(), k)
                    for L in want_layers}
        if kind.startswith("rand:"):
            d = int(kind.split(":")[1])
            return {L: _random_subspace(n, k, a.seed * 104729 + 1000 * d + L) for L in want_layers}
        if kind.startswith("em:"):
            d = int(kind.split(":")[1])
            return {L: Q_em[(L, k, d)] for L in want_layers}
        raise AssertionError(kind)

    # ---- PRE: projector idempotency + base ppl over ALL windows (wiring gate, FUNC population).
    Qt = torch.as_tensor(U_unt[(probe_layers[0], k_grid[0])], device=device, dtype=torch.float64)
    xr = torch.randn(1, 5, n, device=device, dtype=torch.float64)
    x1 = xr - (xr @ Qt) @ Qt.transpose(0, 1)
    proj_resid = float(((x1 - (x1 @ Qt) @ Qt.transpose(0, 1)) - x1).abs().max())
    assert proj_resid < 1e-9 and float((x1 @ Qt).abs().max()) < 1e-9, f"projector not idempotent {proj_resid}"

    base_pw = _probe_forward(model, processor, device, layers, windows, probe_layers)
    base_ppl = _ppl_from_pw(base_pw)                     # ALL windows (FUNC population, wiring gate)
    assert a.smoke or abs(base_ppl - FUNC_BASE_PPL) <= 1.0, \
        f"WIRING GATE FAIL: base ppl {base_ppl} vs FUNC {FUNC_BASE_PPL}"
    print(f"[cc][PRE] base ppl={base_ppl:.4f} (FUNC {FUNC_BASE_PPL})", flush=True)

    disc = [w for w in base_pw if not w["is_conf"]]
    conf = [w for w in base_pw if w["is_conf"]]
    assert disc and conf, f"degenerate split: {len(disc)} disc / {len(conf)} conf"

    # train probes ONCE on baseline DISCOVERY tokens; construct validity + sink floor on baseline.
    probes, base_bands = {}, {}
    for L in probe_layers:
        Xd = np.concatenate([w["feats"][L] for w in disc])
        yposd = np.concatenate([w["pos_labels"] for w in disc])
        yindd = np.concatenate([w["indu_labels"] for w in disc])
        probes[L] = {"pos": P.ridge_fit(Xd, yposd, lam=a.ridge_lam),
                     "indu": P.lsclf_fit(Xd, yindd, lam=a.ridge_lam)}
        Xc = np.concatenate([w["feats"][L] for w in conf])
        ypc = np.concatenate([w["pos_labels"] for w in conf])
        yic = np.concatenate([w["indu_labels"] for w in conf])
        ppos = P.linear_predict(probes[L]["pos"], Xc)
        ppos_shuf = P.linear_predict(P.ridge_fit(Xd, P.shuffle_labels(yposd, a.seed), lam=a.ridge_lam), Xc)
        r2_clean, corr_clean = P.r2_score(ypc, ppos), P.corr_score(ypc, ppos)
        r2_shuf, corr_shuf = P.r2_score(ypc, ppos_shuf), P.corr_score(ypc, ppos_shuf)
        f1_clean = P.f1_at_zero(P.linear_predict(probes[L]["indu"], Xc), yic)
        f1_shuf = P.f1_at_zero(P.linear_predict(P.lsclf_fit(Xd, P.shuffle_labels(yindd, a.seed), lam=a.ridge_lam), Xc), yic)
        sink_base = float(np.mean([w["sink"][L] for w in conf]))
        base_bands[L] = {"pos_r2_clean": r2_clean, "pos_r2_shuffle": r2_shuf,
                         "pos_corr_clean": corr_clean, "pos_corr_shuffle": corr_shuf,
                         "indu_f1_clean": f1_clean, "indu_f1_shuffle": f1_shuf, "sink_base": sink_base}
        print(f"[cc][PRE] L{L} pos-corr={corr_clean:.3f}(shuf {corr_shuf:.3f}) R²={r2_clean:.3f} "
              f"indu-F1={f1_clean:.3f}(shuf {f1_shuf:.3f}) sink_BOS={sink_base:.4f}", flush=True)
    rep["base_probe_bands"] = base_bands
    rep["base_ppl"] = base_ppl
    (out / "conserved_core.json").write_text(json.dumps(rep, indent=2, default=str))

    # synthetic ceiling (baseline) — dynamic-range gate (§8).
    copy_items = P.copy_corpus(a.n_synth, seed=a.seed)
    posq_items = P.position_corpus(a.n_synth, seed=a.seed + 1)
    base_copy_acc, base_copy_pi = _synth_acc(model, processor, device, copy_items)
    base_posq_acc, base_posq_pi = _synth_acc(model, processor, device, posq_items)
    rep["synthetic_baseline"] = {"copy_acc": base_copy_acc, "posq_acc": base_posq_acc,
                                 "copy_has_range": base_copy_acc >= 0.4, "posq_has_range": base_posq_acc >= 0.4}
    print(f"[cc][PRE] synthetic baseline copy={base_copy_acc:.3f} posq={base_posq_acc:.3f}", flush=True)

    # =======================================================================
    # ABL: per k, arms = U_unt, em x N_EM, rand x N_RAND, bottom-σ, top-σ, over ALL windows (disc => θ,
    # conf => headline). Probe contrast PRIMARY = U_unt vs energy-matched (the H_generic+energy killer).
    # =======================================================================
    def _arm(projkind, k, sides="qk"):
        Qby = _Q(projkind, k)
        cproj = "k_proj" if sides == "k" else "q_proj"
        ctx = (_InputProject(layers, Qby, device, dtype) if sides == "qk"
               else _InputProjectSided(layers, Qby, device, dtype, sides))
        with ctx:
            return _probe_forward(model, processor, device, layers, windows, probe_layers, capture_proj=cproj)

    def _preds(arm_pw, L):
        """Per-window predictions of the baseline-trained probe on the arm's activations + sink, keyed
        by window text (paired)."""
        return {w["text"]: {"is_conf": w["is_conf"], "y_pos": w["pos_labels"],
                            "y_indu": w["indu_labels"].astype(np.float64),
                            "pred_pos": P.linear_predict(probes[L]["pos"], w["feats"][L]),
                            "pred_indu": P.linear_predict(probes[L]["indu"], w["feats"][L]),
                            "sink": np.full(w["n_text"], w["sink"][L], dtype=np.float64)} for w in arm_pw}

    def _blocks(treat_preds, null_preds, which, half):
        """Paired per-window blocks (treat,null,y) over the disc/conf half, aligned by window text."""
        key = {"pos": "pred_pos", "indu": "pred_indu", "sink": "sink"}[which]
        wb = []
        for txt, tp in treat_preds.items():
            if (half == "conf") != tp["is_conf"]:
                continue
            np_ = null_preds[txt]
            wb.append({"y_pos": tp["y_pos"], "y_indu": tp["y_indu"], "treat": tp[key], "null": np_[key]})
        assert wb, f"no {half} windows"
        return wb

    def _mfn(which):
        if which == "pos":
            return lambda pool, key: P.corr_score(pool["y_pos"], pool[key])   # BOUNDED (smoke fix)
        if which == "indu":
            return lambda pool, key: P.f1_at_zero(pool[key], pool["y_indu"])
        return lambda pool, key: float(np.mean(pool[key]))

    def _point(pred_dict, L, which):
        """Point metric on the CONFIRMATION pool for one arm — the raw degradation magnitude (banked so
        the result doc shows per-probe drops side-by-side; an energy-only hit is uniform across probes)."""
        key = {"pos": "pred_pos", "indu": "pred_indu", "sink": "sink"}[which]
        cw = [p for p in pred_dict.values() if p["is_conf"]]
        pool = {"y_pos": np.concatenate([p["y_pos"] for p in cw]),
                "y_indu": np.concatenate([p["y_indu"] for p in cw]),
                key: np.concatenate([p[key] for p in cw])}
        return _mfn(which)(pool, key)

    def _excess(treat_preds, null_preds, which, half):
        return P.probe_contrast_block_ci(_blocks(treat_preds, null_preds, which, half),
                                         _mfn(which), "treat", "null", n_boot=a.n_boot, seed=a.seed)

    base_pred = {L: _preds(base_pw, L) for L in probe_layers}      # baseline metric (raw-magnitude anchor)
    abl = []
    for k in k_grid:
        unt = _arm("unt", k)
        ems = [_arm(f"em:{d}", k) for d in range(N_EM_DRAWS)]
        rands = [_arm(f"rand:{d}", k) for d in range(N_RAND_DRAWS)]
        bots = _arm("bots", k)
        tops = _arm("tops", k)

        # ppl gate over ALL windows (FUNC population) — replication assert at headline_k.
        dppl_unt = _ppl_from_pw(unt) - base_ppl
        dppl_em = [_ppl_from_pw(e) - base_ppl for e in ems]
        dppl_rand = [_ppl_from_pw(r) - base_ppl for r in rands]
        dppl_bots = _ppl_from_pw(bots) - base_ppl
        dppl_tops = _ppl_from_pw(tops) - base_ppl
        poscontrol_valid = bool(dppl_tops > max(dppl_rand + dppl_em))
        if k == headline_k:
            assert a.smoke or abs(dppl_unt - FUNC_UNT_DPPL_K128) <= 10.0, \
                f"PPL-REPLICATION GATE FAIL: U_unt k{k} dppl {dppl_unt:.2f} vs FUNC {FUNC_UNT_DPPL_K128} ±10"

        unt_pred = {L: _preds(unt, L) for L in probe_layers}
        em_pred = [{L: _preds(e, L) for L in probe_layers} for e in ems]
        rand_pred = {L: _preds(rands[0], L) for L in probe_layers}
        bots_pred = {L: _preds(bots, L) for L in probe_layers}

        krec = {"k": k, "dppl_unt": dppl_unt, "dppl_em": dppl_em, "dppl_rand": dppl_rand,
                "dppl_bots": dppl_bots, "dppl_tops": dppl_tops, "poscontrol_valid": poscontrol_valid,
                "e_unt": e_unt[(headline_layer, k)], "probes": {}}
        for L in probe_layers:
            krec["probes"][str(L)] = {}
            for which in ("pos", "indu", "sink"):
                # PRIMARY: U_unt vs energy-matched (conf headline + disc θ).
                ex_conf = _excess(unt_pred[L], em_pred[0][L], which, "conf")
                # θ = 95th pct |ematch-vs-ematch contrast| on disc (the energy-controlled null spread).
                em_null = [_excess(em_pred[i][L], em_pred[0][L], which, "disc")["contrast"]
                           for i in range(1, N_EM_DRAWS)]
                theta = float(np.percentile(np.abs(em_null), 95)) if em_null else 0.0
                # context: vs low-energy random + bottom-σ (NOT energy-matched).
                ex_rand = _excess(unt_pred[L], rand_pred[L], which, "conf")
                ex_bots = _excess(unt_pred[L], bots_pred[L], which, "conf")
                krec["probes"][str(L)][which] = {"excess_vs_em": ex_conf, "theta_em": theta,
                                                 "excess_vs_rand": ex_rand, "excess_vs_bots": ex_bots,
                                                 "em_null_disc": em_null,
                                                 "raw": {"base": _point(base_pred[L], L, which),
                                                         "unt": _point(unt_pred[L], L, which),
                                                         "em": _point(em_pred[0][L], L, which),
                                                         "rand": _point(rand_pred[L], L, which),
                                                         "bots": _point(bots_pred[L], L, which)}}
        abl.append(krec)
        rep["ablation"] = abl
        (out / "conserved_core.json").write_text(json.dumps(rep, indent=2, default=str))
        hp = krec["probes"][str(headline_layer)]
        print(f"[cc][ABL] k={k} dppl U_unt={dppl_unt:+.2f} em~{np.mean(dppl_em):+.2f} "
              f"rand~{np.mean(dppl_rand):+.2f} TOPσ={dppl_tops:+.1f}(valid={poscontrol_valid}) | "
              f"L{headline_layer} pos_ex={hp['pos']['excess_vs_em']['contrast']:+.4f}"
              f"[{hp['pos']['excess_vs_em']['lo']:+.3f},{hp['pos']['excess_vs_em']['hi']:+.3f}]θ{hp['pos']['theta_em']:.3f} "
              f"indu_ex={hp['indu']['excess_vs_em']['contrast']:+.4f} "
              f"sink_ex={hp['sink']['excess_vs_em']['contrast']:+.5f}", flush=True)

    # ---- synthetic under {base, U_unt, em[0], rand[0]} at headline_k + paired-bootstrap contrast vs em.
    synth = {}
    synth_pi = {"base": {"copy": base_copy_pi, "posq": base_posq_pi}}
    for kind in ("unt", "em:0", "rand:0"):
        Qby = _Q(kind, headline_k)
        with _InputProject(layers, Qby, device, dtype):
            c_acc, c_pi = _synth_acc(model, processor, device, copy_items)
            p_acc, p_pi = _synth_acc(model, processor, device, posq_items)
        synth[kind] = {"copy_acc": c_acc, "posq_acc": p_acc}
        synth_pi[kind] = {"copy": c_pi, "posq": p_pi}
    synth["copy_contrast_unt_vs_em"] = P.synth_contrast_bootstrap(
        synth_pi["base"]["copy"], synth_pi["unt"]["copy"], synth_pi["em:0"]["copy"], n_boot=a.n_boot, seed=a.seed)
    synth["posq_contrast_unt_vs_em"] = P.synth_contrast_bootstrap(
        synth_pi["base"]["posq"], synth_pi["unt"]["posq"], synth_pi["em:0"]["posq"], n_boot=a.n_boot, seed=a.seed)
    rep["synthetic"] = synth
    print(f"[cc][SYNTH] copy_ex_vs_em={synth['copy_contrast_unt_vs_em']['contrast']:+.3f}"
          f"[{synth['copy_contrast_unt_vs_em']['lo']:+.3f},{synth['copy_contrast_unt_vs_em']['hi']:+.3f}] "
          f"posq_ex_vs_em={synth['posq_contrast_unt_vs_em']['contrast']:+.3f}"
          f"[{synth['posq_contrast_unt_vs_em']['lo']:+.3f},{synth['posq_contrast_unt_vs_em']['hi']:+.3f}]", flush=True)

    # ---- k-only / q-only sensitivity at headline_k (capture the ablated side).
    sided = {}
    for sides in ("k", "q"):
        unt_s = _arm("unt", headline_k, sides=sides)
        em_s = _arm("em:0", headline_k, sides=sides)
        up = {headline_layer: _preds(unt_s, headline_layer)}
        ep = {headline_layer: _preds(em_s, headline_layer)}
        sided[sides] = {"dppl_unt": _ppl_from_pw(unt_s) - base_ppl,
                        "pos": _excess(up[headline_layer], ep[headline_layer], "pos", "conf"),
                        "indu": _excess(up[headline_layer], ep[headline_layer], "indu", "conf"),
                        "sink": _excess(up[headline_layer], ep[headline_layer], "sink", "conf")}
    rep["sided"] = sided
    print(f"[cc][SIDED] k-only dppl={sided['k']['dppl_unt']:+.2f} q-only dppl={sided['q']['dppl_unt']:+.2f}", flush=True)

    # ---- VERDICT (§6 four-conjunct, energy-matched primary). BH-FDR across {pos,sink,indu}×{wiki,synth}.
    rep["verdict"] = _verdict(abl, headline_k, headline_layer, synth, base_bands[headline_layer],
                              rep["synthetic_baseline"], k_grid)
    (out / "conserved_core.json").write_text(json.dumps(rep, indent=2, default=str))
    print("=" * 78, flush=True)
    print(f"[cc] VERDICT: {rep['verdict']['named']} — {rep['verdict']['basis']}", flush=True)
    print(f"CONSERVED_CORE_OK verdict={rep['verdict']['named']} n_ft={len(repos)} k={k_grid}", flush=True)


def _monotone(xs):
    """True iff non-decreasing in magnitude (dose-response over k)."""
    import numpy as np
    a = np.abs(np.asarray(xs, dtype=np.float64))
    return bool(np.all(np.diff(a) >= -1e-9))


def _verdict(abl, headline_k, L, synth, bands, synth_base, k_grid):
    """§6 four-conjunct naming, energy-matched primary null, two-sided sink, BH-FDR, single-dominant by
    magnitude margin. A function X is NAMED iff (1) excess_vs_em CI excludes 0 [pos/indu one-sided lo>0;
    sink two-sided] AND (2) |excess| > θ_em AND (3) monotone in k AND (4) [pos/indu] the synthetic task
    collapses past em with CI>0 and has dynamic range; construct-validity + §8 floor gates required; BH
    across the family controls multiplicity. H_compound iff >=2 named. H_generic iff none beats em but
    U_unt degrades probes vs baseline (energy effect). H_other iff U_unt live but nothing degrades.
    INSTRUMENT_BROKEN iff poscontrol invalid."""
    import numpy as np
    hk = next(r for r in abl if r["k"] == headline_k)
    pr = hk["probes"][str(L)]
    if not hk["poscontrol_valid"]:
        return {"named": "INSTRUMENT_BROKEN", "basis": "top-σ positive control did not exceed the null band",
                "poscontrol_valid": False}

    # construct validity + §8 floor gates.
    valid = {"pos": (bands["pos_corr_clean"] - bands["pos_corr_shuffle"]) > 0.05,
             "indu": (bands["indu_f1_clean"] - bands["indu_f1_shuffle"]) > 0.05,
             "sink": bands["sink_base"] >= 0.05}                              # §8 sink-floor

    # per-probe conjuncts at headline_k.
    pvals, order = [], ["pos", "indu", "sink"]
    detail, clears = {}, {}
    for w in order:
        c = pr[w]["excess_vs_em"]
        theta = pr[w]["theta_em"]
        ci_excl_0 = (c["lo"] > 0) if w != "sink" else (c["lo"] > 0 or c["hi"] < 0)   # sink two-sided
        past_theta = abs(c["contrast"]) > theta
        ks_excess = [next(r["probes"][str(L)][w]["excess_vs_em"]["contrast"] for r in abl if r["k"] == kk)
                     for kk in k_grid]
        monotone = _monotone(ks_excess)
        # synthetic collapse for pos/indu (the opposite-prediction gate). WAIVED if the task lacks
        # baseline dynamic range (e.g. PG1 cannot do zero-shot symbolic position retrieval, posq≈0 —
        # smoke 2026-06-19); then the naming rests on the energy-matched probe excess with a low-power flag.
        if w in ("pos", "indu"):
            hr = synth_base["posq_has_range" if w == "pos" else "copy_has_range"]
            syn = synth["posq_contrast_unt_vs_em" if w == "pos" else "copy_contrast_unt_vs_em"]
            syn_ok = (not hr) or bool(syn["lo"] > 0)
            syn_flag = "no-range(waived,low-power)" if not hr else ("collapses" if syn["lo"] > 0 else "no-collapse")
        else:
            syn, syn_ok, syn_flag = None, True, "n/a (sink wiki readout is functional)"
        pvals.append(c["p_le0"] if w != "sink" else min(c["p_le0"], 1 - c["p_le0"]))
        passed = bool(ci_excl_0 and past_theta and monotone and syn_ok and valid[w])
        clears[w] = passed
        detail[w] = {"excess": c["contrast"], "ci": [c["lo"], c["hi"]], "theta": theta,
                     "ci_excl_0": ci_excl_0, "past_theta": past_theta, "monotone": monotone,
                     "synthetic_ok": syn_ok, "synthetic_flag": syn_flag, "construct_valid": valid[w],
                     "passed": passed, "ks_excess": ks_excess}
    bh = P.bh_adjust(pvals).tolist()
    for i, w in enumerate(order):
        detail[w]["bh_p"] = bh[i]
        detail[w]["bh_sig"] = bool(bh[i] < 0.05)
        clears[w] = bool(clears[w] and detail[w]["bh_sig"])

    named_set = [w for w in order if clears[w]]
    # raw degradation vs baseline (to tell H_generic from H_other): U_unt vs em already controls energy;
    # "core live" = U_unt ppl gate large; if nothing beats em but U_unt ppl large => energy/generic.
    core_live = hk["dppl_unt"] > 5.0
    if len(named_set) >= 2:
        # single-dominant requires a magnitude margin; else compound.
        mags = {w: abs(detail[w]["excess"]) for w in named_set}
        top = max(mags, key=mags.get)
        second = max((mags[w] for w in named_set if w != top), default=0.0)
        if mags[top] > 1.5 * second:
            named_set = [top]
    if len(named_set) == 1:
        x = named_set[0]
        label = {"pos": "H_pos (positional addressing)", "indu": "H_indu (induction/copy)",
                 "sink": "H_sink (attention-sink/BOS routing)"}[x]
        return {"named": label, "basis": f"single-dominant {x} beats the energy-matched null on all §6 conjuncts",
                "named_set": named_set, "detail": detail, "core_live": core_live, "poscontrol_valid": True}
    if len(named_set) >= 2:
        return {"named": "H_compound", "basis": f"multiple functions beat the energy-matched null: {named_set}",
                "named_set": named_set, "detail": detail, "core_live": core_live, "poscontrol_valid": True}
    if core_live:
        return {"named": "H_generic", "basis": ("U_unt is LIVE (ppl) but NO probe beats the energy-matched "
                "null — the probe degradation is an energy/variance effect, not a named function"),
                "named_set": [], "detail": detail, "core_live": True, "poscontrol_valid": True}
    return {"named": "H_other", "basis": "U_unt live but none of {pos,indu,sink} degrades beyond the null",
            "named_set": [], "detail": detail, "core_live": core_live, "poscontrol_valid": True}


if __name__ == "__main__":
    main()
