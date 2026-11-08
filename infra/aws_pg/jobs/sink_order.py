"""sink-order — does the BOS-sink collapse CAUSE the position-loss, or co-occur with it?

Child of conserved-core (studies/.../2026-06-24_sink_order_prereg.md). conserved-core: ablating the
conserved q/k input subspace U_unt craters position-decodability (corr 0.93->~0.10) AND uniquely routes
attention onto the BOS sink (text->BOS mass 0.19->~0.64). THIS job is the SUFFICIENCY test of the causal
ordering: INDUCE the BOS sink with U_unt LEFT INTACT, read position-decodability@L9 + ppl + copy
DOWNSTREAM, vs a MATCHED-ATTENTION-MASS null (the same mass diverted to a random non-sink token), over a
dose-response of induced BOS mass. Positive control = U_unt-ablation reproducing the conserved-core anchors.

H_sink-primary  : inducing the sink reproduces the position/ppl damage past the matched-mass null (sink is
                  the proximate cause; one addressing mechanism).
H_addressing-primary : it does NOT (position-loss needs the U_unt removal; the sink is an epiphenomenon).
H_compound / H_generic / INSTRUMENT_BROKEN per the §6 frozen rule.

The clamp (the one NEW mechanism): a forward_pre_hook(with_kwargs) on each DECODER LAYER adds a per-
(query-row, key-col) additive bias to the eager 4D attention_mask -> a +b on the BOS col raises its
post-softmax mass (induce); +b on a random visible non-sink text col (matched-mass null); -b on BOS
(suppress). Pure bias builders + selection in infra/pg/sink_clamp.py (weight-free unit-tested).

FAIL LOUD: no try/except, no synthetic fallback. Reuses the FUNC + conserved_core harness VERBATIM.

  # smoke (tiny ft set => U_unt meaningless, plumbing + clamp-bite only):
  pg run infra.aws_pg.jobs.sink_order --args="--smoke" --no-weights \
      --type g6e.8xlarge,p5.4xlarge --s3-prefix results/sink_order_smoke --budget-min 90
  # full (24 ft, the MDE):
  pg run infra.aws_pg.jobs.sink_order --args="--bias-grid 1,2,3,4 --n-ppl-seq 120 --n-synth 256" \
      --no-weights --type g6e.8xlarge,p5.4xlarge --s3-prefix results/sink_order --budget-min 300
"""
import argparse
import contextlib
import json
import math
import pathlib

# Reuse the harness VERBATIM (imported) — §13 reuse ledger. (All torch imports are lazy / in-function so
# this module imports torch-free for the weight-free verdict unit tests.)
from infra.aws_pg.jobs.invariant_core import PG1_REPO, N_LAYERS
from infra.aws_pg.jobs.invariant_core_func import (
    _load_full_pg1, _decoder_layers, _InputProject, _blank_image, _locate_text_block,
    _paired_contrast_ci,
)
from infra.aws_pg.jobs.conserved_core import (
    _TokenCapture, _wikitext_windows, _build_u_unt, _ppl_from_pw,
)
from infra.pg import conserved_core_probes as P
from infra.pg import sink_clamp as SC

FUNC_BASE_PPL = 22.4258            # FUNC anchor (recomputed in-run; wiring gate)
CC_UNT_DPPL = 49.14               # conserved-core/FUNC anchor: U_unt k=128 Δppl (positive-control gate, ±15)
# positive-control replication bands for the U_unt-ablation arm (recomputed in-run; INSTRUMENT_BROKEN else):
ABL_POS_BAND = (0.0, 0.30)        # position-corr@L9 under U_unt-ablation must land here (anchor ~0.10)
ABL_SINK_BAND = (0.40, 0.90)      # text->BOS sink mass@L9 under U_unt-ablation (anchor ~0.64)
MED_PRIMARY = 0.50                # mediation fraction cut: >= => sink-primary
MED_INERT = 0.25                  # < => addressing-primary (sink does not reproduce the crater)


# ===========================================================================
# The clamp — additive attention-logit bias on the decoder-layer attention_mask.
# ===========================================================================
class _AttnBiasClamp:
    """forward_pre_hook(with_kwargs) on each DECODER LAYER: add a per-(row,col) additive bias to the 4D
    attention_mask (eager adds it to the QKᵀ logits pre-softmax). self.bias_t (T,T) is set per window."""
    def __init__(self, layers, device):
        self.layers = layers
        self.device = device
        self.bias_t = None
        self._handles = []

    def set_bias(self, bias_2d):
        import torch
        import numpy as np
        self.bias_t = torch.as_tensor(np.asarray(bias_2d, dtype=np.float32), device=self.device)

    def _hook(self, _module, args, kwargs):
        assert self.bias_t is not None, "clamp bias not set before forward"
        q, kv = self.bias_t.shape
        am = kwargs.get("attention_mask", None)
        loc = "kwarg"
        if am is None and len(args) >= 2 and hasattr(args[1], "dim") and args[1].dim() == 4:
            am, loc = args[1], "arg1"
        assert am is not None, ("attention_mask not found (kwarg None and args[1] not a 4D mask) — "
                                "transformers decoder-layer plumbing changed; clamp cannot bite")
        assert am.dim() == 4, f"attention_mask not 4D: {tuple(am.shape)}"
        assert am.shape[-2] == q and am.shape[-1] == kv, (tuple(am.shape), (q, kv))
        new_am = am + self.bias_t.to(am.dtype).to(am.device)[None, None]
        if loc == "kwarg":
            nk = dict(kwargs)
            nk["attention_mask"] = new_am
            return args, nk
        return (args[0], new_am) + tuple(args[2:]), kwargs

    def __enter__(self):
        for layer in self.layers:
            self._handles.append(layer.register_forward_pre_hook(self._hook, with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


# ===========================================================================
# One forward per window: ppl + per-probe-layer position features + BOS-sink mass + target-col mass.
# Mirrors conserved_core._probe_forward EXACTLY (so baseline/ablation reproduce the anchors), with the
# clamp (per-window bias) + the target-col mass readout added. clamp_build(pos,n_text,bos_idx,T) ->
# (bias (T,T), target_cols (T,)) or None for no clamp; ablate_Q -> _InputProject Q (or None).
# ===========================================================================
def _forward_arm(model, processor, device, layers, windows, probe_layers, dtype,
                 clamp=None, clamp_build=None, ablate_Q=None, gray=128):
    import torch
    import numpy as np
    blank = _blank_image(gray)
    out = []
    ctxs = []
    if ablate_Q is not None:
        ctxs.append(_InputProject(layers, ablate_Q, device, dtype))
    if clamp is not None:
        assert clamp_build is not None, "clamp set but no clamp_build"
        ctxs.append(clamp)
    with contextlib.ExitStack() as stack:
        for c in ctxs:
            stack.enter_context(c)
        for window_text, win_ids in windows:
            n_text = int(win_ids.shape[0])
            inp = processor(text=[window_text], images=[blank], return_tensors="pt", padding=False).to(device)
            in_ids = inp["input_ids"][0]
            assert in_ids.shape[0] > n_text, "text span >= prompt — image prefix missing"
            pos = _locate_text_block(in_ids, win_ids)
            assert pos >= 1, "no BOS before text block"
            bos_idx = pos - 1
            T = int(in_ids.shape[0])
            target_cols = None
            if clamp is not None:
                bias_2d, target_cols = clamp_build(pos, n_text, bos_idx, T)
                clamp.set_bias(bias_2d)
            cap = _TokenCapture(layers, probe_layers)
            with cap, torch.inference_mode():
                mo = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                           pixel_values=inp["pixel_values"].to(torch.bfloat16), output_attentions=True)
            logits = mo.logits[0]
            text_logits = logits[pos:pos + n_text - 1, :].float()
            targets = win_ids[1:].to(device)
            assert text_logits.shape[0] == targets.shape[0], (text_logits.shape, targets.shape)
            loss = torch.nn.functional.cross_entropy(text_logits, targets, reduction="sum")
            attns = mo.attentions
            assert attns is not None and len(attns) == N_LAYERS, \
                f"output_attentions returned {None if attns is None else len(attns)} (want {N_LAYERS})"
            feats, sink, tgt_mass = {}, {}, {}
            for L in probe_layers:
                buf = cap.buf[L]
                assert buf is not None, f"no capture at L{L}"
                feats[L] = buf[pos:pos + n_text, :].astype(np.float32)
                a = attns[L][0]                                          # (heads, T, T)
                qslice = a[:, pos:pos + n_text, :].float()               # text query rows
                sink[L] = float(qslice[:, :, bos_idx].mean().item())     # mean text->BOS mass
                if target_cols is not None:
                    cols = target_cols[pos:pos + n_text]                 # per-text-row biased col
                    assert (cols >= 0).all(), "target col unset on a text row"
                    # mean over text rows of attention(row -> its target col)
                    rr = qslice[:, np.arange(n_text), cols].float()      # (heads, n_text)
                    tgt_mass[L] = float(rr.mean().item())
                else:
                    tgt_mass[L] = sink[L]                                # no clamp: target == BOS read
            out.append({"text": window_text, "is_conf": P.is_confirmation(window_text), "n_text": n_text,
                        "ppl_pw": [float(loss.item()), int(targets.numel())],
                        "pos_labels": P.position_labels(n_text),
                        "feats": feats, "sink": sink, "target_mass": tgt_mass})
    return out


def _synth_acc_arm(model, processor, device, layers, items, dtype,
                   clamp=None, clamp_build=None, ablate_Q=None, gray=128):
    """copy-corpus next-token accuracy under an arm (clamp/ablation). Mirrors conserved_core._synth_acc
    with per-item clamp bias. Returns (acc, per_item 0/1)."""
    import torch
    import numpy as np
    tok = processor.tokenizer
    blank = _blank_image(gray)
    per_item = []
    ctxs = []
    if ablate_Q is not None:
        ctxs.append(_InputProject(layers, ablate_Q, device, dtype))
    if clamp is not None:
        ctxs.append(clamp)
    with contextlib.ExitStack() as stack:
        for c in ctxs:
            stack.enter_context(c)
        for prompt, gold in items:
            win_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
            n_text = int(win_ids.shape[0])
            inp = processor(text=[prompt], images=[blank], return_tensors="pt", padding=False).to(device)
            in_ids = inp["input_ids"][0]
            pos = _locate_text_block(in_ids, win_ids)
            T = int(in_ids.shape[0])
            if clamp is not None:
                bias_2d, _ = clamp_build(pos, n_text, pos - 1, T)
                clamp.set_bias(bias_2d)
            with torch.inference_mode():
                logits = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                               pixel_values=inp["pixel_values"].to(torch.bfloat16)).logits[0]
            pred = tok.decode([int(logits[pos + n_text - 1].float().argmax().item())]).strip()
            per_item.append(1 if pred == gold else 0)
    return float(np.mean(per_item)), per_item


# ===========================================================================
# Readout helpers (pure-ish: numpy). Position probe trained ONCE on baseline discovery features.
# ===========================================================================
def _pool_feats(per_window, L, conf):
    import numpy as np
    sel = [w for w in per_window if w["is_conf"] == conf]
    X = np.concatenate([w["feats"][L] for w in sel])
    y = np.concatenate([w["pos_labels"] for w in sel])
    return X, y


def _corr_conf(per_window, probe_w, L):
    """position-corr on the CONFIRMATION pool for one arm with the baseline-trained probe."""
    X, y = _pool_feats(per_window, L, conf=True)
    return P.corr_score(y, P.linear_predict(probe_w, X))


def _pos_preds(per_window, probe_w, L):
    """per-window {text,is_conf,y,pred} for the position contrast (paired by window text)."""
    return {w["text"]: {"is_conf": w["is_conf"], "y": w["pos_labels"],
                        "pred": P.linear_predict(probe_w, w["feats"][L])} for w in per_window}


def _pos_excess(sink_preds, null_preds, n_boot, seed, half="conf"):
    """Excess = corr(null) - corr(sink) on the half (>0 ⇔ sink degrades position more). Paired block
    bootstrap over windows (reuses P.probe_contrast_block_ci with the corr metric)."""
    import numpy as np
    blocks = []
    want_conf = (half == "conf")
    for txt, sp in sink_preds.items():
        if sp["is_conf"] != want_conf:
            continue
        np_ = null_preds[txt]
        blocks.append({"y": sp["y"], "sink": sp["pred"], "null": np_["pred"]})
    assert blocks, f"no {half} windows for excess"
    metric = lambda pool, key: P.corr_score(pool["y"], pool[key])
    return P.probe_contrast_block_ci(blocks, metric, key_treat="sink", key_null="null",
                                     n_boot=n_boot, seed=seed)


def _theta_pos(null_a_preds, null_b_preds, n_boot, seed):
    """θ = 95th pct of |corr(null_a) - corr(null_b)| bootstrapped on the DISCOVERY half (the achievable
    null spread of the position Excess under no sink-specificity)."""
    import numpy as np
    blocks = []
    for txt, ap in null_a_preds.items():
        if ap["is_conf"]:
            continue
        bp = null_b_preds[txt]
        blocks.append({"y": ap["y"], "a": ap["pred"], "b": bp["pred"]})
    assert blocks, "no discovery windows for θ"
    keys = ["y", "a", "b"]
    W = len(blocks)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, W, W)
        pool = {k: np.concatenate([blocks[j][k] for j in idx]) for k in keys}
        boots[i] = P.corr_score(pool["y"], pool["b"]) - P.corr_score(pool["y"], pool["a"])
    return float(np.percentile(np.abs(boots), 95))


def _mediation_pos(base_pw, sink_pw, abl_pw, probe_w, L, n_boot, seed):
    """mediation fraction D_sink/D_abl for position on the CONFIRMATION half, with a paired ratio
    bootstrap CI. D_arm = corr(base) - corr(arm). >=0.5 sink-primary, <0.25 addressing-primary."""
    import numpy as np
    def pools(pw):
        sel = [w for w in pw if w["is_conf"]]
        return ([w["feats"][L] for w in sel], [w["pos_labels"] for w in sel])
    bX, bY = pools(base_pw); sX, _ = pools(sink_pw); aX, _ = pools(abl_pw)
    W = len(bY)
    assert W == len(sX) == len(aX), (W, len(sX), len(aX))
    def frac(idx):
        Y = np.concatenate([bY[j] for j in idx])
        cb = P.corr_score(Y, P.linear_predict(probe_w, np.concatenate([bX[j] for j in idx])))
        cs = P.corr_score(Y, P.linear_predict(probe_w, np.concatenate([sX[j] for j in idx])))
        ca = P.corr_score(Y, P.linear_predict(probe_w, np.concatenate([aX[j] for j in idx])))
        d_abl = cb - ca
        return (cb - cs) / d_abl if abs(d_abl) > 1e-9 else float("nan")
    point = frac(np.arange(W))
    rng = np.random.default_rng(seed)
    boots = np.array([frac(rng.integers(0, W, W)) for _ in range(n_boot)], dtype=np.float64)
    boots = boots[np.isfinite(boots)]
    return {"fraction": float(point), "lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5))}


# ===========================================================================
# §6 frozen verdict — pure (numpy), unit-tested weight-free.
# ===========================================================================
def _verdict(pos_excess, ppl_excess, copy_excess, theta, mediation, monotone_pos,
             poscontrol_valid, clamp_bite_ok, null_self_frac):
    """§6 naming rule. Excess>0 ⇔ sink degrades MORE than the matched-mass null. BH across {pos,ppl,copy}.
    mediation = D_sink/D_abl (position). null_self_frac = the matched-mass NULL's own mediation fraction
    (to separate H_addressing-primary from H_generic). Returns {named, basis, ...}."""
    import numpy as np
    if not (poscontrol_valid and clamp_bite_ok):
        return {"named": "INSTRUMENT_BROKEN",
                "basis": f"poscontrol_valid={poscontrol_valid} clamp_bite_ok={clamp_bite_ok}",
                "poscontrol_valid": poscontrol_valid, "clamp_bite_ok": clamp_bite_ok}
    p_pos = pos_excess["p_le0"]
    p_ppl = ppl_excess["p_le0"]
    p_copy = copy_excess["p_le0"]
    bh = P.bh_adjust([p_pos, p_ppl, p_copy]).tolist()
    bh_sig_pos = bool(bh[0] < 0.05)
    excess_pos = pos_excess["contrast"]
    ci_lo = pos_excess["lo"]
    sink_specific = bool(ci_lo > 0 and abs(excess_pos) > theta and bh_sig_pos)
    med = mediation["fraction"]
    detail = {"pos_excess": excess_pos, "pos_ci": [pos_excess["lo"], pos_excess["hi"]], "theta": theta,
              "monotone_pos": bool(monotone_pos), "bh": bh, "bh_sig_pos": bh_sig_pos,
              "mediation_fraction": med, "mediation_ci": [mediation["lo"], mediation["hi"]],
              "null_self_fraction": null_self_frac, "sink_specific": sink_specific}
    if sink_specific and monotone_pos and med >= MED_PRIMARY:
        return {"named": "H_sink-primary", "detail": detail,
                "basis": (f"inducing the BOS sink reproduces {med:.0%} of the U_unt-ablation position "
                          f"crater, past the matched-mass null (Excess {excess_pos:+.3f} "
                          f"[{pos_excess['lo']:+.3f},{pos_excess['hi']:+.3f}]>θ={theta:.3f}, monotone, BH-sig) "
                          f"⇒ the sink is the proximate cause; one addressing mechanism")}
    if sink_specific and monotone_pos and med < MED_PRIMARY:
        return {"named": "H_compound", "detail": detail,
                "basis": (f"the sink is sink-specific (Excess>θ, BH-sig, monotone) but reproduces only "
                          f"{med:.0%} (<50%) of the ablation crater ⇒ one causal pathway among others")}
    # not sink-specific: separate addressing-primary (sink inert) from generic (any diversion hits).
    if med < MED_INERT and null_self_frac < MED_INERT:
        return {"named": "H_addressing-primary", "detail": detail,
                "basis": (f"inducing the sink does NOT reproduce the position crater (mediation {med:.0%}<25%, "
                          f"Excess CI brackets 0) and the matched-mass null is also inert ⇒ position-loss "
                          f"requires the U_unt removal; the sink is a downstream epiphenomenon")}
    if null_self_frac >= MED_PRIMARY:
        return {"named": "H_generic", "detail": detail,
                "basis": (f"the matched-mass null itself degrades position ({null_self_frac:.0%} of ablation) "
                          f"as much as the sink (Excess CI brackets 0) ⇒ generic attention disruption, not "
                          f"sink-specific")}
    return {"named": "H_compound", "detail": detail,
            "basis": (f"intermediate: Excess not clearly sink-specific (lo {pos_excess['lo']:+.3f}) yet the "
                      f"sink reproduces {med:.0%} of the crater — neither dominant nor inert; reported compound")}


def _monotone(xs):
    import numpy as np
    a = np.abs(np.asarray(xs, dtype=np.float64))
    return bool(np.all(np.diff(a) >= -1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ft-tasks", default="", help="comma slugs; empty => all -224 (24-ft cloud)")
    ap.add_argument("--k", type=int, default=128, help="U_unt rank (headline; conserved-core k=128)")
    ap.add_argument("--headline-layer", type=int, default=9)
    ap.add_argument("--probe-layers", default="4,9,13")
    ap.add_argument("--n-ppl-seq", type=int, default=120)
    ap.add_argument("--ppl-window", type=int, default=512)
    ap.add_argument("--n-synth", type=int, default=256)
    ap.add_argument("--bias-grid", default="1,2,3,4,5", help="SINK additive-bias dose-response grid")
    ap.add_argument("--null-bias-grid", default="2,4,6,8,10,12",
                    help="MATCHED-NULL bias grid — higher range: BOS is a far stronger attractor per "
                         "unit bias than a random text col, so the null needs a bigger bias to reach the "
                         "sink's achieved mass (smoke 2026-06-24: b=4 -> sink BOS 0.64 but null div 0.08)")
    ap.add_argument("--mass-match-tol", type=float, default=0.12,
                    help="|null div mass - sink BOS mass| above this at the headline => low-power flag")
    ap.add_argument("--suppress-bias", type=float, default=10.0)
    ap.add_argument("--ridge-lam", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = [float(x) for x in a.bias_grid.split(",")]
    null_grid = [float(x) for x in a.null_bias_grid.split(",")]
    headline_layer = a.headline_layer
    probe_layers = [int(x) for x in a.probe_layers.split(",") if x.strip()]
    if a.smoke:
        grid = [2.0, 4.0]
        null_grid = [4.0, 8.0, 12.0]          # confirm the null CAN reach the sink's ~0.64 mass
        a.n_ppl_seq = 16
        a.n_synth = 24
        probe_layers = [9]
        a.k = 8
        if not a.ft_tasks.strip():
            a.ft_tasks = "vqav2,gqa,ai2d"
    assert headline_layer in probe_layers, "headline layer must be a probe layer"
    want_layers = list(range(N_LAYERS))

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoProcessor
    assert torch.cuda.is_available(), "GPU required for the pt-224 forward"
    device = "cuda"
    api = HfApi()

    rep = {"object": "sink_order_v1", "base_repo": PG1_REPO, "k": a.k, "headline_layer": headline_layer,
           "probe_layers": probe_layers, "bias_grid": grid, "null_bias_grid": null_grid,
           "n_ppl_seq": a.n_ppl_seq, "n_synth": a.n_synth, "seed": a.seed, "smoke": a.smoke,
           "anchors": {"base_ppl": FUNC_BASE_PPL, "unt_dppl": CC_UNT_DPPL,
                       "abl_pos_band": ABL_POS_BAND, "abl_sink_band": ABL_SINK_BAND}}

    print("[so] building U_unt(L,k) from the consensus Gram over the ft cloud ...", flush=True)
    U_unt, repos, shas, base_sha = _build_u_unt(api, [a.k], want_layers, a.ft_tasks)
    Q_unt = {L: U_unt[(L, a.k)] for L in want_layers}
    rep.update({"base_sha": base_sha, "n_ft": len(repos), "cloud_repos": repos})
    print(f"[so] U_unt built: {len(repos)} ft, k={a.k}", flush=True)

    processor = AutoProcessor.from_pretrained(PG1_REPO, revision=base_sha)
    model = _load_full_pg1(base_sha, device)
    layers = _decoder_layers(model)
    dtype = next(model.parameters()).dtype
    windows = _wikitext_windows(processor, a.n_ppl_seq, a.ppl_window)
    print(f"[so] {len(windows)} wikitext windows", flush=True)

    clamp = _AttnBiasClamp(layers, device)

    def build_sink(b):
        return lambda pos, n, bos, T: SC.bos_bias(pos, n, bos, T, b)

    def build_null(b, seed):
        return lambda pos, n, bos, T: SC.matched_mass_bias(pos, n, bos, T, b, seed + pos * 131 + n)

    def build_suppress(b):
        return lambda pos, n, bos, T: SC.suppress_bias(pos, n, bos, T, b)

    # ---- BASELINE (no clamp, no ablation) — trains the probe + the anchors.
    base_pw = _forward_arm(model, processor, device, layers, windows, probe_layers, dtype)
    base_ppl = _ppl_from_pw(base_pw)
    assert a.smoke or abs(base_ppl - FUNC_BASE_PPL) <= 1.0, f"WIRING GATE: base ppl {base_ppl} vs {FUNC_BASE_PPL}"
    print(f"[so][PRE] base ppl={base_ppl:.4f}", flush=True)
    # probe trained ONCE on baseline DISCOVERY features (per probe layer); construct validity on conf.
    probes, base_bands = {}, {}
    for L in probe_layers:
        Xd, yd = _pool_feats(base_pw, L, conf=False)
        probes[L] = P.ridge_fit(Xd, yd, lam=a.ridge_lam)
        corr_clean = _corr_conf(base_pw, probes[L], L)
        probe_shuf = P.ridge_fit(Xd, P.shuffle_labels(yd, a.seed), lam=a.ridge_lam)
        Xc, yc = _pool_feats(base_pw, L, conf=True)
        corr_shuf = P.corr_score(yc, P.linear_predict(probe_shuf, Xc))
        sink_base = float(np.mean([w["sink"][L] for w in base_pw if w["is_conf"]]))
        base_bands[L] = {"corr_clean": corr_clean, "corr_shuffle": corr_shuf, "sink_base": sink_base}
        print(f"[so][PRE] L{L} pos-corr={corr_clean:.3f}(shuf {corr_shuf:.3f}) sink_BOS={sink_base:.4f}", flush=True)
    rep["base_ppl"] = base_ppl
    rep["base_bands"] = base_bands
    Lh = headline_layer
    construct_valid = bool((base_bands[Lh]["corr_clean"] - base_bands[Lh]["corr_shuffle"]) > 0.05)
    assert a.smoke or construct_valid, f"position probe construct-invalid at L{Lh}: {base_bands[Lh]}"
    (out / "sink_order.json").write_text(json.dumps(rep, indent=2, default=str))

    # ---- POSITIVE CONTROL: U_unt-ablation reproduces the conserved-core anchors (full-damage reference).
    abl_pw = _forward_arm(model, processor, device, layers, windows, probe_layers, dtype, ablate_Q=Q_unt)
    abl_ppl = _ppl_from_pw(abl_pw)
    abl_dppl = abl_ppl - base_ppl
    abl_corr = _corr_conf(abl_pw, probes[Lh], Lh)
    abl_sink = float(np.mean([w["sink"][Lh] for w in abl_pw if w["is_conf"]]))
    poscontrol_valid = bool(
        (a.smoke or abs(abl_dppl - CC_UNT_DPPL) <= 15.0)
        and ABL_POS_BAND[0] <= abl_corr <= ABL_POS_BAND[1]
        and ABL_SINK_BAND[0] <= abl_sink <= ABL_SINK_BAND[1])
    rep["positive_control"] = {"dppl": abl_dppl, "pos_corr": abl_corr, "sink_mass": abl_sink,
                               "valid": poscontrol_valid}
    print(f"[so][POS-CTRL] U_unt-abl dppl={abl_dppl:+.2f} pos-corr={abl_corr:.3f} sink={abl_sink:.3f} "
          f"valid={poscontrol_valid}", flush=True)
    assert a.smoke or poscontrol_valid, f"POSITIVE-CONTROL FAIL (INSTRUMENT_BROKEN): {rep['positive_control']}"
    (out / "sink_order.json").write_text(json.dumps(rep, indent=2, default=str))

    # ---- DOSE-RESPONSE. SINK grid + (separate, higher) NULL grid — BOS is a far stronger attractor per
    # unit bias than a random text col, so the null needs a bigger bias to reach the sink's achieved mass
    # (smoke 2026-06-24). "matched-attention-mass" is matched on ACHIEVED mass, not bias.
    sink_dose, null_dose, sink_arms, null_arms = [], [], {}, {}
    for b in grid:
        sp = _forward_arm(model, processor, device, layers, windows, probe_layers, dtype,
                          clamp=clamp, clamp_build=build_sink(b))
        sink_arms[b] = sp
        s_bos = float(np.mean([w["sink"][Lh] for w in sp if w["is_conf"]]))
        sink_dose.append({"b": b, "sink_bos_mass": s_bos, "sink_pos_corr": _corr_conf(sp, probes[Lh], Lh),
                          "sink_dppl": _ppl_from_pw(sp) - base_ppl})
        print(f"[so][DOSE-sink] b={b} BOS={s_bos:.3f} pos={sink_dose[-1]['sink_pos_corr']:.3f} "
              f"dppl={sink_dose[-1]['sink_dppl']:+.2f}", flush=True)
    for b in null_grid:
        npw = _forward_arm(model, processor, device, layers, windows, probe_layers, dtype,
                           clamp=clamp, clamp_build=build_null(b, a.seed))
        null_arms[b] = npw
        n_div = float(np.mean([w["target_mass"][Lh] for w in npw if w["is_conf"]]))
        null_dose.append({"b": b, "null_div_mass": n_div, "null_pos_corr": _corr_conf(npw, probes[Lh], Lh),
                          "null_dppl": _ppl_from_pw(npw) - base_ppl})
        print(f"[so][DOSE-null] b={b} div={n_div:.3f} pos={null_dose[-1]['null_pos_corr']:.3f} "
              f"dppl={null_dose[-1]['null_dppl']:+.2f}", flush=True)
    rep["dose_response"] = {"sink": sink_dose, "null": null_dose}
    rep["clamp_bite_ok"] = bool(max(d["sink_bos_mass"] for d in sink_dose) > base_bands[Lh]["sink_base"] + 0.05)
    (out / "sink_order.json").write_text(json.dumps(rep, indent=2, default=str))
    print(f"[so] clamp_bite_ok={rep['clamp_bite_ok']} (max sink BOS "
          f"{max(d['sink_bos_mass'] for d in sink_dose):.3f}; max null div "
          f"{max(d['null_div_mass'] for d in null_dose):.3f})", flush=True)

    # ---- HEADLINE: sink level whose BOS mass closest to the ablation sink; null level whose diverted
    # mass matches that sink BOS mass (matched-attention-mass). mass_match_ok flags the §2(iii) low-power.
    hi_sink = SC.closest_level([d["sink_bos_mass"] for d in sink_dose], abl_sink)
    b_sink = grid[hi_sink]
    target_div = sink_dose[hi_sink]["sink_bos_mass"]
    hi_null = SC.closest_level([d["null_div_mass"] for d in null_dose], target_div)
    b_null = null_grid[hi_null]
    mass_match_ok = bool(abs(null_dose[hi_null]["null_div_mass"] - target_div) <= a.mass_match_tol)
    sink_h = sink_arms[b_sink]
    null_h = null_arms[b_null]
    print(f"[so][HEAD] sink b={b_sink} (BOS {target_div:.3f}≈abl {abl_sink:.3f}); "
          f"null b={b_null} (div {null_dose[hi_null]['null_div_mass']:.3f}≈sink {target_div:.3f}) "
          f"mass_match_ok={mass_match_ok}", flush=True)

    # second matched-null draw (seed_b) at the null headline level => θ.
    null_b_pw = _forward_arm(model, processor, device, layers, windows, probe_layers, dtype,
                             clamp=clamp, clamp_build=build_null(b_null, a.seed + 99991))

    sink_preds = _pos_preds(sink_h, probes[Lh], Lh)
    null_preds = _pos_preds(null_h, probes[Lh], Lh)
    null_b_preds = _pos_preds(null_b_pw, probes[Lh], Lh)
    pos_excess = _pos_excess(sink_preds, null_preds, a.n_boot, a.seed, half="conf")
    theta = _theta_pos(null_preds, null_b_preds, a.n_boot, a.seed)
    ppl_excess = _paired_contrast_ci([w["ppl_pw"] for w in sink_h if w["is_conf"]],
                                     [w["ppl_pw"] for w in null_h if w["is_conf"]], seed=a.seed)
    mediation = _mediation_pos(base_pw, sink_h, abl_pw, probes[Lh], Lh, a.n_boot, a.seed)
    # the matched-null's OWN mediation fraction (to separate addressing-primary from generic).
    null_self = _mediation_pos(base_pw, null_h, abl_pw, probes[Lh], Lh, a.n_boot, a.seed)
    monotone_pos = _monotone([d["sink_pos_corr"] - base_bands[Lh]["corr_clean"] for d in sink_dose])  # deg grows
    rep["headline"] = {"b_sink": b_sink, "b_null": b_null, "sink_bos_mass": target_div,
                       "null_div_mass": null_dose[hi_null]["null_div_mass"], "abl_sink_mass": abl_sink,
                       "mass_match_ok": mass_match_ok,
                       "pos_excess": pos_excess, "theta": theta, "ppl_excess": ppl_excess,
                       "mediation": mediation, "null_self_mediation": null_self,
                       "monotone_pos": monotone_pos}
    (out / "sink_order.json").write_text(json.dumps(rep, indent=2, default=str))
    print(f"[so][HEAD] pos Excess={pos_excess['contrast']:+.3f}[{pos_excess['lo']:+.3f},"
          f"{pos_excess['hi']:+.3f}] θ={theta:.3f} | ppl Excess={ppl_excess['contrast']:+.2f} | "
          f"mediation={mediation['fraction']:.2f}[{mediation['lo']:.2f},{mediation['hi']:.2f}] "
          f"null_self={null_self['fraction']:.2f}", flush=True)

    # ---- COPY (addressing-sensitive behavioral readout) under {base, abl, sink_h, null_h}.
    copy_items = P.copy_corpus(a.n_synth, seed=a.seed)
    base_copy, base_copy_pi = _synth_acc_arm(model, processor, device, layers, copy_items, dtype)
    abl_copy, _ = _synth_acc_arm(model, processor, device, layers, copy_items, dtype, ablate_Q=Q_unt)
    sink_copy, sink_copy_pi = _synth_acc_arm(model, processor, device, layers, copy_items, dtype,
                                             clamp=clamp, clamp_build=build_sink(b_sink))
    null_copy, null_copy_pi = _synth_acc_arm(model, processor, device, layers, copy_items, dtype,
                                             clamp=clamp, clamp_build=build_null(b_null, a.seed))
    copy_excess = P.synth_contrast_bootstrap(base_copy_pi, sink_copy_pi, null_copy_pi,
                                             n_boot=a.n_boot, seed=a.seed)
    rep["copy"] = {"base": base_copy, "abl": abl_copy, "sink": sink_copy, "null": null_copy,
                   "excess": copy_excess}
    print(f"[so][COPY] base={base_copy:.3f} abl={abl_copy:.3f} sink={sink_copy:.3f} null={null_copy:.3f} "
          f"excess={copy_excess['contrast']:+.3f}[{copy_excess['lo']:+.3f},{copy_excess['hi']:+.3f}]", flush=True)

    # ---- SUPPRESSION (secondary necessity arm): U_unt-ablation + suppress the BOS sink => recovery?
    sup_pw = _forward_arm(model, processor, device, layers, windows, probe_layers, dtype,
                          ablate_Q=Q_unt, clamp=clamp, clamp_build=build_suppress(a.suppress_bias))
    sup_corr = _corr_conf(sup_pw, probes[Lh], Lh)
    sup_dppl = _ppl_from_pw(sup_pw) - base_ppl
    sup_sink = float(np.mean([w["sink"][Lh] for w in sup_pw if w["is_conf"]]))
    rep["suppression"] = {"pos_corr": sup_corr, "dppl": sup_dppl, "sink_mass": sup_sink,
                          "abl_pos_corr": abl_corr, "abl_dppl": abl_dppl, "base_corr": base_bands[Lh]["corr_clean"],
                          "pos_recovered": bool(sup_corr > abl_corr + 0.05),
                          "ppl_recovered": bool(sup_dppl < abl_dppl - 2.0)}
    print(f"[so][SUPP] abl+suppress: sink {abl_sink:.3f}->{sup_sink:.3f} pos {abl_corr:.3f}->{sup_corr:.3f} "
          f"dppl {abl_dppl:+.2f}->{sup_dppl:+.2f}", flush=True)

    # ---- VERDICT (§6).
    v = _verdict(pos_excess, ppl_excess, copy_excess, theta, mediation, monotone_pos,
                 poscontrol_valid, rep["clamp_bite_ok"], null_self["fraction"])
    rep["verdict"] = v
    (out / "sink_order.json").write_text(json.dumps(rep, indent=2, default=str))
    print("=" * 78, flush=True)
    print(f"[so] VERDICT: {v['named']} — {v['basis']}", flush=True)
    print(f"SINK_ORDER_OK verdict={v['named']} n_ft={len(repos)} k={a.k}", flush=True)


if __name__ == "__main__":
    main()
