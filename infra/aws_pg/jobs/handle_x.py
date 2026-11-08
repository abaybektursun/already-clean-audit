"""HANDLE Arm X (GPU) — verdict-level replication of the core laws on a SECOND frozen public
model, PaliGemma-2 (`google/paligemma2-3b-pt-224`, Gemma-2-2B LM). Spec §3.1 Arm X, §3.2 (the
firewall), §6 R-X, §11 P11.

WHAT THIS ARM IS. Arm X asks, for each banked PG1 core law, ONE categorical question: does the
same frozen rule land the SAME branch on a different frozen public model? The deliverable is one
row per core law — {two-regime energy leg, within-level operator ordering, kernel theorems,
mid-stack spike depletion, (bonus) stiff-carries write-side} × {replicates / lands-elsewhere /
VOID}. A law that lands elsewhere on PG2 ACQUIRES its scope boundary — the external-validity datum
no amount of PG1 work can supply. Every landing is informative; a VOID is an honest
instrument-scoping measurement, never a failure.

THE FIREWALL (spec §3.2, reasoned in code below). This arm compares VERDICTS ONLY.
  1. Every PG2 decision threshold is EITHER (a) calibrated on PG2's OWN control data — matched
     seed bands, split-half operator agreement, PG2 Haar bulk references — OR (b) an ARCHITECTURAL
     / mathematical tolerance that is model-independent (theorem exactness ~0 by RMSNorm/RoPE
     algebra; the finite-difference wiring tolerance; participation-ratio math). NEVER a PG1
     number. No PG1 measurement is imported as a PG2 threshold anywhere in this file.
  2. The `pg1` field fed to `adjudicate_rx` is ONLY a categorical branch LABEL (a verdict string
     from the frozen PG1 result docs, `PG1_LABELS` below with provenance). It is a LABEL, not a
     number; it enters ONLY the final `pg1 == pg2` equality check and NEVER touches any PG2
     computation. `adjudicate_rx` (infra/pg/handle_core.py, NOT redefined here) reads
     {law:{pg1,pg2,void}} → per-law replicates/lands-elsewhere/VOID.
  3. Cross-lineage firewall: PG2 is Gemma-2-based; the RETIRED Gemma-2 deletion object is NOT
     touched — no LATLOC/LATABLATE-class deletion contrasts. Arm X only reproduces the
     operator / energy-law / spike / kernel-theorem CONSTRUCTIONS.
  4. Arm X is FULLY INDEPENDENT of the PG1 arms and never blocks them (spec §9.4).

REUSE LEDGER (the parent machinery reproduced on PG2 — nothing about the operator is re-derived).
  * Model load: `PaliGemmaForConditionalGeneration` (the repo's PG2 path — infra/aws_pg/jobs/
    smoke.py, infra/pg/entities.py), bf16, EAGER (sdpa mishandles the prefix-LM mask →
    reference_paligemma_sdpa_wrong_forward). Decoder layers via
    `gemma_scope.resolve_gemma2_layers` (generic Gemma-2 resolver); config resolved DYNAMICALLY
    (num_hidden_layers / hidden_size / num_key_value_heads / head_dim / rope_theta / softcap) —
    NOTHING hardcoded to PG1's 18 layers / d=2048.
  * Operator (SPECTER): `specter_core.make_residual_injector` / `mode_matvec` / `mode_quadform`
    / `lanczos` / `stable_spikes` / `participation_ratio` / `field_quadform` / `adjudicate_ro2_*`
    / `fd_directional_check`; item construction mirrors `_specter_shared.build_ppl_item` /
    `build_vqa_item` / `wikitext_windows` (PaliGemma-generic) but through local `_op_ppl_item` /
    `_op_vqa_item` that force `use_cache=False` — the PG2 operator-wiring fix (see their docstring:
    the Gemma-2 DynamicCache in-place write drops the jvp tangent). Lanczos top-32 at the
    embedding-output site, both losses ppl+vqa; differentiation route certified at Gate-X.
  * Energy-law damage cells (CARRY/ATLAS): coherent-vs-iid residual fields at matched per-token
    energy, `_handle_shared.safe_ppl_gate` readout (the log(1+Δppl) scale).
  * Kernel theorems (SPECTER/ATLAS): the radial-gauge family (`adjudicate_ro2_radial`) and the
    counter-rotated post-RoPE-K coherent slice (`atlas_core.delta_postrope_coherent` +
    `delta_multihead` for GQA) vs a PG2 Haar bulk.
  * Adjudicator: `handle_core.adjudicate_rx` — FED, not redefined.

No try/except hiding — a wrong assumption RAISES (CLAUDE.md). Heavy imports inside main so the
module py_compiles on the control plane; hf_online_setup() before transformers/datasets.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.handle_x --type p5.4xlarge,g6e.8xlarge --no-weights \
      --s3-prefix results/handle_x --budget-min 720 --args "--losses ppl,vqa"
"""
import argparse
import hashlib
import json
import os
import pathlib

PG2_REPO = "google/paligemma2-3b-pt-224"

# ---- frozen scoped-battery constants (spec §3.1 Arm X: ~20-cell + operator pass + spike census).
N_OP_PPL_X = 16          # ppl operator batch A (A1 = first 8, A2 = next 8)
N_OP_HELD_X = 8          # held-out batch B for the spike Rayleigh check
N_OP_VQA_X = 12          # vqa operator batch A (6 + 6)
LANCZOS_M_X = 44         # Lanczos iterations (top-32 Ritz reported); one restart at 2× on non-conv
TOPK_X = 32              # "Lanczos top-32" (spec §3.1)
EPS_PROBE_FRAC = 0.05    # operator probe amplitude ×site_rms (linear jvp ⇒ cancels; conditioning only)
TR_EPS_FRAC = 0.25       # two-regime DAMAGE amplitude ×site_rms (behavioural cell)
N_SEED_TR = 3            # coherent/iid seeds per level → the self-calibrated two-regime band
N_BULK_DRAWS = 8         # Haar bulk draws for the kernel-theorem references
FD_REL_TOL = 1e-2        # Gate-X finite-difference wiring tolerance (architectural, not PG1)
CONV_RESID_FRAC = 0.10   # top-8 Ritz residual ≤ 0.1·λ₁ ⇒ converged
WHITE_SIGN_SEED = 211    # the frozen white-envelope Rademacher sign-set (specter_core convention)

# ===========================================================================
# PG1 branch LABELS (categorical verdicts from the FROZEN PG1 result docs) — provenance recorded.
# THESE ARE LABELS, NOT NUMBERS. They enter ONLY the final `pg1 == pg2` equality in adjudicate_rx
# and NEVER touch any PG2 computation (the firewall, §3.2). Sourced from the lineage RESEARCH_INDEX
# entries 11–13 (BISPEC / ATLAS / SPECTER) + CARRY R-C — verdict strings, not baseline metrics.
# ===========================================================================
PG1_LABELS = {
    "two_regime_energy": (
        "H_two_regime",
        "BISPEC/ATLAS: at matched per-token residual energy a coherent field damages far more "
        "than an incoherent (iid) field, well-separated across sites (results/bispec_*, atlas_m_*)."),
    "operator_ordering": (
        "coherent_ordered",
        "SPECTER results/specter_o: at the embedding-output site the coherent (dc) operator "
        "carries STABLE spikes while the white (iid) operator is flat (PR 498-980, no stable "
        "spike) — coherent modes lead the within-level operator ordering."),
    "kernel_theorems": (
        "H_kernels-hold",
        "SPECTER results/specter_o: counter-rotated post-RoPE-K coherent slice = 2.3e-5×bulk "
        "(theorem verified) AND the radial gauge = 0.196×bulk (H_skip-share-measured) — both "
        "exact-architecture kernel checks land where the theorems require."),
    "midstack_spike_depletion": (
        "H_mid-depleted",
        "SPECTER results/specter_o: dc-ppl operator carries stable spikes at the stack ENDS "
        "(L0 & L14) but the MID site (L8) is spike-free with near-flat ladders — spikes deplete "
        "mid-stack."),
    "stiff_carries_writeside": (
        "H_partial",
        "CARRY results/carry_c_sealed.rc: the stiff subspace beats the matched-rank band but "
        "the flat complement carries MORE (Δcarry flat−stiff = +0.113) — H_partial."),
}


# ===========================================================================
# PG2 load + DYNAMIC architecture resolution (nothing hardcoded to PG1).
# ===========================================================================
def load_pg2(device="cuda", sha=None):
    """PG2 on GPU, bf16, EAGER (mandatory — sdpa mishandles the prefix-LM mask). Returns
    (model, proc, layers, arch) with `arch` resolved from the config, never assumed."""
    import torch
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

    from infra.pg.entities import resolve_submodule_paths
    from infra.pg.gemma_scope import resolve_gemma2_layers
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        PG2_REPO, revision=sha, torch_dtype=torch.bfloat16,
        device_map=device, attn_implementation="eager").eval()
    proc = PaliGemmaProcessor.from_pretrained(PG2_REPO, revision=sha)
    lm = model.get_submodule(resolve_submodule_paths(model)["language_model"])
    layers = resolve_gemma2_layers(lm)
    cfg = model.config
    tcfg = getattr(cfg, "text_config", cfg)              # PaliGemma nests the LM config
    d = int(tcfg.hidden_size)
    n_heads = int(tcfg.num_attention_heads)
    head_dim = int(getattr(tcfg, "head_dim", d // n_heads))
    arch = {
        "num_hidden_layers": int(tcfg.num_hidden_layers),
        "hidden_size": d,
        "num_attention_heads": n_heads,
        "num_key_value_heads": int(getattr(tcfg, "num_key_value_heads", n_heads)),
        "head_dim": head_dim,
        "rope_theta": float(getattr(tcfg, "rope_theta", 10000.0)),
        # Gemma-2 caps attention logits with tanh — the a-priori break-suspect for the T2 key
        # gauge theorem (a constant added to all keys does NOT commute with tanh(·/softcap)).
        "attn_logit_softcapping": (None if getattr(tcfg, "attn_logit_softcapping", None) is None
                                   else float(tcfg.attn_logit_softcapping)),
        "n_resolved_layers": len(layers),
    }
    assert arch["n_resolved_layers"] == arch["num_hidden_layers"], \
        f"resolved {arch['n_resolved_layers']} layers vs config {arch['num_hidden_layers']}"
    assert d > 0 and arch["num_hidden_layers"] >= 6, "PG2 architecture resolution failed"
    return model, proc, layers, arch


def three_depths(n_layers):
    """Early / mid / late sites for the spike census (spec §3.1: '3 depths'). The early site is
    the embedding-output site (residual entering layer 0 — the operator's home). Chosen
    dynamically from PG2's own depth; NEVER the PG1 (0,8,14) triple."""
    depths = [0, n_layers // 2, n_layers - 3]
    assert len(set(depths)) == 3 and 0 <= depths[0] < depths[1] < depths[2] < n_layers, \
        f"degenerate depths {depths} for {n_layers} layers"
    return depths


# ===========================================================================
# Leg A helpers — the two-regime energy DAMAGE cells (coherent vs iid at matched per-token energy).
# Self-calibrated hooks; no position tracker needed (each token normalised independently).
# ===========================================================================
def coherent_hook(u_unit, eps_abs):
    """One shared random direction added to EVERY token at per-token norm eps_abs (the coherent
    regime). u_unit (d,) unit."""
    from infra.pg.inherit_core import _split_hidden

    def hook(_m, args, kwargs):
        hs, rebuild = _split_hidden(args, kwargs)
        x = hs.float()
        field = float(eps_abs) * u_unit.to(x.device).view(1, 1, -1)
        return rebuild((x + field).to(hs.dtype))
    return hook


def white_hook(seed, eps_abs):
    """A fresh iid draw per token, renormalised to per-token norm eps_abs (the incoherent regime).
    Matched per-token energy to the coherent cell by construction."""
    import torch
    from infra.pg.inherit_core import _split_hidden

    def hook(_m, args, kwargs):
        hs, rebuild = _split_hidden(args, kwargs)
        x = hs.float()
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        g = torch.randn(x.shape, generator=gen).to(x.device)
        g = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12) * float(eps_abs)
        return rebuild((x + g).to(hs.dtype))
    return hook


# ---------------------------------------------------------------------------
# Operator ITEM builders — the PG2 OPERATOR-WIRING FIX (spec R-W2: a wiring miss is fix-and-rerun).
#
# ROOT CAUSE. `_specter_shared.build_ppl_item`/`build_vqa_item` leave `use_cache` at the config
# default. On the Gemma-2 stack a single scoring forward then builds a `DynamicCache` whose
# in-place K/V write DROPS the forward-mode (`torch.func.jvp`) tangent that carries the injected
# residual direction — so the ANALYTIC operator (jvp/vjp `mode_matvec`) reads ≈0 while the
# finite-difference forward (cache-agnostic) sees the full logit change. That is exactly the
# certified signature Arm X observed on PG2: q_analytic 2.1e-6 vs q_fd 1403 (rel_err 1.0, q≈0),
# a ~10⁹ mismatch with the analytic form pinned at zero — a WIRING gap, not a softcap/GQA scope
# boundary (those give a moderate mismatch, never q≈0). Fix: force `use_cache=False` (no cache is
# needed for a single non-generative scoring forward) so the jvp/vjp identity is restored, and
# certify at Gate-X. These mirror the shared builders VERBATIM except for `use_cache=False`; the
# frozen shared PG1 builders are left untouched (cross-model firewall — PG1's SPECTER banks stand).
# ---------------------------------------------------------------------------
def _op_ppl_item(model, proc, dev, wdict, install, holder):
    import torch

    from infra.aws_pg.jobs.invariant_core_func import _locate_text_block
    inp = proc(text=[wdict["text"]], images=[wdict["blank"]], return_tensors="pt",
               padding=False).to(dev)
    inp["pixel_values"] = inp["pixel_values"].to(next(model.parameters()).dtype)
    in_ids = inp["input_ids"][0]
    win_ids = wdict["win_ids"]
    n_text = int(win_ids.shape[0])
    pos = _locate_text_block(in_ids.cpu(), win_ids)
    T = int(in_ids.shape[0])
    n_t = n_text - 1
    loss_w = torch.zeros(T, dtype=torch.float32)
    loss_w[pos:pos + n_t] = 1.0 / n_t
    labels = torch.full((T,), -100, dtype=torch.long)
    labels[pos:pos + n_t] = win_ids[1:]

    def forward_fn():
        with install():
            return model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                         pixel_values=inp["pixel_values"], use_cache=False).logits
    return {"forward_fn": forward_fn, "holder": holder, "loss_w": loss_w.to(dev),
            "labels": labels, "T": T, "hash": wdict["hash"], "loss": "ppl"}


def _op_vqa_item(model, proc, dev, row, task, install, holder):
    import torch

    from infra.aws_pg.jobs._specter_shared import majority_answer
    img = task.image_fn(row).convert("RGB")
    prompt = task.prompt_fn(row)
    ans = majority_answer(row)
    inp = proc(text=[prompt], images=[img], suffix=[ans], return_tensors="pt",
               padding=False).to(dev)
    inp["pixel_values"] = inp["pixel_values"].to(next(model.parameters()).dtype)
    labels = inp["labels"][0].cpu()
    T = int(inp["input_ids"].shape[1])
    assert int((labels != -100).sum()) >= 1, "teacher-forced answer produced no label tokens"
    loss_w = torch.zeros(T, dtype=torch.float32)
    shift_labels = torch.full((T,), -100, dtype=torch.long)
    shift_labels[:-1] = labels[1:]
    lw = shift_labels != -100
    loss_w[lw] = 1.0 / int(lw.sum())

    def forward_fn():
        with install():
            return model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                         pixel_values=inp["pixel_values"], use_cache=False).logits
    return {"forward_fn": forward_fn, "holder": holder, "loss_w": loss_w.to(dev),
            "labels": shift_labels, "T": T, "loss": "vqa"}


# ---------------------------------------------------------------------------
# The FIRST/SECOND-ORDER wiring diagnostic + the fp32 operator-precision fix (the final round).
#
# LOCALIZATION (banked diagnostic on PG2). The injected direction IS connected to the loss —
# ∂loss/∂coef is FINITE and non-None (first order fine, coef_grad_none=False) — but the analytic
# SECOND derivative ∂²loss/∂coef² = vᵀHv (reverse-mode double-backprop) returns ≈0 while the
# finite-difference Hessian is large and real (h_analytic ≈ -1e-3 vs h_fd ≈ -0.52, both soft-cap
# on and off). Fine at 1st order, ≈0 at 2nd order is the signature of bf16 PRECISION LOSS in the
# double-backprop: the tiny second-order terms underflow in bfloat16. PG1 certified in bf16; PG2's
# operator (its own curvature scale) underflows. Soft-capping was RULED OUT (q_fd ~unchanged
# capped/un-capped). FIX: run the operator forwards (the FD wiring check, mode_matvec/mode_quadform,
# Lanczos) in fp32 — the CALLER casts model.float() around the operator legs only; the ppl anchor +
# behavioural leg A stay bf16. Self-calibrated on PG2 (firewall). Honest VOID if fp32 also
# underflows (h_analytic≈0 vs h_fd≠0) — the pre-authorized final outcome.
# ---------------------------------------------------------------------------
def _first_order_diag(model, layers, L, d, dev, probe, v_unit, eps, precision):
    """First/second-order connectivity localization at site L, banked at the `precision` label the
    CALLER set (model dtype). Injects coef·v_unit via the operator's holder-injector. Banks
    g_analytic=∂loss/∂coef vs g_fd, coef_grad_none, h_analytic=∂²loss/∂coef² vs h_fd,
    delta_leaf_grad_norm. Reading: h_analytic≈0 while h_fd≠0 ⇒ 2nd-order underflow (bf16); in fp32
    they should MATCH (rel < 0.01) — the certification the coordinator asked to verify."""
    import torch
    holder = probe["holder"]
    T = probe["T"]
    V = v_unit.to(dev).view(1, -1).expand(T, d).contiguous()      # (T,d) per-token unit direction

    def ce(logits):
        lab = probe["labels"].to(logits.device)
        live = lab != -100
        lse = torch.logsumexp(logits[live].float(), dim=-1)
        nll = lse - logits[live].float().gather(1, lab[live].unsqueeze(1)).squeeze(1)
        return (nll * probe["loss_w"].to(logits.device)[live]).sum()

    def loss_at(coef_val, grad):
        coef = torch.tensor(float(coef_val), device=dev, dtype=torch.float32, requires_grad=grad)
        holder["delta"] = coef * V
        if grad:
            with torch.enable_grad():
                return ce(probe["forward_fn"]()[0]), coef
        with torch.no_grad():
            return ce(probe["forward_fn"]()[0]).detach(), coef

    l0, c0 = loss_at(0.0, True)
    (g,) = torch.autograd.grad(l0, c0, create_graph=True, allow_unused=True)
    g_analytic = None if g is None else float(g)
    h_analytic = None
    if g is not None:
        (h,) = torch.autograd.grad(g, c0, allow_unused=True)
        h_analytic = None if h is None else float(h)
    lp = float(loss_at(+eps, False)[0])
    lm = float(loss_at(-eps, False)[0])
    l0v = float(loss_at(0.0, False)[0])
    dleaf = (eps * V).detach().clone().requires_grad_(True)
    holder["delta"] = dleaf
    with torch.enable_grad():
        ld = ce(probe["forward_fn"]()[0])
    (gd,) = torch.autograd.grad(ld, dleaf, allow_unused=True)
    h_fd = (lp - 2.0 * l0v + lm) / (eps ** 2)
    h_rel = (None if h_analytic is None or abs(h_fd) < 1e-30
             else abs(h_analytic - h_fd) / abs(h_fd))
    return {"precision": precision, "hook_fires": bool(abs(lp - l0v) > 0.0), "loss_clean": l0v,
            "g_analytic": g_analytic, "g_fd": (lp - lm) / (2.0 * eps),
            "coef_grad_none": g_analytic is None, "h_analytic": h_analytic, "h_fd": h_fd,
            "h_rel_err": h_rel, "delta_leaf_grad_none": gd is None,
            "delta_leaf_grad_norm": None if gd is None else float(gd.norm())}


def label_two_regime(levels):
    """Self-calibrated two-regime label over the per-level (coherent-band, iid-band) readings.
    `levels` = [{coherent:[dmg×seeds], white:[dmg×seeds], both_saturated:bool}]. A level SEPARATES
    iff the coherent seed-band and the iid seed-band are DISJOINT with coherent > iid (bands from
    PG2's OWN seeds — the firewall). Ceiling + VOID: both-saturated levels VOID; an inverted
    majority is its own branch. Majority over the valid levels."""
    valid = [L for L in levels if not L["both_saturated"]]
    if len(valid) < 2:
        return "VOID", {"n_valid": len(valid),
                        "reason": "fewer than 2 readable levels (both regimes saturated)"}
    sep, inv = 0, 0
    for L in valid:
        clo, chi = min(L["coherent"]), max(L["coherent"])
        wlo, whi = min(L["white"]), max(L["white"])
        if clo > whi:
            sep += 1
        elif wlo > chi:
            inv += 1
    n = len(valid)
    ev = {"n_valid": n, "n_separated_coherent_high": sep, "n_inverted": inv}
    if sep > n / 2.0:
        return "H_two_regime", ev
    if inv > n / 2.0:
        return "H_inverted", ev
    return "H_flat", ev


# ===========================================================================
# Operator helpers — site rms, matvec factory, one Lanczos operator estimate (SPECTER-faithful).
# ===========================================================================
def site_rms(layers, L, item, d, dev):
    """rms of the residual ENTERING layers[L] on a clean forward of `item` (delta zeroed)."""
    import torch
    from infra.aws_pg.jobs import _specter_shared as SS
    from infra.pg.inherit_core import _split_hidden
    cap = {}

    def hook(_m, args, kwargs):
        hs, _rebuild = _split_hidden(args, kwargs)
        cap["x"] = hs[0].detach().float()
    h = layers[L].register_forward_pre_hook(hook, with_kwargs=True)
    with torch.no_grad():
        SS.clean_forward(item, d, dev)
    h.remove()
    return float(cap["x"].pow(2).mean().sqrt())


def matvec_factory(items, mode, eps_probe, dev, ss_seed=None, path="manual"):
    """H_ω v with the eps_probe conditioning scale (linear in v ⇒ cancels exactly). `path`
    selects the differentiation route (see the Gate-X wiring note)."""
    from infra.pg import specter_core as SC

    def mv(v):
        return SC.mode_matvec(items, v.to(dev) * eps_probe, mode, sign_seed=ss_seed,
                              path=path) / eps_probe
    return mv


def lanczos_estimate(items, mode, eps_probe, d, dev, ss_seed=None, path="manual"):
    """One Lanczos top-32 estimate with a single restart on non-convergence (SPECTER R-O1)."""
    from infra.pg import specter_core as SC
    res = SC.lanczos(matvec_factory(items, mode, eps_probe, dev, ss_seed, path), d, m=LANCZOS_M_X,
                     seed=17, device=dev)
    conv = bool((res["resid"][:8] <= CONV_RESID_FRAC * max(float(res["evals"][0]), 1e-30)).all())
    if not conv:
        res = SC.lanczos(matvec_factory(items, mode, eps_probe, dev, ss_seed, path), d,
                         m=2 * LANCZOS_M_X, seed=17, device=dev)
        res["restarted"] = True
        conv = bool((res["resid"][:8] <= CONV_RESID_FRAC * max(float(res["evals"][0]), 1e-30)).all())
    res["converged"] = conv
    return res


def count_stable_spikes(res_a1, res_a2, held_items, mode, eps_probe, dev, ss_seed=None,
                        path="manual"):
    """Split-half + held-out Rayleigh stable-spike count on PG2's OWN operator (intrinsic
    calibration — split-half overlap ≥ 0.7 and held-out ratio ∈ [0.5, 2], specter_core constants).
    Returns (n_stable, spikes)."""
    from infra.pg import specter_core as SC

    def ray_b(u):
        return SC.mode_quadform(held_items, u.to(dev) * eps_probe, mode,
                                sign_seed=ss_seed, path=path) / (eps_probe ** 2)
    spikes, _clusters = SC.stable_spikes(res_a1, res_a2, ray_b, k=TOPK_X)
    return len(spikes), spikes


# ===========================================================================
# The write-side (bonus) inventory — spec §0.5-4: verify PG2 write sources on our storage.
# ===========================================================================
def writeside_inventory():
    """Record what PG2 write sources exist (spec §0.5-4). The CARRY stiff-carries write-side needs
    a per-task FT ROSTER with measurable per-task gains + range-guarded swap-shell cells (the
    PG1 `google/paligemma-3b-ft-<slug>-224` corpus). Verified inventory: PG2 has NO analogous
    per-task FT roster (`google/paligemma2-3b-ft-*` does not exist); only the base
    `paligemma2-3b-pt-224` and the GENERALIST `paligemma2-3b-mix-224` full fine-tune are seeded
    (infra/aws_pg/seed_weights.sh). A single generalist mix write is NOT the CARRY per-task
    roster, and porting CARRY's dense-Q stiff subspace to PG2 exceeds this scoped operator budget.
    → the write-side row is disclosed ABSENT ⇒ VOID (spec §3.2 'runs only if adapters verified,
    else disclosed absent'). This is an honest VOID row, not a failure."""
    return {
        "pg2_base": PG2_REPO,
        "pg2_generalist_full_ft": "google/paligemma2-3b-mix-224 (seeded; NOT a per-task roster)",
        "pg2_per_task_ft_roster": None,
        "roster_pattern_checked": "google/paligemma2-3b-ft-<slug>-224 (does not exist)",
        "verdict": "disclosed-absent",
        "void": True,
        "reading": "no PG2 per-task FT roster with measurable per-task gains exists; the "
                   "write-side stiff/flat carry cell is disclosed ABSENT and the law row is VOID "
                   "(spec §3.2/§9.4). mix-224 remains a candidate write source for a successor "
                   "once a stiff/flat swap-shell port + range guard are verified.",
    }


# ===========================================================================
# Main.
# ===========================================================================
def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")     # reference_hf_xet_wedge
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--losses", default="ppl,vqa", help="operator-pass losses (spec: both)")
    ap.add_argument("--op-path", default="auto", choices=("auto", "manual", "func"),
                    help="operator differentiation route; auto certifies func then manual at Gate-X")
    ap.add_argument("--n-op-ppl", type=int, default=N_OP_PPL_X)
    ap.add_argument("--n-op-vqa", type=int, default=N_OP_VQA_X)
    ap.add_argument("--n-held", type=int, default=N_OP_HELD_X)
    ap.add_argument("--n-ppl-seq", type=int, default=60)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    losses = [s.strip() for s in a.losses.split(",") if s.strip()]
    assert losses and set(losses) <= {"ppl", "vqa"}, f"bad --losses {a.losses}"

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()                                    # BEFORE transformers/datasets (pg rule)
    import torch

    from infra.aws_pg.jobs import _handle_shared as HS
    from infra.aws_pg.jobs import _specter_shared as SS
    from infra.pg import atlas_core as AC
    from infra.pg import handle_core as HC
    from infra.pg import specter_core as SC

    # Freeze-or-refuse (HANDLE's own tag + spec sha256). This arm is FULLY INDEPENDENT of the
    # PG1 arms (spec §9.4) — its own freeze entry, never blocks them.
    freeze = HS.assert_frozen("x")
    HS.install_pg()
    dev = "cuda"
    assert torch.cuda.is_available(), "no CUDA — Arm X refuses to run"

    rep = {"object": "handle_x_v1", "freeze": freeze, "model": PG2_REPO,
           "firewall": "verdicts-only; every PG2 threshold PG2-self-calibrated or architectural; "
                       "pg1 field is a categorical LABEL from frozen PG1 result docs, used ONLY in "
                       "the final pg1==pg2 equality; no LATLOC/LATABLATE deletion contrasts "
                       "(cross-lineage Gemma-2 firewall).",
           "pg1_labels": {k: {"label": v[0], "provenance": v[1]} for k, v in PG1_LABELS.items()},
           "legs": {}}
    tm = HS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    wd = "/tmp/handle_x"
    pathlib.Path(wd).mkdir(parents=True, exist_ok=True)

    # ---- GATE-X: anchors FIRST. Any leg whose PG2 anchor fails ⇒ honest VOID row. --------------
    with tm.stage("gate_x_load"):
        # Load the S3-cached snapshot (sha=None) and record the ACTUALLY-loaded commit hash from
        # the model config — no network HfApi call (the re-run stages weights from S3; the sha is
        # a provenance record, not load-bearing).
        model, proc, layers, arch = load_pg2(device=dev)
        sha = getattr(model.config, "_commit_hash", None)
        d = arch["hidden_size"]
        depths = three_depths(arch["num_hidden_layers"])
        rep["model_sha"] = sha
        rep["pg2_arch"] = arch
        rep["depths"] = depths
        rep["embedding_output_site"] = depths[0]
    _w()

    # ppl anchor on PG2 — its OWN reference (NEVER compared to PG1's 22.43). VOID-all if it fails.
    with tm.stage("gate_x_ppl_anchor"):
        base = HS.safe_ppl_gate(model, proc, dev, n_seq=a.n_ppl_seq)
        base_ppl = base["ppl"]
        rep["base_ppl_anchor"] = {"ppl": base_ppl, "mean_nll": base["mean_nll"],
                                  "saturated": base["saturated"], "n_tokens": base["n_tokens"]}
        anchor_ok = bool(torch.isfinite(torch.tensor(base["mean_nll"])) and not base["saturated"])
        rep["anchor_ok"] = anchor_ok
        assert anchor_ok, "PG2 ppl anchor non-finite or saturated at baseline — Gate-X halt"
    _w()

    # operator batch A/B (ppl) — B=1 forwards, manifests content-hashed (spec §4.1 reuse).
    with tm.stage("gate_x_items"):
        wins_a = SS.wikitext_windows(proc, a.n_op_ppl, skip=0)
        wins_b = SS.wikitext_windows(proc, a.n_held, skip=a.n_op_ppl)
        rep["manifests"] = {"ppl_A": SS.batch_manifest([w["hash"] for w in wins_a]),
                            "ppl_B": SS.batch_manifest([w["hash"] for w in wins_b])}
    _w()

    # Gate-X finite-difference WIRING check at the embedding-output site. FD_REL_TOL is an
    # ARCHITECTURAL tolerance (model-independent), not a PG1 number. The operator items force
    # use_cache=False (the PG2 wiring fix — see _op_ppl_item). We certify the jvp/vjp identity on
    # PG2 by trying the forward-mode (func, faster) and reverse-mode (manual, robust) routes and
    # using whichever lands rel_err < tol (prefer func). If NEITHER certifies on cache-free items,
    # that is evidence of a genuine PG2 incompatibility ⇒ the operator legs VOID honestly. The
    # damage leg (A) is operator-free and runs regardless.
    L_e = depths[0]
    paths = ["func", "manual"] if a.op_path == "auto" else [a.op_path]

    def certify(probe, delta):
        """FD wiring check over the requested autodiff route(s) at the model's CURRENT dtype."""
        cand = {}
        for p in paths:
            q, q_fd, rel = SC.fd_directional_check(probe, delta, path=p)
            cand[p] = {"q": q, "q_fd": q_fd, "rel_err": rel}
            print(f"[handle-x gate] fd path={p} ({next(model.parameters()).dtype}): "
                  f"q={q:.4g} q_fd={q_fd:.4g} rel={rel:.4g}", flush=True)
        chosen = next((p for p in ("func", "manual")
                       if p in cand and cand[p]["rel_err"] <= FD_REL_TOL), None)
        return cand, chosen

    def probe_delta():
        """A fresh operator probe + dc direction at the model's CURRENT dtype (pixel_values follow
        the model dtype via the item builder). Fixed direction seed 17."""
        inst, hold = SC.make_residual_injector(layers, L_e)
        pb = _op_ppl_item(model, proc, dev, wins_a[0], inst, hold)
        rms = site_rms(layers, L_e, pb, d, dev)
        eps = EPS_PROBE_FRAC * rms
        g = torch.Generator(device="cpu").manual_seed(17)
        vu = torch.randn(d, generator=g)
        vu = (vu / vu.norm()).to(dev)                              # deterministic unit direction
        c = SC.env_coeffs("dc", pb["T"], device=dev)
        return pb, c.unsqueeze(1) * (vu * eps).unsqueeze(0), vu, eps, rms

    with tm.stage("gate_x_fd"):
        # ---- bf16 attempt (as PG1 did) ----
        probe, delta, vu, eps_e, rms_e = probe_delta()
        cand, op_path = certify(probe, delta)
        operator_ok = op_path is not None
        operator_fp32 = False
        rep["fd_wiring"] = {"candidates_bf16": cand, "chosen_path": op_path, "tol": FD_REL_TOL,
                            "use_cache": False, "site_rms_L%d" % L_e: rms_e}
        rep["fd_wiring"]["first_order_bf16"] = _first_order_diag(
            model, layers, L_e, d, dev, probe, vu, eps_e, "bf16")

        # ---- fp32 attempt (the bf16 2nd-order double-backprop underflow fix) ----
        # Cast the model to fp32 ONLY for the operator certification; rebuild the probe so its
        # pixel_values match fp32; certify; bank the fp32 first/second-order diagnostic; restore
        # bf16 for the ppl anchor / leg A. If fp32 certifies, the operator legs run in fp32.
        if not operator_ok:
            model.float()
            probe32, delta32, vu32, eps32, rms32 = probe_delta()
            cand32, op_path32 = certify(probe32, delta32)
            rep["fd_wiring"]["candidates_fp32"] = cand32
            rep["fd_wiring"]["first_order_fp32"] = _first_order_diag(
                model, layers, L_e, d, dev, probe32, vu32, eps32, "fp32")
            model.to(torch.bfloat16)
            if op_path32 is not None:
                operator_ok, op_path, operator_fp32 = True, op_path32, True
                rep["fd_wiring"]["chosen_path"] = op_path

        rep["fd_wiring"]["operator_ok"] = operator_ok
        rep["fd_wiring"]["operator_fp32"] = operator_fp32
        rep["operator_scoping"] = {
            "fp32": operator_fp32,
            "note": ("operator legs (B/D/C) measured in fp32 — the bf16 double-backprop 2nd-order "
                     "underflow fix (the analytic vᵀHv reads ≈0 in bf16); the ppl anchor + "
                     "behavioural leg A stay bf16. Self-calibrated on PG2."
                     if operator_fp32 else
                     "operator measured in bf16 (certified as-is).")}
    _w()

    # =======================================================================================
    # LEG A — two-regime energy law ACROSS LEVELS (behavioural damage cells; operator-free).
    #   coherent (one shared direction) vs iid (fresh per token), matched per-token energy
    #   eps = TR_EPS_FRAC·site_rms, at each of the 3 depths, N_SEED_TR seeds each. The label is
    #   self-calibrated on PG2's OWN seed bands (the firewall). ~= (2 field types × 3 depths ×
    #   3 seeds) + baselines ≈ 20-cell battery.
    # =======================================================================================
    from infra.pg import bispec_core as BC
    with tm.stage("legA_two_regime"):
        levelA = []
        for L in depths:
            item0 = SS.build_ppl_item(model, proc, dev, wins_a[0],
                                      *SC.make_residual_injector(layers, L))
            rms_L = site_rms(layers, L, item0, d, dev)
            eps = TR_EPS_FRAC * rms_L
            coh, wht, sat_c, sat_w = [], [], [], []
            for s in range(N_SEED_TR):
                gseed = int(hashlib.sha1(f"handle_x|coh|{L}|{s}".encode()).hexdigest()[:8], 16)
                g = torch.Generator(device="cpu").manual_seed(gseed)
                u = torch.randn(d, generator=g)
                u = u / u.norm()
                with BC.hook_at(layers, L, coherent_hook(u, eps)):
                    pg = HS.safe_ppl_gate(model, proc, dev, n_seq=a.n_ppl_seq)
                coh.append(HS.safe_log1p_dppl(pg["mean_nll"], base_ppl)); sat_c.append(pg["saturated"])
                wseed = int(hashlib.sha1(f"handle_x|iid|{L}|{s}".encode()).hexdigest()[:8], 16)
                with BC.hook_at(layers, L, white_hook(wseed, eps)):
                    pw = HS.safe_ppl_gate(model, proc, dev, n_seq=a.n_ppl_seq)
                wht.append(HS.safe_log1p_dppl(pw["mean_nll"], base_ppl)); sat_w.append(pw["saturated"])
            both_sat = all(sat_c) and all(sat_w)
            levelA.append({"L": L, "site_rms": rms_L, "eps_abs": eps, "coherent": coh,
                           "white": wht, "coherent_saturated": sat_c, "white_saturated": sat_w,
                           "both_saturated": both_sat})
            print(f"[handle-x A] L{L}: coherent {[round(x,3) for x in coh]} "
                  f"iid {[round(x,3) for x in wht]}", flush=True)
        labelA, evA = label_two_regime(levelA)
        rep["legs"]["two_regime_energy"] = {
            "pg2_label": labelA, "evidence": evA, "levels": levelA, "void": labelA == "VOID",
            "self_calibration": "coherent vs iid seed bands computed on PG2; disjoint-band test; "
                                "no PG1 number used."}
    _w()

    # =======================================================================================
    # OPERATOR PASS (SPECTER, both losses) — feeds LEG B (ordering) and LEG D (spike census).
    # If the Gate-X FD wiring failed, all operator legs VOID honestly.
    # =======================================================================================
    op = {}                                    # {(depth, mode, loss): {res_a1,res_a2,n_stable,...}}
    if operator_ok:
        # run the whole operator pass in fp32 iff that is what certified (the bf16 double-backprop
        # 2nd-order underflow fix); restored to bf16 right after leg C. Operator items are built
        # AFTER the cast so their pixel_values follow the fp32 model dtype. leg A already ran bf16.
        if operator_fp32:
            model.float()
        # ---- per-depth ppl items (batch A split A1/A2 + held-out B). vqa built once at L_e. ----
        with tm.stage("operator_items"):
            vqa_task = None
            vqa_rows = None
            if "vqa" in losses:
                from infra.pg import inherit_pg1 as IP
                vqa_task, _coco = IP.pg1_tasks()
                # A (n_op_vqa) + a DISJOINT held-out B (n_held) — B must not overlap A.
                vqa_rows, _h = IP.load_items(vqa_task, a.n_op_vqa + a.n_held)
        _w()

        def op_pass(L, mode, loss):
            """One (depth, mode, loss) operator estimate: A1/A2 Lanczos + held-out stable-spike
            count. Reuses SPECTER machinery verbatim on PG2. ppl uses wikitext items; vqa uses
            teacher-forced items (built at L_e only, per the spec's embedding-output operator)."""
            inst, hold = SC.make_residual_injector(layers, L)
            if loss == "ppl":
                A = [_op_ppl_item(model, proc, dev, w, inst, hold) for w in wins_a]
                B = [_op_ppl_item(model, proc, dev, w, inst, hold) for w in wins_b]
            else:
                A = [_op_vqa_item(model, proc, dev, vqa_rows[i], vqa_task, inst, hold)
                     for i in range(a.n_op_vqa)]
                B = [_op_vqa_item(model, proc, dev, vqa_rows[i], vqa_task, inst, hold)
                     for i in range(a.n_op_vqa, a.n_op_vqa + a.n_held)]
            rms_L = site_rms(layers, L, A[0], d, dev)
            eps_probe = EPS_PROBE_FRAC * rms_L
            A1, A2 = A[:len(A) // 2], A[len(A) // 2:]
            ss = WHITE_SIGN_SEED if mode == "white" else None
            r1 = lanczos_estimate(A1, mode, eps_probe, d, dev, ss_seed=ss, path=op_path)
            r2 = lanczos_estimate(A2, mode, eps_probe, d, dev, ss_seed=ss, path=op_path)
            n_stable, spikes = count_stable_spikes(r1, r2, B, mode, eps_probe, dev, ss_seed=ss,
                                                   path=op_path)
            rec = {"lambda1_a1": float(r1["evals"][0]), "lambda1_a2": float(r2["evals"][0]),
                   "converged": bool(r1["converged"] and r2["converged"]),
                   "n_stable": int(n_stable), "site_rms": rms_L,
                   "top4_a1": [float(x) for x in r1["evals"][:4]]}
            print(f"[handle-x op] L{L}/{mode}/{loss}: lam1={rec['lambda1_a1']:.3e} "
                  f"n_stable={n_stable} conv={rec['converged']}", flush=True)
            return rec, r1, r2

        # LEG D: mid-stack spike depletion — dc-ppl at all 3 depths.
        with tm.stage("legD_spike_census"):
            spike_by_depth = {}
            depth_recs = {}
            for L in depths:
                rec, _r1, _r2 = op_pass(L, "dc", "ppl")
                depth_recs[L] = rec
                spike_by_depth[L] = rec["n_stable"] if rec["converged"] else None
                op[(L, "dc", "ppl")] = rec
            rep["legs"]["midstack_spike_depletion"] = _label_spike_depletion(depths, spike_by_depth,
                                                                             depth_recs)
        _w()

        # LEG B: within-level operator ordering — coherent (dc) vs incoherent (white), at the
        # embedding-output site, both losses. Label from ppl (primary); vqa banked collateral
        # (SPECTER found the vqa operator spike-free = H_field).
        with tm.stage("legB_operator_ordering"):
            order = {}
            for loss in losses:
                dc = op.get((L_e, "dc", loss)) or op_pass(L_e, "dc", loss)[0]
                op[(L_e, "dc", loss)] = dc
                wh, _r1, _r2 = op_pass(L_e, "white", loss)
                op[(L_e, "white", loss)] = wh
                order[loss] = {"dc_n_stable": dc["n_stable"], "white_n_stable": wh["n_stable"],
                               "dc_lambda1": dc["lambda1_a1"], "white_lambda1": wh["lambda1_a1"],
                               "dc_converged": dc["converged"], "white_converged": wh["converged"]}
            rep["legs"]["operator_ordering"] = _label_operator_ordering(order, primary="ppl")
        _w()

        # LEG C: kernel theorems — the radial gauge + the counter-rotated post-RoPE-K coherent
        # slice, both at the embedding-output site, self-calibrated against PG2's OWN Haar bulk.
        with tm.stage("legC_kernel_theorems"):
            rep["legs"]["kernel_theorems"] = kernel_theorems(model, proc, layers, L_e, d, arch,
                                                             wins_a[:min(len(wins_a), 8)], dev,
                                                             op_path)
        _w()
        if operator_fp32:
            model.to(torch.bfloat16)            # restore bf16
    else:
        # honest VOID rows for the operator legs — Gate-X certified in NEITHER bf16 NOR fp32.
        # Evidence banked: fd_wiring candidates (bf16 + fp32) and the first/second-order diagnostic
        # (h_analytic vs h_fd). fp32 h_analytic STILL ≈0 vs h_fd≠0 ⇒ the SPECTER analytic
        # local-quadratic operator does not transfer to PG2 even at full precision — the
        # pre-authorized final outcome (spec R-W2 / §9.4). leg A's lands-elsewhere stands.
        void_reading = (
            "Gate-X operator wiring did not certify on PG2 in bf16 "
            f"({rep['fd_wiring']['candidates_bf16']}) OR fp32 "
            f"({rep['fd_wiring'].get('candidates_fp32', 'n/a')}); tol {FD_REL_TOL}. "
            f"2nd-order diag bf16={rep['fd_wiring'].get('first_order_bf16', {})} "
            f"fp32={rep['fd_wiring'].get('first_order_fp32', {})}. The SPECTER analytic operator "
            "does not transfer to PG2 even at fp32 ⇒ operator legs VOID (honest, richly diagnosed).")
        for law in ("midstack_spike_depletion", "operator_ordering", "kernel_theorems"):
            rep["legs"][law] = {"pg2_label": None, "void": True, "reading": void_reading}
        _w()

    # =======================================================================================
    # LEG E (bonus) — write-side stiff/flat: inventory + honest VOID (no PG2 per-task FT roster).
    # =======================================================================================
    with tm.stage("legE_writeside"):
        inv = writeside_inventory()
        rep["legs"]["stiff_carries_writeside"] = {"pg2_label": None, "void": inv["void"],
                                                  "inventory": inv, "reading": inv["reading"]}
    _w()

    # =======================================================================================
    # FEED adjudicate_rx (NOT redefined) — the per-law replication scoreboard. The pg1 field is
    # the categorical LABEL only; pg2 is the PG2-self-calibrated label; void short-circuits.
    # =======================================================================================
    law_verdicts = {}
    for law, (pg1_label, _prov) in PG1_LABELS.items():
        leg = rep["legs"].get(law, {"void": True, "pg2_label": None})
        law_verdicts[law] = {"pg1": pg1_label, "pg2": leg.get("pg2_label"),
                             "void": bool(leg.get("void", False))}
    rep["law_verdicts"] = law_verdicts
    rep["rx"] = HC.adjudicate_rx(law_verdicts)

    HS.assert_gpu_seconds(rep)
    rep["green"] = True                              # last statement (status flags go last)
    _w()
    print("HANDLE_X_OK", json.dumps({"per_law": rep["rx"]["per_law"],
                                     "n_lands_elsewhere": rep["rx"]["n_lands_elsewhere"]}))


# ===========================================================================
# Leg labellers (self-calibrated; each names a VOID branch and reachable alternatives).
# ===========================================================================
def _label_spike_depletion(depths, spike_by_depth, depth_recs):
    """LEG D label. `spike_by_depth[L]` = stable-spike count (None if non-converged). Mid-depleted
    iff the MID depth carries strictly fewer stable spikes than at least one END while an end has
    ≥1 (spikes live at the stack's ends). Self-calibrated: spike stability is intrinsic to PG2's
    operator (split-half overlap + held-out Rayleigh). VOID if the mid or both ends non-converged."""
    Le, Lm, Ll = depths
    if spike_by_depth[Lm] is None or (spike_by_depth[Le] is None and spike_by_depth[Ll] is None):
        return {"pg2_label": None, "void": True, "counts": spike_by_depth,
                "reading": "operator non-converged at the mid or both end depths — VOID"}
    mid = spike_by_depth[Lm]
    ends = [spike_by_depth[Le], spike_by_depth[Ll]]
    ends_ok = [e for e in ends if e is not None]
    max_end = max(ends_ok) if ends_ok else 0
    if max_end >= 1 and mid < max_end:
        label = "H_mid-depleted"
    elif max_end == 0 and mid == 0:
        label = "H_no-spikes"
    elif mid >= max_end and mid >= 1:
        label = "H_uniform"           # spikes not depleted mid-stack
    else:
        label = "H_flat-ends"          # ends flat, mid carries — the inverse pattern
    return {"pg2_label": label, "void": False, "counts": spike_by_depth,
            "detail": {str(L): depth_recs[L] for L in depths},
            "self_calibration": "split-half overlap + held-out Rayleigh band on PG2's own "
                                "operator; spatial pattern only; no PG1 number.",
            "reading": f"stable-spike counts by depth {spike_by_depth}; label {label}."}


def _label_operator_ordering(order, primary="ppl"):
    """LEG B label. Coherent-ordered iff the coherent (dc) operator carries ≥1 stable spike AND
    strictly MORE than the incoherent (white) operator at the embedding-output site (both on
    PG2's own split-half-stable operator — the firewall). Inverted / flat are reachable. VOID if
    the primary loss non-converged."""
    o = order.get(primary)
    if o is None or not (o["dc_converged"] and o["white_converged"]):
        return {"pg2_label": None, "void": True, "orders": order,
                "reading": f"primary-loss ({primary}) operator non-converged — VOID"}
    dc, wh = o["dc_n_stable"], o["white_n_stable"]
    if dc >= 1 and dc > wh:
        label = "coherent_ordered"
    elif wh >= 1 and wh > dc:
        label = "incoherent_ordered"
    else:
        label = "flat_ordering"        # neither mode leads (both 0, or tied)
    return {"pg2_label": label, "void": False, "orders": order, "primary_loss": primary,
            "self_calibration": "dc vs white stable-spike counts on PG2's own operator; "
                                "split-half stability intrinsic; no PG1 number.",
            "reading": f"[{primary}] dc n_stable={dc} vs white n_stable={wh}; label {label}."}


def kernel_theorems(model, proc, layers, L_e, d, arch, wins, dev, op_path="manual"):
    """LEG C — the exact-architecture kernel theorems on PG2, self-calibrated vs PG2's OWN Haar
    bulk. Two checks at the embedding-output site:

      (1) RADIAL GAUGE (RMSNorm annihilates x̂): radial residual field δ_t = eps·x̂_t vs a Haar
          tangential bulk. `adjudicate_ro2_radial` — thresholds (0.05 / 0.25) are ARCHITECTURAL
          (the RMSNorm Jacobian kills x̂ exactly), not PG1 numbers.
      (2) KEY GAUGE (a coherent post-RoPE-K shift moves no attention pattern): the counter-rotated
          post-RoPE-K coherent slice vs a Haar-dc bulk at k_proj, replicated across GQA kv-heads.
          On Gemma-2 this theorem is the a-priori BREAK-SUSPECT: attn_logit_softcapping applies
          tanh to the logits, and a per-query-constant key shift does NOT commute with tanh — so
          a large ratio here is a GENUINE scope boundary (lands-elsewhere), NOT a wiring halt.
          Hence we map the ratio to a LABEL and do NOT fail-loud on PG2 (on PG1 the same reading
          is a halt because the theorem is exact there — the firewall-aware interpretation split).

    Bulk references are measured on PG2's own operator ⇒ every threshold is architectural or
    PG2-self-calibrated. Label 'H_kernels-hold' iff BOTH land where the theorems require."""
    import torch

    from infra.aws_pg.jobs import _specter_shared as SS
    from infra.pg import atlas_core as AC
    from infra.pg import specter_core as SC
    from infra.pg.inherit_core import _split_hidden

    # ---- (1) radial gauge at the residual site (use_cache=False operator items — the PG2 fix) ----
    inst, hold = SC.make_residual_injector(layers, L_e)
    A = [_op_ppl_item(model, proc, dev, w, inst, hold) for w in wins]
    rms_L = site_rms(layers, L_e, A[0], d, dev)
    eps = EPS_PROBE_FRAC * rms_L
    # capture the clean residual per item (zeroed delta) → per-token x̂ radial fields. On the
    # FIRST item also capture the k_proj OUTPUT (for the key-gauge conditioning scale) — done in
    # the SAME residual-injector forward so the holder width stays correct (d), never the wrong
    # k_proj width.
    xcache = {}
    kcap = {}

    def _cap(i):
        def hook(_m, args, kwargs):
            hs, _rb = _split_hidden(args, kwargs)
            xcache[i] = hs[0].detach().float()
        return hook

    def _kcap(_m, _a, o):
        kcap["o"] = o.detach().float()
    for i, itx in enumerate(A):
        h = layers[L_e].register_forward_pre_hook(_cap(i), with_kwargs=True)
        hk = layers[L_e].self_attn.k_proj.register_forward_hook(_kcap) if i == 0 else None
        with torch.no_grad():
            SS.clean_forward(itx, d, dev)
        h.remove()
        if hk is not None:
            hk.remove()
    rad_fields = [eps * xcache[i] / xcache[i].norm(dim=-1, keepdim=True).clamp_min(1e-12)
                  for i in range(len(A))]
    q_rad = SC.field_quadform(A, rad_fields, path=op_path) / (eps ** 2)
    gen = torch.Generator(device="cpu").manual_seed(510)
    q_tang = []
    for _s in range(N_BULK_DRAWS):
        g = torch.randn(d, generator=gen)
        u = (g / g.norm()).to(dev)
        tf = []
        for i in range(len(A)):
            x = xcache[i]
            xh = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            t = u.unsqueeze(0) - (xh @ u).unsqueeze(1) * xh          # project off x̂ (tangential)
            tf.append(eps * t / t.norm(dim=-1, keepdim=True).clamp_min(1e-12))
        q_tang.append(SC.field_quadform(A, tf, path=op_path) / (eps ** 2))
    bulk_tang = sum(q_tang) / len(q_tang)
    radial = dict(SC.adjudicate_ro2_radial(mean_rayleigh=q_rad, bulk_mean=bulk_tang),
                  q_radial=q_rad, bulk_tangential=bulk_tang)

    # ---- (2) key gauge (post-RoPE coherent K slice) at k_proj, GQA-aware ----
    hd = arch["head_dim"]
    n_kv = arch["num_key_value_heads"]
    theta = arch["rope_theta"]
    inst_k, hold_k = SC.make_proj_injector(layers[L_e], "k_proj")
    items_k = [_op_ppl_item(model, proc, dev, w, inst_k, hold_k) for w in wins]
    # k_proj-output rms (captured above on the residual-injector forward) → the coherent
    # conditioning scale (cancels in the linear form).
    kproj_width = int(kcap["o"].shape[-1])
    assert kproj_width == n_kv * hd, f"k_proj width {kproj_width} != n_kv*head_dim {n_kv*hd}"
    crms = float(kcap["o"].pow(2).mean().sqrt())
    epsk = 0.05 * crms * (hd ** 0.5)
    gk = torch.Generator(device="cpu").manual_seed(520)
    q_key, q_bulk = [], []
    for _s in range(N_BULK_DRAWS // 2):
        g = torch.randn(hd, generator=gk, dtype=torch.float64)
        u = g / g.norm()
        # post-RoPE coherent shift on ONE head, replicated across all GQA kv-heads (matched energy).
        fn = AC.delta_multihead(AC.delta_postrope_coherent(u * epsk, base=theta), n_kv, hd)
        fields = [fn(torch.arange(itk["T"]), dev) for itk in items_k]
        q_key.append(SC.field_quadform(items_k, fields, path=op_path) / (epsk ** 2))
        gb = torch.randn(kproj_width, generator=gk, dtype=torch.float64)
        ub = (gb / gb.norm()) * epsk
        fb = [ub.float().to(dev).unsqueeze(0).expand(itk["T"], kproj_width).contiguous()
              for itk in items_k]
        q_bulk.append(SC.field_quadform(items_k, fb, path=op_path) / (epsk ** 2))
    mean_key = sum(q_key) / len(q_key)
    mean_bulk = sum(q_bulk) / len(q_bulk)
    r_key = mean_key / mean_bulk if mean_bulk > 0 else float("inf")
    # Firewall-aware mapping: architectural tolerances (theorem is exact ⇒ ~0), NO fail-loud on
    # PG2 (a break is a scope boundary, not a wiring bug).
    if r_key <= 1e-3:
        key_label = "theorem-verified"
    elif r_key < 0.1:
        key_label = "measured-leak"
    else:
        key_label = "theorem-breaks"
    key = {"ratio": r_key, "mean_key": mean_key, "mean_bulk": mean_bulk, "label": key_label,
           "note": ("attn_logit_softcapping=%s — the a-priori theorem-breaker on Gemma-2"
                    % arch["attn_logit_softcapping"])}

    kernels_hold = (radial["verdict"] in ("H_gauge-confirmed", "H_skip-share-measured")
                    and key_label in ("theorem-verified", "measured-leak"))
    label = "H_kernels-hold" if kernels_hold else "H_kernels-break"
    return {"pg2_label": label, "void": False, "radial": radial, "key_coherent": key,
            "self_calibration": "radial/key Rayleigh both normalised by PG2's own Haar bulk; "
                                "thresholds are architectural (RMSNorm/RoPE exactness), never "
                                "PG1 numbers.",
            "reading": f"radial={radial['verdict']} (ratio {radial['ratio']:.3g}); "
                       f"key={key_label} (ratio {r_key:.3g}); label {label}."}


if __name__ == "__main__":
    main()
