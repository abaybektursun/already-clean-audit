"""RATION shared job glue — model+layer load, the `ration|` split, the Q-bank load, the Σ_x capture
at L0, the per-policy behavior harness (VQA per-item + ppl per-window + reconstruction MSE, all under
the substitutor), the oracle band leave-one-out, and the freeze mechanics.

Everything measured (Q, Σ_x, behavior, MSE) reuses verified surfaces (`inherit_pg1`, `_specter_shared`,
`_regrain_shared`, `carry_scope`); the NEW logic is entirely in `ration_core`. No try/except hiding —
a wrong assumption RAISES (CLAUDE.md). Journal amendments A-RATION-1..4 govern the choices here:
  * L* = L0 (the only banked/matched susceptibility operator).
  * behavior = wired PG1 VQA harness (eager, hooked, per-item) + blank-image wikitext ppl.
  * Σ_x from the blank-image ppl windows (matched to the ppl-mode Q); ONE calibration applied to both
    readouts — the ppl leg is modality-matched, the VQA leg tests transfer of the ppl-calibrated map.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import pathlib

from infra.aws_pg.jobs._regrain_shared import S3_BUCKET, fetch, pinned_pg1_sha

# ---- S3 pointers ---------------------------------------------------------------------------------
CARRY_SCOPE = f"{S3_BUCKET}/results/carry_scope/out"
RATION_DISC = f"{S3_BUCKET}/results/ration_disc/out"

# ---- frozen constants (spec §3/§4; journal) ------------------------------------------------------
L_STAR = 0                         # A-RATION-1: primary intervention layer = L0 (matched Q)
D = 2048                           # PG1 residual width (inherit_pg1.D)
N_PPL = 320                        # wikitext windows total (split 50/50 → 160/half)
N_VQA = 1400                       # vqav2 items total (split 50/50 → ~700/half)
N_VQA_CELL = 480                   # items per VQA sweep cell (bounded generation cost)
STIFF_METRIC_KEY = "L0/dc/ppl"


# ===========================================================================
# Model, split, Q bank.
# ===========================================================================
def load_model():
    """PG1 pt-224, eager, bf16; the 18 decoder layers; the pinned sha + FUNC base_ppl anchor."""
    from infra.pg import inherit_pg1 as IP
    sha, base_ppl = pinned_pg1_sha("/tmp/ration")
    model, proc = IP.load_pg1(sha=sha)
    layers = IP.decoder_layers(model)
    assert len(layers) == 18, f"expected 18 PG1 layers, got {len(layers)}"
    return model, proc, layers, sha, base_ppl


def ration_split(hashes, name="ration"):
    """sha1 content-hash bucketing (the comwrite/carry convention, re-salted). held=True → the SEALED
    confirmation half (odd bucket); held=False → discovery. Both halves asserted non-empty."""
    import numpy as np
    held = np.array([int(hashlib.sha1(f"{name}|{h}".encode()).hexdigest(), 16) % 2 == 1
                     for h in hashes], dtype=bool)
    assert held.any() and (~held).any(), f"degenerate split for {name!r}"
    return held


def load_q_bank(workdir):
    """carry_scope Q bank: evals (d,) fp64 desc, evecs (d,d) fp32, k_stiff, metric_key. Loaded exactly
    as CONFLUX loaded it (A-RATION-4)."""
    import torch
    qb = torch.load(fetch(f"{CARRY_SCOPE}/carry_q_bank.pt", pathlib.Path(workdir) / "carry_q_bank.pt"),
                    map_location="cpu", weights_only=False)
    for k in ("evals", "evecs", "k_stiff", "metric_key"):
        assert k in qb, f"q bank missing {k!r}"
    assert qb["metric_key"] == STIFF_METRIC_KEY, (qb["metric_key"], STIFF_METRIC_KEY)
    return qb


# ===========================================================================
# Data splits (ppl windows + VQA items), aligned by content hash.
# ===========================================================================
def _window_locatable(proc, w):
    """True if build_ppl_item can anchor this window's text block after the image prefix. A few
    wikitext windows tokenize differently in-context (PaliGemma boundary-merge quirk,
    reference_paligemma_text_trailing_newline) and are not locatable — a data-quality drop, not
    error-hiding (build_ppl_item would fail-loud on them). Non-asserting rightmost block search."""
    import torch
    inp = proc(text=[w["text"]], images=[w["blank"]], return_tensors="pt", padding=False)
    hay = inp["input_ids"][0].cpu()
    win = w["win_ids"]
    n = int(win.shape[0])
    for i in range(hay.shape[0] - n, -1, -1):
        if torch.equal(hay[i:i + n], win):
            return True
    return False


def wikitext_split(proc):
    """N_PPL blank-image wikitext windows, filtered to build_ppl_item-locatable ones, then split by
    hash into (discovery, confirmation). Filtering BEFORE the split keeps ppl_disc identical to the
    frozen-banks run (disc was all-locatable); it only drops malformed windows from the conf half."""
    from infra.aws_pg.jobs._specter_shared import wikitext_windows
    wins = [w for w in wikitext_windows(proc, N_PPL) if _window_locatable(proc, w)]
    held = ration_split([w["hash"] for w in wins], name="ration_ppl")
    disc = [w for w, h in zip(wins, held) if not h]
    conf = [w for w, h in zip(wins, held) if h]
    return disc, conf


def vqa_task_and_split():
    """The wired PG1 vqav2 task + its rows split by content hash into (discovery, confirmation)."""
    from infra.pg import inherit_pg1 as IP
    task, _coco = IP.pg1_tasks()
    rows, hashes = IP.load_items(task, N_VQA)
    held = ration_split(hashes, name="ration_vqa")
    disc = [r for r, h in zip(rows, held) if not h]
    conf = [r for r, h in zip(rows, held) if h]
    return task, disc, conf


# ===========================================================================
# Σ_x capture at L0 (blank-image ppl windows) → banks.
# ===========================================================================
N_SIGMA_VQA = 256                  # VQA items mixed into the Σ_x calibration (real-image ranges)


def accumulate_sigma(model, proc, layers, windows, vqa_rows=None, task=None, L=L_STAR, device="cuda"):
    """(gram=Σxxᵀ, sum_x, absmean_sum, n) of the entering residual at layers[L], fp64 on device then
    to CPU. Calibrated on a MIX of blank-image ppl windows AND (if given) real-image VQA items so the
    ±K_SIGMA·σ quantiser ranges fit BOTH modalities — the text-relevant ranges (which drive ppl) are
    preserved; the image-relevant ranges widen to fit VQA, removing the blank-vs-real clip confound.
    Standard representative-set calibration. Full-rank when n ≫ d."""
    import torch

    from infra.aws_pg.jobs._specter_shared import build_ppl_item, build_vqa_item
    from infra.pg.inherit_core import _split_hidden
    st = {"gram": torch.zeros(D, D, dtype=torch.float64, device=device),
          "sum": torch.zeros(D, dtype=torch.float64, device=device),
          "abs": torch.zeros(D, dtype=torch.float64, device=device), "n": 0, "n_ppl": 0, "n_vqa": 0}

    def hook(_m, args, kwargs):
        hs, _ = _split_hidden(args, kwargs)
        x = hs.reshape(-1, D).detach().double()
        st["gram"] += x.transpose(0, 1) @ x
        st["sum"] += x.sum(0)
        st["abs"] += x.abs().sum(0)
        st["n"] += x.shape[0]
        # NB: a forward_pre_hook must return None (or a new-args tuple). Returning a scalar would
        # replace the layer inputs — a capture hook never modifies, so return nothing.

    h = layers[L].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        for wd in windows:
            item = build_ppl_item(model, proc, device, wd, contextlib.nullcontext, {})
            with torch.no_grad():
                item["forward_fn"]()
        st["n_ppl"] = st["n"]
        if vqa_rows and task is not None:
            for row in vqa_rows[:N_SIGMA_VQA]:
                item = build_vqa_item(model, proc, device, row, task, contextlib.nullcontext, {})
                with torch.no_grad():
                    item["forward_fn"]()
            st["n_vqa"] = st["n"] - st["n_ppl"]
    finally:
        h.remove()
    assert st["n"] > D, f"Σ_x rank-deficient: n={st['n']} ≤ d={D}"
    return {"gram": st["gram"].cpu(), "sum": st["sum"].cpu(), "abs": st["abs"].cpu(),
            "n": st["n"], "n_ppl": st["n_ppl"], "n_vqa": st["n_vqa"]}


def build_banks(qb, sigma_state, seed=1337):
    """Assemble the discovery-frozen banks consumed by ration_core.policy_basis. Q eigenbasis from the
    Q bank; Σ_x eigenbasis + (mean, C_x, absmean) from the accumulated moments."""
    import torch

    from infra.pg import ration_core as RC
    mean_x, C_x, absmean_x = RC.finalize_sigma(sigma_state["gram"], sigma_state["sum"],
                                               sigma_state["abs"], sigma_state["n"])
    sev, sea = RC._sigma_eig(C_x)
    return {"Q_evecs": torch.as_tensor(qb["evecs"], dtype=torch.float64),
            "Q_evals": torch.as_tensor(qb["evals"], dtype=torch.float64).numpy(),
            "sigma_evecs": sev, "sigma_evals": sea.numpy(),
            "mean_x": mean_x, "C_x": C_x, "absmean_x": absmean_x.numpy(), "seed": seed}


def detect_sink(banks, massive_coords=None):
    """The sink-carrying Σ_x eigendirections the sink-aware rival pins to b_max (frozen on discovery)."""
    from infra.pg import ration_core as RC
    mc = tuple(massive_coords) if massive_coords is not None else RC.MASSIVE_COORDS
    return RC.sink_eigendirections(banks["sigma_evecs"], massive_coords=mc)


# ===========================================================================
# The behavior harness — every readout under the substitutor's CURRENT spec.
# ===========================================================================
def spec_to_device(spec, device="cuda", dtype=None):
    """Move a ration_core.make_spec dict onto the model device (R/centers/stds → fp32; bits int)."""
    import torch
    dtype = dtype or torch.float32
    return {"R": spec["R"].to(device, dtype), "bits": torch.as_tensor(spec["bits"]).to(device),
            "centers": spec["centers"].to(device, dtype), "stds": spec["stds"].to(device, dtype)}


def ppl_per_window(model, proc, windows, sub_install, sub_holder, device="cuda", collect_mse=True):
    """Per-window [Σnll, n_tok] under the CURRENT spec (blank-image, eager). If collect_mse, also
    returns the per-token reconstruction MSE at L0 accumulated over the SAME forwards."""
    import torch

    from infra.aws_pg.jobs._specter_shared import build_ppl_item
    if collect_mse:
        sub_holder["collect_mse"] = True
        sub_holder["mse_sum"] = 0.0
        sub_holder["mse_cnt"] = 0
    out = []
    with sub_install():
        for wd in windows:
            item = build_ppl_item(model, proc, device, wd, contextlib.nullcontext, {})
            with torch.no_grad():
                logits = item["forward_fn"]()[0]
            lab = item["labels"].to(logits.device)
            live = lab != -100
            lse = torch.logsumexp(logits[live].float(), dim=-1)
            nll = lse - logits[live].float().gather(1, lab[live].unsqueeze(1)).squeeze(1)
            out.append([float(nll.sum()), int(live.sum())])
    sub_holder["collect_mse"] = False
    mse = (sub_holder.get("mse_sum", 0.0) / max(sub_holder.get("mse_cnt", 0), 1)) if collect_mse else None
    return out, mse


def vqa_per_item(model, proc, task, rows, sub_install, device="cuda", n=None):
    """Per-item VQA correctness (0/1) under the CURRENT spec, via the wired eval_pg1 + a hook ctx."""
    from infra.pg import inherit_pg1 as IP
    use = rows if n is None else rows[:n]
    r = IP.eval_pg1(model, proc, task, use, device, ctx=sub_install())
    assert r["per_item"] is not None, "vqav2 must be a per-item task"
    return r["per_item"]


def run_cell(name, b_bar, banks, model, proc, task, ppl_windows, vqa_rows, layers, L=L_STAR,
             do_vqa=True, n_vqa=N_VQA_CELL, precomputed_basis=None):
    """One policy cell at one budget: ppl per-window + MSE (+ optional VQA per-item), all under the
    substitutor. `precomputed_basis` = (R_dev, centers_dev, stds_dev, w, pinned) to avoid recomputing
    basis_stats across budgets. name='__clean__' → passthrough (spec=None)."""
    import torch

    from infra.pg import ration_core as RC
    sub_install, sub_holder = RC.make_residual_substitutor(layers, L)
    if name == "__clean__":
        sub_holder["spec"] = None
    else:
        if precomputed_basis is None:
            spec = RC.make_spec(name, b_bar, banks)
            sub_holder["spec"] = spec_to_device(spec)
        else:
            R_dev, cen_dev, std_dev, w, pinned = precomputed_basis
            bits = RC.water_fill(w, b_bar, pinned=pinned)
            sub_holder["spec"] = {"R": R_dev, "bits": torch.as_tensor(bits).to(R_dev.device),
                                  "centers": cen_dev, "stds": std_dev}
    pw, mse = ppl_per_window(model, proc, ppl_windows, sub_install, sub_holder)
    vqa = vqa_per_item(model, proc, task, vqa_rows, sub_install, n=n_vqa) if do_vqa else None
    return {"policy": name, "b_bar": float(b_bar) if name != "__clean__" else None,
            "ppl_per_window": pw, "mse": mse, "vqa_per_item": vqa}


def precompute_basis_dev(name, banks, device="cuda"):
    """(R_dev fp32, centers_dev, stds_dev, w, pinned) — the budget-independent policy basis on device."""
    import torch

    from infra.pg import ration_core as RC
    R, centers, stds, w, pinned = RC.policy_basis(name, banks)
    return (R.to(device, torch.float32), centers.to(device, torch.float32),
            stds.to(device, torch.float32), w, pinned)


# ===========================================================================
# Oracle leave-one-band-out (discovery): per-Q-band Δbehavior.
# ===========================================================================
def oracle_band_damage(banks, model, proc, task, ppl_windows, vqa_rows, layers, L=L_STAR,
                       n_bands=None, device="cuda", n_vqa=N_VQA_CELL):
    """For each contiguous band of Q eigendirections (top-first), quantise ONLY that band to 0 (centre)
    and keep the rest at b_max; measure Δppl and Δvqa vs the all-b_max intact. Returns per-band
    {dppl, dvqa} and the intact readouts — the oracle importance + the range-guard ceiling (§O)."""
    import torch

    from infra.pg import ration_core as RC
    n_bands = n_bands or RC.N_ORACLE_BANDS
    R = banks["Q_evecs"]
    centers, stds = RC.basis_stats(R, banks["mean_x"], banks["C_x"])
    R_dev = R.to(device, torch.float32)
    cen_dev, std_dev = centers.to(device, torch.float32), stds.to(device, torch.float32)
    edges = RC.oracle_band_edges(D, n_bands)
    sub_install, sub_holder = RC.make_residual_substitutor(layers, L)

    import numpy as np

    def _readouts(bits):
        sub_holder["spec"] = {"R": R_dev, "bits": torch.as_tensor(bits).to(device),
                              "centers": cen_dev, "stds": std_dev}
        pw, _ = ppl_per_window(model, proc, ppl_windows, sub_install, sub_holder, collect_mse=False)
        nll = sum(s for s, _ in pw) / sum(n for _, n in pw)
        vq = vqa_per_item(model, proc, task, vqa_rows, sub_install, n=n_vqa) if vqa_rows else None
        return nll, (float(np.mean(vq)) if vq is not None else None)

    intact_nll, intact_vqa = _readouts(np.full(D, RC.B_MAX, dtype=np.int64))
    bands = []
    for j in range(n_bands):
        bits = RC.band_bits(D, edges[j], edges[j + 1])
        nll, vqa = _readouts(bits)
        bands.append({"band": j, "lo": int(edges[j]), "hi": int(edges[j + 1]),
                      "dppl_nll": nll - intact_nll,
                      "dvqa": (intact_vqa - vqa) if vqa is not None else None})
    return {"intact_nll": intact_nll, "intact_vqa": intact_vqa, "bands": bands}


# ===========================================================================
# Per-layer Σ_x (Arm L) — one pass, 18 residual-capture hooks, so the layer sweep gives every
# policy layer-matched quantiser ranges (fair test of whether the L0-Q BASIS transfers upward).
# ===========================================================================
def accumulate_sigma_all_layers(model, proc, layers, windows, vqa_rows=None, task=None, device="cuda"):
    """Return {L: {gram, sum, abs, n}} for every decoder layer, captured in ONE pass over the ppl
    windows + (if given) VQA items — the same MIX calibration as accumulate_sigma, at every layer."""
    import torch

    from infra.aws_pg.jobs._specter_shared import build_ppl_item, build_vqa_item
    from infra.pg.inherit_core import _split_hidden
    nL = len(layers)
    st = {L: {"gram": torch.zeros(D, D, dtype=torch.float64, device=device),
              "sum": torch.zeros(D, dtype=torch.float64, device=device),
              "abs": torch.zeros(D, dtype=torch.float64, device=device), "n": 0} for L in range(nL)}

    def mk(L):
        def hook(_m, args, kwargs):
            hs, _ = _split_hidden(args, kwargs)
            x = hs.reshape(-1, D).detach().double()
            st[L]["gram"] += x.transpose(0, 1) @ x
            st[L]["sum"] += x.sum(0)
            st[L]["abs"] += x.abs().sum(0)
            st[L]["n"] += x.shape[0]
        return hook

    hs = [layers[L].register_forward_pre_hook(mk(L), with_kwargs=True) for L in range(nL)]
    try:
        for wd in windows:
            item = build_ppl_item(model, proc, device, wd, contextlib.nullcontext, {})
            with torch.no_grad():
                item["forward_fn"]()
        if vqa_rows and task is not None:
            for row in vqa_rows[:N_SIGMA_VQA]:
                item = build_vqa_item(model, proc, device, row, task, contextlib.nullcontext, {})
                with torch.no_grad():
                    item["forward_fn"]()
    finally:
        for h in hs:
            h.remove()
    return {L: {"gram": st[L]["gram"].cpu(), "sum": st[L]["sum"].cpu(),
                "abs": st[L]["abs"].cpu(), "n": st[L]["n"]} for L in range(nL)}
