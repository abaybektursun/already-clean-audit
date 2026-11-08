"""LACUNA Arm 2d — text-geometry preservation across the gemma-2b → PG1 install (W8-analogue).

Estimand (spec §2.2): per decoder layer, the top-k principal-subspace overlap between the activation
Gram of BASE gemma-2b and of PG1-under-a-blank-image on MATCHED wikitext windows, normalised into
P = (ov − FLOOR)/(CEILING − FLOOR).

**INSTRUMENT CORRECTION 2026-07-28 (v2).** v1 built the CEILING as base's FIRST vs SECOND contiguous
half of each window's tokens. That is not a ceiling: the two halves are different *text*, so their
activation subspaces differ for reasons that have nothing to do with instrument noise, and the
measured "ceiling" came out BELOW the base↔PG1 overlap it was supposed to bound — P landed at 3–6,
outside [0,1]. The run is discarded and the instrument rebuilt; the numbers are reported in the
journal as a caught instrument failure, not as a result.

v2 matches all three quantities by construction. Each window's token positions are split at random
into two disjoint halves A and B (random, not contiguous, so neither half is a different topic), and
EVERY statistic is a half-vs-half overlap:

    OV_w      = overlap( topk(base on A_w) , topk(PG1 on B_w) )
    CEILING_w = overlap( topk(base on A_w) , topk(base on B_w) )      ← same model, same noise budget
    FLOOR_w   = overlap( topk(base on A_w) , topk(PG1 on B_{π(w)}) )  ← π a derangement of windows
    (also banked: PG1's own self-consistency overlap(topk(PG1 on A), topk(PG1 on B)) — a second
     ceiling, so a broken ceiling shows up as a disagreement between the two rather than silently.)

VOID if the CEILING − FLOOR band's lower-CI ≤ 0 (§6.2), and additionally VOID-INVALID if the pooled
overlap exceeds the pooled ceiling (the v1 failure signature) — a range check that CAN fail.

Alignment: BOTH models see IDENTICAL token ids. The window is tokenized once with the PG1
processor's tokenizer; PG1 runs the blank-image forward and the capture is sliced to the located
TEXT block (never the 256 image tokens); gemma-2b is fed the same ids directly. Instrument: the
invcore-FUNC `q_proj` forward-pre-hook contract (the post-input_layernorm residual), REUSED.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.lacuna_a2_geom --type g6e.2xlarge --no-weights \
      --s3-prefix results/lacuna_a2_geom2 --budget-min 120
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

N_LAYERS = 18


def _topk_basis(X, idx, k):
    """Top-k eigenvector basis of the Gram of rows `idx` of the activation matrix X (fp64)."""
    import numpy as np
    A = np.asarray(X[idx], dtype=np.float64)
    G = A.T @ A
    w, V = np.linalg.eigh(G)
    return V[:, np.argsort(w)[::-1][:k]]


class _ActCapture:
    """Per-layer activation matrix (T, d) of the q_proj INPUT for ONE forward, optionally sliced."""

    def __init__(self, layers):
        self.layers = layers
        self.sl = None
        self.X = {}
        self._h = []

    def _mk(self, L):
        def pre_hook(_m, args):
            x = args[0]
            if self.sl is not None:
                x = x[:, self.sl[0]:self.sl[1], :]
            self.X[L] = x.reshape(-1, x.shape[-1]).float().cpu().numpy()
            return None
        return pre_hook

    def __enter__(self):
        for L, layer in enumerate(self.layers):
            self._h.append(layer.self_attn.q_proj.register_forward_pre_hook(self._mk(L)))
        return self

    def __exit__(self, *exc):
        for h in self._h:
            h.remove()
        self._h = []


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-windows", type=int, default=24)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--split-seed", type=int, default=20260728)
    ap.add_argument("--perm-seed", type=int, default=20260727)
    ap.add_argument("--n-boot", type=int, default=4000)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM

    from infra.aws_pg.jobs.invariant_core_func import _blank_image, _locate_text_block
    from infra.pg import lacuna_core as LC
    from infra.pg import lacuna_shell as LS
    from infra.pg.w8_weight_activation import subspace_overlap

    assert torch.cuda.is_available(), "lacuna_a2_geom needs GPU"
    device = "cuda"
    t0 = time.time()
    rep = {"object": "lacuna_a2_geom_v2", "n_windows": a.n_windows, "window": a.window,
           "topk": a.topk, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
           "instrument_note": ("v2 — all three statistics are half-vs-half overlaps on random "
                               "disjoint token halves of the SAME window (v1's contiguous "
                               "first-vs-second-half ceiling was invalid and is discarded)")}
    per_layer = {}
    rep["per_layer"] = per_layer

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    pg1, proc = LS.load_pg1(device=device)
    tok = proc.tokenizer
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    blank = _blank_image(128)

    windows = []
    for start in range(0, ids.size(0), a.window):
        chunk = ids[start:start + a.window]
        if chunk.size(0) < 64:
            continue
        windows.append(tok.decode(chunk.tolist()))
        if len(windows) >= a.n_windows:
            break
    assert len(windows) == a.n_windows, f"only {len(windows)} windows"

    # ---- PG1 (blank image), per window, TEXT positions only --------------------------------
    from infra.pg.inherit_pg1 import decoder_layers
    pg1_X, win_ids_all = [], []
    cap = _ActCapture(decoder_layers(pg1))
    with cap:
        for wtxt in windows:
            win_ids = tok(wtxt, return_tensors="pt", add_special_tokens=False).input_ids[0]
            inp = proc(text=[wtxt], images=[blank], return_tensors="pt", padding=False).to(device)
            pos = _locate_text_block(inp["input_ids"][0], win_ids)
            n_text = int(win_ids.shape[0])
            cap.sl = (pos, pos + n_text)
            cap.X = {}
            with torch.inference_mode():
                pg1(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                    pixel_values=inp["pixel_values"].to(torch.bfloat16))
            assert all(cap.X[L].shape[0] == n_text for L in range(N_LAYERS)), \
                f"capture length {cap.X[0].shape[0]} ≠ text tokens {n_text}"
            pg1_X.append({L: cap.X[L] for L in range(N_LAYERS)})
            win_ids_all.append(win_ids)
    del pg1
    torch.cuda.empty_cache()
    print(f"[A2D] PG1 activations captured ({len(pg1_X)} windows)", flush=True)
    _w()

    # ---- base gemma-2b on the IDENTICAL token ids -------------------------------------------
    g1 = AutoModelForCausalLM.from_pretrained(
        LS.GEMMA1_REPO, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation="eager").eval()
    assert len(g1.model.layers) == N_LAYERS, f"gemma-2b has {len(g1.model.layers)} layers"
    base_X = []
    cap2 = _ActCapture(g1.model.layers)
    with cap2:
        for win_ids in win_ids_all:
            ii = win_ids.unsqueeze(0).to(device)
            cap2.sl = None
            cap2.X = {}
            with torch.inference_mode():
                g1(input_ids=ii, attention_mask=torch.ones_like(ii))
            base_X.append({L: cap2.X[L] for L in range(N_LAYERS)})
    del g1
    torch.cuda.empty_cache()
    print("[A2D] base activations captured", flush=True)
    _w()

    # ---- matched half-vs-half statistics ----------------------------------------------------
    nW = len(windows)
    rng = np.random.default_rng(a.split_seed)
    halves = []
    for w in range(nW):
        T = base_X[w][0].shape[0]
        assert pg1_X[w][0].shape[0] == T, f"window {w}: token-count mismatch base {T} vs PG1"
        p = rng.permutation(T)
        halves.append((np.sort(p[: T // 2]), np.sort(p[T // 2:])))
    prng = np.random.default_rng(a.perm_seed)
    perm = prng.permutation(nW)
    while any(perm == np.arange(nW)):
        perm = prng.permutation(nW)
    rep["derangement"] = perm.tolist()

    pooled = {"ov": [], "ceiling": [], "floor": [], "ceiling_pg1": []}
    for L in range(N_LAYERS):
        ov, ce, fl, cp = [], [], [], []
        for w in range(nW):
            A, B = halves[w]
            bA = _topk_basis(base_X[w][L], A, a.topk)
            bB = _topk_basis(base_X[w][L], B, a.topk)
            pB = _topk_basis(pg1_X[w][L], B, a.topk)
            pA = _topk_basis(pg1_X[w][L], A, a.topk)
            Bp = halves[perm[w]][1]
            pBperm = _topk_basis(pg1_X[perm[w]][L], Bp, a.topk)
            ov.append(subspace_overlap(bA, pB))
            ce.append(subspace_overlap(bA, bB))
            cp.append(subspace_overlap(pA, pB))
            fl.append(subspace_overlap(bA, pBperm))
        ovm, cem, flm, cpm = (float(np.mean(x)) for x in (ov, ce, fl, cp))
        band = LC.boot_diff_ci(ce, fl, n_boot=a.n_boot, seed=L)
        test = LC.boot_diff_ci(ov, fl, n_boot=a.n_boot, seed=100 + L)
        invalid = ovm > cem + 1e-9
        P = LC.preservation_P(ovm, flm, cem)
        per_layer[str(L)] = {
            "overlap": ovm, "ceiling_base": cem, "ceiling_pg1": cpm, "floor": flm, "P": P,
            "band_gap": band["diff"], "band_lo95": band["lo95"],
            "band_void": bool(band["lo95"] <= 0), "range_invalid": bool(invalid),
            "P_label": ("VOID-INVALID (overlap above ceiling)" if invalid else
                        "VOID" if band["lo95"] <= 0 else LC.band_label_P(P, None, None)),
            "ov_minus_floor": test["diff"], "ov_minus_floor_lo95": test["lo95"],
            "p_le0": test["p_le0"], "n_windows": nW}
        for key, val in (("ov", ovm), ("ceiling", cem), ("floor", flm), ("ceiling_pg1", cpm)):
            pooled[key].append(val)
        print(f"[A2D] L{L:02d} ov={ovm:.4f} ceil_base={cem:.4f} ceil_pg1={cpm:.4f} "
              f"floor={flm:.4f} P={P:.3f} invalid={invalid}", flush=True)
        _w()

    ovm, cem, flm, cpm = (float(np.mean(pooled[k])) for k in ("ov", "ceiling", "floor", "ceiling_pg1"))
    band = LC.boot_diff_ci(pooled["ceiling"], pooled["floor"], n_boot=a.n_boot, seed=999)
    test = LC.boot_diff_ci(pooled["ov"], pooled["floor"], n_boot=a.n_boot, seed=1999)
    invalid = ovm > cem + 1e-9
    P = LC.preservation_P(ovm, flm, cem)
    rep["pooled"] = {"overlap": ovm, "ceiling": cem, "ceiling_pg1": cpm, "floor": flm, "P": P,
                     "band_gap": band["diff"], "band_lo95": band["lo95"],
                     "band_void": bool(band["lo95"] <= 0), "range_invalid": bool(invalid),
                     "P_label": ("VOID-INVALID (overlap above ceiling)" if invalid else
                                 "VOID" if band["lo95"] <= 0 else LC.band_label_P(P, None, None)),
                     "T11_p_le0": test["p_le0"], "ov_minus_floor": test["diff"],
                     "ov_minus_floor_lo95": test["lo95"]}
    rep["elapsed_s"] = round(time.time() - t0, 1)
    _w()
    print("LACUNA_A2_GEOM_OK", json.dumps(rep["pooled"], default=str), flush=True)


if __name__ == "__main__":
    main()
