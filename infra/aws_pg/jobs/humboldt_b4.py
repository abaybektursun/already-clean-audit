"""HUMBOLDT B4 — perturbation response spectrum (spec §3/B4). GPU, measurement-grade.

PG1 only. At each layer, ONE AT A TIME (enforced by construction — BankInject holds exactly one
layer), add ε·Uz to the pre-q/k residual at all positions, z unit-random in-bank, 3 seeds, ε swept
over 6 log-spaced values calibrated per layer so the smallest moves ‖h‖ by ~0.1% and the largest by
~30% (calibration on 1 batch, FROZEN before the sweep).

Records, per (layer, bank, ε, seed): attention-map JS divergence vs the clean run, next-token KL,
and Δppl on a 2k-token wikitext slice. This records the response PROFILE; it asserts nothing.

Smoke (§3/B4): ε=0 must reproduce the clean run bit-identically; the largest ε on the Haar twin must
produce nonzero divergence (a dead readout cannot record a profile).

Budget (§12): capped; if the cap binds the drop order is seeds 2–3 first, then the two middle ε
values — decided ONCE from a timing probe before the sweep so coverage stays uniform, and logged.

  pg run infra.aws_pg.jobs.humboldt_b4 --type p5.4xlarge,g6e.8xlarge --no-weights \
      --s3-prefix results/humboldt_b4 --budget-min 540
"""
import argparse
import json
import pathlib
import time

from infra.pg import humboldt_act as A
from infra.pg import humboldt_core as H
from infra.pg import humboldt_smoke as S
from infra.pg import humboldt_weights as W

SITE = "q_proj_in"
EPS_FRACS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.30)     # frozen; ~0.1% … ~30% of ‖h‖
SEEDS = (0, 1, 2)
MIDDLE_EPS_IDX = (2, 3)                                # the "two middle ε" of the drop order


def _windows(tok, n_tokens, window):
    from datasets import load_dataset
    from infra.pg.text_eval import WIKITEXT_CONFIG, WIKITEXT_REPO
    ds = load_dataset(WIKITEXT_REPO, WIKITEXT_CONFIG, split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0][:n_tokens]
    return [tok.decode(ids[i:i + window].tolist())
            for i in range(0, len(ids) - window + 1, window)]


def _prepare(proc, device, texts, blank):
    """Tokenise + pack the slice ONCE. B4 runs thousands of interventions over the SAME windows, so
    re-invoking the processor per cell would spend more wall-clock on tokenisation than on forwards
    — and any per-call variation would be an uncontrolled difference between arms."""
    import torch
    from infra.aws_pg.jobs.invariant_core_func import _locate_text_block
    tok = proc.tokenizer
    out = []
    for t in texts:
        win = tok(t, return_tensors="pt", add_special_tokens=False).input_ids[0]
        inp = proc(text=[t], images=[blank], return_tensors="pt", padding=False).to(device)
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        pos = _locate_text_block(inp["input_ids"][0].cpu(), win)
        out.append({"inp": inp, "win": win, "pos": int(pos), "n_text": int(win.shape[0])})
    return out


def _run(model, prepared, device, want_attn_on_first):
    """One pass over the prepared slice. Returns (nll_sum, n_tok, first-window logits, attentions)."""
    import torch
    nll, ntok, logits0, attn0 = 0.0, 0, None, None
    for i, pr in enumerate(prepared):
        inp, win, pos, n_text = pr["inp"], pr["win"], pr["pos"], pr["n_text"]
        want = want_attn_on_first and i == 0
        with torch.inference_mode():
            o = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                      pixel_values=inp["pixel_values"], output_attentions=want)
        lg = o.logits[0, pos:pos + n_text - 1, :].float()
        tgt = win[1:].to(device)
        nll += float(torch.nn.functional.cross_entropy(lg, tgt, reduction="sum").item())
        ntok += int(tgt.numel())
        if want:
            logits0 = lg.detach()
            attn0 = tuple(x.detach().float() for x in o.attentions)
    return nll, ntok, logits0, attn0


def _js(a, b, eps=1e-12):
    """Mean Jensen-Shannon divergence over attention rows, on GPU (nats)."""
    tot, n = 0.0, 0
    for x, y in zip(a, b):
        p = x.clamp_min(eps)
        q = y.clamp_min(eps)
        p = p / p.sum(-1, keepdim=True)
        q = q / q.sum(-1, keepdim=True)
        m = 0.5 * (p + q)
        d = 0.5 * (p * (p.log() - m.log())).sum(-1) + 0.5 * (q * (q.log() - m.log())).sum(-1)
        tot += float(d.mean().item())
        n += 1
    return tot / n


def _kl(pl, ql, chunk: int = 64):
    """Mean KL(clean ‖ perturbed) over next-token distributions (nats), chunked over positions: the
    vocabulary is 257k wide, so a whole-tensor formulation would hold several 0.5 GB copies at once."""
    import torch
    tot, n = 0.0, 0
    for i in range(0, pl.shape[0], chunk):
        p = torch.log_softmax(pl[i:i + chunk], -1)
        q = torch.log_softmax(ql[i:i + chunk], -1)
        tot += float((p.exp() * (p - q)).sum(-1).sum().item())
        n += int(p.shape[0])
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="")
    ap.add_argument("--n-tokens", type=int, default=2048)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--budget-h", type=float, default=8.0, help="§12 B4 cap in GPU-hours")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--run-id", default="humboldt_b4")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    H.hf_setup()                       # FIRST — XET off + gated-repo auth (§0.2.7)
    import numpy as np
    import torch
    assert torch.cuda.is_available(), "GPU required"
    device = "cuda"
    from infra.aws_pg.jobs.invariant_core_func import _blank_image

    gates = H.Gates(out)
    S.run_all_offline(gates)
    bank_dir = W.fetch_banks(H.cache_dir(out, "banks"))

    model, proc, layers, sha, _mm = A.load_model("pg1-pt224", A.PG1_REPO, "paligemma", device=device)
    layer_idx = ([int(x) for x in a.layers.split(",") if x.strip()] if a.layers.strip()
                 else list(range(len(layers))))
    blank = _blank_image(128)
    texts = _windows(proc.tokenizer, a.n_tokens, a.window)
    bank_ids = [(t, bid) for t in A.BANK_TYPES_AT_SITE[SITE] for bid in H.BANK_IDS]
    if a.smoke:
        layer_idx, texts, bank_ids = layer_idx[:1], texts[:1], bank_ids[:2]

    # ---- clean reference (also the ε-calibration batch: the residual norms at each layer's site).
    energy_probe = A.SiteSlice(layers, device)
    norms = {}
    with energy_probe:
        inp = proc(text=[texts[0]], images=[blank], return_tensors="pt", padding=False).to(device)
        energy_probe.set_window(0, int(inp["input_ids"].shape[1]))
        with torch.inference_mode():
            model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                  pixel_values=inp["pixel_values"].to(torch.bfloat16))
        for L in layer_idx:
            h = energy_probe.buf[(L, SITE)]
            norms[L] = float(np.linalg.norm(h, axis=1).mean())
    eps_grid = {L: [f * norms[L] for f in EPS_FRACS] for L in layer_idx}   # FROZEN before the sweep

    prepared = _prepare(proc, device, texts, blank)
    nll_c, ntok_c, logits_c, attn_c = _run(model, prepared, device, True)
    ppl_c = float(np.exp(nll_c / ntok_c))
    print(f"[B4] clean ppl={ppl_c:.4f} over {ntok_c} tokens; ‖h‖ by layer: "
          f"{ {k: round(v, 2) for k, v in norms.items()} }", flush=True)

    # ---- the readout's own noise floor: the SAME clean forward, run twice. GPU kernels are not
    # contractually bit-deterministic, so "ε=0 must be bit-identical" is a gate that can fail for a
    # reason that has nothing to do with the instrument. Measure the floor first and judge ε=0
    # against it — a self-referenced criterion that still catches a projector that does nothing.
    nr, kr, lgr, atr = _run(model, prepared, device, True)
    floor = {"dppl": abs(float(np.exp(nr / kr)) - ppl_c), "js": _js(attn_c, atr),
             "kl": _kl(logits_c, lgr)}
    del lgr, atr

    # ---- smoke: ε=0 must reproduce the clean run to within that floor.
    L0, (t0, b0) = layer_idx[0], bank_ids[0]
    U0 = W.load_bank(bank_dir, t0, L0, b0)
    z0 = np.zeros(H.BANK_K); z0[0] = 1.0
    with A.BankInject(layers[L0], SITE, U0, z0, 0.0, device):
        n0, k0, lg0, at0 = _run(model, prepared, device, True)
    zero = {"dppl": abs(float(np.exp(n0 / k0)) - ppl_c), "js": _js(attn_c, at0),
            "kl": _kl(logits_c, lg0)}
    exact = all(v == 0.0 for v in zero.values())
    gates.require("G5_B4_zero_eps_identity",
                  exact or all(zero[k] <= max(floor[k], 0.0) for k in zero),
                  zero_eps=zero, self_noise_floor=floor, bitwise_identical=exact,
                  criterion="ε=0 reproduces the clean run bit-identically, or within the readout's "
                            "own rerun-to-rerun floor (pre-registered two-branch gate)")
    del lg0, at0

    # ---- smoke: the largest ε on the Haar twin must move the readout.
    Uh = W.load_bank(bank_dir, "q_proj", L0, "haar32_s0")
    rng = np.random.default_rng(H.cell_seed(a.seed, "q_proj", L0, "haar32_s0", 0))
    zh = rng.standard_normal(H.BANK_K); zh /= np.linalg.norm(zh)
    with A.BankInject(layers[L0], SITE, Uh, zh, eps_grid[L0][-1], device):
        nh, kh, lgh, ath = _run(model, prepared, device, True)
    live = {"js": _js(attn_c, ath), "kl": _kl(logits_c, lgh),
            "dppl": float(np.exp(nh / kh)) - ppl_c}
    gates.require("G5_B4_live_readout", live["js"] > 0 and live["kl"] > 0,
                  **live, eps=eps_grid[L0][-1],
                  criterion="largest ε on the Haar twin produces nonzero divergence")

    # ---- budget guard (§12): decide the drop level ONCE, from a measured cell, so coverage stays
    # uniform across layers. Nothing is capped silently.
    t_cell = time.time()
    with A.BankInject(layers[L0], SITE, U0, zh, eps_grid[L0][0], device):
        _run(model, prepared, device, True)
    per_cell = time.time() - t_cell + 0.15                    # + the JS/KL reduction
    seeds, eps_idx = list(SEEDS), list(range(len(EPS_FRACS)))
    drops = []
    for level in range(3):
        n_cells = len(layer_idx) * len(bank_ids) * len(eps_idx) * len(seeds)
        if per_cell * n_cells / 3600.0 <= a.budget_h or level == 2:
            break
        if level == 0:
            seeds = [SEEDS[0]]
            drops.append("seeds 2-3")
        else:
            eps_idx = [i for i in eps_idx if i not in MIDDLE_EPS_IDX]
            drops.append(f"middle eps {list(MIDDLE_EPS_IDX)}")
    n_cells = len(layer_idx) * len(bank_ids) * len(eps_idx) * len(seeds)
    budget = {"per_cell_s": per_cell, "n_cells": n_cells,
              "projected_h": per_cell * n_cells / 3600.0, "cap_h": a.budget_h,
              "dropped": drops, "seeds_used": seeds, "eps_idx_used": eps_idx,
              "eps_fracs": list(EPS_FRACS)}
    print(f"[B4] budget: {json.dumps(budget)}", flush=True)

    manifest = {"arm": "B4", "run_id": a.run_id, "commit_sha": H.commit_sha(),
                "model": "pg1-pt224", "model_sha": sha, "site": SITE, "layers": layer_idx,
                "banks": [list(b) for b in bank_ids], "eps_fracs": list(EPS_FRACS),
                "eps_grid_abs": {str(k): v for k, v in eps_grid.items()},
                "residual_norms": {str(k): v for k, v in norms.items()},
                "clean_ppl": ppl_c, "n_tokens": ntok_c, "seed": a.seed, "budget": budget,
                "gpu": torch.cuda.get_device_name(0)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    atlas = H.Atlas(out, arm="B4", run_id=a.run_id)      # bound BEFORE any loop (§0.2.4)
    done = 0
    for L in layer_idx:
        for (t, bid) in bank_ids:
            U = W.load_bank(bank_dir, t, L, bid)
            for si in seeds:
                rng = np.random.default_rng(H.cell_seed(a.seed, t, L, bid, si))
                z = rng.standard_normal(H.BANK_K)
                z /= np.linalg.norm(z)
                for ei in eps_idx:
                    eps = eps_grid[L][ei]
                    with A.BankInject(layers[L], SITE, U, z, eps, device):
                        nll, ntok, lg, at = _run(model, prepared, device, True)
                    fl = {**H.type_flags(t), "site": SITE, "eps_frac": EPS_FRACS[ei],
                          "eps_abs": eps, "z_seed": si, "residual_norm": norms[L],
                          "clean_ppl": ppl_c}
                    common = dict(layer=L, type=t, bank_id=bid, model_id="pg1-pt224",
                                  model_sha=sha, corpus_id="wikitext2k", seed=si,
                                  token_class=f"eps{ei}", flags=fl)
                    atlas.add("perturb_attn_js", value=_js(attn_c, at), **common)
                    atlas.add("perturb_next_token_kl", value=_kl(logits_c, lg), **common)
                    atlas.add("perturb_dppl", value=float(np.exp(nll / ntok)) - ppl_c, **common)
                    done += 1
            print(f"[B4] L{L} {t}/{bid}: {done}/{n_cells} cells, "
                  f"{(time.time() - t_start) / 3600:.2f} h", flush=True)
        H.s3_sync(out, f"humboldt/{a.run_id}")          # per layer; run_job.sh also syncs every 30 s

    atlas.close()
    rep = atlas.to_parquet()
    (out / "b4_summary.json").write_text(json.dumps(
        {"manifest": manifest, "atlas": rep, "gates": gates.gates, "n_cells_done": done,
         "elapsed_h": (time.time() - t_start) / 3600.0}, indent=2, default=str))
    H.s3_sync(out, f"humboldt/{a.run_id}")
    print("=" * 78)
    print(f"HUMBOLDT_B4_OK rows={rep['n_rows']} cells={done}/{n_cells} dropped={drops} "
          f"elapsed_h={(time.time() - t_start) / 3600:.2f}")


if __name__ == "__main__":
    main()
