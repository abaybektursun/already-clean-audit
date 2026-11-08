"""GESTALT harvest — one unit's residual streams at BOTH crosscoder sites (L9, L14), banked.
Spec Arm S population + journal FR-8. The SAME job serves Arm S (unit pg1) and Arm D (re-shells).

What it does, in order (each step gated, fail-loud):
  1. Shell load (eager, bf16, pinned sha) + **ppl-anchor preflight** (the fail-fast known number:
     recomputed, relative 1e-3 vs the lacuna_ladder anchor, GPU name banked).
  2. Items: the four battery tasks at LACUNA sizes, content-hash dedup, `lacuna_core.parity_split`
     -> DISCOVERY half only (confirmation halves never harvested).
  3. Unit install via `plenum_units.install_unit` (pg1 = native ceiling; pi0/pi05/ft-* for Arm D),
     key-count gates asserted by the installer itself (FR-2 b).
  4. Forward every discovery item batch (real images); capture the post-block output residual at
     sites 9 and 14 (`ResidualHarvest`); keep attention-kept tokens; input-ids hash banked per
     batch (the pairing guarantee across units — every unit sees the identical batches).
  5. Bank to S3 (via --out): per-site fp16 shards + per-token item indices + fp64 moments +
     the manifest (hashes, counts, gates). Finite-activations gate (FR-2 c) on every shard.
  6. The frozen FIDELITY ITEM LIST (FR-1): first 800 VQAv2 discovery items, hashes banked.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.gestalt_harvest --args "--unit pg1" \
      --type p5.4xlarge,g6e.8xlarge --s3-prefix results/gestalt_harvest_pg1 --budget-min 240

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import time

SITES = (9, 14)                    # the crosscoder sites (tessera_x.SITES; spec Arm S)
N_ITEMS = {"vqav2": 4000, "textvqa": 2000, "refcoco": 1975, "coco_captions": 2000}
N_FIDELITY = 800                   # FR-1 frozen fidelity item count (VQAv2 discovery head)
SHARD_TOKENS = 250_000             # fp16 shard size ~1GB
BATCH_ITEMS = 16


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--unit", default="pg1",
                    help="pg1 | pi0 | pi05 | ft-<slug> (plenum_units.install_unit ids)")
    ap.add_argument("--n-cap-items", type=int, default=0, help="smoke cap (0 = full discovery)")
    a = ap.parse_args()
    t0 = time.time()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.aws_pg.jobs._inherit_shared import install_pg
    install_pg()               # pycocoevalcap (battery deps) is not in the AMI
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch

    from infra.aws_pg.jobs import _stencil_shared as SS
    from infra.aws_pg.jobs.lacuna_ladder import PPL_ANCHOR, PPL_REL_TOL
    from infra.pg import inherit_pg1 as IP
    from infra.pg import lacuna_core as LC
    from infra.pg import lacuna_shell as LS
    from infra.pg import plenum_units as PU
    from infra.pg.gemma_scope import ResidualHarvest

    assert torch.cuda.is_available(), "gestalt_harvest needs CUDA"
    device = "cuda"
    rep = {"object": "GESTALT harvest", "unit": a.unit, "sites": list(SITES),
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    _w()

    # ---- 1. shell + ppl-anchor preflight (the fail-fast known number) ----------------------
    model, proc, layers, sha, base_ppl_pin = SS.load_model("/tmp/gestalt_harvest")
    pg1_sd = LS.pg1_baseline_sd(model)
    ppl = IP.ppl_gate(model, proc, device)
    val = ppl["ppl"] if isinstance(ppl, dict) else float(ppl)
    rel = abs(float(val) - PPL_ANCHOR) / PPL_ANCHOR
    rep["preflight"] = {"ppl": float(val), "anchor": PPL_ANCHOR, "rel": rel,
                        "pinned_sha": sha, "passes": rel < PPL_REL_TOL}
    print(f"[GH] preflight ppl={val} anchor={PPL_ANCHOR} rel={rel:.2e}", flush=True)
    assert rel < PPL_REL_TOL, f"ppl anchor FAIL: {val} vs {PPL_ANCHOR} (rel {rel:.2e})"
    _w()

    # ---- 2. items: 4 tasks, discovery halves only ------------------------------------------
    tasks = LS.mm_tasks()
    rows_all, hashes_all, item_task, fidelity_hashes = [], [], [], []
    for name in ("vqav2", "textvqa", "refcoco", "coco_captions"):
        # smoke mode caps the STREAMING LOAD too (200/task) — same deterministic load order +
        # hash-parity split as the full run, so the smoke validates the identical path
        n_load = min(N_ITEMS[name], 200) if a.n_cap_items else N_ITEMS[name]
        rows, hashes = IP.load_items(tasks[name], n_load)
        disc, conf = LC.parity_split(hashes)
        d_rows = [r for r, m in zip(rows, disc) if m]
        d_hashes = [h for h, m in zip(hashes, disc) if m]
        if name == "vqav2":
            if not a.n_cap_items:
                assert len(d_rows) >= N_FIDELITY, f"vqav2 discovery {len(d_rows)} < {N_FIDELITY}"
            fidelity_hashes = d_hashes[: min(N_FIDELITY, len(d_hashes))]
        if a.n_cap_items:
            d_rows, d_hashes = d_rows[: a.n_cap_items], d_hashes[: a.n_cap_items]
        rows_all += [(name, r) for r in d_rows]
        hashes_all += d_hashes
        item_task += [name] * len(d_rows)
        print(f"[GH] {name}: {len(d_rows)} discovery items", flush=True)
    rep["items"] = {"n_discovery_total": len(rows_all),
                    "per_task": {t: item_task.count(t) for t in set(item_task)},
                    "fidelity_n": len(fidelity_hashes),
                    "fidelity_sha": hashlib.sha256("".join(fidelity_hashes).encode()).hexdigest()[:16]}
    (out / "item_hashes.json").write_text(json.dumps(
        {"hashes": hashes_all, "tasks": item_task, "fidelity_hashes": fidelity_hashes}))
    _w()

    # ---- 3. unit install --------------------------------------------------------------------
    info = PU.install_unit(a.unit, model, proc, pg1_sd, device,
                           {"stitch": 0, "matched_dw": 0, "rand_dec": 0})
    rep["install"] = info
    print(f"[GH] installed {a.unit}: {str(info)[:200]}", flush=True)
    _w()

    # ---- 4+5. harvest both sites, shard as we go -------------------------------------------
    tok = proc.tokenizer
    orig_side = tok.padding_side
    tok.padding_side = "left"
    buf = {L: [] for L in SITES}
    idxbuf = {L: [] for L in SITES}
    counts = {L: 0 for L in SITES}
    shard_i = {L: 0 for L in SITES}
    moments = {L: {"n": 0, "sum": None, "gram": None} for L in SITES}
    batch_hashes = []
    site_dirs = {L: out / f"site{L}" for L in SITES}
    for L in SITES:
        site_dirs[L].mkdir(exist_ok=True)

    def _flush(L, final=False):
        if not buf[L] or (not final and sum(x.shape[0] for x in buf[L]) < SHARD_TOKENS):
            return
        X = torch.cat(buf[L], 0)
        idx = torch.cat(idxbuf[L], 0)
        assert torch.isfinite(X).all(), f"non-finite activations at site {L} (FR-2 c)"
        np.save(site_dirs[L] / f"acts_{shard_i[L]:03d}.npy", X.numpy())
        np.save(site_dirs[L] / f"item_idx_{shard_i[L]:03d}.npy", idx.numpy())
        X64 = X.to(torch.float64)
        m = moments[L]
        m["sum"] = X64.sum(0) if m["sum"] is None else m["sum"] + X64.sum(0)
        m["gram"] = X64.T @ X64 if m["gram"] is None else m["gram"] + X64.T @ X64
        m["n"] += X.shape[0]
        counts[L] += X.shape[0]
        shard_i[L] += 1
        buf[L].clear()
        idxbuf[L].clear()

    for start in range(0, len(rows_all), BATCH_ITEMS):
        chunk = rows_all[start:start + BATCH_ITEMS]
        images = [tasks[nm].image_fn(r).convert("RGB") for nm, r in chunk]
        prompts = [tasks[nm].prompt_fn(r) for nm, r in chunk]
        inp = proc(text=prompts, images=images, return_tensors="pt", padding="longest").to(device)
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        batch_hashes.append(hashlib.sha256(
            inp["input_ids"].detach().cpu().numpy().tobytes()).hexdigest()[:12])
        keep = inp["attention_mask"].bool()
        item_ids = torch.arange(start, start + len(chunk)).unsqueeze(1).expand(-1, keep.shape[1])
        hs = {L: ResidualHarvest(layers, L) for L in SITES}
        for L in SITES:
            hs[L].__enter__()
        with torch.inference_mode():
            model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                  pixel_values=inp["pixel_values"])
        for L in SITES:
            cap = hs[L].captured
            hs[L].__exit__(None, None, None)
            flat = cap.reshape(-1, cap.shape[-1])[keep.reshape(-1)]
            # fp32 shards: the bf16 residual carries massive-activation dims whose range exceeds
            # fp16 max (65504) — an fp16 cast would inf (and the finite gate would kill the run)
            buf[L].append(flat.float().cpu())
            idxbuf[L].append(item_ids.reshape(-1)[keep.reshape(-1).cpu()].to(torch.int32))
            _flush(L)
        if (start // BATCH_ITEMS) % 25 == 0:
            rep["progress"] = {"items_done": start + len(chunk), "tokens": dict(counts)}
            _w()
            print(f"[GH] {start + len(chunk)}/{len(rows_all)} items, tokens {dict(counts)}",
                  flush=True)
    for L in SITES:
        _flush(L, final=True)
    tok.padding_side = orig_side

    for L in SITES:
        m = moments[L]
        mean = m["sum"] / m["n"]
        cov = m["gram"] / m["n"] - torch.outer(mean, mean)
        np.savez(site_dirs[L] / "moments.npz", mean=mean.numpy(), cov=cov.numpy(), n=m["n"])
        rep[f"site{L}"] = {"n_tokens": counts[L], "n_shards": shard_i[L],
                           "mean_sha": hashlib.sha256(mean.numpy().tobytes()).hexdigest()[:16],
                           "cov_sha": hashlib.sha256(cov.numpy().tobytes()).hexdigest()[:16]}
    rep["batch_input_hash"] = hashlib.sha256("".join(batch_hashes).encode()).hexdigest()[:16]
    rep["n_batches"] = len(batch_hashes)
    (out / "batch_hashes.json").write_text(json.dumps(batch_hashes))
    rep["elapsed_s"] = time.time() - t0
    _w()
    print(f"GESTALT_HARVEST_OK unit={a.unit} tokens={dict(counts)} "
          f"pair_hash={rep['batch_input_hash']}", flush=True)


if __name__ == "__main__":
    main()
    # D7: datasets/cuda extension threads abort the interpreter during finalization
    # (PyGILState_Release fatal, rc=-6) AFTER all artifacts are banked; every write is
    # explicit and already streamed, so skip finalization outright.
    import os as _os
    _os._exit(0)
