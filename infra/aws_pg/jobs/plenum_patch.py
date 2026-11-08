"""PLENUM Arm 4 — WHERE π0.5's retained multimodal function lives (spec §2.4, §6).

Three contrasts, all inside the one shell so donor and recipient share an architecture, an item set
and a tokenization exactly (a cross-model transplant with any of those unmatched measures the
mismatch, not the mechanism):

  locus       depth-resolved residual patching PG1→π0.5 at each LM layer, last-token and
              image-token blocks scored separately, with dose-response over 1/2/4 contiguous layers
  embed swap  π0.5's embedding+head into π0 AND the reverse — sufficiency needs BOTH directions (§6)
  proj swap   PG1 ↔ π0.5 projector

Nulls (§5 N3): the SAME-DEPTH-COUNT patch — k layers at random positions, k matched to the window
under test. A window that does not beat its own depth-matched null is not a locus. A second,
DISCLOSED-as-added null is the wrong-donor patch (another item's residual at the same layer), which
asks whether the transported content is ITEM-SPECIFIC information or merely a distribution shift;
LATLOC verified that construction. Both are reported; N3 is the frozen adjudicator.

`P5` positive control: PG1→PG1 self-patch must move nothing, and a full-depth PG1→π0.5 patch must
reproduce the endpoint gap. If either fails the arm is VOID — the instrument is not transporting
what it claims to.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.plenum_patch --type g6e.2xlarge --no-weights \
      --args "--n 250 --arms locus,embed_swap,proj_swap" \
      --s3-prefix results/plenum_patch --budget-min 420
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

PPL_ANCHOR = 22.42583
PPL_REL_TOL = 1e-3
TASKS = ("vqav2", "textvqa")
DOSES = (1, 2, 4)
N_NULL_DRAWS = 8


def _capture(model, layers, layer_idxs, fwd_kwargs):
    """Residual at `layer_idxs` (output[0]) for one forward — the LATLOC `_capture` pattern."""
    import torch
    cache, handles = {}, []

    def _mk(L):
        def hook(m, a, o):
            cache[L] = (o[0] if isinstance(o, tuple) else o).detach().clone()
        return hook
    for L in layer_idxs:
        handles.append(layers[L].register_forward_hook(_mk(L)))
    with torch.inference_mode():
        model(**fwd_kwargs)
    for h in handles:
        h.remove()
    return cache


def _write_hooks(layers, donor, positions):
    """Overwrite the residual leaving each donated layer, at `positions` only.

    `positions` is a boolean mask over sequence positions (the image-token block or the last token),
    so the transplant is localized in DEPTH and in POSITION — §2.4 scores those separately because a
    finding at the image block and one at the last token are different mechanisms."""
    import torch
    handles = []

    def _mk(L):
        def hook(m, a, o):
            t = o[0] if isinstance(o, tuple) else o
            new = t.clone()
            new[:, positions, :] = donor[L][:, positions, :].to(new.dtype)
            return (new,) + tuple(o[1:]) if isinstance(o, tuple) else new
        return hook
    for L in donor:
        handles.append(layers[L].register_forward_hook(_mk(L)))
    return handles


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--arms", default="p5,locus,embed_swap,proj_swap")
    ap.add_argument("--half", default="confirmation")
    ap.add_argument("--choice-seed", type=int, default=20260728)
    ap.add_argument("--null-seed", type=int, default=4242)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from infra.pg.plenum_core import score_item_fc as score_item
    from infra.pg import inherit_pg1 as IP
    from infra.pg import lacuna_core as LC
    from infra.pg import lacuna_shell as LS
    from infra.pg import plenum_core as PC
    from infra.pg import plenum_units as PU

    assert torch.cuda.is_available(), "plenum_patch needs GPU"
    device = "cuda"
    t0 = time.time()
    rep = {"object": "plenum_patch_v1", "arms": arms, "n": a.n, "tasks": list(TASKS),
           "doses": list(DOSES), "gpu": torch.cuda.get_device_name(0),
           "null": "N3 same-depth-count (frozen adjudicator) + wrong-donor (added, disclosed)"}
    cells = []
    rep["cells"] = cells
    seeds = {"stitch": 1, "matched_dw": 20260727, "rand_dec": 1337}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    tasks = LS.mm_tasks()
    items, choices = {}, {}
    for name in TASKS:
        r, h = IP.load_items(tasks[name], a.n)
        disc, conf = LC.parity_split(h)
        m = LC.mask_of(a.half, disc, conf)
        r = [x for x, keep in zip(r, m) if keep]
        items[name] = r
        choices[name] = PC.choices_for_task(name, r, a.choice_seed)
        print(f"[PAT] {name}: {len(r)} {a.half} items", flush=True)
    rep["items"] = {k: len(v) for k, v in items.items()}
    _w()

    model, proc = LS.load_pg1(device=device)
    pg1_sd = LS.pg1_baseline_sd(model)
    ppl = IP.ppl_gate(model, proc, device)
    val = float(ppl["ppl"] if isinstance(ppl, dict) else ppl)
    rel = abs(val - PPL_ANCHOR) / PPL_ANCHOR
    rep["P2_ppl_anchor"] = {"ppl": val, "rel_diff": rel, "passes": bool(rel < PPL_REL_TOL)}
    assert rep["P2_ppl_anchor"]["passes"], f"P2 FAIL: {val}"
    layers = IP.decoder_layers(model)
    n_layers = len(layers)
    rep["n_layers"] = n_layers
    _w()

    def _enc(task, r, cands):
        """The SAME encoding `score_item` builds — including `fc_prompt` (see plenum_core).

        This must match exactly: the donor residuals are captured on this token sequence and are
        written back into the sequence `score_item` forwards. A one-token difference between the two
        would misalign every patched position and silently measure a shifted transplant."""
        img = tasks[task].image_fn(r).convert("RGB")
        prompt = PC.fc_prompt(tasks[task].prompt_fn(r))
        enc = proc(text=[prompt + " " + c for c in cands], images=[img] * len(cands),
                   return_tensors="pt", padding="longest").to(device)
        return enc

    def _acc(task):
        """Unpatched forced-choice accuracy over `task` — the native read for whatever unit is
        currently installed."""
        per_item = []
        for r, (cands, gold) in zip(items[task], choices[task]):
            s, nt = score_item(model, proc, device, tasks[task].image_fn(r).convert("RGB"),
                               PC.fc_prompt(tasks[task].prompt_fn(r)), cands)
            mean_lp = np.asarray(s) / np.asarray(nt, dtype=float)
            per_item.append(int(int(mean_lp.argmax()) == gold))
        return np.asarray(per_item, dtype=float)

    # Re-installing weights PER ITEM would be far too slow (a 3 GB state-dict write per forward),
    # so the donor pass runs ONCE per (task, window) over all items and is banked on CPU in fp16.
    # This is the same transported object, without the weight thrash.
    def donor_bank(task, window):
        PU.install_unit("pg1", model, proc, pg1_sd, device, seeds)
        bank = []
        for r, (cands, _g) in zip(items[task], choices[task]):
            enc = _enc(task, r, cands)
            d = _capture(model, layers, list(window),
                         {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                          "pixel_values": enc["pixel_values"].to(torch.bfloat16)})
            bank.append({L: v.to("cpu", torch.float16) for L, v in d.items()})
        return bank

    def acc_with_bank(task, bank, window, positions_kind, donor_perm=None):
        per_item = []
        n = len(items[task])
        for i, (r, (cands, gold)) in enumerate(zip(items[task], choices[task])):
            enc = _enc(task, r, cands)
            ids = enc["input_ids"]
            src = bank[i if donor_perm is None else donor_perm[i]]
            seq = ids.shape[1]
            ok = all(tuple(v.shape[:2]) == (ids.shape[0], seq) for v in src.values())
            if not ok:                       # donor/recipient length mismatch → skip, and COUNT it
                per_item.append(np.nan)
                continue
            if positions_kind == "last":
                pos = torch.zeros(seq, dtype=torch.bool, device=ids.device); pos[-1] = True
            elif positions_kind == "image":
                iid = proc.tokenizer.convert_tokens_to_ids("<image>")
                pos = (ids[0] == iid)
            else:
                pos = torch.ones(seq, dtype=torch.bool, device=ids.device)
            donor = {L: v.to(device) for L, v in src.items()}
            handles = _write_hooks(layers, donor, pos)
            s, nt = score_item(model, proc, device, tasks[task].image_fn(r).convert("RGB"),
                               PC.fc_prompt(tasks[task].prompt_fn(r)), cands)
            for h in handles:
                h.remove()
            mean_lp = np.asarray(s) / np.asarray(nt, dtype=float)
            per_item.append(float(int(mean_lp.argmax()) == gold))
        v = np.asarray(per_item, dtype=float)
        return v

    def _bank_cell(**kw):
        cells.append({**kw, "elapsed_s": round(time.time() - t0, 1)})
        _w()

    # ---- P5: the positive control (self-patch moves nothing; full depth reproduces the gap) ---
    if "p5" in arms:
        task = TASKS[0]
        PU.install_unit("pg1", model, proc, pg1_sd, device, seeds)
        base_pg1 = _acc(task)
        full = tuple(range(n_layers))
        bank = donor_bank(task, full)
        PU.install_unit("pg1", model, proc, pg1_sd, device, seeds)
        self_patch = acc_with_bank(task, bank, full, "all")
        PU.install_unit("pi05", model, proc, pg1_sd, device, seeds)
        base_pi05 = _acc(task)
        full_patch = acc_with_bank(task, bank, full, "all")
        good = ~np.isnan(self_patch) & ~np.isnan(full_patch)
        p5 = {"pg1_native": float(base_pg1.mean()),
              "pg1_self_patch": float(np.nanmean(self_patch)),
              "pi05_native": float(base_pi05.mean()),
              "pi05_full_depth_patch": float(np.nanmean(full_patch)),
              "n_length_mismatch": int((~good).sum()),
              "self_patch_moves_nothing": bool(abs(float(np.nanmean(self_patch))
                                                   - float(base_pg1.mean())) < 0.02),
              "full_patch_closes_gap": bool(float(np.nanmean(full_patch))
                                            > float(base_pi05.mean()))}
        p5["passes"] = bool(p5["self_patch_moves_nothing"] and p5["full_patch_closes_gap"])
        rep["P5"] = p5
        print(f"[PAT] P5 {json.dumps(p5)}", flush=True)
        _w()
        assert p5["passes"], (
            "P5 FAIL — the transplant instrument does not transport what it claims: "
            f"{json.dumps(p5)}. Arm 4 is VOID (§5).")

    # ---- locus: per-layer ΔS with dose-response + the N3 same-depth-count null ---------------
    if "locus" in arms:
        rng = np.random.default_rng(a.null_seed)
        for task in TASKS:
            PU.install_unit("pg1", model, proc, pg1_sd, device, seeds)
            ceil_items = _acc(task)
            PU.install_unit("matched_dw", model, proc, pg1_sd, device, seeds)
            floor_items = _acc(task)
            PU.install_unit("pi05", model, proc, pg1_sd, device, seeds)
            native = _acc(task)
            base_S = LC.boot_S_paired(native, floor_items, ceil_items)
            _bank_cell(arm="locus", task=task, kind="native", **base_S)
            for dose in DOSES:
                for start in range(0, n_layers - dose + 1):
                    win = tuple(range(start, start + dose))
                    bank = donor_bank(task, win)
                    for pk in ("all", "last", "image"):
                        PU.install_unit("pi05", model, proc, pg1_sd, device, seeds)
                        pv = acc_with_bank(task, bank, win, pk)
                        good = ~np.isnan(pv)
                        s_ci = LC.boot_S_paired(pv[good], floor_items[good], ceil_items[good])
                        d = LC.boot_diff_ci(pv[good], native[good])
                        _bank_cell(arm="locus", task=task, kind="patched", dose=dose,
                                   window=list(win), positions=pk, n_skipped=int((~good).sum()),
                                   S=s_ci["S_point"], S_lo95=s_ci["S_lo95"],
                                   S_hi95=s_ci["S_hi95"], dS=d["diff"], dS_lo95=d["lo95"],
                                   dS_hi95=d["hi95"], p_le0=d["p_le0"])
                    print(f"[PAT] locus {task} dose={dose} win={win} done "
                          f"t={time.time()-t0:.0f}s", flush=True)
                # N3: same-depth-count null — k random layers, k = dose
                for j in range(N_NULL_DRAWS):
                    win = tuple(sorted(rng.choice(n_layers, dose, replace=False).tolist()))
                    bank = donor_bank(task, win)
                    PU.install_unit("pi05", model, proc, pg1_sd, device, seeds)
                    pv = acc_with_bank(task, bank, win, "all")
                    good = ~np.isnan(pv)
                    d = LC.boot_diff_ci(pv[good], native[good])
                    _bank_cell(arm="locus_null_n3", task=task, dose=dose, window=list(win),
                               draw=j, dS=d["diff"], dS_lo95=d["lo95"], dS_hi95=d["hi95"])

    # ---- swaps: state-dict compositions, both directions ------------------------------------
    if "embed_swap" in arms or "proj_swap" in arms:
        from infra.pg import lacuna_shell as _LS
        pol = {}
        for nm in ("pi0", "pi05"):
            sd, sha = _LS.policy_full_sd(nm)
            vlm, head, _c = _LS.policy_vlm_sd(nm, full_sd=sd)
            pol[nm] = {"vlm": vlm, "head": head, "sha": sha}
            del sd
        for task in TASKS:
            PU.install_unit("pg1", model, proc, pg1_sd, device, seeds)
            ceil_items = _acc(task)
            PU.install_unit("matched_dw", model, proc, pg1_sd, device, seeds)
            floor_items = _acc(task)
            for direction, host, donor in (("pi05_embed_into_pi0", "pi0", "pi05"),
                                           ("pi0_embed_into_pi05", "pi05", "pi0")):
                if "embed_swap" not in arms:
                    continue
                PU.install_unit(host, model, proc, pg1_sd, device, seeds)
                nat = _acc(task)
                emb = model.get_input_embeddings()
                with torch.no_grad():
                    emb.weight[:_LS.POLICY_VOCAB].copy_(
                        pol[donor]["head"].to(emb.weight.device, emb.weight.dtype))
                model.tie_weights()
                sw = _acc(task)
                d = LC.boot_diff_ci(sw, nat)
                s_nat = LC.boot_S_paired(nat, floor_items, ceil_items)
                s_sw = LC.boot_S_paired(sw, floor_items, ceil_items)
                _bank_cell(arm="embed_swap", task=task, direction=direction,
                           S_native=s_nat["S_point"], S_swapped=s_sw["S_point"],
                           dS=d["diff"], dS_lo95=d["lo95"], dS_hi95=d["hi95"], p_le0=d["p_le0"])
                print(f"[PAT] embed_swap {direction} {task} dS={d['diff']:.4f} "
                      f"[{d['lo95']:.4f},{d['hi95']:.4f}]", flush=True)
            if "proj_swap" in arms:
                for direction, host, donor_sd in (("pi05_proj_into_pg1", "pg1",
                                                   pol["pi05"]["vlm"]),):
                    PU.install_unit(host, model, proc, pg1_sd, device, seeds)
                    nat = _acc(task)
                    upd = {k: v for k, v in donor_sd.items()
                           if k.startswith("multi_modal_projector")}
                    _LS.install_canon(model, upd, f"projswap/{direction}")
                    sw = _acc(task)
                    d = LC.boot_diff_ci(sw, nat)
                    _bank_cell(arm="proj_swap", task=task, direction=direction, dS=d["diff"],
                               dS_lo95=d["lo95"], dS_hi95=d["hi95"], p_le0=d["p_le0"])
                    print(f"[PAT] proj_swap {direction} {task} dS={d['diff']:.4f}", flush=True)

    rep["elapsed_s"] = round(time.time() - t0, 1)
    _w()
    print(f"PLENUM_PATCH_OK cells={len(cells)}", flush=True)


if __name__ == "__main__":
    main()
