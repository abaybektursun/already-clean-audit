"""PLENUM Arm 1 — the FORCED-CHOICE battery, one instrument fanned out over every unit (§2.1, §3.2).

Forced choice is the PRIMARY readout family for this program. The reason is structural, not
stylistic: free generation's null band is zero-width (LACUNA measured VQAv2 soft-accuracy at
EXACTLY 0.0 for both the real-image and the gray-image condition on the π0 re-shell, which makes
Δ_perception identically zero and unable to separate "the capability was consumed" from "the
forward is OOD"). A forced choice has chance = 0.25 BY CONSTRUCTION on every task, so the floor is
real and the contrast has range even where generation floors.

What is NEW here vs `lacuna_mmcloze` (which was VQAv2-only and post-hoc): the construction is
generalised to all four mm tasks (§13 ledger, "forced-choice extension"), and it runs over the
whole 32-unit roster rather than four ladder rows. The per-candidate SCORER is imported verbatim —
the same `score_item` LACUNA banked, including its PaliGemma trailing-'\\n' framing assertions.

Incremental save (`feedback_fleet_incremental_save`): `rep["cells"]` is bound BEFORE the loops and
metrics.json is rewritten after every (unit, task, condition).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.plenum_fc --type g6e.2xlarge --no-weights \
      --args "--units ft-vqav2,ft-refcoco-seg --tasks vqav2,refcoco --n 500" \
      --s3-prefix results/plenum_mde --budget-min 300
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

PPL_ANCHOR = 22.42583      # invcore-FUNC reference; recomputed digit-exact on every load (§0)
PPL_REL_TOL = 1e-3         # relative: the anchor is hardware-dependent (LACUNA's note)
MM_TASKS = ("vqav2", "textvqa", "refcoco", "coco_captions")
N_DEFAULT = {"vqav2": 4000, "textvqa": 2000, "refcoco": 1975, "coco_captions": 2000}


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")      # the xet download wedge
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--units", required=True, help="comma unit ids (plenum_map ids)")
    ap.add_argument("--tasks", default=",".join(MM_TASKS))
    ap.add_argument("--conditions", default="real,gray")
    ap.add_argument("--readout", default="forced_choice",
                    choices=("forced_choice", "generation"),
                    help="forced_choice is the PRIMARY family (§2.1). generation fills the "
                         "behavior/*/generation probes, and is the readout for any faculty whose "
                         "forced-choice cell VOIDs on the N1 power gate — grounding did.")
    ap.add_argument("--batch-size", type=int, default=16, help="generation readout only")
    ap.add_argument("--n", type=int, default=0, help="0 = the frozen per-task counts")
    ap.add_argument("--choice-seed", type=int, default=20260728)
    ap.add_argument("--stitch-seed", type=int, default=1)
    ap.add_argument("--matched-dw-seed", type=int, default=20260727)
    ap.add_argument("--rand-dec-seed", type=int, default=1337)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    units = [u.strip() for u in a.units.split(",") if u.strip()]
    tasks_want = [t.strip() for t in a.tasks.split(",") if t.strip()]
    conds = [c.strip() for c in a.conditions.split(",") if c.strip()]
    assert all(t in MM_TASKS for t in tasks_want), f"unknown task in {tasks_want}"

    if a.readout == "generation":
        # CIDEr (COCO captions) needs pycocoevalcap, which is NOT in the env AMI — the same call
        # `lacuna_ladder` makes for the same reason. Only the generation readout scores a corpus
        # task, so the forced-choice path does not pay for it.
        from infra.aws_pg.jobs._inherit_shared import install_pg
        install_pg()
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()                                     # before transformers (pg rule)
    import numpy as np
    import torch
    from infra.pg.plenum_core import score_item_fc as score_item   # framing-robust; = score_item on VQAv2
    from infra.aws_pg.jobs.invariant_core_func import _blank_image
    from infra.pg import inherit_pg1 as IP
    from infra.pg import lacuna_shell as LS
    from infra.pg import plenum_core as PC
    from infra.pg import plenum_units as PU

    assert torch.cuda.is_available(), "plenum_fc needs GPU"
    device = "cuda"
    t0 = time.time()
    seeds = {"stitch": a.stitch_seed, "matched_dw": a.matched_dw_seed,
             "rand_dec": a.rand_dec_seed}
    rep = {"object": "plenum_fc_v1", "units": units, "tasks": tasks_want, "conditions": conds,
           "readout": a.readout, "n_cand": PC.N_CAND, "chance": PC.CHANCE,
           "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "seeds": seeds,
           "note": "PRIMARY readout family (§2.1): chance is 0.25 by construction, so the floor is "
                   "real and Δ_perception has range even where generation floors."}
    cells, banks = [], {}
    rep["cells"] = cells                                  # bound BEFORE the loops (fleet lesson)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
        (out / "fc_banks.json").write_text(json.dumps(banks, default=str))
    _w()

    # ---- items + choices: ONE frozen sample per task, shared by every unit and condition -------
    tasks = LS.mm_tasks()
    items, hashes, choices = {}, {}, {}
    for name in tasks_want:
        n = a.n or N_DEFAULT[name]
        r, h = IP.load_items(tasks[name], n)
        assert len(set(h)) == len(h), f"{name}: duplicate content hashes after dedup"
        items[name], hashes[name] = r, h
        choices[name] = PC.choices_for_task(name, r, a.choice_seed)
        print(f"[FC] items {name}: n={len(r)} uniq={len(set(h))}", flush=True)
    banks["item_hashes"] = hashes
    banks["gold_idx"] = {k: [g for _c, g in v] for k, v in choices.items()}
    rep["items"] = {k: {"n": len(v)} for k, v in items.items()}
    _w()

    # ---- the shell + the P2 anchor (certifies the INSTRUMENT, before any unit is installed) ----
    model, proc = LS.load_pg1(device=device)
    pg1_sd = LS.pg1_baseline_sd(model)
    ppl = IP.ppl_gate(model, proc, device)
    val = float(ppl["ppl"] if isinstance(ppl, dict) else ppl)
    rel = abs(val - PPL_ANCHOR) / PPL_ANCHOR
    rep["P2_ppl_anchor"] = {"ppl": val, "anchor": PPL_ANCHOR, "rel_diff": rel,
                            "rel_tol": PPL_REL_TOL, "gpu": torch.cuda.get_device_name(0),
                            "passes": bool(rel < PPL_REL_TOL)}
    print(f"[FC] P2 base-ppl={val} rel={rel:.3e}", flush=True)
    assert rep["P2_ppl_anchor"]["passes"], (
        f"P2 FAIL: base ppl {val} vs anchor {PPL_ANCHOR} (rel {rel:.3e}) — not the frozen instrument")
    _w()

    gray_img = _blank_image(128)
    for uid in PU.unit_order(units):
        info = PU.install_unit(uid, model, proc, pg1_sd, device, seeds)
        rep.setdefault("unit_info", {})[uid] = info
        print(f"[FC] installed {uid}: {json.dumps(info, default=str)[:220]}", flush=True)
        _w()
        for name in tasks_want:
            task = tasks[name]
            for cond in conds:
                if a.readout == "generation":
                    # REUSED verbatim: the same greedy-decode + task-own-scorer path the LACUNA
                    # ladder banked, so these cells are directly comparable to that read. This is
                    # the readout for any faculty whose forced-choice cell VOIDs on the N1 power
                    # gate — and it fills the behavior/*/generation probes regardless.
                    t = task if cond == "real" else LS.gray_task(task)
                    rr = IP.eval_pg1(model, proc, t, items[name], device,
                                     batch_size=a.batch_size)
                    cells.append({"unit": uid, "task": name, "condition": cond,
                                  "readout": "generation", "acc": rr["score"], "n": rr["n"],
                                  "corpus": bool(rr["corpus"]), "chance": None,
                                  "elapsed_s": round(time.time() - t0, 1)})
                    banks[f"{uid}|{name}|{cond}"] = (
                        {"corpus": True, "cand": rr["cand"], "ref": rr["ref"]} if rr["corpus"]
                        else {"corpus": False, "per_item": rr["per_item"], "preds": rr["preds"]})
                    print(f"[FC] {uid:>18s} {name:>14s} {cond:>5s} gen={rr['score']:.4f} "
                          f"n={rr['n']}", flush=True)
                    _w()
                    continue
                per_item, lps, nts, n_unscorable = [], [], [], 0
                for i, (r, (cands, gold)) in enumerate(zip(items[name], choices[name])):
                    img = task.image_fn(r).convert("RGB") if cond == "real" else gray_img
                    s, nt = score_item(model, proc, device, img,
                                       PC.fc_prompt(task.prompt_fn(r)), cands)
                    if s is None:
                        # Framing does not hold for this item. Record NaN IN PLACE rather than
                        # dropping the row: every downstream contrast is positionally aligned to
                        # the frozen item order, and a shorter array silently shifts every
                        # subsequent item against the hashes. Counted, aligned, not hidden.
                        n_unscorable += 1
                        per_item.append(float("nan")); lps.append(None); nts.append(None)
                        continue
                    lps.append(s); nts.append(nt)
                    mean_lp = np.asarray(s) / np.asarray(nt, dtype=float)
                    per_item.append(int(int(mean_lp.argmax()) == gold))
                    if (i + 1) % 500 == 0:
                        print(f"[FC] {uid}/{name}/{cond} {i+1}/{len(items[name])} "
                              f"acc={np.mean(per_item):.4f}", flush=True)
                acc = float(np.nanmean(per_item))
                cells.append({"unit": uid, "task": name, "condition": cond, "readout":
                              "forced_choice", "acc": acc, "n": len(per_item),
                              "n_unscorable": n_unscorable,
                              "chance": PC.CHANCE, "elapsed_s": round(time.time() - t0, 1)})
                banks[f"{uid}|{name}|{cond}"] = {"per_item": per_item, "logprobs": lps,
                                                 "ntoks": nts}
                print(f"[FC] {uid:>18s} {name:>14s} {cond:>5s} acc={acc:.4f} "
                      f"(chance {PC.CHANCE})", flush=True)
                _w()

    rep["elapsed_s"] = round(time.time() - t0, 1)
    _w()
    print("PLENUM_FC_OK", json.dumps({f"{c['unit']}|{c['task']}|{c['condition']}": c["acc"]
                                      for c in cells}), flush=True)


if __name__ == "__main__":
    main()
