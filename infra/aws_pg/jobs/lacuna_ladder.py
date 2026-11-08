"""LACUNA Arm 1 — THE LADDER. One shell, one battery, every lineage stage (spec §2.1, §3).

Rows (spec §2.1; builders in `infra.pg.lacuna_shell`):
  R1 gemma2b_qonly · R2 stitch0_s{seed} · R3 pg1 · R4 pi0 · R5 pi05 · R6 matched_dw · R7 rand_dec
  (+ the interface controls R4b/R5b = the policy VLM under PG1's embedding table)

Every multimodal row runs the battery PAIRED {real image, gray-128 image} on IDENTICAL items in
identical order, so Δ_perception is a per-item paired contrast. Per-item scores (and, for the
corpus task, the candidate/reference strings) are banked raw — every §6 gate, band, label and
figure is recomputed from the bank by `lacuna_aggregate`, never from a number in this job's log.

Incremental save (`feedback_fleet_incremental_save`): `rep["cells"]` is bound BEFORE the loop and
metrics.json is rewritten after EVERY (row, task, condition), so a crash 6 hours in keeps the
cells already measured.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.lacuna_ladder --type g6e.8xlarge --no-weights \
      --args "--rows pg1,matched_dw --n-vqa 4000" --s3-prefix results/lacuna_ladder_a --budget-min 420
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

MM_TASKS = ("vqav2", "coco_captions", "textvqa", "refcoco")
# DATA-AVAILABILITY AMENDMENT (2026-07-28, pre-unblind): the spec froze n=2,000 per non-VQA task,
# but `lmms-lab/RefCOCO` testA — the standard REC eval split the battery uses — contains only
# **1,975** unique items after content-hash dedup. 1,975 IS the whole split; the frozen 2,000 was
# never available. Recorded here rather than silently truncating or silently over-requesting.
N_CAP = {"refcoco": 1975}
ALL_ROWS = ("gemma2b_qonly", "stitch0_s1", "stitch0_s2", "pg1", "pi0", "pi05",
            "pi0_pg1embed", "pi05_pg1embed", "matched_dw", "matched_dw_pi05", "rand_dec")
# Rows whose purpose is the interface control only — VQAv2 alone (spec §5 own-vs-base-embed analogue).
CONTROL_ROWS = {"pi0_pg1embed", "pi05_pg1embed"}
PPL_ANCHOR = 22.42583          # invcore-FUNC reference; recomputed on every load, never quoted onward
# The anchor is HARDWARE-DEPENDENT. The invcore/preflight value was measured on an L40S and
# reproduces there to |Δ| = 1.4e-6; the same shell on an L4 (g6.2xlarge) gives 22.4192996, a
# relative shift of 2.9e-4 from kernel-level differences alone. An absolute 1e-3 gate therefore
# fails a CORRECT shell on different silicon, while a wrong shell moves ppl by orders of magnitude
# (the random-decoder floor is ~1e57). The gate is relative at 1e-3, and the GPU NAME is banked so
# the aggregator can refuse to pool rows measured on different hardware — which is the real risk
# here, not the gate: every ladder row is compared to every other, so they must share silicon.
PPL_REL_TOL = 1e-3
# Rows grouped so each policy's checkpoint is fetched ONCE and freed before the next (the cache
# holds one). `--rows all` expands to this order.
ROW_GROUPS = ("pg1", "matched_dw", "pi0", "pi0_pg1embed", "rand_dec",
              "matched_dw_pi05", "pi05", "pi05_pg1embed",
              "stitch0_s1", "stitch0_s2", "gemma2b_qonly")


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")     # the xet download wedge
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default=",".join(ALL_ROWS))
    ap.add_argument("--tasks", default=",".join(MM_TASKS))
    ap.add_argument("--n-vqa", type=int, default=4000)
    ap.add_argument("--n-other", type=int, default=2000)
    ap.add_argument("--conditions", default="real,gray")
    # 32 is a pure THROUGHPUT knob (greedy decode is batch-invariant up to padding, and P1
    # verified the harness reproduces the July read exactly). It fits a 24GB A10G fallback.
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--matched-dw-seed", type=int, default=20260727)
    ap.add_argument("--rand-dec-seed", type=int, default=1337)
    ap.add_argument("--skip-ppl-gate", action="store_true",
                    help="only for a rows=gemma2b_qonly shard (no PG1 shell is loaded there)")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = (list(ROW_GROUPS) if a.rows.strip() == "all"
            else [r.strip() for r in a.rows.split(",") if r.strip()])
    tasks_want = [t.strip() for t in a.tasks.split(",") if t.strip()]
    conds = [c.strip() for c in a.conditions.split(",") if c.strip()]
    assert all(r in ALL_ROWS for r in rows), f"unknown row in {rows}"
    assert all(t in MM_TASKS for t in tasks_want), f"unknown task in {tasks_want}"

    from infra.aws_pg.jobs._inherit_shared import install_pg
    install_pg()               # pycocoevalcap (CIDEr) is not in the AMI
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch
    from infra.pg import inherit_pg1 as IP
    from infra.pg import lacuna_shell as LS

    assert torch.cuda.is_available(), "lacuna_ladder needs GPU"
    device = "cuda"
    t0 = time.time()
    rep = {"object": "lacuna_ladder_v1", "rows": rows, "tasks": tasks_want, "conditions": conds,
           "n_vqa": a.n_vqa, "n_other": a.n_other, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0),
           "seeds": {"matched_dw": a.matched_dw_seed, "rand_dec": a.rand_dec_seed}}
    cells, banks, gates = [], {}, {}
    rep["cells"], rep["gates"] = cells, gates          # bound BEFORE the loops (fleet lesson)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
        (out / "ladder_banks.json").write_text(json.dumps(banks, default=str))
    _w()

    # ---- items: ONE frozen sample per task, shared by every row and condition -----------
    tasks = LS.mm_tasks()
    items, hashes = {}, {}
    for name in tasks_want:
        n = min(a.n_vqa if name == "vqav2" else a.n_other, N_CAP.get(name, 10 ** 9))
        r, h = IP.load_items(tasks[name], n)
        items[name], hashes[name] = r, h
        print(f"[LAC] items {name}: n={len(r)} uniq_hash={len(set(h))}", flush=True)
        assert len(set(h)) == len(h), f"{name}: duplicate content hashes after dedup"
    rep["items"] = {k: {"n": len(v), "hash_head": hashes[k][:3]} for k, v in items.items()}
    banks["item_hashes"] = hashes
    _w()

    # ---- R1 is a pure text forward — no PG1 shell involved ------------------------------
    if "gemma2b_qonly" in rows:
        for name in tasks_want:
            sc = LS.textonly_scores(tasks[name], items[name], device, batch_size=a.batch_size)
            cells.append({"row": "gemma2b_qonly", "task": name, "condition": "none",
                          "corpus": sc["corpus"],
                          "score": (None if sc["corpus"] else
                                    float(sum(sc["per_item"]) / len(sc["per_item"])))})
            banks[f"gemma2b_qonly|{name}|none"] = sc
            print(f"[LAC] R1 {name}: {cells[-1]['score']}", flush=True)
            _w()
        rows = [r for r in rows if r != "gemma2b_qonly"]
    if not rows:
        rep["elapsed_s"] = time.time() - t0
        _w(); print("LACUNA_LADDER_OK (text-only shard)", flush=True); return

    # ---- the PG1 VLM shell + its native snapshot ----------------------------------------
    model, proc = LS.load_pg1(device=device)
    pg1_sd = LS.pg1_baseline_sd(model)
    rep["shell"] = {"repo": LS.PG1_REPO, "n_state_keys": len(pg1_sd),
                    "attn": "eager", "dtype": "bfloat16"}
    if not a.skip_ppl_gate:
        ppl = IP.ppl_gate(model, proc, device)
        val = ppl["ppl"] if isinstance(ppl, dict) else float(ppl)
        rel = abs(float(val) - PPL_ANCHOR) / PPL_ANCHOR
        gates["G2_ppl_anchor"] = {"ppl": float(val), "anchor": PPL_ANCHOR,
                                  "abs_diff": abs(float(val) - PPL_ANCHOR), "rel_diff": rel,
                                  "rel_tol": PPL_REL_TOL, "gpu": torch.cuda.get_device_name(0),
                                  "passes": rel < PPL_REL_TOL}
        print(f"[LAC] G2 base-ppl={val} anchor={PPL_ANCHOR} rel={rel:.3e} "
              f"gpu={torch.cuda.get_device_name(0)}", flush=True)
        assert gates["G2_ppl_anchor"]["passes"], (
            f"G2 FAIL: base ppl {val} vs anchor {PPL_ANCHOR} (rel {rel:.3e} ≥ {PPL_REL_TOL}) — "
            f"the shell is not the frozen instrument")
    _w()

    # Policy weights are ~13 GB in fp32 each, so the cache holds AT MOST ONE at a time and the
    # caller is expected to group a policy's rows together (see ROW_GROUPS). Holding both π0 and
    # π0.5 alongside the PG1 CPU snapshot would need ~33 GB of host RAM and OOMs a 2xlarge.
    pol_cache = {}

    def _policy(nm):
        if nm not in pol_cache:
            for old in list(pol_cache):
                del pol_cache[old]
            import gc
            gc.collect()
            sd, sha = LS.policy_full_sd(nm)
            pol_cache[nm] = (sd, sha)
            print(f"[LAC] fetched {nm} @{sha} ({len(sd)} tensors)", flush=True)
        return pol_cache[nm]

    def build(row):
        LS.restore_pg1(model, pg1_sd)
        if row == "pg1":
            return {"row": "pg1", "note": "native PG1 in shell (CEILING)"}
        if row.startswith("stitch0_s"):
            return LS.build_row_stitch0(model, int(row.split("_s")[1]), pg1_sd)
        if row in ("pi0", "pi05"):
            sd, sha = _policy(row)
            return {**LS.build_row_policy(model, row, embed="own", full_sd=sd), "sha": sha}
        if row.endswith("_pg1embed"):
            nm = row[: -len("_pg1embed")]
            sd, sha = _policy(nm)
            return {**LS.build_row_policy(model, nm, embed="pg1", full_sd=sd), "sha": sha}
        if row.startswith("matched_dw"):
            # POST-FREEZE, PRE-UNBLIND CONTROL (added 2026-07-28, no ladder result seen yet):
            # the frozen R6 is matched to the PG1→π0 ΔW (AMENDMENT-A1) but also serves as π0.5's
            # floor. The 2a byte test already shows π0.5 moved its vision tower ~2× further than
            # π0 did, so if its LM ΔW is also larger, scoring π0.5 against a floor calibrated to a
            # SMALLER perturbation is anti-conservative for it — it would make π0.5 look more
            # `consumed` than a matched floor would. `matched_dw_pi05` is the same construction
            # matched to the PG1→π0.5 ΔW; the frozen R6 stays the primary and this is reported
            # beside it as a disclosed sensitivity check.
            nm = "pi05" if row.endswith("_pi05") else "pi0"
            sd, _ = _policy(nm)
            lm, _h, _c = LS.policy_vlm_sd(nm, full_sd=sd)
            target = {k[len("language_model."):]: v for k, v in lm.items()
                      if k.startswith("language_model.")}
            return {**LS.build_row_matched_dw(model, pg1_sd, target, a.matched_dw_seed),
                    "matched_to": f"PG1→{nm} LM ΔW"}
        if row == "rand_dec":
            return LS.build_row_random_decoder(model, pg1_sd, a.rand_dec_seed)
        raise RuntimeError(f"unknown row {row}")

    for row in rows:
        info = build(row)
        rep.setdefault("row_info", {})[row] = info
        print(f"[LAC] built {row}: {json.dumps(info, default=str)[:300]}", flush=True)
        _w()
        row_tasks = ["vqav2"] if row in CONTROL_ROWS else tasks_want
        for name in row_tasks:
            task = tasks[name]
            for cond in conds:
                t = task if cond == "real" else LS.gray_task(task)
                r = IP.eval_pg1(model, proc, t, items[name], device, batch_size=a.batch_size)
                cell = {"row": row, "task": name, "condition": cond, "score": r["score"],
                        "ci95": r["ci95"], "n": r["n"], "corpus": bool(r["corpus"]),
                        "elapsed_s": round(time.time() - t0, 1)}
                cells.append(cell)
                banks[f"{row}|{name}|{cond}"] = (
                    {"corpus": True, "cand": r["cand"], "ref": r["ref"]} if r["corpus"] else
                    {"corpus": False, "per_item": r["per_item"], "preds": r["preds"]})
                print(f"[LAC] {row:>14s} {name:>14s} {cond:>4s} score={r['score']:.4f} n={r['n']}",
                      flush=True)
                _w()

    rep["elapsed_s"] = time.time() - t0
    _w()
    print("LACUNA_LADDER_OK", json.dumps({c["row"] + "|" + c["task"] + "|" + c["condition"]:
                                          c["score"] for c in cells}, default=str), flush=True)


if __name__ == "__main__":
    main()
