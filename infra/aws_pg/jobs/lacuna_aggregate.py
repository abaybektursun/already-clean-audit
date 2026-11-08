"""LACUNA aggregation + unblind (spec §6, §7, §11). CPU; reads ONLY banked per-item artifacts.

Nothing here re-runs a model. Every band, gate, label, test and grid row is recomputed from the
per-item banks the jobs wrote, so the analysis can be re-run, audited and re-run again after a
critic pass without touching a GPU (`feedback_post_critic_artifact_desync`).

Order is the frozen one (spec §6.1): G1 → G2 → G3 → G4 → labels → G5 → BH. Status flags are
computed LAST from numbers already banked.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.lacuna_aggregate --type r7i.4xlarge --no-weights \
      --args "--half confirmation" --s3-prefix results/lacuna_agg --budget-min 90
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess

S3 = "s3://pg-lineage-769063704696"
MM_TASKS = ("vqav2", "coco_captions", "textvqa", "refcoco")
FLOOR_ROW = "matched_dw"
CEIL_ROW = "pg1"
POLICY_ROWS = ("pi0", "pi05")
PRE_ROWS = ("stitch0_s1", "stitch0_s2", "gemma2b_qonly")
ALT_FLOOR_ROW = "matched_dw_pi05"      # the π0.5-matched floor (post-freeze, pre-unblind control)
CIDER_BOOT = 300


LOCAL_ROOT = {"path": ""}


def _pull(prefix, dst):
    """Sync <prefix>/out/ from S3 — or, in --local-root mode, just point at the local copy."""
    if LOCAL_ROOT["path"]:
        return pathlib.Path(LOCAL_ROOT["path"]) / pathlib.Path(prefix).name
    dst = pathlib.Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "sync", f"{S3}/{prefix}/out/", str(dst), "--region", "us-east-1"],
                   check=True, capture_output=True)
    return dst


def _load(d, name):
    p = pathlib.Path(d) / name
    return json.loads(p.read_text()) if p.exists() else None


def _sub(arr, mask):
    return [x for x, m in zip(arr, mask) if m]


def _cider(cand, ref, idx=None):
    from infra.pg.battery_tasks import _cider_corpus
    if idx is None:
        return _cider_corpus(cand, ref)
    c = {i: cand[str(k)] if str(k) in cand else cand[k] for i, k in enumerate(idx)}
    r = {i: ref[str(k)] if str(k) in ref else ref[k] for i, k in enumerate(idx)}
    return _cider_corpus(c, r)


def cider_S_multi(cfgs, floor, ceil, keys, n_boot=CIDER_BOOT, seed=0):
    """Item-paired bootstrap of S for the corpus metric, for EVERY row at once.

    One resampled key list per draw is shared by all rows AND by the floor/ceiling, so (a) the
    floor and ceiling CIDEr are computed ONCE per draw instead of once per row — 2 + n_rows CIDEr
    per draw instead of 3·n_rows — and (b) the draws are genuinely paired across rows, which is
    what makes row-vs-row comparison of the corpus metric legitimate in the first place.
    `cfgs` = {row: (cand, ref)}. Returns {row: S dict}."""
    import numpy as np
    from infra.pg.lacuna_core import s_point
    rng = np.random.default_rng(seed)
    f0, k0 = _cider(*floor, keys), _cider(*ceil, keys)
    point = {r: s_point(_cider(*c, keys), f0, k0) for r, c in cfgs.items()}
    draws = {r: [] for r in cfgs}
    for _ in range(n_boot):
        idx = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        fb, kb = _cider(*floor, idx), _cider(*ceil, idx)
        for r, c in cfgs.items():
            s = s_point(_cider(*c, idx), fb, kb)
            if s == s:
                draws[r].append(s)
    out = {}
    for r in cfgs:
        d = np.asarray(draws[r])
        out[r] = {"S_point": point[r], "S_lo95": float(np.percentile(d, 2.5)),
                  "S_hi95": float(np.percentile(d, 97.5)), "n_boot_valid": int(d.size),
                  "raw_excess_over_ceiling": float(_cider(*cfgs[r], keys) - k0),
                  "raw_mean": float(_cider(*cfgs[r], keys)), "raw_floor": f0, "raw_ceiling": k0,
                  "draws": d.tolist()}
    return out


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ladder-prefixes", default="lacuna_ladder_a,lacuna_ladder_b,lacuna_ladder_c")
    ap.add_argument("--text-prefixes", default="lacuna_text_a,lacuna_text_b")
    ap.add_argument("--bytes-prefix", default="lacuna_a2_bytes")
    ap.add_argument("--geom-prefix", default="lacuna_a2_geom")
    ap.add_argument("--preflight-prefix", default="lacuna_preflight")
    ap.add_argument("--arm3-prefix", default="")
    ap.add_argument("--work", default="/work/agg")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--cider-boot", type=int, default=CIDER_BOOT)
    ap.add_argument("--halves", default="discovery,confirmation,full")
    ap.add_argument("--local-root", default="",
                    help="read <root>/<prefix>/ instead of S3 (offline dry-run of the analysis)")
    ap.add_argument("--no-install", action="store_true", help="skip the pycocoevalcap shim")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    LOCAL_ROOT["path"] = a.local_root

    if not a.no_install:
        from infra.aws_pg.jobs._inherit_shared import install_pg
        install_pg()           # pycocoevalcap (CIDEr) is not in the AMI
    import numpy as np
    from infra.pg import lacuna_core as LC

    rep = {"object": "lacuna_aggregate_v1", "inputs": {}}
    res = {"per_task": {}, "gates": {}, "tests": {}, "grid": {}}
    rep["results"] = res

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    # ---- pull every bank ---------------------------------------------------------------
    work = pathlib.Path(a.work); work.mkdir(parents=True, exist_ok=True)
    banks, cells, row_info = {}, [], {}
    for pfx in [p.strip() for p in a.ladder_prefixes.split(",") if p.strip()]:
        d = _pull(f"results/{pfx}", work / pfx)
        m, b = _load(d, "metrics.json"), _load(d, "ladder_banks.json")
        if m is None or b is None:
            rep["inputs"][pfx] = "MISSING"
            continue
        rep["inputs"][pfx] = {"cells": len(m.get("cells", [])), "rows": m.get("rows"),
                              "gpu": m.get("gpu")}
        cells += m.get("cells", [])
        row_info.update(m.get("row_info", {}))
        for k, v in b.items():
            if k == "item_hashes":
                # cross-shard pairing tripwire: the rows live in different jobs, so if two shards
                # streamed different items every paired contrast between them is meaningless. A
                # silent `.update()` would hide exactly that.
                prev = banks.setdefault("item_hashes", {})
                for task, hs in v.items():
                    assert task not in prev or prev[task] == hs, (
                        f"{pfx}: item hashes for {task} differ from an earlier shard — the ladder "
                        f"rows are NOT on the same items (n={len(hs)} vs {len(prev[task])})")
                    prev[task] = hs
            else:
                assert k not in banks, f"{pfx}: duplicate bank cell {k} across shards"
                banks[k] = v
    assert "item_hashes" in banks, "no item hashes banked — cannot split"
    # HARDWARE-CONSISTENCY tripwire. Every ladder row is compared to every other, and the shell's
    # own ppl anchor moves by ~3e-4 relative between an L40S and an L4 from kernel differences
    # alone. Pooling rows measured on different silicon would put that shift inside every contrast.
    gpus = {v.get("gpu") for v in rep["inputs"].values() if isinstance(v, dict) and v.get("gpu")}
    rep["ladder_gpus"] = sorted(g for g in gpus if g)
    assert len(rep["ladder_gpus"]) <= 1, (
        f"ladder shards ran on DIFFERENT GPUs {rep['ladder_gpus']} — rows are not comparable; "
        f"re-run the odd shard on the majority hardware")
    _w()

    text_banks, text_meta = {}, {}
    for pfx in [p.strip() for p in a.text_prefixes.split(",") if p.strip()]:
        d = _pull(f"results/{pfx}", work / pfx)
        m, b = _load(d, "metrics.json"), _load(d, "lacuna_text_banks.json")
        if m is None:
            rep["inputs"][pfx] = "MISSING"
            continue
        rep["inputs"][pfx] = {"rows": m.get("rows"), "acc": m.get("acc"), "text_gpu": m.get("gpu")}
        text_banks.update(b or {})
        text_meta.update(m.get("meta", {}))
    tgpus = sorted({v["text_gpu"] for v in rep["inputs"].values()
                    if isinstance(v, dict) and v.get("text_gpu")})
    rep["text_gpus"] = tgpus
    assert len(tgpus) <= 1, (f"text shards ran on DIFFERENT GPUs {tgpus} — the text rows are not "
                             f"comparable; re-run the odd shard on the majority hardware")
    byte_m = _load(_pull(f"results/{a.bytes_prefix}", work / "bytes"), "metrics.json")
    geom_m = _load(_pull(f"results/{a.geom_prefix}", work / "geom"), "metrics.json")
    pre_m = _load(_pull(f"results/{a.preflight_prefix}", work / "pre"), "metrics.json")
    arm3_m = (_load(_pull(f"results/{a.arm3_prefix}", work / "arm3"), "metrics.json")
              if a.arm3_prefix else None)
    _w()

    # ---- G1 / G2 from the preflight ----------------------------------------------------
    g = res["gates"]
    g["G1_shell_identity"] = (pre_m or {}).get("controls", {}).get("P1_harness", {"passes": None})
    g["G2_ppl_anchor"] = (pre_m or {}).get("controls", {}).get("G2_ppl_anchor", {"passes": None})
    g["G4_shell_perception_preflight"] = (pre_m or {}).get("controls", {}).get(
        "G4_shell_perception", {"passes": None})
    g["S1_answer_shuffle"] = (pre_m or {}).get("controls", {}).get("S1_answer_shuffle")
    g["S2_caption_permutation"] = (pre_m or {}).get("controls", {}).get("S2_caption_permutation")
    g["P2_text_wiring"] = (pre_m or {}).get("controls", {}).get("P2_text_wiring", {"passes": None})
    _w()

    def bank(row, task, cond):
        return banks.get(f"{row}|{task}|{cond}")

    halves = [h.strip() for h in a.halves.split(",") if h.strip()]

    # =====================================================================================
    # Arm 1 — per task, per half
    # =====================================================================================
    for task in MM_TASKS:
        hs = banks["item_hashes"].get(task)
        if hs is None:
            res["per_task"][task] = {"status": "NO ITEMS BANKED"}
            continue
        disc, conf = LC.parity_split(hs)
        corpus = task == "coco_captions"
        tblock = {"n_items": len(hs), "corpus": corpus, "halves": {}}
        for half in halves:
            mask = LC.mask_of(half, disc, conf)
            keys = [i for i, m in enumerate(mask) if m]
            hb = {"n": len(keys)}
            fl, ce = bank(FLOOR_ROW, task, "real"), bank(CEIL_ROW, task, "real")
            if fl is None or ce is None:
                hb["status"] = f"missing floor/ceiling row for {task}"
                tblock["halves"][half] = hb
                continue
            if corpus:
                cf = (fl["cand"], fl["ref"])
                cc = (ce["cand"], ce["ref"])
                ck = [str(k) for k in keys]
                gap = _cider(*cc, ck) - _cider(*cf, ck)
                hb["G3_power"] = {"gap": gap, "min_gap": 0.15, "passes": bool(gap >= 0.15),
                                  "note": "CIDEr score-scale equivalent of the 15pp gate"}
                floor_scores = None
            else:
                floor_scores = _sub(fl["per_item"], mask)
                ceil_scores = _sub(ce["per_item"], mask)
                hb["G3_power"] = LC.power_gate(floor_scores, ceil_scores, n_boot=2000, seed=11)
                th = LC.theta_null(floor_scores, ceil_scores, n_boot=a.n_boot, seed=20260727)
                hb["theta_null"] = th
            hb["rows"] = {}
            corpus_todo = {}
            for row in sorted({c["row"] for c in cells}):
                b_real, b_gray = bank(row, task, "real"), bank(row, task, "gray")
                b_none = bank(row, task, "none")
                entry = {}
                src = b_real or b_none
                if src is None:
                    continue
                if corpus:
                    if not src.get("corpus"):
                        continue
                    entry["cider"] = _cider(src["cand"], src["ref"], [str(k) for k in keys])
                    if hb["G3_power"]["passes"] and row not in (FLOOR_ROW, CEIL_ROW):
                        corpus_todo[row] = (src["cand"], src["ref"])
                    if b_gray is not None:
                        entry["cider_gray"] = _cider(b_gray["cand"], b_gray["ref"],
                                                     [str(k) for k in keys])
                        entry["delta_perception"] = entry["cider"] - entry["cider_gray"]
                else:
                    sc_real = _sub(src["per_item"], mask)
                    entry["mean"] = LC.boot_mean_ci(sc_real, n_boot=2000, seed=17)
                    if row not in (FLOOR_ROW,):
                        sci = LC.boot_S_paired(sc_real, floor_scores, ceil_scores,
                                               n_boot=a.n_boot, seed=19)
                        draws = LC.boot_S_draws(sc_real, floor_scores, ceil_scores,
                                                n_boot=a.n_boot, seed=19)
                        entry["S"] = sci
                        entry["p_above_band"] = LC.p_above_band(draws, hb["theta_null"]["theta_null"])
                    if b_gray is not None:
                        entry["delta_perception"] = LC.delta_perception(
                            sc_real, _sub(b_gray["per_item"], mask), n_boot=a.n_boot, seed=23)
                    if src.get("preds"):
                        entry["has_preds"] = True
                hb["rows"][row] = entry
            if corpus and corpus_todo and hb["G3_power"]["passes"]:
                sc = cider_S_multi(corpus_todo, cf, cc, [str(k) for k in keys],
                                   n_boot=a.cider_boot, seed=13)
                # The corpus FLOOR is exactly 0 CIDEr, so its null band is zero-width for the
                # same reason the per-item bands are — θ_null = 0. Computing `p_above_band` here
                # is what lets the corpus tests enter the BH family at all; without it T4/T5 were
                # VOID purely because the corpus branch never produced a p, not because any gate
                # had failed.
                hb["theta_null_corpus"] = 0.0
                for row, s in sc.items():
                    hb["rows"][row]["S"] = {k: v for k, v in s.items() if k != "draws"}
                    hb["rows"][row]["S_draws_n"] = len(s["draws"])
                    hb["rows"][row]["p_above_band"] = LC.p_above_band(s["draws"], 0.0)
            # labels (only after G3; G1/G2/G4 are job-level and checked above)
            if not corpus and hb["G3_power"]["passes"] and not hb["theta_null"]["saturated"]:
                for row, e in hb["rows"].items():
                    if "S" in e:
                        e["label"] = LC.label_row(e["S"], hb["theta_null"]["theta_null"],
                                                  e.get("delta_perception"))
            elif corpus and hb["G3_power"]["passes"]:
                for row, e in hb["rows"].items():
                    if "S" in e:
                        e["label"] = {"label": ("retained" if e["S"]["S_lo95"] >= 0.5 else
                                                "partial" if e["S"]["S_hi95"] >= 0.05 else "consumed"),
                                      "note": "corpus metric: no per-item null band; bands only",
                                      **e["S"]}
            # sensitivity: rescore the π0.5 rows against the π0.5-MATCHED floor when it exists
            alt = bank(ALT_FLOOR_ROW, task, "real")
            if alt is not None and not corpus:
                alt_scores = _sub(alt["per_item"], mask)
                hb["alt_floor_pi05"] = {
                    "floor_mean": float(np.mean(alt_scores)),
                    "primary_floor_mean": float(np.mean(floor_scores)),
                    "power": LC.power_gate(alt_scores, ceil_scores, n_boot=2000, seed=53),
                    "theta_null": LC.theta_null(alt_scores, ceil_scores, n_boot=4000, seed=20260728),
                    "rows": {}}
                for row in ("pi05", "pi05_pg1embed"):
                    b = bank(row, task, "real")
                    if b is None or b.get("corpus"):
                        continue
                    sc = _sub(b["per_item"], mask)
                    s = LC.boot_S_paired(sc, alt_scores, ceil_scores, n_boot=a.n_boot, seed=59)
                    hb["alt_floor_pi05"]["rows"][row] = {
                        "S": s,
                        "label": LC.label_row(s, hb["alt_floor_pi05"]["theta_null"]["theta_null"])}
            tblock["halves"][half] = hb
            print(f"[AGG] {task}/{half}: n={len(keys)} "
                  f"G3={hb['G3_power']['passes']} rows={len(hb['rows'])}", flush=True)
            _w()
        res["per_task"][task] = tblock
        _w()

    # ---- N-controls -----------------------------------------------------------------
    def _mean(row, task, cond="real", half="full"):
        b = bank(row, task, cond)
        if b is None or b.get("corpus"):
            return None
        hs = banks["item_hashes"][task]
        d, c = LC.parity_split(hs)
        return float(np.mean(_sub(b["per_item"], LC.mask_of(half, d, c))))

    n1 = {t: {"floor": _mean(FLOOR_ROW, t), "ceiling": _mean(CEIL_ROW, t)} for t in MM_TASKS
          if _mean(FLOOR_ROW, t) is not None}
    for t, v in n1.items():
        v["gap"] = v["ceiling"] - v["floor"]
        v["null_too_weak"] = bool(v["gap"] < 0.15)
    g["N1_matched_random_floor"] = n1
    n2 = {t: {"rand_dec": _mean("rand_dec", t), "matched_dw": _mean(FLOOR_ROW, t)}
          for t in MM_TASKS if _mean("rand_dec", t) is not None}
    for t, v in n2.items():
        v["ordering_ok"] = bool(v["rand_dec"] <= v["matched_dw"] + 0.02)
    g["N2_random_decoder_ordering"] = n2
    n3 = {t: {"stitch_s1": _mean("stitch0_s1", t), "stitch_s2": _mean("stitch0_s2", t),
              "question_only": _mean("gemma2b_qonly", t, cond="none")}
          for t in MM_TASKS}
    for t, v in n3.items():
        if v["stitch_s1"] is not None and v["question_only"] is not None:
            v["seed_spread"] = (abs(v["stitch_s1"] - v["stitch_s2"])
                                if v["stitch_s2"] is not None else None)
            v["leak"] = bool(v["stitch_s1"] > v["question_only"] + 0.15)
    g["N3_stitch_pre_point"] = n3
    _w()

    # =====================================================================================
    # The confirmatory family (spec §7 + AMENDMENT-A7)
    # =====================================================================================
    HALF = "confirmation" if "confirmation" in halves else halves[-1]
    T = {}

    def _entry(task, row):
        return (res["per_task"].get(task, {}).get("halves", {}).get(HALF, {})
                .get("rows", {}).get(row))

    def _gated(task):
        h = res["per_task"].get(task, {}).get("halves", {}).get(HALF, {})
        return bool(h.get("G3_power", {}).get("passes"))

    for tid, task, row in (("T1", "vqav2", "pi0"), ("T2", "vqav2", "pi05"),
                           ("T4", "coco_captions", "pi0"), ("T5", "coco_captions", "pi05"),
                           ("T15", "textvqa", "pi0"), ("T16", "textvqa", "pi05"),
                           ("T17", "refcoco", "pi0"), ("T18", "refcoco", "pi05")):
        e = _entry(task, row)
        T[tid] = {"what": f"S_{task}({row}) above the null band", "task": task, "row": row,
                  "p": (e.get("p_above_band") if (e and _gated(task)) else None),
                  "S": (e or {}).get("S"),
                  "void_reason": (None if (e and _gated(task)) else
                                  ("G3 power gate failed" if e else "row not measured"))}

    # T3 / T14-style paired policy contrasts
    for tid, task in (("T3", "vqav2"),):
        b0, b5 = bank("pi0", task, "real"), bank("pi05", task, "real")
        if b0 and b5 and not b0.get("corpus") and _gated(task):
            hs = banks["item_hashes"][task]
            d, c = LC.parity_split(hs)
            m = LC.mask_of(HALF, d, c)
            df = LC.boot_diff_ci(_sub(b5["per_item"], m), _sub(b0["per_item"], m),
                                 n_boot=a.n_boot, seed=29)
            T[tid] = {"what": "paired π0.5 − π0 on VQAv2 (registered direction: positive)",
                      "p": df["p_le0"], **df}
        else:
            T[tid] = {"what": "paired π0.5 − π0 on VQAv2", "p": None,
                      "void_reason": "rows missing or G3 failed"}

    # T6 Δ_install
    b_ce = bank(CEIL_ROW, "vqav2", "real")
    pre_best, pre_name = None, None
    for r in PRE_ROWS:
        b = bank(r, "vqav2", "real") or bank(r, "vqav2", "none")
        if b is None or b.get("corpus"):
            continue
        mval = float(np.mean(b["per_item"]))
        if pre_best is None or mval > pre_best[0]:
            pre_best, pre_name = (mval, b), r
    if b_ce and pre_best:
        hs = banks["item_hashes"]["vqav2"]
        d, c = LC.parity_split(hs)
        m = LC.mask_of(HALF, d, c)
        df = LC.boot_diff_ci(_sub(b_ce["per_item"], m), _sub(pre_best[1]["per_item"], m),
                             n_boot=a.n_boot, seed=31)
        T["T6"] = {"what": f"Δ_install = acc(PG1) − max(pre-point) on VQAv2; best pre-point = {pre_name}",
                   "pre_point_row": pre_name, "p": df["p_le0"], **df,
                   "note": "LOWER BOUND on the install — the stitch retains PG1's embedding (A4)"}
    else:
        T["T6"] = {"what": "Δ_install", "p": None, "void_reason": "pre-point rows missing"}

    # T7 / T8 Δ_perception
    for tid, row in (("T7", "pi0"), ("T8", "pi05")):
        e = _entry("vqav2", row)
        dp = (e or {}).get("delta_perception")
        T[tid] = {"what": f"Δ_perception({row}) > 0 on VQAv2",
                  "p": (dp["p_le0"] if dp else None), **(dp or {}),
                  "void_reason": (None if dp else "row/condition not measured")}

    # T9 vision-overlap vs null (per policy) — a sign test over matrices, or terminal if byte-frozen
    for tid, pol in (("T9a", "pi0"), ("T9b", "pi05")):
        blk = ((byte_m or {}).get("results", {}).get("2a", {}) or {}).get(pol)
        if blk is None:
            T[tid] = {"what": f"vision subspace overlap({pol}) above the matched-rotation null",
                      "p": None, "void_reason": "2a not run"}
        elif blk.get("byte_frozen"):
            T[tid] = {"what": f"vision tower {pol} vs PG1", "p": None,
                      "verdict": "H_vision-frozen CONFIRMED (max|Δ| = 0 in stored dtype)",
                      "void_reason": "terminal by the byte test — no subspace test is defined"}
        else:
            prof = blk.get("profile", {})
            k = sum(1 for p in prof.values() if p["left"]["above_null"])
            n = len(prof)
            from math import comb
            p = sum(comb(n, i) * (0.05 ** i) * (0.95 ** (n - i)) for i in range(k, n + 1)) if n else None
            T[tid] = {"what": f"vision subspace overlap({pol}) above the matched-rotation null",
                      "n_matrices": n, "n_above_null": k, "p": p,
                      "median_overlap": blk.get("median_left_overlap"),
                      "median_null_p95": blk.get("median_null_p95")}

    # T10 srank-A vs the random band (sign test over matrices, one-sided BELOW = low-rank)
    c2 = (byte_m or {}).get("results", {}).get("2c")
    if c2:
        per = [r for r in c2["per_matrix"] if r.get("srank_delta") is not None]
        k = sum(1 for r in per if r["srank_delta"] < r["srank_null"]["p2_5"])
        from math import comb
        n = len(per)
        p = sum(comb(n, i) * (0.025 ** i) * (0.975 ** (n - i)) for i in range(k, n + 1)) if n else None
        base_med = c2.get("median_srank_base")
        ratio = c2.get("srank_delta_over_base")
        n_base_below = c2.get("n_base_below_band")
        T["T10"] = {"what": "gemma→PG1 LM ΔW is LOW-RANK (below the Frobenius-matched random band)",
                    "n_matrices": n, "n_below_band": k, "p": p,
                    "median_srank": c2["median_srank_delta"],
                    "band": [c2["median_null_p2_5"], c2["median_null_p97_5"]],
                    "fullrank_like": c2["fullrank_like"],
                    # THE NULL IS UNFAILABLE — disclosed, not swapped.
                    "median_srank_base": base_med, "srank_delta_over_base": ratio,
                    "n_base_below_band": n_base_below,
                    "null_is_diagnostic": (None if n_base_below is None
                                           else bool(n_base_below < 0.5 * n)),
                    "note": ("H_fullrank-A is the registered hypothesis and this p tests its "
                             "NEGATION. READ WITH THE CONTROL: if n_base_below_band is also ~n, "
                             "the Frobenius-matched GAUSSIAN band is cleared by ordinary trained "
                             "weights, so 'below the band' means only 'not Gaussian' and carries "
                             "no information about the UPDATE. The interpretable quantity is then "
                             "srank(ΔW)/srank(W_base): >1 means the update is MORE spread than the "
                             "matrices it edits. The frozen verdict is reported as computed; the "
                             "scientific reading follows the control.")}
    else:
        T["T10"] = {"what": "srank-A", "p": None, "void_reason": "2c not run"}

    # T11 geometry preservation P
    if geom_m and geom_m.get("pooled"):
        pl = geom_m["pooled"]
        T["T11"] = {"what": "2d text-geometry overlap above the window-permuted floor",
                    "p": (None if pl["band_void"] else pl["T11_p_le0"]),
                    "P": pl["P"], "P_label": pl["P_label"],
                    "void_reason": ("band has no range (CEILING−FLOOR lower-CI ≤ 0)"
                                    if pl["band_void"] else None)}
    else:
        T["T11"] = {"what": "P(2d)", "p": None, "void_reason": "2d not run"}

    # T12–T14 Arm 3
    for tid, what in (("T12", "savings(π0) > 0"), ("T13", "savings(π0.5) > 0"),
                      ("T14", "paired savings π0.5 − π0")):
        blk = (arm3_m or {}).get("tests", {}).get(tid)
        T[tid] = blk or {"what": what, "p": None,
                         "void_reason": "Arm 3 not run / VOID (spec §12: first cut if the cap binds)"}

    res["tests"] = T
    res["bh"] = LC.bh_reject({k: v.get("p") for k, v in T.items()}, q=0.05)
    print(f"[AGG] BH m={res['bh']['m']} rejected={res['bh']['rejected']} void={res['bh']['void']}",
          flush=True)
    _w()

    # ---- G5 scope cap ------------------------------------------------------------------
    labels = {}
    dperc = {}
    for pol in POLICY_ROWS:
        labels[pol] = {}
        for task in MM_TASKS:
            e = _entry(task, pol)
            if e and "label" in e:
                labels[pol][task] = e["label"]["label"]
        e = _entry("vqav2", pol)
        dp = (e or {}).get("delta_perception")
        dperc[pol] = (dp["lo95"] if isinstance(dp, dict) else -1.0)
    g["G5_scope_cap"] = LC.scope_cap_g5(labels, dperc) if labels and all(labels.values()) else {
        "g5_fired": False, "note": "not evaluable — some policy/task labels are missing or VOID"}
    _w()

    # ---- the text ladder (coverage; no confirmatory weight — A10) -----------------------
    def _text_items(row, axis):
        """Per-item 0/1 for a text axis. The MMLU-cloze axis banks (Σlogprob, ntok, gold) rather
        than a decision, so that any length-normalisation λ stays a downstream recompute; λ=1 is
        the banked default (`policy_content_readout.cloze_acc_lambda`)."""
        b = (text_banks.get(row) or {}).get(axis)
        if not b:
            return None
        if "per_item" in b:
            return b["per_item"]
        if "logprobs" in b:
            from infra.aws_pg.jobs.policy_content_readout import cloze_acc_lambda
            return [int(x) for x in cloze_acc_lambda(b["logprobs"], b["ntoks"],
                                                     b["correct_idx"], lam=1.0)]
        return None

    if text_banks:
        tl = {"note": "coverage arm — fills matrix cell 4A-cloze; no confirmatory weight (A10). "
                      "S reported against BOTH PG1 (lineage) and base gemma-2b (power)."}
        for axis in ("mcq", "cloze", "rfp_cloze", "rfp_mcq"):
            have = [r for r in text_banks if axis in (text_banks[r] or {})]
            if not have or CEIL_ROW not in have:
                continue
            block = {"rows": {}}
            for denom_name, denom_row in (("vs_pg1", "pg1"), ("vs_base", "base")):
                if denom_row not in have or FLOOR_ROW not in have:
                    continue
                fl = _text_items(FLOOR_ROW, axis)
                ce = _text_items(denom_row, axis)
                if fl is None or ce is None:
                    continue
                pw = LC.power_gate(fl, ce, n_boot=2000, seed=37)
                block[denom_name] = {"power": pw, "rows": {}}
                if pw["passes"]:
                    th = LC.theta_null(fl, ce, n_boot=4000, seed=20260727)
                    block[denom_name]["theta_null"] = th
                for r in have:
                    pi = _text_items(r, axis)
                    if pi is None or len(pi) != len(fl):
                        continue
                    e = {"acc": float(np.mean(pi)),
                         "ci": LC.boot_mean_ci(pi, n_boot=2000, seed=41)}
                    if pw["passes"]:
                        e["S"] = LC.boot_S_paired(pi, fl, ce, n_boot=4000, seed=43)
                        if not block[denom_name]["theta_null"]["saturated"]:
                            e["label"] = LC.label_row(
                                e["S"], block[denom_name]["theta_null"]["theta_null"])
                    block[denom_name]["rows"][r] = e
            tl[axis] = block
        tl["ppl"] = {r: (text_banks[r].get("ppl") or {}).get("ppl") for r in text_banks
                     if "ppl" in (text_banks[r] or {})}
        res["text_ladder"] = tl
        _w()

    # ---- the §11 grid, filled whatever landed ------------------------------------------
    def _fmt(e, key="S"):
        if not e or key not in e:
            return "not measured"
        s = e[key]
        lab = e.get("label", {}).get("label", "")
        return f"{s['S_point']:.3f} [{s['S_lo95']:.3f}, {s['S_hi95']:.3f}] {lab}".strip()

    grid = {
        "1_S_vqa_pi0": _fmt(_entry("vqav2", "pi0")),
        "2_S_vqa_pi05": _fmt(_entry("vqav2", "pi05")),
        "2b_S_textvqa_refcoco": {
            "textvqa_pi0": _fmt(_entry("textvqa", "pi0")),
            "textvqa_pi05": _fmt(_entry("textvqa", "pi05")),
            "refcoco_pi0": _fmt(_entry("refcoco", "pi0")),
            "refcoco_pi05": _fmt(_entry("refcoco", "pi05"))},
        "3_delta_perception": {p: T.get(t, {}).get("diff") for t, p in (("T7", "pi0"), ("T8", "pi05"))},
        "4_pi05_minus_pi0_mm": T.get("T3", {}).get("diff"),
        "5_delta_install_T6": T.get("T6", {}).get("diff"),
        "6_vision_byte_compare": {p: ((byte_m or {}).get("results", {}).get("2a", {})
                                      .get(p, {}) or {}).get("global_max_abs_delta")
                                  for p in POLICY_ROWS},
        "7_srank_A_vs_band": T.get("T10", {}).get("median_srank"),
        "8_P_2d_pooled": T.get("T11", {}).get("P"),
        "9_savings": {t: T.get(t, {}).get("savings") for t in ("T12", "T13")},
        "10_gates_fired": {k: (v.get("passes") if isinstance(v, dict) else v)
                           for k, v in g.items()},
    }
    res["grid"] = grid
    rep["headline_half"] = HALF
    _w()
    print("LACUNA_AGG_OK", json.dumps({"bh": res["bh"], "grid": grid}, default=str)[:3000],
          flush=True)


if __name__ == "__main__":
    main()
