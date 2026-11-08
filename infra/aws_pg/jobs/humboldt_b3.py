"""HUMBOLDT B3 — ICL shift projection (spec §3/B3). GPU, forwards only.

gemma-2b + PG1-text. Three few-shot text tasks (ledger correction 2 — the house text-side battery is
infra/pg/text_eval.py, not battery_tasks.py, whose TASKS are all multimodal):
    MMLU (cais/mmlu, per-subject exemplars), WMDP-bio, WMDP-cyber
all sharing text_eval._q_block as the prompt builder, so 0/1/4/16-shot differ ONLY in prefix length.

Contrast: the identical final query token BLOCK under 0/1/4/16-shot contexts, located by exact
token-block match (the PaliGemma trailing-newline gotcha applies). Record, per layer and per bank
in the battery: ‖UUᵀΔh‖²/‖Δh‖², Δh = h(k-shot) − h(0-shot) at the query positions.

Smoke (§3/B3): Δh between two DIFFERENT queries (not two contexts) must show a different projection
profile than the ICL contrast. If the two are indistinguishable, the contrast is not isolating
context-induced shift — journal and still record, with flag contrast_weak=true.

  pg run infra.aws_pg.jobs.humboldt_b3 --type p5.4xlarge,g6e.8xlarge --no-weights \
      --s3-prefix results/humboldt_b3 --budget-min 180
"""
import argparse
import json
import pathlib

from infra.pg import humboldt_act as A
from infra.pg import humboldt_core as H
from infra.pg import humboldt_smoke as S
from infra.pg import humboldt_weights as W

SHOTS = (0, 1, 4, 16)
TASKS = ("mmlu", "wmdp-bio", "wmdp-cyber")
B3_MODELS = ("gemma-2b", "pg1-pt224")
CONTRAST_WEAK_RATIO = 1.25       # ICL vs cross-query profile distance must exceed this


def _item_id(task, i, question):
    """A collision-free item id. `question[:32]` is NOT one: WMDP-cyber questions share long
    boilerplate prefixes, so distinct items collapse onto the same id — which merges their Δh in
    the per-item accumulator and makes the alignment gate report a mismatch that is really a
    collision. Index + full-question digest is unique by construction and still deterministic."""
    import hashlib
    return f"{task}::{i:04d}::{hashlib.sha1(question.encode()).hexdigest()[:10]}"


def _build_items(task, n_items, n_max_shots):
    """(exemplar_pool, queries) as lists of rendered strings. Exemplars never appear as queries."""
    from datasets import load_dataset
    from infra.pg.text_eval import CHOICES, MMLU_REPO, WMDP_REPO, _q_block, _subject_header
    assert len(CHOICES) == 4
    if task == "mmlu":
        ds = load_dataset(MMLU_REPO, "all")
        by_subj = {}
        for split in ("dev", "validation"):
            for r in ds[split]:
                by_subj.setdefault(r["subject"], []).append(r)
        test_by_subj = {}
        for r in ds["test"]:
            test_by_subj.setdefault(r["subject"], []).append(r)
        out = []
        for subj in sorted(test_by_subj):
            pool = by_subj.get(subj, [])
            if len(pool) < n_max_shots:
                continue
            for r in test_by_subj[subj][:2]:
                out.append({"item_id": _item_id(f"mmlu::{subj}", len(out), r["question"]),
                            "header": _subject_header(subj),
                            "shots": [_q_block(s["question"], s["choices"], s["answer"])
                                      for s in pool[:n_max_shots]],
                            "query": _q_block(r["question"], r["choices"])})
                if len(out) >= n_items:
                    return out
        return out
    cfg = task                    # "wmdp-bio" / "wmdp-cyber" are the config names verbatim
    ds = load_dataset(WMDP_REPO, cfg, split="test")
    rows = [dict(r) for r in ds]
    pool = rows[:n_max_shots]
    qs = rows[n_max_shots:n_max_shots + n_items]
    return [{"item_id": _item_id(cfg, i, r["question"]), "header": "",
             "shots": [_q_block(s["question"], s["choices"], int(s["answer"])) for s in pool],
             "query": _q_block(r["question"], r["choices"])}
            for i, r in enumerate(qs)]


def _locate(tok, full_ids, query_text):
    """Start index of the query block inside the tokenised prompt, by exact token-block match."""
    from infra.aws_pg.jobs.invariant_core_func import _locate_text_block
    q_ids = tok(query_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    return _locate_text_block(full_ids, q_ids), int(q_ids.shape[0])


def _forward_capture(model, proc, cap, tok, prompt, query_text, is_mm, device, blank):
    import torch
    if is_mm:
        inp = proc(text=[prompt], images=[blank], return_tensors="pt", padding=False).to(device)
        kw = {"pixel_values": inp["pixel_values"].to(torch.bfloat16)}
    else:
        inp = proc(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        kw = {}
    ids = inp["input_ids"][0]
    start, n_q = _locate(tok, ids.cpu(), query_text)
    cap.set_window(start, n_q)
    with torch.inference_mode():
        model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"], **kw)
    return dict(cap.buf), A.alignment_hash(ids[start:start + n_q].cpu().tolist()), n_q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=",".join(B3_MODELS))
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--n-items", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--run-id", default="humboldt_b3")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    want_models = [m.strip() for m in a.models.split(",") if m.strip()]
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    if a.smoke:
        want_models, tasks, a.n_items = want_models[:1], tasks[:1], 4

    H.hf_setup()                       # FIRST — XET off + gated-repo auth (§0.2.7)
    import numpy as np
    import torch
    assert torch.cuda.is_available(), "GPU required for Phase B forwards"
    device = "cuda"

    gates = H.Gates(out)
    S.run_all_offline(gates)
    bank_dir = W.fetch_banks(H.cache_dir(out, "banks"))

    manifest = {"arm": "B3", "run_id": a.run_id, "commit_sha": H.commit_sha(),
                "models": want_models, "tasks": tasks, "shots": list(SHOTS),
                "n_items": a.n_items, "seed": a.seed,
                "task_source": "infra/pg/text_eval.py (ledger correction 2)",
                "gpu": torch.cuda.get_device_name(0)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    atlas = H.Atlas(out, arm="B3", run_id=a.run_id)      # bound BEFORE any loop (§0.2.4)
    coverage = []                                        # bound BEFORE any loop (§0.2.4)

    for mid, repo, kind in A.ROSTER:
        if mid not in want_models:
            continue
        model, proc, layers, sha, is_mm = A.load_model(mid, repo, kind, device=device)
        tok = proc.tokenizer if is_mm else proc
        blank = None
        if is_mm:
            from infra.aws_pg.jobs.invariant_core_func import _blank_image
            blank = _blank_image(128)
        cap = A.SiteSlice(layers, device)
        # Stack the battery per (layer, site) into one matrix so a projection is one GEMM, and load
        # every bank exactly once (the naive per-item load would re-read ~750 files per forward).
        order = {site: [(t, bid) for t in A.BANK_TYPES_AT_SITE[site] for bid in H.BANK_IDS]
                 for site in H.SITES_PHASE_B}
        stack = {(L, site): np.concatenate([W.load_bank(bank_dir, t, L, bid)
                                            for t, bid in order[site]], axis=1)
                 for L in range(len(layers)) for site in H.SITES_PHASE_B}

        def _fracs(dh, L, site):
            """‖UUᵀΔh‖²/‖Δh‖² for every bank at (L, site), in order[site] order."""
            den = float((dh ** 2).sum())
            if den == 0:
                return None
            C = dh @ stack[(L, site)]
            nb = len(order[site])
            return (C ** 2).reshape(dh.shape[0], nb, H.BANK_K).sum(axis=(0, 2)) / den

        for task in tasks:
            items = _build_items(task, a.n_items, max(SHOTS))
            assert len(items) >= min(a.n_items, 4), f"{task}: only {len(items)} items"
            acc, cross, align = {}, {}, {}
            prev_h = None
            with cap:
                for it in items:
                    h = {}
                    for k in SHOTS:
                        prompt = it["header"] + "".join(it["shots"][:k]) + it["query"]
                        buf, ah, n_q = _forward_capture(model, proc, cap, tok, prompt,
                                                        it["query"], is_mm, device, blank)
                        h[k] = buf
                        align.setdefault(it["item_id"], {})[k] = ah
                    for k in SHOTS[1:]:
                        for (L, site), h0 in h[0].items():
                            f = _fracs(h[k][(L, site)] - h0, L, site)
                            if f is None:
                                continue
                            for j, (t, bid) in enumerate(order[site]):
                                acc.setdefault((L, site, t, bid, k), []).append(float(f[j]))
                    # cross-query control: Δh between THIS item and the previous one, both 0-shot.
                    if prev_h is not None:
                        for (L, site), h0 in h[0].items():
                            n = min(h0.shape[0], prev_h[(L, site)].shape[0])
                            f = _fracs(h0[:n] - prev_h[(L, site)][:n], L, site)
                            if f is None:
                                continue
                            for j, (t, bid) in enumerate(order[site]):
                                cross.setdefault((L, site, t, bid), []).append(float(f[j]))
                    prev_h = h[0]

            # the token-alignment gate: the query block must be token-IDENTICAL across shot counts.
            bad = [i for i, d in align.items() if len(set(d.values())) != 1]
            gates.require(f"G_shot_alignment__{mid}__{task}", not bad, n_items=len(align),
                          n_mismatched=len(bad), mismatched=bad[:5],
                          criterion="query block token-identical across 0/1/4/16 shots")

            # the B3 smoke (§3/B3): the ICL profile must differ from the cross-query profile.
            keys16 = [k for k in sorted(acc) if k[4] == 16 and k[:4] in cross]
            assert keys16, "no comparable (ICL, cross-query) cells — the smoke cannot be evaluated"
            icl16 = np.array([np.mean(acc[k]) for k in keys16])
            crossv = np.array([np.mean(cross[k[:4]]) for k in keys16])
            dist = float(np.abs(icl16 - crossv).mean() / max(float(np.abs(crossv).mean()), 1e-12))
            weak = bool(1.0 + dist < CONTRAST_WEAK_RATIO)
            gates.record(f"G5_B3_contrast__{mid}__{task}", not weak, profile_distance=dist,
                         profile_ratio=1.0 + dist, required=CONTRAST_WEAK_RATIO,
                         contrast_weak=weak, n_cells=int(icl16.size),
                         note="§3/B3: record either way; a weak contrast is flagged, not dropped")

            for (L, site, t, bid, k), vals in sorted(acc.items()):
                fl = {**H.type_flags(t), "task": task, "site": site, "shots": k,
                      "contrast_weak": weak, "chance": H.BANK_K / H.IN_DIM[t]}
                atlas.add_ci("icl_shift_bank_fraction", vals, layer=L, type=t, bank_id=bid,
                             model_id=mid, model_sha=sha, corpus_id=task, seed=a.seed,
                             token_class=f"{k}shot", flags=fl)
            for (L, site, t, bid), vals in sorted(cross.items()):
                atlas.add_ci("cross_query_shift_bank_fraction", vals, layer=L, type=t,
                             bank_id=bid, model_id=mid, model_sha=sha, corpus_id=task,
                             seed=a.seed, token_class="cross_query_control",
                             flags={**H.type_flags(t), "task": task, "site": site,
                                    "role": "smoke control (different query, same 0-shot context)",
                                    "chance": H.BANK_K / H.IN_DIM[t]})
            coverage.append({"model": mid, "task": task, "n_items": len(items),
                             "contrast_weak": weak, "profile_distance": dist,
                             "status": "recorded"})
            (out / "coverage_b3.json").write_text(json.dumps(coverage, indent=2))
            print(f"[B3] {mid}/{task}: {len(items)} items, weak={weak} ({atlas.n_rows} rows)",
                  flush=True)
            H.s3_sync(out, f"humboldt/{a.run_id}")
        del model, cap
        torch.cuda.empty_cache()

    atlas.close()
    rep = atlas.to_parquet()
    (out / "b3_summary.json").write_text(json.dumps(
        {"manifest": manifest, "atlas": rep, "gates": gates.gates, "coverage": coverage},
        indent=2, default=str))
    H.s3_sync(out, f"humboldt/{a.run_id}")
    print("=" * 78)
    print(f"HUMBOLDT_B3_OK rows={rep['n_rows']} cells={len(coverage)}")


if __name__ == "__main__":
    main()
