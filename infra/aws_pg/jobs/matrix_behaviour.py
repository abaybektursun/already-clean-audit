"""N7 — MENDELEEV behaviour job: every behaviour column of ONE row, both models, both halves
(contract §5; spec §1.2, §3.2, §6.1–6.5, §12.3).

  python -m infra.aws_pg.jobs.matrix_behaviour --row 2 --cols S,S-k,F1,F2,F3,F4,F5,H,K,K-null,RFP,U,ppl,A,TQ,P \
      --halves discovery,confirmation --out /work/out [--smoke] [--preflight-c1 <carrel_metrics.json>] [--secondary-head]
  --load-test-only     row 4: load each model, one forward, binds, exit 0 (S9-pre, B2).
  --after-text         OPT-IN, install rows only: re-read each RESIDENT shell member through a plain text
                       forward (no image) and bank it under `before-shell-text` / `after-text`, so the
                       install contrast has a reading with no shell on EITHER arm. Loads nothing extra.
  --dryrun             weight-free: stub Forward + stub loaders, every stage, incremental writes; prints PASS.

Models are loaded SEQUENTIALLY (`before-text` → `before` → `after` [→ `after-pg1`]); never two resident.
Stages, each appended to out/metrics.json on completion (temp file + os.replace):
  manifest → (per member: throughput → columns) → binds → gates → lost → preflight → done.
Layout: metrics["cols"][col][model][half] = the readout record (contract §4 shape; per-item vectors also in
out/per_item/<col>_<model>_<half>.json), metrics["gates"], metrics["binds"], metrics["controls"],
out/manifest.json, out/controls.json, out/events.jsonl (stage, t, gpu_util + dmon samples).
Half-dependent columns (MMLU halves): S, S-0, S-1, F1–F5, H, K, K-null, P. Half-independent (stored under
"all"): RFP, U (+U-shortcut), A, TQ, ppl, O. Stored under "eval" (the WMDP eval split, rows 7 only):
S_W, K_W — the §6.10 unlearning readouts, off by default; the healthy `before` arm of these two IS the
parent margin the unlearning verdict is a fraction of, and nothing else in the program measures it.
Exit 0 only when every requested column is present for `before` and `after` on every requested half
(a VOID-tokenizer stamp is a present record).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

CHOICES = ("A", "B", "C", "D")
DIGITS = ("1", "2", "3", "4")
YESNO = ("Yes", "No")
ALL_COLS = ["S", "S-k", "F1", "F2", "F3", "F4", "F5", "H", "K", "K-null", "RFP", "U", "ppl", "A", "TQ", "P", "O",
            "S_W", "K_W"]
DEFAULT_COLS = [c for c in ALL_COLS if c not in ("O", "S_W", "K_W")]
HALF_COLS = {"S", "S-0", "S-1", "F1", "F2", "F3", "F4", "F5", "H", "K", "K-null", "P"}
# S_W / K_W are the §6.10 unlearning readouts on the WMDP EVAL SPLIT — neither an MMLU half nor a
# half-independent "all" set — so they are banked under the half key "eval", the spelling
# `matrix_recovery.wmdp_readouts` already uses for the same two readouts at every dose.
EVAL_COLS = {"S_W", "K_W"}
LETTER_SURFACES = {"S": (CHOICES, True), "S-0": (CHOICES, True), "S-1": (CHOICES, True), "F1": (CHOICES, True),
                   "F2": (DIGITS, True), "F3": (YESNO, True), "F4": (CHOICES, False), "U": (CHOICES, True),
                   "S_W": (CHOICES, True)}
FLOOR = {"S": 0.25, "S-0": 0.25, "S-1": 0.25, "F1": 0.25, "F2": 0.25, "F3": 0.5, "F4": 0.25, "F5": 0.25,
         "K": 0.25, "K-null": 0.25, "RFP": 0.27, "U": 0.25, "U-shortcut": 0.25, "A": 0.0, "TQ": 0.25}
PPL_MAX_TOKENS = 120_000
SMOKE_N = 24                     # ≥ 24 items per half-independent set in --smoke (preflight_match / bootstrap need ≥ 20)
MODULE = "infra.aws_pg.jobs.matrix_behaviour"
# `POLICY_ROWS = ("1b-pi0", "1b-pi05", "3c-Act")` stood here "for provenance" after the O dispatch was
# moved to `matrix_models.o_tasks`. It is DELETED. A hand-maintained copy of a registry answer with no
# consumer is not provenance, it is the next editor's trap — the same shape as the dead `floors` table in
# `matrix_models` and the dead `_FLOORS_DEAD_DO_NOT_USE`. The history belongs in git and in the comment
# at the O dispatch, which names what the tuple cost (19 cells). `matrix_audit.audit_constants` now
# compares every remaining hand-typed row set against the registry on every self-test.


def MM_o_tasks(row):
    from infra.pg import matrix_models as MM
    return MM.o_tasks(row)
VQA_PREFIX = "answer en "


# ------------------------------------------------------------------------------------------ io
def _write_json(path, obj):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


class Job:
    def __init__(self, out, row, cols, halves, smoke, n_boot):
        self.out = out
        self.per_item_dir = os.path.join(out, "per_item")
        os.makedirs(self.per_item_dir, exist_ok=True)
        self.events = os.path.join(out, "events.jsonl")
        self.metrics_path = os.path.join(out, "metrics.json")
        self.m = {"kind": "behaviour", "row": row, "cols_requested": cols, "halves": halves, "smoke": smoke,
                  "n_boot": n_boot, "stages": {}, "cols": {}, "gates": {}, "binds": {}, "controls": {},
                  "manifest": {}, "members": [], "t0": time.time()}
        self.sampler = None
        self.write()

    def write(self):
        _write_json(self.metrics_path, self.m)

    def stage(self, name, **fields):
        from infra.pg.util_probe import log_event
        util = self.sampler.mean_sm_last(10) if self.sampler else None
        self.m["stages"][name] = {"t": time.time(), "gpu_util_10min": util, **fields}
        log_event(self.events, stage=name, gpu_util=util, **fields)
        self.write()
        print(f"[N7 {self.m['row']}] stage {name} {fields if fields else ''} (util {util})", flush=True)

    def bank(self, col, model, half, rec):
        self.m["cols"].setdefault(col, {}).setdefault(model, {})[half] = rec
        # smoke 4 (2026-09-03): the letter readout banks a second answer surface from the same forward.
        # In the blank-image shell the processor appends a newline, so the NO-SPACE letters are the
        # on-format answer there; `<col>-native` carries them as a full column so the format artifact and
        # the fine-tune's effect can be separated (spec confound 31). Zero extra compute.
        alt = rec.get("extra", {}).get("alt_surface") if isinstance(rec, dict) else None
        if alt:
            nat = {"col": col + "-native", "acc": alt["acc"], "n": len(alt["per_item"]), "floor": rec["floor"],
                   "per_item": alt["per_item"], "per_item_hash": rec["per_item_hash"],
                   "per_item_scores": alt["per_item_scores"], "pred": alt["pred"],
                   "extra": {"surface": "no-space letters", "letter_ids": alt["letter_ids"],
                             "argmax_in_letters_frac": alt["argmax_in_letters_frac"],
                             "p_any_letter": alt["p_any_letter"], "primary_acc": rec["acc"],
                             # §12.11. This record is hand-built here and omitted `samples`, so every
                             # `-native` column in the program was a number alone — the exact readout the
                             # 2026-09-03 instrument alarm was raised about. `_letter_readout` now banks
                             # the alt surface's own samples (selected on ITS correctness) and they are
                             # carried straight through.
                             "samples": alt["samples"]}}
            self.m["cols"].setdefault(col + "-native", {}).setdefault(model, {})[half] = nat
        _write_json(os.path.join(self.per_item_dir, f"{col}_{model}_{half}.json"), rec)
        self.write()
        acc = rec.get("acc") if isinstance(rec, dict) else None
        print(f"[N7 {self.m['row']}] {col:10s} {model:11s} {half:12s} acc={acc}", flush=True)


# ------------------------------------------------------------------------------------------ items
def _smoke_subset(items, per_subject=1):
    seen, out, pos = {}, [], []
    for i, it in enumerate(items):
        k = seen.get(it["subject"], 0)
        if k < per_subject:
            out.append(it); pos.append(i); seen[it["subject"]] = k + 1
    return out, pos


def build_items(halves, smoke, cols, row=None):
    from infra.pg import matrix_items as MI
    from infra.pg import two_hop_items as TH
    from infra.pg import matrix_models as MM
    items = {"mmlu": {}, "mmlu_pos": {}}
    for h in halves:
        full = MI.mmlu_eval(h)
        items["mmlu"][h], items["mmlu_pos"][h] = (_smoke_subset(full) if smoke else (full, list(range(len(full)))))
    n = SMOKE_N if smoke else None
    if "RFP" in cols:
        r = MI.rfp_items()
        items["rfp"] = {"items": r["items"][:n] if n else r["items"], "exemplars": r["exemplars"], "meta": r["meta"]}
    if "U" in cols:
        b = TH.build_two_hop()
        items["two_hop"] = {**b, "items": b["items"][:n] if n else b["items"], "shortcut": b["shortcut"][:n] if n else b["shortcut"]}
    if "A" in cols:
        items["arith"] = {"items": MI.arith_items()[:n] if n else MI.arith_items(), "exemplars": MI.arith_exemplars()}
    if "TQ" in cols:
        t = MI.triviaqa_items()
        items["tq"] = {"items": t["items"][:n] if n else t["items"], "exemplars": t["exemplars"]}
    if "P" in cols:
        from infra.pg.probe_readout import mmlu_probe_train_items
        items["probe_train"] = mmlu_probe_train_items(n_per_subject=2 if smoke else None)
    if {"S_W", "K_W"} & set(cols):
        w = MI.wmdp_items("eval")
        items["wmdp"] = w[:n] if n else w
    # `o_tasks(row) == []` is the registry's own statement that O is not a per-item vision readout of
    # this pair (policy rows, unlearning rows, TEXT rows, the gated row 4). Building the vision index
    # for such a row is wasted work at best; passing it to `_o_column` is fatal, because a text row's
    # `load_pair` returns `proc: None` and the processor call raises
    # `TypeError: 'NoneType' object is not callable`. That killed row 7a's 19 cells.
    if "O" in cols and (row is None or MM.o_tasks(row)):
        items["vision"] = MI.vision_index_lists()
        if smoke:
            for k in items["vision"]:
                items["vision"][k] = {**items["vision"][k], "indices": items["vision"][k]["indices"][:n],
                                      **({"hashes": items["vision"][k]["hashes"][:n]} if "hashes" in items["vision"][k] else {})}
    return items


def items_manifest(items):
    from infra.pg.matrix_items import _sha256
    man = {}
    for h, its in items["mmlu"].items():
        man[f"mmlu_{h}"] = {"n": len(its), "sha256": _sha256(its), "positions_sha256": _sha256(items["mmlu_pos"][h])}
    for key in ("rfp", "arith", "tq"):
        if key in items:
            man[key] = {"n": len(items[key]["items"]), "sha256": _sha256(items[key]["items"]),
                        "exemplars_sha256": _sha256(items[key]["exemplars"])}
    if "two_hop" in items:
        man["two_hop"] = {"n": len(items["two_hop"]["items"]), "sha256": _sha256(items["two_hop"]["items"]),
                          "shortcut_sha256": _sha256(items["two_hop"]["shortcut"]), "meta": items["two_hop"]["meta"]}
    if "probe_train" in items:
        man["probe_train"] = {"n": len(items["probe_train"]), "sha256": _sha256(items["probe_train"])}
    if "wmdp" in items:
        # the same identity the recovery ladder manifests for its own WMDP eval set: the item hashes
        man["wmdp_eval"] = {"n": len(items["wmdp"]), "sha256": _sha256([it["hash"] for it in items["wmdp"]])}
    if "vision" in items:
        # The identity is the ROWS (indices + content hashes), not the record: `hash_path` says whether the
        # hashes came from the cached 200-row subset or from materialising the split, and hashing the whole
        # record would make the same frozen item set manifest differently on the two paths.
        man["vision"] = {k: {"n": len(v["indices"]), "sha256": _sha256(v["hashes"]),
                             "indices_sha256": _sha256(v["indices"]), "hash_path": v["hash_path"],
                             "dataset": v["dataset"], "split": v["split"], "n_split": v["n_split"]}
                         for k, v in items["vision"].items()}
    return man


# ------------------------------------------------------------------------------------------ column runners
def _letter_ok(fwd, col):
    surfaces, space = LETTER_SURFACES[col]
    counts = [int(c) for c in fwd.letter_token_counts(surfaces, space)]
    return all(c == 1 for c in counts), counts


def _void_tokenizer(col, counts):
    return {"col": col, "status": "VOID-tokenizer", "letter_token_counts": counts, "acc": None, "n": 0,
            "per_item": [], "per_item_hash": [], "per_item_scores": None, "pred": [], "extra": {}}


def _ppl(fwd, max_tokens):
    """The ppl column, through the member's OWN shell forward (spec §1.2 + §6.5).

    Until 2026-09-03 this called `clean_text_perplexity`, which runs `model(input_ids=...)` with no
    pixel_values. On an install row that is a TEXT-ONLY forward of a VLM: neither §6.5's blank-image
    condition (which every other column of the row uses) nor a bare text model. Row 2's `before`
    member banked 18558.6 that way while the same decoder reads 61.28 bare / 7.966 BOS-prefixed.
    `shell_text_perplexity` puts both arms of a row through the identical in-distribution forward;
    the record names it in `forward` so two protocols can never again be read as one number."""
    from infra.pg import text_eval as TE
    r = TE.shell_text_perplexity(fwd, max_tokens=max_tokens)
    return {"col": "ppl", "ppl": r["ppl"], "mean_nll": r["mean_nll"], "n_tokens": r["n_tokens"], "window": r["window"],
            "per_window": r["per_window"], "shell": fwd.shell, "protocol": r["protocol"],
            "forward": r["forward"], "note": r["note"],
            # §12.11: this record carried no `extra` at all, so ppl was the one column in the program
            # reported as a bare number with no text, no prediction and nothing to look at. The samples
            # are the actual wikitext windows the model saw with its actual next-token prediction at the
            # last scored position, best and worst by per-token NLL.
            "extra": {"col": "ppl", "acc": None, "n": r["n_tokens"], "floor": None,
                      "score_units": "per-token NLL / perplexity, NOT an accuracy",
                      "samples": r["samples"]},
            # no-start-token shells (Qwen, row 3a/3b) drop one unscorable target per window; the
            # artifact says so rather than hiding a token-count difference between rows.
            "dropped_leading_target": r["dropped_leading_target"],
            "n_dropped_leading_targets": r["n_dropped_leading_targets"]}


# ------------------------------------------------------------------- §6.5 forward causality (2026-09-04)
# The four texts the causality probe is built from. `z_a` and `z_b` START WITH THE SAME WORD on purpose:
# the span|z token boundary is then identical in both recoveries, so the only thing that differs between
# them is what comes AFTER that boundary — which is exactly the perturbation the gate asks about.
CAUSAL_PROBE = {
    "prompt": "In July 1969 the Apollo 11 mission carried three astronauts to the Moon,",
    "span": "and the lunar module rested on the surface for about twenty one hours",
    "z": ("before the ascent stage lifted off to rejoin the command module.",
          "before the argument over the programme's cost resumed in Washington."),
    "scrambled": "surface twenty the about hours module for one rested lunar and on the",
}


def _causality_probe(fwd, probe=None) -> dict:
    """Is this member's forward AUTOREGRESSIVE? Measured directly, not inferred from a perplexity ratio.

    WHY NOT THE RATIO. `matrix_core.ppl_forward_gate` compares the shell's perplexity with the bare
    reading, and the failure it was written for — a bidirectional mask, so every position can read its
    own target — drives BOTH perplexities toward 1.0 at once. Its lower arm needs `ppl_shell <
    ppl_bare / 20`, perplexity is ≥ 1, and the banked bare reading is 7.966, so that arm cannot fire at
    all: a shell reading the text PERFECTLY is stamped VALID at ratio 0.125. The property in question is
    not a magnitude, so it is measured as a property: does a token's score depend on a token that comes
    after it?

    THE MEASUREMENT, from `Forward.continuation_logprobs` alone — no new forward API, no second copy
    resident, two batched calls:
      lp(P, S·z_a) − lp(P·S, z_a)   the logprob of span S after prompt P, RECOVERED from a sequence in
                                    which S is followed by z_a. Under the chain rule of an
                                    autoregressive model this is lp(S | P) whatever z_a is.
      lp(P, S·z_b) − lp(P·S, z_b)   the same, with a different following run.
      lp(P, S) and lp(P, S_scrambled)   the true span and a word-scrambled span of the same words,
                                    per token — the control that catches the degenerate case where the
                                    forward reads every target and BOTH recoveries are ~0, so the
                                    invariance clause above is satisfied by 0 == 0.
    `matrix_core.causality_gate` scores them; the numbers are banked per member so a reader can see the
    drift and the separation, not just the stamp."""
    p = dict(CAUSAL_PROBE if probe is None else probe)
    from infra.pg import matrix_core as MC
    P, S, (za, zb), scr = p["prompt"], p["span"], p["z"], p["scrambled"]
    joint = fwd.continuation_logprobs([P, P, P, P], [f"{S} {za}", f"{S} {zb}", S, scr])
    tail = fwd.continuation_logprobs([f"{P} {S}", f"{P} {S}"], [za, zb])
    (J_a, _), (J_b, _), (lp_true, n_true), (lp_scr, n_scr) = joint
    (Z_a, _), (Z_b, _) = tail
    span_a, span_b = (J_a - Z_a) / n_true, (J_b - Z_b) / n_true
    g = MC.causality_gate(span_a, span_b, lp_true / n_true, lp_scr / max(n_scr, 1))
    return {**g, "recovered_span_logprob_per_token": {"after_z_a": span_a, "after_z_b": span_b},
            "per_token": {"true": lp_true / n_true, "scrambled": lp_scr / max(n_scr, 1)},
            "raw": {"joint_a": J_a, "joint_b": J_b, "tail_a": Z_a, "tail_b": Z_b,
                    "span_only": lp_true, "scrambled_only": lp_scr,
                    "n_span_tokens": n_true, "n_scrambled_tokens": n_scr},
            "shell": fwd.shell, "probe": p}


def _vqa_norm(s):
    from infra.pg.text_eval import _normalize_answer
    return _normalize_answer(str(s))


def _cand_options(golds, seed=1337):
    """4 options per item: gold + 3 other items' golds (distinct, case-insensitive), shuffled; fixed seed."""
    rng = random.Random(seed)
    out = []
    for i, g in enumerate(golds):
        others = []
        tries = 0
        while len(others) < 3:
            j = rng.randrange(len(golds))
            tries += 1
            assert tries < 10_000, "cannot draw 3 distinct distractors"
            o = golds[j]
            if j != i and o.lower() != g.lower() and o.lower() not in {x.lower() for x in others}:
                others.append(o)
        opts = [g] + others
        rng.shuffle(opts)
        out.append((opts, opts.index(g)))
    return out


def _cand_record(fwd, col, prompts, images, golds, hashes, batch_note):
    """4-way candidate likelihood with REAL images through Forward.continuation_logprobs (per-token mean)."""
    import numpy as np
    opts = _cand_options(golds)
    fp, fc, fi = [], [], []
    for p, im, (o, _) in zip(prompts, images, opts):
        for c in o:
            fp.append(p); fc.append(" " + str(c).strip()); fi.append(im)
    res = []
    bs = int(fwd.batch_size)
    for s in range(0, len(fp), bs):
        res.extend(fwd.continuation_logprobs(fp[s:s + bs], fc[s:s + bs], images=fi[s:s + bs]))
    sums = np.array([r[0] for r in res]).reshape(len(prompts), 4)
    ntok = np.array([r[1] for r in res]).reshape(len(prompts), 4)
    mean = sums / ntok
    pred = mean.argmax(1)
    gold_idx = np.array([g for _, g in opts])
    per_item = (pred == gold_idx).astype(int).tolist()
    from infra.pg.format_readouts import _samples
    return {"col": col, "acc": float(np.mean(per_item)), "n": len(per_item), "floor": 0.25, "per_item": per_item,
            "per_item_hash": hashes, "per_item_scores": mean.tolist(), "pred": pred.tolist(),
            "extra": {"options": [o for o, _ in opts], "gold_idx": gold_idx.tolist(), "cloze_logprobs": sums.tolist(),
                      "cloze_ntoks": ntok.tolist(), "prompt_example": prompts[0], "note": batch_note,
                      # §12.11: without this every O number in the program is unquotable, and four
                      # recovery hypotheses consume O. The candidates and their per-token-mean scores
                      # are what this readout actually chose between, so they go in the sample.
                      "samples": _samples(prompts, per_item, pred, gold_idx, mean.tolist(),
                                          options=[o for o, _ in opts],
                                          extra_per_item={"gold_text": [o[g] for o, g in opts],
                                                          "candidate_sum_logprob": sums.tolist(),
                                                          "candidate_n_tokens": ntok.tolist()})}}


def cand_column(fwd, task, rows, hashes):
    """The 4-way candidate-likelihood O record for `task` on a PALIGEMMA-shell model — the ONE
    implementation of this readout, called by the behaviour cell here and re-called by the recovery
    ladder at every dose (`matrix_recovery.o_record`). It has to be one implementation: the D18
    identity gate pairs the two per item and demands ≥ 99 % agreement, which is only meaningful if the
    prompts, the images, the golds and the distractor draw are literally the same code."""
    from infra.pg import battery_tasks as B
    note = "continuation scored as prompt + ' ' + text before the processor's trailing newline (the C1 K convention)"
    if task == "vqav2":
        prompts = [VQA_PREFIX + r["question"] for r in rows]
        images = [r["image"] for r in rows]
        golds = [r["multiple_choice_answer"] for r in rows]
    else:
        assert task == "textvqa", task
        prompts = [B.TEXTVQA_TASK.prompt_fn(r) for r in rows]
        images = [B.TEXTVQA_TASK.image_fn(r) for r in rows]
        golds = [B._textvqa_majority(r) for r in rows]
    return _cand_record(fwd, f"O-{task}-cand", prompts, images, golds, list(hashes), note)


# The O record keys each shell banks under `cols.O.<member>.all`. They are NOT the same set — PaliGemma
# scores VQAv2/TextVQA by exact match through the e4 battery, Molmo2 by the OFFICIAL VQA soft accuracy
# through N14 — and that is exactly why the verdicts consume neither. `O_VERDICT_KEY` is the ONE key
# common to both and the ONE key any verdict reads (spec §1.2 v2.3, D4: exact match collapses under
# answer-format drift, so it is reported beside and never scored). `_o_column` asserts its output
# against this table, so a key a consumer names can never again be a key no producer writes.
O_KEYS = {"paligemma": ("refcoco", "textvqa", "textvqa_em", "vqav2_em", "vqav2_cand", "textvqa_cand"),
          "molmo2": ("refcoco", "vqav2", "textvqa", "vqav2_cand", "textvqa_cand")}
O_VERDICT_KEY = "vqav2_cand"


def _o_column_molmo2(fwd, repo, rows):
    """O for a Molmo2-shell row through N14 `infra/pg/vlm_skill_eval`: its OWN prompt conventions and
    its OWN shell (`Molmo2SkillForward`, whose `_encode` binds the item's real image). `fwd` is built
    and template-gated by the caller, which is where the processor lives.

    FIXED 2026-09-03 (SEAMS audit). This branch read `getattr(VS, name)(member, vision[...])` for the
    five column names, and `vlm_skill_eval` defines none of them as callables of that shape: `refcoco`
    exists with a different signature, and `vqav2` / `textvqa` / `vqav2_cand` / `textvqa_cand` are
    COLUMN names produced by the dispatcher `evaluate(col, fwd, rows, prompts)`. Row 3's entire O
    column was dead code — an AttributeError after the models were already resident. Only the dry-run's
    wholesale `_o_column` stub kept it looking green."""
    from infra.pg import vlm_skill_eval as VS
    prompts = VS.molmo2_prompts()
    out = {}
    for col in O_KEYS["molmo2"]:
        task = col[:-len("_cand")] if col.endswith("_cand") else col
        out[col] = VS.evaluate(col, fwd, rows[task], prompts)
    # N14 acceptance (spec §10) — a GATE, not a note (2026-09-03 ruling). It used to be BANKED and never
    # asserted, on the argument that a ±3-point band fails a perfect harness about a fifth of the time.
    # That argument was right about ±3 and wrong about the gate: the spec's tolerance is ±6 points
    # (≈ 2.4 binomial SE at n=200, p≈0.86), and at ±6 the band is a real test. So it FIRES here.
    #
    # Firing means: the O records of this member are stamped `status = "VOID-acceptance"`, which is what
    # the offline verdict script keys on to refuse to score the row's O column. It does NOT raise — the
    # cell's other columns are untouched by a vision-anchor miss, so the failure is banked loud (in every
    # O record, in the printed line below, and in `job.m["gates"]["o_acceptance"]` via the caller) and
    # the job continues. A silent pass is the failure mode we are removing; a killed job is not the fix.
    failed = []
    for task in ("refcoco", "vqav2", "textvqa"):
        acc = (VS.acceptance(repo, task, out[task]) if repo in VS.PUBLISHED
               else {"note": f"no published-anchor table for {repo}", "pass_": None,
                     "verdict": f"N/A — no published-anchor table for {repo}"})
        out[task]["extra"]["acceptance"] = acc
        if acc.get("pass_") is False:
            failed.append(acc["verdict"])
    if failed:
        for col in out:
            out[col]["status"] = "VOID-acceptance"
            out[col].setdefault("extra", {})["acceptance_failed"] = failed
        print(f"[N7] O ACCEPTANCE FAIL on {repo}: {failed} — this member's O column is VOID-acceptance "
              f"(unusable, not scored); the job continues", flush=True)
    return out


def _o_column(member, vision, batch):
    """O for one VLM row, dispatched on the shell. PaliGemma: the e4_capability battery (RefCOCO
    acc@0.5, TextVQA ANLS + EM, VQAv2 EM, greedy generate with the model's processor on REAL images)
    plus the VQAv2/TextVQA 4-way candidate likelihood. Molmo2: N14 `infra/pg/vlm_skill_eval`. Both end
    at `vqav2_cand` — the O every verdict consumes (§1.2) — and the return is asserted against
    `O_KEYS`."""
    import numpy as np
    # `MI` was read as a module global here and this module has none: every REAL O column would have
    # died with `NameError: name 'MI' is not defined` after both models were already resident. The
    # dry-run cannot see it because `_o_column` is stubbed wholesale there, and no O cell has banked
    # yet (wave-1 was killed before its vision packs ran), so it had never fired. Found 2026-09-03.
    from infra.pg import matrix_items as MI
    from infra.pg.format_readouts import _samples
    fwd, model, proc, device = member["fwd"], member["model"], member["proc"], member["fwd"].device
    rows = {}
    for task, spec in vision.items():                  # cached 200-row subset if present, else the full split
        rows[task] = MI.vision_rows(task)[:len(spec["indices"])]
        assert len(spec["hashes"]) == len(rows[task]), (task, len(spec["hashes"]), len(rows[task]))
        assert [r["_index"] for r in rows[task]] == list(spec["indices"]), \
            f"O-{task}: the loaded rows are not the index list this run banked"
    if member["shell"] == "molmo2":
        from infra.pg import vlm_skill_eval as VS
        skill = VS.Molmo2SkillForward(model, proc, device)
        skill.batch_size = batch
        VS.molmo2_template_parity(proc)                # the card's static rendering IS the processor's, or raise
        out = _o_column_molmo2(skill, member["repo"], rows)
        assert set(out) == set(O_KEYS["molmo2"]), (sorted(out), sorted(O_KEYS["molmo2"]))
        return out
    from infra.aws_pg.jobs import e4_capability as E4
    from infra.pg import battery_tasks as B
    out = {}
    # RefCOCO acc@0.5 and TextVQA ANLS: the battery's own TaskSpecs, verbatim
    for task, tspec in (("refcoco", B.REFCOCO_TASK), ("textvqa", B.TEXTVQA_TASK)):
        preds = E4._generate(model, proc, tspec, rows[task], device, batch)
        sc = E4._per_item_scores(tspec, rows[task], preds)
        prompts = [tspec.prompt_fn(r) for r in rows[task]]
        gold_txt = [str(tspec.target_fn(r)) for r in rows[task]]
        rec = {"col": f"O-{task}", "acc": float(sc.mean()), "n": int(sc.size), "floor": 0.0, "per_item": sc.tolist(),
               "per_item_hash": list(vision[task]["hashes"]), "per_item_scores": None, "pred": preds,
               "extra": {"metric": tspec.metric_name, "prompt": tspec.prompt_prefix,
                         "samples": _samples(prompts, sc.tolist(), preds, gold_txt, decoded=preds)}}
        if task == "textvqa":
            golds = [B._textvqa_majority(r) for r in rows[task]]
            em = [int(_vqa_norm(p) == _vqa_norm(g)) for p, g in zip(preds, golds)]
            rec["extra"].update({"anls_per_item": sc.tolist(), "em_acc": float(np.mean(em)), "em_per_item": em})
            out["textvqa_em"] = {**rec, "col": "O-textvqa-em", "acc": float(np.mean(em)), "per_item": em,
                                 "extra": {**rec["extra"],
                                           "samples": _samples(prompts, em, preds, golds, decoded=preds)}}
        out[task] = rec
    # VQAv2 EM: PaliGemma's VQA prefix "answer en {question}", greedy 8 tokens, vs the majority answer
    vq = B.TaskSpec(name="vqav2", dataset=vision["vqav2"]["dataset"], train_split="train", eval_split="validation",
                    prompt_prefix=VQA_PREFIX, metric_name="exact match vs multiple_choice_answer",
                    metric_source="VQAv2 (Goyal 2017) majority answer; EM after text_eval._normalize_answer",
                    prompt_source="big_vision proj/paligemma/transfers/vqav2.py PREFIX='answer en '",
                    target_fn=lambda r: r["multiple_choice_answer"], prompt_fn=lambda r: VQA_PREFIX + r["question"],
                    score_fn=lambda p, r: float(_vqa_norm(p) == _vqa_norm(r["multiple_choice_answer"])), max_new_tokens=8)
    preds = E4._generate(model, proc, vq, rows["vqav2"], device, batch)
    sc = E4._per_item_scores(vq, rows["vqav2"], preds)
    out["vqav2_em"] = {"col": "O-vqav2-em", "acc": float(sc.mean()), "n": int(sc.size), "floor": 0.0, "per_item": sc.tolist(),
                       "per_item_hash": list(vision["vqav2"]["hashes"]), "per_item_scores": None, "pred": preds,
                       "extra": {"metric": vq.metric_name,
                                 "samples": _samples([vq.prompt_fn(r) for r in rows["vqav2"]], sc.tolist(), preds,
                                                     [str(vq.target_fn(r)) for r in rows["vqav2"]], decoded=preds)}}
    # candidate likelihood (gold + 3 other items' golds) on the same rows through the Forward — the
    # SHARED `cand_column`, which the recovery ladder re-runs verbatim at every dose (D18 pairs them).
    for task in ("vqav2", "textvqa"):
        out[f"{task}_cand"] = cand_column(fwd, task, rows[task], vision[task]["hashes"])
    assert set(out) == set(O_KEYS["paligemma"]), (sorted(out), sorted(O_KEYS["paligemma"]))
    return out


def _wmdp_columns(job, fwd, model_key, cols, items):
    """S_W / K_W — the §6.10 unlearning readouts on the WMDP eval split, for BOTH members of a row-7
    pair, banked under `cols.<col>.<member>.eval`.

    Why they live here and not only in the recovery ladder (2026-09-03): the §6.10 safety verdict is
    "did unlearning remove the skill (S_W at floor) while the knowledge survives (K_W or P_W retained
    as a fraction of the PARENT's above-floor margin)". The PARENT is this row's healthy `before`
    model, which only the behaviour cell loads. Without these columns `matrix_recovery.identity_gate`
    stamped "ABSENT in behaviour cell" on every real run (a dose-0 gate that could not fire) and
    `matrix_recovery.fractions` stamped "ABSENT (parent record not in the behaviour cell)" on
    S_W_rec / K_W_rec, so `unlearn_verdict` had no reachable path to a number at all.

    The construction MIRRORS `matrix_recovery.wmdp_readouts` line for line — same prompts
    (`_q_block` / `_q_block_cloze`), same 0-shot, same letter ids, same item hashes — because the D18
    identity gate pairs the two records per item and demands ≥ 99 % agreement, which is only
    meaningful if both sides are literally the same readout. The one addition is the second letter
    surface every letter readout in this job now banks (the 2026-09-03 instrument ruling); it rides
    in `extra.alt_surface` and leaves the primary record identical."""
    from infra.pg import format_readouts as FR
    from infra.pg.text_eval import _q_block, _q_block_cloze
    hashes = [it["hash"] for it in items]
    golds = [it["answer"] for it in items]
    if "S_W" in cols:
        ok, counts = _letter_ok(fwd, "S_W")
        rec = (FR._letter_readout(fwd, "S_W", [_q_block(it["question"], it["choices"]) for it in items],
                                  fwd.letter_ids(CHOICES, True), golds, hashes,
                                  {"shots": 0, "set": "wmdp-eval"}, alt_ids=fwd.letter_ids(CHOICES, False))
               if ok else _void_tokenizer("S_W", counts))
        job.bank("S_W", model_key, "eval", rec)
    if "K_W" in cols:
        job.bank("K_W", model_key, "eval",
                 FR._cloze_readout(fwd, "K_W", [_q_block_cloze(it["question"]) for it in items],
                                   [it["choices"] for it in items], golds, hashes,
                                   {"shots": 0, "set": "wmdp-eval"}))


def run_member(job, member, model_key, cols, halves, items, smoke, probe_state):
    """Every requested column on ONE resident model. Returns nothing; banks into job.m."""
    from infra.pg import format_readouts as FR
    from infra.pg import probe_readout as PR
    from infra.pg import util_probe as UP
    fwd = member["fwd"]
    job.m["members"].append({"which": model_key, "repo": member["repo"], "shell": member["shell"], "info": member["info"]})
    # throughput probe (N13) on this model's own S prompts
    h0 = halves[0]
    probe_prompts = [FR.s_prompt(it, FR._shots(FR.MI.mmlu_dev_shots, it["subject"], 5)) for it in items["mmlu"][h0]]
    tp = UP.choose_batch(fwd, probe_prompts[:max(len(probe_prompts), 4)], start=4, max_bs=8 if smoke else 64)
    job.stage(f"throughput_{model_key}", batch=tp["batch"], prompts_per_s=tp["prompts_per_s"], stop=tp["stop"])
    tok_stats = {}
    for half in halves:
        its = items["mmlu"][half]
        for col in ("S", "S-0", "S-1", "F1", "F2", "F3", "F4"):
            if col not in cols:
                continue
            ok, counts = _letter_ok(fwd, col)
            if not ok:
                job.bank(col, model_key, half, _void_tokenizer(col, counts))
                continue
            if col in ("S", "S-0", "S-1"):
                rec = FR.S(fwd, its, {"S": 5, "S-0": 0, "S-1": 1}[col])
                if col == "S":
                    tok_stats[half] = FR.tokenizer_gate_stats(fwd, rec)
            else:
                rec = getattr(FR, col)(fwd, its, 5)
            job.bank(col, model_key, half, rec)
        if "F5" in cols:
            job.bank("F5", model_key, half, FR.F5(fwd, its, 5))
        if "K" in cols:
            job.bank("K", model_key, half, FR.K(fwd, its, 5))
        if "K-null" in cols:
            job.bank("K-null", model_key, half, FR.K_shuffle_null(fwd, its, 5))
        if "H" in cols:
            hs = {c: FR.H_record(job.m["cols"][c][model_key][half]) for c in ("S", "S-0", "S-1", "F1", "F2", "F4")
                  if c in job.m["cols"] and half in job.m["cols"][c].get(model_key, {})
                  and job.m["cols"][c][model_key][half].get("status") != "VOID-tokenizer"}
            job.bank("H", model_key, half, {"col": "H", "per_format": hs})
    if tok_stats:
        job.m["gates"].setdefault("tokenizer_stats", {})[model_key] = tok_stats
    if "RFP" in cols:
        job.bank("RFP", model_key, "all", FR.RFP_readout(fwd, items["rfp"]["items"], items["rfp"]["exemplars"]))
    if "U" in cols:
        ok, counts = _letter_ok(fwd, "U")
        if ok:
            job.bank("U", model_key, "all", FR.U(fwd, items["two_hop"], 5))
            job.bank("U-shortcut", model_key, "all", FR.U_shortcut(fwd, items["two_hop"], 5))
        else:
            job.bank("U", model_key, "all", _void_tokenizer("U", counts))
            job.bank("U-shortcut", model_key, "all", _void_tokenizer("U-shortcut", counts))
    if {"S_W", "K_W"} & set(cols):
        _wmdp_columns(job, fwd, model_key, cols, items["wmdp"])
    if "A" in cols:
        job.bank("A", model_key, "all", FR.A(fwd, items["arith"]["items"], items["arith"]["exemplars"]))
    if "TQ" in cols:
        job.bank("TQ", model_key, "all", FR.TQ(fwd, items["tq"]["items"], items["tq"]["exemplars"]))
    if "ppl" in cols:
        job.bank("ppl", model_key, "all", _ppl(fwd, 5_000 if smoke else PPL_MAX_TOKENS))
        # The §6.5 causality gate on THIS member's own forward, while it is resident (two batched calls;
        # nothing extra is loaded). It licenses the same `ppl` column the ratio gate licenses and it is
        # the only one of the two that can see a non-causal mask — see `_causality_probe`.
        job.m["gates"].setdefault("causality", {})[model_key] = _causality_probe(fwd)
    if "O" in cols:
        # Ask the frozen registry, not a hand-maintained tuple. `POLICY_ROWS` duplicated what
        # `matrix_models.o_tasks` already asserts and omitted every TEXT row, so row 7a
        # (zephyr-7b-beta -> Zephyr_RMU) reached `_o_column` with `proc: None` and died on
        # `TypeError: 'NoneType' object is not callable`, taking 19 cells with it. Rows 7b and 4 carry
        # the identical latent crash and would have reproduced it verbatim when unparked.
        if not MM_o_tasks(job.m["row"]):
            job.bank("O", model_key, "all",
                     {"status": f"not measured here — o_tasks({job.m['row']!r}) == [] "
                                "(policy / unlearning / text row, or the gated row 4)"})
        else:
            job.bank("O", model_key, "all", _o_column(member, items["vision"], batch=8 if not smoke else 4))
    if "P" in cols and model_key in ("before", "after"):
        layers = PR.probe_layers(fwd.n_layers)
        feats = PR.probe_features(fwd, items["probe_train"], {h: items["mmlu"][h] for h in halves}, layers, 5)
        probe_state[model_key] = feats
        job.stage(f"probe_features_{model_key}", layers=layers, n_train=feats["train"]["n"])
        if model_key == "after":
            rec = PR.probe_column(probe_state["before"], feats, n_boot=job.m["n_boot"],
                                  save_dir=os.path.join(job.out, "probe"))
            rec = PR.finalize_with_random_init(rec, probe_state["random_init"])
            for half in halves:
                dec = rec["halves"][half][str(rec["decisive_layer"])]["after"]
                job.bank("P", "after", half, {"col": "P", **rec["halves"][half], "layers": layers,
                                              # §12.11: P is banked keyed by LAYER and had no `extra` at
                                              # all, so `samples_stage` collected nothing and every P
                                              # number in the program was unquotable. The decisive
                                              # layer's after-probe is the arm the verdict reads.
                                              "extra": {"col": "P", "acc": dec["acc"], "n": dec["n"],
                                                        "floor": 0.25, "samples": dec["samples"],
                                                        "decisive_layer": rec["decisive_layer"]},
                                              "decisive_layer": rec["decisive_layer"], "n_train": rec["n_train"],
                                              # the lower bound's own provenance travels WITH the column:
                                              # how far the re-initialised decoder moved the features, and
                                              # what it re-drew. Without it a reader cannot tell a real
                                              # bound from the wave-1 no-op by looking at the artifact.
                                              "random_init_feature_shift": rec["random_init_feature_shift"],
                                              "random_init_reinit": rec["random_init_reinit"]})
                job.bank("P", "before", half, {"col": "P", "note": "P is one paired record; see cols.P.after", "layers": layers})


def _random_init_last(job, member, cols, halves, items, probe_state):
    """DESTRUCTIVE (re-inits the resident healthy decoder) — the last act on `before`.

    `feats_healthy` is this same model's features from `run_member`, and passing them is what makes
    the control falsifiable: `random_init_bound` asserts the re-initialised decoder's residuals are
    not bit-identical to the healthy ones. Wave-1 banked 60 records where they were (2026-09-03)."""
    from infra.pg import probe_readout as PR
    if "P" not in cols:
        return
    layers = PR.probe_layers(member["fwd"].n_layers)
    probe_state["random_init"] = PR.random_init_bound(member["fwd"], items["probe_train"],
                                                      {h: items["mmlu"][h] for h in halves}, layers, 5,
                                                      feats_healthy=probe_state["before"])
    ri = probe_state["random_init"]
    job.stage("probe_random_init", n_reinit_params=ri["n_reinit_params"], n_kept=ri["reinit"]["n_kept"],
              feature_shift={L: round(v["mean_abs_delta"], 4) for L, v in ri["feature_shift"].items()})


# ------------------------------------------------------------------------------------------ gates / lost / preflight
def _rec(job, col, model, half):
    return job.m["cols"].get(col, {}).get(model, {}).get(half)


def gates_stage(job, cols, halves):
    from infra.pg import format_readouts as FR
    from infra.pg import matrix_core as MC
    g = job.m["gates"]
    ts = g.get("tokenizer_stats", {})
    for half in halves:
        sb, sa = _rec(job, "S", "before", half), _rec(job, "S", "after", half)
        if sb is None or sa is None:
            continue
        if sb.get("status") == "VOID-tokenizer" or sa.get("status") == "VOID-tokenizer":
            g.setdefault("tokenizer", {})[half] = {"S_family": "VOID-tokenizer",
                                                   "letter_token_counts": {"before": sb.get("letter_token_counts"), "after": sa.get("letter_token_counts")}}
        else:
            g.setdefault("tokenizer", {})[half] = FR.tokenizer_gate(ts["before"][half], ts["after"][half])
        if "K-null" in cols:
            g.setdefault("k_null", {})[half] = {m: MC.k_shuffle_null(_rec(job, "K-null", m, half)["acc"]) for m in ("before", "after")}
        # shell parity (§6.5, D15): healthy LM text-only vs the shell, S and K, on this half
        st, kt = _rec(job, "S", "before-text", half), _rec(job, "K", "before-text", half)
        if st is not None and kt is not None and "K" in cols:
            g.setdefault("shell_parity", {})[half] = MC.k_shell_parity(kt["acc"], _rec(job, "K", "before", half)["acc"],
                                                                        st["acc"], sb["acc"])
            # confound 31: the same parity on the shell's NATIVE letter surface. If the primary surface
            # fails parity and the native one passes, the shell's letter loss is a prompt-format artifact
            # of the processor's trailing newline, not a property of the model.
            sn = _rec(job, "S-native", "before", half)
            if sn is not None:
                g["shell_parity"][half]["S_shell_native"] = MC.k_shell_parity(
                    kt["acc"], _rec(job, "K", "before", half)["acc"], st["acc"], sn["acc"])["S_shell"]
                g["shell_parity"][half]["S_text_vs_native_delta"] = float(st["acc"] - sn["acc"])
    if "U" in cols:
        u, us = _rec(job, "U", "before", "all"), _rec(job, "U-shortcut", "before", "all")
        g["u_valid"] = MC.u_valid(u["acc"], us["acc"]) if u.get("status") != "VOID-tokenizer" else {"U": "VOID-tokenizer"}
    if "ppl" in cols:
        rb, ra = _rec(job, "ppl", "before", "all"), _rec(job, "ppl", "after", "all")
        pb, pa = rb["ppl"], ra["ppl"]
        assert rb["forward"] == ra["forward"], (
            f"ppl arms read through different forwards: {rb['forward']!r} vs {ra['forward']!r} — "
            "a fluency ratio across two protocols is not a measurement")
        g["ppl"] = {"before": pb, "after": pa, "ratio": pa / pb, "fluency_broken": bool(pa / pb > 20),
                    "forward": rb["forward"], "protocol": rb["protocol"]}
        # The §1.2 threshold is UNCHANGED and one-sided (`after/before > 20`), so any defect that
        # INFLATES the before arm deflates the ratio and SUPPRESSES the flag; a text-only forward of
        # row 2's VLM shell did exactly that (before 18558.6, ratio 0.0032, no flag). `ratio_inverted`
        # stays a DIAGNOSTIC (it cannot separate a broken instrument from a real fluency improvement).
        g["ppl"]["ratio_inverted"] = bool(pa / pb < 1 / 20)
        # GATE (2026-09-03, §6.5, §13 lesson 39): on an install row the `before-text` member banks the
        # BARE-text anchor, so the shell arm can be checked against the SAME weights read bare. That is
        # a parity check, not an invented perplexity threshold — the same shape as `k_shell_parity` —
        # and `before_forward_suspect` is therefore promoted from diagnostic to verdict here.
        rt = _rec(job, "ppl", "before-text", "all")
        if rt is not None:
            fg = MC.ppl_forward_gate(pb, rt["ppl"])
            g["ppl"]["before_text"] = rt["ppl"]
            g["ppl"]["before_shell_vs_text"] = fg["shell_vs_bare"]
            g["ppl"]["before_forward_suspect"] = bool(fg["ppl_forward"] != "VALID")
            g["ppl"]["ppl_forward"] = fg["ppl_forward"]
            g["ppl"]["ppl_forward_factor"] = fg["factor"]
    job.stage("gates")


def samples_stage(job):
    """Collect the real-inference samples every readout banked (actual prompt tail, the model's actual
    prediction and top token, the candidates it scored, the decoded generation where there is one) into
    ONE `out/samples.json` per cell, so the sample report can show real input and real output for every
    eval without re-running anything. Requested 2026-09-03: every eval in the report carries real data."""
    samples = {}
    for col, per_model in job.m["cols"].items():
        for model, per_half in per_model.items():
            for half, rec in per_half.items():
                if not isinstance(rec, dict):
                    continue
                # O is banked ONE LEVEL DEEPER (`cols.O.<member>.all = {O column: record}`), so this loop
                # read a dict-of-records as a record, found no `extra`, and collected nothing — the third
                # reader in this repo to make that assumption (2026-09-03 audit; see `matrix_recovery.
                # _beh_o` and `matrix_verdicts.o_verdicts`). Flattened here as `O:<column>`.
                for sub_col, sub in ({f"{col}:{k}": v for k, v in rec.items() if isinstance(v, dict)}.items()
                                     if col == "O" and "per_item" not in rec else ((col, rec),)):
                    ex = sub.get("extra") or {}
                    sm = ex.get("samples")
                    if sm:
                        # acc/n/floor fall back to `extra`: P is keyed by layer and ppl reports a
                        # perplexity, so neither carries a top-level accuracy of its own.
                        def _f(k, _sub=sub, _ex=ex):
                            v = _sub.get(k)
                            return _ex.get(k) if v is None else v
                        samples.setdefault(sub_col, {}).setdefault(model, {})[half] = {
                            "acc": _f("acc"), "n": _f("n"), "floor": _f("floor"), "samples": sm}
    job.m["samples_index"] = {c: sorted(v) for c, v in samples.items()}
    _write_json(os.path.join(job.out, "samples.json"),
                {"row": job.m["row"], "smoke": job.m.get("smoke"), "cols": samples})
    job.stage("samples", n_cols=len(samples))


def lost_stage(job, cols, halves):
    import numpy as np
    from infra.pg import matrix_core as MC
    lost = {}
    for col, per_model in job.m["cols"].items():
        if col in ("H", "P", "ppl", "O"):
            continue
        for half, rb in per_model.get("before", {}).items():
            ra = per_model.get("after", {}).get(half)
            if ra is None or rb.get("status") == "VOID-tokenizer" or ra.get("status") == "VOID-tokenizer":
                lost.setdefault(col, {})[half] = {"status": "VOID-tokenizer"} if ra is not None else {"status": "missing after"}
                continue
            assert rb["per_item_hash"] == ra["per_item_hash"], f"{col}/{half}: item pairing broke"
            ml = MC.margin_lost(rb["acc"], ra["acc"], rb["floor"])
            boot = MC.bootstrap_fraction(rb["per_item"], ra["per_item"], ra["per_item"], n_boot=job.m["n_boot"])
            lost.setdefault(col, {})[half] = {**ml, "bootstrap": {k: boot[k] for k in ("n_items", "lost_point", "lost_ci")}}
    if "O" in job.m["cols"]:
        ob, oa = _rec(job, "O", "before", "all"), _rec(job, "O", "after", "all")
        if isinstance(ob, dict) and "status" not in ob:
            for k in ob:
                # a VOID record still carries `per_item`, so the acceptance gate has to be asked FIRST or
                # an unusable O column would be scored anyway (that is exactly the defect being closed)
                if k in oa and str(ob[k].get("status") or oa.get(k, {}).get("status") or "").startswith("VOID"):
                    lost.setdefault("O", {})[k] = {"status": ob[k].get("status") or oa[k]["status"]}
                    continue
                if k in oa and "per_item" in ob[k]:
                    ml = MC.margin_lost(ob[k]["acc"], oa[k]["acc"], ob[k]["floor"])
                    boot = MC.bootstrap_fraction(ob[k]["per_item"], oa[k]["per_item"], oa[k]["per_item"], n_boot=job.m["n_boot"])
                    lost.setdefault("O", {})[k] = {**ml, "bootstrap": {kk: boot[kk] for kk in ("n_items", "lost_point", "lost_ci")}}
    if "S" in lost and "S-0" in lost:
        for half in halves:
            a, b = lost["S"].get(half, {}), lost["S-0"].get(half, {})
            if a.get("status") in ("ok", "below-floor") and b.get("status") in ("ok", "below-floor"):
                lost.setdefault("sk_repairable", {})[half] = MC.sk_repairable(a["lost"], b["lost"])
    def _scrub(r):
        """NaN → None for JSON. A per-half record is a dict; `sk_repairable` is a bare bool (smoke 4)."""
        if not isinstance(r, dict):
            return r
        return {k: (None if isinstance(v, float) and np.isnan(v) else _scrub(v)) for k, v in r.items()}
    job.m["controls"]["lost"] = {c: {h: _scrub(r) for h, r in hs.items()} for c, hs in lost.items()}
    job.stage("lost")


def _text_pair_lost(job, before_key, after_key, n_boot):
    """`margin_lost` + the paired bootstrap for one pair of TEXT members, column by column.

    The SAME two `matrix_core` functions `lost_stage` calls, with the same arguments in the same order —
    no threshold is invented here and none is moved. Only the two member keys differ."""
    from infra.pg import matrix_core as MC
    out = {}
    for col, per in job.m["cols"].items():
        rb_all, ra_all = per.get(before_key), per.get(after_key)
        if not rb_all or not ra_all:
            continue
        for half in sorted(set(rb_all) & set(ra_all)):
            rb, ra = rb_all[half], ra_all[half]
            if not isinstance(rb, dict) or not isinstance(ra, dict):
                continue
            if rb.get("status") == "VOID-tokenizer" or ra.get("status") == "VOID-tokenizer":
                out.setdefault(col, {})[half] = {"status": "VOID-tokenizer"}
                continue
            if not (rb.get("per_item") and ra.get("per_item")):
                continue                      # ppl carries no per-item vector; it is reported by `validity`
            assert rb["per_item_hash"] == ra["per_item_hash"], f"{col}/{half}: {before_key} vs {after_key} pairing broke"
            ml = MC.margin_lost(rb["acc"], ra["acc"], rb["floor"])
            boot = MC.bootstrap_fraction(rb["per_item"], ra["per_item"], ra["per_item"], n_boot=n_boot)
            out.setdefault(col, {})[half] = {**ml, "bootstrap": {k: boot[k] for k in ("n_items", "lost_point", "lost_ci")}}
    return out


def text_shell_stage(job, halves):
    """`--after-text` only: the install contrast with NO shell on either arm, and its validity control.

    Two pairs, both scored by rules that already exist:
      `contrast`  before-shell-text → after-text. The two arms are the SAME architecture, the SAME
                  tokenizer and the SAME call; only the decoder weights differ. This is the reading
                  `k_shell_parity` cannot void, because there is no shell on either side of it.
      `validity`  before-text → before-shell-text. ONE set of base-decoder weights read two ways: in its
                  own architecture, and inside the shell text-only. `matrix_core.k_shell_parity` (§6.5,
                  |Δ| ≤ 0.05 on S and K) and `matrix_core.ppl_forward_gate` (§6.5, factor 20) are the
                  frozen rules for exactly that question, applied here to the new read path. If they
                  stamp VOID the `contrast` above is NOT a damage measurement — the wrapper read
                  text-only cannot carry a decoder faithfully — and that is the whole point of banking
                  it beside the number instead of underneath it.
      `encoders`  the two text arms' probe encodings. Different ids for the same string (a missing BOS,
                  a diverged vocabulary) makes the contrast a tokenizer comparison, whatever the gates say.

    Nothing here feeds a §6 verdict: `matrix_verdicts` reads `before`/`after` and is untouched by this
    block. It is banked under `controls`, where the lead reads it."""
    from infra.pg import matrix_core as MC
    tm = job.m["gates"].get("text_members", {})
    out = {"pairs": {"contrast": ["before-shell-text", "after-text"],
                     "validity": ["before-text", "before-shell-text"]},
           "contrast": _text_pair_lost(job, "before-shell-text", "after-text", job.m["n_boot"]),
           "validity": _text_pair_lost(job, "before-text", "before-shell-text", job.m["n_boot"]),
           "gates": {}, "encoders": {k: v for k, v in tm.items() if isinstance(v, dict)}}
    for half in halves:
        st, kt = _rec(job, "S", "before-text", half), _rec(job, "K", "before-text", half)
        ss, ks = _rec(job, "S", "before-shell-text", half), _rec(job, "K", "before-shell-text", half)
        if None not in (st, kt, ss, ks) and not any(r.get("status") == "VOID-tokenizer" for r in (st, kt, ss, ks)):
            out["gates"].setdefault("text_parity", {})[half] = MC.k_shell_parity(kt["acc"], ks["acc"],
                                                                                 st["acc"], ss["acc"])
    pt, ps = _rec(job, "ppl", "before-text", "all"), _rec(job, "ppl", "before-shell-text", "all")
    if pt is not None and ps is not None:
        out["gates"]["ppl_text_forward"] = {**MC.ppl_forward_gate(ps["ppl"], pt["ppl"]),
                                            "before_text": pt["ppl"], "before_shell_text": ps["ppl"]}
    # THE PAIRS THIS CONTROL ACTUALLY SCORES, every one of them (2026-09-04 repair). This gate used to
    # compare `before-text` vs `after-text` — a pair that appears in NEITHER of the two comparisons above.
    # `contrast` is before-shell-text vs after-text and `validity` is before-text vs before-shell-text, so
    # the encoder check and the numbers it was meant to license were about different objects. It fired on
    # 2026-09-03 only by luck of which pair it happened to compare: both CONTRAST arms losing their start
    # token identically would have read VALID, because the arm the comparison did include was the one that
    # kept it. The rule is now derived from `out["pairs"]` — the same dict the scoring reads — so an arm
    # added to a pair cannot escape the encoder check, and a pair cannot be checked that is not scored.
    ids = {k: v["probe_ids"] for k, v in out["encoders"].items()}
    per_pair, missing = {}, []
    for name, (lhs, rhs) in out["pairs"].items():
        if {lhs, rhs} <= set(ids):
            per_pair[name] = {"arms": [lhs, rhs], "identical_probe_ids": bool(ids[lhs] == ids[rhs])}
        else:
            missing.append(f"{name} ({', '.join(sorted({lhs, rhs} - set(ids)))} not encoded)")
    if per_pair:
        bad = sorted(n for n, v in per_pair.items() if not v["identical_probe_ids"])
        out["gates"]["text_encoder_parity"] = {
            "pairs": per_pair, "pairs_not_checked": missing,
            "identical_probe_ids": all(v["identical_probe_ids"] for v in per_pair.values()),
            "status": "VALID" if not bad else f"VOID-text-encoder (pairs: {', '.join(bad)})",
            "bos_present": {k: v["bos_present"] for k, v in out["encoders"].items()},
            "note": "the same probe string through the OWN encoder of each arm of each SCORED pair; "
                    "different ids mean that pair's two arms were not asked the same question (a "
                    "missing BOS is the ERLANGEN c6b defect). Checking one pair the scoring does not "
                    "use is how two arms could lose the start token together and still read VALID."}
    job.m["controls"]["text_shell"] = out
    job.stage("text_shell", pairs=sorted(out["contrast"]))
    return out


def preflight_c1(job, path, halves, items):
    """§12.10 / D19: reproduce C1 per item on row 2, discovery half. S: carrel_metrics.json configs.all (healthy,
    134/285) vs configs.none (damaged, 80/285); K: carrel_tail_metrics.json cloze.per_item_before/after
    (126 / 105 of 285). Both banked in text_eval order (sorted subject, rows[:5]) without hashes → positional."""
    from infra.pg import matrix_core as MC
    assert job.m["row"] == "2", "the C1 preflight is a row-2 instrument"
    assert "discovery" in halves
    ref = json.load(open(path))
    tail_path = os.path.join(os.path.dirname(path), "carrel_tail_metrics.json")
    assert os.path.exists(tail_path), f"K reference missing: {tail_path}"
    tail = json.load(open(tail_path))
    pos = items["mmlu_pos"]["discovery"]
    refs = {"S": {"before": ref["configs"]["all"]["per_item"], "after": ref["configs"]["none"]["per_item"]},
            "K": {"before": tail["cloze"]["per_item_before"], "after": tail["cloze"]["per_item_after"]}}
    out = {"reference": {"S": path, "K": tail_path}, "n_ref": len(refs["S"]["before"]), "n_compared": len(pos)}
    for col in ("S", "K"):
        for model in ("before", "after"):
            r = _rec(job, col, model, "discovery")
            assert r is not None and len(r["per_item"]) == len(pos), (col, model)
            assert len(refs[col][model]) == 285, (col, model, len(refs[col][model]))
            sub = [refs[col][model][p] for p in pos]
            out[f"{col}_{model}"] = MC.preflight_match(r["per_item"], sub)
    out["pass"] = all(out[k]["pass"] for k in ("S_before", "S_after", "K_before", "K_after"))
    job.m["controls"]["preflight_c1"] = out
    job.stage("preflight_c1", passed=out["pass"])
    return out


def completeness(job, cols, halves):
    missing = []
    for col in cols:
        names = {"U": ["U", "U-shortcut"]}.get(col, [col])
        for name in names:
            for model in ("before", "after"):
                hs = halves if name in HALF_COLS else (["eval"] if name in EVAL_COLS else ["all"])
                for h in hs:
                    if _rec(job, name, model, h) is None:
                        missing.append((name, model, h))
    assert not missing, f"columns missing: {missing}"
    return True


# ------------------------------------------------------------------------------------------ main
def parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", default=None)
    ap.add_argument("--cols", default=",".join(DEFAULT_COLS))
    ap.add_argument("--halves", default="discovery,confirmation")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--load-test-only", action="store_true")
    ap.add_argument("--preflight-c1", default=None, help="path to carrel_metrics.json (row 2)")
    ap.add_argument("--secondary-head", action="store_true", help="1b rows: also run the PG1-table head (after-pg1)")
    ap.add_argument("--after-text", action="store_true",
                    help="install rows: also re-read each RESIDENT shell member through a plain text "
                         "forward (no image) as `before-shell-text` / `after-text` — the install "
                         "contrast with no shell on either arm, plus its §6.5 validity control. "
                         "Loads no extra model; OFF by default, so no queued cell changes shape.")
    ap.add_argument("--adapter-uri", default=None, help="row 5c: S3 URI or presigned URL of adapter.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dryrun", action="store_true")
    a = ap.parse_args(argv)
    assert a.dryrun or (a.row and a.out), "--row and --out are required (except --dryrun)"
    cols = sorted({c.strip() for c in a.cols.split(",") if c.strip()}, key=ALL_COLS.index)
    unknown = [c for c in cols if c not in ALL_COLS]
    assert not unknown, f"unknown columns {unknown}; known {ALL_COLS}"
    a.cols = [x for c in cols for x in (("S-0", "S-1") if c == "S-k" else (c,))]     # S-k = 0-shot + 1-shot letter
    a.halves = [h.strip() for h in a.halves.split(",") if h.strip()]
    assert a.halves and all(h in ("discovery", "confirmation") for h in a.halves), a.halves
    if a.smoke:
        a.n_boot = min(a.n_boot, 10)
    return a


def run(a):
    import subprocess
    from infra.pg import matrix_models as MM
    from infra.pg import util_probe as UP
    from infra.pg.matrix_items import _sha256
    row = MM.ROWS[a.row]
    os.makedirs(a.out, exist_ok=True)
    job = Job(a.out, a.row, a.cols, a.halves, a.smoke, a.n_boot)
    if a.load_test_only:
        os.environ["MDL_LOAD_TEST"] = "1"
    if _has_nvidia_smi():
        job.sampler = UP.DmonSampler(job.events, interval_s=30).start()
    # ---- manifest ----
    env = _env_pins()
    repos = {r: MM.repo_hashes(r, MM.hf_files(r)) for r in MM.row_repos(a.row)}
    shift = json.loads(os.environ["MDL_HARNESS_SHIFT"]) if os.environ.get("MDL_HARNESS_SHIFT") else None
    job.m["manifest"] = {"row": {k: v for k, v in row.items() if k != "floors"}, "repos": repos,
                         "known": {r: MM.KNOWN[r] for r in MM.row_repos(a.row)}, "env": env, "args": vars(a),
                         "harness_shift": shift,
                         "seeds": {"items": 1337, "bootstrap": 1337, "probe_shuffle": 1337, "random_init": 0}}
    if a.load_test_only:
        job.stage("manifest")
        return load_test(job, a, MM)
    items = build_items(a.halves, a.smoke, a.cols, a.row)
    job.m["manifest"]["items"] = items_manifest(items)
    _write_json(os.path.join(a.out, "manifest.json"), job.m["manifest"])
    job.stage("manifest")
    # ---- members, sequential ----
    adapter_path = None
    if row["adapter"] and a.adapter_uri:
        adapter_path = os.path.join(a.out, "..", "h2_adapter.pt")
        MM.fetch_uri(a.adapter_uri, adapter_path)
    members = MM.members(a.row) + (["after-pg1"] if a.secondary_head and row["policy"] else [])
    # --after-text: which RESIDENT members get an extra text-shell reading, asked of the REGISTRY.
    # `{}` on a row that has none, and the reason is banked rather than crashed or silently skipped.
    text_keys = MM.text_members(a.row) if a.after_text else {}
    if a.after_text and not text_keys:
        job.m["gates"]["text_members"] = {
            "status": f"not run — row {a.row} has no text-shell re-reading",
            "reason": f"matrix_models.text_members({a.row!r}) == {{}}: the row has no `before-text` parity "
                      f"member (members({a.row!r}) == {MM.members(a.row)}), so its `after` is either "
                      f"already read through the text shell (shell == {row['shell']!r}) or has no text "
                      f"reading at all (policy row)"}
    probe_state, snapshot = {}, None
    for which in members:
        job.stage(f"load_{which}")
        member = MM.load_pair(a.row, a.device, which, adapter_path)
        if which == "before-text":
            _run_parity_member(job, member, a.halves, items, a.cols, a.smoke)
            if text_keys:       # the same three encoder lines the re-read banks, for the arm it is compared to
                job.m["gates"].setdefault("text_members", {})["before-text"] = _text_shell_check(
                    member["fwd"], member, "before-text", "text (the row's bare base LM)")
        else:
            run_member(job, member, which, a.cols, a.halves, items, a.smoke, probe_state)
            # BEFORE `_random_init_last`, which re-initialises the resident healthy decoder in place.
            if which in text_keys:
                _run_text_shell_member(job, member, text_keys[which], a.halves, items, a.cols, a.smoke)
            if which == "before":
                snapshot = MM.snapshot_lm(member, os.path.join(a.out, "..", "lm_snapshot"))
                job.stage("snapshot_before", n_layers=snapshot["n_layers"])
                _random_init_last(job, member, a.cols, a.halves, items, probe_state)
            if which == "after":
                job.m["binds"] = MM.binds(snapshot, member)
                job.stage("binds", frozen_lm=job.m["binds"]["frozen_lm"])
        MM.free(member)
        job.stage(f"freed_{which}")
    gates_stage(job, a.cols, a.halves)
    lost_stage(job, a.cols, a.halves)
    if text_keys:
        text_shell_stage(job, a.halves)
    samples_stage(job)
    if a.preflight_c1:
        preflight_c1(job, a.preflight_c1, a.halves, items)
    _write_json(os.path.join(a.out, "controls.json"), {"gates": job.m["gates"], **job.m["controls"]})
    completeness(job, a.cols, a.halves)
    job.m["elapsed_s"] = time.time() - job.m["t0"]
    job.stage("done", elapsed_s=job.m["elapsed_s"])
    if job.sampler:
        job.sampler.stop()
    return job


def _run_parity_member(job, member, halves, items, cols, smoke, model_key="before-text", fwd=None, shell="text"):
    """The healthy LM text-only: S (5-shot) and K on every half — the §6.5 parity member.

    ADDITIVE 2026-09-03: it also banks `ppl`, the BARE-text fluency anchor for the row. The §1.2
    ratio rule is one-sided, so nothing in the grid could catch a defect that inflates the BEFORE
    arm — and a broken before-forward is exactly what happened on row 2 (18558.6 through the shell
    against 7.966 bare, ratio 0.0032, no flag). With this anchor the gate can say out loud when the
    shell arm and the bare arm disagree by orders of magnitude. Costs one extra ppl pass on a model
    that is already resident.

    ADDITIVE 2026-09-04: `model_key` / `fwd` / `shell` default to EXACTLY the call this function has
    always made (`before-text`, the member's own Forward, shell "text"), so a run without
    `--after-text` executes the identical code with the identical arguments. `--after-text` reuses the
    body with a text Forward built over an already-resident SHELL member (`_run_text_shell_member`):
    same items, same halves, same columns, one banking key apart."""
    from infra.pg import format_readouts as FR
    from infra.pg import util_probe as UP
    fwd = member["fwd"] if fwd is None else fwd
    job.m["members"].append({"which": model_key, "repo": member["repo"], "shell": shell, "info": member["info"]})
    h0 = halves[0]
    prompts = [FR.s_prompt(it, FR._shots(FR.MI.mmlu_dev_shots, it["subject"], 5)) for it in items["mmlu"][h0]]
    tp = UP.choose_batch(fwd, prompts, start=4, max_bs=64)
    job.stage(f"throughput_{model_key}", batch=tp["batch"], prompts_per_s=tp["prompts_per_s"], stop=tp["stop"])
    for half in halves:
        # only the columns this job was asked for: a single-column re-run (`--cols ppl`, the §6.5
        # forward fix) must not re-buy the letter readouts, nor bank a one-armed S/K that reads
        # downstream as an INCOMPLETE column. A full behaviour job always asks for S and K.
        if "S" in cols:
            ok, counts = _letter_ok(fwd, "S")
            job.bank("S", model_key, half, FR.S(fwd, items["mmlu"][half], 5) if ok else _void_tokenizer("S", counts))
        if "K" in cols:
            job.bank("K", model_key, half, FR.K(fwd, items["mmlu"][half], 5))
    if "ppl" in cols:
        job.bank("ppl", model_key, "all", _ppl(fwd, 5_000 if smoke else PPL_MAX_TOKENS))
        # The text-shell re-read is where this question is sharpest: it is the path that banked row 2's
        # 18558.6, so its forward is a suspect instrument until it says otherwise about itself.
        job.m["gates"].setdefault("causality", {})[model_key] = _causality_probe(fwd)


# ------------------------------------------------------------------ the opt-in text-shell re-reading
TEXT_PROBE = "The capital of France is"


def _text_shell_check(fwd, member, model_key, shell):
    """What the text forward of this member ACTUALLY encodes, banked before any readout is bought.

    One probe string through the member's own encoder. It costs nothing and it is the only thing that
    can catch the failure this member is most exposed to: a shell whose processor owns the framing may
    hand back a tokenizer that does NOT prepend BOS under `add_special_tokens=True`, and a Gemma read
    without a start token is the ERLANGEN c6b VOID instrument (61.28 vs 7.966 on the same weights).
    `before-text` records the same three lines, so the two text arms can be compared as ENCODINGS and
    not only as accuracies."""
    enc = fwd._encode([TEXT_PROBE])                                    # noqa: SLF001 — the shell's own call
    ids = [int(i) for i in enc["input_ids"][0]]
    bos = getattr(fwd.tok, "bos_token_id", None)
    return {"member": model_key, "repo": member["repo"], "read_through": shell,
            "resident_member": member["which"], "resident_shell": member["shell"],
            "probe": TEXT_PROBE, "probe_ids": ids, "n_probe_ids": len(ids),
            "first_token_id": ids[0], "bos_token_id": bos, "bos_present": bos is not None and ids[0] == bos,
            "letter_token_counts": [int(c) for c in fwd.letter_token_counts(CHOICES, True)]}


def _run_text_shell_member(job, member, model_key, halves, items, cols, smoke):
    """`--after-text`: the SAME resident model, read through a plain text forward with no image.

    Nothing is loaded and nothing is copied — `matrix_models.text_forward_for` wraps the model that is
    already on the GPU, so the sequential rule (never two models resident) holds by construction and the
    cost is the S/K/ppl passes alone. Runs AFTER the member's own columns are banked, and on `before`
    strictly BEFORE `_random_init_last`, which destroys the resident decoder."""
    from infra.pg import matrix_models as MM
    fwd = MM.text_forward_for(member)
    shell = f"text, no image (the resident {member['shell']} `{member['which']}` member re-read)"
    job.m["gates"].setdefault("text_members", {})[model_key] = _text_shell_check(fwd, member, model_key, shell)
    _run_parity_member(job, member, halves, items, cols, smoke, model_key=model_key, fwd=fwd, shell=shell)


def load_test(job, a, MM):
    """Row 4 (B2 S9-pre): load each model, one forward, binds; exit 0."""
    snapshot = None
    for which in ("before", "after"):
        job.stage(f"load_{which}")
        t0 = time.time()
        member = MM.load_pair(a.row, a.device, which)
        import numpy as np
        lg = np.asarray(member["fwd"].next_token_logits(["The capital of France is", "Answer:"]))
        job.m["controls"].setdefault("load_test", {})[which] = {
            "load_s": time.time() - t0, "logits_shape": list(lg.shape), "finite": bool(np.isfinite(lg).all()),
            "n_layers": member["fwd"].n_layers, "family": member["fwd"].family, "repo": member["repo"],
            "letter_token_counts": member["fwd"].letter_token_counts(CHOICES, True)}
        job.stage(f"forward_{which}")
        if which == "before":
            snapshot = MM.snapshot_lm(member, os.path.join(a.out, "..", "lm_snapshot"))
        else:
            job.m["binds"] = MM.binds(snapshot, member)
            job.stage("binds", frozen_lm=job.m["binds"]["frozen_lm"])
        MM.free(member)
    job.stage("done", load_test_only=True)
    if job.sampler:
        job.sampler.stop()
    return job


def _has_nvidia_smi():
    import shutil
    return shutil.which("nvidia-smi") is not None


def _env_pins():
    import platform
    import subprocess
    pins = {"python": platform.python_version(), "hostname": platform.node(),
            "instance_type": os.environ.get("PG_INSTANCE_TYPE") or os.environ.get("INSTANCE_TYPE"),
            "vast": bool(os.environ.get("VAST_CONTAINERLABEL") or os.environ.get("CONTAINER_ID")),
            "hf_hub_disable_xet": os.environ.get("HF_HUB_DISABLE_XET")}
    for mod in ("torch", "transformers", "numpy", "datasets"):
        if mod in sys.modules or _importable(mod):
            m = __import__(mod)
            pins[mod] = getattr(m, "__version__", None)
    if "torch" in pins and pins["torch"]:
        import torch
        pins["cuda"] = torch.version.cuda
        pins["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=os.path.dirname(os.path.abspath(__file__)))
    pins["git_sha"] = git.stdout.strip() if git.returncode == 0 else os.environ.get("PG_GIT_SHA")
    return pins


def _importable(mod):
    import importlib.util
    return importlib.util.find_spec(mod) is not None


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    a = parse(argv)
    if a.dryrun:
        return _dryrun(a)
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()                                   # BEFORE any transformers/datasets import
    from infra.pg.matrix_models import ensure_transformers_pin
    if ensure_transformers_pin(a.row, MODULE, argv) is not None:     # §8b: the child under the pin venv did the work
        return
    run(a)
    print(f"[N7 {a.row}] DONE — every requested column present", flush=True)


# ------------------------------------------------------------------------------------------ dry-run
class _StubForward:
    """format_readouts.FakeForward + the N3 harvest + a stub model/tokenizer, keyed by `which` so before ≠ after."""
    def __init__(self, which, n_layers=6, d=16, bos=2, start=None):
        import types
        from infra.pg.format_readouts import FakeForward
        self._f = FakeForward()
        self.which, self.n_layers, self.family, self.shell, self.batch_size, self.device = which, n_layers, "stub", "text", 3, "cpu"
        self.d = d
        # `.model` back-references the Forward so the dry-run's stub decoder re-init can actually change
        # what this member reports (probe_readout.random_init_bound now asserts the features moved).
        # `.tok` carries a BOS id because `_text_shell_check` reads one: a text arm that does not start
        # its sequence with the model's start token is the ERLANGEN c6b void. `start` is the id the
        # ENCODER actually emits first and defaults to the BOS — a processor-owned tokenizer that drops
        # the start token under `add_special_tokens=True` is the realistic failure, and `start != bos`
        # is how the stub expresses it, so the VOID branch of the gate is reachable.
        self.model, self.tok = types.SimpleNamespace(stub_forward=self), types.SimpleNamespace(bos_token_id=bos)
        self._start = bos if start is None else start

    def _encode(self, texts, images=None):
        """The shell's own encoding, stubbed: start token + one deterministic id per word, left-padded
        to the longest row. Deterministic in the TEXT, so two text arms encoding the same probe agree
        exactly unless their start token differs — the comparison `text_encoder_parity` makes."""
        import torch
        assert images is None or all(im is None for im in images), "text shell takes no images"
        rows = [[self._start] + [sum(map(ord, w)) % 997 for w in t.split()] for t in texts]
        n = max(len(r) for r in rows)
        return {"input_ids": torch.tensor([[0] * (n - len(r)) + r for r in rows])}

    def _salt(self, prompts):
        return [f"{self.which}|{p}" for p in prompts]

    def next_token_logits(self, prompts, images=None):
        return self._f.next_token_logits(self._salt(prompts))

    def continuation_logprobs(self, prompts, continuations, images=None, sep=" "):
        return self._f.continuation_logprobs(self._salt(prompts), continuations)

    def generate(self, prompts, max_new_tokens, images=None, skip_special_tokens=True):
        return self._f.generate(self._salt(prompts), max_new_tokens)

    def letter_ids(self, letters=CHOICES, space=True):
        return self._f.letter_ids(letters, space)

    def letter_token_counts(self, letters=CHOICES, space=True):
        return [2 if (self.which == "after-void" and L == "A" and space) else 1 for L in letters]

    def final_residuals(self, prompts, layers):
        import numpy as np
        import torch
        out = {}
        for L in layers:
            rows = [self._f._rng(f"R{L}|{self.which}|{p}").normal(size=self.d) for p in prompts]
            out[L] = torch.tensor(np.stack(rows), dtype=torch.float16)
        return out


def _stub_member(which, row, salt=None) -> dict:
    """One `matrix_models.load_pair` member over a `_StubForward`, CHECKED against the real loader's
    contract (`matrix_models.assert_member`). A dry-run stub is a contract, not a convenience: on
    2026-09-03 matrix_graft's stub loader returned a different shape from the real one, its `--dryrun`
    passed, and three GPU packs died in `attach`. `salt` keys the stub's outputs (before ≠ after ≠ the
    VOID-tokenizer variant) without changing the member's own `which`."""
    from infra.pg import matrix_models as MM
    fwd = _StubForward(salt or which)
    return MM.assert_member({"fwd": fwd, "model": fwd.model, "tok": fwd.tok, "proc": None,
                             "repo": MM.ROWS[row][which.split("-")[0]], "which": which, "shell": "text",
                             "info": {"stub": True}, "local": None}, which)


class _CandStubForward:
    """Duck-typed Forward for the PaliGemma cand seam: `_cand_record` needs `batch_size` and
    `continuation_logprobs(prompts, continuations, images=...)` and nothing else. `gold_of` maps the
    prompt string to that item's gold answer; the stub prefers the gold."""
    batch_size = 4

    def __init__(self, gold_of: dict):
        self.gold_of = gold_of
        self.seen_images = 0

    def continuation_logprobs(self, prompts, continuations, images=None):
        assert images is not None and len(images) == len(prompts), "the cand readout must pass REAL images"
        self.seen_images += len(images)
        out = []
        for p, c in zip(prompts, continuations):
            n = max(1, len(c.strip().split()))
            out.append(((-1.0 if _vqa_norm(c) == _vqa_norm(self.gold_of[p]) else -6.0) * n, n))
        return out


def _o_seams():
    """Run the two REAL O producers weight-free, so neither can rot into a name nothing defines again.

    This is the check that was missing when `_o_column`'s Molmo branch called four functions
    `vlm_skill_eval` does not define and one with the wrong signature: the dry-run replaced the whole
    of `_o_column`, so no test ever touched the seam. Here the only thing stubbed is the FORWARD (the
    one part that genuinely needs weights); the prompts, the candidate draw, the record shape, the
    dispatcher and the key sets are the real ones."""
    from infra.pg import vlm_skill_eval as VS
    # --- Molmo2: the real `VS.evaluate` dispatcher over the real prompt set --------------------------
    P = VS.molmo2_prompts()
    rows = {t: VS._stub_rows(t, 8) for t in ("refcoco", "vqav2", "textvqa")}
    gens, golds = {}, {}
    for r in rows["refcoco"]:
        W, H = r["image"].size
        x, y, w, h = r["bbox"]
        gens[P["refcoco"](r)] = f'<points alt="o" coords="1 1 {int(1000 * (x + w / 2) / W):03d} {int(1000 * (y + h / 2) / H):03d}"/>'
    for t in ("vqav2", "textvqa"):
        for r in rows[t]:
            g = VS.gold_answer(t, r)
            gens[P[t](r)] = g
            golds[P[t](r)] = g
    fwd = VS._StubForward(gens, golds)
    out = _o_column_molmo2(fwd, "allenai/Molmo2-4B", rows)
    assert set(out) == set(O_KEYS["molmo2"]), (sorted(out), sorted(O_KEYS["molmo2"]))
    assert O_VERDICT_KEY in out and out[O_VERDICT_KEY]["floor"] == 0.25 and out[O_VERDICT_KEY]["acc"] == 1.0
    assert out["vqav2"]["acc"] == 1.0 and out["refcoco"]["extra"]["metric"] == "point_in_gt_box"
    assert out["vqav2"]["extra"]["acceptance"]["published"] == 86.6
    assert out["refcoco"]["extra"]["acceptance"]["pass_"] is None          # no published RefCOCO anchor
    # THE ACCEPTANCE GATE FIRES (2026-09-03 ruling). This stub answers every item correctly, so vqav2
    # reads 100.0 against a published 86.6 — delta +13.4, outside the ±6-point band. That must not be
    # scored silently: every O record of the member is stamped VOID-acceptance and the FAIL verdict is
    # legible in the record. If the band or the propagation is ever loosened back to a note, this fails.
    assert out["vqav2"]["extra"]["acceptance"]["pass_"] is False
    assert out["vqav2"]["extra"]["acceptance"]["verdict"].startswith("FAIL — vqav2 100.00 vs published 86.60")
    assert all(out[c]["status"] == "VOID-acceptance" for c in O_KEYS["molmo2"]), \
        "acceptance FAILED and the O column was left scorable — the gate did not fire"
    assert out[O_VERDICT_KEY]["extra"]["acceptance_failed"], "the failure must be named on the verdict column"
    # and the negative branch: no anchor table -> pass_ None everywhere -> NOTHING is voided
    unp = _o_column_molmo2(fwd, "stub/unpublished", rows)
    assert unp["vqav2"]["extra"]["acceptance"]["pass_"] is None
    assert all("status" not in unp[c] for c in O_KEYS["molmo2"]), "no anchor must not void the column"
    # --- PaliGemma: the real `cand_column` (prompts, images, distractor draw, record shape) ----------
    from infra.pg import battery_tasks as _B
    vrows = rows["vqav2"]
    cf = _CandStubForward({VQA_PREFIX + r["question"]: VS.gold_answer("vqav2", r) for r in vrows})
    rec = cand_column(cf, "vqav2", vrows, [f"h{i}" for i in range(len(vrows))])
    assert rec["col"] == "O-vqav2-cand" and rec["floor"] == 0.25 and rec["acc"] == 1.0
    assert cf.seen_images == 4 * len(vrows), cf.seen_images        # 4 candidates per item, REAL images
    assert set(rec) == {"col", "acc", "n", "floor", "per_item", "per_item_hash", "per_item_scores", "pred", "extra"}
    assert len(rec["extra"]["options"][0]) == 4 and rec["extra"]["prompt_example"].startswith(VQA_PREFIX)
    # §12.11 on the O column every verdict consumes (2026-09-03): `_cand_record` banked no samples, so
    # under the frozen rule no vision number in this program was quotable — and O feeds four recovery
    # hypotheses. Same for the five N14 columns, through the real dispatcher.
    sm = rec["extra"]["samples"]
    assert len(sm) == 3 and all(d["prompt_tail"].startswith(VQA_PREFIX) for d in sm), sm
    assert all(len(d["candidates"]) == 4 and len(d["candidate_scores"]) == 4 and d["gold_text"] for d in sm)
    assert all(d["gold"] is not None and d["predicted"] is not None for d in sm)
    for col in O_KEYS["molmo2"]:
        assert out[col]["extra"]["samples"], col
    trows = rows["textvqa"]
    tf = _CandStubForward({_B.TEXTVQA_TASK.prompt_fn(r): VS.gold_answer("textvqa", r) for r in trows})
    assert cand_column(tf, "textvqa", trows, [f"t{i}" for i in range(len(trows))])["acc"] == 1.0
    print("PASS O seams: vlm_skill_eval.evaluate over molmo2_prompts + cand_column, both real, "
          f"keys {sorted(O_KEYS['molmo2'])} / {sorted(O_KEYS['paligemma'])}, verdict key {O_VERDICT_KEY!r}")


def _dryrun(a):
    import tempfile
    from infra.pg import matrix_items as MI
    from infra.pg import matrix_models as MM
    from infra.pg import probe_readout as PR
    from infra.pg import util_probe as UP
    me = sys.modules[__name__]
    MI._hf = MI.fake_load_dataset                                   # every item builder → synthetic rows
    MI.vision_index_lists = lambda *a, **k: {t: {"dataset": f"stub/{t}", "split": "val", "indices": list(range(200)),
                                                 "hashes": [f"{t}{i}" for i in range(200)], "n_split": 1000,
                                                 "hash_path": "stub"} for t in ("refcoco", "vqav2", "textvqa")}
    MI.vision_rows = lambda task, load_dataset=None: [{"_index": i} for i in range(200)]
    MM.hf_files = lambda repo, revision=None: "/nonexistent"
    MM.repo_hashes = lambda repo, local: {"repo": repo, "sha": MM.KNOWN[repo]["sha"], "stub": True}
    MM.load_pair = lambda row, device, which, adapter_path=None: _stub_member(which, row)
    MM.snapshot_lm = lambda member, d: {"dir": d, "n_layers": 6, "stub": True}
    MM.binds = lambda snap, member: {"frozen_lm": False, "per_layer": [{"layer": L, "rel_delta": 0.1 * L} for L in range(6)], "stub": True}
    MM.free = lambda member: None

    def _stub_reinit(model, seed=0):
        """The stub Forward carries no real decoder, so a stub re-init re-draws its FEATURE SALT: every
        residual this member reports afterwards is a different draw, exactly as a re-initialised decoder
        would be. It used to return the integer 6 and change nothing — which is precisely the shape of
        the real defect it was hiding, so the dry-run certified a control that could not fail."""
        model.stub_forward.which += "|reinit"
        return {"seed": seed, "n_random": 6, "n_kept": 6, "rule": "stub: the Forward's feature salt is re-drawn"}
    PR.reinit_decoder = _stub_reinit
    def _stub_ppl_extra():
        """The `extra.samples` shape the real `_ppl` now returns (§12.11). The stub carries it so the
        dry-run exercises `samples_stage` on a ppl-only cell: the old assertion checked only that
        `samples.json` EXISTED, which is exactly the check that certified nine empty `{"cols": {}}`
        shells onto S3. The real selection is exercised against `text_eval.shell_text_perplexity`."""
        return {"col": "ppl", "acc": None, "n": 100, "floor": None,
                "score_units": "per-token NLL / perplexity, NOT an accuracy",
                "samples": [{"item_index": 0, "selection": "best-scoring (lowest per-token NLL)",
                             "score_kind": "per-token NLL (not right/wrong)",
                             "score_label": "per-token NLL 2.0000 (window perplexity 7.39) over 100 tokens",
                             "prompt_tail": "stub window text", "prompt_chars": 16, "n_tokens": 100,
                             "nll_per_token": 2.0, "model_top_token_id": 42, "model_top_token_p": 0.5,
                             "model_top_token_text": "x", "predicted": "x", "gold": "y",
                             "gold_token_id": 43, "scored_correct": None}]}
    me._ppl = lambda fwd, max_tokens: {"col": "ppl", "n_tokens": 100,
                                       "ppl": {"before": 20.0, "before-text": 16.0}.get(fwd.which, 25.0),
                                       "forward": "text shell", "extra": _stub_ppl_extra(),
                                       "protocol": "shell-framed windows, BOS from the shell's own encoder (ERLANGEN c6b)"}
    # The O readouts themselves need a processor, real images and a vision tower, so the pipeline test
    # below runs a placeholder — but its KEY SET is `O_KEYS["paligemma"]`, not a hand-typed list, and
    # `_o_column` asserts the real return against the same table. The two REAL producers are exercised
    # for real, weight-free, in `_o_seams()` immediately after.
    _o_seams()
    me._o_column = lambda member, vision, batch: {k: {"col": f"O-{k}", "acc": 0.5, "n": len(vision["vqav2"]["indices"]), "floor": 0.25,
                                                      "per_item": [i % 2 for i in range(len(vision["vqav2"]["indices"]))],
                                                      "per_item_hash": [f"h{i}" for i in range(len(vision["vqav2"]["indices"]))], "per_item_scores": None,
                                                      "pred": [0] * len(vision["vqav2"]["indices"]), "extra": {}}
                                                  for k in O_KEYS["paligemma"]}
    me._has_nvidia_smi = lambda: False
    UP.choose_batch = lambda fwd, prompts, start=4, max_bs=64: {"batch": 4, "prompts_per_s": 1.0, "trace": [], "stop": "stub"}
    with tempfile.TemporaryDirectory() as d:
        argv = ["--row", "2", "--cols", ",".join(ALL_COLS), "--halves", "discovery,confirmation", "--out", os.path.join(d, "out"),
                "--smoke", "--preflight-c1", os.path.join(_repo_root(), "studies/lineage/cross-lineage/results/carrel_metrics.json")]
        b = parse(argv)
        job = run(b)
        m = json.load(open(job.metrics_path))
        assert m["kind"] == "behaviour" and m["row"] == "2"
        for col in ("S", "S-0", "S-1", "F1", "F2", "F3", "F4", "F5", "K", "K-null", "H", "P"):
            for model in ("before", "after"):
                assert set(m["cols"][col][model]) == {"discovery", "confirmation"}, (col, model)
        for col in ("RFP", "U", "U-shortcut", "A", "TQ", "ppl", "O"):
            assert set(m["cols"][col]["before"]) == {"all"} and set(m["cols"][col]["after"]) == {"all"}, col
        assert set(m["cols"]["S"]["before-text"]) == {"discovery", "confirmation"} and "K" in m["cols"] and "before-text" in m["cols"]["K"]
        assert m["gates"]["tokenizer"]["discovery"]["S_family"].startswith(("VALID", "VOID-tokenizer"))
        assert {"S_shell", "K_shell", "S_shell_native", "S_text_vs_native_delta"} == set(m["gates"]["shell_parity"]["discovery"]) and "u_valid" in m["gates"]
        gp = m["gates"]["ppl"]
        assert gp["ratio"] == 1.25 and not gp["fluency_broken"] and gp["forward"] == "text shell"
        # the ppl column carries its forward, both arms read the same one, and the install row's
        # bare-text anchor is banked and compared (the one-sided-§1.2 blind spot, 2026-09-03)
        assert m["cols"]["ppl"]["before-text"]["all"]["ppl"] == 16.0
        assert gp["before_text"] == 16.0 and gp["before_shell_vs_text"] == 1.25
        assert not gp["before_forward_suspect"] and not gp["ratio_inverted"] and gp["ppl_forward"] == "VALID"
        # ---- THE GATE BRIDGE, closed end to end (2026-09-04) ---------------------------------------
        # `matrix_audit._run_verdict_wiring` proves each `GATE_SPECS` entry WITHHOLDS a verdict when it
        # VOIDs — but it builds the record it tests FROM the spec, so it cannot see a spec that addresses
        # a key the job never banks (a mutation test found exactly that hole in it on 2026-09-04). Only
        # a REAL job artifact can close that end. Every gate the verdict layer reads must therefore be
        # present in the half record `behaviour_verdicts` builds out of THIS cell's own metrics.
        from scripts import matrix_verdicts as MV
        _half = MV.behaviour_verdicts(m, "2")["halves"]["discovery"]
        _reach = {n for n, (key, field, _c) in MV.GATE_SPECS.items()
                  if key in _half and MV._stamps(_half[key], field)}
        assert _reach == set(MV.GATE_SPECS), (
            f"GATE_SPECS entries that address nothing this job banks: "
            f"{sorted(set(MV.GATE_SPECS) - _reach)}; the job's gate keys are {sorted(m['gates'])} "
            f"and the half record's are {sorted(_half)}")
        assert m["controls"]["lost"]["S"]["discovery"]["status"] in ("ok", "below-floor", "VOID-low-margin")
        assert "vqav2_cand" in m["controls"]["lost"]["O"] and "K" in m["controls"]["lost"]
        assert m["cols"]["O"]["after"]["all"]["vqav2_cand"]["n"] == 24 and m["cols"]["RFP"]["after"]["all"]["n"] == 24
        assert m["manifest"]["harness_shift"] is None
        pf = m["controls"]["preflight_c1"]
        assert pf["n_ref"] == 285 and pf["n_compared"] == 57 and set(pf) >= {"S_before", "S_after", "K_before", "K_after", "pass"}
        assert m["binds"]["stub"] and m["stages"]["done"] and m["stages"]["binds"]["frozen_lm"] is False
        pa = m["cols"]["P"]["after"]["discovery"]
        assert pa[str(pa["decisive_layer"])]["verdict"]["verdict"] and pa["decisive_layer"] == 2   # v3 probe_recovered landed
        # the random-init lower bound is a control that CAN fail now: the decoder re-init is asserted
        # to have moved the readout features, and the degenerate case (bound == healthy probe) is
        # banked per (half, layer) instead of passing silently as it did on 60 wave-1 records.
        assert m["stages"]["probe_random_init"]["n_reinit_params"] and m["stages"]["probe_random_init"]["feature_shift"]
        assert set(pa[str(pa["decisive_layer"])]) >= {"random_init", "random_init_identical_to_before"}
        assert pa["random_init_feature_shift"] and pa["random_init_reinit"]["n_random"]
        # §6.10 S_W / K_W (WMDP eval split) — the parent record the unlearning verdict had no source for
        from infra.aws_pg.jobs.matrix_recovery import _beh_record
        for col in ("S_W", "K_W"):
            for model in ("before", "after"):
                r = m["cols"][col][model]["eval"]
                assert set(r["per_item_hash"]) and r["n"] == 24 and r["floor"] == 0.25 and r["extra"]["shots"] == 0
                assert r["extra"]["samples"] and r["extra"]["set"] == "wmdp-eval"
                # the seam that matters: matrix_recovery reads THIS shape for its dose-0 identity gate
                # and its S_W_rec / K_W_rec fractions, and used to find nothing here ("ABSENT").
                assert _beh_record(m, col, model) is r, (col, model)
        assert m["cols"]["S_W-native"]["before"]["eval"]["acc"] is not None      # both letter surfaces
        assert m["manifest"]["items"]["wmdp_eval"]["n"] == 24
        assert {"S_W", "K_W"} <= set(m["samples_index"]) and m["samples_index"]["S_W"] == ["after", "before"]
        assert m["controls"]["lost"]["S_W"]["eval"]["status"] in ("ok", "below-floor", "VOID-low-margin")
        per_item = sorted(os.listdir(os.path.join(d, "out", "per_item")))
        assert "S_before_discovery.json" in per_item and "O_after_all.json" in per_item and "S_before-text_confirmation.json" in per_item
        assert os.path.exists(os.path.join(d, "out", "manifest.json")) and os.path.exists(os.path.join(d, "out", "controls.json"))
        ev = [json.loads(l) for l in open(os.path.join(d, "out", "events.jsonl"))]
        assert [e["stage"] for e in ev][:2] == ["manifest", "load_before-text"] and ev[-1]["stage"] == "done"
        assert m["stages"]["probe_random_init"]["t"] < m["stages"]["freed_before"]["t"] < m["stages"]["load_after"]["t"]
        # VOID-tokenizer reachable: an after-model whose ' A' is 2 tokens stamps S-family VOID, K still runs, exit 0
        members_orig, load_pair_orig = MM.members, MM.load_pair
        MM.members = lambda row: ["before", "after"]
        MM.load_pair = lambda row, device, which, adapter_path=None: _stub_member(
            which, row, salt="after-void" if which == "after" else which)
        c = parse(["--row", "ctrl-g2", "--cols", "S,S-k,F1,F2,F3,F4,F5,H,K,K-null,U", "--halves", "discovery",
                   "--out", os.path.join(d, "out2"), "--smoke"])
        os.environ["MDL_HARNESS_SHIFT"] = json.dumps({"transformers": "4.57.1 (model-card pin; base env 5.9.0)"})
        job2 = run(c)
        del os.environ["MDL_HARNESS_SHIFT"]
        m2 = json.load(open(job2.metrics_path))
        assert m2["manifest"]["harness_shift"]["transformers"].startswith("4.57.1")
        assert str(m2["cols"]["S"]["after"]["discovery"]["status"]).startswith("VOID-tokenizer") and m2["cols"]["F2"]["after"]["discovery"]["acc"] is not None
        assert m2["gates"]["tokenizer"]["discovery"]["S_family"] == "VOID-tokenizer" and m2["controls"]["lost"]["S"]["discovery"]["status"] == "VOID-tokenizer"
        assert m2["cols"]["K"]["after"]["discovery"]["acc"] is not None and m2["gates"]["u_valid"]["U"] in ("VALID", "VOID-U")
        # load-test path (row 4 shape) with stubs
        MM.ROWS["4"]["gated_on"] = None
        l = parse(["--row", "4", "--out", os.path.join(d, "out3"), "--load-test-only"])
        job3 = run(l)
        m3 = json.load(open(job3.metrics_path))
        assert set(m3["controls"]["load_test"]) == {"before", "after"} and m3["stages"]["done"]["load_test_only"]
        # SINGLE-COLUMN RE-RUN + the VOID branch of the forward gate (2026-09-03). `--cols ppl` on an
        # install row must bank a NORMAL cell — the three members, the bare-text anchor, gates, samples,
        # completeness — without re-buying any other column; and the measured defect (before 18558.6
        # through the shell against 7.966 bare) must stamp VOID-ppl-forward. The one-sided §1.2 flag
        # stays OFF here (ratio 0.0032), which is exactly why the gate exists.
        MM.members, MM.load_pair = members_orig, load_pair_orig
        me._ppl = lambda fwd, max_tokens: {"col": "ppl", "n_tokens": 100, "forward": "paligemma shell, blank gray image present (§6.5)",
                                           "ppl": {"before": 18558.6, "before-text": 7.966}.get(fwd.which, 59.2),
                                           "extra": _stub_ppl_extra(),
                                           "protocol": "shell-framed windows, BOS from the shell's own encoder (ERLANGEN c6b)"}
        p = parse(["--row", "2", "--cols", "ppl", "--halves", "discovery", "--out", os.path.join(d, "out4"), "--smoke"])
        job4 = run(p)
        m4 = json.load(open(job4.metrics_path))
        assert list(m4["cols"]) == ["ppl"] and set(m4["cols"]["ppl"]) == {"before-text", "before", "after"}
        g4 = m4["gates"]["ppl"]
        assert g4["ppl_forward"] == "VOID-ppl-forward" and g4["before_forward_suspect"] and g4["ppl_forward_factor"] == 20.0
        assert abs(g4["before_shell_vs_text"] - 2329.5) < 1 and g4["ratio_inverted"] and not g4["fluency_broken"]
        # §12.11 is a CONTENT gate, not a file-exists gate. `os.path.exists` was the whole check, and a
        # ppl cell writes `samples.json` whether or not anything is in it — three ppl cells banked
        # `{"cols": {}}` on S3 under a dry-run that passed. The samples must actually be there.
        assert m4["stages"]["done"]
        s4 = json.load(open(os.path.join(d, "out4", "samples.json")))
        assert "ppl" in s4["cols"], f"ppl banked no samples: {list(s4['cols'])}"
        b4 = s4["cols"]["ppl"]["before"]["all"]
        assert b4["samples"] and b4["samples"][0]["prompt_tail"], "ppl sample carries no prompt"
        assert m4["stages"]["samples"]["n_cols"] >= 1
        # ---- §6.5 CAUSALITY PROBE (2026-09-04) -----------------------------------------------------
        # The ppl cell now also banks, per member, the direct answer to the question the RATIO gate is
        # structurally unable to ask (its lower arm cannot fire below a bare ppl of 20 — see
        # `matrix_core.ppl_forward_gate`). The probe is banked here on the pipeline's own stub, whose
        # `continuation_logprobs` is a hash of (prompt, continuation) and is therefore NOT
        # autoregressive — so the end-to-end VOID branch is genuinely reachable in the job, not only in
        # a unit test:
        cz = m4["gates"]["causality"]
        assert set(cz) == {"before-text", "before", "after"}, cz
        assert all(str(v["causality"]).startswith("VOID-causality") for v in cz.values()), cz
        assert all(set(v["raw"]) >= {"joint_a", "joint_b", "tail_a", "tail_b", "n_span_tokens"} for v in cz.values())
        # …and BOTH clauses, against forwards whose behaviour is known by construction.
        class _CausalLP:
            """Scores each continuation word from THE WORD BEFORE IT ONLY: strictly autoregressive, so
            the chain rule is exact, and order-sensitive, so a scrambled span scores worse. A known
            bigram of the probe costs 0.5 nats and anything else costs 4.0."""
            shell = "text"
            _known = {(a, b) for t in (CAUSAL_PROBE["prompt"] + " " + CAUSAL_PROBE["span"] + " " + z
                                       for z in CAUSAL_PROBE["z"]) for a, b in zip(t.split(), t.split()[1:])}

            def continuation_logprobs(self, prompts, conts, images=None, sep=" "):
                out = []
                for p, c in zip(prompts, conts):
                    full, n = (p + " " + c).split(), len(c.split())
                    out.append((sum(-0.5 if (full[i - 1], full[i]) in self._known else -4.0
                                    for i in range(len(full) - n, len(full))), n))
                return out

        class _Leaky(_CausalLP):
            """The same, except every scored word also sees the LAST word of the sequence — the defining
            behaviour of a bidirectional mask, and the reason a later token can move an earlier span's
            logprob. (A whole-sequence term that is uniform per word would cancel in the chain-rule
            recovery; this one does not, because z_a and z_b end differently.)"""
            def continuation_logprobs(self, prompts, conts, images=None, sep=" "):
                out = []
                for (s, n), p, c in zip(super().continuation_logprobs(prompts, conts), prompts, conts):
                    out.append((s - 0.02 * n * len((p + " " + c).split()[-1]), n))
                return out

        class _ReadsTargets:
            """The ppl -> 1.0 failure: every target readable from its own position, so everything scores
            ~0 and the true and scrambled continuations are indistinguishable. This one PASSES the
            invariance clause (0 == 0) and is why the second clause exists."""
            shell = "text"

            def continuation_logprobs(self, prompts, conts, images=None, sep=" "):
                return [(-1e-4 * len(c.split()), len(c.split())) for c in conts]

        assert _causality_probe(_CausalLP())["causality"] == "VALID", _causality_probe(_CausalLP())
        leak = _causality_probe(_Leaky())
        assert "not autoregressive" in leak["causality"] and leak["later_token_drift"] > 0.01, leak
        copy = _causality_probe(_ReadsTargets())
        assert copy["later_token_drift"] < 1e-12, "the copying forward must PASS the invariance clause"
        assert "reading its targets" in copy["causality"], copy
        # ---- --after-text (2026-09-04): the install contrast with NO shell on either arm ------------
        # The seam that is stubbed here is `text_forward_for` ONLY (a text Forward over a resident VLM
        # needs weights); the REAL one is exercised against a real torch model in
        # `matrix_models._dryrun`. Everything else below is the real code: the registry's eligibility
        # answer, the member ordering, the banking keys, the encoder probe, and the two frozen §6.5
        # rules applied to the new read path.
        MM.text_forward_for = lambda member: _StubForward(f"{member['which']}|text")
        me._ppl = lambda fwd, max_tokens: {"col": "ppl", "n_tokens": 100, "forward": "text shell",
                                           # the MEASURED row-2 defect: the shell read text-only puts the
                                           # base decoder at 18558.6 where it reads 7.966 bare, so the
                                           # validity gate's VOID branch is the one this run exercises
                                           "ppl": {"before-text": 7.966, "before|text": 18558.6}.get(fwd.which, 59.2),
                                           "extra": _stub_ppl_extra(),
                                           "protocol": "shell-framed windows, BOS from the shell's own encoder (ERLANGEN c6b)"}
        t = parse(["--row", "2", "--cols", "S,K,ppl", "--halves", "discovery", "--out", os.path.join(d, "out5"),
                   "--smoke", "--after-text"])
        job5 = run(t)
        m5 = json.load(open(job5.metrics_path))
        # five text/shell members, and the two new keys are banked on the SAME columns as before-text
        assert [x["which"] for x in m5["members"]] == ["before-text", "before", "before-shell-text", "after", "after-text"]
        for col in ("S", "S-native", "K"):
            assert set(m5["cols"][col]) == {"before-text", "before", "before-shell-text", "after", "after-text"}, col
            assert m5["cols"][col]["after-text"]["discovery"]["acc"] is not None, col
        # the first member key of a column is UNCHANGED (matrix_verdicts._halved_cols reads next(iter(...)))
        assert next(iter(m5["cols"]["S"])) == "before-text" and next(iter(m5["cols"]["ppl"])) == "before-text"
        assert set(m5["cols"]["ppl"]) == {"before-text", "before", "before-shell-text", "after", "after-text"}
        ts = m5["controls"]["text_shell"]
        assert ts["pairs"]["contrast"] == ["before-shell-text", "after-text"]
        for pair in ("contrast", "validity"):
            for col in ("S", "K"):
                assert ts[pair][col]["discovery"]["status"] in ("ok", "below-floor", "VOID-low-margin"), (pair, col)
                assert ts[pair][col]["discovery"]["bootstrap"]["n_items"] == 57, (pair, col)
        # the frozen §6.5 rules, applied to the new read path — both verdicts reachable, neither invented
        assert set(ts["gates"]["text_parity"]["discovery"]) == {"S_shell", "K_shell"}
        assert ts["gates"]["ppl_text_forward"]["ppl_forward"] == "VOID-ppl-forward"
        assert ts["gates"]["ppl_text_forward"]["before_shell_text"] == 18558.6
        tep = ts["gates"]["text_encoder_parity"]
        assert tep["status"] == "VALID" and not tep["pairs_not_checked"]
        # EVERY scored pair is checked, and the pairs checked are exactly the pairs scored
        assert set(tep["pairs"]) == set(ts["pairs"]) == {"contrast", "validity"}, tep
        assert all(ts["encoders"][k]["bos_present"] for k in ("before-text", "before-shell-text", "after-text"))
        assert ts["encoders"]["after-text"]["repo"] == MM.ROWS["2"]["after"]
        assert ts["encoders"]["before-shell-text"]["repo"] == MM.ROWS["2"]["before"]
        assert m5["stages"]["text_shell"] and m5["samples_index"]["S"] == sorted(m5["cols"]["S"])
        assert "S_after-text_discovery.json" in os.listdir(os.path.join(d, "out5", "per_item"))
        # the text pass runs while the member is resident and BEFORE the destructive re-init / the free
        st5 = m5["stages"]
        assert st5["throughput_before-shell-text"]["t"] < st5["freed_before"]["t"] < st5["load_after"]["t"]
        assert st5["throughput_after-text"]["t"] < st5["freed_after"]["t"]
        # VOID branch of the encoder gate: a text arm whose encoder does not start with the model's BOS
        MM.text_forward_for = lambda member: _StubForward(
            f"{member['which']}|text", start=None if member["which"] == "before" else 5)
        t2 = parse(["--row", "2", "--cols", "S,K", "--halves", "discovery", "--out", os.path.join(d, "out6"),
                    "--smoke", "--after-text"])
        m6 = json.load(open(run(t2).metrics_path))
        g6 = m6["controls"]["text_shell"]["gates"]["text_encoder_parity"]
        assert g6["status"].startswith("VOID-text-encoder") and not g6["identical_probe_ids"]
        assert g6["pairs"]["contrast"]["identical_probe_ids"] is False, g6      # the arm that lost BOS
        assert g6["pairs"]["validity"]["identical_probe_ids"] is True, g6
        assert g6["bos_present"] == {"before-text": True, "before-shell-text": True, "after-text": False}
        # THE CASE THE OLD RULE WAS BLIND TO (2026-09-04). Both CONTRAST arms lose the start token
        # IDENTICALLY: `before-shell-text` and `after-text` agree with each other, so the pair the old
        # gate compared — before-text vs after-text — is the only thing that could have caught it, and
        # only by accident, since that pair is scored nowhere. Now the VALIDITY pair carries it.
        MM.text_forward_for = lambda member: _StubForward(f"{member['which']}|text", start=5)
        t3 = parse(["--row", "2", "--cols", "S,K", "--halves", "discovery", "--out", os.path.join(d, "out6b"),
                    "--smoke", "--after-text"])
        g6b = json.load(open(run(t3).metrics_path))["controls"]["text_shell"]["gates"]["text_encoder_parity"]
        assert g6b["pairs"]["contrast"]["identical_probe_ids"] is True, g6b     # both arms lost it together
        assert g6b["pairs"]["validity"]["identical_probe_ids"] is False, g6b
        assert g6b["status"].startswith("VOID-text-encoder") and "validity" in g6b["status"], g6b
        # a row with no text-shell reading SKIPS it and says so on the record — it does not crash
        s7 = parse(["--row", "ctrl-g2", "--cols", "S,K", "--halves", "discovery",
                    "--out", os.path.join(d, "out7"), "--smoke", "--after-text"])
        m7 = json.load(open(run(s7).metrics_path))
        assert m7["gates"]["text_members"]["status"].startswith("not run — row ctrl-g2")
        assert "text_shell" not in m7["controls"] and [x["which"] for x in m7["members"]] == ["before", "after"]
        assert set(m7["cols"]["S"]) == {"before", "after"}
        # completeness must fail loud when a column is missing
        raised = False
        try:
            completeness(job2, ["S", "RFP"], ["discovery"])
        except AssertionError:
            raised = True
        assert raised
    n_cols = len(m["cols"])
    print(f"PASS matrix_behaviour dry-run (row 2 smoke: {n_cols} column records incl. S_W/K_W, 3 members sequential, gates/lost/"
          f"preflight/binds stages, incremental metrics + per-item files; falsifiable random-init bound; "
          f"VOID-tokenizer reachable; load-test path; "
          f"`--cols ppl` single-column re-run banks a normal cell and VOID-ppl-forward is reachable; "
          f"`--after-text` banks before-shell-text/after-text on the resident members, both §6.5 gates "
          f"and an ineligible row skips it on the record; the §6.5 CAUSALITY probe banked per member "
          f"with both clauses reachable; the encoder gate now checks every SCORED pair, incl. the case "
          f"where both contrast arms lose the start token together, which the old pair could not see)")


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


if __name__ == "__main__":
    main()
