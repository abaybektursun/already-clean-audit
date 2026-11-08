"""PLENUM pure core — the NEW decision surfaces, weight-free and unit-testable on the control plane.

Spec: studies/lineage/paligemma1-pi0/research/2026-07-28_plenum_program_spec.md
Everything LACUNA already froze (S, the paired bootstrap, θ_null, the labels, BH) is IMPORTED from
`infra.pg.lacuna_core`, never re-implemented — the label rules must be the same object.

What is NEW here, and why:
  * `choices_for_task`  — the forced-choice readout generalised from VQAv2 to all four mm tasks
                          (§2.1: forced-choice is the PRIMARY readout family, because the
                          generation null band is zero-width);
  * `loo_r2_grouped` / `delta_loo_grouped` — the law's ΔLOO is leave-one-**UNIT**-out (§2.1), and
                          each unit contributes 6 faculty rows, so the fold must drop a GROUP.
                          `savb.loo_r2` drops one ROW, which would leak the held-out unit's other
                          five faculties into its own prediction;
  * `onset_statistic`   — the film's pre-registered order statistic (§2.3);
  * SAE gate predicates (§2.2) and the prediction-scoring table (§7, §11).

No try/except (CLAUDE.md). Heavy imports inside functions.
"""
from __future__ import annotations

from infra.pg.lacuna_core import (  # noqa: F401  (re-exported: the labels must be the same object)
    N_BOOT, S_RETAINED, bh_reject, boot_diff_ci, boot_mean_ci, boot_S_draws, boot_S_paired,
    delta_perception, label_row, p_above_band, power_gate, s_point, theta_null,
)

N_CAND = 4
CHANCE = 1.0 / N_CAND
# A distractor box this close to the gold box would be silently correct (RefCOCO is scored by IoU
# at 0.5 — `battery_tasks._refcoco_score`). 0.3 leaves a margin below the scoring threshold.
REFCOCO_MAX_DISTRACTOR_IOU = 0.3


# ===========================================================================
# §2.1 — the forced-choice readout, generalised over the four mm tasks.
# ===========================================================================
def _vqa_answers(row):
    return [a["answer"].strip().lower() for a in row["answers"]]


def gold_and_banned(task_name, row):
    """(gold_string, banned_strings) for a forced-choice item.

    `banned` is every string that would be SILENTLY CORRECT if drawn as a distractor — for the VQA
    tasks that is the full annotator set, for captions the item's other references. This is the one
    property that makes chance the true floor: without it a "distractor" can be a right answer and
    the readout's floor drifts below 0.25 for reasons that have nothing to do with the model.
    """
    from infra.pg import battery_tasks as BT
    if task_name == "vqav2":
        ans = _vqa_answers(row)
        from collections import Counter
        g = Counter(ans).most_common(1)[0][0]
        return g, set(ans) | {g}
    if task_name == "textvqa":
        ans = [a.strip().lower() for a in BT._textvqa_answers(row)]
        g = BT._textvqa_majority(row).strip().lower()
        return g, set(ans) | {g}
    if task_name == "coco_captions":
        g = BT._coco_target(row)
        return g, set(BT._coco_caps(row)) | {g}
    if task_name == "refcoco":
        return BT._refcoco_target(row), set()      # geometric exclusion, see `choices_for_task`
    raise RuntimeError(f"no forced-choice construction for task {task_name!r}")


def fc_prompt(prompt):
    """The prompt the forced-choice readout presents — the NATIVE one, unchanged.

    HISTORY, kept because the wrong version of this function produced a measurement that looked
    fine and meant nothing. v1 stripped the prompt's trailing newline to dodge a framing assertion
    (see `score_item_fc`). It did dodge it — and it also destroyed the readout: `PG1 | refcoco`
    came back at **0.2593** against a chance floor of 0.25. PaliGemma's '\\n' is the prompt/answer
    SEPARATOR, so removing it asks the model to continue `detect the man <loc…>` in a format it was
    never trained on. The crash was fixed and the instrument was broken, which is worse than the
    crash: a crash announces itself.

    The framing problem is now solved where it belongs — in the continuation extraction
    (`score_item_fc`) — so the prompt stays native for every task.
    """
    return prompt


def score_item_fc(model, proc, device, image, prompt, cands, trailing=1):
    """(summed logprob, ntok) per candidate — the forced-choice scorer, framing-robust.

    Same estimand as `lacuna_mmcloze.score_item`, and PROVABLY the same arithmetic wherever that
    one applies: it takes the continuation to be `full[lcp : -trailing]`, which is exactly what
    `score_item` computes when its `lcs` equals `trailing`. On VQAv2 and TextVQA the two therefore
    agree token-for-token, which is what keeps P1 (`pg1 | vqav2 | real = 0.7747`) a live check of
    this code rather than a check of code it replaced.

    What it does NOT do is require a shared SUFFIX between the prompt-only and prompt+candidate
    encodings. That requirement is what broke on RefCOCO: `REFCOCO_PREFIX_FMT` and `COCO_PREFIX`
    already end in '\\n', PaliGemma's processor appends another, and two adjacent newlines tokenize
    as one unit while newlines separated by a candidate do not — so the suffix vanished (`lcs=0`)
    even though nothing was wrong with the item.

    For a newline-terminated prompt the continuation then legitimately begins with the separator
    token. That token is shared by every candidate and sits after an identical prefix, so it
    contributes the SAME constant to all four sums and cannot move the argmax.
    """
    import torch
    from infra.pg.text_eval import _common_prefix_len, _score_cloze_window
    tok = proc.tokenizer
    orig = tok.padding_side
    tok.padding_side = "left"
    pe = proc(text=[prompt], images=[image], return_tensors="pt", padding="longest").to(device)
    prompt_real = pe["input_ids"][0][int((pe["attention_mask"][0] == 0).sum()):].tolist()
    enc = proc(text=[prompt + " " + c for c in cands], images=[image] * len(cands),
               return_tensors="pt", padding="longest").to(device)
    ids, am = enc["input_ids"], enc["attention_mask"]
    real = am.sum(dim=1).tolist()
    rows = [ids[k][ids.shape[1] - int(real[k]):].tolist() for k in range(len(cands))]
    # `trailing` stays FIXED at 1 whenever the candidates share their final token, which is the
    # arithmetic `score_item` performs and the reason P1 (`pg1 | vqav2 | real = 0.7747`) reproduces
    # exactly. Self-calibrating a maximal shared suffix would have been more general and would ALSO
    # have silently changed VQAv2 scores on any item whose four candidates happen to share a
    # penultimate token — trading away the one check that proves this code is the banked instrument.
    #
    # When the candidates do NOT share a final token the framing this scorer assumes does not hold
    # for that item. That happens on COCO captions, where sentence-final punctuation merges with the
    # processor's appended '\n' for some candidates and not others. Such an item is UNSCORABLE here,
    # and the honest response is to return None so the caller counts and skips it — not to crash the
    # run, and not to invent a continuation by guessing where the boundary was.
    if len({r[-1] for r in rows}) != 1:
        tok.padding_side = orig
        return None, None
    cont_ids, ntok, lcp0, tails = [], [], None, set()
    for k in range(len(cands)):
        full_real = rows[k]
        p = _common_prefix_len(prompt_real, full_real)
        cont = full_real[p:len(full_real) - trailing]
        assert len(cont) >= 1, f"empty continuation for candidate {k}"
        if lcp0 is None:
            lcp0 = p
        else:
            assert p == lcp0, f"candidates disagree on the shared prefix ({p} vs {lcp0})"
        tails.add(tuple(full_real[len(full_real) - trailing:]))   # shared by construction
        cont_ids.append(cont)
        ntok.append(len(cont))
    assert len(tails) == 1, (
        f"candidates do not share the {trailing}-token separator {tails} — `trailing` is wrong "
        f"for this processor and the scored window would be misaligned")
    K = max(ntok) + trailing + 1
    with torch.inference_mode():
        out = model(input_ids=ids, attention_mask=am,
                    pixel_values=enc["pixel_values"].to(torch.bfloat16), logits_to_keep=K)
        lw = torch.log_softmax(out.logits.float(), dim=-1)
    sums = _score_cloze_window(lw, cont_ids, ntok, trailing=trailing)
    tok.padding_side = orig
    assert torch.isfinite(sums).all(), "non-finite candidate logprob"
    return [float(x) for x in sums.tolist()], [int(m) for m in ntok]


def choices_for_task(task_name, rows, seed):
    """[(candidates, gold_idx)] per row — gold + (N_CAND-1) IN-DISTRIBUTION distractors.

    The distractor pool is the corpus's OWN answers/captions/boxes, so a wrong choice is a
    plausible output of the same task rather than an out-of-distribution string that any model
    could reject on form alone. Chance is 0.25 BY CONSTRUCTION for every task (§2.1) — a real
    floor, unlike the generation readout whose null band is zero-width.

    RefCOCO is the one task whose "already correct" test is geometric rather than textual: a
    distractor box that overlaps the gold box above `REFCOCO_MAX_DISTRACTOR_IOU` is rejected and
    redrawn, because the task's own scorer would count it correct.
    """
    import numpy as np
    from collections import Counter
    from infra.pg import battery_tasks as BT
    rng = np.random.default_rng(seed)
    golds, banned = [], []
    for r in rows:
        g, b = gold_and_banned(task_name, r)
        golds.append(g)
        banned.append(b)

    if task_name == "refcoco":
        boxes = [BT._refcoco_gt_box(r) for r in rows]
        out = []
        for i, g in enumerate(golds):
            picked, guard = [], 0
            while len(picked) < N_CAND - 1:
                j = int(rng.integers(0, len(rows)))
                guard += 1
                assert guard < 10_000, "refcoco distractor draw did not converge"
                if j == i or BT._iou(boxes[j], boxes[i]) > REFCOCO_MAX_DISTRACTOR_IOU:
                    continue
                cand = BT._refcoco_loc_str(boxes[j])
                if cand in picked or cand == g:
                    continue
                picked.append(cand)
            opts = picked + [g]
            order = rng.permutation(N_CAND)
            opts = [opts[k] for k in order]
            out.append((opts, opts.index(g)))
        return out

    if task_name == "coco_captions":
        pool = list(golds)                       # every image's own reference caption
    else:
        counts = Counter()
        for r in rows:
            counts.update(_vqa_answers(r) if task_name == "vqav2"
                          else [a.strip().lower() for a in BT._textvqa_answers(r)])
        pool = [a for a, _ in counts.most_common(200)]
    out = []
    for i, g in enumerate(golds):
        cand_pool = [c for c in pool if c not in banned[i]]
        assert len(cand_pool) >= N_CAND - 1, f"{task_name}: distractor pool exhausted at item {i}"
        d = [str(x) for x in rng.choice(cand_pool, size=N_CAND - 1, replace=False)]
        opts = d + [g]
        order = rng.permutation(N_CAND)
        opts = [opts[k] for k in order]
        out.append((opts, opts.index(g)))
    return out


# ===========================================================================
# §2.1 — the law: leave-one-UNIT-out predictive gain.
# ===========================================================================
def _fit_predict(X_tr, y_tr, X_te):
    """OLS (intercept + per-fold standardized predictors) — no leakage across the fold boundary.
    Same estimator as `savb._loo_pred`, applied to a GROUP of held-out rows."""
    import numpy as np
    X_tr = np.asarray(X_tr, np.float64)
    y_tr = np.asarray(y_tr, np.float64)
    X_te = np.asarray(X_te, np.float64)
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
    assert (sd > 0).all(), "constant predictor column in a LOO fold — the fit is not identified"
    A = np.column_stack([np.ones(len(y_tr)), (X_tr - mu) / sd])
    w, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
    return w[0] + ((X_te - mu) / sd) @ w[1:]


def loo_r2_grouped(X, y, groups):
    """Leave-one-GROUP-out R². `groups` is the unit id per row: dropping a unit drops ALL of its
    faculty rows at once, which is what "leave-one-UNIT-out" means (§2.1).

    Using row-wise LOO here would let a unit's other five faculties predict its sixth — the model
    would score well by memorising the unit's mean rather than by predicting from the recipe, which
    is the exact claim under test."""
    import numpy as np
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)
    g = np.asarray(groups)
    uniq = list(dict.fromkeys(g.tolist()))
    assert len(uniq) >= 4, f"need ≥4 units to leave one out (got {len(uniq)})"
    preds = np.empty(y.size)
    for u in uniq:
        te = g == u
        preds[te] = _fit_predict(X[~te], y[~te], X[te])
    sst = float(((y - y.mean()) ** 2).sum())
    assert sst > 0, "constant outcome — nothing to predict"
    return float(1.0 - ((y - preds) ** 2).sum() / sst), preds


def delta_loo_grouped(X_base, x_add, y, groups, n_perm=2000, seed=1337):
    """ΔLOO-R²(+x_add | X_base) with a group-permutation null.

    The permutation shuffles `x_add` ACROSS UNITS (not across rows), preserving each unit's
    within-unit faculty pattern. Shuffling rows would destroy the unit structure and make the null
    trivially easy to beat — a one-sided threshold that could not fail
    (`feedback_one_sided_thresholds`).

    Returns both tails: `p_perm` is the registered positive direction, `p_perm_neg` makes a
    NEGATIVE landing (the predictor actively hurting) visible rather than hidden."""
    import numpy as np
    X_base = np.asarray(X_base, np.float64)
    x_add = np.asarray(x_add, np.float64).reshape(-1, 1)
    y = np.asarray(y, np.float64)
    g = np.asarray(groups)
    uniq = list(dict.fromkeys(g.tolist()))
    r2_b, _ = loo_r2_grouped(X_base, y, g)
    r2_a, _ = loo_r2_grouped(np.column_stack([X_base, x_add]), y, g)
    obs = r2_a - r2_b
    # the per-unit value of the added predictor (constant within a unit for membership/‖ΔW‖ is NOT
    # assumed — we permute the unit's whole block of values, which handles both cases)
    blocks = {u: x_add[g == u] for u in uniq}
    rng = np.random.default_rng(seed)
    hi = lo = 0
    for _ in range(n_perm):
        order = rng.permutation(len(uniq))
        xp = np.empty_like(x_add)
        for k, u in enumerate(uniq):
            src = blocks[uniq[order[k]]]
            tgt = g == u
            n_t = int(tgt.sum())
            # unit blocks are equal-length by construction (same faculty grid per unit); assert it
            assert src.shape[0] == n_t, "unit blocks differ in length — the grid is not uniform"
            xp[tgt] = src
        d = loo_r2_grouped(np.column_stack([X_base, xp]), y, g)[0] - r2_b
        hi += int(d >= obs)
        lo += int(d <= obs)
    return {"r2_base": r2_b, "r2_aug": r2_a, "delta": float(obs),
            "p_perm": float((1 + hi) / (1 + n_perm)),
            "p_perm_neg": float((1 + lo) / (1 + n_perm)),
            "n_perm": int(n_perm), "n_units": len(uniq), "n_rows": int(y.size)}


def boot_delta_loo(X_base, x_add, y, groups, n_boot=1000, seed=11):
    """Bootstrap CI for ΔLOO, resampling UNITS with replacement (§2.1 "bootstrap CI").
    H_recipe is supported only if this CI excludes 0 (§6) — a point estimate is not enough."""
    import numpy as np
    X_base = np.asarray(X_base, np.float64)
    x_add = np.asarray(x_add, np.float64).reshape(-1, 1)
    y = np.asarray(y, np.float64)
    g = np.asarray(groups)
    uniq = list(dict.fromkeys(g.tolist()))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx, grp = [], []
        for rep, k in enumerate(pick):
            sel = np.nonzero(g == uniq[k])[0]
            idx.extend(sel.tolist())
            grp.extend([f"{uniq[k]}#{rep}"] * sel.size)     # resampled units are distinct folds
        idx = np.asarray(idx)
        if len({s.split("#")[0] for s in grp}) < 4:
            continue
        Xb, xa, yy = X_base[idx], x_add[idx], y[idx]
        if not (Xb.std(axis=0) > 0).all() or xa.std() == 0 or yy.std() == 0:
            continue
        d = (loo_r2_grouped(np.column_stack([Xb, xa]), yy, grp)[0]
             - loo_r2_grouped(Xb, yy, grp)[0])
        draws.append(d)
    d = np.asarray(draws, dtype=float)
    assert d.size > 0.5 * n_boot, f"ΔLOO bootstrap degenerate: {d.size}/{n_boot} usable draws"
    return {"lo95": float(np.percentile(d, 2.5)), "hi95": float(np.percentile(d, 97.5)),
            "median": float(np.median(d)), "n_draws": int(d.size),
            "excludes_zero": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)}


def mde_pattern(grid):
    """§3.1 — classify the 2×2 cross-survival grid, mechanically.

    `grid[(unit, faculty)] = S`. Units are the two FTs, faculties the two each was trained on.
    H_recipe → ANTI-DIAGONAL (each unit keeps its own faculty, drops the other's);
    H_fragility → SAME-ORDER (both units rank the two faculties identically);
    H_magnitude → UNIFORM (no within-unit split).

    All three can lose: `unpatterned` is a legal, reportable outcome (§6 "all four can lose").
    """
    units = sorted({u for u, _ in grid})
    facs = sorted({f for _, f in grid})
    assert len(units) == 2 and len(facs) == 2, f"MDE grid must be 2×2, got {units} × {facs}"
    u1, u2 = units
    f1, f2 = facs
    d1 = grid[(u1, f1)] - grid[(u1, f2)]
    d2 = grid[(u2, f1)] - grid[(u2, f2)]
    split = 0.15          # pre-set convention: a within-unit faculty split worth naming
    if abs(d1) < split and abs(d2) < split:
        return {"pattern": "uniform", "supports": "H_magnitude", "d1": d1, "d2": d2}
    if d1 * d2 < 0:
        return {"pattern": "anti_diagonal", "supports": "H_recipe", "d1": d1, "d2": d2}
    if d1 * d2 > 0:
        return {"pattern": "same_order", "supports": "H_fragility", "d1": d1, "d2": d2}
    return {"pattern": "unpatterned", "supports": None, "d1": d1, "d2": d2}


# ===========================================================================
# §2.3 — the film's order statistic.
# ===========================================================================
def onset_step(steps, s_values):
    """First checkpoint whose S drops BELOW HALF its step-0 value (§2.3). None if it never does —
    reported as undefined, never interpolated (§8 threat row)."""
    assert len(steps) == len(s_values) and len(steps) >= 2, "need a checkpoint grid"
    assert steps[0] == 0, f"the film's grid must start at step 0 (got {steps[0]})"
    s0 = float(s_values[0])
    if s0 <= 0:
        return {"onset": None, "reason": "step-0 S is not positive — no decline is definable",
                "s0": s0}
    thr = 0.5 * s0
    for st, v in zip(steps[1:], s_values[1:]):
        if float(v) < thr:
            return {"onset": int(st), "threshold": thr, "s0": s0}
    return {"onset": None, "reason": "S never fell below half its step-0 value", "threshold": thr,
            "s0": s0}


def acquisition_step(steps, pc_success):
    """First checkpoint with pc_success ≥ 0.5 × its FINAL value (§2.3). None if never."""
    assert len(steps) == len(pc_success) and len(steps) >= 2, "need a checkpoint grid"
    final = float(pc_success[-1])
    if final <= 0:
        return {"acquisition": None, "reason": "control never acquired (final pc_success is 0)",
                "final": final}
    thr = 0.5 * final
    for st, v in zip(steps, pc_success):
        if float(v) >= thr:
            return {"acquisition": int(st), "threshold": thr, "final": final}
    return {"acquisition": None, "reason": "no checkpoint reached half the final competence",
            "threshold": thr, "final": final}


def onset_statistic(steps, s_by_faculty, pc_success):
    """The pre-registered order statistic `onset(faculty) − acquisition(control)` per faculty (§2.3).

    Sign convention, frozen here so it cannot be reinterpreted later:
      negative → consumption PRECEDES control acquisition  (H_consume-to-install)
      ≈ zero   → co-timed
      positive → the install lands FIRST, consumption follows
      undefined → either term is None; reported as undefined, and an undefined term NEVER
                  contributes a sign to the verdict.
    """
    acq = acquisition_step(steps, pc_success)
    out = {"acquisition": acq, "per_faculty": {}}
    for f, s in s_by_faculty.items():
        ons = onset_step(steps, s)
        d = (None if (ons["onset"] is None or acq["acquisition"] is None)
             else int(ons["onset"]) - int(acq["acquisition"]))
        out["per_faculty"][f] = {"onset": ons, "delta": d,
                                 "reading": ("undefined" if d is None else
                                             "consume_first" if d < 0 else
                                             "install_first" if d > 0 else "co_timed")}
    defined = [v["delta"] for v in out["per_faculty"].values() if v["delta"] is not None]
    out["n_defined"] = len(defined)
    out["n_undefined"] = len(out["per_faculty"]) - len(defined)
    out["median_delta"] = (None if not defined else float(sorted(defined)[len(defined) // 2]))
    return out


# ===========================================================================
# §2.2 — the SAE validity gates. Each can FAIL; failure is terminal (VOID), not a retry loop.
# ===========================================================================
DEAD_FRAC_MAX = 0.5
BEHAVIOR_TOL = 0.10          # frozen tolerance: reconstruction-patched battery within 10pp
# PRE-FREEZE AMENDMENT (found by the G0 dry-run, 2026-07-28). The identity gate was written as the
# bare inequality `same_seed_cos > cross_stream_cos`. That is a threshold that cannot fail: a
# featurizer with same=0.45 / cross=0.44 PASSES on a 0.01 margin that is pure sampling noise, which
# is the exact shape `feedback_one_sided_thresholds` names. The gate now requires BOTH
#   (a) DISCRIMINATION — a margin at least IDENTITY_MIN_MARGIN wide, and
#   (b) SELF-CONSISTENCY — the two seeds of one stream actually agree (IDENTITY_MIN_SELF),
# because "the same feature" is undefined if a featurizer does not reproduce itself across seeds.
# Both are pre-set conventions, disclosed as such, and each can fail on its own.
IDENTITY_MIN_MARGIN = 0.15   # the house POWER_MIN_GAP convention, applied on the cosine scale
IDENTITY_MIN_SELF = 0.50


def sae_gates(dead_frac, same_seed_cos, cross_stream_cos, behav_native, behav_patched):
    """The three §2.2 gates + P4, computed together so a stream's verdict is one object.

    identity gate: the two seeds of the SAME stream must agree with EACH OTHER (self-consistency)
    and must do so by a real margin over the cross-STREAM match (discrimination) — see the
    AMENDMENT note above. A featurizer that matches every stream equally well fails, and one that
    does not reproduce itself across seeds fails, so the gate has a ceiling as well as a floor.
    """
    dead_ok = bool(dead_frac <= DEAD_FRAC_MAX)
    margin = float(same_seed_cos) - float(cross_stream_cos)
    ident_ok = bool(margin >= IDENTITY_MIN_MARGIN and same_seed_cos >= IDENTITY_MIN_SELF)
    behav_ok = bool(abs(behav_patched - behav_native) <= BEHAVIOR_TOL)
    gates = {
        "G_dead": {"value": float(dead_frac), "max": DEAD_FRAC_MAX, "passes": dead_ok},
        "G_identity": {"same_seed_cos": float(same_seed_cos),
                       "cross_stream_cos": float(cross_stream_cos),
                       "margin": margin, "min_margin": IDENTITY_MIN_MARGIN,
                       "min_self": IDENTITY_MIN_SELF,
                       "fails_on": (None if ident_ok else
                                    "no_discrimination" if margin < IDENTITY_MIN_MARGIN
                                    else "seeds_disagree_with_themselves"),
                       "passes": ident_ok},
        "G_behavior": {"native": float(behav_native), "patched": float(behav_patched),
                       "abs_diff": float(abs(behav_patched - behav_native)),
                       "tol": BEHAVIOR_TOL, "passes": behav_ok},
    }
    failed = [k for k, v in gates.items() if not v["passes"]]
    gates["all_pass"] = not failed
    gates["void_gate"] = None if not failed else failed[0]
    gates["note"] = ("gates pass — feature cells are measurable" if not failed else
                     f"stream VOID at {failed[0]} — feature cells terminate as VOID (§2.2)")
    return gates


def emergent_frac(child_to_parent_cos, identity_floor):
    """Fraction of CHILD features with no parent match at or above the cross-seed identity floor.

    The floor is not a free parameter: it is the same-stream cross-seed matched cosine (what "the
    same feature" means for this featurizer). Reported with the floor attached — an emergent_frac
    without its floor is uninterpretable."""
    import numpy as np
    c = np.asarray(child_to_parent_cos, dtype=float)
    assert c.ndim == 1 and c.size > 0, "need per-feature best-match cosines"
    return {"emergent_frac": float((c < identity_floor).mean()),
            "identity_floor": float(identity_floor), "n_features": int(c.size),
            "median_match": float(np.median(c))}


# ===========================================================================
# §7, §11 — sealed predictions and the prediction-scoring table.
# ===========================================================================
def score_predictions(sealed, observed):
    """The §11 row-3 deliverable: sealed per-cell predictions vs what landed.

    A cell the program did not measure scores as `unmeasured` and is EXCLUDED from the hit rate —
    the rate is over cells that were actually adjudicated, and the exclusions are counted in the
    open so the denominator can never quietly shrink (`feedback_signals_that_cannot_fail`).
    """
    rows, hit, miss, unmeasured = [], 0, 0, 0
    for cell, pred in sorted(sealed.items()):
        obs = observed.get(cell)
        if obs is None:
            rows.append({"cell": cell, "predicted": pred, "observed": None, "outcome": "unmeasured"})
            unmeasured += 1
            continue
        ok = (str(pred) == str(obs))
        rows.append({"cell": cell, "predicted": pred, "observed": obs,
                     "outcome": "hit" if ok else "miss"})
        hit += int(ok)
        miss += int(not ok)
    n = hit + miss
    return {"n_sealed": len(sealed), "n_scored": n, "n_unmeasured": unmeasured,
            "hit": hit, "miss": miss,
            "hit_rate": (None if n == 0 else hit / n),
            "note": ("hit_rate is over ADJUDICATED cells only; unmeasured cells are listed, not "
                     "dropped silently"),
            "rows": rows}


def predictor_matrix(unit_rows, faculty_order):
    """(X_membership, X_dwnorm, X_similarity, y, groups) for the law fit.

    `unit_rows`: [{unit, dw_norm, recipe_faculties, similarity: {faculty: float},
                   S: {faculty: float}}]. Rows with a missing S are dropped and COUNTED (a VOID
    cell must not silently become a zero).
    """
    import numpy as np
    mem, dw, sim, y, grp, dropped = [], [], [], [], [], []
    for u in unit_rows:
        for f in faculty_order:
            s = u["S"].get(f)
            if s is None:
                dropped.append(f"{u['unit']}|{f}")
                continue
            mem.append(1.0 if f in u["recipe_faculties"] else 0.0)
            dw.append(float(u["dw_norm"]))
            sim.append(float(u["similarity"][f]))
            y.append(float(s))
            grp.append(u["unit"])
    return (np.asarray(mem), np.asarray(dw), np.asarray(sim), np.asarray(y), grp,
            {"n_rows": len(y), "n_dropped": len(dropped), "dropped": dropped})
