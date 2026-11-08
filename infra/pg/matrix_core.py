"""MENDELEEV pure core v3 — every frozen rule of §6 of
studies/lineage/cross-lineage/research/2026-09-02_mendeleev_spec.md, as code.

Pure numpy; asserts; `python -m infra.pg.matrix_core` runs the dry-run gate: every verdict branch
reachable (including every INDETERMINATE band), degenerate cases handled, NaN inputs REJECTED (a dead
readout must raise, never land in an indeterminate band), no model, no GPU.
Function names carry the spec rule number so a reader can check each against its sentence.
v3 folds the 2026-09-03 second-pass cold-reader audit (D1–D35 in the journal); every changed rule is
mirrored in the spec (v2.3) in the same commit.
"""
import numpy as np

LN4 = float(np.log(4.0))
MIN_HEALTHY_MARGIN = 0.15      # §6.1: a cell is VOID-low-margin below this
INDET = "INDETERMINATE (counts for neither)"
FORMAT_VANISH = 0.3            # §6.2b: the bar below which a format's loss "vanishes" — and, by the same
                               # number, the minimum PRIMARY loss that poses the H_format/H_stage question
                               # at all (deviation D-E, 2026-09-03; see `format_verdict`). One threshold,
                               # used twice; no new number was introduced by the domain clause.
PRIMARY_RECIPE = "r512"        # §6.3/§6.6: the REPRICE recipe; the only one licensed on every recovery row
CONTENTION_MARGIN = 0.10       # §6.6 D9: how far `overlap` must clear the unrelated-update null. THE one
                               # place this number exists — `contention` and `subspace_overlap` both
                               # default to it and `scripts/matrix_verdicts` reads it from here.


def _finite(*xs):
    for x in xs:
        assert x is not None and np.isfinite(x), f"non-finite input to a verdict rule: {xs} (VOID must be stamped upstream, never scored)"


def _isfin(x) -> bool:
    """True only for a real, present number. `None` and NaN are both 'no measurement'."""
    return x is not None and bool(np.isfinite(x))


# ---------------------------------------------------------------- §6.1 margins
def margin_lost(acc_before: float, acc_after: float, floor: float) -> dict:
    """Fraction of the above-floor margin lost. VOID-low-margin when the healthy margin is
    < MIN_HEALTHY_MARGIN (a broken readout with both models at floor must not read 'kept').
    Values > 1 mean the after-model is below floor: reported as 'below floor', clipped to 1 for verdicts.
    A VOID cell carries lost = nan and MUST NOT be passed to any verdict function (they assert).

    DEFECT (2026-09-03 adversarial audit, wiring repair). A non-finite `acc_before` fell straight through
    every branch: `nan - floor < MIN_HEALTHY_MARGIN` is False, so the VOID-low-margin branch did not
    fire; `min(nan, 1.0)` is nan; and `nan > 1` is False, so the record came back with
    `status = "ok"` — a dead readout stamped healthy. This is THE producer of the `lost` every §6 rule
    reads, so it is also the right place to say 'no measurement'. It now returns lost = nan with an
    explicit VOID status, which `_scored`/`_absent` downstream already read as INPUTS-MISSING.
    NO THRESHOLD MOVED: on finite inputs every branch and every number is exactly as before."""
    if not (_isfin(acc_before) and _isfin(acc_after) and _isfin(floor)):
        return {"lost": np.nan, "healthy_margin": np.nan,
                "status": f"VOID-nonfinite (acc_before={acc_before}, acc_after={acc_after}, "
                          f"floor={floor} — a non-finite accuracy or floor is not a measurement)"}
    mb = acc_before - floor
    if mb < MIN_HEALTHY_MARGIN:
        return {"lost": np.nan, "status": "VOID-low-margin", "healthy_margin": float(mb)}
    raw = float((mb - (acc_after - floor)) / mb)
    return {"lost": min(raw, 1.0), "raw": raw, "status": "below-floor" if raw > 1 else "ok",
            "healthy_margin": float(mb)}


def recovered_fraction(acc_x: float, acc_after: float, acc_before: float) -> float:
    lost = acc_before - acc_after
    assert lost > 1e-9, f"nothing lost (before {acc_before}, after {acc_after})"
    return float((acc_x - acc_after) / lost)


def bootstrap_fraction(items_before, items_after, items_x, n_boot=2000, seed=1337):
    """§6.1. Paired-item bootstrap; every draw enters the lost-margin and difference CIs. The ratio
    (recovered fraction) is reported only when the CI of the lost margin (before − after) excludes 0;
    the ratio's own CI is then computed over the draws with lost > 0 (≤ 2.5 % of draws by construction,
    the excluded fraction is reported). Otherwise the statistic is the absolute difference with its CI and
    the ratio is N/A. `p_wrong_side` = fraction of draws with diff ≤ 0 — the bootstrap p used by BH (§6.13).

    DEFECT (2026-09-03 adversarial audit, wiring repair). The per-item vectors were never checked for
    finiteness. An all-NaN arm makes every `diff` draw NaN, `(diff <= 0)` is then False everywhere, and
    `p_wrong_side` comes back 0.0 — the MOST significant value the statistic can take — which
    `bh_reject` then accepts. A dead readout was the strongest evidence the bootstrap could produce.
    NaN in, assertion out (the module contract): VOID is stamped upstream, never scored."""
    b, a, x = (np.asarray(v, float) for v in (items_before, items_after, items_x))
    n = b.size
    assert a.size == x.size == n and n >= 20, n
    assert np.isfinite(b).all() and np.isfinite(a).all() and np.isfinite(x).all(), (
        f"non-finite per-item values reached the bootstrap (before {int((~np.isfinite(b)).sum())}, "
        f"after {int((~np.isfinite(a)).sum())}, x {int((~np.isfinite(x)).sum())} of {n}): every draw "
        "would be NaN and `p_wrong_side` would read 0.0, maximal significance, from a dead readout")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    lost = b[idx].mean(1) - a[idx].mean(1)
    diff = x[idx].mean(1) - a[idx].mean(1)
    lost_ci = (float(np.percentile(lost, 2.5)), float(np.percentile(lost, 97.5)))
    diff_ci = (float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5)))
    out = {"n_items": int(n), "lost_point": float(b.mean() - a.mean()), "lost_ci": lost_ci,
           "diff_point": float(x.mean() - a.mean()), "diff_ci": diff_ci,
           "p_wrong_side": float((diff <= 0).mean())}
    if lost_ci[0] > 0:
        ok = lost > 0
        r = diff[ok] / lost[ok]
        out.update({"ratio_point": recovered_fraction(x.mean(), a.mean(), b.mean()),
                    "ratio_ci": (float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))),
                    "ratio_excluded_frac": float(1 - ok.mean()), "ratio_status": "reported"})
    else:
        out.update({"ratio_point": None, "ratio_ci": None, "ratio_status": "N/A (lost-margin CI includes 0)"})
    return out


def bh_reject(pvals, q: float = 0.05) -> list:
    """§6.13 Benjamini–Hochberg at q over one family of bootstrap p-values (fraction of draws on the wrong
    side). Returns the reject mask. Used only for families of INDEPENDENT tests (H_split across hops,
    F1–F4 within a row, IM signs); never for a conjunctive rule."""
    p = np.asarray(pvals, float); assert p.size >= 1 and np.all(np.isfinite(p))
    m = p.size; order = np.argsort(p); ranked = p[order]
    thresh = q * (np.arange(1, m + 1) / m)
    below = np.flatnonzero(ranked <= thresh)
    k = int(below.max()) + 1 if below.size else 0
    rej = np.zeros(m, bool); rej[order[:k]] = True
    return rej.tolist()


# ---------------------------------------------------------------- §6.2 H_split (with indeterminate band)
def split_verdict(S_lost: float, K_lost: float, diff_ci_lo: float) -> dict:
    """H_split ⟺ S_lost ≥ 0.6 ∧ K_lost ≤ 0.5 ∧ (S − K) ≥ 0.3 ∧ CI-low(S − K) > 0.
    KILLED ⟺ (S − K) < 0.15 ∨ K_lost ≥ S_lost. Otherwise INDETERMINATE. Inputs must be finite."""
    _finite(S_lost, K_lost, diff_ci_lo)
    d = S_lost - K_lost
    if S_lost >= 0.6 and K_lost <= 0.5 and d >= 0.3 and diff_ci_lo > 0:
        v = "H_split"
    elif d < 0.15 or K_lost >= S_lost:
        v = "H_split KILLED"
    else:
        v = "H_split " + INDET
    return {"verdict": v, "S_lost": S_lost, "K_lost": K_lost, "diff": d, "diff_ci_lo": diff_ci_lo}


def k_positive_control(K_lost_on_destroyed_pair: float) -> dict:
    """§6.2: the K column is VALID program-wide only if K drops ≥ 0.5 on a pair whose content
    is known to be destroyed (row 5c). Otherwise every K cell is VOID (the C6 lesson)."""
    _finite(K_lost_on_destroyed_pair)
    ok = K_lost_on_destroyed_pair >= 0.5
    return {"K_column": "VALID" if ok else "VOID (K did not fall on a content-destroyed pair)"}


def k_shuffle_null(acc_shuffled: float, floor: float = 0.25, tol: float = 0.05) -> dict:
    """§6.13 K null: option texts of item i scored against item π(i)'s question must land at floor:
    acc ≤ floor + tol (0.30 at n = 285). Otherwise the row's K is VOID-K-null (a length/frequency prior
    is doing the scoring)."""
    _finite(acc_shuffled)
    ok = acc_shuffled <= floor + tol
    return {"K_null": "at floor" if ok else "VOID-K-null", "acc_shuffled": acc_shuffled}


def k_shell_parity(K_text_only: float, K_shell: float, S_text_only: float, S_shell: float, tol: float = 0.05) -> dict:
    """§6.5 (D15): on an install row the healthy LM's text-only and blank-image-shell readouts must agree
    to |Δ| ≤ tol for BOTH S and K, or the shell path is VOID-shell for that readout."""
    _finite(K_text_only, K_shell, S_text_only, S_shell)
    return {"S_shell": "VALID" if abs(S_text_only - S_shell) <= tol else "VOID-shell",
            "K_shell": "VALID" if abs(K_text_only - K_shell) <= tol else "VOID-shell"}


def ppl_forward_gate(ppl_shell: float, ppl_bare: float, factor: float = 20.0) -> dict:
    """§6.5 ppl forward parity: the SAME (before) model read through the row's shell and read BARE must
    agree within `factor`, or the row's `ppl` column is VOID-ppl-forward. Two-sided, exactly like
    `k_shell_parity` above: this compares two measurements of ONE set of weights, so it needs no
    per-family re-justification — unlike an absolute perplexity threshold, which would be an invented
    number that means something different on every family.

    Promoted from diagnostic to verdict on 2026-09-03 (§13 lesson 39). The §1.2 fluency rule is
    ONE-SIDED (after/before > 20), so a defect that INFLATES the before arm deflates the ratio and
    suppresses the flag; this gate is what sees that direction.

    Why 20, from artifacts, not taste. Re-framing the SAME weights moves perplexity by SINGLE DIGITS on
    this family: PaliGemma-2's decoder reads 42.79 with no start token and 17.29 with one through the
    identical call — 2.48x from a one-token framing change (`results/bosgate_ppl/a1/out/metrics.json`,
    20k tokens); PaliGemma-1 reads 36.8 through a blank-image shell against 22.4 bare, 1.64x
    (`memory/reference_code_traps.md`). The forward defect this gate exists to catch moved it 2329x
    (row 2 `before`: 18558.6 through the shell against 7.966 bare). A factor of 20 therefore sits ~8x
    above the largest framing effect ever measured here and ~100x below the observed break: no healthy
    re-framing is voided, and nothing of that character can pass. It is a gate on the INSTRUMENT, so a
    VOID stamp means the row's fluency ratio is not a measurement — not that the model is broken.

    WHAT THIS GATE CANNOT SEE, stated (2026-09-04; `matrix_audit` proves it on every run). It is written
    as two-sided and the LOWER arm is UNREACHABLE in the regime the instrument produces. That arm needs
    `ppl_shell < ppl_bare / 20`; perplexity is ≥ 1; the banked bare perplexity of row 2's `before` is
    7.966; so the ratio bottoms out at 1/7.966 = 0.126 while the arm needs < 0.05, and it can never fire.
    A shell that made the text read PERFECTLY — the signature of a bidirectional mask letting each
    position see its own target, driving perplexity toward 1.0 — is stamped VALID at ratio 0.125.
    Widening the factor does not help, and neither does a symmetric `max(a/b, b/a)` repair: 8.43x sits
    comfortably inside a 20x tolerance. NO ratio of two perplexities can see this failure, because the
    failure drives BOTH arms toward 1.0 together, so their ratio stays near 1 exactly when the instrument
    is most broken. `causality_gate` below answers that question by measuring the property directly
    instead of inferring it from a magnitude. This gate keeps its one reachable job — catching a shell
    arm that reads far HARDER than bare, which is what actually happened on row 2 at 2329x — and its
    threshold is UNCHANGED."""
    _finite(ppl_shell, ppl_bare)
    assert ppl_shell > 0 and ppl_bare > 0 and factor > 1, f"ppl must be positive and factor > 1: {(ppl_shell, ppl_bare, factor)}"
    ratio = float(ppl_shell) / float(ppl_bare)
    ok = (1.0 / factor) <= ratio <= factor
    return {"ppl_forward": "VALID" if ok else "VOID-ppl-forward", "shell_vs_bare": ratio, "factor": float(factor)}


# `causality_gate`'s tolerance is NOT a new number: 1e-2 in logit/logprob units is the bar this program
# already uses for "the same computation, twice, in bf16" — `graft_identity_gate` requires
# max|Δlogit| ≤ 1e-2 for a self-graft at batch 4. Reused here for the same reason, on the same hardware.
CAUSAL_TOL = 1e-2


def causality_gate(span_lp_a: float, span_lp_b: float, lp_true: float, lp_scrambled: float,
                   n_tokens: int = 1, tol: float = CAUSAL_TOL) -> dict:
    """§6.5 forward causality — is the shell's forward actually AUTOREGRESSIVE? Two clauses, both required.

    WHY THIS EXISTS AND WHY IT IS NOT A PERPLEXITY RATIO. `ppl_forward_gate` above is a ratio of two
    magnitudes, and the defect it was written for — a bidirectional mask, so every position can read its
    own target and the text reads perfectly — moves BOTH magnitudes toward 1.0 at once. Its lower arm is
    then unreachable (see that docstring: the ratio cannot fall below 0.126 at the banked bare
    perplexity, and the arm needs 0.05). The property in question is not "how big is the perplexity"; it
    is "can a token's score depend on a token that comes after it". So measure THAT.

    THE MEASUREMENT (`matrix_behaviour._causality_probe` produces these four numbers on the resident
    member, from `Forward.continuation_logprobs` alone — no new forward API, no weights loaded twice):
      span_lp_a  the logprob of a span S after prompt P, recovered from a sequence in which S is
                 FOLLOWED by token-run z_a:  lp(P, S·z_a) − lp(P·S, z_a).  Under the chain rule of an
                 autoregressive model this is exactly lp(S | P), whatever z_a is.
      span_lp_b  the same quantity recovered with a DIFFERENT following run z_b.
      lp_true / lp_scrambled   the per-token logprob of the true continuation and of a scrambled
                 continuation of the same token length after the same prompt.

    CLAUSE 1 — later-token invariance. |span_lp_a − span_lp_b| / n_tokens ≤ tol. An earlier span's
    logprob must not move when a LATER token is perturbed. Under a causal mask the two recoveries are
    the same number; under any leakage of the future into the past they are not.

    CLAUSE 2 — target discrimination. (lp_true − lp_scrambled) / n_tokens > tol. Clause 1 alone is
    satisfied by the WORST case: a forward that reads every target off its own position scores
    everything at ~0 nats, so both recoveries are ~0 and 0 == 0 passes. A forward that predicts a
    scrambled continuation exactly as well as the true one is not predicting — it is copying. This is a
    contrast between two readings of ONE model, not an absolute perplexity threshold, so it needs no
    per-family justification.

    A VOID here is a statement about the INSTRUMENT: the row's `ppl` column, and every readout taken
    through that forward, is not a measurement."""
    _finite(span_lp_a, span_lp_b, lp_true, lp_scrambled)
    assert n_tokens >= 1 and tol > 0, f"n_tokens ≥ 1 and tol > 0: {(n_tokens, tol)}"
    drift = float(abs(span_lp_a - span_lp_b)) / n_tokens
    sep = float(lp_true - lp_scrambled) / n_tokens
    fails = []
    if drift > tol:
        fails.append(f"a later token moved an earlier span's logprob by {drift:.4g}/token (> {tol}) — "
                     "the forward is not autoregressive")
    if sep <= tol:
        fails.append(f"the true and a scrambled continuation score alike ({sep:.4g}/token ≤ {tol}) — the "
                     "forward is reading its targets, not predicting them")
    return {"causality": "VALID" if not fails else "VOID-causality: " + "; ".join(fails),
            "later_token_drift": drift, "true_minus_scrambled": sep, "tol": float(tol),
            "n_tokens": int(n_tokens)}


# ---------------------------------------------------------------- §6.2b H_format vs H_stage
ACC_INVARIANCE_TOL = 0.03


def tokenizer_gate(letter_token_counts, healthy_argmax_in_choices_frac: float,
                   p_letter_before: float, p_letter_after: float,
                   native_argmax_frac: float = None, native_p_after: float = None,
                   surface_agreement: float = None, acc_scored: float = None,
                   acc_native: float = None, agreement_when_right: float = None,
                   agreement_when_wrong: float = None) -> dict:
    """§6.2b (v3, D14): the four letter tokens must be single tokens and the healthy model's full-vocab
    argmax must be one of the four letters on ≥ 95 % of items (S/F/H VALID). Separately, the AFTER
    model's mean probability mass on the four letters must be ≥ 0.5 for a letter histogram to mean
    anything: below that the after-model is 'off-format' (abstaining from letters) and H_stage cannot be
    claimed on the letter readout — F5 becomes the primary letter-with-options readout."""
    _finite(healthy_argmax_in_choices_frac, p_letter_before, p_letter_after)
    single = all(c == 1 for c in letter_token_counts)
    # v4 (2026-09-03, measured): "on-format" is a property of the model-plus-surface, not of the scored
    # surface alone. In the blank-image shell the healthy model is 0 % on the space-prefixed surface and
    # 100 % on the no-space one, yet BOTH surfaces give the same accuracy (the four-way ranking is
    # surface-invariant). Voiding the S family on the scored surface alone would have voided a readout
    # that demonstrably measures the right thing. The gate therefore asks: are the letters single tokens;
    # is the healthy model on-format on SOME scored surface; and does reading the OTHER surface change
    # the measurement.
    #
    # v5 (2026-09-03, deviation D-B, journal + §13.38): the third clause was per-item surface AGREEMENT
    # ≥ 0.90. That is the wrong estimand and it voided demonstrably valid data. On ctrl-g1
    # (gemma-2b → gemma-1.1-2b-it) agreement was 0.74 / 0.77, yet the two surfaces returned the SAME
    # accuracy (0.4386 vs 0.4386 discovery; 0.4456 vs 0.4491 confirmation). The disagreement sits on the
    # items the model gets WRONG: near the floor the model is guessing, so two near-tied surfaces pick
    # different wrong options without the measurement moving. The 0.90 threshold had been calibrated on
    # the blank-image shell, where disagreement happened to be 4/57 and 1/57 — a sample of one regime.
    # The clause now tests what the rule is FOR: reading the alternative surface must not change the
    # measurement, i.e. |acc_scored − acc_native| ≤ ACC_INVARIANCE_TOL (0.03). Per-item agreement is kept
    # as a REPORTED diagnostic (overall, and split by whether the model was right), never as a gate.
    on_some = healthy_argmax_in_choices_frac >= 0.95 or (native_argmax_frac is not None and native_argmax_frac >= 0.95)
    have_acc = acc_scored is not None and acc_native is not None
    if have_acc:
        _finite(acc_scored, acc_native)
        acc_delta = abs(acc_scored - acc_native)
    else:
        acc_delta = None                       # the pair was not banked (pre-v5 artifact): clause N/A
    acc_ok = acc_delta is None or acc_delta <= ACC_INVARIANCE_TOL
    ok = single and on_some and acc_ok
    why = [] if ok else ([] if single else ["letters are not single tokens"]) + (
        [] if on_some else ["the healthy model is off-format on every scored surface"]) + (
        [] if acc_ok else [f"the measurement is surface-dependent (|Δacc| = {acc_delta:.3f} > {ACC_INVARIANCE_TOL})"])
    after_mass = max(p_letter_after, native_p_after) if native_p_after is not None else p_letter_after
    # UNCHECKABLE CLAUSE (2026-09-03 adversarial audit, wiring repair). With no alt-surface accuracy
    # banked the surface-invariance clause is N/A and `acc_ok` is True by default, so the clause CANNOT
    # FAIL on such an artifact and the gate returned a bare "VALID" that is indistinguishable from one
    # where all three clauses were tested. The DECISION is unchanged — voiding a pre-v5 artifact would
    # be a rule change, escalated — but the stamp now says which clauses actually ran.
    surface_clause = ("checked" if acc_delta is not None else
                      "NOT CHECKED (no alt-surface accuracy banked: `acc_scored`/`acc_native` absent, "
                      "so the surface-invariance clause cannot fail on this artifact)")
    stamp = "VALID" if ok else "VOID-tokenizer (" + "; ".join(why) + ")"
    if acc_delta is None:
        stamp += " [surface-invariance clause NOT CHECKED: no alt-surface accuracy banked]"
    return {"S_family": stamp, "surface_invariance_clause": surface_clause,
            "clauses_checked": {"single_tokens": True, "on_format_surface": True,
                                "surface_invariance": acc_delta is not None},
            "after_on_format": bool(after_mass >= 0.5),
            "p_letter_before": p_letter_before, "p_letter_after": p_letter_after,
            "native_p_after": native_p_after,
            "acc_scored": acc_scored, "acc_native": acc_native, "acc_delta": acc_delta,
            "acc_invariance_tol": ACC_INVARIANCE_TOL,
            # diagnostics only — never gates
            "surface_agreement": surface_agreement,
            "surface_agreement_when_right": agreement_when_right,
            "surface_agreement_when_wrong": agreement_when_wrong,
            "on_format_surface": ("scored" if healthy_argmax_in_choices_frac >= 0.95 else
                                  ("native" if (native_argmax_frac or 0) >= 0.95 else "none"))}


def sk_repairable(S_lost_5shot: float, S_lost_0shot: float) -> bool:
    """§1.2 S-k rule: in-context repairable ⟺ S-lost(5-shot) ≤ S-lost(0-shot) − 0.3."""
    _finite(S_lost_5shot, S_lost_0shot)
    return bool(S_lost_5shot <= S_lost_0shot - 0.3)


def format_verdict(S_lost_formats: dict, entropy_after_bits: float, entropy_before_bits: float,
                   S_lost_primary: float, after_on_format: bool = True, repairable: bool = False) -> dict:
    """S_lost_formats: {F1, F2, F3, F4: margin lost} (F3 by balanced accuracy, floor 0.5). F5 and S-k are
    NOT members of this family (F5 is the bridge to K; S-k enters as `repairable`).
    `S_lost_primary` is the PRIMARY letter readout's own margin lost — the loss both hypotheses exist to
    explain — and is required, never defaulted.

    DOMAIN (deviation D-E, 2026-09-03). Both hypotheses are explanations OF a loss, so the rule only
    applies where there is one. If `S_lost_primary < FORMAT_VANISH` the verdict is
    'N/A: no loss to explain (S-lost = X)' — for neither hypothesis and against neither. Without this
    clause H_format is VACUOUSLY TRUE on any benign row: nothing is lost under any format, every
    alternative is therefore below the vanish bar, and the rule reports 'the loss vanishes under another
    format' where no loss existed (it did exactly that on rows 3a and ctrl-g2, whose primary losses are
    0.04–0.09 and in places NEGATIVE — the tuned model is better). The minimum IS `FORMAT_VANISH` = 0.3,
    the same number the rule already uses as its 'vanishes' bar: no threshold was added or changed by
    this clause, only the rule's domain of applicability was stated. `entropy_ratio` is reported on this
    branch too — the ratio accumulates on every row, whether or not a verdict is reachable.

    Precedence (D29): H_format ⟺ ANY of F1–F4 has S_lost < 0.3 — it wins whenever it holds.
    Else H_stage ⟺ S_lost ≥ 0.6 under ≥ 2 formats ∧ entropy(after) ≤ 0.5 × entropy(before) ∧ the after
    model is on-format (D14) ∧ NOT in-context repairable (S-k rule). ≥ 2 formats at ≥ 0.6 without the
    collapse → H_stage-PARTIAL. Off-format after-model → 'off-format (abstention)'. Repairable → 'H_stage
    KILLED (in-context repairable)'. Anything else INDETERMINATE."""
    assert S_lost_formats, "need ≥1 alternative format"
    for v in S_lost_formats.values():
        _finite(v)
    _finite(entropy_after_bits, entropy_before_bits, S_lost_primary)
    vanishes = [k for k, v in S_lost_formats.items() if v < FORMAT_VANISH]
    persists = [k for k, v in S_lost_formats.items() if v >= 0.6]
    ratio = entropy_after_bits / entropy_before_bits if entropy_before_bits > 0 else np.inf
    if S_lost_primary < FORMAT_VANISH:
        return {"verdict": f"N/A: no loss to explain (S-lost = {S_lost_primary:.3f} < {FORMAT_VANISH})",
                "S_lost_primary": float(S_lost_primary), "vanishes": vanishes, "persists": persists,
                "entropy_ratio": float(ratio)}
    if vanishes:
        v = "H_format (loss vanishes under: " + ",".join(vanishes) + ")"
    elif not after_on_format:
        v = "off-format (after-model abstains from letters; H_stage not claimable on S — F5 is primary)"
    elif len(persists) >= 2 and repairable:
        v = "H_stage KILLED (in-context repairable by 5-shot)"
    elif len(persists) >= 2 and ratio <= 0.5:
        v = "H_stage"
    elif len(persists) >= 2:
        v = "H_stage-PARTIAL (format-invariant loss, histogram not collapsed)"
    else:
        v = "H_format/H_stage " + INDET
    return {"verdict": v, "vanishes": vanishes, "persists": persists, "entropy_ratio": float(ratio),
            "S_lost_primary": float(S_lost_primary)}


def letter_entropy(pred_letters, n_choices: int = 4) -> float:
    """Shannon entropy (bits) of the argmax-letter histogram; report beside the answer key's entropy.

    DEFECT (2026-09-03 adversarial audit, wiring repair). An EMPTY prediction list returned -0.0 bits:
    `bincount` gave four zeros, `p / p.sum()` divided 0 by 0, the `p > 0` filter emptied the vector and
    the sum of nothing is -0.0. `format_verdict` reads that as `entropy_ratio` 0.0 ≤ 0.5 — a TOTAL
    entropy collapse, the H_stage conjunct's strongest possible satisfaction — so a histogram that
    produced no predictions at all was the best evidence this program could manufacture for its own
    hypothesis. A dead histogram now raises; VOID is stamped upstream."""
    a = np.asarray(pred_letters, int)
    assert a.size >= 1, ("empty prediction list: a dead letter histogram has no entropy (it used to "
                         "return -0.0 bits, which format_verdict reads as a total entropy collapse)")
    p = np.bincount(a, minlength=n_choices).astype(float)
    assert p.sum() > 0, f"letter histogram carries no mass: {p}"
    p = p / p.sum(); p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def balanced_accuracy(tp: int, fn: int, tn: int, fp: int) -> float:
    """F3 true/false: four prompts per item (1 true, 3 false); balanced accuracy; floor 0.5."""
    return float(0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)))


def u_valid(healthy_U: float, healthy_shortcut: float, floor: float = 0.25) -> dict:
    """§1.2 U (D16): VALID ⟺ healthy ≥ floor + 0.15 on the two-hop set AND ≤ floor + 0.10 on the
    first-hop-removed shortcut set (otherwise the second hop alone answers the item)."""
    _finite(healthy_U, healthy_shortcut)
    ok = healthy_U >= floor + MIN_HEALTHY_MARGIN and healthy_shortcut <= floor + 0.10
    return {"U": "VALID" if ok else "VOID-U", "healthy_U": healthy_U, "healthy_shortcut": healthy_shortcut}


# ---------------------------------------------------------------- §6.4 grafts
def graft_identity_gate(selfgraft_item_match_frac: float, max_abs_dlogit: float,
                        donorhead_item_match_frac: float, donorhead_max_abs_dlogit: float,
                        lastlayer_R_recipient_head: float) -> dict:
    """§6.4 (v3, D2). Two parts, both required per pair and per resolver:
    (a) self-graft at 3 layers: 100 % per-item argmax match AND max|Δlogit| ≤ 1e-2 (bf16, batch 4);
    (b) WIRING test: the donor's residual captured at the LAST decoder layer, passed through the DONOR's
        own final norm + head, must reproduce the donor's per-item logits (argmax match ≥ 99 %,
        max|Δlogit| ≤ 1e-2). A wrong hook point passes (a) trivially and fails (b).
    The prefix graft at the last layer read through the RECIPIENT's norm + head is a RESULT, not a gate:
    reported as R_G(last); < 0.9 is labelled 'head-stage damage' (the recipient's head cannot read
    healthy residuals — real on rows whose head changed: 1b, 3b, 3c-Act, 4)."""
    _finite(selfgraft_item_match_frac, max_abs_dlogit, donorhead_item_match_frac, donorhead_max_abs_dlogit,
            lastlayer_R_recipient_head)
    a = selfgraft_item_match_frac >= 1.0 - 1e-12 and max_abs_dlogit <= 1e-2
    b = donorhead_item_match_frac >= 0.99 and donorhead_max_abs_dlogit <= 1e-2
    return {"identity_ok": bool(a), "wiring_ok": bool(b), "gate": "VALID" if (a and b) else "VOID-wiring",
            "R_G_last": lastlayer_R_recipient_head,
            "head_stage": "head-stage damage" if lastlayer_R_recipient_head < 0.9 else "head reads healthy residuals"}


def null_spread(null_Rs) -> float:
    """DEFECT (2026-09-03 adversarial audit, wiring repair). One non-finite mismatched-item cut made
    `np.abs(a).max()` NaN, `threshold_T` then returned NaN (`max(nan, 0.10)` keeps the first argument),
    and `reader_verdict` compared every R against NaN: `max_R <= nan` False, `early_quiet` False,
    `n_above > n/3` False — so a graft sweep whose NULL DID NOT RUN came back
    'H_reader INDETERMINATE', an inconclusive RESULT reported from an instrument with no calibration.
    A dead null must VOID; it now raises here and the caller stamps VOID rather than scoring."""
    a = np.asarray(list(null_Rs), float); assert a.size >= 5, "need ≥5 mismatched-item cuts"
    assert np.isfinite(a).all(), (
        f"non-finite mismatched-item cut(s) {[i for i, v in enumerate(a) if not np.isfinite(v)]} of "
        f"{a.size}: the null did not run, so there is no threshold T — a dead null must VOID, never "
        "produce an INDETERMINATE verdict")
    return float(np.abs(a).max())


def threshold_T(spread: float) -> float:
    _finite(spread)
    return float(max(2.0 * spread, 0.10))


def graft_band(layers, R_G, T: float, hi: float = 0.6, exclude_terminal_control: bool = True) -> dict:
    """L_hi = first sweep layer with R_G ≥ hi that is 'sustained': of the next 3 sweep layers, ≥ 2 are ≥ hi;
    at the tail (fewer than 3 sweep layers left) ALL remaining must be ≥ hi. L_lo = last sweep layer
    before L_hi with R_G ≤ T; band = L_hi − L_lo in layer units.

    DEFECT (2026-09-03 adversarial audit). `matrix_recovery_core.sweep_layers` ALWAYS appends
    `n_layers - 1`, and `matrix_graft` sets that last sweep point to R_G(last) — the full-prefix graft,
    which is the WIRING CONTROL and equals 1.0 by construction whenever the recipient's head is intact
    (banked: 5c_G has `R_G_last_recipient_head` = 1.0). At the tail `need = len(nxt)` is 0 at the last
    index, so L_hi was set there UNCONDITIONALLY. On the real threshold (T_G = 0.10345, n_layers = 34) a
    sweep where NOTHING transfers — 17 zeros plus the mandatory terminal 1.0 — returned H_reader, band 1.
    H_lost was therefore unreachable on any completed sweep and a null result reported the causal
    column's headline hypothesis. The terminal point is a control, NOT a data point: it is EXCLUDED from
    band-finding and from `max_R` by default, and `R_G(last)` is banked and reported on its own as the
    head-stage diagnostic (`graft_identity_gate`), which is its real job. Pass
    `exclude_terminal_control=False` only to score a curve that is known not to end at n_layers − 1."""
    L = list(layers); R = list(R_G)
    assert L == sorted(L) and len(L) == len(R) >= 4
    for r in R:
        _finite(r)
    terminal = None
    if exclude_terminal_control:
        terminal = {"layer": L[-1], "R_G": R[-1]}
        L, R = L[:-1], R[:-1]
    l_hi = None
    for i in range(len(L)):
        if R[i] >= hi:
            nxt = R[i + 1:i + 4]
            need = 2 if len(nxt) >= 3 else len(nxt)
            if sum(1 for r in nxt if r >= hi) >= need:
                l_hi = L[i]; break
    l_lo = None
    if l_hi is not None:
        below = [l for l, r in zip(L, R) if l < l_hi and r <= T]
        l_lo = max(below) if below else None
    return {"L_lo": l_lo, "L_hi": l_hi,
            "band": (l_hi - l_lo) if (l_hi is not None and l_lo is not None) else None, "max_R": float(max(R)),
            "scored_layers": L, "excluded_terminal_control": terminal}


def layers_covered_above(layers, R_G, T: float, n_layers: int) -> int:
    """D3: count in LAYER units — each sweep point covers the layers from itself up to the next sweep
    point (the last point covers up to n_layers)."""
    L = list(layers); cov = 0
    for i, (l, r) in enumerate(zip(L, R_G)):
        nxt = L[i + 1] if i + 1 < len(L) else n_layers
        if r > T:
            cov += nxt - l
    return int(cov)


def reader_verdict(layers, R_G, T: float, n_layers: int, exclude_terminal_control: bool = True) -> dict:
    """H_reader ⟺ L_hi exists ∧ band ≤ round(n_layers/6) ∧ every sweep layer in the first third ≤ T.
    H_lost ⟺ max R_G ≤ T. H_distributed ⟺ > n_layers/3 LAYERS (stride-weighted, D3) above T with no
    band ≤ n/6. Otherwise INDETERMINATE.

    The terminal sweep point is the full-prefix WIRING CONTROL (see `graft_band`) and is scored by
    NOTHING here: the trim happens once, at the top, so band, `max_R` and the stride-weighted
    `layers_above_T` all read the same curve. Dropping it costs no layer coverage — the preceding real
    sweep point's stride then runs to `n_layers`, which is the correct stride-weighted accounting.
    `R_G(last)` is reported beside the verdict as `R_G_terminal_control`, never inside it."""
    g = graft_band(layers, R_G, T, exclude_terminal_control=exclude_terminal_control)
    L, R = list(layers), list(R_G)
    if exclude_terminal_control:
        L, R = L[:-1], R[:-1]
    early = [r for l, r in zip(L, R) if l < n_layers / 3]
    early_quiet = bool(early) and all(r <= T for r in early)
    n_above = layers_covered_above(L, R, T, n_layers)
    if g["max_R"] <= T:
        v = "H_lost"
    elif g["L_hi"] is not None and g["band"] is not None and g["band"] <= round(n_layers / 6) and early_quiet:
        v = "H_reader"
    elif n_above > n_layers / 3 and (g["band"] is None or g["band"] > round(n_layers / 6)):
        v = "H_distributed"
    else:
        v = "H_reader " + INDET
    out = {"verdict": v, **g, "early_quiet": early_quiet, "layers_above_T": n_above, "T": T}
    t = g.get("excluded_terminal_control")
    if t is not None:
        out["R_G_terminal_control"] = t     # the head-stage diagnostic: REPORTED, never scored
    return out


def overlap_precision(items_graft, items_healthy, acc_healthy: float) -> dict:
    """§6.4 (v3, D10): chance precision under independence is acc_healthy, so 'restores the same items'
    is allowed iff |right_graft ∩ right_healthy| / |right_graft| ≥ acc_healthy + 0.15."""
    _finite(acc_healthy)
    g = set(np.flatnonzero(np.asarray(items_graft, int))); h = set(np.flatnonzero(np.asarray(items_healthy, int)))
    prec = float(len(g & h) / len(g)) if g else 0.0
    thr = acc_healthy + 0.15
    return {"precision": prec, "threshold": thr, "restores_same_items": bool(prec >= thr)}


def transplant_wiring_gate(R_prefix_L: float, R_G_L: float, tol: float = 0.05) -> dict:
    """§6.4 W (D17): the weight prefix control 0..L must reproduce the activation prefix graft at L to
    |Δ| ≤ tol (C5 saw exact equality); otherwise the transplant is VOID-wiring and a low R(band) is
    NOT a kill."""
    _finite(R_prefix_L, R_G_L)
    ok = abs(R_prefix_L - R_G_L) <= tol
    return {"gate": "VALID" if ok else "VOID-wiring", "delta": float(R_prefix_L - R_G_L)}


def weight_band_verdict(R_band: float, R_band_tail: float) -> dict:
    """H_weight-band ⟺ R(band) ≥ 0.6; band-plus-tail ⟺ R(band) < 0.6 ∧ R(band+tail) ≥ 0.6 (the band's
    weights need the co-adapted late layers too); KILLED ⟺ both < 0.3; otherwise INDETERMINATE."""
    _finite(R_band, R_band_tail)
    if R_band >= 0.6:
        v = "H_weight-band"
    elif R_band_tail >= 0.6:
        v = "band-plus-tail"
    elif R_band < 0.3 and R_band_tail < 0.3:
        v = "H_weight-band KILLED"
    else:
        v = "H_weight-band " + INDET
    return {"verdict": v, "R_band": R_band, "R_band_tail": R_band_tail}


# ---------------------------------------------------------------- §6.6 recovery
def n_half(doses, recovered, target: float = 0.5):
    """Smallest dose at which the recovered fraction reaches target, linear in log2(dose) between
    the bracketing measured doses. None if never reached. Doses in ITEMS (one number per row; the
    recipe's epoch count is fixed, so items × epochs is proportional)."""
    d = list(doses); r = list(recovered)
    assert d == sorted(d) and len(d) == len(r) >= 2 and all(x > 0 for x in d)
    for i in range(len(d)):
        if r[i] >= target:
            if i == 0:
                return float(d[0])
            x0, x1 = np.log2(d[i - 1]), np.log2(d[i]); y0, y1 = r[i - 1], r[i]
            t = (target - y0) / (y1 - y0) if y1 != y0 else 1.0
            return float(2 ** (x0 + t * (x1 - x0)))
    return None


def recovery_void(final_answer_ce: float, scratch_final_answer_ce: float, chance_ce: float = LN4,
                  slack: float = 0.05) -> dict:
    """§6.6: 'dose too weak' VOID only if BOTH the run and the fair-scratch run under the identical
    recipe stay at the chance floor of the answer-token CE. A run stuck at ln 4 while scratch moves is a
    RESULT (the substrate did not learn), not a void.

    DEFECT (2026-09-03 adversarial audit, wiring repair). NaN cross-entropy on BOTH arms made both
    comparisons False and returned `void: False` — two training runs that produced no loss at all read
    as 'the dose was fine'. The VOID that exists to catch a dead run was hardest to fire when the run
    was deadest. Non-finite CE now raises; the caller stamps INPUTS-MISSING."""
    _finite(final_answer_ce, scratch_final_answer_ce, chance_ce, slack)
    stuck = final_answer_ce >= chance_ce - slack
    scratch_stuck = scratch_final_answer_ce >= chance_ce - slack
    return {"void": bool(stuck and scratch_stuck), "stuck": bool(stuck), "scratch_stuck": bool(scratch_stuck)}


def permuted_label_void(S_rec_permuted: float, tol: float = 0.2) -> dict:
    """§6.13 recovery shuffle control: training on the same items with permuted answer letters (the H-2
    recipe) at 4k must NOT restore the letter: S_rec ≤ tol, else the recovery column is VOID-format."""
    _finite(S_rec_permuted)
    return {"REC": "VALID" if S_rec_permuted <= tol else "VOID-format (any label training restores the letter)",
            "S_rec_permuted": S_rec_permuted}


def k_moved(K_at_nstar: float, K_after: float, K_before: float, floor: float = 0.25):
    """D11: K_moved = |K(n*) − K_after| / (K_before − floor) at n* = the first dose with S_rec ≥ 0.8.
    None when the healthy K margin is below MIN_HEALTHY_MARGIN (then H_repair is scored without it, stated)."""
    mb = K_before - floor
    if mb < MIN_HEALTHY_MARGIN:
        return None
    return float(abs(K_at_nstar - K_after) / mb)


def recipe_curve(doses, S_rec, O_lost) -> dict:
    """Per recipe (D5): n_star = first dose with S_rec ≥ 0.8 (None if never); O_lost_at_S08 at that dose;
    corner_reached ⟺ ∃ dose: S_rec ≥ 0.8 ∧ O_lost ≤ 0.2; S_rec_at_max."""
    d = list(doses); s = list(S_rec); o = list(O_lost)
    assert len(d) == len(s) == len(o) >= 1 and d == sorted(d)
    for v in s + o:
        _finite(v)
    n_star = next((dd for dd, ss in zip(d, s) if ss >= 0.8), None)
    o_at = next((oo for dd, ss, oo in zip(d, s, o) if ss >= 0.8), None)
    corner = any(ss >= 0.8 and oo <= 0.2 for ss, oo in zip(s, o))
    return {"n_star": n_star, "O_lost_at_S08": o_at, "corner_reached": bool(corner), "S_rec_at_max": float(s[-1]),
            "n_half_S": n_half(d, s)}


def aggregate_recipes(curves: dict) -> dict:
    """D5: curves = {recipe: recipe_curve(...)}. `all_repurchase` ⟺ every recipe that reaches S_rec 0.8
    has O_lost_at_S08 ≥ 0.5 AND at least one reaches it; `any_corner` ⟺ some recipe reached the corner;
    `n_reaching` counts recipes with n_star.

    DEGENERATE QUANTIFIER (2026-09-03 adversarial audit — REPORTED, not gated). `all_rep` is an `all()`,
    so over a SINGLE reaching recipe it says 'under EVERY recipe' on the evidence of one ladder. The
    decision is untouched here (moving it would be a rule change, escalated to the lead), but the
    quantifier's width is now on the record: `n_quantified` is the number of recipes the universal
    actually ranged over and `quantifier_degenerate` is True when that is < 2. `recipe_invariant`
    already asserts >= 2 recipes for a cross-recipe statement; this one does not, and a reader must be
    able to see that without re-deriving it."""
    assert PRIMARY_RECIPE in curves, f"primary recipe {PRIMARY_RECIPE} missing"
    reach = {k: c for k, c in curves.items() if c["n_star"] is not None}
    all_rep = bool(reach) and all(c["O_lost_at_S08"] >= 0.5 for c in reach.values())
    return {"all_repurchase": all_rep, "any_corner": any(c["corner_reached"] for c in curves.values()),
            "n_reaching": len(reach), "primary": curves[PRIMARY_RECIPE],
            "n_recipes": len(curves), "recipes": sorted(curves), "recipes_reaching": sorted(reach),
            "n_quantified": len(reach), "quantifier_degenerate": bool(len(reach) < 2),
            "quantifier_note": ("`all_repurchase` ranged over "
                                f"{len(reach)} recipe(s) {sorted(reach) or '[]'}"
                                + (" — a universal over fewer than two recipes says 'under EVERY recipe'"
                                   " on one ladder" if len(reach) < 2 else ""))}


def sub_contention_frac(per_layer_contention) -> dict:
    """D5: fraction of layers with SUB contention; 'contention' at the row level ⟺ ≥ half the layers.

    DEGENERATE AGGREGATION (2026-09-03 adversarial audit — REPORTED, not gated). Over ONE banked layer
    the fraction is 1.0, 'at least half the layers' is satisfied by that single layer, and that is the
    conjunct that upgrades H_repurchase to `H_repurchase + H_capacity` in `recovery_verdict`. The
    decision is untouched (moving it would be a rule change, escalated); `n_layers` and
    `aggregation_degenerate` now ride on the record so a one-layer 'half the layers' cannot be read as
    a sweep."""
    a = np.asarray(list(per_layer_contention), bool); assert a.size >= 1
    return {"frac": float(a.mean()), "contention": bool(a.mean() >= 0.5),
            "n_layers": int(a.size), "n_contended": int(a.sum()),
            "aggregation_degenerate": bool(a.size < 2),
            "aggregation_note": f"'>= half the layers' evaluated over {int(a.size)} banked layer(s)"
                                + (" — a single layer satisfies it by itself" if a.size < 2 else "")}


def recovery_verdict(primary: dict, K_moved, agg: dict, corner_reached_reh: bool, sub_contention: bool,
                     gb_unreadable) -> dict:
    """§6.6 (v3, D4/D5). `primary` = recipe_curve of the PRIMARY recipe (r512); `agg` = aggregate_recipes.
    gb_unreadable ∈ {True, False, None (GB not run)}.
    H_repair ⟺ n_half(S) ≤ 4000 ∧ (K_moved ≤ 0.15 or K_moved is None) ∧ O_lost_at_S08 ≤ 0.2 (primary).
    H_repurchase ⟺ O_lost_at_S08 ≥ 0.5 under every recipe that reaches S_rec 0.8 ∧ GB is not 'unreadable'
    (an unreadable-but-present skill is rearrangement, not consumption — D4).
    H_neglect ⟺ rehearsal reaches the corner (S_rec ≥ 0.8 ∧ O_lost ≤ 0.2) by 16k.
    H_capacity ⟺ no recipe and not rehearsal reaches the corner ∧ SUB contention in ≥ half the layers.
    0.2 < O_lost_at_S08 < 0.5 on the primary recipe with no corner and no contention → INDETERMINATE
    (the audit's D5 band, stated). Plain labels: repair = rearrangement; repurchase/capacity =
    consumption; neglect = neglect."""
    n_star, o_at = primary["n_star"], primary["O_lost_at_S08"]
    if K_moved is not None:
        _finite(K_moved)
    if corner_reached_reh:
        return {"verdict": "H_neglect (rehearsal reaches the corner)" + ("" if n_star else " — sequential recovery never reached S_rec 0.8"),
                "n_half_S": primary["n_half_S"], "K_moved": K_moved, "O_lost_at_S08": o_at}
    if n_star is None:
        base = f"corner not reached (S_rec at max dose = {primary['S_rec_at_max']:.2f})"
        if (not agg["any_corner"]) and sub_contention:
            return {"verdict": "H_capacity (" + base + ")", "n_half_S": None, "K_moved": K_moved, "O_lost_at_S08": None}
        return {"verdict": "recovery " + INDET + " — " + base, "n_half_S": None, "K_moved": K_moved, "O_lost_at_S08": None}
    repair = primary["n_half_S"] is not None and primary["n_half_S"] <= 4000 and (K_moved is None or K_moved <= 0.15) and o_at <= 0.2
    repurchase = agg["all_repurchase"] and (gb_unreadable is not True)
    if repair:
        v = "H_repair"
    elif repurchase and (not agg["any_corner"]) and sub_contention:
        v = "H_repurchase + H_capacity"
    elif repurchase:
        v = "H_repurchase"
    elif agg["all_repurchase"] and gb_unreadable is True:
        v = "rearrangement (O lost at S_rec 0.8 but graft-back reads it: unread, not consumed)"
    elif (not agg["any_corner"]) and sub_contention:
        v = "H_capacity (corner unreached under every recipe; contention)"
    else:
        v = "recovery " + INDET
    return {"verdict": v, "n_half_S": primary["n_half_S"], "K_moved": K_moved, "O_lost_at_S08": o_at}


def tradeoff_curve(doses, S_rec_vals, O_lost_vals):
    pts = [(float(s), float(o)) for s, o in zip(S_rec_vals, O_lost_vals)]
    dist = [float(np.hypot(1 - s, o)) for s, o in pts]
    return {"points": pts, "min_distance_to_corner": min(dist), "dose_at_min": int(doses[int(np.argmin(dist))])}


def graft_back_verdict(layers, O_restored, T_O: float) -> dict:
    """§6.6 GB (v3, D12): after recovery has cost O (O_lost ≥ 0.3), graft the pre-recovery model's
    activations into the recovered model per layer. T_O = threshold_T(null_spread of ≥ 5 mismatched-item
    cuts ON THE O READOUT). 'unreadable' ⟺ O restored ≥ 0.6 at some layer; 'overwritten' ⟺ never above
    T_O; otherwise INDETERMINATE."""
    for v in O_restored:
        _finite(v)
    _finite(T_O)
    m = float(max(O_restored))
    v = "GB: unreadable (skill still present, not read)" if m >= 0.6 else (
        "GB: overwritten" if m <= T_O else "GB " + INDET)
    return {"verdict": v, "max_O_restored": m, "layer_at_max": int(layers[int(np.argmax(O_restored))]),
            "unreadable": bool(m >= 0.6), "T_O": T_O}


# ---------------------------------------------------------------- §6.3 recipe invariance
def rho_status(nh_S, nh_K, K_lost: float):
    """D1 three-way status per recipe: 'K-never-lost' (K_lost < 0.15: ρ undefined by design),
    'K-never-recovered' (K lost but n_half(K) None — the C14 pattern), or ('finite', ρ = n_half(S)/n_half(K)).
    n_half(S) None with K lost and recovered → 'S-never-recovered'."""
    _finite(K_lost)
    if K_lost < 0.15:
        return {"status": "K-never-lost", "rho": None}
    if nh_K is None:
        return {"status": "K-never-recovered", "rho": None}
    if nh_S is None:
        return {"status": "S-never-recovered", "rho": None}
    return {"status": "finite", "rho": float(nh_S / nh_K)}


def recipe_invariant(status_by_recipe: dict, factor: float = 2.0) -> dict:
    """§6.3 (v3, D1): invariant ⟺ every recipe has the same status AND, when finite, the ρ's lie within a
    factor of `factor`. Inputs are rho_status() records."""
    vals = list(status_by_recipe.values()); assert len(vals) >= 2
    statuses = {v["status"] for v in vals}
    if len(statuses) > 1:
        return {"invariant": False, "reason": f"status differs across recipes: {sorted(statuses)}"}
    st = statuses.pop()
    if st != "finite":
        return {"invariant": True, "reason": f"all recipes: {st}"}
    rhos = [v["rho"] for v in vals]
    lo, hi = min(rhos), max(rhos)
    return {"invariant": bool(hi / lo <= factor), "ratio": float(hi / lo), "reason": "finite ρ compared"}


# ---------------------------------------------------------------- §6.7 probe
def probe_recovered(acc_after_probe: float, acc_before_probe: float, acc_shuffle: float,
                    acc_randominit_probe: float, ci_lo_after: float, chance: float = 0.25) -> dict:
    """P_rec = (after − chance)/(before − chance). VOID if shuffle-label acc − chance > 0.05 or the healthy
    probe margin < 0.15. 'linearly present' ⟺ P_rec ≥ 0.6 AND the CI-low of P_rec ≥ 0.3 (D30; the CI is
    the item bootstrap of the after-probe accuracy); P_rec < 0.3 → 'not linearly present'; between →
    INDETERMINATE. The random-init-model probe is the lower bound (reported).

    UNITS DEFECT + DEAD NULL (2026-09-03 adversarial audit — reported, not gated). `lower_bound` was
    `acc_randominit - chance`, an ABSOLUTE accuracy margin, while the quantity it is supposed to bound,
    `P_recovered`, is a FRACTION of the healthy margin. The two were never comparable, so nothing could
    read the bound even by eye, and no rule consumes it: a random-init model decoding as well as the
    healthy one still returned 'linearly present'. The bound is now also reported in P's own units
    (`P_lower_bound`), together with the two comparisons a reader needs — whether the null sits below
    the measurement at all, and whether P clears it. GATING on that is a rule change (the frozen rule
    says the random-init probe is REPORTED) and is escalated to the lead, not taken here."""
    _finite(acc_after_probe, acc_before_probe, acc_shuffle, acc_randominit_probe, ci_lo_after)
    void = (acc_shuffle - chance > 0.05) or (acc_before_probe - chance < MIN_HEALTHY_MARGIN)
    if void:
        return {"P_recovered": np.nan, "void": True, "verdict": "VOID-probe"}
    p = float((acc_after_probe - chance) / (acc_before_probe - chance))
    p_lo = float((ci_lo_after - chance) / (acc_before_probe - chance))
    p_null = float((acc_randominit_probe - chance) / (acc_before_probe - chance))
    if p >= 0.6 and p_lo >= 0.3:
        v = "linearly present"
    elif p < 0.3:
        v = "not linearly present"
    else:
        v = "probe " + INDET
    return {"P_recovered": p, "P_ci_lo": p_lo, "void": False, "lower_bound": float(acc_randominit_probe - chance),
            "P_lower_bound": p_null, "null_below_measurement": bool(p_null < p),
            "null_separation": float(p - p_null),
            "null_note": ("the random-init null is REPORTED, never gated (frozen rule). "
                          f"P_recovered {p:.3f} vs random-init lower bound {p_null:.3f} in the same "
                          "units" + ("" if p_null < p else " — THE NULL IS NOT BELOW THE MEASUREMENT")),
            "linearly_present": v == "linearly present", "verdict": v}


# ---------------------------------------------------------------- §6.8 construct + interference matrix
# NAMING ALIGNMENT 2026-09-03 (lead's ruling; NOT a change of prediction). The MEASURED vision task was
# keyed `"vqav2"`, but the readout the recovery artifacts bank is the candidate-likelihood column
# `vqav2_cand` (spec §1.2 v2.3, D4: exact match is reported beside and never scored;
# `matrix_behaviour.O_VERDICT_KEY == "vqav2_cand"`), so `im_sign_hits` would have found no observed cell
# under that name and silently scored n = 0 for it. The three measured keys are renamed to `vqav2_cand`.
# The TRAIN key stays `"vqav2"`: it names the training SOURCE, which is `matrix_recovery.IM_TRAIN_TASKS`'s
# `"vqav2"`, not a readout column — renaming it would break the alignment it is here to keep.
# EVERY PREDICTED SIGN IS EXACTLY AS FROZEN. 15 cells before, 15 cells after, same signs, same pairs; the
# only difference is the spelling of one measured-task name. This is not a post-hoc edit of a
# pre-registration and must never be read as one.
IM_PREDICTED_SIGNS = {  # frozen (train task → {measured task: sign}); '0' = CI ⊂ (−0.15, 0.15); '+' = ci_lo > 0; '-' = ci_hi < 0
    "mmlu_letter": {"mmlu_cloze": "0", "arith": "+", "vqav2_cand": "-", "refcoco": "-", "triviaqa_cloze": "0"},
    "vqav2": {"mmlu_letter": "-", "mmlu_cloze": "0", "refcoco": "+", "triviaqa_cloze": "0"},
    "triviaqa_cloze": {"mmlu_cloze": "+", "mmlu_letter": "0", "vqav2_cand": "0"},
    "arith": {"mmlu_letter": "+", "mmlu_cloze": "0", "vqav2_cand": "-"},
}  # 15 cells, frozen 2026-09-03 (v3: '+0' retired — D13; skill-heavy tasks predicted to co-move with S)


def im_sign_hits(observed: dict, predicted: dict = IM_PREDICTED_SIGNS, tol: float = 0.15, min_n: int = 10) -> dict:
    """observed[train][measured] = (delta_point, ci_lo, ci_hi), or None when the measured task is
    VOID-low-margin on the damaged model (already at floor: a '-' is unobservable by construction —
    D13).

    SIGN CONVENTION (unambiguous, 2026-09-03): Δ is an ACCURACY CHANGE and POSITIVE MEANS IMPROVED.
    Precisely, Δ = (acc_after_IM_training − acc_damaged) / (acc_damaged − floor): the accuracy change
    the training task caused on the measured task, as a fraction of the damaged model's above-floor
    margin. `scripts/matrix_verdicts.IM_DELTA` is the producer of exactly this quantity (it negates
    `matrix_recovery_core.lost_fraction` and banks the unnegated record beside every cell).
    This docstring used to say "in margin-lost units", which is the SAME NUMBER WITH THE OPPOSITE
    SIGN, and that contradiction is what made the convention ambiguous. The frozen predictions
    disambiguate it and are the reason the reading above is the right one: `mmlu_letter → vqav2_cand`
    is predicted '-' because training on the letter task should HURT the vision skill, and
    `arith → mmlu_letter` / `triviaqa_cloze → mmlu_cloze` are '+' because skill and knowledge should
    transfer. Under margin-lost units those three signs would each be inverted. D13 says the same
    thing independently: a measured task already at floor cannot show '-', which is only true if '-'
    means "got worse". NOT A CHANGE OF PREDICTION — every sign in IM_PREDICTED_SIGNS is exactly as
    frozen (15 cells: 4 '+', 4 '-', 7 '0'); only this comment changed.

    Void cells are excluded and n counted after exclusion. Hits: '+' → ci_lo > 0;
    '-' → ci_hi < 0; '0' → (ci_lo, ci_hi) ⊂ (−tol, tol). Separates ⟺ n ≥ min_n and ≥ 70 % hits.
    IM has NO shuffle control (stated; §6.13)."""
    n = hits = excluded = 0
    for tr, row in predicted.items():
        for me, sign in row.items():
            if tr not in observed or me not in observed[tr]:
                continue
            cell = observed[tr][me]
            if cell is None:
                excluded += 1; continue
            pt, lo, hi = cell; _finite(pt, lo, hi); n += 1
            ok = {"+": lo > 0, "-": hi < 0, "0": (lo > -tol and hi < tol)}[sign]
            hits += int(ok)
    frac = hits / n if n else np.nan
    return {"n": n, "hits": hits, "excluded_void": excluded, "frac": float(frac),
            "separates": bool(n >= min_n and frac >= 0.7)}


def task_pattern(A_lost: float, S_lost: float, TQ_lost: float, K_lost: float) -> dict:
    """§6.8 task-pattern test (D30): where S_lost ≥ 0.6 the skill-heavy task must fall (A_lost ≥ 0.5) and
    where K_lost ≤ 0.5 the knowledge-heavy task must survive (TQ_lost ≤ 0.5). Only rows meeting the
    premise are scored; a row meeting neither premise returns 'not applicable'."""
    _finite(A_lost, S_lost, TQ_lost, K_lost)
    checks = {}
    if S_lost >= 0.6:
        checks["A_with_S"] = bool(A_lost >= 0.5)
    if K_lost <= 0.5:
        checks["TQ_with_K"] = bool(TQ_lost <= 0.5)
    return {"applicable": bool(checks), "pass": bool(checks) and all(checks.values()), "checks": checks}


def construct_verdict(n_stage_hops: int, probe_verdict: str, im_separates: bool, task_pattern_pass) -> dict:
    """§6.8 (v3, D30). H_construct ⟺ H_stage on ≥ 2 install hops ∧ probe 'linearly present' where K
    survived ∧ IM separates ∧ task-pattern passes on every applicable row. NOT supported ⟺ n_stage_hops
    ≤ 1 or probe 'not linearly present' or task-pattern fails. Otherwise INDETERMINATE (e.g. the probe
    in its band, or IM n < 10). task_pattern_pass ∈ {True, False, None (not applicable anywhere)}."""
    if n_stage_hops >= 2 and probe_verdict == "linearly present" and im_separates and task_pattern_pass is not False:
        v = "H_construct"
    elif n_stage_hops <= 1 or probe_verdict == "not linearly present" or task_pattern_pass is False:
        v = "H_construct NOT supported"
    else:
        v = "H_construct " + INDET
    return {"verdict": v, "n_stage_hops": n_stage_hops, "probe": probe_verdict, "IM": im_separates,
            "task_pattern": task_pattern_pass}


# ---------------------------------------------------------------- §6.9 scale (descriptive across families)
def scale_within_family(diff_point: float, diff_ci: tuple, direction: str = "+") -> dict:
    """The only confirmatory scale statement: a within-family, same-hop-kind pair (PaliGemma-2 3B vs
    10B). direction '+' predicts the larger model loses LESS (diff = small − large > 0)."""
    _finite(diff_point, *diff_ci)
    lo, hi = diff_ci
    if direction == "+":
        ok, killed = lo > 0, hi < 0
    else:
        ok, killed = hi < 0, lo > 0
    v = "H_scale(within-family)" if ok else ("H_scale(within-family) KILLED" if killed else "H_scale " + INDET)
    return {"verdict": v, "diff": diff_point, "ci": diff_ci}


def scale_descriptive(scales, values) -> dict:
    """Across families: Spearman rho REPORTED, never scored (families, hop kinds and MoE are confounded with scale)."""
    s = np.asarray(scales, float); v = np.asarray(values, float)
    if s.size < 3:
        return {"rho": np.nan, "n": int(s.size), "status": "descriptive (n<3)"}
    rs = np.argsort(np.argsort(s)); rv = np.argsort(np.argsort(v))
    return {"rho": float(np.corrcoef(rs, rv)[0, 1]), "n": int(s.size), "status": "descriptive only"}


# ---------------------------------------------------------------- §6.10 unlearning
def unlearn_verdict(S_W_after: float, K_or_P_fraction: float, nh_S_items, min_dose: int = 250,
                    floor: float = 0.25) -> dict:
    """'at floor' := S_W − floor ≤ 0.05. Doses in ITEMS (ladder {250, 500, 1000}, 2 epochs each — D31).
    H_unlearn-shallow ⟺ at floor ∧ (K_W or P_W) ≥ 0.5 of the parent's margin ∧ n_half(S_W) ≤ min_dose.
    DEEP ⟺ at floor ∧ K/P < 0.5 ∧ (n_half > 4 × min_dose or never). Otherwise INDETERMINATE.

    WIRING DEFECT (2026-09-03 adversarial audit). The floor was the literal 0.25 while the caller
    (`matrix_verdicts.unlearn_row_verdict`) reads the WMDP block's OWN banked floor and uses it for
    every retained fraction. Two floors for one readout is a wiring bug, not a threshold: the rule now
    takes the block's floor as a parameter and the caller passes it. The default is the same 0.25, so
    on every artifact whose banked floor is 0.25 — which is every one banked today — nothing moves."""
    _finite(S_W_after, K_or_P_fraction, floor)
    at_floor = S_W_after - floor <= 0.05
    shallow = at_floor and K_or_P_fraction >= 0.5 and nh_S_items is not None and nh_S_items <= min_dose
    deep = at_floor and K_or_P_fraction < 0.5 and (nh_S_items is None or nh_S_items > 4 * min_dose)
    v = "H_unlearn-shallow" if shallow else ("H_unlearn DEEP" if deep else "H_unlearn " + INDET)
    return {"verdict": v, "at_floor": at_floor, "K_or_P": K_or_P_fraction, "nh_S": nh_S_items,
            "floor": float(floor)}


def unlearn_locus(L_hi, edited_layers) -> str:
    """§6.10 / §9: trivial ⟺ L_hi ≤ max(edited) + 2; informative ⟺ L_hi > max(edited) + 2."""
    if L_hi is None:
        return "no band"
    return "trivial (restoration at the edited layers)" if L_hi <= max(edited_layers) + 2 else "informative (restoration later than the edited layers)"


# ---------------------------------------------------------------- §6.11 instruments
def validity_triple(fvu_before: float, fvu_after: float, plant_fired: bool, identity_floor_computed: bool,
                    fvu_max: float = 0.3) -> dict:
    _finite(fvu_before, fvu_after)
    ok = fvu_before < fvu_max and fvu_after < fvu_max and plant_fired and identity_floor_computed
    return {"valid": bool(ok), "stamp": "VALID" if ok else "VOID"}


def contention(overlap: float, overlap_unrelated: float, haar_p95: float,
               margin: float = CONTENTION_MARGIN) -> bool:
    """§6.6 SUB (D9) — THE contention decision, over the three statistics and nothing else:
    contention ⟺ overlap ≥ overlap_unrelated + margin ∧ overlap > haar_p95.

    Split out of `subspace_overlap` on 2026-09-03 because that function's inputs are RAW weight deltas,
    gigabytes per dose, which are never banked — so every consumer downstream of a banked artifact had
    no way to call the rule and `scripts/matrix_verdicts._sub` had restated the decision line by hand.
    `subspace_overlap` now calls this, `matrix_recovery_core.sub_statistics` calls this, and
    `matrix_verdicts` calls this: one definition, three call sites, no restatement anywhere."""
    _finite(overlap, overlap_unrelated, haar_p95)
    return bool(overlap >= overlap_unrelated + margin and overlap > haar_p95)


def subspace_overlap(dW_rec: np.ndarray, dW_inst: np.ndarray, k: int, dW_unrelated: np.ndarray,
                     seed: int = 1337, n_null: int = 200, margin: float = CONTENTION_MARGIN) -> dict:
    """§6.6 SUB (v3, D9). overlap = ‖P_k ΔW_rec‖_F² / ‖ΔW_rec‖_F² with P_k the projector onto the top-k left
    singular subspace of ΔW_inst. Two nulls: (i) the SAME statistic for an UNRELATED update on the same
    base (an adapter trained on another task; the binding null — two gradient updates on one base share
    the input-covariance directions, so Haar alone cannot fail); (ii) Haar-random k-subspaces (floor,
    reported). The decision itself is `contention` above — called, never repeated."""
    U, _, _ = np.linalg.svd(dW_inst, full_matrices=False)
    Uk = U[:, :k]
    def frac(dw):
        den = float(np.linalg.norm(dw) ** 2)
        return float(np.linalg.norm(Uk.T @ dw) ** 2 / den)
    ov = frac(dW_rec); ov_unrel = frac(dW_unrelated)
    rng = np.random.default_rng(seed); d = dW_inst.shape[0]
    nulls = []
    for _ in range(n_null):
        Q, _ = np.linalg.qr(rng.standard_normal((d, k)))
        nulls.append(float(np.linalg.norm(Q.T @ dW_rec) ** 2 / float(np.linalg.norm(dW_rec) ** 2)))
    p95 = float(np.percentile(nulls, 95))
    return {"overlap": ov, "overlap_unrelated": ov_unrel, "haar_p95": p95, "margin": margin,
            "contention": contention(ov, ov_unrel, p95, margin)}


# ---------------------------------------------------------------- preflight (§12.10, D19)
def preflight_match(per_item_new, per_item_ref, tol_acc: float = 0.01, min_agree: float = 0.99) -> dict:
    """The harness reproduces a banked artifact: ≥ 99 % per-item agreement and |Δacc| ≤ 0.01 (bf16 is not
    bit-stable across instance types/batches, so 'exact' is not a reachable bar)."""
    a = np.asarray(per_item_new, int); b = np.asarray(per_item_ref, int); assert a.size == b.size >= 20
    agree = float((a == b).mean()); dacc = float(abs(a.mean() - b.mean()))
    return {"agree": agree, "dacc": dacc, "pass": bool(agree >= min_agree and dacc <= tol_acc)}


# ---------------------------------------------------------------- dry-run gate
def _raises(fn, *a, **k):
    try:
        fn(*a, **k)
    except AssertionError:
        return True
    return False


def _dryrun():
    # THE GATE REGISTRY (2026-09-04). Every gate in this module declares, per clause, an input that must
    # make it fail, and `matrix_audit.run_gates` asserts it does. Coverage is re-derived from THIS FILE's
    # source with `ast`, so a gate added below without a registered failing input breaks this dry-run.
    from infra.pg import matrix_audit
    matrix_audit.run_gates("core", verbose=True)
    m = margin_lost(0.47, 0.281, 0.25); assert abs(m["lost"] - 0.859) < 0.01 and m["status"] == "ok"
    assert margin_lost(0.30, 0.28, 0.25)["status"] == "VOID-low-margin"
    m2 = margin_lost(0.47, 0.20, 0.25); assert m2["status"] == "below-floor" and m2["lost"] == 1.0
    # D8: NaN never scores
    assert _raises(split_verdict, np.nan, 0.3, 0.1) and _raises(format_verdict, {"F1": np.nan}, 1.0, 1.9, 0.8)
    assert _raises(weight_band_verdict, np.nan, 0.2) and _raises(graft_back_verdict, [1, 2], [np.nan, 0.1], 0.2)
    rng = np.random.default_rng(0)
    b = np.ones(300); a = (rng.random(300) < 0.3).astype(float); x = np.where(rng.random(300) < 0.5, 1.0, a)
    bs = bootstrap_fraction(b, a, x)
    assert bs["ratio_status"] == "reported" and bs["ratio_ci"][0] <= bs["ratio_point"] <= bs["ratio_ci"][1] and bs["p_wrong_side"] < 0.01
    a2 = b.copy(); a2[:3] = 0
    bs2 = bootstrap_fraction(b, a2, b)
    assert bs2["ratio_point"] is None and "N/A" in bs2["ratio_status"] and bs2["diff_ci"][0] >= 0
    assert bh_reject([0.001, 0.02, 0.04, 0.5]) == [True, True, False, False] and bh_reject([0.3, 0.6]) == [False, False]
    # split incl. indeterminate; K controls
    assert split_verdict(0.86, 0.38, 0.2)["verdict"] == "H_split"
    assert split_verdict(0.5, 0.45, 0.0)["verdict"] == "H_split KILLED"
    assert INDET in split_verdict(0.7, 0.5, 0.05)["verdict"]
    assert k_positive_control(0.9)["K_column"] == "VALID" and k_positive_control(0.2)["K_column"].startswith("VOID")
    assert k_shuffle_null(0.27)["K_null"] == "at floor" and k_shuffle_null(0.36)["K_null"] == "VOID-K-null"
    kp = k_shell_parity(0.44, 0.43, 0.47, 0.40); assert kp["K_shell"] == "VALID" and kp["S_shell"] == "VOID-shell"
    # ppl forward parity (§6.5, 2026-09-03): both branches, plus the boundary on each side
    assert ppl_forward_gate(42.789, 17.286)["ppl_forward"] == "VALID"            # measured framing effect, 2.48x
    br = ppl_forward_gate(18558.6, 7.966); assert br["ppl_forward"] == "VOID-ppl-forward" and abs(br["shell_vs_bare"] - 2329.5) < 1
    assert ppl_forward_gate(160.0, 8.0)["ppl_forward"] == "VALID"                # exactly 20x: just inside
    assert ppl_forward_gate(161.0, 8.0)["ppl_forward"] == "VOID-ppl-forward"     # 20.125x: just outside
    assert ppl_forward_gate(8.0, 160.0)["ppl_forward"] == "VALID"                # two-sided: 1/20 just inside
    assert ppl_forward_gate(8.0, 161.0)["ppl_forward"] == "VOID-ppl-forward"     # two-sided: just outside
    assert _raises(ppl_forward_gate, np.nan, 8.0) and _raises(ppl_forward_gate, 8.0, 0.0)
    # format / stage / tokenizer (D14, D29)
    tg = tokenizer_gate([1, 1, 1, 1], 0.99, 0.9, 0.8)
    assert tg["S_family"].startswith("VALID") and tg["after_on_format"]
    # …and with no alt-surface accuracy banked the third clause is UNCHECKABLE, so the stamp says so
    assert tg["clauses_checked"]["surface_invariance"] is False and "NOT CHECKED" in tg["S_family"], tg
    assert tokenizer_gate([1, 2, 1, 1], 0.99, 0.9, 0.8)["S_family"].startswith("VOID-tokenizer")
    assert not tokenizer_gate([1, 1, 1, 1], 0.99, 0.9, 0.2)["after_on_format"]
    # the measured shell case: off-format on the scored surface, on-format natively, accuracies agree → VALID
    shell = tokenizer_gate([1, 1, 1, 1], 0.0, 1.8e-07, 5.7e-05, native_argmax_frac=1.0, native_p_after=0.765,
                           surface_agreement=0.98, acc_scored=0.4737, acc_native=0.4737)
    assert shell["S_family"] == "VALID" and shell["after_on_format"] and shell["on_format_surface"] == "native"
    # off-format on BOTH surfaces → VOID
    assert tokenizer_gate([1, 1, 1, 1], 0.0, 1e-7, 1e-7, native_argmax_frac=0.1, native_p_after=0.05,
                          surface_agreement=0.99, acc_scored=0.25, acc_native=0.25)["S_family"].startswith("VOID-tokenizer")
    # v5 (D-B) BOTH branches of the surface clause, measured against ctrl-g1:
    #  (a) low per-item agreement but INVARIANT accuracy → must PASS (the v4 rule voided this)
    inv = tokenizer_gate([1, 1, 1, 1], 1.0, 0.9950, 0.9998, native_argmax_frac=0.0, native_p_after=6.0e-05,
                         surface_agreement=0.7368, acc_scored=0.4386, acc_native=0.4386,
                         agreement_when_right=0.98, agreement_when_wrong=0.53)
    assert inv["S_family"] == "VALID" and inv["acc_delta"] == 0.0 and inv["surface_agreement"] == 0.7368
    assert inv["surface_agreement_when_wrong"] == 0.53      # reported, never gating
    #  (b) genuinely surface-DEPENDENT accuracy → must still VOID, even with high per-item agreement
    dep = tokenizer_gate([1, 1, 1, 1], 1.0, 0.99, 0.99, native_argmax_frac=1.0, native_p_after=0.9,
                         surface_agreement=0.97, acc_scored=0.44, acc_native=0.36)
    assert dep["S_family"].startswith("VOID-tokenizer") and "surface-dependent" in dep["S_family"]
    assert abs(dep["acc_delta"] - 0.08) < 1e-9
    #  boundary: exactly at the tolerance passes, just past it voids
    assert tokenizer_gate([1, 1, 1, 1], 1.0, 0.9, 0.9, acc_scored=0.40, acc_native=0.43)["S_family"] == "VALID"
    assert tokenizer_gate([1, 1, 1, 1], 1.0, 0.9, 0.9, acc_scored=0.40, acc_native=0.4301
                          )["S_family"].startswith("VOID-tokenizer")
    #  a pre-v5 artifact banks no accuracy pair: the clause is N/A, the other clauses still rule.
    #  2026-09-03 audit: the DECISION is unchanged (still VALID — voiding it would be a rule change),
    #  but "VALID" alone was indistinguishable from a VALID where all three clauses ran, so the stamp
    #  now names the clause that could not fail. This is the whole repair: visibility, not a threshold.
    na = tokenizer_gate([1, 1, 1, 1], 1.0, 0.9, 0.9, surface_agreement=0.55)
    assert na["S_family"].startswith("VALID") and na["acc_delta"] is None
    assert "surface-invariance clause NOT CHECKED" in na["S_family"], na
    assert na["surface_invariance_clause"].startswith("NOT CHECKED") and not na["clauses_checked"]["surface_invariance"]
    #  and where the pair IS banked the stamp is the bare VALID it always was — no text drift
    assert tokenizer_gate([1, 1, 1, 1], 1.0, 0.9, 0.9, acc_scored=0.44, acc_native=0.44)["S_family"] == "VALID"
    assert sk_repairable(0.3, 0.8) and not sk_repairable(0.8, 0.9)
    assert format_verdict({"F1": 0.1, "F2": 0.8}, 0.4, 1.9, 0.8)["verdict"].startswith("H_format")
    assert format_verdict({"F1": 0.8, "F2": 0.9, "F3": 0.7}, 0.5, 1.9, 0.9)["verdict"] == "H_stage"
    assert format_verdict({"F1": 0.8, "F2": 0.9}, 1.8, 1.9, 0.9)["verdict"].startswith("H_stage-PARTIAL")
    assert format_verdict({"F1": 0.8, "F2": 0.9}, 0.5, 1.9, 0.9, after_on_format=False)["verdict"].startswith("off-format")
    assert format_verdict({"F1": 0.8, "F2": 0.9}, 0.5, 1.9, 0.9, repairable=True)["verdict"].startswith("H_stage KILLED")
    assert format_verdict({"F1": 0.1, "F2": 0.9}, 0.5, 1.9, 0.9, after_on_format=False)["verdict"].startswith("H_format")  # precedence
    assert INDET in format_verdict({"F1": 0.5, "F2": 0.4}, 1.8, 1.9, 0.5)["verdict"]
    # D-E (2026-09-03): the rule has a DOMAIN. With no primary loss to explain, neither hypothesis is
    # posed and the verdict is N/A — the branch H_format used to answer VACUOUSLY on every benign row.
    na = format_verdict({"F1": 0.05, "F2": 0.09}, 1.984, 1.997, 0.088)      # 3a discovery, as measured
    assert na["verdict"] == "N/A: no loss to explain (S-lost = 0.088 < 0.3)" and na["vanishes"] == ["F1", "F2"]
    assert abs(na["entropy_ratio"] - 1.984 / 1.997) < 1e-12                 # REPORTED even with no verdict
    # a NEGATIVE primary loss (the tuned model is BETTER) is the same case, not evidence for H_format
    assert format_verdict({"F1": -0.30, "F2": -0.41}, 1.990, 1.983, -0.278)["verdict"].startswith("N/A: no loss")
    # exactly AT the bar the rule applies (0.3 is not < 0.3), so the H_format branch is still reachable
    assert format_verdict({"F1": 0.1, "F2": 0.8}, 0.4, 1.9, FORMAT_VANISH)["verdict"].startswith("H_format")
    # the domain bar IS the vanish bar — one number, used twice, never two
    assert FORMAT_VANISH == 0.3 and _raises(format_verdict, {"F1": 0.5}, 1.0, 1.9, np.nan)
    assert abs(letter_entropy([0, 0, 0, 0])) < 1e-12 and abs(letter_entropy([0, 1, 2, 3]) - 2.0) < 1e-12
    assert abs(balanced_accuracy(10, 0, 0, 30) - 0.5) < 1e-12 and abs(balanced_accuracy(10, 0, 30, 0) - 1.0) < 1e-12
    assert u_valid(0.55, 0.30)["U"] == "VALID" and u_valid(0.55, 0.45)["U"] == "VOID-U" and u_valid(0.35, 0.30)["U"] == "VOID-U"
    # grafts: identity + wiring (D2) + head-stage result
    g = graft_identity_gate(1.0, 5e-3, 1.0, 5e-3, 0.95); assert g["gate"] == "VALID" and g["head_stage"].startswith("head reads")
    g2 = graft_identity_gate(1.0, 5e-3, 1.0, 5e-3, 0.2); assert g2["gate"] == "VALID" and g2["head_stage"] == "head-stage damage"  # NOT void
    assert graft_identity_gate(1.0, 5e-3, 0.5, 3.0, 0.95)["gate"] == "VOID-wiring"   # wrong hook: identity passes, wiring fails
    L = list(range(0, 26, 2)) + [25]
    T = threshold_T(null_spread([-.13, -.02, -.11, .02, .04])); assert abs(T - 0.26) < 1e-9
    R2 = [.0, .0, .0, .0, .0, .0, .0, .05, .2, .75, .85, .9, .95, 1.0]
    rv2 = reader_verdict(L, R2, 0.10, 26); assert rv2["verdict"] == "H_reader" and rv2["band"] == 4, rv2
    assert reader_verdict(L, [0.02] * 14, 0.10, 26)["verdict"] == "H_lost"
    Rd = [.05, .12, .2, .28, .35, .42, .5, .55, .6, .65, .7, .72, .75, .8]
    rd = reader_verdict(L, Rd, 0.10, 26); assert rd["verdict"] == "H_distributed" and rd["layers_above_T"] == 24, rd
    assert layers_covered_above([0, 8, 16], [0.5, 0.0, 0.5], 0.1, 24) == 16   # stride-weighted (D3)
    Rm = [.3, .05, .02, .02, .02, .02, .02, .05, .2, .75, .85, .9, .95, 1.0]
    assert INDET in reader_verdict(L, Rm, 0.10, 26)["verdict"]
    assert graft_band([0, 8, 16, 24], [0.0, 0.0, 0.7, 0.9], 0.1)["L_hi"] == 16
    # a high-but-UNSUSTAINED point is not a band — checked on the SCORED curve, so the sweep carries the
    # terminal control that the rule drops (spelt as 4 points, that test used to score the control)
    assert graft_band([0, 8, 16, 24, 25], [0.0, 0.0, 0.7, 0.2, 0.9], 0.1)["L_hi"] is None
    # RESIDUAL, stated not hidden (2026-09-03): `need = len(nxt)` is 0 at the LAST index, so the last
    # SCORED point still qualifies with no sustaining evidence. Excluding the control moves that vacuity
    # one point earlier; it does not remove it. It is no longer unfalsifiable — the last scored point is
    # a real graft, not a wiring identity worth 1.0 by construction — but it is a weak L_hi and it is
    # asserted here so a reader sees it. Removing it is a rule change, not an audit fix.
    assert graft_band([0, 8, 16, 24], [0.0, 0.0, 0.7, 0.2], 0.1)["L_hi"] == 16
    # the terminal sweep point is the full-prefix WIRING CONTROL: excluded from band-finding and max_R.
    # Spelt with the control scored (the pre-2026-09-03 behaviour) the same curve claims a band AT it.
    assert graft_band([0, 8, 16, 24], [0.0, 0.0, 0.2, 0.9], 0.1)["L_hi"] is None
    assert graft_band([0, 8, 16, 24], [0.0, 0.0, 0.2, 0.9], 0.1, exclude_terminal_control=False)["L_hi"] == 24
    gt = graft_band([0, 8, 16, 24], [0.0, 0.0, 0.2, 0.9], 0.1)
    assert gt["excluded_terminal_control"] == {"layer": 24, "R_G": 0.9} and gt["max_R"] == 0.2 and gt["scored_layers"] == [0, 8, 16]
    # THE REGRESSION (2026-09-03 adversarial audit), on the real threshold and depth of the banked 5c_G
    # graft: T_G = 0.10345, n_layers = 34, sweep = sweep_layers(34, 2) — 17 real points plus the
    # mandatory terminal n_layers-1. A sweep where NOTHING transfers must read H_lost; scored WITH the
    # control it read "H_reader, band 1", which made H_lost unreachable on any completed sweep.
    T_G, N_G = 0.10344827586206883, 34
    L_G = list(range(0, N_G, 2)) + [N_G - 1]
    assert len(L_G) == 18 and L_G[-1] == 33
    dead = [0.0] * 17 + [1.0]                                     # nothing transfers + the wiring control
    rvd = reader_verdict(L_G, dead, T_G, N_G)
    assert rvd["verdict"] == "H_lost" and rvd["max_R"] == 0.0 and rvd["band"] is None, rvd
    assert rvd["R_G_terminal_control"] == {"layer": 33, "R_G": 1.0} and rvd["layers_above_T"] == 0, rvd
    assert reader_verdict(L_G, dead, T_G, N_G, exclude_terminal_control=False)["verdict"] == "H_reader"   # the defect
    # …and a GENUINE late band on the same sweep must still read H_reader, band intact.
    live = [0.0] * 12 + [0.7, 0.8, 0.85, 0.9, 0.9] + [1.0]
    rvl = reader_verdict(L_G, live, T_G, N_G)
    assert rvl["verdict"] == "H_reader" and rvl["L_hi"] == 24 and rvl["L_lo"] == 22 and rvl["band"] == 2, rvl
    assert rvl["max_R"] == 0.9 and rvl["R_G_terminal_control"] == {"layer": 33, "R_G": 1.0}, rvl
    h = np.array([1, 1, 1, 0, 0, 0])
    assert not overlap_precision(np.array([1, 1, 0, 1, 0, 0]), h, 0.6)["restores_same_items"]
    assert overlap_precision(np.array([1, 1, 0, 0, 0, 0]), h, 0.5)["restores_same_items"]
    assert not overlap_precision(np.array([1, 1, 0, 0, 0, 0]), h, 0.9)["restores_same_items"]   # D10: chance = acc_healthy
    assert transplant_wiring_gate(0.204, 0.2037)["gate"] == "VALID" and transplant_wiring_gate(0.5, 0.2)["gate"] == "VOID-wiring"
    assert weight_band_verdict(0.02, 0.0)["verdict"] == "H_weight-band KILLED"
    assert weight_band_verdict(0.7, 0.8)["verdict"] == "H_weight-band"
    assert weight_band_verdict(0.4, 0.7)["verdict"] == "band-plus-tail" and INDET in weight_band_verdict(0.4, 0.4)["verdict"]
    # recovery (D1, D4, D5, D11, D12)
    nh = n_half([250, 1000, 4000, 16000], [0.05, 0.2, 0.6, 0.8]); assert 2800 < nh < 3000, nh
    assert n_half([250, 1000], [0.1, 0.2]) is None and n_half([250, 1000], [0.7, 0.9]) == 250.0
    assert recovery_void(LN4 - 0.01, LN4 - 0.01)["void"] and not recovery_void(LN4 - 0.01, 0.9)["void"]
    assert permuted_label_void(0.1)["REC"] == "VALID" and permuted_label_void(0.5)["REC"].startswith("VOID-format")
    assert abs(k_moved(0.40, 0.368, 0.442) - 0.167) < 0.01 and k_moved(0.3, 0.3, 0.35) is None
    D = [250, 1000, 4000, 16000]
    c_rep = recipe_curve(D, [0.1, 0.4, 0.85, 0.95], [0.0, 0.05, 0.1, 0.15])
    c_buy = recipe_curve(D, [0.1, 0.4, 0.85, 0.95], [0.1, 0.3, 0.6, 0.7])
    c_mid = recipe_curve(D, [0.1, 0.4, 0.85, 0.95], [0.1, 0.2, 0.35, 0.4])
    c_never = recipe_curve(D, [0.05, 0.1, 0.2, 0.5], [0.0, 0.0, 0.1, 0.2])
    assert c_rep["corner_reached"] and c_rep["n_star"] == 4000 and not c_buy["corner_reached"] and c_never["n_star"] is None
    agg_rep = aggregate_recipes({"r512": c_rep, "r64": c_rep, "full": c_rep})
    agg_buy = aggregate_recipes({"r512": c_buy, "r64": c_buy, "full": c_buy})
    agg_mid = aggregate_recipes({"r512": c_mid, "r64": c_mid, "full": c_mid})
    agg_never = aggregate_recipes({"r512": c_never, "r64": c_never, "full": c_never})
    assert agg_buy["all_repurchase"] and not agg_mid["all_repurchase"] and not agg_never["all_repurchase"]
    assert _raises(aggregate_recipes, {"r64": c_rep})
    assert recovery_verdict(c_rep, 0.05, agg_rep, False, False, None)["verdict"] == "H_repair"
    assert recovery_verdict(c_rep, None, agg_rep, False, False, None)["verdict"] == "H_repair"           # K_moved undefined → scored without it
    assert recovery_verdict(c_buy, 0.05, agg_buy, False, False, None)["verdict"] == "H_repurchase"
    assert recovery_verdict(c_buy, 0.05, agg_buy, False, False, False)["verdict"] == "H_repurchase"
    assert recovery_verdict(c_buy, 0.05, agg_buy, False, True, False)["verdict"] == "H_repurchase + H_capacity"
    assert recovery_verdict(c_buy, 0.05, agg_buy, False, False, True)["verdict"].startswith("rearrangement")   # D4: GB reads it back
    assert recovery_verdict(c_buy, 0.05, agg_buy, True, False, None)["verdict"].startswith("H_neglect")
    assert INDET in recovery_verdict(c_mid, 0.05, agg_mid, False, False, None)["verdict"]                   # D5 band
    assert recovery_verdict(c_mid, 0.05, agg_mid, False, True, None)["verdict"].startswith("H_capacity")   # D5: unreached corner + contention
    assert recovery_verdict(c_never, 0.05, agg_never, False, True, None)["verdict"].startswith("H_capacity")
    assert INDET in recovery_verdict(c_never, 0.05, agg_never, False, False, None)["verdict"]
    assert recovery_verdict(c_never, 0.05, agg_never, True, False, None)["verdict"].startswith("H_neglect")
    sc = sub_contention_frac([True, True, False, True]); assert sc["contention"] and not sub_contention_frac([True, False, False])["contention"]
    assert tradeoff_curve([1000, 4000, 16000], [0.1, 0.5, 0.9], [0.0, 0.2, 0.7])["dose_at_min"] == 4000
    assert graft_back_verdict([4, 8, 12], [0.1, 0.7, 0.3], 0.2)["unreadable"]
    assert graft_back_verdict([4, 8, 12], [0.1, 0.15, 0.05], 0.2)["verdict"] == "GB: overwritten"
    assert INDET in graft_back_verdict([4, 8, 12], [0.1, 0.4, 0.3], 0.2)["verdict"]
    # recipe invariance three-way (D1) incl. the C14 pattern (K lost, never recovered)
    s_c14 = rho_status(2900.0, None, 0.38); assert s_c14["status"] == "K-never-recovered"
    assert rho_status(2900.0, None, 0.05)["status"] == "K-never-lost" and rho_status(None, 900.0, 0.5)["status"] == "S-never-recovered"
    assert recipe_invariant({"r512": s_c14, "r64": s_c14, "full": s_c14})["invariant"]
    assert not recipe_invariant({"r512": s_c14, "full": rho_status(2000.0, 1000.0, 0.4)})["invariant"]
    assert recipe_invariant({"r512": rho_status(2000.0, 1000.0, 0.4), "full": rho_status(3000.0, 1000.0, 0.4)})["invariant"]
    assert not recipe_invariant({"r512": rho_status(2000.0, 1000.0, 0.4), "full": rho_status(9000.0, 1000.0, 0.4)})["invariant"]
    # probe (D30) — present / not / band / void
    assert probe_recovered(0.55, 0.65, 0.27, 0.26, 0.50)["verdict"] == "linearly present"
    assert INDET in probe_recovered(0.55, 0.65, 0.27, 0.26, 0.33)["verdict"]
    assert probe_recovered(0.33, 0.65, 0.27, 0.26, 0.28)["verdict"] == "not linearly present"
    assert probe_recovered(0.55, 0.65, 0.40, 0.26, 0.5)["void"] and probe_recovered(0.30, 0.35, 0.26, 0.26, 0.28)["void"]
    # IM (D13) + task pattern + construct (D30)
    # measured keys are the BANKED column names (`vqav2_cand`); the train key is the training source
    obs = {"mmlu_letter": {"mmlu_cloze": (0.02, -0.1, 0.12), "arith": (0.3, 0.1, 0.5), "vqav2_cand": (-0.4, -0.5, -0.3),
                           "refcoco": (-0.5, -0.6, -0.4), "triviaqa_cloze": (0.01, -0.1, 0.1)},
           "vqav2": {"mmlu_letter": None, "mmlu_cloze": (0.05, -0.05, 0.14), "refcoco": (0.2, 0.05, 0.35),
                     "triviaqa_cloze": (0.0, -0.1, 0.1)},
           "triviaqa_cloze": {"mmlu_cloze": (0.2, 0.1, 0.3), "mmlu_letter": (0.05, -0.05, 0.14), "vqav2_cand": (0.0, -0.1, 0.1)},
           "arith": {"mmlu_letter": None, "mmlu_cloze": (0.02, -0.1, 0.1), "vqav2_cand": (-0.3, -0.4, -0.2)}}
    im = im_sign_hits(obs); assert im["separates"] and im["n"] == 13 and im["excluded_void"] == 2, im
    obs_bad = {k: {mm: (0.0, -0.05, 0.05) for mm in v} for k, v in obs.items()}
    assert not im_sign_hits(obs_bad)["separates"]
    # the alignment itself is testable: spelt the OLD way, the three vision cells match NOTHING and the
    # matrix silently shrinks to 10 scored cells. n is the tell — that is why the rename was needed.
    obs_old = {tr: {("vqav2" if me == "vqav2_cand" else me): v for me, v in r.items()} for tr, r in obs.items()}
    assert im_sign_hits(obs_old)["n"] == 10, im_sign_hits(obs_old)
    # and every predicted sign is EXACTLY as frozen (naming alignment, never a prediction edit)
    assert sorted(sg for r in IM_PREDICTED_SIGNS.values() for sg in r.values()) == list("++++----0000000")
    assert IM_PREDICTED_SIGNS["mmlu_letter"]["vqav2_cand"] == "-" and IM_PREDICTED_SIGNS["arith"]["vqav2_cand"] == "-"
    assert IM_PREDICTED_SIGNS["triviaqa_cloze"]["vqav2_cand"] == "0" and IM_PREDICTED_SIGNS["vqav2"]["refcoco"] == "+"
    assert not im_sign_hits({"mmlu_letter": obs["mmlu_letter"]})["separates"]   # n < 10
    assert task_pattern(0.7, 0.86, 0.3, 0.38)["pass"] and not task_pattern(0.2, 0.86, 0.3, 0.38)["pass"]
    assert not task_pattern(0.1, 0.1, 0.1, 0.8)["applicable"]
    assert construct_verdict(2, "linearly present", True, True)["verdict"] == "H_construct"
    assert construct_verdict(2, "linearly present", True, None)["verdict"] == "H_construct"
    assert construct_verdict(1, "linearly present", True, True)["verdict"].startswith("H_construct NOT")
    assert construct_verdict(2, "not linearly present", True, True)["verdict"].startswith("H_construct NOT")
    assert construct_verdict(2, "linearly present", True, False)["verdict"].startswith("H_construct NOT")
    assert INDET in construct_verdict(2, "probe " + INDET, True, True)["verdict"]
    # scale
    assert scale_within_family(0.2, (0.05, 0.35))["verdict"] == "H_scale(within-family)"
    assert scale_within_family(-0.2, (-0.35, -0.05))["verdict"].endswith("KILLED")
    assert INDET in scale_within_family(0.05, (-0.1, 0.2))["verdict"]
    assert scale_descriptive([3, 4, 10, 284], [0.86, 0.8, 0.6, 0.3])["status"] == "descriptive only"
    # unlearn (min dose 250 items) + locus
    assert unlearn_verdict(0.27, 0.7, 250)["verdict"] == "H_unlearn-shallow"
    assert unlearn_verdict(0.27, 0.1, None)["verdict"] == "H_unlearn DEEP"
    assert INDET in unlearn_verdict(0.27, 0.7, 600)["verdict"]
    assert unlearn_locus(8, [5, 6, 7]).startswith("trivial") and unlearn_locus(20, [5, 6, 7]).startswith("informative")
    # instruments + SUB (D9)
    assert validity_triple(0.05, 0.1, True, True)["stamp"] == "VALID" and validity_triple(0.05, 8.6, True, True)["stamp"] == "VOID"
    d = 40; inst = rng.standard_normal((d, d)); U, _, _ = np.linalg.svd(inst)
    rec_in = U[:, :4] @ rng.standard_normal((4, d)); rec_out = U[:, 20:24] @ rng.standard_normal((4, d))
    assert subspace_overlap(rec_in, inst, 4, rec_out, n_null=50)["contention"]
    assert not subspace_overlap(rec_in, inst, 4, rec_in, n_null=50)["contention"]     # unrelated update shares the subspace → no contention
    assert not subspace_overlap(rec_out, inst, 4, rec_out, n_null=50)["contention"]
    # `contention` itself, on statistics alone — BOTH branches and each conjunct separately, because this
    # is the entry point every banked-artifact consumer uses and `subspace_overlap` can no longer be the
    # only thing that exercises it.
    assert contention(0.50, 0.30, 0.20) is True                       # both conjuncts hold
    assert contention(0.50, 0.40, 0.20) is True                       # margin met exactly (≥, not >)
    assert contention(0.50, 0.30, 0.20, margin=0.25) is False         # margin is a real knob
    assert contention(0.50, 0.45, 0.20) is False                      # unrelated null not cleared by the margin
    assert contention(0.50, 0.30, 0.60) is False                      # above the unrelated null, below Haar p95
    assert _raises(contention, float("nan"), 0.30, 0.20) and _raises(contention, 0.50, None, 0.20)
    assert subspace_overlap(rec_in, inst, 4, rec_out, n_null=50)["margin"] == CONTENTION_MARGIN
    # preflight (D19)
    ref = (rng.random(285) < 0.47).astype(int); new = ref.copy(); new[:2] ^= 1
    assert preflight_match(new, ref)["pass"] and not preflight_match(1 - ref, ref)["pass"]

    # ======================================================================================
    # 2026-09-03 ADVERSARIAL AUDIT — the checks that could not fail. Every assertion below is
    # an input that USED TO SCORE and now cannot. No threshold, band, floor or constant moved;
    # each case is a dead or degenerate input being refused, or a degeneracy being named.
    # ======================================================================================
    # (C5) a non-finite accuracy read as status "ok": nan - floor < 0.15 is False, so the
    #      VOID-low-margin branch never fired, min(nan, 1.0) is nan, and nan > 1 is False.
    for bad in (np.nan, None):
        mb = margin_lost(bad, 0.28, 0.25)
        assert not np.isfinite(mb["lost"]) and mb["status"].startswith("VOID-nonfinite"), mb
    assert margin_lost(0.47, np.nan, 0.25)["status"].startswith("VOID-nonfinite")
    assert margin_lost(0.47, 0.281, 0.25)["status"] == "ok"                # the healthy path is untouched
    # (C4) an all-NaN arm made every bootstrap draw NaN, `(diff <= 0).mean()` 0.0 — maximal
    #      significance — and `bh_reject([0.0])` accepted it.
    nanv = [float("nan")] * 30
    assert _raises(bootstrap_fraction, nanv, nanv, nanv) and _raises(bootstrap_fraction, list(np.ones(30)), nanv, nanv)
    assert bh_reject([0.0])[0] is True                                     # …which is what it fed
    # (C6) an empty prediction list returned -0.0 bits = a total entropy collapse, the strongest
    #      possible satisfaction of the H_stage conjunct, from a histogram with no predictions.
    assert _raises(letter_entropy, []) and abs(letter_entropy([0, 1, 2, 3]) - 2.0) < 1e-12
    assert format_verdict({"F1": 0.8, "F2": 0.9}, 0.0, 1.9, 0.9)["verdict"] == "H_stage"   # what -0.0 bought
    # (C7) one NaN cut made T NaN and `reader_verdict` returned an INDETERMINATE *result* from a
    #      null that did not run. A dead null must VOID, so it raises and the caller stamps.
    assert _raises(null_spread, [np.nan, 0.1, 0.1, 0.1, 0.1]) and _raises(threshold_T, np.nan)
    assert abs(threshold_T(null_spread([-.13, -.02, -.11, .02, .04])) - 0.26) < 1e-9        # healthy path
    # (C8) NaN CE on BOTH arms: both `>=` comparisons False, so `void` was False — two runs that
    #      produced no loss at all read as "the dose was fine".
    assert _raises(recovery_void, np.nan, np.nan) and _raises(recovery_void, LN4, np.nan)
    assert recovery_void(LN4 - 0.01, LN4 - 0.01)["void"]                   # the healthy path is untouched
    # (D9/D10) degenerate quantifiers: NAMED, never re-decided (a rule change; escalated to the lead).
    agg_one = aggregate_recipes({"r512": c_buy})
    assert agg_one["all_repurchase"] and agg_one["quantifier_degenerate"] and agg_one["n_quantified"] == 1
    assert not agg_buy["quantifier_degenerate"] and agg_buy["n_quantified"] == 3
    s_one = sub_contention_frac([True])
    assert s_one["contention"] and s_one["aggregation_degenerate"] and s_one["n_layers"] == 1
    assert not sc["aggregation_degenerate"] and sc["n_layers"] == 4
    # (D11) the random-init null was echoed in ABSOLUTE accuracy units while P_recovered is a
    #       FRACTION, so the "lower bound" could not be compared with the thing it bounds. Same
    #       units now, plus the comparison; gating it is a rule change and is escalated.
    dead_null = probe_recovered(0.55, 0.65, 0.27, 0.55, 0.50)
    assert dead_null["verdict"] == "linearly present"                      # DECISION UNCHANGED
    assert not dead_null["null_below_measurement"] and abs(dead_null["null_separation"]) < 1e-12
    assert "THE NULL IS NOT BELOW THE MEASUREMENT" in dead_null["null_note"], dead_null
    live_null = probe_recovered(0.55, 0.65, 0.27, 0.26, 0.50)
    assert live_null["null_below_measurement"] and live_null["P_lower_bound"] < live_null["P_recovered"]
    # (D15) the floor was the literal 0.25 while the caller read the block's own banked floor.
    assert unlearn_verdict(0.27, 0.7, 250)["verdict"] == "H_unlearn-shallow"          # default unchanged
    assert unlearn_verdict(0.27, 0.7, 250)["floor"] == 0.25
    at_other = unlearn_verdict(0.27, 0.7, 250, floor=0.1)     # a block whose banked floor is not 0.25
    assert not at_other["at_floor"] and INDET in at_other["verdict"] and at_other["floor"] == 0.1
    print("MENDELEEV matrix_core v3 dry-run: ALL PASS")


if __name__ == "__main__":
    _dryrun()
