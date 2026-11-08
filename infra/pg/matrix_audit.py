"""MENDELEEV self-audit — the two mechanisms that stop the two defect classes of 2026-09-03/04.

Roughly a dozen separate defects in sixteen hours reduced to exactly two mechanisms, and fixing the
instances one at a time did not work. This module fixes the mechanisms.

  1. DERIVED-VALUE AUDIT (`audit_derived`). Every registry field that a machine-readable source already
     knows is compared against that source and FAILS LOUD on divergence. The sources are each repo's own
     `config.json` / `model.safetensors.index.json` and the weight bucket's own listing, banked in
     `infra/pg/matrix_repo_facts.json` by `matrix_models --refresh-facts` and never typed by hand. Runs
     inside `matrix_models --dryrun`, which gates every launch; `--s3` re-reads the live bucket.
         python -m infra.pg.matrix_audit            # offline: registry + constants + gates
         python -m infra.pg.matrix_audit --s3       # + the live bucket listing

  2. GATE REGISTRY (`run_gates`). Every gate declares, per CLAUSE, an input that must make it fail, and
     the runner asserts the gate actually fails on it. A gate with no registered failing input, a clause
     whose declared input does not trip it, or a gate-shaped function missing from the registry is a
     TEST FAILURE. Coverage is not a list to maintain: `_gate_functions` re-derives it from the source
     with `ast`, so a new gate cannot be added without a failing input.

WHY A CLAUSE AND NOT A GATE. `matrix_core.ppl_forward_gate` had a failing input, passed its dry-run, and
was still blind to the exact defect it was written for: its lower arm needs `ppl_shell < ppl_bare / 20`,
perplexity is ≥ 1, so at the banked bare perplexity of 7.966 the arm is UNREACHABLE — the ratio bottoms
out at 0.126 and a shell that made the text read PERFECTLY would read VALID. The gate as a whole could
fail; the clause that mattered could not. So a clause that cannot fail must either be repaired or name
the gate that supersedes it, and the runner PROVES unreachability by running the physical extreme rather
than accepting the claim.

Fail-loud; no try/except that hides. Pure stdlib + numpy; no torch, no network except under `--s3`.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from infra.pg import matrix_core as MC        # noqa: E402  (pure numpy)
from infra.pg import matrix_models as MM      # noqa: E402  (weight-free registry + the facts file)

# ============================================================================ 1. derived-value audit
# The one aliasing in the program between a repo's declared text model type and the row's `family`, and
# the evidence for it. Molmo2 wraps a Qwen3-shaped decoder and declares `molmo2_text`; the rows call that
# family "qwen3" because the LINEAGE is Qwen3-4B-Base -> Instruct -> Molmo2. That is a claim about
# anatomy, so `audit_derived` CHECKS the anatomy (layers/hidden) against a repo that really declares the
# aliased type, instead of taking the alias on trust.
FAMILY_ALIAS = {"molmo2": "qwen3", "molmoact2": "qwen3"}
# architecture -> the Forward shell `matrix_models._forward` dispatches for it. Gemma-3 and Gemma-4 ship
# `*ForConditionalGeneration` classes and are deliberately read TEXT-ONLY here (their vision towers are
# never fed), so the choice lives in this one table rather than once per row.
SHELL_OF_ARCH = {
    "GemmaForCausalLM": "text", "Gemma2ForCausalLM": "text", "Qwen3ForCausalLM": "text",
    "MistralForCausalLM": "text", "LlamaForCausalLM": "text", "DeepseekV4ForCausalLM": "text",
    "Gemma3ForConditionalGeneration": "text", "Gemma4ForConditionalGeneration": "text",
    "PaliGemmaForConditionalGeneration": "paligemma",
    "Molmo2ForConditionalGeneration": "molmo2", "MolmoAct2ForConditionalGeneration": "molmo2",
}
# `params_b` is a ROUNDED human-readable figure; the safetensors index is exact. The band is fixed by the
# data, not by taste: the largest honest rounding among the eighteen rows is 2.6 % (row 1a, 3.0 declared
# against 2.923 derived) and the two errors this check exists to catch were 7.8 % and 15.8 % (the Molmo
# rows' 4.5 / 4.7 against 4.851 / 5.442). The geometric mean of 2.6 % and 7.8 % is 4.5 %; 5 % is that,
# rounded outward, and it admits every current row with ≥ 1.9x headroom while rejecting both banked
# errors. It is a PROVENANCE check, not a memory budget: `params_b` is copied into every manifest.
PARAMS_B_TOL = 0.05
# `params_b` is NOT re-derived and written back, deliberately. It is banked inside
# `mdl_metrics/2_beh_smoke/metrics.json` (`manifest.row.params_b` 3.0, `gb_one` 14.6, `gb_pair` 21.2), so
# replacing 3.0 with the exact 3.032 would move a number a banked artifact already records. The check
# makes a WRONG value impossible to commit without moving a right one.

# ---------------------------------------------------------------------------- acknowledged divergences
# A divergence the program has SEEN and cannot resolve without a decision that is not this module's to
# take. It is not a waiver: the audit still computes it, still prints it on every run as an ESCALATION,
# and ASSERTS THAT IT IS STILL TRUE — an acknowledgement of something that has since been fixed is
# itself a failure, so this table cannot rot into a list of ignored checks. Same shape as a gate clause
# marked `cannot_fail`.
ACKNOWLEDGED = {
    "pin-major:3c-Act:allenai/Molmo2-ER":
        "Row 3c-Act is Molmo2-ER -> MolmoAct2 and BOTH members ship remote code declaring DIFFERENT "
        "transformers majors (4.56.2 and 5.3.0). One pin cannot serve both, so the row is a genuine "
        "conflict, not a typo. It is NOT resolved here because every option costs something the lead "
        "must price: (a) keep 5.9.0 — MolmoAct2 loads, and Molmo2-ER's remote code runs on a library it "
        "was not written for. ROW3 verified that the Molmo2 remote code does NOT run on 5.9.0 for "
        "Molmo2-4B (create_causal_mask signature, processor kwargs); whether Molmo2-ER behaves the same "
        "is UNTESTED, and the only observation of it loading at all is under 4.57.1 "
        "(results/mdl/S4/3c-Act_beh/a134124: `before` loaded and snapshotted, then the job died in "
        "`load_after`); "
        "(b) pin 4.57.1 — Molmo2-ER loads and MolmoAct2's major-5 tokenizer config dies as "
        "\"AttributeError: 'list' object has no attribute 'keys'\", the failure that already cost this "
        "row 13 cells twice; (c) pin PER MEMBER (the members load sequentially, so ensure_transformers_pin "
        "could build both venvs) — mechanically possible, but the two arms of one contrast would then be "
        "read by two libraries, which is the very defect the cross-row check below exists to catch, "
        "moved inside a single cell; (d) re-pin the whole Molmo chain to one major and re-run 3b and "
        "3c-ER, orphaning their banked results.",
    "pin-cross:allenai/Molmo2-ER":
        "The same weights (Molmo2-ER) are read under transformers 4 in row 3c-ER and under transformers "
        "5 in row 3c-Act, so the `3b -> 3c-ER -> 3c-Act` chain does not read one object with one library. "
        "3c-ER is DONE with a clean, gate-valid result banked at 4.57.1; changing its pin would orphan "
        "that result, and it is NOT changed here. Note that 3c-ER's own minor mismatch (pin 4.57.1 "
        "against its config's 4.56.2) is NOT what this flags — same major, reported as a note — the "
        "flag is the major split with 3c-Act. Same four options as above.",
}


def _facts(repo: str) -> dict | None:
    return MM.FACTS[repo]["config"]


def _row_repo_facts(row: str) -> list:
    return [(r, _facts(r)) for r in (MM.ROWS[row]["before"], MM.ROWS[row]["after"])]


def _major(v: str) -> str:
    return str(v).split(".", 1)[0]


def audit_derived() -> dict:
    """Every registry field with a machine-readable source, against that source. Returns the report;
    raises AssertionError naming every divergence at once (a caller fixing one should see all of them)."""
    bad, unchecked, notes = [], [], []
    F, K, R = MM.FACTS, MM.KNOWN, MM.ROWS

    # --- the facts file itself covers the registry, at the pinned revisions -----------------------
    assert set(F) == set(K), f"facts/KNOWN repo sets differ: {set(F) ^ set(K)}"
    for repo, k in K.items():
        if F[repo]["revision"] != k["sha"]:
            bad.append(f"{repo}: facts read revision {F[repo]['revision']} but KNOWN pins {k['sha']} — "
                       "re-run `matrix_models --refresh-facts`")
        # `access` is a claim that the token CANNOT read this repo. It is checked against the fetch that
        # the refresher actually made, so "403 licence" cannot be carried on a repo that reads fine.
        readable = F[repo]["config"] is not None
        if (k["access"] == "ok") != readable:
            bad.append(f"{repo}: KNOWN access={k['access']!r} but its config.json "
                       f"{'reads fine' if readable else 'could not be read'} "
                       f"({F[repo]['config_source']})")

    # --- KNOWN vs each repo's own config.json ------------------------------------------------------
    # A field the config does not state is UNCHECKED and is listed as such: silence is reported, never
    # counted as agreement. (`tie_word_embeddings` is absent from most configs — it then comes from the
    # architecture class default, which no offline reader can resolve.)
    for repo, c in ((r, _facts(r)) for r in K):
        k = K[repo]
        if c is None:
            for f in ("arch", "model_type", "n_layers", "hidden", "vocab", "tie", "dtype"):
                if k[f] is not None:
                    bad.append(f"{repo}: config unreadable ({K[repo]['access']}) yet KNOWN states {f}={k[f]!r}")
            continue
        pairs = [("arch", c["architectures"][0] if c["architectures"] else None),
                 ("model_type", c["text_model_type"]), ("n_layers", c["n_layers"]),
                 ("hidden", c["hidden"]), ("vocab", c["vocab"]), ("tie", c["tie"]),
                 ("remote_code", c["auto_map"])]
        for field, src in pairs:
            if src is None:
                unchecked.append(f"{repo}.{field}: the config does not state it (KNOWN says {k[field]!r})")
            elif k[field] != src:
                bad.append(f"{repo}: KNOWN {field}={k[field]!r} but its config.json says {src!r}")
        # dtype: KNOWN carries prose on the quantized rows ("bfloat16 + fp8 quant …"), so the config's
        # value must be its PREFIX rather than its equal.
        if c["dtype"] is None:
            unchecked.append(f"{repo}.dtype: the config does not state it (KNOWN says {k['dtype']!r})")
        elif not str(k["dtype"]).startswith(c["dtype"]):
            bad.append(f"{repo}: KNOWN dtype={k['dtype']!r} does not start with the config's {c['dtype']!r}")

    # --- row fields against the repos they name ----------------------------------------------------
    for row, r in R.items():
        rf = _row_repo_facts(row)
        readable = [(repo, c) for repo, c in rf if c is not None]
        # shell: the loader dispatches on the AFTER repo's architecture (the BEFORE repo's where the
        # after is a bare LeRobot policy or unreadable); a policy row is its own shell.
        archs = [c["architectures"][0] for _, c in reversed(readable) if c["architectures"]]
        if r["policy"]:
            if r["shell"] != "pi0-pg1":
                bad.append(f"row {row}: policy={r['policy']!r} but shell={r['shell']!r}")
        elif archs:
            want = SHELL_OF_ARCH.get(archs[0])
            if want is None:
                bad.append(f"row {row}: architecture {archs[0]!r} is in no SHELL_OF_ARCH entry — add it "
                           "with the shell it dispatches, do not guess at the call site")
            elif want != r["shell"]:
                bad.append(f"row {row}: shell={r['shell']!r} but {archs[0]} dispatches {want!r}")
        else:
            unchecked.append(f"row {row}.shell: neither repo's config declares an architecture")
        # family: the row's LM lineage. Equal to a repo's own text model type, or its declared alias —
        # and the alias is only accepted when the anatomy backs it.
        types = {c["text_model_type"] for _, c in readable if c["text_model_type"]}
        fams = {FAMILY_ALIAS.get(t, t) for t in types}
        if not types:
            unchecked.append(f"row {row}.family: neither repo's config declares a text model type")
        elif r["family"] not in fams:
            bad.append(f"row {row}: family={r['family']!r} but its repos declare {sorted(types)} "
                       f"(families {sorted(fams)})")
        # params_b against the exact index totals of the repos that publish one.
        derived = [c["params_b_from_index"] for _, c in readable if c["params_b_from_index"] is not None]
        if not derived:
            why = ("quantized release: the index publishes bytes at several widths at once"
                   if any(c["quantized"] for _, c in readable) else
                   "no safetensors index at the pinned revision")
            unchecked.append(f"row {row}.params_b: not derivable ({why}); declared {r['params_b']}")
        else:
            want = max(derived)
            off = abs(r["params_b"] - want) / want
            if off > PARAMS_B_TOL:
                bad.append(f"row {row}: params_b={r['params_b']} but the safetensors index totals give "
                           f"{want:.3f} B ({off:.1%} off, tolerance {PARAMS_B_TOL:.0%})")
            else:
                notes.append(f"row {row}: params_b {r['params_b']} vs index {want:.3f} B ({off:.1%})")
        # needs_remote_code is DERIVED in `_row`; this asserts the derivation still selects exactly the
        # rows whose repos ship an auto_map, so the deletion of the hand flag stays honest.
        want_remote = any(c["auto_map"] for _, c in readable)
        if bool(r["needs_remote_code"]) != want_remote:
            bad.append(f"row {row}: needs_remote_code={r['needs_remote_code']} but auto_map={want_remote}")
        # s3_cached, against the bucket's own listing.
        for repo, flag in r["s3_cached"].items():
            have = ("models--" + repo.replace("/", "--")) in MM.FACTS_META["bucket_dirs"]
            if bool(flag) != have:
                bad.append(f"row {row}: s3_cached[{repo}]={flag} but the bucket listing "
                           f"({MM.FACTS_META['bucket_listed']}) says {have}")

    ids = _audit_pins(notes, unchecked)
    # Route the findings the program has already seen and priced into ESCALATIONS; everything else fails
    # the build. An acknowledgement that no longer fires is itself a failure, so the table cannot rot.
    escalated = []
    for fid, msg in ids:
        if fid in ACKNOWLEDGED:
            escalated.append(f"{msg}\n      {ACKNOWLEDGED[fid]}")
        else:
            bad.append(msg)
    stale = sorted(set(ACKNOWLEDGED) - {fid for fid, _ in ids})
    assert not stale, (f"ACKNOWLEDGED names divergences that no longer fire: {stale} — delete the "
                       "entries; an acknowledgement of a fixed defect is a check that cannot fail")
    assert not bad, ("DERIVED-VALUE AUDIT FAILED — a registry field disagrees with the source that "
                     "already knows it:\n  - " + "\n  - ".join(bad))
    return {"checked_rows": len(R), "checked_repos": len(K), "notes": notes, "unchecked": unchecked,
            "escalated": escalated}


def _audit_pins(notes, unchecked) -> list:
    """`transformers_pin` against each repo's own declared `transformers_version`, and across rows.

    THE RULE, and why it is not "the pin equals the declared version". Almost every config here was
    saved by an ancient transformers (gemma-2b says 4.38.0.dev0) and runs fine on the AMI's 5.9.0,
    because an IN-LIBRARY architecture is maintained across versions. The pin is only load-bearing for a
    repo that ships its OWN modelling code (`auto_map`): that code was written against the version the
    config names, and a major bump breaks it — ROW3 verified Molmo2's remote code does not run on 5.9.0
    (create_causal_mask signature, processor kwargs). So: for a remote-code repo, the row's pin must
    share the repo's declared MAJOR. Two rows that load the same weights must also agree, or the same
    object is read by two different libraries and the two readings are not comparable."""
    bad = []
    for row, r in MM.ROWS.items():
        pin = r["transformers_pin"]
        for repo in (r["before"], r["after"]):
            c = _facts(repo)
            if c is None or not c["auto_map"]:
                continue
            decl = c["transformers_version"]
            if decl is None:
                unchecked.append(f"row {row}: {repo} ships remote code but declares no transformers_version")
            elif _major(decl) != _major(pin):
                bad.append((f"pin-major:{row}:{repo}",
                            f"row {row}: transformers_pin={pin} but its remote-code repo {repo} declares "
                            f"{decl} — different MAJOR, so the repo's own modelling code is run by a "
                            "library it was not written against"))
            elif decl != pin:
                notes.append(f"row {row}: pin {pin} vs {repo}'s declared {decl} (same major; "
                             "in-major drift is tolerated, see _audit_pins)")
    # one repo, one pin, program-wide
    by_repo = {}
    for row, r in MM.ROWS.items():
        for repo in (r["before"], r["after"]):
            if _facts(repo) is not None and _facts(repo)["auto_map"]:
                by_repo.setdefault(repo, {}).setdefault(_major(r["transformers_pin"]), []).append(row)
    for repo, majors in by_repo.items():
        if len(majors) > 1:
            bad.append((f"pin-cross:{repo}",
                        f"{repo} is loaded under transformers majors " +
                        ", ".join(f"{m} (rows {sorted(rs)})" for m, rs in sorted(majors.items())) +
                        " — the same weights read by two libraries; one reading is not the other"))
    return bad


def audit_s3() -> dict:
    """The bucket listing banked in the facts file against the LIVE bucket. Network; not in `--dryrun`."""
    r = subprocess.run(["aws", "s3", "ls", f"s3://{MM.S3_BUCKET}/weights/hf/hub/", "--profile", "fuelos",
                        "--region", "us-east-1"], capture_output=True, text=True)
    assert r.returncode == 0, f"cannot list the weight cache: {r.stderr.strip()[:300]}"
    live = sorted(ln.split("PRE ", 1)[1].rstrip("/") for ln in r.stdout.splitlines()
                  if " PRE " in ln and "PRE models--" in ln)
    banked = sorted(MM.FACTS_META["bucket_dirs"])
    assert live == banked, ("the weight cache has moved since the facts file was written "
                            f"(listed {MM.FACTS_META['bucket_listed']}): only in the bucket "
                            f"{sorted(set(live) - set(banked))}, only in the facts file "
                            f"{sorted(set(banked) - set(live))} — run `matrix_models --refresh-facts`")
    return {"bucket_dirs": len(live)}


def audit_constants() -> dict:
    """Values duplicated ACROSS modules, compared. Imports the verdict layer and the job, so it runs from
    `matrix_verdicts --selftest` rather than from a GPU job's registry dry-run.

    The floors are the live case: `scripts/matrix_verdicts.FLOORS` scores every verdict and
    `infra/aws_pg/jobs/matrix_behaviour.FLOOR` is what the job stamps onto each readout record. Two live
    copies of one scoring table is exactly the shape that produced the dead eighth copy in
    `matrix_models` — so they are compared here on every key they share."""
    from infra.aws_pg.jobs import matrix_behaviour as MB
    from scripts import matrix_verdicts as MV
    from scripts import matrix_roster_build as MRB
    bad = []
    shared = set(MV.FLOORS) & set(MB.FLOOR)
    for k in sorted(shared):
        if MV.FLOORS[k] != MB.FLOOR[k]:
            bad.append(f"floor {k!r}: matrix_verdicts.FLOORS={MV.FLOORS[k]} but "
                       f"matrix_behaviour.FLOOR={MB.FLOOR[k]}")
    assert shared, "the two floor tables share no key — one of them has been renamed"
    # the contention margin has ONE definition and the verdict layer reads it rather than retyping it
    if MV.SUB_MARGIN != MC.CONTENTION_MARGIN:
        bad.append(f"SUB_MARGIN {MV.SUB_MARGIN} != matrix_core.CONTENTION_MARGIN {MC.CONTENTION_MARGIN}")
    # every hand-typed set of ROW NAMES anywhere must be the registry's row set or a subset of it. Four
    # such sets diverged in one day (POLICY_ROWS, S3_CACHED, O_ROWS, the 3c-Act pin), so the rule is
    # mechanical: a name that is not a row is a typo, and the roster's row table must be complete.
    if set(MRB.ROWS) != set(MM.ROWS):
        bad.append(f"matrix_roster_build.ROWS != matrix_models.ROWS: "
                   f"only in roster {sorted(set(MRB.ROWS) - set(MM.ROWS))}, "
                   f"only in registry {sorted(set(MM.ROWS) - set(MRB.ROWS))}")
    for name in ("U_ROWS", "P_ROWS", "G_ROWS", "REC3", "REC_LORA", "REH", "GB", "SUB", "IM", "INS",
                 "O_ROWS", "UNLEARN_ROWS", "AT_ROWS"):
        stray = sorted(set(getattr(MRB, name)) - set(MM.ROWS))
        if stray:
            bad.append(f"matrix_roster_build.{name} names {stray}, which are not rows")
    # `matrix_launch` re-exports the cache list rather than keeping one; assert it is the same object's
    # value, so a future edit there cannot fork it again.
    from scripts import matrix_launch as ML
    if tuple(ML.S3_CACHED_REPOS) != tuple(MM.S3_CACHED):
        bad.append("matrix_launch.S3_CACHED_REPOS has forked from matrix_models.S3_CACHED")
    assert not bad, "CROSS-MODULE CONSTANT AUDIT FAILED:\n  - " + "\n  - ".join(bad)
    return {"floor_keys_compared": len(shared), "roster_rows": len(MRB.ROWS)}


# ================================================================================ 2. the gate registry
# Every gate declares, per CLAUSE, an input that must make it fail. `run_gates` asserts each one does.
#
# A clause carries EITHER `reachable` (prose citing where such an input actually occurs) OR, when the
# clause provably cannot fire in the regime the instrument produces, `cannot_fail` + `superseded_by`.
# The second form is not a waiver: the runner RUNS the physical extreme, asserts it does NOT trip the
# gate (so the hole is proved, not claimed), asserts the superseding gate is itself registered with a
# clause that DOES fail, and prints the hole as an ESCALATION on every run.
def _lt(key, val):
    return lambda r: r[key] == val


def _starts(key, val):
    return lambda r: str(r[key]).startswith(val)


GATES = {
    # -------------------------------------------------------------------------------- matrix_core §6.1
    "margin_lost": {
        "scope": "core", "fn": lambda: MC.margin_lost,
        "pass": {"args": (0.47, 0.281, 0.25), "expect": _lt("status", "ok"),
                 "anchor": "C1 row 2: healthy 0.47, damaged 0.281, letter floor 0.25"},
        "clauses": {
            "healthy margin too small to read anything": {
                "args": (0.30, 0.28, 0.25), "expect": _lt("status", "VOID-low-margin"),
                "reachable": "a readout where both arms sit near the floor — the C6 case the rule exists for"},
            "a dead readout must not read healthy": {
                "args": (float("nan"), 0.28, 0.25), "expect": _starts("status", "VOID-nonfinite"),
                "reachable": "a column whose per-item vector came back empty (D8) — before the "
                             "2026-09-03 repair this returned status 'ok'"},
        }},
    "k_positive_control": {
        "scope": "core", "fn": lambda: MC.k_positive_control,
        "pass": {"args": (0.9,), "expect": _lt("K_column", "VALID"),
                 "anchor": "row 5c, the content-destroyed pair (H-2 skill_r2 MCQ 0.656 -> 0.263)"},
        "clauses": {
            "K did not fall on destroyed content": {
                "args": (0.2,), "expect": _starts("K_column", "VOID"),
                "reachable": "the C6 lesson: K read 'kept' on a pair whose content was destroyed"}}},
    "k_shuffle_null": {
        "scope": "core", "fn": lambda: MC.k_shuffle_null,
        "pass": {"args": (0.27,), "expect": _lt("K_null", "at floor"), "anchor": "floor 0.25 + tol at n=285"},
        "clauses": {
            "a length/frequency prior is doing the scoring": {
                "args": (0.36,), "expect": _lt("K_null", "VOID-K-null"),
                "reachable": "options scored against a permuted question landing above floor"}}},
    "k_shell_parity": {
        "scope": "core", "fn": lambda: MC.k_shell_parity,
        "pass": {"args": (0.44, 0.43, 0.47, 0.46), "expect": lambda r: r["S_shell"] == r["K_shell"] == "VALID",
                 "anchor": "the healthy LM read bare and through the blank-image shell, agreeing"},
        "clauses": {
            "the shell moves S away from the bare reading": {
                "args": (0.44, 0.43, 0.47, 0.40), "expect": _lt("S_shell", "VOID-shell"),
                "reachable": "row 2: the blank-image shell costs 0.158 MMLU accuracy (confound 32)"},
            "the shell moves K away from the bare reading": {
                "args": (0.44, 0.30, 0.47, 0.46), "expect": _lt("K_shell", "VOID-shell"),
                "reachable": "the same shell cost on the cloze readout"}}},
    "ppl_forward_gate": {
        "scope": "core", "fn": lambda: MC.ppl_forward_gate,
        "pass": {"args": (42.789, 17.286), "expect": _lt("ppl_forward", "VALID"),
                 "anchor": "bosgate_ppl/a1: the same weights, 2.48x from a one-token framing change"},
        "clauses": {
            "the shell makes the text read far HARDER": {
                "args": (18558.6, 7.966), "expect": _lt("ppl_forward", "VOID-ppl-forward"),
                "reachable": "row 2 `before`: 18558.6 through the shell against 7.966 bare (2329x)"},
            "the shell makes the text read EASIER (a non-causal mask)": {
                # THE HOLE, PROVED RATHER THAN CLAIMED. The lower arm needs ppl_shell < ppl_bare / 20.
                # Perplexity is >= 1, so at the banked bare 7.966 the ratio bottoms out at 0.126 and the
                # arm can NEVER fire: a shell that predicted every token PERFECTLY reads VALID at 0.125.
                # A symmetric max(a/b, b/a) repair does not help — 8.43x is inside a 20x tolerance. No
                # ratio of two perplexities can see this failure, because the failure DRIVES BOTH ARMS
                # TOWARD 1.0. `causality_gate` measures the property directly instead.
                "args": (1.0, 7.966), "expect": _lt("ppl_forward", "VOID-ppl-forward"),
                "cannot_fail": "perplexity >= 1, so at the banked bare ppl 7.966 the ratio cannot go "
                               "below 1/7.966 = 0.126, and the lower arm needs < 0.05. A shell that read "
                               "the text PERFECTLY would be stamped VALID at ratio 0.125.",
                "superseded_by": ("causality_gate", "a later token changed an earlier token's logprob")}}},
    "causality_gate": {
        "scope": "core", "fn": lambda: MC.causality_gate,
        "pass": {"args": (-8.31, -8.31, -4.02, -11.60), "expect": _lt("causality", "VALID"),
                 "anchor": "a causal decoder: the recovered span logprob is identical under two "
                           "different following tokens, and the true continuation beats a scrambled one"},
        "clauses": {
            "a later token changed an earlier token's logprob": {
                "args": (-8.31, -5.02, -4.02, -11.60), "expect": _starts("causality", "VOID-causality"),
                "reachable": "a bidirectional mask over the joint sequence: the span is scored with the "
                             "following token visible, so which token follows changes its logprob"},
            "the forward predicts everything perfectly (targets are visible)": {
                "args": (-0.001, -0.001, -0.002, -0.002), "expect": _starts("causality", "VOID-causality"),
                "reachable": "the ppl -> 1.0 failure: every target readable from its own position, so "
                             "the true and the scrambled continuation score alike and the invariance "
                             "clause above is satisfied by 0 == 0"}}},
    "tokenizer_gate": {
        "scope": "core", "fn": lambda: MC.tokenizer_gate,
        "pass": {"args": ([1, 1, 1, 1], 0.99, 0.9, 0.8), "expect": _starts("S_family", "VALID"),
                 "anchor": "single-token letters, healthy argmax in the four letters on 99% of items"},
        "clauses": {
            "a letter is not a single token": {
                "args": ([1, 2, 1, 1], 0.99, 0.9, 0.8), "expect": _starts("S_family", "VOID-tokenizer"),
                "reachable": "any family whose ' A'..' D' do not tokenize alike"},
            "off-format on BOTH surfaces": {
                "args": ([1, 1, 1, 1], 0.0, 1e-7, 1e-7),
                "kwargs": {"native_argmax_frac": 0.1, "native_p_after": 0.05, "surface_agreement": 0.99,
                           "acc_scored": 0.25, "acc_native": 0.25},
                "expect": _starts("S_family", "VOID-tokenizer"),
                "reachable": "the 2026-09-03 instrument alarm: a readout scoring a token neither model emits"},
            "accuracy is surface-DEPENDENT": {
                "args": ([1, 1, 1, 1], 1.0, 0.99, 0.99),
                "kwargs": {"native_argmax_frac": 1.0, "native_p_after": 0.9, "surface_agreement": 0.97,
                           "acc_scored": 0.44, "acc_native": 0.36},
                "expect": lambda r: "surface-dependent" in r["S_family"],
                "reachable": "ctrl-g1: the two letter surfaces disagreeing on accuracy, not just per item"}}},
    "u_valid": {
        "scope": "core", "fn": lambda: MC.u_valid,
        "pass": {"args": (0.45, 0.30), "expect": _lt("U", "VALID"),
                 "anchor": "healthy two-hop above floor+0.15, shortcut set at floor+0.05"},
        "clauses": {
            "the healthy model cannot do the two-hop task": {
                "args": (0.30, 0.30), "expect": _lt("U", "VOID-U"),
                "reachable": "a two-hop set the healthy model reads at floor"},
            "the second hop alone answers the item": {
                "args": (0.45, 0.42), "expect": _lt("U", "VOID-U"),
                "reachable": "D16: the shortcut set scoring as well as the full item"}}},
    "graft_identity_gate": {
        "scope": "core", "fn": lambda: MC.graft_identity_gate,
        "pass": {"args": (1.0, 1e-3, 1.0, 1e-3, 0.95), "expect": _lt("gate", "VALID"),
                 "anchor": "5c_G: self-graft exact, donor-head reproduction exact"},
        "clauses": {
            "the self-graft does not reproduce the model": {
                "args": (0.8, 1e-3, 1.0, 1e-3, 0.95), "expect": _lt("gate", "VOID-wiring"),
                "reachable": "matrix_graft's own dry-run: item_match_frac 0.0 at a wrong hook"},
            "the hook is at the wrong point (donor head does not reproduce the donor)": {
                "args": (1.0, 1e-3, 0.5, 2.6, 0.95), "expect": _lt("gate", "VOID-wiring"),
                "reachable": "the D2 wiring test: a wrong hook point passes the self-graft trivially"}}},
    "transplant_wiring_gate": {
        "scope": "core", "fn": lambda: MC.transplant_wiring_gate,
        "pass": {"args": (0.71, 0.70), "expect": _lt("gate", "VALID"), "anchor": "C5 saw exact equality"},
        "clauses": {
            "the weight prefix does not reproduce the activation graft": {
                "args": (0.71, 0.40), "expect": _lt("gate", "VOID-wiring"),
                "reachable": "D17: a transplant whose low R(band) would otherwise read as a kill"}}},
    "permuted_label_void": {
        "scope": "core", "fn": lambda: MC.permuted_label_void,
        "pass": {"args": (0.05,), "expect": _lt("REC", "VALID"),
                 "anchor": "the H-2 permuted-label recipe leaving the letter unrestored"},
        "clauses": {
            "any label training restores the letter": {
                "args": (0.6,), "expect": _starts("REC", "VOID-format"),
                "reachable": "§6.13: the shuffle control recovering as well as the real recipe"}}},
    "probe_recovered": {
        "scope": "core", "fn": lambda: MC.probe_recovered,
        "pass": {"args": (0.62, 0.66, 0.26, 0.30, 0.50),
                 "expect": lambda r: not str(r["verdict"]).startswith("VOID"),
                 "anchor": "a probe whose shuffle control sits at chance and whose healthy margin is 0.41"},
        "clauses": {
            "the probe's own shuffle control is not at chance": {
                "args": (0.62, 0.66, 0.55, 0.30, 0.50), "expect": _starts("verdict", "VOID-probe"),
                "reachable": "a probe reading structure out of shuffled labels"},
            "the healthy probe has no margin to lose": {
                "args": (0.62, 0.30, 0.26, 0.30, 0.50), "expect": _starts("verdict", "VOID-probe"),
                "reachable": "a layer where the healthy model's own probe sits near chance"}}},
    "validity_triple": {
        "scope": "core", "fn": lambda: MC.validity_triple,
        "pass": {"args": (0.2, 0.25, True, True), "expect": _lt("stamp", "VALID"),
                 "anchor": "fit under 0.3 FVU on both arms, plant fired, identity floor computed"},
        "clauses": {
            "the plant did not fire": {
                "args": (0.2, 0.25, False, True), "expect": _lt("stamp", "VOID"),
                "reachable": "HELMHOLTZ SUCCESSOR: the CKA plant control failed on gemma-2-2b residuals"},
            "the fit is too poor to read": {
                "args": (0.2, 0.45, True, True), "expect": _lt("stamp", "VOID"),
                "reachable": "an SAE/probe whose FVU on one arm exceeds 0.3"},
            "no identity floor was computed": {
                "args": (0.2, 0.25, True, False), "expect": _lt("stamp", "VOID"),
                "reachable": "an instrument reported without its own null"}}},
    "recovery_void": {
        "scope": "core", "fn": lambda: MC.recovery_void,
        "pass": {"args": (0.9, 0.9), "expect": lambda r: r["void"] is False,
                 "anchor": "both arms moved off the chance floor of the answer-token CE"},
        "clauses": {
            "the dose and its fair-scratch control are both stuck at chance": {
                "args": (MC.LN4, MC.LN4), "expect": lambda r: r["void"] is True,
                "reachable": "a ladder whose training did nothing on either arm"},
            "a training run that produced no loss must raise, not read 'fine'": {
                "args": (float("nan"), float("nan")), "raises": True,
                "reachable": "the 2026-09-03 defect: NaN CE on both arms returned void=False"}}},
    "preflight_match": {
        "scope": "core", "fn": lambda: MC.preflight_match,
        "pass": {"args": ([1] * 20 + [0] * 20, [1] * 20 + [0] * 20), "expect": lambda r: r["pass"] is True,
                 "anchor": "D19: the harness reproducing a banked artifact per item"},
        "clauses": {
            "the harness does not reproduce the banked artifact": {
                "args": ([1] * 20 + [0] * 20, [0] * 20 + [1] * 20), "expect": lambda r: r["pass"] is False,
                "reachable": "a C1 preflight run under a different item order or prompt"}}},
    "null_spread": {
        "scope": "core", "fn": lambda: MC.null_spread,
        "pass": {"args": ([0.01, -0.02, 0.03, 0.0, 0.05],), "expect": lambda v: v == 0.05,
                 "anchor": "five finite mismatched-item cuts"},
        "clauses": {
            "the null did not run": {
                "args": ([0.01, float("nan"), 0.03, 0.0, 0.05],), "raises": True,
                "reachable": "the 2026-09-03 defect: one NaN cut made T NaN and the sweep read "
                             "'H_reader INDETERMINATE' from an uncalibrated instrument"}}},
}
# Gate-shaped functions in `matrix_core` that the AST sweep would not spot, because they decide with a
# bool or an exception rather than a "VOID"/"VALID" string. Named here so the coverage rule stays total.
CORE_EXTRA_GATES = ("recovery_void", "preflight_match", "null_spread")


def _gate_functions(path: str) -> set:
    """Every gate-shaped function in a module, re-derived from the SOURCE. A function is a gate when its
    body contains a "VOID…"/"VALID" string constant — that is what a gate is, a thing that stamps a
    measurement valid or void. Coverage is therefore not a list anyone maintains: adding a gate without
    registering a failing input for it breaks the run."""
    tree = ast.parse(open(path).read())
    out = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
            continue
        if any(isinstance(n, ast.Constant) and isinstance(n.value, str)
               and (n.value.startswith("VOID") or n.value == "VALID") for n in ast.walk(node)):
            out.add(node.name)
    return out


def run_gates(scope: str = "all", verbose: bool = False) -> dict:
    """Run every registered gate's pass case and every clause's declared failing input."""
    import numpy as np
    sel = {k: v for k, v in GATES.items() if scope == "all" or v["scope"] == scope}
    escalations, ran = [], 0

    def call(fn, case):
        return fn(*case["args"], **case.get("kwargs", {}))

    for name, g in sorted(sel.items()):
        fn = g["fn"]()
        p = g["pass"]
        assert p.get("anchor"), f"{name}: the passing case must name the regime it comes from"
        r = call(fn, p)
        assert p["expect"](r), f"{name}: the PASSING case did not pass — {r} ({p['anchor']})"
        assert g["clauses"], f"{name}: a gate with no clause has no input that makes it fail"
        for clause, c in g["clauses"].items():
            ran += 1
            assert ("reachable" in c) ^ ("cannot_fail" in c), (
                f"{name}/{clause}: a clause declares EITHER where its failing input occurs (`reachable`) "
                "OR that it provably cannot fire (`cannot_fail` + `superseded_by`)")
            if c.get("raises"):
                assert MC._raises(fn, *c["args"], **c.get("kwargs", {})), (
                    f"{name}/{clause}: the declared input did not raise — the gate cannot fail here")
                continue
            out = call(fn, c)
            fired = bool(c["expect"](out))
            if "cannot_fail" in c:
                # The hole is PROVED, not asserted: the physical extreme must genuinely fail to trip it.
                assert not fired, (f"{name}/{clause} is marked `cannot_fail` but the declared input DOES "
                                   f"trip it ({out}) — delete the marking and the supersession")
                sup, sup_clause = c["superseded_by"]
                assert sup in GATES and sup_clause in GATES[sup]["clauses"], (
                    f"{name}/{clause} names {sup}/{sup_clause!r} as its replacement, which is not a "
                    "registered gate clause — a hole may only be transferred to a gate that is tested")
                assert "cannot_fail" not in GATES[sup]["clauses"][sup_clause], (
                    f"{name}/{clause} is superseded by another clause that also cannot fail")
                escalations.append(f"{name} / {clause}: {c['cannot_fail']} Covered instead by "
                                   f"{sup}/{sup_clause!r}.")
            else:
                assert fired, (f"{name}/{clause}: the declared failing input {c['args']} did NOT trip the "
                               f"gate — it returned {out}. A clause that cannot fail is not a check.")
    if scope in ("all", "core"):
        want = _gate_functions(MC.__file__) | set(CORE_EXTRA_GATES)
        missing = sorted(want - set(GATES))
        assert not missing, (f"gates in matrix_core with no registered failing input: {missing} — "
                             "register a clause with the input that must make each one fail")
    if scope == "all":
        _run_verdict_wiring()
    for e in escalations:
        print(f"  GATE ESCALATION — {e}")
    if verbose:
        print(f"  gate registry: {len(sel)} gates, {ran} clauses, each with an input that trips it")
    return {"gates": len(sel), "clauses": ran, "escalations": escalations}


def _run_verdict_wiring() -> None:
    """The verdict layer's own can-it-fail question: a VOIDed gate must actually withhold a verdict.

    `GATE_SPECS` maps each banked gate to the readout columns it licenses. For every entry, a half record
    carrying that gate's VOID stamp must (a) be seen by `_gate_voids` on a column it licenses and (b) make
    `_inherit_gate_void` withhold the verdict. Seventeen gates were computed and read by nothing on
    2026-09-03; this is what makes that state a test failure rather than a code review finding.

    WHAT THIS ONE CANNOT DO, and where it is closed. The record tested here is built FROM the spec, so a
    spec entry that addresses a key or field the JOB never banks would still pass — a mutation test found
    exactly that hole in this function on 2026-09-04. Only a real artifact can close that end, so
    `matrix_behaviour._dryrun` runs `behaviour_verdicts` over its own row-2 cell and asserts every
    GATE_SPECS entry finds a stamp there. Both ends together are the check; neither alone is."""
    from scripts import matrix_verdicts as MV
    for name, (key, field, covers) in MV.GATE_SPECS.items():
        stamp = False if name == "preflight_c1" else "VOID-x"
        cols = ["S"] if covers is MV.ALL_COLUMNS else list(covers)
        h = {key: {field: stamp}}
        voids = MV._gate_voids(h, *cols)
        assert any(g == name for g, _, _ in voids), (
            f"GATE_SPECS[{name!r}] voids on {key}.{field} but `_gate_voids` did not see it on {cols}")
        v = MV._inherit_gate_void({"verdict": "H_x"}, h, *cols)
        assert v["verdict"].startswith("VOID-inherited") and v["verdict_withheld_by_gate"] == "H_x", (
            f"GATE_SPECS[{name!r}] VOIDs but the verdict was not withheld: {v}")


def main(argv=None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    d = audit_derived()
    print(f"PASS derived-value audit ({d['checked_rows']} rows, {d['checked_repos']} repos against "
          f"their own config.json / safetensors index / the bucket listing)")
    print(f"  {len(d['unchecked'])} field(s) UNCHECKED because the source is silent about them "
          "(reported, never counted as agreement)")
    if "--verbose" in argv:
        for n in d["unchecked"] + d["notes"]:
            print(f"    {n}")
    for e in d["escalated"]:
        print(f"  REGISTRY ESCALATION — {e}")
    g = run_gates("all", verbose=True)
    c = audit_constants()
    print(f"PASS cross-module constants ({c['floor_keys_compared']} floor keys, "
          f"{c['roster_rows']} roster rows)")
    if "--s3" in argv:
        s = audit_s3()
        print(f"PASS live bucket listing ({s['bucket_dirs']} cached repos)")
    print(f"PASS matrix_audit ({g['gates']} gates, {g['clauses']} clauses, "
          f"{len(g['escalations'])} escalated hole(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
