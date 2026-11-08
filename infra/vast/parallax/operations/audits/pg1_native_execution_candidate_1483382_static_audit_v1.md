# PG1-native candidate `1483382` static audit

Verdict: **STATIC_NO_GO** against prior NO_GO `ccf3593` and trusted science
base `8bba643`. This was a tracked-source-only audit. It made no cloud, Vast,
AWS, price, paid, archive/model/data/confirmation-body, Docker, real-ID,
freeze, claim, authority, science-journal, or science-result action.

## Reproduced evidence

- Focused capsule/native suite: **101 passed**.
- Credential-stripped complete tracked `infra/vast/parallax/tests` suite:
  **788 passed**. This is not a repository-wide test count.
- AST guard: **GREEN** with 19 protected functions, 14 constants, 2 protected
  run slices, and the expected fixed controller and worker traces.
- Candidate delta from `ccf3593`: exactly the stated **15 tracked files**.
  The changed files compile; `ccf3593..1483382` and `8bba643..1483382` pass
  `git diff --check`.
- All eight legacy/hash-bound critic, PG1, shared-cell, and primary-cell files
  are byte-identical to `8bba643`. The candidate itself changes neither
  `studies/**` nor the existing operations tree.

## Prior-blocker adjudication

| Requirement | Result |
|---|---|
| Mixed-manifest selection and selected visual-body reads | **CLOSED**. Hydration filters with the exact science selector and opens image bodies only for selected visual rows; valid image-less rows pass. |
| Complete result/checkpoint durability and actual `artifact_key` use | **CLOSED statically**. The exact output inventory, checkpoint bodies, singleton evidence, bundle, manifest, and final receipt are conditionally uploaded and downloaded-byte verified. |
| Non-circular final teardown evidence | **CLOSED statically**. The five payload/evidence objects are verified, their final receipt is then separately persisted and downloaded, and only that verification authorizes teardown. |
| Reject false/wrong/null/nonzero/invalid/timeout Vast outcomes | **CLOSED only for the invented unit-test envelope; OPEN for the tracked provider interface**. |
| Tracked guardian, canonical marker, replay/race/timeout/state verification | **PARTIAL**. It is launched before the worker and has bounded calls, but ordinary provider/key failures can terminate or bypass it without fail-closed action. |
| Absolute executable/config/credential paths and AWS/current-instance binding | **CLOSED as a contract shape; OPEN as an executable provider proof** because the expected Vast response schema is unsupported by the tracked provider adapter. |
| Tracked bootstrap materializes exact runtime and execs exact production argv | **PARTIAL**. The consumer exists and the local order is correct; no tracked deployment path invokes/provisions it, and pre-import failures remain outside its stop coverage. |
| Mechanical contract/backend cannot authorize production | **CLOSED**. Mechanical schemas/store/transport remain separate and production validators reject them. |
| Unchanged verified science, legacy PG1/PG2 bindings, and no journal mutation | **CLOSED**. |

## Hard blockers

1. **The fixed Vast parser does not match the repository's tracked Vast
   interface, so production identity and terminal-state checks are not
   executable.** `run_json` rejects list responses, and `_vast_semantics`
   requires every raw show/stop/destroy response to contain `success: true`,
   `instance_id`, and `state`; show additionally must contain the frozen offer,
   machine, and container fields. The established tracked Vast normalizer
   explicitly accepts list/nested-dict show payloads and reads instance `id`
   plus `actual_status`/`cur_state`. Its destruction proof is repeated absence,
   not a surviving row whose synthetic `state` is `destroyed`. The new tests
   exercise only hand-built `{success, instance_id, state, ...}` dictionaries.
   Therefore provider proof, stop confirmation, and especially post-destroy
   confirmation can fail even when Vast performed the requested action. The
   mechanical CPU rehearsal never calls this provider and cannot adjudicate
   the mismatch.

2. **The fallback/guardian path is not fail closed.** The production stopper
   loads and validates the Vast key outside its primary-attempt `try` blocks.
   A missing, unreadable, wrong-mode, or wrong-hash key raises before the S3
   guardian marker is attempted, although that fallback does not require the
   Vast key. In the guardian poll loop, a single key failure, show timeout,
   nonzero response, malformed response, or identity parse error escapes the
   loop and terminates the guardian. The bootstrap checks the child only once
   before `execve` and has no post-exec supervisor/restart path. Thus the
   candidate's timeout and race plants do not cover a normal transient provider
   failure in the actual guardian.

3. **The bootstrap is not yet a complete tracked deployment path or complete
   setup-failure boundary.** It imports the capsule, backend, and provider before
   `main` can preserve a pointer or construct a stopper. Source search finds no
   tracked launcher/provisioner that creates the bound stop context, installs
   the fixed receipts/secrets/capsule/source layout, and invokes
   `bootstrap_argv`; outside metadata, tests, and the bootstrap's self-check,
   that argv has no consumer. A manually staged CPU rehearsal could prove the
   copy/exec analogue but cannot prove that a future paid instance is forced
   through it or protected when interpreter/import setup fails.

## Rehearsal authorization

**Exactly one bounded no-H100 cloud CPU staged-layout rehearsal is NOT
AUTHORIZED for `1483382`.** The provider schema, fallback bypass, guardian exit,
and deployment-entry gaps are source defects outside the deterministic
rehearsal transport. Fixing them changes runtime hashes and the capsule digest,
so a receipt for this candidate would be obsolete.
