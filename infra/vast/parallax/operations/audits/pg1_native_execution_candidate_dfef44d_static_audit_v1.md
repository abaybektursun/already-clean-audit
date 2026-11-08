# PG1-native candidate `dfef44d` static audit

Verdict: **STATIC_NO_GO** against prior NO_GO `0af283a` and trusted science
base `8bba643`. This was a tracked-source-only audit. It made no cloud, Vast,
AWS, price, paid, provider, archive/model/data/confirmation-body, Docker,
real-ID, freeze, claim, authority, science-journal, or science-result action.

## Reproduced evidence

- Focused capsule/native suite: **119 passed**.
- Credential-stripped complete tracked `infra/vast/parallax/tests` suite:
  **806 passed**. This is not a repository-wide test count.
- AST guard: **GREEN** with 19 protected functions, 14 constants, 2 protected
  run slices, and the expected controller/worker traces and slice hashes.
- Candidate delta from `0af283a`: exactly the stated **10 tracked files**.
  Changed Python files pass `py_compile`; `0af283a..dfef44d` and
  `8bba643..dfef44d` pass `git diff --check`.
- All eight legacy/hash-bound critic, PG1, shared-cell, and primary-cell files
  are byte-identical to `8bba643`. The candidate itself changes neither
  `studies/**` nor the existing operations tree.

## Prior-blocker adjudication

| Requirement | Result |
|---|---|
| One tracked Vast response adapter; no invented success envelope | **CLOSED**. The capsule hash-binds one adapter for list/direct/nested rows, exact `id`, and `actual_status`/`cur_state`. |
| Mutation and terminal semantics | **CLOSED**. Mutation stdout is not trusted; zero exit only starts show polling, STOP requires a tracked terminal row, and DESTROY requires three canonical listing absences. |
| Key provenance inside STOP attempts; AWS fallback independent and replay-safe | **CLOSED in `ProductionStopper`**. Key failures reach the fixed conditional S3 marker path and exact readback. |
| Guardian retries transient key/show/parse/nonzero/timeout failures | **CLOSED within its action deadline**, with bounded backoff, but its DESTROY policy and failure signaling remain unsafe. |
| Fixed-layout provisioner and deployment receipt | **CLOSED after the provisioner has started**. It uses fixed paths/modes, exact source hashes, no mechanical selector, and invokes the capsule bootstrap argv. |
| Bootstrap pointer/import/argument supervision | **PARTIAL**. Project imports and argv parsing occur after pointer/guardian arming, but guardian spawn and the outer deployment CLI/import boundary are still uncovered. |
| Complete artifact inventory, layout receipt, hydration, and non-circular durability | **CLOSED statically**. The layout receipt joins the previously complete result/checkpoint bundle and separately persisted final receipt. |
| Mechanical/production separation and unchanged science/legacy bindings | **CLOSED**. |

## Hard blockers

1. **The guardian can destroy undurable science results.** Every controller,
   science, critic, upload, or verification failure may publish the same stop
   marker while checkpoints and result bodies exist only on local scratch. In
   `_act`, three failed or unconfirmed STOP iterations switch directly to
   `vast_destroy_command`; there is no final durability receipt, artifact
   manifest verification, or “science never started” condition on this path.
   This bypasses `ProductionBackend.request_destroy`, whose purpose is to allow
   destroy only after the complete downloaded-byte durability proof. A
   transient STOP failure followed by a successful DESTROY therefore erases
   the bodies the failure pointer claims to preserve. The focused test named
   `test_guardian_nonzero_stop_retries_then_falls_back_to_destroy_absence`
   positively exercises this unsafe transition with no representative
   undurable output tree.

2. **A brief S3 marker outage can still burn almost the entire H100 deadline.**
   Bootstrap failure signaling is only the S3 marker. If its conditional put
   and one readback both fail, bootstrap never retries publication. The
   guardian does not treat the already-written local failure pointer as an
   action trigger; with a healthy Vast key it merely observes the instance as
   running until the 10,800-second watch deadline, then begins STOP. Thus a
   transient AWS failure during an immediate import/setup failure can leave the
   paid instance running for about three hours. If provider action remains
   unavailable at the action deadline, the guardian persists
   `terminal_state: unverified`; bootstrap rejects it but no tracked external
   deadman remains to make the instance nonbillable. This is fail-loud, not
   verified shutdown.

3. **The outer production entry and guardian-start setup boundary are not
   forced or fail-closed.** `production_argv` runs
   `/usr/local/bin/python3 -m infra.vast.parallax.pg1_native_deployment`, while
   the module is declared under `/opt/parallax/image/source`; neither the argv
   nor the deployment contract binds an initial working directory or
   `PYTHONPATH`, and source search finds no tracked invoker that supplies one.
   Module resolution therefore still depends on the caller/image environment
   before the fixed provisioner can establish its environment. Moreover,
   deployment `main` validates argv before entering `provision_and_launch`, and
   bootstrap calls `_start_guardian()` outside its protected `try`. An initial
   module/CLI error or guardian spawn failure can exit with at most a local
   pointer and no marker, supervised guardian, or verified nonbillable state.
   The deterministic rehearsal only copies files and records that the argv was
   “invoked”; its own source states that it never starts that command.

## Rehearsal authorization

**Exactly one bounded no-H100 cloud CPU staged-layout rehearsal is NOT
AUTHORIZED for `dfef44d`.** The unsafe pre-durability DESTROY and missing local
failure handoff are production source defects, not layout uncertainties. Their
repair changes runtime hashes and the capsule digest, making a rehearsal receipt
for this candidate obsolete.
