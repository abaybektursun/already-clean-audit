# PG1-native candidate `b0a9aec` static audit

Verdict: **STATIC_NO_GO** against prior NO_GO `5ed5099`, earlier candidates,
and trusted science base `8bba643`. This audit used tracked source only. It made
no cloud, Vast, AWS, price, paid, archive/model/data/confirmation-body, Docker,
real-ID, freeze, claim, authority, or science-journal action.

## Reproduced evidence

- Focused capsule/native suite: **84 passed**.
- Credential-stripped complete tracked `infra/vast/parallax/tests` suite:
  **771 passed**. This is not a repository-wide test count.
- AST guard: **GREEN** with 19 protected functions, 14 constants, 2 protected
  run slices, and the expected fixed controller and worker traces.
- All eight legacy/hash-bound critic, PG1, shared-cell, and primary-cell files
  are byte-identical to `8bba643`; the candidate has no tracked `studies/**`
  diff. `5ed5099..b0a9aec` and `8bba643..b0a9aec` both pass `git diff --check`.

## Prior-blocker adjudication

| Requirement | Result |
|---|---|
| Launcher invokes exact capsule `normal_controller_argv`; seven fixed owners, no callbacks | **CLOSED inside the new isolated launcher/controller**; no tracked production spec invokes that launcher. |
| Production cannot select mechanical schema/backend | **CLOSED**; separate executable and schemas, production-only issuer validation, no controller rehearsal mode. |
| Canonical authorization/receipt identity and independently loaded marker body | **CLOSED**; bucket/namespace/keys are derived, receipt key is canonical, and exact versioned marker bytes are downloaded and checked. |
| Conditional issue/base consume/owner consume replay and race safety | **CLOSED statically** by fixed `If-None-Match: *` writes and exact checksum/SSE evidence. |
| Literal AWS profile/region and fixed Vast command/key path | **CLOSED for argv shape; OPEN for executable, credential-scope, account, and current-instance identity**. |
| Conditional, checksummed, encrypted, downloaded-byte durability | **CLOSED for terminal, scratch pointer, and critic JSON only; real science artifacts are omitted**. |
| Pointer-first idempotent stop, fallback, and no false/double/premature lifecycle success | **OPEN — hard blocker**. |
| Post-authority exact model/tokenizer/task hydration before unchanged science | **OPEN — the canonical task bank cannot pass the new validator**. |
| Science AST/output/adjudication and PG2/legacy bindings unchanged | **CLOSED**. |

## Hard blockers

1. **Canonical hydration deterministically rejects the canonical mixed task-bank
   schema.** The tracked task-bank generator deliberately emits image-less
   interface rows with `image: null` and no `image_sha256`/`image_bytes`.
   `_task_bodies` requires those image fields on every discovery-manifest row
   before filtering to the registered visual `task_id`. It therefore raises on
   a valid non-image row and never reaches science. The mechanical harness uses
   a one-row all-image fixture, so 84/771 cannot expose this production failure.

2. **Real science artifacts are still destroyed rather than made durable.** The
   science path creates the output tree and checkpoint `.pt` bodies, and the
   capsule declares `artifact_key`, but no native production module consumes
   that key. The worker uploads and downloads-verifies only raw terminal,
   scratch-pointer JSON, and critic JSON, then destroys the instance. A pointer
   and hashes do not preserve the checkpoint/output bodies. The synthetic
   rehearsal creates no representative artifact tree and cannot license destroy.

3. **Lifecycle success is not fail-closed.** Both primary stop and destroy accept
   any JSON object returned with exit code zero. A source-only probe confirmed
   that `{"success": false}` is recorded as `PRIMARY_STOP_ACCEPTED` and
   `DESTROY_REQUEST_ACCEPTED_AFTER_DURABILITY`. No instance-state confirmation is
   required after either mutation.

4. **The fallback is a request with no tracked executor, not a stop channel.**
   Source search finds creation of the S3 `stop-guardian` marker but no guardian
   that reads it and stops the bound instance. Before authorization is loaded,
   the marker also carries `instance_id: null`. Every provider subprocess lacks
   a timeout, so a hung primary stop never reaches fallback. Crash windows around
   conditional fallback creation are not reconciled, and concurrent local state
   writers are not serialized. This is neither resilient nor idempotent.

5. **Failure coverage still begins too late in executable CLIs.** Launcher,
   controller, and worker `main` functions load/validate the capsule before
   constructing their stopper. A capsule/path/runtime failure there bypasses the
   wrapped callable. Pointer creation or prior-state parsing failure also aborts
   before either stop channel. The synthetic `pre_setup` plant starts after its
   own deterministic transport exists and does not cover these production edges.

6. **Provider and instance identity remain environment-selected or unproved.**
   `shutil.which` and bare `aws`/`vastai` names trust `PATH`; AWS profile files and
   credentials remain process-environment/config selected. Version-string and
   configured-region checks do not verify the running image digest, executable
   digest, AWS caller account/scoped policy, Vast key scope, or that the
   authorization's caller-supplied positive instance ID equals the current Vast
   container. A wrong ID can run science on one machine and stop/destroy another.

7. **The isolated production entrypoint is not yet the tracked deployment path.**
   `production_argv` is referenced only by the capsule/new modules/tests; no
   tracked launch spec, stager, or provider launcher consumes it or stages the
   capsule, fixed AWS profile, and scoped Vast secret. `materialize` validates an
   already-populated `/workspace/parallax/source`; it does not materialize that
   layout. Manually invoking it in a rehearsal would not prove the eventual paid
   launcher is forced through this path.

## Minimal remediation boundary

Filter to the registered task/splits before requiring selected-image fields;
conditionally upload and download-verify the complete science artifact tree
before destroy; require semantic Vast success plus observed terminal instance
state; implement and bind a real timeout-bounded guardian with crash/race
reconciliation; cover CLI-load and pointer failures; bind absolute hashed CLI,
AWS scoped caller identity, Vast scoped key, and current container ID; and wire
the production argv and secret/source staging into one tracked launch entrypoint.
Preserve the current authorization, legacy-byte, and science-AST improvements.

## Rehearsal authorization

**Exactly one bounded no-H100 cloud CPU staged-layout rehearsal is NOT AUTHORIZED
for `b0a9aec`.** The hydration failure, false lifecycle acceptance, absent guardian,
and omitted artifact durability are source defects that a synthetic staged layout
cannot safely adjudicate. Remediation changes runtime hashes and the capsule
digest, making a present-candidate receipt obsolete.
