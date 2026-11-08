# PG1-native candidate `83bfff1` static audit

Verdict: **STATIC_NO_GO** against prior NO_GO `bc654ed` and trusted science base
`8bba643`. This was a tracked-source-only audit. It made no cloud, Vast, AWS,
price, paid, archive/model/data/confirmation-body, Docker, run-ID, freeze, claim,
authority, or science-journal action.

## Reproduced evidence

- Focused capsule/native suite: **74 passed**.
- Credential-stripped tracked PARALLAX suite: **761 passed**. This is the complete
  `infra/vast/parallax/tests` suite, not a repository-wide count.
- AST guard: **GREEN**, with 19 protected functions, 14 constants, 2 protected
  run slices, the expected seven-owner controller trace, and the expected
  terminal/critic/durability worker trace.
- All eight legacy/hash-bound critic, PG1, shared-cell, and primary-cell files are
  byte-identical to `8bba643`; the candidate has no tracked `studies/**` diff.
  Both `bc654ed..83bfff1` and `8bba643..83bfff1` pass `git diff --check`.
- Installed CLI help confirms AWS CLI 2.33.2 accepts the conditional/checksum/SSE,
  profile, and region flags, and that the declared Vast stop/destroy grammar is
  valid. No provider call was made.

## Prior-blocker adjudication

| Requirement | Result |
|---|---|
| Complete declared argv and seven fixed native owners without callback injection | **CLOSED in the isolated modules**; controller/job/cell/worker/critic select fixed implementations. |
| Production cannot select rehearsal | **PARTIAL**; `--execute` does not import the rehearsal harness, but production validators and `load_issued` still accept the rehearsal issuer. |
| Concrete conditional issuer and race-safe atomic consume | **PARTIAL**; fixed S3 conditional issue/base/owner markers are structurally race-safe, but the consumption receipt is not bound to its canonical marker. |
| Capsule-bound matrix/config/task/tokenizer/entity/model/paths actually consumed | **PARTIAL**; the declared metadata is fixed and most paths are consumed, but model-cache and task-image bodies remain unverified. |
| Normal path requires consumed authority; preflight cannot grant it | **PARTIAL**; preflight is authority-free and the controller consumes, but a fabricated receipt can pass schema validation and satisfy downstream admission with an unrelated S3 object. |
| Concrete terminal -> critic -> verified audit -> lifecycle action | **PARTIAL**; source order is correct inside the worker, but durability proof and failure-stop coverage are incomplete. |
| Materialization/runtime path is production-executable | **OPEN — hard blocker**. |
| Legacy/PG2/frozen science bindings remain exact | **CLOSED** by bytes, AST guard, and the tracked PARALLAX suite. |

## Hard blockers

1. **The new normal controller is not connected to any production launcher or
   materializer.** Repository search finds `normal_controller_argv` only in the
   capsule/tests/new native modules. The frozen v5 PG1 launch still starts
   `/opt/parallax/primary_interface_control_job.py`; nothing stages the capsule,
   native runtime, materialization receipt, or authority files at their canonical
   paths and then starts the new controller. A declared argv is therefore not an
   executable Vast path.

2. **The authorization receipt remains self-assertable.**
   `validate_authorization_receipt` requires only nonempty caller-supplied
   `consume_key`, `consume_version_id`, and `consume_etag`; it never requires the
   canonical `ProductionAuthorizationStore.consumed_key`. `consume_owner` then
   HEADs that supplied object identity without reading or validating the marker
   body. An audit probe confirmed that a receipt naming an arbitrary unrelated
   key/version/etag passes the production schema validator. Production validation
   also accepts the rehearsal issuer. The conditional writes themselves are
   sound, but they do not establish that this authorization was the marker
   consumed.

3. **The production transport is neither candidate-bound nor fail-closed.** S3
   commands omit the mandatory literal `--profile fuelos` and explicit region;
   Vast stop/destroy omit a bound scoped `--api-key`. No tracked launch stages
   those credentials or proves the fixed worker image contains a usable AWS CLI.
   The dependency lock does not deliver either CLI. Vast command grammar is valid,
   but command availability and authority on the worker are not established.

4. **“Durable and verified” is weaker than the frozen storage contract.** Result
   `put-object` is unconditional and omits collision exclusion, SHA-256 checksum,
   and SSE. HEAD verification compares only VersionId and ETag; it does not verify
   checksum, length, encryption, or the locally computed source SHA-256. Thus an
   existing attempt key may be superseded and byte-integrity/encryption are not
   proven before destroy.

5. **Stop is not reached on every failure and has no resilient fallback.** In the
   controller, source validation, authorization load, and backend construction
   precede its `try`, while `finalize_attempt` is outside that `try`. In the worker,
   validation, owner consumption, path setup, and backend construction precede its
   `try`. Any such failure bypasses stop. Stop itself is attempted once; if the CLI
   or key is absent, no retry, remote guardian, or second lifecycle channel acts.
   The four rehearsal plants exercise only a synthetic transport after setup and
   therefore do not close these production edges.

6. **The exact scientific bodies used at runtime are not fully verified.** The
   model hydration manifest is hash-checked as a file but never parsed or used to
   verify the resolved snapshot's 12 files; `resolve_snapshot` checks only the
   revision directory and `config.json`. The discovery manifest is exact, but the
   selected image files are opened without checking each row's registered
   `image_sha256`/`image_bytes`. Exact paths and manifests do not by themselves
   exclude model/data body drift.

## Minimal remediation boundary

Bind the new controller into one concrete staged launch; canonicalize and
body-verify the consumption marker while requiring the production issuer; use a
candidate-bound, scoped provider transport with mandatory AWS profile/region and
resilient lifecycle fallback; require collision-safe SHA-256/SSE durability; move
all failure-capable setup under stop coverage; and verify exact model-cache and
selected task-image bytes before ML execution. Preserve the current legacy and
science AST guarantees.

## Rehearsal authorization

**Exactly one bounded no-H100 cloud CPU staged-layout rehearsal is NOT AUTHORIZED
for `83bfff1`.** Closing the source blockers changes runtime hashes and the capsule
digest, making a receipt from the present candidate knowingly obsolete.
