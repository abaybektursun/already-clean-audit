# PG1-native candidate `8ef2c8f` static audit

Verdict: **STATIC_NO_GO** against prior audit `0599d32` and trusted
science/legacy base `8bba643`. This was a tracked-source-only audit. It made no
cloud, Vast, AWS, price, paid, provider, protected-body, Docker, real-ID,
freeze, claim, authority, ledger, science-journal, or science-result action.

## Reproduced evidence

- Credential-stripped focused capsule suite: **143 passed**.
- Credential-stripped exact tracked `infra/vast/parallax/tests` suite:
  **830 passed**. This is not a repository-wide count.
- The requested launch/lifecycle transition subset: **11 passed**; the named
  exact-copied-chain test: **1 passed**.
- AST/trace guard: **GREEN**, with 19 protected functions, 14 constants, 2
  protected run slices, and the expected controller/worker traces.
- `0599d32..8ef2c8f` is exactly the stated 11 tracked source/test files.
  Changed Python files pass `py_compile`; both candidate diffs pass
  `git diff --check`.
- All eight legacy/hash-bound files have the exact `8bba643` blob identities.
  There is no candidate `studies/**`, config, operations, or ledger mutation;
  unrelated untracked files were preserved.
- Two direct source-only adversarial probes reproduced that a locally fabricated
  authorization with `offline-version` issuance is accepted and that a
  PRE_SCIENCE destroy request is selected even when the current transaction says
  `science_released: true`.

## Source adjudication

The candidate closes important prior defects. The compiler now binds an explicit
positive numeric offer into a canonical packet and unique label. The gateway
uses the central tracked create/show parser, requires the realistic successful
create response and scoped instance key, binds instance/machine/container and
fixed handoff material, and puts the start lease last. The idle entry is
stdlib-only and science-disabled before a bound lease. The worker now emits a
durable external destroy request instead of destroying and polling its own
instance. The control deadman retries ordinary exceptions, exercises CLI and
REST channels independently, and requires three STOP observations or three
destroy absences. Local lifecycle ancestry and conflict installation close the
prior local/durable CAS defect. Secrets remain out of argv and receipts, and the
prior parser, hydration, artifact, provider, protected-science, and
production/mechanical separation invariants remain intact.

Hard blockers:

1. **Launch authority is still self-asserted, not canonical.** Gateway validation
   trusts a local `0400` JSON's issuer/status and merely nonempty issuance fields
   (`pg1_native_launch_gateway.py:180-213`). It never downloads the claimed
   `issued_key` and version or compares exact body, checksum, ETag, encryption,
   and metadata. Conditional creation of a `.consumed` marker is genuinely
   one-shot and before create, but proves only marker uniqueness, not that the
   production issuer authorized the launch (`:251-270,537-559`). No tracked
   source issues this authorization schema. The reproduced offline fabricated
   authorization therefore licenses the paid create path.

2. **The start-release boundary can misclassify live science as pre-science.**
   The start lease is copied directly to its final worker path. `_run` can time
   out after the remote side effect, while `science_released` becomes true only
   after a successful local return (`pg1_native_launch_gateway.py:132-140,
   620-627`). The exception path then invokes cleanup with false, which may emit
   `PRE_SCIENCE_ZERO_BODY_DURABILITY` and a destroy request (`:636-648,472-533`),
   even though the idle worker may already have observed the lease and started
   production. The deadman validates only the request's false flag; it does not
   reject that gate when the current transaction is released
   (`pg1_native_control_deadman.py:369-405`). The direct adversarial probe
   selected destroy under exactly that contradictory state. This can destroy
   post-release scratch without complete-artifact durability.

3. **The advertised fixed control/worker layout is not source-runnable.** The
   new idle bootstrap must already exist inside the immutable worker image before
   onstart, but the pinned image digest is unchanged from `0599d32` and the new
   file was added only in this candidate. The packet's source bundle is validated
   and copied into a layout receipt but is never downloaded, extracted, or
   installed; the create command supplies no source mount
   (`pg1_native_launch_packet.py:113-142,219-241`;
   `pg1_native_launch_gateway.py:310-318,410-449`). Likewise no tracked
   provisioner establishes or hash-verifies the compiler, gateway, deadman,
   binaries, authorization, and inputs under `/opt/parallax/control`. Handoff
   uses one-shot `vast copy` calls without a readiness loop, while the declared
   bind timeout is unused (`pg1_native_launch_gateway.py:59,330-331,452-456`).
   Static source therefore cannot execute the claimed fixed-layout transaction.

4. **The external deadman is not restart-persistent and terminal publication
   does not converge.** Every invocation first loops on `arm`, which accepts only
   `PRELAUNCH_CLEANUP_INTENT`; a restart after the transaction advances retries
   forever and never resumes monitoring
   (`pg1_native_control_deadman.py:237-257,527-560`). Gateway does not supervise
   the process after its ACK. Terminal receipt and destroyed-lifecycle event are
   two independent fixed-key conditional writes; the event is always sequence 1
   with no ancestry to the worker lifecycle (`:412-456`). If the first write
   succeeds and the second fails, retry changes observations/errors, exact-replay
   CAS fails, and no startup reconciliation exists. The separate external event
   also never installs into or descends from the worker's local/durable lifecycle,
   so `TERMINAL_DESTROYED_DURABLE` does not establish one converged, no-split-brain
   lifecycle.

5. **The claimed exact copied production chain is not executed.** The rehearsal
   copies/hash-checks the runtime inventory but executes only the rebased
   deployment, which executes the mechanical bootstrap receipt path; it never
   invokes the copied compiler, gateway, idle entry, or control deadman
   (`pg1_native_rehearsal.py:491-567`;
   `pg1_native_deployment.py:834-898`). The named test copies only compiler,
   gateway, idle, deployment, and bootstrap, omits the copied deadman, mocks
   gateway deadman/provider/context/handoff operations, mocks copied idle `run`,
   and runs deployment/bootstrap mechanically as a separate action
   (`tests/test_pg1_execution_capsule.py:1494-1642`). Its green result is not an
   exact six-component production transaction proof.

## Rehearsal authorization

**Exactly one bounded no-H100 cloud CPU staged-layout rehearsal is NOT
AUTHORIZED for `8ef2c8f`.** It would enter a paid create through unauthenticated
local authority evidence and an unproven fixed image/control layout, while the
ambiguous start boundary can authorize pre-science destroy after release and the
deadman cannot prove persistent, converged terminal cleanup. Repair changes
runtime hashes and the capsule/packet digests, so a receipt for this candidate
would not transfer.
