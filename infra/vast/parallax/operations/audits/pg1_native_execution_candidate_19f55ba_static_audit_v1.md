# PG1-native candidate `19f55ba` static audit

Verdict: **STATIC_NO_GO** against prior audit `9351deb` and trusted
science/legacy base `8bba643`. This was a tracked-source-only audit. It made no
cloud, provider, paid, protected-body, authority, freeze, ledger,
science-journal, or science-result action.

## Reproduced evidence

- Credential-stripped focused capsule suite: **156/156 passed**.
- Credential-stripped exact tracked `infra/vast/parallax/tests` suite:
  **843/843 passed**. This is not a repository-wide count.
- The named exact emitted-chain case: **1 passed in 4.68s**.
- All 11 formerly displaced cases are collected and pass; the two fresh
  restart/destroy cases also pass. The 13-case increase from 143 to 156 is real
  restored/additional coverage rather than another replacement.
- The standalone onstart materializer imports and renders help successfully
  under the root-working-directory, empty-environment shape. Its pre-download
  imports are stdlib-only, and the offline fake executes the gateway-emitted
  onstart string.
- AST/trace guard: **GREEN**, with 19 protected functions, 14 constants, 2
  protected run slices, and the expected controller/worker traces. Changed
  Python files pass `py_compile`; both candidate diffs pass `git diff --check`.
- `9351deb..19f55ba` is exactly 12 tracked source/test files. All eight
  legacy/hash-bound files have the exact `8bba643` blob identities. The
  candidate has no tracked studies, config, operations, or ledger mutation;
  unrelated untracked files were preserved.

The canonical authority exact-version/body validation, conservative start
classification, sequence-independent bound-ACK body, protected-science guards,
parser/hydration/artifact gates, and restored legacy coverage are materially
improved or remain sound. They do not close the following production blockers.

## Hard blockers

1. **The real worker rejects the new bound ACK.** Control deadman emits the
   sequence-independent `pg1-native-instance-bound-deadman-ack/v3`, gateway
   copies that exact envelope, and idle accepts v3
   (`pg1_native_control_deadman.py:343-375`;
   `pg1_native_launch_gateway.py:596-616`;
   `pg1_native_idle_bootstrap.py:232-239`). Production deployment still requires
   v1 and the removed `transaction_sha256`
   (`pg1_native_deployment.py:236-267`). Therefore the real emitted chain fails
   closed before production provisioning even though the v3 body itself is now
   restart-idempotent.

2. **PRE_SCIENCE destroy is reachable only without a crash in a non-atomic
   gap.** Gateway durably publishes the destroy request and then separately
   advances `CLEANUP_STOPPING` to `CLEANUP_DESTROYING`
   (`pg1_native_launch_gateway.py:1024-1039`). If it exits between those writes,
   deadman returns STOP immediately for `CLEANUP_STOPPING` before reading the
   request (`pg1_native_control_deadman.py:630-647`), and terminal STOP cannot
   later advance to destroy. The fresh test manually performs the second
   transition and does not exercise restart in this gap
   (`tests/test_pg1_execution_capsule.py:832-847`). Contradictory, released, and
   unknown evidence is safely stop-only, but pre-science cleanup is not
   crash-convergent.

3. **Ambiguous terminal-state reread is not a full canonical-child check.** It
   validates structure, ancestry, metadata, receipt, and only a selected field
   subset (`pg1_native_control_deadman.py:725-756`). Canonical transition accepts
   arbitrary field overrides, while state validation does not constrain every
   carried field (`pg1_native_launch_state.py:101-189,209-249`). A direct child
   can therefore change omitted identity/evidence fields and still pass the
   ambiguous reread. The new test plants bad metadata only
   (`tests/test_pg1_execution_capsule.py:731-742`), not an otherwise valid child
   with altered omitted fields.

4. **Production entrypoints expose test-only rebasing and probe modes.** The
   authority, gateway, deadman, installer, materializer, and idle CLIs accept an
   offline-root selector, while worker accepts a mechanical exact-probe mode
   (`pg1_native_launch_authority.py:250-270`;
   `pg1_native_launch_gateway.py:1091-1103`;
   `pg1_native_control_deadman.py:1033-1044`;
   `pg1_native_control_installer.py:255-282`;
   `pg1_native_onstart_materializer.py:296-316`;
   `pg1_native_idle_bootstrap.py:377-395`;
   `pg1_native_worker.py:401-413`). Checking only that the former selector name
   is absent does not prove production rejection of fake/mechanical entry.

5. **The green exact-chain case is not an exact production transaction.** The
   test manually invokes installer, obtains and rebases the command list, adds
   offline selectors, and runs compiler, issuer, and gateway separately; it
   also pre-stages later runtime roots
   (`tests/test_pg1_execution_capsule.py:2110-2202`). Emitted onstart now runs,
   but deployment selects its mechanical dispatch instead of the production
   verifier/provisioner, bootstrap selects its mechanical branch, and worker
   executes a receipt probe rather than its production lifecycle
   (`pg1_native_deployment.py:841-976`;
   `pg1_native_bootstrap.py:75-148`;
   `pg1_native_worker.py:51-91`). This bypasses the v1/v3 failure and does not
   prove the requested copied production boundaries.

6. **The production installer still has no causal invoker.** Its helper now
   resolves offer and attempt values, but no production source consumes it; the
   installer installs and exits (`pg1_native_control_installer.py:209-282`).
   Only the test calls the helper through ad hoc orchestration
   (`tests/test_pg1_execution_capsule.py:2170-2202`). Thus source does not provide
   one production entry that causally executes install, compile, issue, and
   gateway launch.

## Rehearsal authorization

**Exactly one bounded no-H100 cloud CPU staged-layout rehearsal is NOT
AUTHORIZED for `19f55ba`.** The production worker deterministically rejects its
bound ACK, cleanup and terminal convergence remain incomplete, and the green
chain relies on production-exposed mechanical branches rather than the real
production transaction. Repairs change runtime hashes and capsule/packet
identities, so a receipt for this candidate would not transfer.
