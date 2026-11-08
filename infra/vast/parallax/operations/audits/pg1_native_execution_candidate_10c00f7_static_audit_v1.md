# PG1-native candidate `10c00f7` static audit

Verdict: **STATIC_NO_GO** against prior audit `f2e4d6b` and trusted
science/legacy base `8bba643`. This was a tracked-source-only audit. It made no
cloud, provider, paid, protected-body, production-authority, freeze, ledger,
science-journal, or science-result action.

## Reproduced evidence

- Credential-stripped focused capsule suite: **164/164 passed**; the isolated
  exact causal chain: **1/1 passed in 5.05s**. Collection is exactly 165.
- Credential-stripped exact tracked `infra/vast/parallax/tests` suite:
  **852/852 passed**. This is not a repository-wide count.
- The copied top-level invoker now resolves its source tree before importing the
  installer (`pg1_native_launch_invoker.py:22-32`). Direct `env -i` help from `/`
  succeeds without `PYTHONPATH`; the named chain also invokes it from an
  arbitrary cwd with a replacement environment containing no `PYTHONPATH`
  (`tests/test_pg1_execution_capsule.py:2592-2613`).
- The named chain pre-seeds the immutable capsule, inputs, source archive,
  mechanical root contract, credentials/receipts, and tracked fake services.
  The control source, packet, worker image source, bootstrap source, entry
  contract, deadman ACK, and authority binding are absent before the one
  invoker and present after it. Fake Vast executes the emitted onstart, which
  causally reaches copied materializer, idle gate, deployment, bootstrap, and
  the explicitly allowed no-model mechanical worker receipt
  (`tests/test_pg1_execution_capsule.py:2480-2639`). There are no downstream
  manual calls or monkeypatches in this case.
- AST/trace guard: **GREEN**, with 19 protected functions, 14 constants, 2
  protected run slices, and the expected controller/worker traces. Changed
  Python files pass `py_compile`; `f2e4d6b..10c00f7` and
  `8bba643..10c00f7` pass `git diff --check`.
- The candidate diff is exactly 12 tracked implementation/test files. All eight
  legacy/hash-bound files have the exact `8bba643` blob identities. It changes
  no tracked studies, operations, readiness, config, ledger, journal, or freeze
  artifact; unrelated untracked files were preserved.
- Five net-new collected cases cover concurrent cleanup adoption, disjoint
  authorization consumers, mechanical issuer selection, and two raw-state
  corruption plants. The restored and existing canonical authority/start,
  materializer, deadman, hydration, artifact, provider, protected-science, and
  source/runtime invariants remain green in the exact tracked suite.

## Closed findings

The v3 bound ACK is now a closed envelope and binds packet, capsule, attempt,
instance/provider identity, stop context, authorization body and receipt,
mount inventory, nonce/ID, exact replay envelope, payload digest, checksum, and
storage receipt. Gateway, deployment, and idle reject v1, extras, nonce or
checksum changes, and replay drift. Production and mechanical launch
authorizations now have distinct schemas, statuses, issuers, validators, and
issuer functions; production gateway mode rejects the mechanical body and the
mechanical CLI does not call the production issuer. Pre- and post-science
destroy requests are v2 bodies embedded in the single `CLEANUP_DESTROYING`
canonical raw-state CAS rather than standalone destroy objects. Gateway and
backend rereads now verify canonical raw bytes, metadata, state identity, and
the direct legal edge. These are material improvements, but the following
blockers remain.

## Hard blockers

1. **A permanent gateway crash can still strand a billable instance with the
   deadman intentionally idle.** Gateway durably creates the stop-unconfirmed
   cleanup receipt (`pg1_native_launch_gateway.py:1240-1246`) before embedding
   its destroy body in the state CAS (`pg1_native_launch_gateway.py:1444-1453`).
   Once that receipt exists, deadman `_action_required` returns no action both
   for `CLEANUP_STOPPING` and stale pre-science states
   (`pg1_native_control_deadman.py:668-700,713-739`), then only heartbeats
   (`pg1_native_control_deadman.py:1062-1073`). The top invoker executes gateway
   once and propagates any nonzero result; it has no restart loop
   (`pg1_native_launch_invoker.py:73-85`). The crash plant itself demonstrates
   this state, asserts that deadman does nothing, and converges only after the
   test explicitly calls the production recovery entry
   (`tests/test_pg1_execution_capsule.py:1121-1134`). Thus restart adoption and
   concurrent CAS convergence work when another gateway is started, but no
   independent owner guarantees that restart or terminal control in this gap.

2. **Deadman's ambiguous CAS path still accepts parsed equality instead of the
   requested exact storage proof.** On a failed write or incomplete durability
   response, `_put` returns success whenever `existing["body"] == payload`
   (`pg1_native_control_deadman.py:229-248`). It does not compare canonical raw
   bytes/raw digest, expected metadata, expected checksum, or the prior/legal
   state edge. Because `_put` returns, `_publish_state` never reaches its
   stronger exception reread (`pg1_native_control_deadman.py:307-340`). A direct
   credential-free in-memory plant supplied the same parsed body with
   noncanonical raw bytes, a false raw digest, wrong metadata, and wrong
   checksum; `_put` accepted it as `replayed_exact=True`. This affects ordinary
   and terminal state publication and leaves the requested complete ambiguous
   CAS proof open despite the gateway/backend repairs.

3. **Terminal receipt restart adoption is not a closed, canonical-raw proof.**
   `_validate_terminal_receipt` checks selected values but not the exact field
   set (`pg1_native_control_deadman.py:750-788`); a direct in-memory plant added
   an unexpected field and validation accepted it. Existing receipt adoption
   checks parsed fields/reference and metadata but does not require observed
   raw bytes to equal the canonical payload bytes
   (`pg1_native_control_deadman.py:840-846,886-903`). This is a second recovery
   boundary at which full fields and exact bytes are not established.

4. **Worker-side production CLIs still select mechanical dispatch from caller
   contract content.** Deployment and bootstrap accept a generic
   `--entry-contract`; when its kind is mechanical, they derive the supposedly
   fixed entry path from that same contract's `rebase_root` and dispatch before
   production validation (`pg1_native_deployment.py:1025-1043`;
   `pg1_native_bootstrap.py:774-790`). Their mechanical branches bind copied
   script locations and exact argv/hashes, but neither discovers the immutable
   root contract from its own source location nor rejects `rebase_root == /`
   as `pg1_native_mechanical_rebase.py:48-50` does. Consequently the production
   deployment/bootstrap entrypoints do not themselves reject a mechanical
   entry body placed at their canonical path. The worker later rejects its
   mechanical flag in production, so this fails closed before model work, but
   it does not satisfy the requested complete production/mechanical CLI split.

## Rehearsal authorization

**Exactly one bounded no-H100 cloud CPU staged-layout rehearsal is NOT
AUTHORIZED for `10c00f7`.** The unrestarted cleanup gap can abandon lifecycle
ownership, and deadman ambiguity/terminal recovery still lack exact raw-state
proof. Repairs alter bound runtime bytes and capsule/packet identities, so a
receipt for this candidate would not transfer.
