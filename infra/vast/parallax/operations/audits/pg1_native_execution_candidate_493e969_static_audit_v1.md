# PG1-native candidate `493e969` static audit

Verdict: **STATIC_NO_GO** against prior audit `2b89bec` and trusted
science/legacy base `8bba643`. This was a tracked-source-only audit. It made no
cloud, provider, paid, protected-body, authority, freeze, ledger,
science-journal, or science-result action.

## Reproduced evidence

- Credential-stripped focused capsule suite: **160/160 passed in 18.40s**.
- Credential-stripped exact tracked `infra/vast/parallax/tests` suite:
  **847/847 passed**. This is not a repository-wide count.
- Named one-invoker emitted-chain case: **1/1 passed in 4.65s**.
- Standalone onstart import/help succeeds under an empty environment from `/`,
  with no offline/mechanical CLI option. The fake service executes the exact
  gateway-emitted onstart string.
- AST/trace guard: **GREEN**, with 19 protected functions, 14 constants, 2
  protected run slices, and the expected controller/worker traces. Existing
  changed Python files pass `py_compile`; both candidate diffs pass
  `git diff --check`.
- `2b89bec..493e969` is exactly 15 tracked source/test files. All eight
  legacy/hash-bound files have the exact `8bba643` blob identities. The
  candidate has no tracked studies, config, operations, or ledger mutation;
  unrelated untracked files were preserved.
- The 11 restored cases remain collected. Four net-new collected cases cover
  v3 ACK compatibility, CLI option inventory, resolved invocation arguments,
  stricter CAS bodies, and a partial cleanup crash plant.

The candidate fixes the standalone materializer import, removes explicit
offline/mechanical selectors and the worker probe option, resolves invocation
placeholders, makes the bound-ACK payload state-sequence-independent, accepts
v3 in deployment, and introduces one invoker that really installs and launches
compiler, authority, and gateway subprocesses. Canonical authority fetch/start
classification and the prior parser, hydration, artifact, protected-science,
and restart guards remain green. These gains do not close the following hard
blockers.

## Hard blockers

1. **The copied top-level invoker is not standalone source-runnable.** It imports
   the project package before configuring its copied source location
   (`pg1_native_launch_invoker.py:13-16,34-39`). Direct empty-environment
   execution exits before help with `ModuleNotFoundError: No module named
   'infra'`. The named chain succeeds because the test explicitly injects the
   stage source as `PYTHONPATH`
   (`tests/test_pg1_execution_capsule.py:2356-2365`). The capsule declares only
   the invoker path, hash, and mode, not this undeclared import precondition.

2. **PRE_SCIENCE cleanup still has a request-to-CAS crash window and live race.**
   Gateway durably publishes and rereads the immutable destroy request, returns
   from `_cleanup`, and only then attempts its single
   `CLEANUP_DESTROYING` CAS
   (`pg1_native_launch_gateway.py:879-890,1063-1080`). In that gap, deadman sees
   the exact request under the prior state, selects STOP, and can CAS
   `CLEANUP_STOPPING`, permanently preempting destroy
   (`pg1_native_control_deadman.py:663-677,869-879`). The crash test restarts the
   private `_cleanup` helper and then manually constructs the decisive state;
   no production restart path performs that recovery
   (`tests/test_pg1_execution_capsule.py:1027-1045`). A real gateway restart
   begins launch/one-shot consumption again rather than resuming this boundary.
   The immutable cleanup receipt is also based on live STOP observations, so a
   later replay need not reproduce its original body.

3. **Production ACK v3 validation is compatible but not exact.** Deployment now
   rejects v1 and the three removed state-derived fields, but it does not bind
   `packet_sha256`, require the durable checksum/replay envelope, or reject
   arbitrary extra fields (`pg1_native_deployment.py:236-270`). The new test
   deliberately supplies an unrelated packet digest and omits checksum/replay
   fields while expecting acceptance
   (`tests/test_pg1_execution_capsule.py:639-668`). Thus the exact stable v3
   envelope is not independently established at the production deployment
   boundary.

4. **Production rejection of mechanical authority and dispatch is not enforced.**
   Explicitly named test selectors are gone and source-location derivation
   correctly returns production mode for root `/`. However, copied authority
   rebases its paths and then calls the ordinary issuer, producing
   `ISSUED_IMMUTABLE_BY_CANONICAL_PRODUCTION_AUTHORITY` even though the
   mechanical contract says it may not mint production authority
   (`pg1_native_launch_authority.py:25-27,73-107,227-271`;
   `pg1_native_mechanical_rebase.py:28-47`). Deployment and bootstrap also accept
   a caller-selected generic entry-contract path and branch to mechanical
   dispatch before enforcing the fixed production path/argv
   (`pg1_native_deployment.py:823-948`;
   `pg1_native_bootstrap.py:55-96,774-784`). The option-name inventory test does
   not exercise these content-selected modes.

5. **The green one-invoker case still bypasses downstream production behavior.**
   The controller half is now causal and has no module monkeypatches: one copied
   invoker installs and executes compiler, issuer, and gateway. Emitted onstart
   also reaches materializer and idle. But the test manually pre-stages the
   future-freeze runtime and fabricates the entry/service layout before invoking
   that entry (`tests/test_pg1_execution_capsule.py:2305-2354`). Gateway obtains
   the entry path from a separate mutable service JSON rather than deriving the
   whole layout only from the immutable mechanical contract and copied source
   (`pg1_native_launch_gateway.py:1125-1139`). Deployment and bootstrap then take
   mechanical receipt branches, while worker intercepts normal-looking
   preflight argv to emit an execution receipt without running its production
   preflight/lifecycle (`pg1_native_deployment.py:844-948`;
   `pg1_native_bootstrap.py:75-150,774-784`;
   `pg1_native_worker.py:75-105,416-440`). The named case therefore cannot expose
   failures in the real provisioner, deployment verifier, bootstrap lifecycle,
   or worker path.

6. **Ambiguous CAS checks are semantically stricter but not complete raw-byte
   proofs.** Gateway and deadman now require the parsed winner to equal the exact
   attempted canonical object, and terminal state publication uses that common
   path (`pg1_native_launch_gateway.py:353-390`;
   `pg1_native_control_deadman.py:294-329,817-833`). Gateway nevertheless
   reserializes parsed JSON and never binds the observed raw-object checksum to
   the expected canonical bytes, so a semantically identical noncanonical body
   can pass. Its ambiguity helper receives only the attempted state and prior
   ETag, so its ancestry check compares that object to itself rather than proving
   a one-step legal transition from the prior canonical body. This falls short
   of the requested complete bytes/digest/metadata/ancestry/legal-transition
   proof.

## Rehearsal authorization

**Exactly one bounded no-H100 cloud CPU staged-layout rehearsal is NOT
AUTHORIZED for `493e969`.** The declared top-level invoker is not standalone,
pre-science cleanup can lose its destroy transition, production/mechanical
authority remains conflated, and the green chain still bypasses downstream
production boundaries. Repairs change runtime hashes and capsule/packet
identities, so a receipt for this candidate would not transfer.
