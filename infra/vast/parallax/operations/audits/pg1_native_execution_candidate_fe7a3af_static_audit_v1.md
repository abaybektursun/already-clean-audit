# PG1-native candidate `fe7a3af` static audit

Verdict: **STATIC_NO_GO** against prior audit `77f6d23` and trusted
science/legacy base `8bba643`. This was a tracked-source-only audit. It made no
cloud, Vast, AWS, price, paid, provider, archive/model/data/confirmation-body,
Docker, real-ID, freeze, claim, authority, science-journal, or science-result
action.

## Reproduced evidence

- Credential-stripped focused capsule suite: **130 passed**.
- Credential-stripped exact tracked `infra/vast/parallax/tests` suite:
  **817 passed**. This is not a repository-wide count.
- AST/trace guard: **GREEN**, with 19 protected functions, 14 constants, 2
  protected run slices, and the expected controller/worker traces.
- `77f6d23..fe7a3af` is exactly the stated 11 tracked source/test files.
  Changed Python files pass `py_compile`; both candidate diffs pass
  `git diff --check`.
- All eight legacy/hash-bound files have the exact `8bba643` blob identities.
  There is no tracked `studies/**`, config, or pre-existing operations mutation;
  unrelated untracked files were preserved.

## Source adjudication

The candidate genuinely closes several prior defects. The mechanical path now
copies, verifies, and executes the exact rebased deployment and bootstrap bytes.
The local lifecycle uses `fcntl`, expected-sequence CAS, digests, atomic replace,
and fsync. The local guardian remains STOP-only and both it and the provider now
require three consecutive canonical STOP observations. Existing parser,
provider-identity, selected-body hydration, artifact-durability, protected
science, and production/mechanical separation checks remain intact.

`TERMINAL_STOPPED_PRESERVED` now has the intended repository meaning when the
local path validly produces it: three consecutive observations of actual
`stopped`/`exited`/`offline`, which is the established nonbillable STOP state
with instance scratch retained. The complete system does not yet prove that
state for every failure path.

Hard blockers:

1. **The new control-plane gateway is not an executable future launch.** It does
   construct the capsule's exact `production_argv` under the fixed worker
   cwd/environment after an arm ACK, but packet validation requires the literal
   offer string `${PG1_VAST_OFFER_ID}` and passes it directly in a list argv, so
   no shell or environment expansion occurs
   (`pg1_native_launch_gateway.py:162-200`). No tracked source produces the
   fixed launch packet or consumes a one-shot launch authorization; the only
   packet status accepted is `FUTURE_FRESH_NOT_AUTHORITY`. The gateway also uses
   an otherwise unused `--api-key-file` form and accepts `new_contract` without
   the repository's established `success is True` plus per-instance-key proof.

2. **A created worker cannot reach the declared fixed layout.** The create
   command supplies only image/disk/SSH/onstart. Its onstart writes only the
   entry contract and pre-create arm receipt before invoking production
   (`pg1_native_launch_gateway.py:114-140,190-200`), while deployment requires
   the capsule, instance-bound stop context, AWS config/credentials,
   per-instance Vast key, and image receipt as fixed `0400` mounts
   (`pg1_native_deployment.py:36-71`). The stop context and scoped key depend on
   the create response, but the gateway neither consumes `instance_api_key` nor
   stages post-create material. Therefore the advertised control plane cannot
   boot its production entry contract.

3. **The outside-worker deadman is neither orphan-safe nor persistent.** The
   gateway records the instance ID only after create returns and parses cleanly
   (`pg1_native_launch_gateway.py:235-270`). A timeout, malformed response, or
   crash after provider creation leaves an unlabeled paid worker while the
   armed deadman sees `instance_id: null` forever. A stale same-attempt local arm
   ACK may also be accepted before a newly spawned deadman completes its S3 arm.
   The deadman implements only CLI STOP (the REST module is imported but never
   used), lets file/S3/parser/timeout/CAS exceptions terminate the process,
   uses a reduced provider parser, and treats conditional STOP-intent conflicts
   as fatal (`pg1_native_control_deadman.py:164-215,218-271,301-357`). Its three
   terminal observations are returned to stdout, which the gateway discards;
   no durable terminal receipt is written.

4. **Lifecycle reconciliation can remain split-brain.** `reconcile` compares
   sequence and self digest but does not prove the winning state's
   `previous_sha256` ancestry or a legal transition from the loser
   (`pg1_native_lifecycle.py:195-209`). When an S3 conflict selects the durable
   winner, deployment records and returns it without replacing the losing local
   lifecycle, and callers ignore the returned winner
   (`pg1_native_deployment.py:408-430`). Thus local and durable state need not
   converge, and a higher nonterminal state can supersede a lower failure state
   without ancestry proof.

5. **Production destroy cannot reliably establish the terminal lifecycle.**
   The worker asks the provider to destroy the same instance it is running on,
   then expects to survive three absence polls and a local CAS. That CAS uses a
   sequence read before the provider mutation and does not retry a legal
   transition after concurrent advancement
   (`pg1_native_backend.py:625-660`). It also does not publish the destroyed
   lifecycle to the durable lease. Meanwhile the enclosing bootstrap/deployment
   success path always waits for a guardian STOP receipt and then writes
   `TERMINAL_STOPPED_PRESERVED` (`pg1_native_deployment.py:783-795`); the
   STOP-only guardian correctly refuses to reinterpret destroyed absence as
   STOP. The normal destroy path therefore cannot coherently reach one atomic
   local-and-durable terminal state.

6. **Outer setup still relies on the deficient external deadman.** Deployment
   imports lifecycle, parses the entry contract, and loads context/capsule
   before its protected failure block (`pg1_native_deployment.py:28-31,883-919`);
   bootstrap similarly parses its entry contract before its protected block
   (`pg1_native_bootstrap.py:725-743`). Those failures have no local
   attempt-bound guardian state. Because blockers 1-3 prevent the independent
   external envelope from being reliably armed and persistent, setup is not
   fail-closed end to end.

## Rehearsal authorization

**Exactly one bounded no-H100 cloud CPU staged-layout rehearsal is NOT
AUTHORIZED for `fe7a3af`.** The mechanical proof is now representative, but a
rehearsal cannot repair the non-executable launch packet/layout handoff, orphan
window, nonpersistent external deadman, reconciliation gap, or incoherent
destroy terminal. Repair changes runtime hashes and the capsule digest, so a
receipt for this candidate would not transfer.
