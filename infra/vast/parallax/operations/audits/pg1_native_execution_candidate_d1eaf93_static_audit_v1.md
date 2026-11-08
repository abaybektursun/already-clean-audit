# PG1-native candidate `d1eaf93` static audit

Verdict: **STATIC_NO_GO** against prior audit `e3e5fe7` and trusted
science/legacy base `8bba643`. This was a tracked-source-only audit. It made no
cloud, Vast, AWS, price, paid, provider, protected-body, Docker, real-ID,
freeze, claim, authority, ledger, science-journal, or science-result action.

## Reproduced evidence

- Credential-stripped focused capsule suite: **143 passed in 19.15s**.
- Credential-stripped exact tracked `infra/vast/parallax/tests` suite:
  **830 passed in 25.99s**. This is not a repository-wide count.
- The requested launch/lifecycle transition subset: **11 passed**; the named
  exact-copied-chain test: **1 passed in 5.07s**.
- AST/trace guard: **GREEN**, with 19 protected functions, 14 constants, 2
  protected run slices, and the expected controller/worker traces.
- `e3e5fe7..d1eaf93` is exactly the stated 16 tracked source/test files.
  Changed Python files pass `py_compile`; both candidate diffs pass
  `git diff --check`.
- All eight legacy/hash-bound files have the exact `8bba643` blob identities.
  There is no candidate `studies/**`, config, operations, or ledger mutation;
  unrelated untracked files were preserved.
- A direct source-only invocation under the gateway's root-working-directory,
  empty-environment shape reproduced that the onstart materializer exits during
  import with `ModuleNotFoundError: No module named 'infra'`, before its pointer
  or source download can run.

## Source adjudication

The candidate materially closes the former self-asserted-authority and ambiguous
start-copy defects. The gateway obtains canonical authority with HEAD followed by
an exact-version GET, validates checksum, encryption, length, type, ETag,
metadata, identity, and body, and consumes a conditional one-shot marker before
create (`pg1_native_launch_gateway.py:234-302,365-390,772-797`). The start commit
is published only after the start intent copy returns, and absent, present, and
inconclusive evidence classify conservatively (`:650-674,867-905`). Canonical
state transitions now have hash ancestry and CAS recovery. The source archive
installer rejects unsafe members and verifies its exact member inventory. The
old protected-science, parser, hydration, artifact, and hash-bound invariants
remain green.

Hard blockers:

1. **The real pinned-image onstart cannot bootstrap.** Gateway embeds only
   `pg1_native_onstart_materializer.py`, writes it under `/run`, and executes it
   from `/` with an empty/sanitized environment
   (`pg1_native_launch_gateway.py:419-447`). Before writing its pointer or
   downloading the exact source bundle, that standalone file imports
   `pg1_native_mechanical_rebase`; its relative import has no package and its
   fallback requires the not-yet-installed `infra` package
   (`pg1_native_onstart_materializer.py:19-22,209-218,275-292`). The direct
   probe reproduced this pre-pointer failure. Thus create can succeed and start
   commit can be published while the worker never reaches the science-disabled
   materializer/idle boundary.

2. **Deadman restart is not live after an instance-bound ACK.** The immutable
   bound ACK binds the current canonical sequence and full state digest and is
   conditionally written to one fixed key
   (`pg1_native_control_deadman.py:215-252,342-373`). On process restart,
   `bound_state_sha` resets; a later eligible canonical phase causes `_bind_ack`
   to write a different body to that same immutable key
   (`:770-798`). Exact-replay CAS rejects it, so the monitor catches and retries
   forever instead of supervising STOP/destroy. The new restart test stops at
   `CREATE_INTENT_ARMED`, before the bound-ACK path
   (`tests/test_pg1_execution_capsule.py:538-566`).

3. **The pre-science destroy fallback is unreachable and terminal ambiguity is
   under-validated.** Gateway first transitions the canonical transaction to
   `CLEANUP_STOPPING`, then, only if its STOP is unconfirmed, publishes a
   PRE_SCIENCE destroy request (`pg1_native_launch_gateway.py:744-768,942-956`).
   Deadman returns STOP immediately for `CLEANUP_STOPPING`, before it reads that
   request (`pg1_native_control_deadman.py:542-555`), so the request cannot
   advance the lifecycle to `CLEANUP_DESTROYING`. The test proves only that an
   abstract `START_COMMIT_PROVEN_ABSENT -> CLEANUP_DESTROYING` transition is
   allowed (`tests/test_pg1_execution_capsule.py:524-528`). Separately, after an
   ambiguous terminal-state publication, deadman accepts a reread based only on
   phase and receipt reference; it does not validate canonical state identity,
   ancestry, or metadata before exiting (`pg1_native_control_deadman.py:643-655`).

4. **The claimed exact copied production chain remains test choreography.** The
   deterministic fake Vast create records the onstart string but never executes
   it (`tests/fixtures/pg1_deterministic_fake_vast.py:46-55`); the test invokes
   the materializer manually after gateway success
   (`tests/test_pg1_execution_capsule.py:2013-2026`). Its component runner calls
   production-module mechanical rebase hooks, materializer replaces idle with
   the fixture runner, and idle replaces the validated production command with
   a separate legacy mechanical rehearsal
   (`tests/fixtures/pg1_exact_chain_component_runner.py:61-85`;
   `pg1_native_onstart_materializer.py:250-270`;
   `pg1_native_idle_bootstrap.py:283-299,305-348`). It never proves causal
   create-to-onstart execution or copied native worker execution. No production
   source consumes the installer's `production_invocation_argv`, which still
   contains literal offer/attempt placeholders
   (`pg1_native_control_installer.py:184-199`).

## Coverage audit

The focused total remains 143 because the candidate adds exactly 11 collected
cases while renaming 11 prior cases out of collection. The removed coverage is
the seven-case post-create failure matrix plus the prior idle validation and
three external-deadman fault/retry/destroy cases
(`tests/test_pg1_execution_capsule.py:2051-2420`). The replacement tests do not
exercise the real onstart boundary, post-bound-ACK restart, or the gateway to
deadman PRE_SCIENCE destroy path. The unchanged green count therefore does not
establish lifecycle coverage preservation.

## Rehearsal authorization

**Exactly one bounded no-H100 cloud CPU staged-layout rehearsal is NOT
AUTHORIZED for `d1eaf93`.** The source-runnable worker bootstrap fails before
its science-disabled pointer, the external deadman can lose cleanup liveness on
restart, pre-science destroy cannot converge, and the named exact-chain proof
bypasses those production boundaries. Repairs change runtime hashes and
capsule/packet identities, so a receipt for this candidate would not transfer.
