# PG1-native candidate `c48e301` static audit

Verdict: **STATIC_NO_GO** against prior NO_GO `2a46b76` and trusted
science/legacy base `8bba643`. This was a tracked-source-only audit. It made no
cloud, Vast, AWS, price, paid, provider, archive/model/data/confirmation-body,
Docker, real-ID, freeze, claim, authority, science-journal, or science-result
action.

## Reproduced evidence

- Credential-stripped focused capsule suite: **125 passed**.
- Credential-stripped exact tracked `infra/vast/parallax/tests` suite:
  **812 passed**. This is not a repository-wide count.
- AST/trace guard: **GREEN**, with 19 protected functions, 14 constants, 2
  protected run slices, and the expected controller/worker traces.
- `2a46b76..c48e301` is exactly the stated eight tracked files. Changed Python
  files pass `py_compile`; both candidate diffs pass `git diff --check`.
- All eight legacy/hash-bound files have the exact `8bba643` blob identities;
  the candidate has no tracked `studies/**` diff and leaves PG2/legacy tests
  green.

## Source adjudication

The prior destructive-guardian defect is closed: the guardian and deadman are
STOP-only, failed STOPs never escalate to DESTROY, local
`FAILURE_STOP_REQUIRED` is checked independently of S3, and the only PG1-native
destroy mutation is behind `ProductionBackend.request_destroy`. The fixed Vast
parser still binds exact `id` plus `actual_status`/`cur_state`; provider
identity, selected-body hydration, complete artifact inventory, exact-byte
download verification, and the final non-circular receipt path remain intact.

Hard blockers:

1. **No tracked provisioner executes the production entry contract.** Source
   search finds `production_argv` only in capsule construction/validation,
   launcher reporting, and the rehearsal. No tracked `Popen`/`run` or instance
   provisioner consumes it or establishes the declared outer cwd/environment.
   `pg1_native_deployment` does consume the exact `bootstrap_argv` once it is
   already running, but how the future instance starts its absolute entry is
   still outside the audited source.

2. **The mechanical analogue does not execute path-rebased production bytes.**
   It copies the real deployment file, then creates and executes a different
   generated stub at the rebased argv path (`pg1_native_rehearsal.py:521-589`).
   Its separate production/stub hashes prove non-equivalence; `bootstrap_argv`
   is merely recorded as `invoked_argv` and is not executed. The green plant
   therefore proves argv plumbing for the stub, not the outer entry, layout,
   imports, deadman, or bootstrap boundary that a CPU rehearsal would use.

3. **The advertised lifecycle is neither complete nor concurrency-monotonic.**
   `ProductionBackend.request_destroy` returns a verified destroy receipt but
   never advances `TERMINAL_DESTROYED_DURABLE`; no production call to that
   transition exists. Conversely, deployment always waits for a STOP receipt
   and writes `TERMINAL_STOPPED_PRESERVED` after bootstrap. Deployment,
   bootstrap, stopper, guardian, and deadman also independently read, increment,
   and atomically replace the same pointer without a shared lock/CAS. Two actors
   can derive the same sequence and a later failure write can overwrite a
   terminal write, so the serial plant does not prove the multiprocess chain.

4. **The claimed external fail-closed envelope still has uncovered setup and a
   common-mode stop dependency.** Deployment loads context and capsule and
   writes `ARMED` before entering its protected `try`
   (`pg1_native_deployment.py:751-758`), so those setup failures create no
   attempt-bound failure pointer, marker, or supervised deadman. The “external”
   deadman is a process spawned inside the target instance, not a tracked
   outside-instance lease consumer. On action it must import
   `pg1_native_provider` and load the key before either CLI STOP or alternate
   REST STOP is reachable (`pg1_native_deadman.py:509-558`); an import/layout
   failure disables both channels.

5. **`TERMINAL_STOPPED_PRESERVED` overclaims the provider proof.** Repository
   semantics do establish that actual `stopped`/`exited`/`offline` is
   nonbillable and STOP preserves the instance disk, but the established
   enforcement requires three consecutive observations
   (`ephemeral_s3_launch.py:671-711`). Guardian and deadman promote one row to a
   durable terminal receipt (`pg1_native_guardian.py:564-595`,
   `pg1_native_deadman.py:511-528`). Thus the state has the intended meaning but
   not the repository's established proof strength for actual nonbilling and
   retained scratch.

## Rehearsal authorization

**Exactly one bounded no-H100 cloud CPU staged-layout rehearsal is NOT
AUTHORIZED for `c48e301`.** It would require an untracked outer provisioner or
would execute the non-equivalent stub, while the real lifecycle/deadman and
terminal-state gaps remain source defects. Repair changes runtime hashes and
the capsule digest, so a receipt for this candidate would not transfer.
