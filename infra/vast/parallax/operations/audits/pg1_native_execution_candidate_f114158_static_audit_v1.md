# PG1-native candidate `f114158` static audit

Verdict: **STATIC_NO_GO** against trusted science base `8bba643` and prior audit `4fe65b2`. This audit used tracked source only. It performed no cloud, Vast, AWS, price, paid, archive/model/data/confirmation-body, Docker, run-ID, freeze, claim, authority, or science-journal action.

## Reproduced evidence

- Native matrix: `PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider infra/vast/parallax/tests/test_pg1_execution_capsule.py` -> **40 passed**.
- Tracked PARALLAX suite: credential-free `pytest ... infra/vast/parallax/tests` -> **727 passed**.
- Restored legacy subset: the same command excluding `test_pg1_execution_capsule.py` -> **687 passed**.
- `python3 -m infra.vast.parallax.pg1_science_ast_guard --root .` -> **GREEN**: 19 registered functions, 14 constants, 2 registered run slices, copied native science slice, primary adjudication/output slice, controller trace, and worker trace match the guard.
- Independent byte comparison confirms all eight legacy/hash-bound critic, PG1, shared-cell, and primary-cell modules exactly equal `8bba643`; this restores the supporting-PG2 gates. Old study specs/configs/manifests/ledgers/freezes are absent from the candidate diff. Both candidate diffs pass `git diff --check`.

The 727 result is meaningful for all tracked tests under `infra/vast/parallax/tests` (the 40 new tests plus 687 legacy tests). It is not a repository-wide test count; the repository contains additional tracked suites outside that directory. No untracked test was included.

## Prior-blocker adjudication

| Requirement | Result |
|---|---|
| Restore eight legacy/hash-bound files and PG2 | **CLOSED** byte-for-byte. |
| Seven distinct PG1-native source owners; no legacy patching, compatibility aliases, mutable module globals, or ML/provider imports in preflight | **CLOSED for source shape/preflight**. |
| Arbitrary future attempt identity distinct from source; exact-one `primary_config` | **CLOSED** by schema and planted tests. |
| Preflight body/confirmation/ML/hydration/science firewall | **CLOSED** by guarded source test and inspection. |
| Registered scientific functions, science slice, primary output/adjudication slice, and critic math/firewall unchanged | **CLOSED conditionally on exact inputs**; ASTs and restored critic bytes agree. |
| Native materialization receipt enumerates and byte-checks all declared runtime paths | **CLOSED statically**; remote archive truth remains intentionally unrehearsed. |
| Seven owners are forced as the actual normal execution path | **OPEN — hard blocker**. |
| Digest-bound, truly one-shot execution authorization with no normal bypass/replay | **OPEN — hard blocker**. |
| Immutable exact normal-path model/data/tokenizer/policy/path inputs | **OPEN — hard blocker**. |
| Durable terminal -> critic -> durable verified audit -> destroy, with safe failure handling | **OPEN — ordering is present, enforcement is not complete**. |

## Hard blockers

1. **The capsule does not describe an executable native production path.** `consumer_execution_argv` gives the controller its preflight command, gives job/cell argv that omit most arguments their normal CLIs require, and gives worker an argv for a CLI whose `--capsule-preflight` flag is required and which exposes no normal finalizer. The controller CLI itself exposes preflight only. Thus the declared commands cannot traverse native execution.

2. **The seven owners are not forced by the callable path.** `pg1_native_controller.execute` accepts caller-supplied `run_job`, `run_cell`, `durable_sink`, `verify_durability`, `run_critic`, and `request_destroy`; job and worker likewise execute supplied callbacks. The trace guard checks callback names/order, not that capsule-bound native functions ran. The green authority test substitutes a fake cell and fake critic/durability/lifecycle providers, so it does not exercise the registered normal cell or critic. A caller can satisfy the controller without invoking the capsule-declared owners.

3. **Authorization is self-asserted and replayable.** Any nonempty `authorization_id` plus matching JSON content hashes is accepted. There is no bound issuer/trust root or tracked implementation of atomic consumption. The consumption receipt is an exact dictionary containing `one_shot_consumed: true`; the same authorization and receipt are accepted repeatedly by job, cell, worker, and critic. The only consumption implementation in the candidate is an injected callback, and the tests manufacture its success receipt. This is digest binding, not one-shot authority enforcement.

4. **Scientific inputs are explicit but not immutable or capsule-bound.** Controller/job accept caller-controlled `matrix`, `task_root`, `tokenizer_preflight`, and `lineage_policy`. Native cell binds only the source-contract and primary-config paths. Its scientific prefix uses the caller's matrix to choose the checkpoint, caller's `lineage_policy["entity_role"]` to choose the model role, caller's task root to choose the manifest/image bodies, and caller's tokenizer-preflight file; none is checked against its exact registered path/evidence. In particular, the exact frozen external processor terminal is verified as a staged artifact but is not the required source of the tokenizer result. The protected science AST is exact only after these unbound choices have already selected its inputs.

5. **Lifecycle safety depends on unbound callbacks and failure does not stop the instance.** The worker source orders terminal receipt/verification, critic, critic receipt/verification, then destroy, so `request_destroy` is textually unreachable after an exception from those calls. But upload, HEAD verification, critic, and destroy are arbitrary callbacks with no concrete capsule-bound implementation. On any failure the code raises without an explicit stop-and-preserve-scratch action, leaving immediate cost shutdown to an unspecified outer system. This does not close the prior lifecycle remediation boundary.

## Minimal remediation boundary

- Bind complete normal CLI/callable wiring for controller, job, cell, worker, critic, durability verification, stop, and destroy; normal controller code must select the capsule-bound implementations internally rather than accept substitutable execution callbacks.
- Add a trust-root-bound, atomic consume-once mechanism whose receipt cannot be replayed by downstream entrypoints.
- Put exact matrix, tokenizer evidence, task-root/manifest, lineage, fused entity role, and canonical terminal/output paths in the content-addressed contract and reject all caller drift before ML imports.
- Make every pre-destroy failure invoke a concrete stop-and-preserve-scratch path, then plant every failure edge against the actual bound providers. Retain the current AST/legacy guarantees and require all three test counts to remain green.

## Rehearsal authorization

**Exactly one bounded no-H100 cloud CPU staged-layout rehearsal is NOT AUTHORIZED for `f114158`.** Closing these source issues changes native runtime bytes and invalidates any materialization receipt produced now; the rehearsal would be knowingly obsolete and would not test the missing normal execution or lifecycle bindings.
