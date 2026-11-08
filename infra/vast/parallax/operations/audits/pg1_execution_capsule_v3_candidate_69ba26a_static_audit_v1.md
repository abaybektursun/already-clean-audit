# PG1 v3 candidate `69ba26a` static audit

Verdict: **STATIC_NO_GO** against base `8bba643`. This was a tracked-source-only audit: no cloud, Vast, AWS, market, paid, archive/model/data/confirmation-body, authority, freeze, claim, or science-journal action was performed.

## Reproduced evidence

- `PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider infra/vast/parallax/tests/test_pg1_execution_capsule.py` -> **27 passed**.
- `PYTHONDONTWRITEBYTECODE=1 python3 -m infra.vast.parallax.pg1_science_ast_guard --root .` -> **GREEN**, 19 protected functions, 14 constants, and 2 protected normal-run slices identical to `8bba643`.
- Credential-free full legacy suite -> **698 passed, 16 failed**.
- `git diff --check 8bba643..69ba26a` -> clean. No old study spec/config/manifest/ledger/freeze or operations record is changed by the candidate commit.

The 27-test matrix does establish seven distinct deployed owner modules, one capsule digest across their receipts, three parameterized attempt identities distinct from the frozen source identity, exactly one `primary_config` at the primary cell, critic CLI binding to the attempt ID, authority-false rejection before ML/hydration, confirmation/body-read firewalls, exact production staged-root rejection, and receipt enumeration plus byte verification of every declared native runtime path. The AST proof establishes scientific-kernel/output-construction equivalence only inside its declared protected functions and slices; it does not establish whole-entrypoint executability.

## Every changed file/hunk class

| Changed file | Classification |
|---|---|
| `audit_primary_interface_independent.py` | Allowed mechanical preflight/CLI binding; registered constants, `recompute`, `_assert_producer_agreement`, and `build_audit` are AST-identical. |
| `pg1_execution_capsule.py` | Allowed capsule identity/receipt/runtime-wiring mechanics, but its `ADDITIVE_PREFLIGHT_ONLY` label overstates changes that also alter normal APIs/routing. |
| `pg1_execution_capsule_preflight.py` | Allowed dispatcher wiring to the native controller graph. |
| `pg1_science_ast_guard.py` and baseline fixture | Allowed source-only audit evidence. |
| `primary_interface_control_source_gate.py` | Allowed source/attempt and confirmation-firewall preflight. |
| `primary_interface_control_overlay_stage.py` | Allowed exact materialization-receipt and staged-byte preflight. |
| `training_interface_control_cell.py` | Mechanical preflight/identity/config routing; protected science from `lineage_rows` onward is identical. Its changed bytes nevertheless invalidate the frozen PG2 and PG1 source gates. |
| `training_interface_control_primary_cell.py` | Mechanical preflight plus explicit lineage policy; protected result/adjudication slice is identical and the old `ALLOWED` mutation is removed. Its changed bytes invalidate the frozen PG1 source gate. |
| `primary_interface_control_job.py` | **Forbidden retained adapter architecture:** normal execution still mutates legacy globals and monkey-patches legacy functions; a compatibility-alias implementation remains in the deployed module. |
| `primary_interface_control_launch.py` | **Forbidden retained adapter architecture:** normal execution still mutates the legacy controller module and replaces `_stage`; it also reaches an unchanged frozen spec whose source hashes reject this candidate. |
| `primary_interface_control_worker.py` | **Forbidden retained adapter architecture and unsafe critic wiring:** normal execution mutates legacy globals; critic execution begins only after the legacy worker has uploaded its terminal and requested instance destruction. |
| `test_pg1_execution_capsule.py` | Allowed source-only planted matrix, but it covers only authority-false/preflight paths and does not catch the legacy-global, frozen-hash, or post-terminal-destruction defects. |

## Blocking findings

1. **The explicit no-legacy-mutable-globals/no-compatibility-alias condition is false.** The controller's `_patched_legacy` uses `setattr` and replaces `legacy._stage`; the job's `_patched_legacy` mutates execution identity, paths, hashes, and expected values, then replaces `_verify_source`, `_run_tokenizer_preflight`, and `_run_control`; the worker's `_patch` mutates `legacy.RUN_ID` and `legacy.EXPECTED_COMMAND`. `legacy_compatible_source_contract` and its materializer also remain. These are the actual v3 controller/job/worker modules, not retired aliases.

2. **The authority-true normal controller is statically non-executable with the untouched frozen bindings.** After v3 preflight it calls `validate_spec`, whose current fresh spec still binds the old critic and five changed primary implementation hashes; it then validates the old PG1 source contract, whose cell hashes also differ. The full suite exposes this rather than an ignorable test update.

3. **The 16 legacy failures cannot be certified harmless.** They classify as 11 frozen-hash failures, 4 intentionally rejected no-capsule API calls, and 1 obsolete direct-call signature. The latter five are mechanical API drift. The 11 hash failures are live regressions: four normal PG1 launcher tests stop on the old critic binding, two PG1 source/overlay tests stop on changed cells, two current-gate tests stop on the changed critic, and three supporting-PG2 launch/source/overlay tests stop because the shared cell changed. Updating tests alone would conceal that normal launch/source gates reject the deployed bytes.

4. **The independent critic is not durably sequenced.** `primary_interface_control_worker.run` calls `legacy.run` first. That legacy worker writes and uploads the terminal and calls the Vast lifecycle action (`destroy` for archive-durable or pre-science cases) before returning. Only then does v3 invoke the critic. The critic can race destruction, and its output is absent from the terminal that was already made durable.

5. **Only authority-false capsules are valid today.** The capsule schema requires all authority fields false, while normal controller/job/cell/worker paths require `h100_authority is True`. This is a safe preflight stop, but no content-address-valid authorized capsule can exercise the audited normal path without another code/contract change; therefore the green matrix cannot license science execution.

## Minimal remediation boundary

1. Restore every pre-existing hash-bound module changed here to its `8bba643` bytes so the untouched PG1 and supporting-PG2 frozen contracts remain replayable. Move v3 behavior to new PG1-only native modules and bind those exact modules in the capsule; they must own execution, not wrap or mutate legacy owners.
2. In the new controller/job/worker/cell path, pass attempt identity, paths, command, source validator, tokenizer preflight, control callback, and lineage policy as explicit immutable arguments. Remove all `setattr`/module-global patching and compatibility-contract aliases from the v3 path.
3. Keep the authority-free content-addressed capsule immutable. Normal execution must require a separate exact execution-authorization object bound to that capsule digest; source-only tests must prove missing/wrong/false authorization stops before hydration and a planted valid binding reaches the native normal control path without cloud or ML work.
4. Keep the registered scientific functions and normal-run slices AST-identical. Sequence `terminal durable -> bound critic -> critic output durable -> immediate destroy`; on critic or critic-upload failure, stop and preserve scratch. Include the durable critic proof in final attempt evidence.
5. Require the focused v3 matrix, AST guard, and the entire legacy suite to pass with zero failures before reconsidering a CPU rehearsal.

## Authorization consequence

**One bounded no-H100 cloud CPU staged-layout rehearsal is NOT AUTHORIZED on this candidate.** It would only repeat the already-green preflight graph and cannot close the normal-path frozen-binding, legacy-global, or critic-durability blockers. Resolve those in tracked source and rerun the source-only matrix, AST proof, and full legacy suite before any cloud rehearsal.
