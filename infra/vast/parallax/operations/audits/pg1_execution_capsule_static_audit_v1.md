# PG1 execution-capsule static audit v1

- Created: `2026-07-19T06:50:22Z`
- Candidate commit: `c39c67906da28ef3b810bbbe3281fb7bbc0a1f6c`
- Prior architecture audit: `infra/vast/parallax/operations/audits/pg1_equal_token_interface_pre_science_contract_system_audit_v1.md`
- Review mode: small committed text only; no tests, archives, checksums, dependencies, model/data bodies, cloud calls, market calls, or paid actions
- Verdict: **STATIC_NO_GO**
- Bounded cloud CPU plants: **NOT_AUTHORIZED**

## Outcome

The candidate is a useful contract scaffold, but it does not yet implement the architecture required by the prior audit. It creates one strict content-addressed capsule, separates source and attempt identities, removes legacy-global mutation from the new files, and binds confirmation absence and critic bytes. However, its six named “consumers” are six branches in one new synthetic preflight module. The dispatcher does not reach the actual source-stage, overlay, job, worker, controller, or critic programs that would execute the future attempt.

Consequently, this candidate can prove that its own centralized model is internally consistent. It cannot yet prove that the deployed producer-to-consumer graph accepts that model. That is the precise class of missing proof that allowed both historical H100 failures.

## Requirement adjudication

| Requirement | Static result | Evidence and finding |
|---|---|---|
| One canonical capsule schema | **PASS for the candidate scaffold** | `pg1_execution_capsule.py:23-24,135-149,451-704` defines one exact envelope, content address, top-level field set, and strict contract validator. |
| Explicit `source_run_id` versus `attempt_run_id` | **PASS** | `pg1_execution_capsule.py:25-26,345-376,486-527` binds distinct exact values and rejects equality or drift. |
| No mutable legacy globals or monkey patches in the new path | **PASS** | Neither new runtime module imports the retired primary/ephemeral adapter or mutates module globals. The path passes an explicit capsule to subprocesses. |
| Every actual consumer reads the same compiled contract | **FAIL** | All six consumer records bind the same module, `infra.vast.parallax.pg1_execution_capsule_preflight`, at `pg1_execution_capsule.py:325-344`. The dispatcher invokes that same module six times at `pg1_execution_capsule_preflight.py:77-139`; it never invokes six actual consumer implementations. |
| Exact staged-layout dispatcher reaches source gate, overlay, job, worker, controller, and critic | **FAIL** | The labels map to `preflight_source_gate`, `preflight_overlay_stage`, `preflight_job`, `preflight_worker`, `preflight_controller`, and `preflight_critic` inside `pg1_execution_capsule.py:855-988`. They are centralized simulations, not entrypoints owned by the programs they purport to check. |
| Exact source-bundle/overlay materialization path | **FAIL** | `render_staged_layout` copies already-present checkout files at `pg1_execution_capsule.py:721-794`. `preflight_overlay_stage` compares receipt maps at `885-909`. Neither executes the source-bundle extraction nor overlay application mechanism used by a deployment, so archive/layout/conflict behavior remains untested. |
| Confirmation absence bound | **PASS statically** | Compilation requires both bundle confirmation flags false (`259-270`), the contract binds confirmation manifests/images false and reads zero (`392-409,665-682`), and staged inventory equality rejects undeclared bodies (`797-851`). |
| Critic identity bound | **PARTIAL / blocking** | Commit, path, byte identity, and freeze identity are bound (`58-64,410-433,642-645,683-691`). But `preflight_critic` only validates bytes and AST imports (`959-978`). The capsule does not bind or invoke the actual critic CLI with `--expected-run-id` set to `attempt_run_id`, so critic execution identity remains unproved. |
| Historical field-name plant targets deployed failure site | **FAIL** | `test_pg1_execution_capsule.py:80-86` renames the capsule key and is rejected by the centralized canonical validator (`pg1_execution_capsule.py:537-546`) before any actual job/cell consumer reads it. It does not exercise the historical downstream lookup site. |
| Historical source/attempt identity plant targets deployed failure site | **FAIL** | `test_pg1_execution_capsule.py:89-94` mutates the canonical source identity and is rejected by the central exact-identity check (`pg1_execution_capsule.py:486-513`). It does not cross an actual source verifier or job boundary with distinct valid source and attempt identities. |

## Precise blockers

### 1. Consumer names are aliases for one validator, not consumer-owned preflights

Every `consumers.<name>.module` is the same dispatcher module. The functions named source gate, overlay stage, job, worker, controller, and critic live in the capsule compiler module itself. No independent consumer implementation imports and calls a shared pure preflight over the capsule.

This violates the core proof obligation: the code that will later consume the contract must also own and execute the preflight before paid authority. A centralized re-description can agree with itself while the deployed consumer disagrees, as happened twice already.

### 2. The future deployed path is not bound

The implementation inventory includes `dag.py` and `launch_packet.py`, but the rehearsal never imports or invokes either. There is no PG1-native job, worker, controller, or launch entrypoint bound to this capsule. The actual critic file is copied and inspected but never run. Thus the receipt cannot constrain the eventual execution graph.

### 3. “Exact staged layout” is a checkout projection

The renderer copies individual files from the worktree and infers their origins from receipts. It does not traverse the deployment's exact bundle extraction and overlay application path. The overlay consumer checks a map, not the overlay stage. This is content-identity validation, not execution-equivalent staging.

### 4. Critic attempt identity is not part of the executable contract

The frozen critic's actual interface requires an expected run identity. The capsule binds critic bytes but no actual critic argv and no equality between its `expected_run_id` argument and `attempt_identity.attempt_run_id`. A future caller could invoke the correct critic bytes with the wrong identity and still satisfy this static capsule.

### 5. Both historical plants stop at the new central validator

The plants demonstrate that malformed capsules are rejected. They do not demonstrate that a valid capsule crosses the exact sites that failed historically. The first plant must reach the actual job/cell config-label consumer; the second must pass two distinct valid identities through the actual source verifier and job/worker/critic chain. Until those sites exist in the replacement path, the plants cannot be execution-relevant.

## Minimum changes required before cloud CPU plants may be considered

This audit does not authorize or implement these changes. A new static review should require:

1. Separate PG1-native consumer modules for source staging, overlay application, job, worker, controller, and critic invocation. Each must expose and call the same pure capsule preflight that its real execution path calls.
2. Capsule `consumers` entries must bind each actual module/path, implementation identity, exact argv, and declared input/output contract—not six aliases to the dispatcher.
3. The dispatcher must invoke those exact modules over the exact materialization path that a future paid launch uses. Shared low-level transport is acceptable; a second synthetic path is not.
4. The actual job preflight must validate `primary_config` before processor or weight access. The actual source verifier must consume `source_run_id`; worker/controller/result keys and actual critic `--expected-run-id` must consume `attempt_run_id`.
5. The two historical plants must reach and fail at those actual consumer-owned checks, while a valid distinct-source/distinct-attempt capsule crosses all of them.
6. Only after a fresh static audit is `STATIC_GO` should one bounded, no-GPU cloud CPU rehearsal of the exact staged path be considered. That rehearsal still must carry no H100 authority and no model/data bodies.

## Authority boundary

- Candidate capsule scaffold: useful but **STATIC_NO_GO**
- Existing legacy PG1 launch path: remains retired and **NOT_TRUSTWORTHY**
- H100 authority, execution freeze, retry, or fresh attempt identity: **not licensed**
- Bounded cloud CPU plants now: **not authorized**, because they would only exercise the synthetic dispatcher and could not close the blockers above
- Scientific status: unchanged; this is operations-only evidence and no model outcome was observed

## Paid-compute assertion

This review made no cloud or market call, created no resource, read no model/data body, and authorized no run. It did not alter paid-compute state. **Paid compute remains OFF under the latest banked account evidence.**
