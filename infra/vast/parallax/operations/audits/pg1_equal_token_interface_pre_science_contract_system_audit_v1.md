# PG1 equal-token pre-science contract-system audit v1

- Created: `2026-07-19T06:19:41Z`
- Scope: the two banked PG1 equal-token terminals, their frozen launch/source contracts, and the exact implementation revisions bound by those freezes
- Method: read-only inspection of committed artifacts and `git show` at the frozen commits; no model or dataset bodies read; no cloud or market calls; no resource creation; no retry
- Current launch-path verdict: **NOT_TRUSTWORTHY**
- Root-cause class: **SYSTEMIC_DEPLOYED_CONTRACT_GRAPH_NOT_EXECUTED_BEFORE_H100_AUTHORIZATION**
- Disposition: **RETIRE_AND_REPLACE_THE_PG1_LEGACY_ADAPTER_LAUNCH_PATH**

## Executive finding

The terminals are two different manifestations of one design defect. The controller validates individual manifests, hashes, constructors, and selected helper functions, but never executes the exact staged producer-to-consumer contract graph before authorizing the H100. The paid runtime is therefore the first process to discover whether the frozen source contract can actually be consumed by every downstream program.

The first attempt exposed a schema-name mismatch only after all three weight shards were downloaded. The attempted correction covered that one consumer field but retained a mutable legacy-global adapter. The fresh-v2 attempt then exposed a second hidden coupling: the same `run_id` field was treated both as immutable source identity and fresh attempt identity. The source gate explicitly accepted that split, while the deployed legacy verifier rejected it.

The prior bounded v1 audit correctly identified the immediate missing field, but its classification as a one-field omission was incomplete. Fresh-v2 is direct falsifying evidence for that narrow diagnosis.

## Failure 1 — original v1 attempt

**Exact immediate cause:** the frozen PG1 source contract binds its training configuration under `artifacts.primary_config`; the exact deployed legacy cell reads `contract["artifacts"]["interface_config"]`. The v1 adapter passed the unmodified frozen source contract to that cell. Python raised `KeyError: 'interface_config'`.

The failure occurred after the runtime had marked `SCIENCE_STARTED` and downloaded three weight objects. It produced no result and no candidate rows.

Exact evidence:

- Terminal: `infra/vast/parallax/operations/results/pg1_equal_token_interface_seed17_discovery_v1/raw-terminal.json`, SHA-256 `e6a9e00e89aec149dec68cdc4765dcd91728dcf36db30aac0a9dcf3275a1f650`
  - terminal status `RED_JOB_TERMINAL`
  - job return code `1`
  - `phase.phase = SCIENCE_STARTED`
  - `phase.weight_objects_downloaded = 3`
  - traceback terminates at `training_interface_control_cell.py` with `KeyError: 'interface_config'`
- Frozen source contract: `studies/lineage/cross-lineage/config/parallax_training_interface_control_source_contract_pg1_primary_v1.json`, SHA-256 `a95f65a78d5dc6b7f4b80416d40c6b9dcc45cc1c91fde74154da90dd117e5544`
  - contains `artifacts.primary_config`
  - contains no `artifacts.interface_config`
- Exact v1 adapter at commit `340ec9c`:
  - `infra/vast/parallax/primary_interface_control_job.py`, SHA-256 `1c890ad198d9d3c7badf1f7225e1213c57de08f94fdcd679fe3cc8fdf8099210`
  - `_run_primary_control` passes `source / SOURCE_CONTRACT_PATH` directly as `--source-contract`
- Exact downstream consumer at commit `340ec9c`:
  - `infra/vast/parallax/training_interface_control_cell.py`, SHA-256 `2bdfb7bef9137eccf51211b5708d2bef9f98fe3d2577593479ab2c24f284647e`
  - `run` indexes `contract["artifacts"]["interface_config"]["path"]`
- Freeze: `infra/vast/parallax/operations/freezes/pg1_equal_token_interface_seed17_one_shot_v1.json`, SHA-256 `9a3b89f7f88276d3d0defc28b1a6e07abee3b5b6a0f59bbd24b2750d4af4111c`
- Green no-create that missed the defect: `infra/vast/parallax/operations/readiness/pg1_equal_token_interface_seed17_final_no_create_v1.json`, SHA-256 `d55e5229ddaa12a2e28301338f0964846041747bfd9def52f3a5e2b2f1fd1427`
- Banked adjudication boundary: `infra/vast/parallax/operations/results/pg1_equal_token_interface_seed17_discovery_v1/independent-critic-invocation.json`, SHA-256 `d3949bb25f94c726bc0dc6863f8c8e12fc4ddc7dcd02bb7702e6ad2366c870ba`

## Failure 2 — fresh-v2 attempt

**Exact immediate cause:** the fresh adapter defines execution `RUN_ID = parallax-pg1-primary-interface-s3-v2` and immutable `SOURCE_RUN_ID = parallax-pg1-primary-interface-s3-v1`. It then monkey-patches the legacy module's `RUN_ID` to v2. The exact legacy `_verify_source` routine requires the frozen source contract's `run_id` to equal that patched `RUN_ID`. The frozen source contract remains correctly hash-bound to v1, so the comparison is deterministically `v1 != v2` and raises `InterfaceJobError: interface source contract escaped its boundary`.

This failure occurred during `SOURCE_STAGING`, before science and before model-weight hydration. It produced no result and no candidate rows.

Exact evidence:

- Terminal: `infra/vast/parallax/operations/readiness/pg1_equal_token_interface_seed17_fresh_v2_attempt/terminal.json`, SHA-256 `ebf586b54047a5faf4550a0a1976fd6ecf2207f98cb070f8a971bc60935a4da4`
  - terminal status `RED_JOB_TERMINAL`
  - job return code `1`
  - `phase.phase = SOURCE_STAGING`
  - `phase.science_started = false`
  - traceback terminates in `_verify_source` with `interface source contract escaped its boundary`
- Fresh operational spec: `infra/vast/parallax/operations/readiness/pg1_equal_token_interface_seed17_fresh_v2_spec.json`, SHA-256 `287deb60b3cd577d3bb428090f797340013f973b56af20390098711d657d8516`
  - declares fresh `run_id = parallax-pg1-primary-interface-s3-v2`
  - preserves predecessor/source contract unchanged
- Exact fresh adapter at commit `d5176479d7f72fa21a07c43b0d14b89364f13e6f`:
  - `infra/vast/parallax/primary_interface_control_job.py`, SHA-256 `bc1d1f51d769fe0e8902bad2f286f8c3b98a2ccef1e923562ca6384fec6dd686`
  - defines v2 `RUN_ID` and v1 `SOURCE_RUN_ID`
  - `_patched_legacy` overwrites legacy `RUN_ID` with v2
- Exact legacy verifier at commit `d5176479d7f72fa21a07c43b0d14b89364f13e6f`:
  - `infra/vast/parallax/ephemeral_interface_control_job.py`, SHA-256 `c2ba15dc19b020b87f3a7212dbca72adb87a102baaf2575707e860d8d770df7b`
  - `_verify_source` requires `frozen.get("run_id") == RUN_ID`
- Frozen source contract remains the exact v1 object, SHA-256 `a95f65a78d5dc6b7f4b80416d40c6b9dcc45cc1c91fde74154da90dd117e5544`
- Freeze: `infra/vast/parallax/operations/freezes/pg1_equal_token_interface_seed17_fresh_v2_one_shot_v1.json`, SHA-256 `e176f36c94203ac31bf8281a9a399d6e9c9fa0337cf196a413d6c93477d1dde2`
- Green no-create that missed the defect: `infra/vast/parallax/operations/readiness/pg1_equal_token_interface_seed17_fresh_v2_final_no_create_v2.json`, SHA-256 `0e988e67b2f8f80dc7b0b19c6e84fd52168314af78910f1767f8cf1e2ab97cd4`
- Green plants that missed the defect: `infra/vast/parallax/operations/readiness/pg1_equal_token_interface_seed17_fresh_v2_plants_v1.json`, SHA-256 `a498cf3f82571dd1d62961123aec5764319c732d8e2167a9a3596070d1cbfb30`
- Banked adjudication boundary: `infra/vast/parallax/operations/readiness/pg1_equal_token_interface_seed17_fresh_v2_attempt/independent-critic.json`, SHA-256 `5dc002a27a4b96f21c797db7dccbebc7f7aa25682659c5cf8c029b5b720041b3`

## Shared systemic defect

Both defects cross an internal producer/consumer boundary that the pre-paid checks did not execute:

1. v1: source-contract producer calls the field `primary_config`; downstream child consumer requires `interface_config`.
2. v2: fresh-attempt producer intentionally separates source and execution identities; legacy source consumer assumes they are identical.

The adapter architecture makes these hidden couplings likely. It reuses a PG2-oriented executor by mutating module globals (`RUN_ID`, lineage, paths, hashes, model inventory, overlays, and expected outputs) at runtime and overrides selected functions. Contract validity is therefore distributed across Python globals, JSON fields, staged filesystem layout, command-line arguments, and downstream consumer assumptions. No single canonical schema or compiled capsule owns the complete graph.

The result is a **contract composition failure**: each component can be locally green while their composition is impossible.

## Why no-create and planted tests did not catch either failure

The receipts prove construction, identity, mutation, and safety invariants; they do not prove executable compatibility.

- The no-create path builds and individually validates job, upload, and worker contracts, then inspects source text for lifecycle ordering. It does not extract the exact bundle and overlays into the deployed layout and invoke the exact job entrypoint through every pre-science consumer.
- The v1 job test replaces `legacy.run` with a planted function. The control test replaces `_run_command` and prewrites a synthetic result. Consequently the real child never reads the real frozen source contract.
- The fresh-v2 tests explicitly assert that the fresh execution `run_id` differs from the predecessor while preserving the predecessor source fields. They never exercise that state through legacy `_verify_source`, which assumes equality.
- The v2 no-create calls the primary local source gate, which is aware of separate `RUN_ID` and `SOURCE_RUN_ID`, and calls `legacy_compatible_source_contract` in isolation. The paid job later invokes the older legacy source verifier, which is aware of only one `RUN_ID`. Passing the aware checker says nothing about the unaware deployed consumer.
- Hash closure confirms that the wrong composition is reproduced exactly. It cannot prove that the exact bytes agree on field names or identity semantics.
- Processor, confirmation-firewall, market, durability, and lifecycle plants are orthogonal to these contract-consumer mismatches.

This explains why increasing the number of unit tests from the first attempt to 687 green tests in fresh-v2 did not reduce this risk: the missing proof obligation was execution of the composition, not another local invariant.

## Minimal trustworthy pre-science architecture

The following is the minimum architecture that prevents this failure class from reaching an H100:

1. **One immutable compiled execution capsule.** A compiler produces one canonical contract with distinct, mandatory `source_identity` and `attempt_identity` objects. Runtime programs must not infer one from the other. The capsule binds exact staged paths, artifact labels, hashes, entrypoint arguments, and consumer schema versions.
2. **No mutable legacy-global adapter.** A PG1-native entrypoint receives the capsule explicitly. Shared lifecycle and S3 transport primitives may remain below this boundary, but model/science job globals, verifier globals, and function monkey-patches are forbidden.
3. **Consumer-owned pure preflight.** Every actual consumer exposes one pure `preflight(capsule, staged_root)` function. The paid path calls the same function before any processor/model hydration, imports of GPU frameworks, CUDA check, or science-start phase. There is no second validator with similar-but-different semantics.
4. **Exact staged-layout replay.** Before H100 authorization, a bounded remote CPU rehearsal extracts the exact source bundle, applies the exact overlays, materializes the exact capsule, and invokes every real entrypoint through a `--pre-science-only` route. It must cross the same source verifier, child CLI, and contract readers used by the paid route. Model/data bodies are replaced by inventory-only sentinels and are inaccessible.
5. **Fail-closed readiness receipt.** The replay emits a canonical receipt containing the capsule hash, source bundle/version hash, overlay set hash, all entrypoint hashes, ordered consumer-preflight results, and an explicit proof that zero model/data body reads and zero science/GPU operations occurred. A separate launcher may create an H100 only when the receipt hash is frozen verbatim and all cloud result/claim identities are still absent.
6. **Paid-path phase order.** `SOURCE_STAGING -> ALL_CONSUMER_PREFLIGHTS_GREEN -> PROCESSOR_HYDRATION -> WEIGHT_HYDRATION -> SCIENCE_STARTED`. Any consumer incompatibility terminates before weight URLs are released.
7. **One deliberate planted contradiction per boundary.** At minimum: missing/renamed config label, differing source/attempt IDs, duplicated config aliases, wrong staged path, child CLI schema drift, and a consumer that reads an undeclared field. Each plant must fail in the exact staged-layout replay, not only in a unit helper.

With this design, a producer/consumer contract mismatch cannot be first discovered by paid H100 execution: H100 authority is downstream of successful execution of the same consumer preflights over the exact compiled capsule.

## Repair-or-retire decision

**Retire and replace the current PG1 adapter launch path. Do not patch and retry it.**

Preserve the two attempts and current code as forensic evidence. Reuse only independently bounded lifecycle, cheapest-offer selection, S3 durability, and cleanup primitives beneath a new PG1-native interface. The current `primary_interface_control_{launch,job,worker}.py` composition must not regain paid authority through another field alias or identity exception.

Reason: the first repair fixed the first encountered downstream assumption but the very next execution found another assumption earlier in the same path. This proves the legacy-global adapter has an unenumerated compatibility surface. A third patch would provide no evidence that the next hidden coupling is absent.

## Verdict and authority boundary

- Existing v1/v2 PG1 launch path: **NOT_TRUSTWORTHY**
- Existing green no-create/plants as proof of deployable composition: **NOT_TRUSTWORTHY**
- Existing terminals as evidence of no scientific outcome: **TRUSTWORTHY**; both are durable, exact banked failures with no candidate rows
- Any automatic or manual paid retry under the existing adapter: **FORBIDDEN BY THIS AUDIT RECOMMENDATION**
- Required next action: replace the adapter, then independently demonstrate the exact staged-layout pre-science receipt before considering a new execution identity or freeze

## Resource assertion

This audit performed no cloud or market query and created no resource. It read no model or dataset body and authorized no retry. Paid-compute state was not mutated; the latest exact banked account observations in both critic receipts report `Vast instances = 0`, `Vast volumes = 0`, and AWS experiment compute/pending-or-running instances `= 0`. **Paid compute remains OFF.**
