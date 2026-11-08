# PARALLAX on Vast.ai

This directory preserves the auditable PARALLAX Vast execution history. Vast.ai
remains the compute provider, but Amazon S3 is now the only durable system of
record. Vast disks are disposable instance scratch and must disappear with the
instance. No local Docker build or heavyweight laptop workload is part of this
path.

## Current state (2026-07-18)

- The 4 TB persistent Vast volume `45042496` is deleted.
- Vast reports zero instances and zero volumes; recurring Vast storage cost is zero.
- Unique results and receipts are stored in a private, versioned, checksum-verified
  S3 archive. The exact object and restore proof are recorded in
  `parallax_vast_storage_retirement_v1.json`.
- The volume-backed mutation CLI in `guarded_launch.py` is hard-retired. It remains
  importable only to audit historical receipts and tests.
- The active storage contract is `parallax_cloud_storage_policy_v3.json`: direct S3
  hydration/result publication, only measured per-job ephemeral instance scratch, and
  no persistent Vast volume or volume gateway. The current rehearsed range cell needs
  80 GB; there is no standing 1 TB Vast allocation.
- The frozen research spec's section 12.5 still describes the retired Vast-volume
  design because changing that hash-bound scientific document after execution would
  destroy its audit value. `parallax_execution_safety_amendment_v2.yaml` and
  `parallax_launch_packet.s3_v2.template.yaml` explicitly supersede that clause for
  every current and future launch while preserving the original hashes.
- A cheap guardian on a different machine is armed before worker creation. Live v2
  evidence proves it remains present through provider loading and independently STOPs
  an over-budget worker; worker/guardian cleanup returned the account to zero.
- The corrected stock transport gate is live-GREEN: the worker atomically produced its
  terminal JSON, received S3 HTTP 200 plus immutable version
  `BGUs7E9w22iFPBf9X3cqP05Ne1x2acqQ`, self-destroyed, and the controller restored that
  exact version byte-for-byte before proving zero instances and zero volumes.
- Post-v2 code requires ten clean absent observations, excludes listing errors from
  absence, checks guardian liveness throughout worker bootstrap, starts hard runtime at
  first provider `running`, and stops S3 polling as soon as a worker becomes terminal.
  These final two corrections are unit-tested but not live-promoted.
- Execution freeze v75 keeps paid action, H100, model/data reads, training, and science
  off. Transport is cleared only; the registered PG1 optimizer-scale gate still needs a
  separate current-source and market audit/freeze. There is no automatic retry.

### Why Vast still shows a disk size

It is not durable Vast storage. A Vast worker needs ordinary instance-local scratch so
the H100 can read model shards, task inputs, optimizer state, and temporary outputs at
local-disk speed. That scratch is disposable and is destroyed with the worker. The
launcher requests only the measured peak plus 30% contingency; the current rehearsed
range-cell class requests 80 GB. There is no fixed Vast scratch minimum or 1 TB Vast
allowance.

The 1 TB number is the current AWS S3 planning allocation. It is unrelated to Vast
storage and does not make any Vast volume, gateway, or larger scratch request eligible.

AWS S3 owns every durable responsibility: immutable inputs, model-cache hydration,
checkpoints, terminal results, lifecycle receipts, version history, and restore tests.
Workers receive narrowly scoped, version-bound presigned URLs rather than AWS
credentials. No current path creates, attaches, or pays for a persistent Vast volume.

The frozen optimizer-scale source contract predates this correction and still contains
the historical string `vast_volume_45042496`. That string has no runtime authority and
is superseded, without changing the hash-bound scientific inputs, by
`parallax_optimizer_scale_s3_execution_amendment_v1.json` and the global storage policy
v2. The run remains locked until a separate execution freeze binds those policies.

## Retired volume architecture (audit history only; never execute)

Everything in this section documents the 2026-07-15--17 implementation and its failure
boundaries. It is retained so old receipts remain interpretable. It is not an available
fallback, and none of its gateway, volume, `vast copy`, or `/volume` instructions may be
used by a current launcher.

Lifecycle v13 is live-proven on Vast: the exact stock-image command gate passed,
the job exited zero, all three frozen artifacts (2,752,640 bytes) were durable,
STOP was confirmed 0.64 seconds after the durable manifest, DESTROY followed,
and a separately created gateway reattached the volume and reverified every
byte and terminal receipt. A planted first-final-sync timeout was contained by
the frozen second attempt; teardown did not begin until the missing file was
copied and hashed. The account returned to zero after both the launch and the
inspection.

This success does not itself authorize research compute. The current post-v13
freeze revokes launch, inspection, control-probe, H100, GPU-work, and science
paths until a new content-hashed smoke-only packet is audited. The guarded
launcher selects the live offer itself; it does not accept offer IDs from a
caller, and any future H100 scope must therefore use the literal cheapest
suitable all-region H100 visible at creation time.

The guardian separates Vast's unbilled `Loading` phase from a nonrenewable
60-second billable pre-arm clock. The latter begins at the first non-`Loading`
or unknown provider interval; recovered `Loading` pauses it but never erases
accumulated uncertain exposure. Provider polling is bounded at five seconds,
and uncertain transition intervals are charged conservatively. The lifecycle
suite has 102 passing tests, including green and planted non-zero inspection
account audits.

The control plane readies the cheap persistent-volume gateway before creating
any worker, uses an exact-ID dedicated relay key accepted by Vast's SSH proxy,
and gives the gateway guardian a scoped lifecycle key that can read and stop
instances but cannot mutate volumes or account credentials.
The historical `storage_rehearsal` module is also hard-revoked at its function
and low-level mutation boundaries, so its old phrase cannot bypass this gate.

Before the gateway exists, a detached local exact-label reaper is active. Before
the worker exists, a detached guardian and durable registry are active on the
persistent volume. The guardian owns the loading, billable pre-arm, heartbeat,
and 150-second hard-runtime clocks. Once worker SSH is genuinely ready, local
scoped-key deadmen add independent pre-arm and hard-STOP backstops. A two-phase
gate prevents the job from starting until the worker, supervisor, and execution
release markers are all present. While the job runs, atomic artifacts are
mirrored to the volume every 0.5--5 seconds and the guardian watches a heartbeat
that does not share the artifact-copy thread.

The terminal ordering is fixed:

1. Verify required sentinels and a content-addressed manifest on the volume.
2. Request worker `STOP` so compute billing ends.
3. Request worker `DESTROY` to remove scratch storage.
4. Fsync the lifecycle receipt to the volume.
5. Destroy the gateway from its per-instance API key.

A separate inspection command is also fail-closed: after its gateway receives
STOP and DESTROY, it waits for account-zero, persists that audit, and rewrites
the inspection verdict to `RED_ACCOUNT_NOT_ZERO` instead of returning success
if any paid instance remains.

If durability is unproven, the system never destroys the only copy: it stops
the worker and preserves its disk. If the supervisor hangs or dies, the
detached watchdog stops the worker and destroys the gateway. If both control
processes fail, the provider guards act without a laptop or account-wide API
key. An unexpected Python exception requests `STOP` immediately rather than
waiting for either fallback.

- The market audit is read-only and writes a sanitized decision record.
- H100-local disk is disposable scratch. Versioned private S3 is the system of
  record for inputs, checkpoints, results, and receipts.
- A 15 GB disposable volume rehearsal must prove create, attach, write,
  checksum, detach/re-attach, restore, and delete behavior before the durable
  research volume is created.
- The first two truth-sprint stages form a hard six-H100-hour preflight
  envelope. Crossing hour six requires a valid, content-hashed `GO` packet.
- Startup installs nothing. Any package or CUDA mismatch is a failed image and
  stops before a model or dataset is loaded.
- Historical references below to an active persistent Vast volume describe the
  retired 2026-07-15--17 architecture and must not be used as launch instructions.

## Local read-only audit

```bash
python -m infra.vast.parallax.live_audit \
  --output studies/lineage/cross-lineage/results/readiness/vast_market.json
```

The command searches every region and chooses the lowest current Vast
`dph_total` among verified, reliable, available one-H100 offers with at least
500 GB scratch. The offer is always re-queried immediately before creation;
offer IDs are observations, never configuration.

## Immutable image

The Dockerfile pins the Vast CUDA image by OCI digest and every PARALLAX
top-level dependency exactly. The Vast SSH-capable CUDA base itself is
pinned by its linux/amd64 digest; dependencies are delivered either in the
built OCI image or as one content-addressed bundle built once on the persistent
gateway. Installing or downloading packages on the H100 is forbidden. The runtime is not launchable until
its digest, full `pip freeze`, `pip check`, H100 smoke output, and PG1/PG2 model
contract checks are captured in readiness evidence.

The one-command H100 action is deliberately limited to environment, mounted
storage, and already-cached model verification:

```bash
python -m infra.vast.parallax_run \
  --matrix studies/lineage/cross-lineage/config/parallax_matrix.yaml \
  --lock infra/vast/parallax/requirements.lock \
  --cache-root /parallax/cache/huggingface \
  --transfer-receipt /workspace/parallax/evidence/inbound-transfer.json \
  --cache-hydration-receipt /workspace/parallax/evidence/cache-hydration.json \
  --model-cache-manifest /workspace/parallax/evidence/model_cache_2026-07-15.json \
  --instance-id "$VAST_CONTAINER_ID" \
  --output /parallax/evidence/h100-preflight.json \
  --require-h100
```

It has a hard 30-minute ceiling, performs no science, and contains no download
path. A cache miss is a refusal, not a reason to install on the H100.

The measured direct-SSH rate would take hours to move the complete cache. After
the runtime and small immutable inputs arrive from the durable gateway, a
separate no-science command may hydrate only the exact pinned revisions from
Hugging Face into H100 scratch. It verifies every byte against the durable
manifest. Successful preflight writes `.parallax_cache_gate.json`; hydration
refuses to run if that marker exists, enforcing the frozen ban on downloads
after the cache gate.

Vast volumes are machine-local. The cheapest all-region H100 is therefore fed
through a temporary cheap gateway on the volume's machine. A one-GiB,
cross-machine, direct-SSH round trip must match by SHA-256 before H100 rental.
The Vast copy helper is forbidden because rehearsal produced false `Done`
reports and altered effective SSH access. Instance bootstrap uses one existing,
exact-ID Vast account key that is verified read-only before creation. The
launcher creates no account SSH keys. It installs an unregistered per-run
public key directly on the worker over that bootstrap connection, sends only
the matching per-run private key to the gateway, and deletes the local private
copy at handoff. The stable bootstrap private key never leaves the local
machine. The preflight re-hashes every local model file; its output returns by
direct SSH and must be verified on the durable volume before the smoke can turn
green.

## Frozen G1 sequence

The first scientific action is one sequential, stop-on-error G1 sequence on the
same Vast H100 that passed its instance-bound preflight. The sequence runs the
primary core rail, the isolated native pi rail, the supporting core rail, and
only then the paired adjudicator. Core and pi subprocesses receive mutually
exclusive `PYTHONPATH` values; neither runtime can see the other's packages.

```bash
python -m infra.vast.parallax.g1_sequence \
  --root /workspace/parallax/source \
  --matrix /workspace/parallax/source/studies/lineage/cross-lineage/config/parallax_matrix.yaml \
  --anchors /workspace/parallax/source/studies/lineage/cross-lineage/config/parallax_g1_anchors_v2.yaml \
  --cache-root /workspace/.hf_home/hub \
  --task-root /workspace/parallax/tasks/parallax_tasks_v1 \
  --core-site /workspace/parallax/runtime/core/site-packages \
  --pi-site /workspace/parallax/runtime/pi/site-packages \
  --instance-id "$VAST_CONTAINER_ID" \
  --output-dir /workspace/parallax/results/paired-g1
```

Any failed hook, strict weight load, cache contract, prompt/image contract, or
preprocessing comparison stops the sequence and still produces a hashed failure
packet. A green result licenses only the registered shared-SigLIP bridge. It
does not merge the Gemma-1 primary lineage and Gemma-2 supporting lineage, and
it never compares their raw language weights or logits.
