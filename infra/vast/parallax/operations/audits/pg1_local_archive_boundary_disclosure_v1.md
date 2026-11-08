# PG1 local archive boundary disclosure

This is an operations-only boundary record. It is not a science journal,
result, claim, freeze, ledger, or execution authorization.

Before the boundary was made explicit, local verification of candidate
`10c00f7` ran
`test_exact_copied_compiler_gateway_idle_and_production_chain_fake_provider`.
That test created and read one temporary, synthetic `exact-source.tar` beneath
pytest's temporary directory. It contained tracked source/test fixture bytes;
it did not contain a protected model, data, confirmation, or science body.

The test is now marked `cloud_cpu_only` and begins with a guard that skips it
unless `PG1_APPROVED_AWS_FUELOS_CLOUD_RUNNER=1`. Current candidate bytes have
not run that test. Local verification must exclude the marker and must not
create, read, list, extract, or inspect archives. No cloud or provider action
was taken while adding or verifying this boundary.
