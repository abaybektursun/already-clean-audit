# Qwen3-8B persona-combination rental

Completed 2026-10-27. Instance **58022910**, RTX PRO 5000 Blackwell,
offer56821109, advertised **$0.7555556/hour**. Invoice charge **$0.105**.
Program realized spend **$32.123** against unchanged **$36** ceiling.

Vast reports no live instances. Station `_done.json` is banked. Watchdog PID79570
was stopped after verifying zero instances. No ongoing compute.

Bank: `s3://pg-lineage-769063704696/results/mdl/mendeleev_neuron_combo_20261027_a1/`.
All14 completed responses and startup JSON match the final S3 bank after canonical
JSON serialization; see `bank_verification.json`. They contain16 generated rows,
including duplicated batch2 rows, not16 independent samples.

Manifest declares45-minute billed limit,30-minute idle shutdown, no respawn.
Code shipped at71f7e494; live redeploy90ac8817 fixed an unrelated study-plan
dependency after weights loaded but before startup or inference completed.
The first request also arrived before station directory readiness. Both incidents
are preserved in `incidents.json` and transport artifacts. No extra rental was needed.

Local request validation rejected four registered malformed inputs; Tier1 preflight
passed26 policy scenarios and static checks. GPU startup identity/realized-addition
checks passed. Different-model scientific audit is recorded in
[the result report](../../../../studies/neuronshow/COMBO_RESULTS.md).

The readout server is specific to this two-neuron experiment. It does not expand
the production probe API. No broad infrastructure tuning or speed claim was made.

```sh
python3 infra/vast/controller.py plan infra/vast/manifests/neuron_combo_20261027_a1.json
python3 infra/vast/controller.py rent infra/vast/manifests/neuron_combo_20261027_a1.json 56821109
python3 infra/vast/watchdog.py
python3 infra/vast/probe.py --rental neuron_combo_20261027_a1 status
python3 infra/vast/probe.py --rental neuron_combo_20261027_a1 send REQUEST.json --timeout 600
python3 infra/vast/probe.py --rental neuron_combo_20261027_a1 stop
```

These are provenance commands, not a reusable rental launch: a future attempt needs
a fresh rental/prefix and current offer. Never print raw Vast instance objects.

<summary>
User requested running nautamine post 2115116538533392836: combine two screenshot
personas. Exact targets: Qwen3-8B L8N9994 positive alpha1 and L16N1887 negative
alpha.75. Completed COMBO_RESULTS.md, COMBO_TRANSCRIPT.md, COMBO_SPEC.md, raw bank
probes/2026-10-27_combo, and updated RESEARCH.md. Original strengths yield an
uncertainty loop; 75% a price loop; 50% a punctuation loop; 25% ordinary Qwen.
A reproduces a Tokyo student, but not topology; B matches the performer opening.
Two numerical variants still loop. Half-dose B already repeats, so no claim of
two clean components or a special pair interaction. A different-model gpt-6-astra
auditor read all 14 responses (16 rows) and startup checks. Source precision,
batch and hook details are unknown; no exact numerical replication or novelty claim.
Rental 58022910 destroyed; no instances remain; watchdog 79570 stopped. Cost $0.105;
program total $32.123/$36. All 15 response/startup JSON artifacts match S3.
Worktree interp_research-probe-session, branch neurons/probe-session.
No further compute promised for this post.
</summary>
