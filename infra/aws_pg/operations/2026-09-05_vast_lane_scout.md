# Lane scout: vast.ai vs AWS for the STAGEDIFF W2 fits (2026-09-05T03:09Z, read-only, nothing rented)

Question (user): would vast.ai speed up the remaining critical path, counting image creation, spin-up and access?
Method: an Explore subagent ran read-only `vastai`, `aws ecr/iam/s3` and file reads; every number below came from a
command output or a file. Decision: stay on AWS for W2; re-evaluate per wave under the "fastest option" rule.

| item | measured |
|---|---|
| image | `shackleton:mdl-v1` exists in ECR, 3.33 GB, torch 2.12.0+cu130 + transformers 5.9.0 (= the AMI pins); vendored jlens needs torch only → no build |
| offers now | 4 qualifying H100-class offers (H100 NVL $2.67/h at 920 Mb/s; three H100 SXM $4.54–6.18/h) = the AWS 4-slot quota → no parallelism gain |
| rent gate | `infra/vast/.preflight_pass` absent; `controller.py` refuses to rent until `preflight.sh` stamps HEAD:infra (~5 min if it passes) |
| loading (unbilled) | 5–10 min measured on two prior mdl-v1 rentals |
| billed setup | net probe 7 s; weights from S3 18–20 GB per model → 2 min at 200 MB/s, 20 min at the 17 MB/s one host measured; wikitext-103 from the Hub (not in the S3 cache), hub reads timed out repeatedly on a prior host |
| fit on H100 | 12–17 min per 64-prompt cell (L40S measured 32–39 s/prompt at 100 % SM; dlperf ratio 3.3–4.0×, assume 2.5–3×) |
| IAM | `mendeleev-box` user + policy already read `weights/hf/*` and write `results/mdl/*`; a `results/stagediff/*` prefix would take 10–15 min plus a new net-probe object |
| result layout | the pair job fetches lens.pt/banks/metrics.json from any prefix; the verdict needs one `aws s3 cp --recursive` per vast cell |
| untested | the transformers 4.57.1 pin venv (C1/C2) has never been built inside this image |
| cost | 6 cells ≈ $11–14 against a $30 program ceiling and $36.17 credit |

Wall clock: AWS as queued ≈ 1.8–2.0 h (C0a/C0b/C1/C2 running; C3–C6 one more wave, done ≈ 04:55–05:05Z).
Vast best case ≈ 1.1–1.5 h including setup, only for the four movable cells, and only with a fast host; the one
prior mdl-v1 rental pair had one net-probe abort. Expected saving ≈ 20 min; not taken.
