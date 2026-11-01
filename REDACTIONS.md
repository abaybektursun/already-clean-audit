# Redactions

28 lines were redacted, with 31 replacements in 30 logged rows (two lines matched two rules each), in 10 of the 17 copied text files. No line was added or removed, so each
line number below is the same in the study's file and in this copy. The figures were not changed.
One line was edited, not redacted: `SPEC_D1.md` line 1, "and the first developmental test" became "and a
developmental test", because the study's own correction (§1, line 135; `checks/literature_W5.md` §4d) withdrew the
priority claim.
A second line was paraphrased, not redacted, and held no secret: `SPEC_D1.md` line 724, the verbatim quote of the
user's order in the backend amendment's heading became "the user's instruction to answer the Sonnet and Opus roles
with Claude Code subagents".

Sources: the study files at commit 85ad5e37 of `drafts/already-clean` (private working repository). A script applied the rules below line by line, logged
every replacement, and then refused to finish if any of the searched patterns survived.

## Rules

| pattern class | replacement |
|---|---|
| local session scratchpad path (contains the username and a session id) | `<local-scratchpad>` |
| local agent-transcript directory under the home `.claude` directory (contains the username and a session id) | `<local-agent-transcripts>/` |
| local session-log file name (session id) | `<local-session-log>.jsonl` |
| local absolute path containing the username (the home-directory prefix up to and including `projects/`) | `<local>/` |
| S3 bucket name (contains the AWS account id) | `<s3-bucket>` |
| AWS profile name | `<aws-profile>` |
| credential store parameter name (SSM path of the Hugging Face token) | `<ssm-parameter>` |
| API key or token environment-variable name (three names; one holds a Hugging Face token, not an API key) | `<api-key-env-var>` |

## Every redaction

| file | line | pattern class | replacement | count |
|---|---|---|---|---|
| `SPEC_D1.md` | 324 | credential store parameter name (SSM path of the Hugging Face token) | `<ssm-parameter>` | 1 |
| `SPEC_D1.md` | 658 | AWS profile name | `<aws-profile>` | 1 |
| `SPEC_D1.md` | 797 | API key environment-variable name | `<api-key-env-var>` | 1 |
| `SPEC_D1.md` | 798 | credential store parameter name (SSM path of the Hugging Face token) | `<ssm-parameter>` | 1 |
| `SPEC_D1.md` | 815 | API key environment-variable name | `<api-key-env-var>` | 1 |
| `DATA_CONTRACT.md` | 5 | S3 bucket name (contains the AWS account id) | `<s3-bucket>` | 1 |
| `DATA_CONTRACT.md` | 303 | token environment-variable name (holds a Hugging Face token, not an API key) | `<api-key-env-var>` | 1 |
| `INDEX.md` | 17 | S3 bucket name (contains the AWS account id) | `<s3-bucket>` | 1 |
| `INDEX.md` | 19 | local agent-transcript directory under the home `.claude` directory | `<local-agent-transcripts>/` | 1 |
| `INDEX.md` | 19 | S3 bucket name (contains the AWS account id) | `<s3-bucket>` | 1 |
| `results/RESULTS.md` | 96 | local absolute path containing the username | `<local>/` | 1 |
| `results/README.md` | 7 | S3 bucket name (contains the AWS account id) | `<s3-bucket>` | 1 |
| `checks/closeout_slips_2026-10-25.md` | 8 | local session scratchpad path | `<local-scratchpad>` | 1 |
| `checks/closeout_slips_2026-10-25.md` | 8 | local absolute path containing the username | `<local>/` | 1 |
| `checks/closeout_slips_2026-10-25.md` | 130 | S3 bucket name (contains the AWS account id) | `<s3-bucket>` | 1 |
| `checks/closeout_W5_recheck.md` | 6 | S3 bucket name (contains the AWS account id) | `<s3-bucket>` | 1 |
| `checks/closeout_W5_recheck.md` | 28 | local agent-transcript directory under the home `.claude` directory | `<local-agent-transcripts>/` | 1 |
| `checks/closeout_W5_recheck.md` | 79 | S3 bucket name (contains the AWS account id) | `<s3-bucket>` | 1 |
| `checks/closeout_W5_recheck.md` | 81 | local session scratchpad path | `<local-scratchpad>` | 1 |
| `checks/closeout_W5_recheck.md` | 82 | local agent-transcript directory under the home `.claude` directory | `<local-agent-transcripts>/` | 1 |
| `checks/closeout_W5.md` | 5 | S3 bucket name (contains the AWS account id) | `<s3-bucket>` | 1 |
| `checks/W4-W5.md` | 17 | local agent-transcript directory under the home `.claude` directory | `<local-agent-transcripts>/` | 1 |
| `checks/W4-W5.md` | 18 | local session-log file name (session id) | `<local-session-log>.jsonl` | 1 |
| `checks/w5_rerun_2026-10-25.md` | 62 | local session scratchpad path | `<local-scratchpad>` | 1 |
| `checks/w5_rerun_2026-10-25.md` | 63 | local absolute path containing the username | `<local>/` | 1 |
| `checks/w5_rerun_2026-10-25.md` | 79 | local absolute path containing the username | `<local>/` | 1 |
| `checks/w5_rerun_2026-10-25.md` | 177 | local absolute path containing the username | `<local>/` | 2 |
| `checks/w5_rerun_2026-10-25.md` | 178 | local absolute path containing the username | `<local>/` | 1 |
| `checks/w5_rerun_2026-10-25.md` | 210 | local absolute path containing the username | `<local>/` | 1 |
| `checks/w5_rerun_2026-10-25.md` | 217 | local session scratchpad path | `<local-scratchpad>` | 1 |

Copied text files with no redaction: `results/REGISTRY.md`, `checks/acceptance_final.md`, `checks/CHECKERS.md`,
`checks/literature_W5.md`, `checks/W1-W3.md`, `checks/substantive_25.txt`, `checks/nonpassing_172.txt`.

## Kept on purpose

- Provenance: git commit hashes, tags and branch names, model ids, prompt sha256s, metrics and input sha256s, S3 object
  counts, and S3 key paths below the bucket (for example `results/neurons/w5/`).
- The AWS region and instance type (`SPEC_D1.md` line 658–659); they carry no account information.
- `.claude/skills/aws-pg/SKILL.md` (`SPEC_D1.md` line 23): a file inside the private repository, named relative to
  it, not a home-directory path.
- Sibling directory names such as `interp_research-neurons-explain` and `interp_research-check-closeout`; they carry
  no username.
- The container path `/workspace/hf/hub`, the image name `shackleton:mdl-v1` and the rental project name `mendeleev`
  (`DATA_CONTRACT.md`); names, not credentials.
- The prose phrase "The Anthropic API key" (`SPEC_D1.md` line 796); it names no variable and holds no value.

## Searched and not found

No AWS instance id, IP address, vast.ai offer or instance id, email address, access key, API key value or token
appears in any copied file. The searches covered `i-` followed by hex, dotted quads, runs of seven or more digits
outside hashes and step numbers, UUIDs, the AWS access-key prefix, the OpenAI-style and GitHub key prefixes, the
Hugging Face token prefix, private-key headers, `Bearer`, and the username. In the PDF metadata, Author and Creator
are "anonymous" and the Producer is "ReportLab PDF Library"; in the PNG metadata the creator is "anonymous" and the
creator tool is ReportLab. No figure contains a path, bucket or account string.

## Final check (2026-10-31)

The check in `PUBLISH.md` searches every text file except `PUBLISH.md` for the account id, the AWS profile name in any
case, the bucket prefix, the home-directory prefix, the at sign, the AWS access-key prefix, the OpenAI-style key prefix
and the SSH key prefix, and searches every figure file for the same strings without the at sign and the key prefix
(compressed figure streams contain those two by chance). Result: zero hits for the account id, the profile name, the
bucket prefix, the home-directory prefix and the key prefixes, in every file. Six lines contain an at sign, each
justified:

| file | line | why |
|---|---|---|
| `DATA_CONTRACT.md` | 95, 181 | the matrix-product operator between `W_U` and `(gamma … W_down[:, i])` |
| `checks/literature_W5.md` | 12, 225 | the matrix-product operator between `W_U` and `(gamma_final … W_down[:, i])` |
| `checks/w5_rerun_2026-10-25.md` | 14 | the shell's quoted argument list in a quoted command line |
| `README.md` | 269 (114 before the 2026-11-01 rewrite) | the BibTeX entry type `misc` |

The SVG figures have no hit at all. Every PDF and PNG figure contains the bytes of an at sign and of the OpenAI-style
key prefix inside compressed image or page streams; these are not text (the same figures as SVG text have no such
string). `PUBLISH.md`, which is not committed, contains the search patterns themselves and the local path of this
directory in its commands.

## Fixes applied 2026-10-31

From the second reader's review of this release (fixes M1–M8 and R1–R5). Line numbers are those of this release after
the fixes.

| fix | file | line | change |
|---|---|---|---|
| M1 | `README.md` | 7–11 | the opening sentence states the mixed developmental outcome and gives the post-closeout sensitivity 20.6% [13.2%, 27.1%] beside p_u |
| M2 | `README.md` | 38–39 | the Result paragraph gives the sensitivity 0.206 [0.132, 0.271] beside p_u, with "the rule is not applied to it" |
| M3 | `README.md` | 18–19 | O7 is named as OLMo-3-1025-7B at its end-of-stage-1 checkpoint `stage1-step1413814`, not the released model |
| M4 | `README.md` | 88–91 | the 17:15Z amendment superseded the model half of the freeze: explainer for 456 of the 1,800 audit poles, writer and job writer for every pole, and the second reader |
| M5 | `README.md` | 141 | license set to "CC BY 4.0 for text and figures" with the license URL, no longer a placeholder; the author may change it before publishing |
| M6 | `SPEC_D1.md` | 1 | title: "and the first developmental test" became "and a developmental test"; logged above (lines 5–7) as a wording change |
| M7 | release directory | — | `ONE_PAGER.md` removed from the release (moved out of this directory before any `git init`) |
| M7 | `README.md` | file map (after line 68) | the `ONE_PAGER.md` row deleted |
| M7 | `PUBLISH.md` | 31, 47 | `ONE_PAGER.md` dropped from the `git add` list; "38 added files" became "37 added files" |
| M7 | `REDACTIONS.md` | 3, 12 | opening counts now cover the 17 copied study files; the one-pager source note deleted; its two redaction rows and its kept-on-purpose bullet deleted |
| M8 | `PUBLISH.md` | 28–30 | `git config user.email` set to the GitHub no-reply address of account 12763264, and `git config user.name "Abay Bektursun"`, placed in the chain right after `git init -b main`, before `git add` |
| M8 | `PUBLISH.md` | 11–13 | the no-reply step says the command sets the address and the name |
| R1 | `README.md` | 47–49 | "about 28%" scoped: on the 117-pole discovery set, 33 poles (28%) pass §6.1 under one explainer model and fail under the other; the explainer model also tracks depth on the audit sets |
| R2 | `README.md` | 49–52 | on S, every judge line and most clusterer answers came from Claude Sonnet 5.5 as Claude Code subagents (the manual backend), not from the API that served the gate |
| R3 | `README.md` | 53–55 | F42 added: in Gemma-3-4B layers 10–11, 25 CRLF-split windows hold 3.44% and 5.19% of top-50 slots; three sampled G4 poles affected, all UNK, no count moves |
| R4 | `SPEC_D1.md` | 724 | the verbatim quote of the user's order replaced by the paraphrase "the user's instruction to answer the Sonnet and Opus roles with Claude Code subagents"; logged above (lines 8–10) as a paraphrase, not a secret |
| R5 | `README.md` | 130 | "23 MB per Q1 model" became "23–24 MB per Q1 model" |
| R5 | `REDACTIONS.md` | 84–86 | figure metadata described correctly: PDF Author and Creator "anonymous", Producer ReportLab; PNG creator "anonymous", creator tool ReportLab |
| R5 | `REDACTIONS.md` | 26, 38 | the rule and the `DATA_CONTRACT.md` line 303 row say the variable holds a Hugging Face token, not an API key |
| consistency | `README.md` | 4–5, 76 | the subtitle and the file-map row for this file mention the two reworded `SPEC_D1.md` lines |
| consistency | `PUBLISH.md` | 9–10 | the license step says the license is set |
| consistency | `REDACTIONS.md` | 102 | the README at-sign line moved to 114 after the README edits |

After the fixes: the 17 copied study files differ from their sources on exactly 30 lines, the 28 logged redaction lines
plus `SPEC_D1.md` lines 1 and 724, with equal line counts; the 18 figure files are byte-identical; the leakage checks
give the same six justified at-sign lines and nothing else.

## Presentation rewrite 2026-11-01

`README.md` was rewritten for presentation before publishing: a summary and a "Start here" table first, figures F1 and
F5 embedded, the class fractions, kappas and Q2 readouts set out in tables, and the freeze record as a table. Every
number, date, hash, tag, commit id and model name of the 2026-10-31 README is kept; the numbers added (the full class
table, the kappas per model, `A`, `a_ctrl`, the 1,238 and 574 counts, the formula) are copied from `results/RESULTS.md`
and `SPEC_D1.md`. The BibTeX block is byte-identical. The `README.md` line numbers in the tables above are those of the
2026-10-31 README. Two files were added, `CITATION.cff` (citation metadata) and `LICENSE` (the CC BY 4.0 notice);
neither has a leakage-check hit. The copied study files and the figures are unchanged.
