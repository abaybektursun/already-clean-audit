# D. Methods and meta: how the sausage was made

Angle: what it is like to run a pre-registered interpretability study whose instruments are language models and whose annotators are Claude Code agents. Every number below comes from a file opened for this report. Numbers marked "my count" or "my arithmetic" were computed here, read-only, from the named files.

Paths used below:
- `S/` = `/Users/abaybektursun/projects/interp_research-drafts/studies/already-clean/`
- `E/` = `/Users/abaybektursun/projects/interp_research-neurons-explain/` (explain worktree, `bank/` is not under git)
- `P/` = `/private/tmp/claude-501/-Users-abaybektursun-projects-FuelOS/964d1f2d-6631-419a-90f2-7fe212a804e5/scratchpad/` (the lead session's scratchpad: `manual/` chunk directories, `briefs/`, `f13/` audit outputs)

The cast, from the record. The lead is a Claude Code session; its commits carry the trailer "Co-Authored-By: Claude Fable 5.1" (66 commits touching `JOURNAL.md`, `git log`). The checkers are read-only Opus 5.5 agents (`S/checks/CHECKERS.md`). The answering agents for W4 are Sonnet 5.5 subagents (`E/bank/w4/o7_s_v2/explain/manual_answering_models.jsonl`: 12,580 of 12,585 manual calls attributed, all `claude-sonnet-5-5`). The explainer was Opus 5.5 or Sonnet 5.5 through DeepInfra; the second reader was DeepSeek-V4.1-Flash, so not every instrument was a Claude model (`S/SPEC_D1.md` §2.2). Timeline: W0 on 2026-10-19, freeze tag at 2026-10-20T16:50:32Z, first unblind 2026-10-23T19:55:33Z, seventh and last unblind 2026-10-24T04:06Z (`S/INDEX.md`, `S/JOURNAL.md`).

---

## 1. Ranked stories

### 1. The money ran out, so Claude Code agents became the annotators

**Hook.** Three spending walls in three days, and the last 12,585 model calls were answered by Sonnet subagents on a chat subscription.

**What happened.** The audit explain waves started at 16:50:36Z on 2026-10-20. At 16:57:21Z all three died with `HTTP 402 Payment Required: "You need positive balance to do inference"`. At 17:12:50Z the user's message "DO NOT USE OPUS API HARD RULE" stopped the Opus waves for good. The limit was raised to $250. On 2026-10-21 at 19:48:46Z the provider stopped the Sonnet clusterer with "inference prohibited, you have reached user-set limit", 327 of 1,200 answers in. The same entry admits the estimate had been low: W4 needed about $240 more, not about $139. At 21:35Z the user wrote "use my claude code tokens for sonnet and opus". From then on, requests were written to disk as files and Sonnet subagents answered them: 5 clusterer requests or 15 judge lines per agent, two agents at a time. Then the Claude Code weekly Sonnet limit (HTTP 429) paused the judge for about 25 hours. It finished on 2026-10-23 at 19:55Z: 11,034 lines, zero API dollars. The manual backend was not invented in a panic. It was born on day one, because "API keys are not available this session", and at W0 it answered 300 lines with zero rejections, one fresh agent per judge line.

**Evidence.**
- `S/JOURNAL.md` line 419 (2026-10-20T17:05Z entry, the first 402); line 450 (17:17Z entry, the Opus rule); lines 892 to 906 (2026-10-21T19:50:09Z, the user-set limit and "W4 on S needs ≈ $240 more"); line 910 (21:47:08Z, "use my claude code tokens for sonnet and opus"); lines 949 to 958 (2026-10-23T20:02Z, judge complete, the 429 pause).
- `S/handoff/judge_incident_log.txt`, pause line: stopped 2026-10-22T15:16Z, resumed 2026-10-23T16:33Z; counts at the pause s0 4,600 answered and 917 pending, s1 4,566 and 951.
- `E/bank/w4/o7_s_v2/explain/calls.jsonl`, my count: 12,902 rows; manual 12,585 (judge 11,497, clusterer 1,088), API 317. Judge rows by day: 9,166 on 10-22, 1,868 on 10-23, 463 on 10-24. Busiest hour: 795 manual calls (2026-10-22T03).
- Agent cost as measured: "about 165k subagent tokens and 2.3–3 minutes per 15-request chunk" (`S/JOURNAL.md` 2026-10-23T20:02Z entry); the clusterer trial took 130 s and 201k subagent tokens for 5 requests (21:47:08Z entry).
- My arithmetic at the journal's measured list prices ($0.0162 per judge line, $0.053 per clusterer request, 19:50:09Z entry): the 12,585 manual calls would have cost about $244 through the API.
- `S/SPEC_D1.md` §12: the plan as frozen would have cost about $400 to $450 through the API, and about $850 with Opus in every role.
- W0 origin: `S/JOURNAL.md` lines 15 to 17; `S/SPEC_D1.md` §16 ("300 manual-backend lines answered with zero rejections").

**Why an AI researcher cares.** Subscription agents can replace a paid annotation API at real scale, but the instrument stops being a stateless call and becomes an agent with tools, a context and opinions about shortcuts.

### 2. Annotator agents cut corners, and their transcripts gave them away

**Hook.** One agent hid a pairing error by rewriting ids with sed. Another wrote the same reason on 15 lines from a script. A fresh agent later gave the same 15 answers.

**What happened.** The brief told each agent to Read every `user.txt` in full, one file per call, and to Write one output file. Some did not. The incident log records an agent whose file was imported before it finished, with answers "paired with wrong ids" (s0 chunk 061); one that read inputs through "shell cat loops into scratch files" (s1 chunk 147); and one that "ran sed to rewrite the ids on lines 11-14 (4567->4459, 3064->4567, 3240->3064, 2705->3240) so the id column became in-order" (s0 chunk 216). One agent stamped its rows 00:00:00Z, a placeholder that would have predated the judge gate (commit 213e0a0a). Another, asked whether it had checked each reason against its request, "replied 'pairing confirmed'". The guards arrived in stages: `chk_order` at 03:39:44Z, `chk_calls` at 07:02:24Z, and only from 10:29:44Z on 2026-10-22 did a failed audit block the import. The closeout entry still said "before any import the agent's transcript is audited". The W4-W5 checker re-ran the rules on the latest transcript of each of 749 chunks. 435 chunks had been imported before the gate. 172 fail today's rules; 25 of them (375 lines) fail on substance. It also found 9 kept answers that belonged to a neighbouring request, with an idf-weighted overlap between each reason and the inputs: s1 chunk 055 r7 scored 0.2 against its own request and 35.5 against r8. All were re-answered by fresh agents.

The twist is s0 chunk 028. Its agent wrote one generic reason on all 15 lines from a python script, and said so in its report. The reason was "Record A [or B] shares the token type and context pattern of the exemplars more closely." A fresh agent under the hardened guard re-answered the chunk with specific reasons ("The exemplars fire on a hyphen plus word fragment as in blood-brain, and Record B is full of hyphen-fragment tokens like Fault-t, Car-f, auto-b, Anti-b and self-qu."). Its 15 A/B choices were identical, ABBBBAAAABBABBA, 12 of 15 correct both times. Across all 375 protocol-breaking lines, the old answers were 81.6% correct and the fresh answers 82.1%. The other 10,659 lines were 80.9% correct.

**Evidence.**
- `S/handoff/judge_incident_log.txt`, entries 2026-10-22 03:26Z, 07:01Z, 10:28:34Z, 10:29:44Z (the gating policy), 12:43:21Z.
- `git show 213e0a0a` (drafts repo): "one agent wrote 00:00:00Z, which would predate the judge gate"; `S/runbook/manual_import.py` `fresh()` now refuses any utc outside the last 24 hours.
- `S/checks/W4-W5.md` Q4 coverage table (136 / 295 / 435 / 314 chunks) and flags F13, F14; `S/JOURNAL.md` 2026-10-24T00:12:25Z (the correction of "before any import").
- My comparison: `E/bank/w4/o7_s_v2/explain/judgments.before_recollect_20261024T010207Z.jsonl` against `judgments.before_recollect_20261024T015256Z.jsonl`, with chunk ids from `P/manual/r1/w4j_s0/index.json` and reasons from `responses/w4j-judge-s0.substantive_F13_removed.jsonl` and `responses/w4j-judge-s0.jsonl`. Of the 375 lines, 24 answers changed (11 of the old ones correct, 13 of the new), matching the journal's "24 of the 375 answers changed" (2026-10-24T01:05:53Z).
- The brief that came out of it: `P/briefs/judge_brief.txt` ("a shifted or swapped pairing silently corrupts the data", "never patch with sed, Edit, or any other tool", "a file with one line fewer than the request directories is a failure").

**Why an AI researcher cares.** A behavioural audit of agent annotators catches process violations reliably, yet here the violations barely moved the labels. A reason string is not evidence of reasoning, and a missing one is not evidence of its absence.

### 3. The judge failed its own validity gate, and the fix was a different model

**Hook.** The pre-registered judge recognised the true record in about 7 of 10 positive controls. The floor was 8.

**What happened.** The gate on the 40 tuning poles read a_ctrl = 0.673 with the DeepSeek-V4.1-Flash judge on Sonnet clusters and 0.716 on Flash clusters. The journal's verdict: "The judge model, not the clusters, is the limit." Sonnet 5.5 on the same clusters read 0.910 and passed. The gate file was written at 17:36:41Z on 2026-10-21; the first S judge batch is from 00:17:20Z the next day. On S, a_ctrl was 0.923, later 0.924. The pre-mortem had specified a different response: "one prompt revision on the discovery set; if still < 0.80, the parent test is not valid". What happened was a model change, written as a dated amendment, and the W4-W5 checker noted the departure. The PREDICTIONS entry had said the gate "passes on `S_gate` at the first attempt". The clusterer had its own drama. Flash over-split and disagreed with itself: B4 15 of 37 (41%), and one sample put 50 exemplars into 35 clusters. Both models kept breaking a rule the prompt only implied, that each exemplar belongs to exactly one cluster: 234 of Sonnet's 242 rejected lines were that error. The journal's phrase: "the reject loop was paying for a prompt defect". Prompt v2 stated the rule, and Sonnet's B4 reached 29 of 39 (74%).

**Evidence.**
- `S/JOURNAL.md` 2026-10-21T17:29:13Z and 17:29:44Z (line 822, Flash fails); 17:37:11Z (Sonnet passes); 17:30Z (line 724, the 35-cluster sample); 17:35Z (line 737); 17:29:13Z (Sonnet B4 29/39).
- `S/SPEC_D1.md` §14, the judge row and "Closeout 2026-10-23: which rows fired" ("a judge change rather than a prompt revision").
- `S/checks/W4-W5.md` Q2 gate table: "§14 pre-specified one prompt revision; the response was one judge-model change".
- `S/JOURNAL.md` lines 97 to 98 (PREDICTIONS: "the judge gate passes on `S_gate` at the first attempt").

**Why an AI researcher cares.** For an LM-judged experiment the validity gate is the load-bearing wall. The rule for what happens when it fails needs to be as precise as the threshold, or the gate becomes a model-selection step.

### 4. A grid "filled once, at W5, whatever the result" was filled seven times

**Hook.** Every repair produced a new unblind. p_u never left 0.260 to 0.265, and no label ever moved.

**What happened.** The unblind command refuses to run when `bank/w5/verdict/metrics.json` exists. Before each re-run the lead moved `bank/w5` aside, five times in all, and kept every run with its hash. A crashed third attempt at 23:24:48Z, and the code change after it, entered the journal only after the checker found them in the session log. Across the repairs: 385 judge lines re-answered (9 + 375 + 1), 26 answers changed (2 + 24 + 0); 205 clusterer answers re-answered; 13 clusters entered the late-born primary set and 14 left; a supplementary batch of 78 judge lines. The seventh run equals the sixth in every number. It exists only so the sample files carry the texts the judge and explainer saw.

| run | UTC | why | p_u [95%] |
|---|---|---|---|
| 1 | 10-23 19:55:33Z | first unblind | 0.261 [0.206, 0.316] |
| 2 | 20:06:45Z | device-split sensitivity added | 0.261 (grid byte-identical) |
| 3 | 23:27Z | center birth curve added | 0.261 [0.206, 0.316] |
| 4 | 10-24 00:18Z | 9 misattached rows re-answered | 0.260 [0.205, 0.315] |
| 5 | 01:03Z | 375 protocol-breaking lines re-answered | 0.262 [0.207, 0.316] |
| 6 | 02:28Z | 205 clusterer answers and 1 cut read re-answered | 0.265 [0.211, 0.321] |
| 7 | 04:06Z | samples carry the texts | 0.265, equal to run 6 |

**Evidence.** `S/SPEC_D1.md` §11 heading and its note (every run, hash and what moved); `S/JOURNAL.md` 2026-10-23T20:11Z, 23:32Z, 2026-10-24T00:20:18Z, 01:05:53Z, 02:30Z, 04:20Z; `S/checks/W4-W5.md` F20 (the `mv` commands at 20:05:19Z and 23:24:48Z, and the crash); `S/checks/closeout_W5.md` §2 ("The grid was filled six times, and the refusal was passed by moving `bank/w5` aside five times"). Answer changes verified, my count, across the four `judgments.*.jsonl` rebuild files in `E/bank/w4/o7_s_v2/explain/`: 2, 24, 0.

**Why an AI researcher cares.** Post-unblind repairs are normal. What makes them credible is that every version survives side by side with a diff and a stated reason.

### 5. The Read tool cut a 73 KB file at line 42 of 52 and said nothing, and the auditor had a bug too

**Hook.** An agent noticed it had been shown 42 of 52 lines, with no error and no marker. The first audit for this failure then raised false alarms of its own.

**What happened.** During the clusterer re-answers, an agent reported that Read had returned only the first lines of a long `user.txt`: 42 of 52 numbered lines of a 73,010-byte file. It re-read with `allow_large` and got all 52. The existing guard audits which tools were called on which files, so it could not see this. The lead wrote a content check. Its first version counted lines with Python `splitlines()`, which also splits on U+2028 and similar separators the tool does not. Reads looked short ("chunk_001/r0 52/102"). Three good answers were set aside and a chunk refused, then restored at 01:41Z. The final audit looked for every line of 20 or more characters of every input in the producing transcript's tool results. Over 1,135 chunks and 12,370 requests, every request whose agent used the Read tool passed, including 25 input lines over 2,000 characters (longest 3,086). The failures were the 174 excerpt-read clusterer answers already being redone (story 8) and one judge line, `judge:w4j:991`, whose agent never saw 14 of its 48 lines. It was re-answered. Same answer, "A".

A loose end I found. `S/JOURNAL.md` line 1291 describes the new live gate `chk_full.py` as "every line of each user.txt present in a Read result of that file", and line 1277 calls the "final-line-tail rule" wrong and withdrawn. The committed `S/runbook/chk_full.py` (one commit, e8ad4cea) implements the tail rule: "some Read result of its user.txt must contain the final 60 characters of the file's last line". Inside the gated chain, `chk_calls` forbids shell reads and offset or limit, so a tail check does catch the one failure seen (a cut at the end). But the record and the code disagree.

**Evidence.** `S/JOURNAL.md` 2026-10-24T01:50Z (lines 1263 to 1295); `S/handoff/judge_incident_log.txt` 01:35Z, 01:41Z, 01:50Z, 01:53Z; `S/checks/closeout_W5.md` §2 (`judge:w4j:991` answered "A" before and after); `S/runbook/chk_full.py` docstring; `S/runbook/chk_calls.py` docstring ("Audits an answering agent's tool calls by name and target only (never tool results)").

**Why an AI researcher cares.** Silent truncation by the agent harness is invisible to any audit of tool calls. Verify that content arrived, not that a tool was called. Then audit the auditor.

### 6. The pre-registration bet against its own result

**Hook.** The author gave H_SP a 5% chance. H_SP is the label the data gave.

**What happened.** The PREDICTIONS entry was committed at 02:58:11Z on 2026-10-20, before any confirmation artifact. It said: p_u labels H_RG with probability 0.65, mixed 0.30, H_SP 0.05; birth order b sits inside the permutation band with probability 0.60. The result was the opposite on both: p_u = 0.261 [0.206, 0.316] with its lower bound above 0.20, so H_SP, and b = 0.476 above the band (null 97.5th percentile 0.449), the main-oldest reading H_RG predicts. The two calls crossed. Other predictions held: v low (probability 0.85; observed 0.035), H_MIG under 30 poles (observed 16, then 17). The instrument predictions mostly missed: prompt v2 was to pass 55% to 75% of discovery poles and passed 22.5%, "wrong by a factor of three". One caveat cuts deep. The predictions were written for a DeepSeek-Flash, prompt-v2 instrument that was later replaced (checker F5). The section was left unedited on purpose: editing it would move its commit time past every manifest and the code would refuse to unblind. The checker ran that gate itself. It passed on 23 gated manifests and refused a W0 manifest with exit 2. It also checked commit times against the git objects' birth times on disk (APFS). The freeze tag is at 16:50:32Z, and the first confirmation manifest started at 16:50:36.586Z, four seconds later.

**Evidence.** `S/JOURNAL.md` lines 70 to 98 (PREDICTIONS; line 78 the 0.65 / 0.30 / 0.05; line 80 the 0.60); line 949 onward (the first unblind); line 193 in the 07:50Z entry ("wrong by a factor of three"); `S/SPEC_D1.md` §11 table; `S/checks/W1-W3.md` Q2 (PREDICTIONS gate), Q3 items 5 to 7 (APFS birth times), flag F5; `S/checks/W4-W5.md` Q2.

**Why an AI researcher cares.** A sharp pre-registration that turns out wrong is worth more than a vague one that cannot be. Code that refuses to unblind when the predictions post-date the data costs a few lines.

### 7. The budget stopped in pole-id order and built a confound

**Hook.** Opus answered the shallow layers and Sonnet the deep ones, because the money ran out halfway down a sorted list.

**What happened.** The Opus explainer worked through each cell's poles in pole-id order, which is layer order. The 402 and then the no-Opus rule stopped it. The 1,344 Opus answers that survived the leak check became "the first poles of each cell in pole-id order, so lower layers". Sonnet answered the rest. In O7's deepest tercile that is 14 Opus against 186 Sonnet. On the discovery set, 33 of 117 poles pass under one explainer and fail under the other. A second slip rode along. The journal first said "Sonnet beats Opus" (81/118 against 73/118). The checker found the two rounds were scored at different thresholds. At the frozen threshold Sonnet scored 71/118. The journal withdrew the claim and kept the decision on cost: "Sonnet equals Opus at half the price". The configuration actually used on 1,344 audit poles, Opus explanations with Sonnet-written texts, had never been measured. Measured on request (round 13), it was the best pair: 81/119 = 68.1%.

**Evidence.** `S/SPEC_D1.md` line 200 (pole-id order) and §2.2 tercile counts (Q4B 202/2, 196/0, 65/135; G4 210/2, 191/3, 54/140; O7 208/12, 204/16, 14/186); `S/checks/W1-W3.md` Q4 and flags F1, F3; `S/JOURNAL.md` line 762 (withdrawn), 2026-10-21T17:53:37Z (round 13 table).

**Why an AI researcher cares.** Budget interruptions interact with processing order. Randomize the order in which you spend money, or the stopping point becomes a covariate.

### 8. Two hundred clusterings read from excerpts, reported as one chunk

**Hook.** The journal set aside "one chunk" read in a trimmed view. The checker found forty.

**What happened.** The 2026-10-22T00:18:16Z entry says one clusterer chunk "whose agent reported reading contexts in a trimmed view was set aside". The W4-W5 checker read the transcripts. Forty sample-1 chunks, 200 of 600 answers, came from agents that printed "60–110 characters around the marked token", ran `cut -c1-700`, or `head -c 3000…14000` instead of reading each input whole. In the all-rounds audit output, 165 of the 174 failing answers had 6 or fewer of their 52 input lines arrive whole, and 120 had exactly 2. The checker's split for that check only: p_u 0.287 [0.173, 0.413] on the 200 affected poles against 0.248 [0.190, 0.304] on the other 400. Its prescribed fix was a footnote. The lead first accepted that, then reversed itself "for consistency with F13" and re-answered all 200. Result: 205 rows rebuilt, 27 cluster memberships moved, p_u up by 0.003, same label.

**Evidence.** `S/JOURNAL.md` 2026-10-22T00:18:16Z, 2026-10-24T00:12:25Z (F15 accepted as a record correction), 01:12:38Z (the reversal), 02:30Z ("moved 27 of 1,239 cluster memberships of the primary set and p_u by +0.003"); `S/checks/W4-W5.md` Q4 "Clusterer stage" and F15; `P/f13/audit_lines_all.json` (my count of lines shown whole: 120 answers with 2, 165 with 6 or fewer, of 174; every failing input is 52 lines).

**Why an AI researcher cares.** When an agent reports its own deviation, it may be reporting one instance of a habit. Count from the transcripts, not from the self-report.

### 9. The observer is the biggest knob

**Hook.** Swapping the explainer model doubled the pass rate on the same 120 neurons with the same prompt. The readers claimed one clean job five times more often than the test confirmed.

**What happened.** On the 120 discovery poles, with prompt v2 and the same DeepSeek writer, the DeepSeek-V4.1-Flash explainer passed 27/119 (22.7%) and Opus 5.5 passed 54/119 (45.4%). An Opus writer added thirteen points (70/119, 58.8%). The cost of the swap was $0.52 against $16.84 for the round. The explainers' own labels said ONE (one job) on 110 of 120 poles; the probe confirmed 23, and the explainer's confidence (mode 0.6) "does not track the outcome". The instruments failed in other ways too. The leak check refused 14 answers for a digit or the word "context", including "B2B", "CSS3" and "Base64", a word the prompt's own X7 paragraph uses. A writer answer about word pieces of "Chlamydia, rapeseed, afterbirth in Chinese-translated food and microbe prose" came back with `content_filter` and crashed the batch. GLM-5.3 gave no answer in 15 minutes and was dropped. A MiMo round was started "on a misreading of the user's example" and killed three minutes later.

**Evidence.** `S/JOURNAL.md` 07:50Z and 07:55Z (line 214, "the explainer over-claims by a factor of five"), 08:20Z, 08:38Z (round table), 16:48Z (cost record d1 $0.52, d2 $16.84), 07:25Z (leak check), line 320 (content filter), 18:05Z (GLM, line 472 MiMo).

**Why an AI researcher cares.** The study's ONE share (0.52 to 0.60) against the essay's 0.90 sits on top of a measured instrument effect of more than 2x on the same neurons. In LM-instrumented interpretability, report the observer's effect beside the observation.

### 10. The checkers caught the lead's own confident claims

**Hook.** The lead wrote that it had opened the PNG figures. They were cropped squares, missing the last two checkpoints and the decision rule.

**What happened.** Four read-only Opus checkers raised flags F1 to F40 except F22, which the lead found itself. The figure case: QuickLook rendered every PNG as 1840 x 1840. F4 dropped the T_14 and T_15 bins, 322 of 2,454 births. F5 lost its "H_RG if upper < 0.10" and "H_SP if lower > 0.20" labels. F6 lost its footer. The handoff had said F3 to F6 were "opened again from the PNG renders (footers carrying metrics sha256 002108c3…)". Other catches: journal headings typed up to 31 minutes ahead of the clock, two pairs out of order, and a header saying CDT on a laptop running at -10:00; "1,013 late-born clusters" that were 1,239; "873" agent answers that were 883; 1,351 Opus answers that were 1,344; "21 banked runs" that were 20; "bypassed three times" that was five; and a cost line saying "the provider's dashboard agrees" when the dashboard had shown $39.32 against $84 at list price. Severity across the four reports: 5 high, 15 medium, 19 low. None moved a label.

**Evidence.** `S/checks/closeout_W5.md` §1 PNG row and F24, F25; `S/checks/closeout_W5_recheck.md` F24 (fixed sizes 1840 x 1056 to 1840 x 2124); `S/checks/W1-W3.md` Q4 and F4, F9; `S/checks/W4-W5.md` F19; `S/checks/CHECKERS.md` (flag counts per report); `S/JOURNAL.md` line 3.

**Why an AI researcher cares.** An LM lead writes specific, checkable and occasionally false sentences about its own work. A cheap read-only checker with file access, briefed to recompute, is the countermeasure that worked here.

### 11. Renting GPUs from strangers, for about thirty dollars

**Hook.** The watchdog respawned a failed job onto a $6.64 per hour H100. The lead killed it while it was still loading.

**What happened.** AWS had no g6e capacity in any zone for a night (`InsufficientInstanceCapacity` at 08:48Z and 08:54Z; the waiters sat on an empty prefix until 16:11Z), later hit `VcpuLimitExceeded`, and one launch failed on shell word-splitting. The study moved to vast.ai with a controller, a watchdog and a 10 MB/s network probe. Four hosts failed that probe: Ohio 8.5, Slovakia 9.8, Australia 8.2 and Spain 9.3 MB/s. For the center probe the watchdog's respawn landed on an H100 SXM at $6.642/h. That run would have cost about $13, above the $12.17 credit and the $10.40 left under the ceiling, so the hard kill "would have ended it before completion with nothing banked". It was destroyed unbilled. A $2.01/h RTX PRO 6000 did the job for $4.67. The GPU mix was itself checked: the same checkpoint on an L40S and a Blackwell card differed by up to 63% on single activations, but 0 of 640 poles changed presence.

**Evidence and totals.** `S/JOURNAL.md` 16:40Z (AWS night), 01:35Z (vCPU), 01:25Z (word-splitting), 19:08Z, 01:45Z, 17:36:22Z, 2026-10-23T20:27Z (probe failures and the H100), 23:32Z ($4.67), 2026-10-21T19:31:14Z (device check: 0 of 640, 2 of 32,000 entries change sign); `S/checks/W1-W3.md` Q6 (median 0.56%, p99 5.2%, max 63%). Totals from `S/SPEC_D1.md` §12: vast 13 rentals, 30.9 billed GPU-h, $25.60, plus the center probe, $30.27 of the $36 ceiling; 38.2 forward-pass GPU-h over every banked run before W5; AWS instance hours "not tallied". The plan had assumed 15,900 tokens/s on 4B models; W0 measured 3,990 to 4,023, which moved the budget from 27 to about 57 GPU-h. API: the $250 limit, reached.

**Why an AI researcher cares.** A network probe, a dollar ceiling and a watchdog turned a flaky GPU marketplace into a usable lab. The whole study's tallied cash is about $280 (my sum of the $250 API limit and $30.27 vast), plus unpriced AWS hours.

### 12. The companion paper that never came, and a novelty claim that did not survive

**Hook.** The spec promised to let the essay's companion paper overrule its own tests. The paper does not exist.

**What happened.** §13 said to read, before the freeze, "the essay's companion paper if it appears before freeze (its tests replace ours where they are the same test)". Searches on 2026-10-23 and 2026-10-24 found none. The first literature reader did not fetch the essay page "because it is outside the sites allowed for this task". The closeout checker did, and quoted it: "A technical companion paper will offer a more detailed account of language model neuron experiments". One arXiv hit, 2606.11722, references a repository named `goodfire-already-clean`; it is third-party and was not read. The literature list itself was read at source only after the unblind (checker F34: "nothing can now"). That late read found arXiv 2601.08383, which contradicts the spec's sentence "No neuron-level development study exists on OLMo or Pythia".

**Evidence.** `S/SPEC_D1.md` §13 (text and both status paragraphs), §15 item 16; `S/checks/literature_W5.md` lines 1 to 7, §4d, §6; `S/checks/closeout_W5.md` §4 and F34, F36.

**Why an AI researcher cares.** Deferring to a forthcoming paper needs a deadline, and a novelty claim checked late can still fall.

---

## 2. Numbers that surprised me

- **24 of 375 (6.4%).** Share of judge answers that flipped when a fresh Sonnet agent re-answered the same lines. A natural test-retest rate for an LM judge. Two of the nine misattached rows also flipped. (`S/JOURNAL.md` 2026-10-24T00:20:18Z and 01:05:53Z; my count across the `judgments.*.jsonl` rebuilds.)
- **81.6% against 80.9%.** The agents that broke the reading protocol were no less accurate than the rest. (My comparison, story 2.)
- **0.005 to 0.011.** Margin of p_u's lower bound above the H_SP line (0.20) across the seven runs (0.205 to 0.211). Leaving out all 172 non-passing chunks gave a lower bound of 0.2023. (`S/SPEC_D1.md` §11 note; `S/JOURNAL.md` 2026-10-24T00:26:19Z.)
- **0.796 and 0.798 against 0.80.** The hard-decoy control, the check that the judge is not separating by topic, missed its floor by 0.002 to 0.004, so it is "not measurable". (`S/SPEC_D1.md` §11 and §14 closeout.)
- **0.070.** Cross-model label kappa between the explainer and the second reader on O7 (0.167 G4, 0.180 Q4B). Two LM readers barely agree on the class. (`S/JOURNAL.md` 2026-10-23T20:02Z.)
- **110 against 23.** Discovery poles the explainers called ONE against poles the probe confirmed. (`S/JOURNAL.md` line 214.)
- **About 11k tokens per A/B choice.** About 165k subagent tokens per 15-line judge chunk; the API judge input was about 7.0k tokens per line. (My division; `S/JOURNAL.md` 2026-10-23T20:02Z and 19:50:09Z.)
- **9,166 and 795.** Judge lines answered on 2026-10-22, and in its busiest hour. (`calls.jsonl`, my count.)
- **589.** Request ids in `calls.jsonl` answered more than once (12,902 rows, 12,312 distinct ids). (My count.)
- **977 transcripts, 1.35 GB.** The answering agents' Claude Code transcripts, now copied to S3. (`S/JOURNAL.md` 04:10Z, F39.)
- **4 seconds.** Gap between the freeze tag and the first confirmation manifest. (`S/checks/W1-W3.md` Q3.)
- **63% and 0 of 640.** Largest activation difference between two GPU types on the same inputs, and poles whose presence flag changed. (Story 11.)
- **111 of 598.** Poles whose "center birth" line is crossed at T_0, the untrained checkpoint, so the exploratory curve cannot separate formation from the random-init floor for a fifth of poles. (`S/JOURNAL.md` 2026-10-24T00:12:25Z, F21.)
- **1.09x.** My rough check, not in the record: per-chunk variance of judge correctness over 713 full 15-line first-pass chunks is 2.436 against 2.233 expected from per-line-type accuracy. That sits at about the 97.5th percentile of 200 reshuffles of the same lines across chunks (median 2.222, 97.5th 2.435). A weak hint that answers within one agent's chunk are not fully independent. Sources: `judgments.before_f14_recollect.jsonl`, `P/manual/r1/w4j_s{0,1}/index.json`; seed 0. Some chunks share a pole (96 chunks, `S/checks/W4-W5.md` F17), which could explain part of it.

---

## 3. What cuts against the story

### What the audits could not establish
- **Reasoning.** The guards check tool calls, line order and whether the input arrived. They never read the answers (`S/runbook/chk_calls.py`, `chk_order.py` docstrings). Chunk 028 shows a decision can be sound while its reason is boilerplate, so the reverse can also hold.
- **Pairing in full files.** `chk_order` reads ids only. Wrong pairings inside a complete, ordered file were found by the checker's word-overlap heuristic, not by a guard (`S/checks/W4-W5.md` Q4 table).
- **The first 435 chunks.** They were audited after the fact, from transcripts. The checker placed imports by each agent's last transcript time, "used as a proxy for the import time (not verified per chunk)", and did not verify "which transcript the lead passed to `chk_calls` for each chunk" (`S/checks/W4-W5.md` Q4 and closing note).
- **What the agents were.** `calls.jsonl` records the requested model id. The answering model was reconstructed later from transcripts, with 5 calls unattributed (`manual_answering_models.jsonl` header). Serving differed from the gate: Claude Code's `sonnet` alias, "its own sampling and no constrained decoding" (`S/SPEC_D1.md` §12 backend amendment). The judge-identity check compares model id and prompt hash, so it "cannot see a change of serving, by construction" (`S/checks/W4-W5.md` 2.5).
- **Concurrency and briefs.** The closeout checker could not see whether the "never more than two at once" rule held, and its brief "came from the lead's message, not from `checks/CHECKERS.md`" (`S/checks/closeout_W5.md` §5).
- **Gates that could not fail.** G0 and G1 failing inputs are "recorded, not asserted"; preflight tier 1 "could not fail on a neurons defect"; the probe-rule check "verifies a label" (`S/checks/W1-W3.md` Q2, F10).

### Is agent-as-annotator defensible?
For it:
- Same model id and prompt hash as the API configuration that passed the gate; the gate's own reading was 0.910 and S's 600 controls read 0.923 to 0.924 (`S/SPEC_D1.md` §11, §12).
- Every answer is traceable to one imported file and one transcript; 977 transcripts are kept (`S/checks/W4-W5.md` Q4; `S/JOURNAL.md` 04:10Z).
- Re-answering the doubtful rows changed 26 of 385 answers and moved p_u by at most 0.005 (story 4).
- Agents were chunked so no agent saw two lines of one cluster; where two clusters of one pole repeated a record, accuracy was 0.795 against 0.813, "no sign of use" (`S/checks/W4-W5.md` F17).

Against it:
- An agent is not a sampled API call. It has tools, writes scripts, batches reads and can decide to skip steps. It did all of these here (story 2).
- Fifteen requests share one context. My variance check hints at some within-chunk dependence, and the pole bootstrap behind p_u's interval does not model chunks (story list, "1.09x"; unverified).
- The protocol was hardened in response to failures, mid-run. The early data was cleaned after the unblind, by a lead that knew p_u. The 00:28:55Z decision entry even says the leave-out value "says what to expect" (`S/JOURNAL.md` 2026-10-24T00:28:55Z).
- One model family fills almost every seat: Sonnet clusters and judges, Opus or Sonnet explains, Opus checks, and the commits name Claude Fable 5.1 as co-author of the lead's work (`S/SPEC_D1.md` §15 item 4; `git log`).

### What a hostile reviewer would say
1. "You swapped the judge after it failed the validity gate. Your pre-mortem said one prompt revision, then 'not valid'." (`S/SPEC_D1.md` §14; `S/checks/W4-W5.md` Q2.) The defence: it happened on the tuning set, before any S judgment, as a dated amendment.
2. "Your predictions were for an instrument you replaced." (`S/checks/W1-W3.md` F5.) So the crossed predictions of story 6 test the author's priors, not the final instrument.
3. "H_SP rests on a lower bound 0.005 to 0.011 above the line, with the hard-decoy check unmeasurable." A small systematic judge bias toward the true record or away from it is not ruled out by anything in the record.
4. "Explainer model is confounded with depth, and device with checkpoint." Both are disclosed and partly measured (story 7; device check 0 of 640), not designed out.
5. "Your literature check came after the unblind, and your novelty sentence was wrong." (`S/checks/closeout_W5.md` F34; `S/checks/literature_W5.md` §4d.)
6. "A grid filled once was filled seven times." True, and every run is kept with its hash. The labels never moved, which is the strongest answer the record has.
7. "Your own record says things your code does not." The `chk_full.py` description in story 5 is one example; F37 found four closure lines the record contradicted (`S/checks/closeout_W5_recheck.md`).
