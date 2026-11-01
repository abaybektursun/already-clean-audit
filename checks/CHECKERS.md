# Checkers (SPEC_D1 §9.6)

Read-only, non-blocking sanity checks after each wave, run as Opus agents only, never more than two at once. A checker
reads the spec, the journal, the data contract, the code and the OPEN banks (discovery sets, S/S_gate draw, W1 and W3
manifests and samples); it never opens a sealed confirmation output (W2-C explanations, texts, probe scores; W4; W5)
before W5, and says "deferred to W5" where a check would need one. It writes `checks/<waves>.md` with a summary table
(check, verdict pass/flag/deferred, one line), evidence per question, and ranked flags with the exact fix each needs.
The lead answers each flag in a dated journal entry (accepted with the fix, or declined with the reason).

| report | waves | run | flags |
|---|---|---|---|
| `W1-W3.md` | W0–W3 and the explain side of W2-C | 2026-10-21, 16:55–17:15Z | 12 (2 high, 2 medium, 8 low); answered in the journal entry of 2026-10-21T17:19:15Z |
| `W4-W5.md` | W4 (S clustering and judge on the manual backend) and W5 (three unblind runs, results, the post-unblind additions) | 2026-10-23 23:40Z – 2026-10-24 00:05Z | 9 (0 high, 3 medium, 6 low); answered in the journal entry of 2026-10-24 "Answers to the W4–W5 checker" |
| `closeout_W5.md` | closeout after the sixth unblind run: the S6 acceptance table row by row, §11 against grid.md and metrics.json, the derived run logs, §13 item 6 with its own search, §9.4 and §9.6 | 2026-10-24 02:54–03:10Z (Opus, read-only, detached worktree at 39260448) | 14 (F23–F36: 3 high, 9 medium, 2 low); answered in the journal entry of 2026-10-24 03:30Z |
| `closeout_W5_recheck.md` | re-check of what was done for F23–F36 (pointers followed to the request lines, PNGs measured and opened, §15 closures' cited commits, registry hashes, run logs, answering models) | 2026-10-24 03:27–03:48Z (Opus, read-only, detached worktree at c86e308c) | F23–F36: 5 resolved as written, 9 in substance, 0 open; 4 new (F37–F40: 1 medium, 3 low), answered in the journal entry of 2026-10-24 04:10Z |
