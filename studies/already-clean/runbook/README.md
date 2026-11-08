# Runbook scripts (copied from the session scratchpad, 2026-10-21)

Control-plane scripts that ran the study from the laptop. Paths inside them point at the three worktrees
(`interp_research-neurons-explain`, `interp_research-neurons-harvest`, this drafts tree) and at the session
scratchpad; adjust `SC`/`NE` when re-running from elsewhere. The key is read from
`~/.config/interp_research/keys.env` (`NEURONS_API_KEY_1`), never from a file in any repo.

| file | what it does |
|---|---|
| `ne` | wrapper: `uv run … python -m infra.pg.neurons_<explain|llm|verdict>` in the explain worktree |
| `split_sample.py` | discovery/audit split of a harvest's sampled neurons (§4.4) |
| `q1_explain_wave.sh` | one Q1 explain wave: prepare → explainer+second reader (reject loop) → writer+job writer (reject loop) |
| `w2c_explain.sh` | W2-C wave per cell (poles, harvest dirs, batch c1) |
| `w2c_finish.sh` | W2-C under the model policy amendment: writer chain on Sonnet or second-reader chain on Flash |
| `w2c_score.sh` | upload a cell's texts and launch its scoring job on AWS or vast |
| `w4_chain.sh` | W4: preflight, S_gate clusters, S_gate judge + gate, S clusters, S judge |
| `w4_w5_resume.sh` | W4 on S and W5 end to end; retries on the provider's HTTP 402 only, stops on anything else |
| `manual_chunks.py` | chunk a batch's pending requests for Claude Code answering agents; no two of one plan group (pole, cluster) per chunk |
| `manual_import.py` | import agents' chunk answers into a mixed batch's responses with `backend: manual` |
| `manual_cycle.sh` | import all chunk outputs, `reject --backend mixed`, print pending per batch |
| `settings/` | the DeepInfra settings files used for the calibration rounds and the audit (also committed on `neurons/explain`) |
| `steps/` | rewritten step manifests for the vast mirror, the W2-D plan and the prompt v3 draft |
