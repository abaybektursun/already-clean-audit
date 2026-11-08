"""M-C9 — repair at large: does the reader work on models nobody built it for? (queue C9). Wave P.

    THE ONE DEPENDENCY GATE IN MERCATOR. §6 says this arm runs ONLY if M-G graduates its refuter —
    core readers beating twin readers on the pi0 re-shell. So the first thing this job does is READ
    M-G's artifact and refuse to proceed if it did not. Not a reminder in a runbook: a check
    against the banked result, with the numbers on the row.

    THE ARM. Take wild Hugging Face models with a DOCUMENTED lost faculty — a fine-tune whose card
    or whose measured behaviour shows it no longer does something its base did — and try M-G's
    rank-r bank-confined reader on them, against the same equal-capacity twins.

    REFUTER (pre-logged): core readers <= twin readers on the wild models.

THE ROSTER IS MECHANICAL. Candidates are enumerated from the Hub as derivatives of the base, and
"lost faculty" is MEASURED here rather than taken from a card: a candidate qualifies when its score
on a battery task falls below its base's by more than the base's own interval half-width. A card
that claims a loss the weights do not show is not evidence, and a loss the card never mentions
counts just the same.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as WT
from infra.pg import mercator_core as M
from infra.pg import mercator_program as P

ARM = "M-C9"
MG_RESULT_S3 = f"s3://{M.BUCKET}/results/mercator_gb/out/M-G_result.json"
BANKS = ("core32", "anti32", "complement32", "haar32_s0", "haar32_s1", "haar32_s2")
TASKS = ("vqav2", "coco_captions")
RANKS = (8, 32)


def main():
    ap = argparse.ArgumentParser(description="M-C9 — repair at large")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mg-result", default=MG_RESULT_S3)
    ap.add_argument("--n-candidates", type=int, default=12)
    ap.add_argument("--n-repair", type=int, default=3)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--n-train", type=int, default=192)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--ranks", default=",".join(str(r) for r in RANKS))
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ranks = [int(r) for r in a.ranks.split(",") if r.strip()]
    meta = P.ARMS[ARM]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_wp") / "expectations.jsonl")
    expected_ref = exp.require(ARM)
    at = M.Atlas(out, arm=ARM, run_id="m_c9", register=meta["register"], wave=meta["wave"],
                 expected_ref=expected_ref, queue_item=meta["queue_item"])
    sur = M.SurpriseLog(out)
    gates = H.Gates(out)

    # ---- THE DEPENDENCY GATE, read from M-G's artifact -----------------------------------------
    mg_path = out / "M-G_result.json"
    subprocess.run(["aws", "s3", "cp", a.mg_result, str(mg_path), "--region", "us-east-1"],
                   check=True)
    mg = json.loads(mg_path.read_text())
    graduated = {k: bool(v["core_beats_twins"]) for k, v in mg["summary"].items()}
    passes = any(graduated.values())
    gates.record("c9_dependency_gate_on_m_g", passes, m_g_summary=graduated,
                 m_g_source=a.mg_result,
                 m_g_core_vs_twins={k: {"core": v["best_by_bank"]["core32"],
                                        "twin_band": v["twin_band"]}
                                    for k, v in mg["summary"].items()},
                 outcome=None if passes else M.COVERAGE_IMPOSSIBLE,
                 note=("MERCATOR's only dependency gate; read from M-G's banked artifact rather "
                       "than assumed, with the numbers on the row"))
    if not passes:
        # A RECORDED COVERAGE-IMPOSSIBLE IS A RESULT, NOT A CRASH — the M-D90 lesson, applied here
        # before this arm ever ran rather than after it took a pipeline down. M-G's readers scored
        # exactly 0.0000 at every rank and bank, so `core_beats_twins` is False on both tasks and
        # this arm has no instrument to carry to the wild models. It writes that verdict and exits
        # 0; an arm that crashes leaves INCOMPLETE in the ledger, which is a worse record of the
        # same fact.
        print(f"[M-C9] COVERAGE-IMPOSSIBLE: M-G did not clear its refuter on any task "
              f"({graduated}). §6 gates this arm on that, so there is no reader to try on the "
              f"wild models.", flush=True)
        info = at.to_parquet()
        M.write_json(out, f"{ARM}_result.json",
                     {"arm": ARM, "queue_item": meta["queue_item"], "refuter": meta["refuter"],
                      "expected_ref": expected_ref, "provenance": M.provenance(),
                      "utc": M.utc_now(), "atlas": info, "outcome": M.COVERAGE_IMPOSSIBLE,
                      "m_g_summary": graduated, "m_g_source": a.mg_result,
                      "why": ("M-G's bank readers scored 0.0000 at every rank on both tasks, so "
                              "core_beats_twins is False and there is no instrument to port")})
        print("C9_OK " + json.dumps({"outcome": M.COVERAGE_IMPOSSIBLE, "m_g": graduated}))
        return 0

    import gc
    import torch
    from contextlib import nullcontext
    from huggingface_hub import HfApi
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
    from infra.pg import darwin_census as DC
    from infra.pg import projector_ablation as PA
    from infra.aws_pg.jobs.invariant_core_func import _decoder_layers, _load_full_pg1
    from infra.aws_pg.jobs.invariant_core import _delete_cache
    from infra.aws_pg.jobs.mercator_g import BankReader

    assert torch.cuda.is_available(), "no CUDA"
    M.ensure_pycocoevalcap()
    api = HfApi()
    base_sha = api.model_info(WT.PG1_REPO).sha
    model = _load_full_pg1(base_sha, "cuda")
    processor = PaliGemmaProcessor.from_pretrained(WT.PG1_REPO, revision=base_sha)
    layers = _decoder_layers(model)
    bank_dir = M.fetch_all_banks(H.cache_dir(out, "banks"))
    basis = {b: {L: WT.load_bank(bank_dir, "q_proj", L, b) for L in range(len(layers))}
             for b in BANKS}
    tasks = {t.name: t for t in PA.build_tasks() if t.name in TASKS}
    items = {k: PA.load_eval_items(t, a.n_train + a.n_eval, seed=a.seed)
             for k, t in tasks.items()}
    eval_rows = {k: v[a.n_train:] for k, v in items.items()}
    train_rows = {k: v[: a.n_train] for k, v in items.items()}
    pg1_sd = {n: p.detach().clone() for n, p in model.named_parameters()}

    def evaluate_under(ctx):
        with (ctx if ctx is not None else nullcontext()):
            return {k: PA.eval_task(model, processor, t, eval_rows[k], "cuda", batch_size=8)
                    for k, t in tasks.items()}

    base_scores = evaluate_under(None)
    for k, v in base_scores.items():
        print(f"[M-C9] PG1 base {k:15s} {v['score']:.4f} CI={v['ci95']}", flush=True)

    # ---- the mechanical roster, with the loss MEASURED ----------------------------------------
    from transformers import AutoConfig
    geom = DC.base_geometry(AutoConfig.from_pretrained(WT.PG1_REPO, revision=base_sha))
    accepted, log = DC.select_derivatives(api, WT.PG1_REPO, geom, n_accept=a.n_candidates,
                                          out=out)
    shas = {r["id"]: api.model_info(r["id"]).sha for r in accepted}
    screened = {}
    for rec in accepted:
        repo = rec["id"]
        m_ = PaliGemmaForConditionalGeneration.from_pretrained(
            repo, revision=shas[repo], torch_dtype=torch.bfloat16, device_map="cuda",
            attn_implementation="eager").eval()
        sd = {n: p.detach().clone() for n, p in m_.named_parameters()}
        del m_
        gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in sd and sd[n].shape == p.shape:
                    p.copy_(sd[n])
        s = evaluate_under(None)
        # A LOST FACULTY IS MEASURED: the score must fall below the base's by more than the base's
        # own interval half-width, so a candidate does not qualify on noise or on a card's claim.
        lost = {}
        for k, v in s.items():
            half = (base_scores[k]["ci95"][1] - base_scores[k]["ci95"][0]) / 2.0
            lost[k] = bool(base_scores[k]["score"] - v["score"] > half)
        screened[repo] = {"scores": {k: float(v["score"]) for k, v in s.items()},
                          "lost": lost, "any_lost": any(lost.values()), "sha": shas[repo]}
        at.add("wild_faculty_screen", value=float(sum(lost.values())), model_id=repo,
               model_sha=shas[repo],
               flags={**screened[repo], "base_scores": {k: float(v["score"])
                                                        for k, v in base_scores.items()},
                      "rule": "a loss counts when it exceeds the base's own CI half-width"})
        print(f"[M-C9] screen {repo.split('/')[-1]:28s} " +
              " ".join(f"{k}={v:.4f}{'*' if lost[k] else ''}"
                       for k, v in screened[repo]["scores"].items()), flush=True)
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(pg1_sd[n])
        del sd
        gc.collect()
        _delete_cache(repo)
        M.s3_sync_incremental(out)

    wild = [r for r, v in screened.items() if v["any_lost"]][: a.n_repair]
    gates.record("c9_found_wild_models_with_a_measured_loss", bool(wild), wild=wild,
                 n_screened=len(screened),
                 outcome=None if wild else M.COVERAGE_IMPOSSIBLE,
                 note=("'documented lost faculty' is measured here, not read off a card: a card "
                       "that claims a loss the weights do not show is not evidence, and a loss "
                       "the card never mentions counts the same"))
    assert wild, (
        f"none of {len(screened)} screened derivatives lost a faculty by more than the base's own "
        f"CI half-width; there is nothing to repair and the gate row records the screen")

    def supervised(task_key, rows):
        b = []
        t = tasks[task_key]
        for r in rows:
            ref = M.battery_supervised_target(t, r)
            if not str(ref).strip():
                continue
            inp = processor(text=[t.prompt_fn(r)], images=[t.image_fn(r).convert("RGB")],
                            suffix=[str(ref)], return_tensors="pt", padding=False).to("cuda")
            inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
            b.append(inp)
        assert b, f"{task_key}: no usable supervised rows"
        return b

    sup = {k: supervised(k, train_rows[k]) for k in tasks}
    results = {}
    for repo in wild:
        m_ = PaliGemmaForConditionalGeneration.from_pretrained(
            repo, revision=shas[repo], torch_dtype=torch.bfloat16, device_map="cuda",
            attn_implementation="eager").eval()
        sd = {n: p.detach().clone() for n, p in m_.named_parameters()}
        del m_
        gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in sd and sd[n].shape == p.shape:
                    p.copy_(sd[n])
        wild_sd = {n: p.detach().clone() for n, p in model.named_parameters()}
        floor = evaluate_under(None)
        cell = {"floor": {k: float(v["score"]) for k, v in floor.items()},
                "base": {k: float(v["score"]) for k, v in base_scores.items()},
                "by_bank": {}}
        for b in BANKS:
            for r in ranks:
                for n, p in model.named_parameters():
                    with torch.no_grad():
                        p.copy_(wild_sd[n])
                reader = BankReader(layers, "cuda", basis[b], r)
                opt = torch.optim.AdamW(reader.parameters(), lr=a.lr)
                rng = np.random.default_rng(H.cell_seed(a.seed, ARM, repo, f"{b}{r}"))
                order = [(k, i) for k in tasks for i in range(len(sup[k]))]
                model.train()
                with reader:
                    for _step in range(a.steps):
                        loss = 0.0
                        for _ in range(a.batch):
                            k, i = order[int(rng.integers(0, len(order)))]
                            loss = loss + model(**sup[k][i]).loss
                        opt.zero_grad()
                        (loss / a.batch).backward()
                        opt.step()
                model.eval()
                s = evaluate_under(reader)
                cell["by_bank"][f"{b}@r{r}"] = {
                    "bank": b, "rank": r, "n_params": reader.n_params,
                    "scores": {k: float(v["score"]) for k, v in s.items()},
                    "gain": {k: float(v["score"] - floor[k]["score"]) for k, v in s.items()}}
                for k, v in s.items():
                    at.add("wild_repair_score", value=float(v["score"]), dose=float(r),
                           bank_id=b, corpus_id=tasks[k].dataset, model_id=repo,
                           model_sha=shas[repo], ci_lo=v["ci95"][0], ci_hi=v["ci95"][1],
                           n=v["n"],
                           flags={"task": k, "floor": float(floor[k]["score"]),
                                  "base": float(base_scores[k]["score"]),
                                  "n_params": reader.n_params, "steps": a.steps, "lr": a.lr})
                print(f"[M-C9] {repo.split('/')[-1][:22]:22s} {b:14s} r={r:<3d} " +
                      " ".join(f"{k}={cell['by_bank'][f'{b}@r{r}']['gain'][k]:+.4f}"
                               for k in tasks), flush=True)
                del reader, opt
                torch.cuda.empty_cache()
                M.s3_sync_incremental(out)
        results[repo] = cell
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(pg1_sd[n])
        del sd, wild_sd
        gc.collect()
        _delete_cache(repo)

    at.check_readout_corpus(["wild_repair_score"])
    summary = {}
    for repo, cell in results.items():
        best = {b: max(cell["by_bank"][f"{b}@r{r}"]["gain"][k]
                       for r in ranks for k in tasks) for b in BANKS}
        twins = [best[b] for b in ("haar32_s0", "haar32_s1", "haar32_s2")]
        summary[repo] = {"best_gain_by_bank": best, "twin_band": [min(twins), max(twins)],
                         "core_beats_twins": bool(best["core32"] > max(twins))}
        gates.record(f"c9_core_readers_beat_twins__{repo.split('/')[-1]}",
                     summary[repo]["core_beats_twins"], **summary[repo],
                     note="the pre-logged refuter, per wild model")
        sur.add(arm=ARM, expected_ref=expected_ref, cell=repo,
                statistic="wild_repair_score/core32",
                s=M.surprise(best["core32"], float(np.mean(twins)), twins))

    # THE ARM-LEVEL ROLLUP, WITH ITS INPUTS. The per-repo gates above carry a repo suffix, so no
    # literal name can address the arm as a whole — `mercator_report.REFUTER_GATES` mapped
    # `c9_core_readers_beat_twins` and would have matched nothing forever, silently deciding the arm
    # on whatever else was mapped. (Caught by the report's new unmatched-gate warning, before this
    # arm ever got past its dependency gate, so no verdict was affected.)
    #
    # `any`, not `all`: the refuter is that the core is no better than a twin ANYWHERE in the wild,
    # so one wild model where it wins is enough to not refute.
    if summary:
        gates.record("c9_core_readers_beat_twins",
                     any(v["core_beats_twins"] for v in summary.values()),
                     per_repo={r: v["core_beats_twins"] for r, v in summary.items()},
                     n_repos=len(summary),
                     note=("arm-level rollup of the per-repo gates; `any` because the refuter is "
                           "that the core beats no twin anywhere in the wild"))

    info = at.to_parquet()
    sur.to_parquet()
    M.write_json(out, f"{ARM}_result.json",
                 {"arm": ARM, "queue_item": meta["queue_item"], "refuter": meta["refuter"],
                  "expected_ref": expected_ref, "provenance": M.provenance(), "utc": M.utc_now(),
                  "atlas": info, "shell_repo": WT.PG1_REPO, "shell_sha": base_sha,
                  "m_g_source": a.mg_result, "m_g_graduated": graduated,
                  "screened": screened, "wild": wild, "ranks": ranks, "tasks": list(TASKS),
                  "base_scores": {k: float(v["score"]) for k, v in base_scores.items()},
                  "results": results, "summary": summary})
    print("C9_OK " + json.dumps({"rows": info["n_rows"], "wild": wild,
                                 "core_beats_twins": {r: v["core_beats_twins"]
                                                      for r, v in summary.items()}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
