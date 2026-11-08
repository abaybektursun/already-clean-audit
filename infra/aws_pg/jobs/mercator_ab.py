"""M-AB stage (i) — the ICL task-screening probe (queue standing W1-A/B). Wave 1.

WHY THIS IS THE FIRST STAGE AND WHY IT GATES THE REST. DARWIN's W1-A and W1-B were recorded
VOID-INSTRUMENT: the context-burning and amplifier arms were run on a task that showed no real
in-context lift at 2B, so neither arm could have detected anything and neither result meant
anything. The ICL-sink causal question was never refuted — it was never tested. So the chain starts
by asking, with power, WHICH tasks show a real ICL lift on this model at all. Nothing downstream
runs on a task that fails here.

THE MEASUREMENT. For each candidate task, accuracy at a shot ladder n in {0, 1, 2, 4}, with the
shots drawn from the SAME corpus and the query item held out of its own prompt. The lift is
acc(n_max) − acc(0) with a bootstrap CI over items, and the arm reports the MINIMUM DETECTABLE
LIFT at the item count actually used, so a null reads as "no lift above X" rather than "no lift".

THE CONTROL THAT MAKES IT A SCREEN RATHER THAN A DEMONSTRATION. `position_corpus` is
explicit-position retrieval: every item carries its own answer key in the prompt, so demonstrations
teach FORMAT but not content, and it should show far less lift than the copy task if the lift being
measured is in-context learning rather than format acclimatisation. Running only tasks expected to
lift would measure the harness.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

from infra.pg import conserved_core_probes as CP
from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as WT
from infra.pg import mercator_core as M
from infra.pg import mercator_program as P

ARM = "M-AB"
SHOTS = (0, 1, 2, 4)
# The lift a task must clear to be worth burning context on downstream. Frozen before the run.
MIN_USEFUL_LIFT = 0.10


def build_prompt(item, shots) -> str:
    """An n-shot prompt: demonstrations, then the query. The query item is never among its own
    demonstrations — the caller guarantees that, and this asserts it."""
    q_prompt, q_gold = item
    assert all(p != q_prompt for p, _g in shots), "the query item leaked into its own shots"
    parts = [f"{p} -> {g}" for p, g in shots]
    parts.append(f"{q_prompt} ->")
    return "\n".join(parts)


def score_exact(pred: str, gold: str) -> float:
    """Exact match on the first whitespace-delimited token of the continuation.

    The golds here are single letters, so anything beyond the first token is the model continuing
    past the answer and is not evidence about the answer.
    """
    p = pred.strip().split()
    return float(bool(p) and p[0].strip(".,:;") == gold.strip())


def lift_ci(acc_hi, acc_lo, *, n_boot: int = 2000, seed: int = 1337) -> dict:
    """Paired bootstrap of acc(n_max) − acc(0) over ITEMS, plus the minimum detectable lift.

    Paired because the same items are scored at both shot counts; an unpaired interval would be
    wider than the design and would understate what the probe can see.
    """
    a = np.asarray(acc_hi, dtype=np.float64)
    b = np.asarray(acc_lo, dtype=np.float64)
    assert a.shape == b.shape and a.size >= 8, f"need >=8 paired items, got {a.shape}"
    rng = np.random.default_rng(seed)
    d = a - b
    boot = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(n_boot)])
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return {"lift": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
            "ci_halfwidth": (hi - lo) / 2.0,
            # the MDE in the lift's own units: two half-widths is the separation two arms of this
            # size must show before their intervals stop overlapping
            "min_detectable_lift": (hi - lo), "n_items": int(d.size), "n_boot": int(n_boot),
            "acc_hi": float(a.mean()), "acc_lo": float(b.mean())}


def main():
    ap = argparse.ArgumentParser(description="M-AB stage (i) — ICL task screening with power")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-items", type=int, default=96)
    ap.add_argument("--max-new", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = P.ARMS[ARM]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_w1") / "expectations.jsonl")
    expected_ref = exp.require(ARM)
    at = M.Atlas(out, arm=ARM, run_id="m_ab_screen", register=meta["register"], wave=meta["wave"],
                 expected_ref=expected_ref, queue_item=meta["queue_item"])
    gates = H.Gates(out)

    import torch
    from huggingface_hub import HfApi
    from transformers import PaliGemmaProcessor
    from infra.aws_pg.jobs.invariant_core_func import _blank_image, _load_full_pg1

    assert torch.cuda.is_available(), "no CUDA"
    api = HfApi()
    base_sha = api.model_info(WT.PG1_REPO).sha
    model = _load_full_pg1(base_sha, "cuda")
    processor = PaliGemmaProcessor.from_pretrained(WT.PG1_REPO, revision=base_sha)
    blank = _blank_image(128)

    tasks = {"copy": CP.copy_corpus(a.n_items + max(SHOTS), seed=20260619),
             "position": CP.position_corpus(a.n_items + max(SHOTS), seed=20260620)}
    rng = np.random.default_rng(a.seed)
    results = {}
    for name, corpus in tasks.items():
        pool, items = corpus[a.n_items:], corpus[:a.n_items]
        per_shot = {}
        for n in SHOTS:
            correct = []
            for it in items:
                shots = [pool[i] for i in rng.choice(len(pool), size=n, replace=False)] if n else []
                text = build_prompt(it, shots)
                inp = processor(text=[text], images=[blank], return_tensors="pt",
                                padding=False).to("cuda")
                # the processor emits fp32 pixel_values; the model is bf16 (house convention)
                inp["pixel_values"] = inp["pixel_values"].to(next(model.parameters()).dtype)
                with torch.inference_mode():
                    gen = model.generate(**inp, max_new_tokens=a.max_new, do_sample=False)
                new = gen[0, inp["input_ids"].shape[1]:]
                pred = processor.tokenizer.decode(new, skip_special_tokens=True)
                correct.append(score_exact(pred, it[1]))
            per_shot[n] = np.asarray(correct, dtype=np.float64)
            s = at.add_ci("icl_accuracy", per_shot[n], dose=float(max(n, 0.5)),
                          corpus_id=f"synthetic:{name}", model_id=WT.PG1_REPO,
                          model_sha=base_sha,
                          flags={"task": name, "n_shot": n, "n_items": len(items),
                                 "max_new_tokens": a.max_new})
            print(f"[M-AB] {name} n={n}: acc={s['mean']:.4f}", flush=True)
        lc = lift_ci(per_shot[max(SHOTS)], per_shot[0], seed=a.seed)
        at.add("icl_lift", value=lc["lift"], ci_lo=lc["ci_lo"], ci_hi=lc["ci_hi"],
               n=lc["n_items"], corpus_id=f"synthetic:{name}", model_id=WT.PG1_REPO,
               model_sha=base_sha,
               flags={**lc, "task": name, "shots_compared": [0, max(SHOTS)],
                      "min_useful_lift": MIN_USEFUL_LIFT})
        passes = bool(lc["ci_lo"] > MIN_USEFUL_LIFT)
        underpowered = bool(lc["min_detectable_lift"] > MIN_USEFUL_LIFT and not passes)
        results[name] = {**lc, "accuracy_by_shot": {n: float(v.mean())
                                                    for n, v in per_shot.items()},
                         "passes_screen": passes, "underpowered": underpowered,
                         "outcome": (None if passes else
                                     (M.VOID_UNDERPOWERED if underpowered else "no-lift"))}
        print(f"[M-AB] {name}: lift={lc['lift']:+.4f} CI=[{lc['ci_lo']:+.4f},{lc['ci_hi']:+.4f}] "
              f"MDL={lc['min_detectable_lift']:.4f} passes={passes}", flush=True)

    at.check_readout_corpus(["icl_accuracy", "icl_lift"])
    passing = [t for t, r in results.items() if r["passes_screen"]]
    gates.record("ab_any_task_shows_real_icl_lift", bool(passing), passing=passing,
                 min_useful_lift=MIN_USEFUL_LIFT,
                 note=("stages ii-iv of the ICL chain do not run without one; W1-A/B were "
                       "VOID-INSTRUMENT for exactly this reason"))
    # The screen is only a screen if the control can fail it differently from the treatment.
    gates.record("ab_control_task_separates_from_copy",
                 results["copy"]["lift"] > results["position"]["lift"],
                 copy_lift=results["copy"]["lift"], position_lift=results["position"]["lift"],
                 note=("position retrieval carries its own answer key, so demonstrations teach "
                       "FORMAT not content; if it lifted as much as copy, the probe would be "
                       "measuring format acclimatisation"))

    info = at.to_parquet()
    M.write_json(out, f"{ARM}_screen_result.json",
                 {"arm": ARM, "stage": "i (task screening)", "queue_item": meta["queue_item"],
                  "refuter": meta["refuter"], "expected_ref": expected_ref,
                  "provenance": M.provenance(), "utc": M.utc_now(), "atlas": info,
                  "base_repo": WT.PG1_REPO, "base_sha": base_sha, "shots": list(SHOTS),
                  "n_items": a.n_items, "min_useful_lift": MIN_USEFUL_LIFT,
                  "tasks": results, "passing_tasks": passing,
                  "gate_note": ("stages ii-iv are gated on a passing task; a task that fails here "
                                "cannot support the burning or amplifier arms, which is what made "
                                "W1-A/B VOID-INSTRUMENT")})
    print("AB_SCREEN_OK " + json.dumps({"rows": info["n_rows"], "passing": passing,
                                        "lifts": {t: r["lift"] for t, r in results.items()}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
