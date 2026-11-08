"""M-12 stage (c) — the position-vs-induction discriminator (queue item 12). Wave 1.

WHAT THIS RESOLVES. conserved-core (2026-06-19) left a WAIVED discriminator: its label was
under-determined between "the core carries positional machinery" and "the core carries induction
machinery", because no experiment separated them. M-12 is chartered to resolve it, and the two
corpora that do so already exist in `conserved_core_probes` — so the arm forks nothing and simply
runs the contrast that was never run.

THE DESIGN. Ablate a 32-dimensional subspace from what attention READS (project it out of the
shared q_proj/k_proj input at every layer, which is `invariant_core_func._InputProject` verbatim)
and measure the damage to two tasks that are pure by construction:

    copy       repeat-3 induction copy      — an induction head can do it, a position index cannot
    position   explicit-position retrieval  — a position index can do it, induction cannot

The DISCRIMINATOR is the DIFFERENCE of damages, damage(copy) − damage(position), read against the
same difference under the full twin battery. Either damage alone would confound "this subspace
matters" with "this subspace matters FOR INDUCTION"; the difference is what separates the labels.

TWO THINGS MAKE IT A TEST RATHER THAN A DEMONSTRATION:

  * both tasks' INTACT accuracies are measured in the same run, and M-AB's screen already
    established that copy is learnable in context here (+0.41 lift) while position is not — a task
    the model cannot do has no damage to lose, and a discriminator built on one would report the
    floor;
  * the twin battery is ablated identically. A random 32-dimensional subspace removed from the
    read path also costs accuracy, and the question is only ever whether the core costs MORE, and
    more ASYMMETRICALLY between the two tasks.

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

ARM = "M-12"
FULL_BATTERY = ("core32", "anti32", "topsigma32", "bottomsigma32", "complement32",
                "haar32_s0", "haar32_s1", "haar32_s2")
SHOTS = 2          # M-AB measured the copy lift saturating at two shots


def score_task(model, processor, blank, items, pool, rng, *, shots: int, max_new: int):
    """Accuracy on one corpus at a fixed shot count. Reuses M-AB's prompt builder and scorer so the
    intact accuracies here are the same measurement the screen made."""
    import torch
    from infra.aws_pg.jobs.mercator_ab import build_prompt, score_exact
    correct = []
    for it in items:
        sh = [pool[i] for i in rng.choice(len(pool), size=shots, replace=False)] if shots else []
        inp = processor(text=[build_prompt(it, sh)], images=[blank], return_tensors="pt",
                        padding=False).to("cuda")
        inp["pixel_values"] = inp["pixel_values"].to(next(model.parameters()).dtype)
        with torch.inference_mode():
            gen = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
        pred = processor.tokenizer.decode(gen[0, inp["input_ids"].shape[1]:],
                                          skip_special_tokens=True)
        correct.append(score_exact(pred, it[1]))
    return np.asarray(correct, dtype=np.float64)


def paired_diff_ci(a, b, *, n_boot: int = 2000, seed: int = 1337) -> dict:
    """Paired item bootstrap of mean(a) − mean(b)."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    assert x.shape == y.shape and x.size >= 8, f"need >=8 paired items, got {x.shape}"
    rng = np.random.default_rng(seed)
    d = x - y
    boot = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(n_boot)])
    return {"diff": float(d.mean()), "ci_lo": float(np.percentile(boot, 2.5)),
            "ci_hi": float(np.percentile(boot, 97.5)), "n": int(d.size)}


def main():
    ap = argparse.ArgumentParser(description="M-12 stage (c) — position vs induction discriminator")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-items", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = P.ARMS[ARM]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_w1") / "expectations.jsonl")
    expected_ref = exp.require(ARM)
    at = M.Atlas(out, arm=ARM, run_id="m_12c", register=meta["register"], wave=meta["wave"],
                 expected_ref=expected_ref, queue_item=meta["queue_item"])
    sur = M.SurpriseLog(out)
    gates = H.Gates(out)

    import torch
    from huggingface_hub import HfApi
    from transformers import PaliGemmaProcessor
    from infra.aws_pg.jobs.invariant_core_func import (_InputProject, _blank_image,
                                                       _decoder_layers, _load_full_pg1)

    assert torch.cuda.is_available(), "no CUDA"
    api = HfApi()
    base_sha = api.model_info(WT.PG1_REPO).sha
    model = _load_full_pg1(base_sha, "cuda")
    processor = PaliGemmaProcessor.from_pretrained(WT.PG1_REPO, revision=base_sha)
    dec = _decoder_layers(model)
    blank = _blank_image(128)
    bank_dir = M.fetch_all_banks(H.cache_dir(out, "banks"))

    n_pool = max(SHOTS, 4)
    corpora = {"copy": CP.copy_corpus(a.n_items + n_pool, seed=20260619),
               "position": CP.position_corpus(a.n_items + n_pool, seed=20260620)}
    split = {k: (v[:a.n_items], v[a.n_items:]) for k, v in corpora.items()}

    def run_all(label):
        rows = {}
        for name, (items, pool) in split.items():
            rng = np.random.default_rng(a.seed)      # same shots in every arm: the ONLY difference
            rows[name] = score_task(model, processor, blank, items, pool, rng,
                                    shots=SHOTS, max_new=a.max_new)
            s = at.add_ci("probe_accuracy", rows[name], bank_id=label,
                          corpus_id=f"synthetic:{name}", model_id=WT.PG1_REPO,
                          model_sha=base_sha,
                          flags={"task": name, "ablated": label, "n_shot": SHOTS})
            print(f"[M-12c] {label:14s} {name:9s} acc={s['mean']:.4f}", flush=True)
        return rows

    intact = run_all("intact")
    # HEADROOM, ON BOTH ARMS, AT THE SAME BAR. The first run of this stage gated the position arm
    # at "> 0.0", which 2 correct items out of 64 passed — and 2/64 is the floor, so the position
    # damage could only range over +/-0.03 and the DIFFERENCE of damages was just the copy damage
    # wearing a discriminator's name. M-AB's screen had already measured position at 0.031 and
    # failed it; the bar is now the same on both arms and is the screen's own threshold.
    MIN_HEADROOM = 0.15
    acc_c, acc_p = float(intact["copy"].mean()), float(intact["position"].mean())
    gates.record("m12c_copy_has_headroom", acc_c > MIN_HEADROOM, accuracy=acc_c,
                 min_headroom=MIN_HEADROOM)
    gates.record("m12c_position_has_headroom", acc_p > MIN_HEADROOM, accuracy=acc_p,
                 min_headroom=MIN_HEADROOM)
    discriminator_valid = bool(acc_c > MIN_HEADROOM and acc_p > MIN_HEADROOM)
    gates.record("m12c_discriminator_is_valid", discriminator_valid,
                 copy=acc_c, position=acc_p, min_headroom=MIN_HEADROOM,
                 outcome=None if discriminator_valid else M.VOID_READOUT,
                 note=("the DIFFERENCE of damages only separates the two labels if BOTH tasks can "
                       "lose something; at the floor the difference is the other task's damage"))

    results = {}
    for bid in FULL_BATTERY:
        Q = {L: WT.load_bank(bank_dir, "q_proj", L, bid) for L in range(H.N_LAYERS)}
        with _InputProject(dec, Q, "cuda", next(model.parameters()).dtype):
            abl = run_all(bid)
        dmg = {t: paired_diff_ci(intact[t], abl[t], seed=a.seed) for t in split}
        disc = paired_diff_ci(intact["copy"] - abl["copy"],
                              intact["position"] - abl["position"], seed=a.seed)
        for t, d in dmg.items():
            at.add("probe_damage", value=d["diff"], ci_lo=d["ci_lo"], ci_hi=d["ci_hi"], n=d["n"],
                   bank_id=bid, corpus_id=f"synthetic:{t}", model_id=WT.PG1_REPO,
                   model_sha=base_sha, flags={"task": t, "ablated": bid})
        at.add("induction_minus_position_damage", value=disc["diff"], ci_lo=disc["ci_lo"],
               ci_hi=disc["ci_hi"], n=disc["n"], bank_id=bid, model_id=WT.PG1_REPO,
               model_sha=base_sha,
               flags={"definition": "damage(copy) - damage(position); the label discriminator"})
        results[bid] = {"damage": dmg, "discriminator": disc,
                        "accuracy": {t: float(v.mean()) for t, v in abl.items()}}
        print(f"[M-12c] {bid}: discriminator={disc['diff']:+.4f} "
              f"CI=[{disc['ci_lo']:+.4f},{disc['ci_hi']:+.4f}]", flush=True)

    at.check_controls_per_statistic(["probe_damage", "induction_minus_position_damage"])
    twins = [results[b]["discriminator"]["diff"] for b in ("haar32_s0", "haar32_s1", "haar32_s2")]
    core = results["core32"]["discriminator"]["diff"]
    s = M.surprise(core, float(np.mean(twins)), twins)
    sur.add(arm=ARM, expected_ref=expected_ref, cell="discriminator",
            statistic="induction_minus_position_damage/core32", s=s)
    band = [float(min(twins)), float(max(twins))]
    gates.record("m12c_core_discriminator_outside_twin_band",
                 not (band[0] <= core <= band[1]), core=core, twin_band=band,
                 note=("positive means ablating the core costs the INDUCTION task more than the "
                       "position task; negative means the reverse; inside the band means the "
                       "core is not distinguishable from a random subspace on this contrast"))

    info = at.to_parquet()
    sur.to_parquet()
    M.write_json(out, f"{ARM}_stage_c_result.json",
                 {"arm": ARM, "stage": "c (position vs induction discriminator)",
                  "queue_item": meta["queue_item"], "refuter": meta["refuter"],
                  "expected_ref": expected_ref, "provenance": M.provenance(),
                  "utc": M.utc_now(), "atlas": info, "base_repo": WT.PG1_REPO,
                  "base_sha": base_sha, "n_items": a.n_items, "shots": SHOTS,
                  "intact_accuracy": {t: float(v.mean()) for t, v in intact.items()},
                  "by_bank": results, "core_discriminator": core, "twin_band": band,
                  "surprise": s, "discriminator_valid": discriminator_valid,
                  "discriminator_outcome": (None if discriminator_valid else M.VOID_READOUT),
                  "resolves": ("conserved-core's WAIVED position-vs-induction label IF AND ONLY "
                               "IF both arms have headroom; otherwise the label stays waived and "
                               "this run records WHY, which is itself the missing information")})
    print("M12C_OK " + json.dumps({"rows": info["n_rows"], "core_discriminator": core,
                                   "twin_band": band}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
