"""M-C7 — sampled telemetry: how little can you look and still see the drift? (queue C7). Wave P.

    THE DELIVERABLE. A training run that logs a sink statistic every step pays for it every step.
    A 1-in-N sampled version pays 1/N of that. The question is the exchange rate: at what N does
    the telemetry stop being able to see a real distribution shift?

    REFUTER (pre-logged): sampled telemetry cannot detect the planted drift at the overhead the
    ladder buys.

THE DRIFT IS PLANTED, so detection has a ground truth. Halfway through the run the training data
switches from the narrow-grammar code task to wikitext — a real distribution shift of the kind a
production run actually suffers, not an injected perturbation of the statistic itself. A detector
tuned on a synthetic spike would report its own tuning.

THE DETECTOR IS FIXED AND SIMPLE, and it is the same detector at every N: a two-window bootstrap
on the sampled series, comparing the last quarter of samples against the first quarter. Changing
the detector with N would confound "sampling less" with "detecting differently".

FALSE POSITIVES ARE MEASURED, not assumed. Every N is also run on a NO-DRIFT control whose data
never switches, and the detection rate there is the false-positive rate. A detector that fires at
N=64 on the drift run and also fires on the control has detected nothing; the arm reports both
rates and gates on their difference.

OVERHEAD IS MEASURED, not modelled: the wall-clock cost of a step with telemetry against a step
without, on this hardware.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

from infra.pg import darwin_train as T
from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as WT
from infra.pg import mercator_core as M
from infra.pg import mercator_program as P

ARM = "M-C7"
N_LADDER = (1, 4, 16, 64)
# The detector compares two windows, so it needs at least this many telemetry samples.
# A rung samples every N steps, so the TOP rung sets the step budget: steps >= this x
# max(N). At the default 400 steps the N=64 rung yields 7 samples and the sweep died
# after three rungs had already been measured (defect M-D62).
MIN_DETECTOR_SAMPLES = 8
DETECT_ALPHA = 0.05          # frozen: the bootstrap tail the detector fires at
PPL_CORPUS = "wikitext2:window512"


def two_window_detector(series, *, alpha: float = DETECT_ALPHA, n_boot: int = 2000,
                        seed: int = 1337) -> dict:
    """Fires when the last quarter of the sampled series differs from the first quarter.

    A bootstrap over samples rather than a t-test, because the telemetry values are not
    independent across steps and a parametric tail would be a stronger claim than the data
    supports. The window fraction and alpha are arguments so they appear in the artifact.
    """
    v = np.asarray(series, dtype=np.float64)
    q = max(2, v.size // 4)
    assert v.size >= MIN_DETECTOR_SAMPLES, (
        f"need >= {MIN_DETECTOR_SAMPLES} samples for a two-window detector, got {v.size}")
    a, b = v[:q], v[-q:]
    rng = np.random.default_rng(seed)
    obs = float(b.mean() - a.mean())
    pool = np.concatenate([a, b])
    null = np.empty(n_boot)
    for i in range(n_boot):
        p = rng.permutation(pool)
        null[i] = p[q:].mean() - p[:q].mean()
    p_two = float((np.abs(null) >= abs(obs)).mean())
    return {"observed_shift": obs, "p_value": p_two, "fires": bool(p_two < alpha),
            "n_samples": int(v.size), "window": int(q), "alpha": float(alpha)}


def main():
    ap = argparse.ArgumentParser(description="M-C7 — sampled telemetry")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-ladder", default=",".join(str(n) for n in N_LADDER))
    ap.add_argument("--types", default="q_proj")
    ap.add_argument("--layers", default="12,13,14,15,16,17")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--seeds", default="1337,1338,1339,1340")
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ladder = [int(x) for x in a.n_ladder.split(",") if x.strip()]
    M.check_dose_grid([float(n) for n in ladder], arm=ARM)
    # FEASIBILITY BEFORE ANYTHING RUNS. Every rung's sample count is decided by (steps, N) alone,
    # so a budget that cannot support the top rung is knowable at second zero rather than three
    # rungs in — and the message says the number to set rather than leaving it to be derived.
    need_steps = MIN_DETECTOR_SAMPLES * max(ladder)
    assert a.steps >= need_steps, (
        f"--steps {a.steps} gives the N={max(ladder)} rung only {a.steps // max(ladder)} telemetry "
        f"samples and the two-window detector needs {MIN_DETECTOR_SAMPLES}. Set --steps "
        f"{need_steps} or higher, or shorten --n-ladder")
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    types = tuple(t.strip() for t in a.types.split(",") if t.strip())
    layers = [int(x) for x in a.layers.split(",") if x.strip()]
    meta = P.ARMS[ARM]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_wp") / "expectations.jsonl")
    expected_ref = exp.require(ARM)
    at = M.Atlas(out, arm=ARM, run_id="m_c7", register=meta["register"], wave=meta["wave"],
                 expected_ref=expected_ref, queue_item=meta["queue_item"])
    sur = M.SurpriseLog(out)
    gates = H.Gates(out)
    optinfo = M.check_optimizer(ARM, "darwin_train.ExactAdam", offload_exact_fp32=False)

    import torch
    from huggingface_hub import HfApi
    from transformers import PaliGemmaProcessor
    from infra.aws_pg.jobs.invariant_core_func import _decoder_layers, _load_full_pg1
    from infra.aws_pg.jobs.darwin_w1_u import build_batches, text_task_rows, wikitext_rows
    from infra.aws_pg.jobs.mercator_m5 import spike_coords

    assert torch.cuda.is_available(), "no CUDA"
    api = HfApi()
    base_sha = api.model_info(WT.PG1_REPO).sha
    model = _load_full_pg1(base_sha, "cuda")
    processor = PaliGemmaProcessor.from_pretrained(WT.PG1_REPO, revision=base_sha)
    for p in model.parameters():
        p.requires_grad_(False)
    mods = T.trainable_modules(_decoder_layers(model), types=types, layers=layers)
    for mod in mods.values():
        mod.weight.requires_grad_(True)
    pristine = {k: m.weight.detach().to(torch.float32).clone() for k, m in mods.items()}
    train_rows, _ = text_task_rows(a.n_train, 16, seeds[0])
    batches = {"code": build_batches(processor, train_rows, "cuda"),
               "wikitext": build_batches(processor, wikitext_rows(a.n_train, seeds[0]), "cuda")}
    bank_dir = M.fetch_all_banks(H.cache_dir(out, "banks"))
    sink = {k: torch.as_tensor(spike_coords(WT.load_bank(bank_dir, k[0], k[1], "core32")),
                               device="cuda", dtype=torch.long) for k in mods}
    n_cols = {k: int(m.weight.shape[1]) for k, m in mods.items()}
    print(f"[M-C7] {len(mods)} cells; sink columns "
          f"{min(len(v) for v in sink.values())}-{max(len(v) for v in sink.values())}", flush=True)

    def telemetry(mods_):
        """The logged statistic: the sink's share of gradient energy, over the flat expectation.

        Cheap by construction — one gather and two sums per cell — which is the whole point of
        asking what sampling it costs.
        """
        vals = []
        with torch.no_grad():
            for k, m in mods_.items():
                g = m.weight.grad.to(torch.float32)
                tot = float((g * g).sum())
                if tot == 0:
                    continue
                share = float((g[:, sink[k]] ** 2).sum()) / tot
                vals.append(share / (len(sink[k]) / n_cols[k]))
        return float(np.mean(vals))

    def run(n_sample, seed, drift):
        for k, m in mods.items():
            m.weight.data.copy_(pristine[k].to(m.weight.dtype))
        opt = T.ExactAdam([m.weight for m in mods.values()], lr=a.lr)
        rng = np.random.default_rng(H.cell_seed(seed, ARM, str(n_sample), str(drift)))
        series, steps_logged = [], []
        t_tel, t_step = 0.0, 0.0
        switch = a.steps // 2
        model.train()
        for step in range(a.steps):
            src = "wikitext" if (drift and step >= switch) else "code"
            b = batches[src][int(rng.integers(0, len(batches[src])))]
            t0 = time.perf_counter()
            opt.zero_grad()
            model(input_ids=b["input_ids"], labels=b["labels"]).loss.backward()
            t1 = time.perf_counter()
            if step % n_sample == 0:
                series.append(telemetry(mods))
                steps_logged.append(step)
            t2 = time.perf_counter()
            opt.step()
            t_step += t1 - t0
            t_tel += t2 - t1
        det = two_window_detector(series, seed=seed)
        return {"n_sample": n_sample, "seed": seed, "drift": drift, "series": series,
                "steps_logged": steps_logged, "switch_step": switch,
                "telemetry_seconds": t_tel, "training_seconds": t_step,
                "overhead_fraction": t_tel / max(t_step, 1e-9), **det}

    cells, by_n = {}, {}
    for n in ladder:
        rows = []
        for seed in seeds:
            for drift in (True, False):
                c = run(n, seed, drift)
                cells[f"n{n}_s{seed}_{'drift' if drift else 'control'}"] = c
                rows.append(c)
                at.add("telemetry_detector_p", value=float(c["p_value"]), dose=float(n),
                       flags={**{k: v for k, v in c.items() if k != "series"},
                              "steps": a.steps, "optimizer": optinfo},
                       incumbent_id="no_drift_control" if not drift else None)
                print(f"[M-C7] N={n:3d} s{seed} {'drift  ' if drift else 'control'} "
                      f"samples={c['n_samples']:4d} p={c['p_value']:.4f} "
                      f"fires={c['fires']} overhead={c['overhead_fraction']:.4f}", flush=True)
        hit = [r for r in rows if r["drift"]]
        fp = [r for r in rows if not r["drift"]]
        power = float(np.mean([r["fires"] for r in hit]))
        false_pos = float(np.mean([r["fires"] for r in fp]))
        overhead = float(np.mean([r["overhead_fraction"] for r in rows]))
        by_n[n] = {"power": power, "false_positive_rate": false_pos, "overhead": overhead,
                   "n_seeds": len(seeds), "excess_power": power - false_pos}
        at.add("telemetry_power", value=power, dose=float(n), n=len(hit),
               flags={**by_n[n], "definition": "fraction of drift runs the detector fires on"})
        at.add("telemetry_overhead", value=overhead, dose=float(n),
               flags={"definition": "telemetry seconds / training seconds, measured on this box"})
        print(f"[M-C7] N={n:3d}: power={power:.3f} false_pos={false_pos:.3f} "
              f"overhead={overhead:.4f}", flush=True)
        M.s3_sync_incremental(out)

    # ---- the exchange rate --------------------------------------------------------------------
    usable = [n for n in ladder if by_n[n]["excess_power"] >= 0.5]
    gates.record("c7_sampled_telemetry_detects_the_drift", bool(usable),
                 usable_n=usable, by_n=by_n,
                 note=("excess power = detection rate on drift runs minus the rate on no-drift "
                       "controls; a detector that fires on both has detected nothing"))
    cheapest = max(usable) if usable else None
    gates.record("c7_cheapest_usable_sampling_is_cheaper_than_every_step",
                 bool(cheapest is not None and cheapest > 1),
                 cheapest_usable_n=cheapest,
                 overhead_at_cheapest=(by_n[cheapest]["overhead"] if cheapest else None),
                 overhead_at_every_step=by_n[ladder[0]]["overhead"],
                 note=("the arm's practical claim: the largest N that still detects, and what it "
                       "costs against logging every step"))
    sur.add(arm=ARM, expected_ref=expected_ref, cell=f"N={ladder[-1]}",
            statistic="telemetry_power",
            s=M.surprise(by_n[ladder[-1]]["power"], by_n[ladder[-1]]["false_positive_rate"],
                         [by_n[n]["false_positive_rate"] for n in ladder]))

    info = at.to_parquet()
    sur.to_parquet()
    M.write_json(out, f"{ARM}_result.json",
                 {"arm": ARM, "queue_item": meta["queue_item"], "refuter": meta["refuter"],
                  "expected_ref": expected_ref, "provenance": M.provenance(), "utc": M.utc_now(),
                  "atlas": info, "base_repo": WT.PG1_REPO, "base_sha": base_sha,
                  "types": list(types), "layers": layers, "steps": a.steps, "lr": a.lr,
                  "n_ladder": ladder, "seeds": seeds, "alpha": DETECT_ALPHA,
                  "optimizer": optinfo, "by_n": by_n, "cells": cells,
                  "usable_n": usable, "cheapest_usable_n": cheapest,
                  "drift": ("training data switches from the code task to wikitext at the "
                            "midpoint; a real distribution shift, not an injected spike")})
    print("C7_OK " + json.dumps({"rows": info["n_rows"], "by_n": by_n,
                                 "cheapest_usable_n": cheapest}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
