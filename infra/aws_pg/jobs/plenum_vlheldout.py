"""Held-out VL loss across U3c's checkpoint trajectory — the discriminating test for whether the
co-training stream teaches the faculty or fits something specific.

WHY THIS EXISTS. With the label defect fixed, U3c's VL loss still falls to near-zero
(9.14 → 0.2836 @512 → 0.0362 @1024). That is ~8× above the newline-collapse run at matched steps, so
it is not that failure — but 0.0362 nats is a perplexity of 1.037, i.e. ≈96% mass on the correct
answer token for rows the model has not seen. Implausibly good for VQAv2. Two readings fit, with
opposite consequences for Arm 3:

  1. real but shallow — PG1 is pretrained on VQA-style data, so most of the initial 9.14 may be FORMAT
     miscalibration, and calibrating it drops CE fast and legitimately;
  2. degenerate in a way not yet identified — in which case U3c's intervention is nominal again and
     `U3c ≈ U3` would again be an artifact rather than a finding.

The training-batch loss cannot separate these. A HELD-OUT loss can: rows the training loop never
sampled. `plenum_a3` draws its supervision from `load_items(vqav2, 4000)` with
`torch.randint(0, 4000, ...)` under a seeded generator, so rows are only ever drawn from that first
4,000. Taking a LATER slice of the same stream (offset 4000+) gives rows from the same distribution
that no training step can have touched — the clean held-out split, without needing to replay the RNG.

Reported against two reference levels, so the number is interpretable rather than merely small:
  * `answer_prior_ce`  — CE of always predicting the empirical answer-token marginal. A loss AT this
    level means the stream taught the marginal and nothing else.
  * `step0` (the pre-training policy) — the same held-out loss before any co-training step.

Runs on the checkpoints already on S3, so it costs no training time and yields the whole trajectory.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.plenum_vlheldout --type g6e.2xlarge --no-weights \\
      --args "--unit U3c --n-heldout 256" --s3-prefix results/plenum_vlheldout --budget-min 300
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import pathlib
import subprocess

S3 = "s3://pg-lineage-769063704696"


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--unit", default="U3c")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--battery-n", type=int, default=4000,
                    help="the battery's frozen vqav2 count — plenum_a3 now draws its supervision "
                         "ABOVE this, so the held-out slice must start above BOTH")
    ap.add_argument("--train-pool", type=int, default=8000,
                    help="first row index NOT reachable by the training stream. plenum_a3 supervises "
                         "`_all[battery_n : battery_n+4000]` = rows 4000..8000 after the contamination "
                         "fix, so held-out must start at 8000. An earlier default of 4000 was correct "
                         "for the CONTAMINATED run (which trained on 0..4000) and would have been "
                         "wrong for the clean one — the probe would have scored the model on rows it "
                         "trained on, contaminating the adjudicator itself.")
    ap.add_argument("--n-heldout", type=int, default=256)
    ap.add_argument("--work", default="/work/plenum_vlho")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # THE SAME STACK THE TRAINER USES, not a trimmed one. `load_policy_for` instantiates the policy
    # class the checkpoint names, which imports `lerobot.policies.factory` — and that module imports
    # EVERY policy family including `groot`, whose deps are absent from the `[pi]` extra. So `[pi]`
    # alone dies in the factory import chain (after an 8GB checkpoint sync), and `[libero,pi]` plus the
    # apt EGL stack is the combination `plenum_a3` and `plenum_control` have both proven.
    #
    # LIBERO is pulled in for its dependency closure, not used: this job runs a text+image forward and
    # never creates an env. Matching a known-good install beats guessing which transitive package the
    # truncated traceback wanted.
    import subprocess as _sp
    import sys as _sys
    print("[VLHO] installing the trainer's stack (apt EGL + lerobot[libero,pi])", flush=True)
    _sp.run(["bash", "-lc",
             "apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y "
             "libegl1 libgles2 libglvnd0 libgl1 libosmesa6 libglfw3 libglib2.0-0 libsm6 "
             "libxext6 libxrender1 libgomp1 git python3-dev python3.12-dev build-essential"],
            check=True)
    _sp.run([_sys.executable, "-m", "pip", "install", "-q",
             "lerobot[libero,pi] @ git+https://github.com/huggingface/lerobot.git"], check=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch
    from infra.aws_pg.jobs.plenum_a3 import load_policy_for, vl_batch
    from infra.pg import inherit_pg1 as IP
    from infra.pg import lacuna_shell as LS
    from infra.pg import plenum_core as PC
    from transformers import AutoProcessor

    assert torch.cuda.is_available(), "needs a GPU forward"
    rep = {"object": "plenum_vlheldout_v1", "unit": a.unit, "seed": a.seed,
           "battery_n": a.battery_n, "train_pool": a.train_pool, "n_heldout": a.n_heldout,
           "disjointness": (f"battery evaluates [0,{a.battery_n}); training supervises "
                            f"[{a.battery_n},{a.train_pool}); this probe evaluates "
                            f"[{a.train_pool},{a.train_pool + a.n_heldout}) — all three disjoint"),
           "what": "VL loss on rows the training loop never sampled, across the checkpoint trajectory"}
    cells = []
    rep["cells"] = cells                       # bound BEFORE the loop (fleet lesson)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    proc = AutoProcessor.from_pretrained(LS.PG1_REPO)
    tok = proc.tokenizer
    tasks = LS.mm_tasks()
    # One stream, two disjoint slices: [0, train_pool) is what training could draw; the tail is not.
    allrows, _h = IP.load_items(tasks["vqav2"], a.train_pool + a.n_heldout)
    assert len(allrows) > a.train_pool, (
        f"only {len(allrows)} rows available; cannot cut a held-out split above {a.train_pool}")
    heldout = allrows[a.train_pool:]
    rep["n_heldout_actual"] = len(heldout)

    # Reference level 1: the answer-token marginal, estimated on the rows the stream could actually
    # have learned the prior of — the supervision slice, not everything below the held-out cut.
    ctr, n_tok = collections.Counter(), 0
    for r in allrows[a.battery_n:a.train_pool]:
        g, _b = PC.gold_and_banned("vqav2", r)
        for t in tok(" " + g, add_special_tokens=False).input_ids:
            ctr[t] += 1; n_tok += 1
    rep["answer_prior_ce_nats"] = -sum(c / n_tok * math.log(c / n_tok) for c in ctr.values())
    rep["how_to_read"] = ("held-out loss AT answer_prior_ce_nats => the stream taught the answer "
                          "marginal and little else; meaningfully BELOW => content; near ZERO => a "
                          "degenerate target; NOT FALLING from step0 => the stream taught nothing "
                          "that generalises and U3c's intervention is nominal")
    _w()

    # Every checkpoint on the film prefix, plus the pre-training policy as step 0.
    ls = subprocess.run(["aws", "s3", "ls", f"{S3}/weights/plenum_film/{a.unit}/seed{a.seed}/"],
                        check=True, capture_output=True, text=True).stdout
    steps = sorted(int(x.strip("/")) for x in
                   (l.split()[-1] for l in ls.splitlines() if l.strip().endswith("/")))
    assert steps, f"no checkpoints under plenum_film/{a.unit}/seed{a.seed}"
    rep["checkpoints"] = steps
    srcs = [(0, f"{S3}/weights/lacuna_units/U3")] + [
        (st, f"{S3}/weights/plenum_film/{a.unit}/seed{a.seed}/{st:08d}/pretrained_model")
        for st in steps]
    _w()

    for step, uri in srcs:
        d = pathlib.Path(a.work) / f"{a.unit}_{step}"
        d.mkdir(parents=True, exist_ok=True)
        subprocess.run(["aws", "s3", "sync", uri, str(d)], check=True)
        policy = load_policy_for(str(d), "cuda", dtype="bf16")
        pali = policy.model.paligemma_with_expert.paligemma
        tot, ntok_sup = 0.0, 0
        with torch.inference_mode():
            for i in range(0, len(heldout), 1):
                enc, labels = vl_batch(proc, tasks, [heldout[i]], "cuda")
                o = pali(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                         token_type_ids=enc["token_type_ids"],
                         pixel_values=enc["pixel_values"].to(next(pali.parameters()).dtype),
                         labels=labels)
                loss = o.loss if hasattr(o, "loss") else o["loss"]
                k = int((labels != -100).sum())
                assert k > 0, "a held-out row carries no supervision — the label gate would have caught this"
                tot += float(loss) * k
                ntok_sup += k
        ce = tot / ntok_sup
        cells.append({"step": step, "heldout_ce_nats": ce, "n_supervised_tokens": ntok_sup,
                      "vs_prior": ce / rep["answer_prior_ce_nats"]})
        print(f"[VLHO] {a.unit} step={step:>8} held-out CE={ce:.4f} "
              f"({ce / rep['answer_prior_ce_nats']:.3f}× the answer prior)", flush=True)
        del policy, pali
        torch.cuda.empty_cache()
        subprocess.run(["rm", "-rf", str(d)], check=True)
        _w()

    base = cells[0]["heldout_ce_nats"]
    last = cells[-1]["heldout_ce_nats"]
    rep["verdict"] = {
        "step0_ce": base, "final_ce": last, "drop": base - last,
        "generalises": bool(last < base - 0.05),
        "note": ("the stream teaches something that generalises to unseen rows" if last < base - 0.05
                 else "the held-out loss did NOT fall materially: whatever the training-batch loss was "
                      "tracking does not generalise, and U3c's intervention is nominal")}
    _w()
    print(f"PLENUM_VLHELDOUT_OK step0={base:.4f} final={last:.4f} "
          f"generalises={rep['verdict']['generalises']}", flush=True)


if __name__ == "__main__":
    main()
