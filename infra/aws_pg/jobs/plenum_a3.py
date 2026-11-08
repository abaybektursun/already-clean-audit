"""PLENUM Arm 3 — the film, the savings, and the U3c causal recipe test (spec §2.3, §6).

Population re-scope, decided in the spec and frozen here (the LACUNA-recorded defect):
`HuggingFaceVLA/libero` carries no per-episode suite tag, so the TRAINING population is the
**combined LIBERO dataset** (`build_loader(..., suites=None)`) and the EVALUATION population is
per-suite via the env (`--env.task=libero_spatial`) — the suite lives in the env, not the dataset.

Three things this job does that `lacuna_a3_run` does not:

  U3c        the causal recipe intervention: U3 trained with a VQA supervision stream interleaved
             at a FROZEN mixing ratio. Does adding VL co-training to the same control fine-tune
             spare the faculties U3 loses? This is what turns Arm 1's law from correlational into
             interventional, so it gets its own unit rather than a post-hoc reweighting.
  film       the full forced-choice battery at EVERY checkpoint (step 0 = the on-chain zero-shot
             point, then every B*/8), so consumption and acquisition are observed on one clock.
  order stat `onset(faculty) − acquisition(control)` per §2.3, computed by `plenum_core`.

FEASIBILITY GATE (`--dry-run`): the co-train step runs for 2 steps on real batches before any real
budget is committed. If the policy's PaliGemma half will not take a VL supervision batch, the
spec's pre-registered pivot fires — recipe-causality is UNTESTED and the cell terminates as
VOID-feasibility, with the film proceeding. That is a terminal state, not a gap, and it is decided
by a 5-minute job rather than by a crash 200 GPU-h in.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.plenum_a3 --type g6e.4xlarge --no-weights \
      --args "--unit U3c --dry-run" --s3-prefix results/plenum_a3_dry --budget-min 60
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

S3 = "s3://pg-lineage-769063704696"
DATASET_REPO = "HuggingFaceVLA/libero"
EVAL_SUITE = "libero_spatial"          # eval population lives in the ENV (the re-scope)
COTRAIN_RATIO = 4                      # FROZEN: one VL step per 4 action steps (§8 threat row)
# U1 is the PUBLISHED policy (no banked copy); U3/U4 are the LACUNA F0 units; U3c STARTS from U3
# and diverges through co-training, so it shares U3's source and writes its own checkpoints.
UNIT_SRC = {"U1": "hf:lerobot/pi0_base", "U2": "hf:lerobot/pi05_base",
            "U3": f"{S3}/weights/lacuna_units/U3", "U4": f"{S3}/weights/lacuna_units/U4",
            "U3c": f"{S3}/weights/lacuna_units/U3"}


def load_policy_for(src, device, dtype=None):
    """Instantiate the policy class named by the checkpoint's own config.json.

    `dtype=bfloat16` is what makes the REAL §2.3 intervention runnable on available hardware. AdamW
    over a ~3B policy in fp32 holds a master copy plus two moments — ~36GB of state, hence the 80GB
    requirement that AWS has not supplied on any H100/H200 family. In bf16 the same optimizer holds
    params 6 + grads 6 + two moments 12 ≈ 24GB, which fits a 44GB L40S.
    
    This is a PRECISION choice, not a recipe change: the optimizer is still AdamW, unlike the
    `--optim sgd` mechanics path which is a different recipe and cannot produce a valid U3c. bf16
    Adam is standard practice in real training runs. The dtype is recorded in the artifact so the
    precision is never invisible.
    """
    import json as _json
    import pathlib as _p
    import torch as _t
    # `src` is a local dir for the banked units and an HF id for U1/U2 (the published policies have
    # no banked copy). Reading config.json off a bare repo id would crash, and U1 is a film unit.
    _local = _p.Path(src) / "config.json"
    if _local.exists():
        cfg = _json.loads(_local.read_text())
    else:
        from huggingface_hub import hf_hub_download
        cfg = _json.loads(_p.Path(hf_hub_download(src, "config.json")).read_text())
    kind = str(cfg.get("type") or cfg.get("policy_type") or "").lower()
    if "pi05" in kind or "pi0.5" in kind:
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy as K
    else:
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy as K
    pol = K.from_pretrained(src)
    if dtype == "bf16":
        pol = pol.to(_t.bfloat16)
    return pol.to(device).eval()


# The UNIT path (flat: safetensors directly inside, like U3/U4 — `s3_policy_sd` globs non-recursively).
S3_U3C = "s3://pg-lineage-769063704696/weights/lacuna_units/U3c"
# The FILM path, a SEPARATE prefix. Writing checkpoints under the unit path would make installing
# `u3c` sync every checkpoint with it — tens of GB pulled to read one state dict — and the same
# `<step>/pretrained_model` layout for every film unit keeps every film URI one shape.
S3_FILM = "s3://pg-lineage-769063704696/weights/plenum_film"
CALIB_CAP = 16384                      # the frozen cap; reaching it is VOID-budget, not a retry


def _sh(cmd, **kw):
    import subprocess
    print(f"[A3] $ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, check=True, **kw)


def sync_unit(uri, dest):
    """Local path for a unit. `hf:` ids pass straight through — lerobot resolves them itself.

    §2.3 films U1 (π0), which is the PUBLISHED policy with no banked S3 copy, so the S3-only
    version of this function made `--unit U1` an illegal choice and the π0 film cell unreachable.
    """
    if uri.startswith("hf:"):
        return uri[len("hf:"):]
    import subprocess
    d = pathlib.Path(dest)
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "sync", uri, str(d)], check=True)
    assert list(d.glob("*.safetensors")), f"no weights under {uri}"
    return str(d)


def vl_batch(proc, tasks, rows, device):
    """One VL supervision batch, built with the processor's `suffix=` API.

    THE PREFIX-LM LEAK. PaliGemma is a prefix-LM: the prefix (image + prompt) is attended
    BIDIRECTIONALLY and only the suffix is causal. Which tokens are prefix and which are suffix is
    carried by `token_type_ids` (0 = prefix, 1 = suffix).

    The previous version concatenated the answer into `text` — `proc(text=[prompt + " " + answer])` —
    and never passed `token_type_ids` to the model. That puts the answer inside the BIDIRECTIONAL
    PREFIX, so the model can attend to the answer token while predicting it. The loss then measures
    copying, not answering, and drives to zero: observed 9.14 → 0.284 → 0.036 → 0.012, converging on
    the same near-zero regime as the newline bug it replaced. Two different defects, one symptom —
    a VL loss that goes to zero because the target is trivially available.

    (This is the same family as `memory/reference_paligemma_sdpa_wrong_forward`: mishandling the
    prefix-LM mask silently produces a confidently wrong number. Other modules in this repo keep
    PaliGemma's training-loss path OFF for exactly this reason.)

    The processor's `suffix=` argument is the documented training API: it emits `token_type_ids` AND
    `labels` with the prefix masked to -100, so the answer is in the causal suffix where predicting it
    requires actually answering. `token_type_ids` must then be forwarded to the model — without it the
    mask is rebuilt wrongly and the leak returns.
    """
    import torch
    from infra.pg import plenum_core as PC
    texts, images, answers = [], [], []
    for r in rows:
        g, _b = PC.gold_and_banned("vqav2", r)
        texts.append(tasks["vqav2"].prompt_fn(r))
        images.append(tasks["vqav2"].image_fn(r).convert("RGB"))
        answers.append(g)
    enc = proc(text=texts, images=images, suffix=answers, return_tensors="pt",
               padding="longest").to(device)
    assert "token_type_ids" in enc, (
        "the processor returned no token_type_ids — without it the prefix/suffix split is lost and "
        "the answer is attended bidirectionally while being predicted")
    assert "labels" in enc, "suffix= should have produced labels; the processor API changed"
    labels = enc.pop("labels")

    # EVERY supervised position must live in the CAUSAL SUFFIX. This is the assertion whose absence
    # let the leak run: a supervised position with token_type_ids == 0 is a token the model can see
    # while predicting it.
    tti = enc["token_type_ids"]
    sup = labels != -100
    assert int(sup.sum()) > 0, "no supervised positions — the VL stream would be inert"
    assert bool((tti[sup] == 1).all()), (
        f"{int((tti[sup] != 1).sum())} supervised positions are in the bidirectional PREFIX "
        f"(token_type_ids == 0) — the answer would be visible while being predicted")

    # And the supervision must still BE the answers (the check that caught the newline bug).
    tok = proc.tokenizer
    expect = sum(len(tok(a, add_special_tokens=False).input_ids) for a in answers)
    n_sup = int(sup.sum())
    assert abs(n_sup - expect) <= len(answers), (
        f"{n_sup} supervised positions vs ~{expect} answer tokens (±1/row for the EOS the suffix "
        f"path appends) — the labels are not the answers")
    return enc, labels


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--unit", required=True, choices=tuple(UNIT_SRC))
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=2,
                    help="AdamW over a ~3B policy does not fit a 44GB L40S at 8; the "
                         "spec prefers p5 H100 for this arm (§12)")
    ap.add_argument("--vl-batch-size", type=int, default=2)
    ap.add_argument("--cotrain-ratio", type=int, default=COTRAIN_RATIO)
    ap.add_argument("--film-points", type=int, default=8, help="checkpoints after step 0")
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--task-ids", default="[0,1,2,3,4,5,6,7,8,9]")
    ap.add_argument("--n-action-steps", type=int, default=10)
    ap.add_argument("--work", default="/work/plenum_a3")
    ap.add_argument("--dtype", default="fp32", choices=("fp32", "bf16"),
                    help="bf16 makes the REAL AdamW intervention fit a 44GB card (~24GB of state vs "
                         "~36GB in fp32). A precision choice, not a recipe change.")
    ap.add_argument("--optim", default="adamw", choices=("adamw", "sgd"),
                    help="sgd is for the DRY-RUN ONLY: it verifies loop mechanics on a 44GB card. "
                         "A U3c trained under sgd is a different recipe, not a valid U3c.")
    ap.add_argument("--calib-eval-only", action="store_true",
                    help="walk the frozen ladder over checkpoints ALREADY on S3, without training. "
                         "Lets an interrupted calibration be finished from what it banked: the "
                         "procedure stops at the first rung reaching the threshold, so if B* is among "
                         "the banked rungs no further training is needed.")
    ap.add_argument("--calibrate", action="store_true",
                    help="the LACUNA frozen budget procedure (§2.3 defers to it): on U1 only, "
                         "B* = smallest 2^k steps reaching pc_success >= 0.5 on a 3-task probe, "
                         "capped at 16k. Cap hit => VOID-budget (a pre-registered terminal state).")
    ap.add_argument("--probe-task-ids", default="[0,1,2]",
                    help="the 3-task probe the frozen calibration is defined on")
    ap.add_argument("--dry-run", action="store_true",
                    help="feasibility gate: 2 co-train steps on real batches, then exit")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    from infra.aws_pg.jobs.n5a_pi0_libero_eval import _install
    _install()
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch
    from infra.pg import ctrliface_lerobot as CL

    t0 = time.time()
    rep = {"object": "plenum_a3_v1", "unit": a.unit, "steps": a.steps, "seed": a.seed,
           "cotrain_ratio": a.cotrain_ratio, "dataset": DATASET_REPO, "eval_suite": EVAL_SUITE,
           "gpu": torch.cuda.get_device_name(0), "dry_run": bool(a.dry_run),
           "population_note": ("TRAINING = combined LIBERO (no per-episode suite tag exists); "
                               "EVAL = per-suite via the env. Re-scoped vs LACUNA §2.3, frozen "
                               "before any training step.")}
    cells = []
    rep["cells"] = cells                       # bound BEFORE the loops (fleet lesson)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    src = sync_unit(UNIT_SRC[a.unit], f"{a.work}/{a.unit}")
    rep["unit_path"] = src
    _w()

    # ---- THE BUDGET CALIBRATION (LACUNA frozen procedure, which §2.3 defers to) ----------
    # "on U1 only, B* = smallest 2^k steps reaching pc_success >= 0.5 on a 3-task probe, capped at
    # 16k steps. Cap hit -> VOID-budget: learning curves reported exploratory, no confirmatory
    # verdict."
    #
    # B* was never computed. Every Arm-3 savings number is stated at {B*/4, B*}, so without it the
    # savings cells have no budget to be measured at -- and the VOID-budget branch is the honest
    # terminal state if the cap is hit, reachable BY MEASUREMENT rather than by an excuse about
    # hardware. That is the whole reason to run this rather than pick a number.
    if a.calibrate or a.calib_eval_only:
        assert a.unit == "U1", "the frozen procedure calibrates on U1 only"
        # Same unification as below: the calibration trains through THIS job's loop, not through a
        # `lerobot-train` subprocess, so B* is measured on the same harness the film and U3c use.
        # A B* calibrated with a different trainer would not be the budget the other units run at.
        a.steps = CALIB_CAP
        a.film_points = CALIB_CAP // 512      # save every 512 -> every 2^k rung is a real checkpoint
        # `rep` was built with the DEFAULT --steps before this branch ran, so the artifact would claim
        # the calibration ran 4000 steps while the loop actually runs to the 16k cap. The run was
        # correct and the cap is recorded under `calibration.cap`, but a field that disagrees with what
        # happened is a field that will be quoted wrongly later.
        rep["steps"] = a.steps
        rep["film_points"] = a.film_points
        rep["calibration"] = {
            "procedure": "LACUNA frozen: smallest 2^k steps reaching pc_success >= 0.5 on a 3-task "
                         "probe, capped at 16k. Cap hit => VOID-budget (learning curves exploratory, "
                         "no confirmatory savings verdict).",
            "ladder": [2 ** k for k in range(9, 15)], "probe_task_ids": a.probe_task_ids,
            "cap": CALIB_CAP, "harness": "this job's loop (not lerobot-train)", "rungs": []}
        _w()

    # `cotrain` is the ONLY thing that differs between U3 and U3c (see the note below). It was
    # deleted by accident when the lerobot-train branch was removed, which cost both U3 and U1 a run
    # with `NameError: name 'cotrain' is not defined`.
    cotrain = (a.unit == "U3c")

    # ONE TRAINING LOOP FOR EVERY FILM UNIT. This used to hand U1/U3/U4 to `lerobot-train` as a
    # subprocess and keep the custom loop for U3c. That is a confound, not a shortcut: §2.3 reads
    # recipe-causality as U3c − U3, so those two units must differ in EXACTLY ONE thing — whether a
    # VL supervision batch is interleaved. Training them with two different harnesses would make
    # every U3c−U3 difference ambiguous between "the co-training did it" and "the trainer did it",
    # which is the same "two harnesses" failure the map's uniformity principle exists to prevent
    # (§3.2).
    #
    # It is also the only path that is known to run: the bf16 AdamW dtype work that passed the
    # feasibility gate lives here, and `lerobot-train` would need its own (differently spelled)
    # precision flags to fit a 44GB card at all. The immediate trigger was `lerobot-train: error:
    # unrecognized arguments: --eval_freq=0`, but the flag was never the real problem.
    #
    # So `cotrain` now decides only whether the VL interleave happens, and nothing else differs.

    # ---- the U3c co-training loop (and the feasibility dry-run) ---------------------------
    from transformers import AutoProcessor
    from infra.pg import inherit_pg1 as IP
    from infra.pg import lacuna_shell as LS

    # U3/U3c are PI0-architecture checkpoints (lacuna_units built them with PI0Config), while
    # `CL.load_policy` hardcodes PI05Policy. Load the class the checkpoint's own config names.
    policy = load_policy_for(src, "cuda", dtype=a.dtype)
    rep["policy_dtype"] = a.dtype
    meta = CL.dataset_meta(DATASET_REPO)
    dts = CL.build_delta_timestamps(policy, meta)
    # `build_loader` returns (loader, n_episodes, filtered) — not a bare DataLoader. suites=None
    # is the PLENUM re-scope: train on the COMBINED LIBERO dataset, because no per-episode suite
    # tag exists to express a suite-scoped training population (the LACUNA-recorded defect).
    loader, n_ep, filtered = CL.build_loader(DATASET_REPO, dts, suites=None,
                                             batch_size=a.batch_size, shuffle=True)
    assert not filtered, "the training population must be the COMBINED dataset (§2.3 re-scope)"
    rep["n_train_episodes"] = n_ep
    loss_fn = CL.loss_fn_factory(policy)
    proc = AutoProcessor.from_pretrained(LS.PG1_REPO)
    tasks = LS.mm_tasks()
    # THE VL STREAM MUST NOT DRAW FROM THE BATTERY'S EVALUATION ITEMS. `plenum_fc`'s frozen count for
    # vqav2 is 4000 and it calls the SAME `load_items(tasks["vqav2"], 4000)`, so supervising from the
    # first 4000 trains the model on the exact items its VQA cell is scored against. At
    # cotrain_ratio=4 over 4096 steps that is ~1024 draws from the 4000 evaluated items — roughly a
    # quarter of the eval set seen during training.
    #
    # Take a DISJOINT slice above the battery's sample, the same construction the held-out VL probe
    # uses. The supervision is the same distribution and the same size; only the overlap is removed.
    BATTERY_VQA_N = 4000
    _all, _h = IP.load_items(tasks["vqav2"], BATTERY_VQA_N + 4000)
    assert len(_all) > BATTERY_VQA_N, (
        f"only {len(_all)} vqav2 rows available; cannot cut a supervision slice disjoint from the "
        f"battery's first {BATTERY_VQA_N}")
    vqa_rows = _all[BATTERY_VQA_N:]
    rep["vl_stream_disjoint_from_battery"] = {
        "battery_n": BATTERY_VQA_N, "supervision_rows": len(vqa_rows),
        "why": "supervising from the battery's own evaluation sample is train/test contamination of "
               "the unit's VQA cell"}
    rep["n_vqa_supervision_rows"] = len(vqa_rows)

    # The VL head: the policy's own PaliGemma half.
    #
    # Dry-run 1 fired the pivot for the WRONG reason — `CL.get_backbone` returns a 3-tuple
    # `(module, layers, d)` whose module is the language_model deep inside, so a `.paligemma`
    # attribute lookup on it was always going to be None. Declaring "recipe-causality UNTESTED"
    # because I misread a return type would have been a FALSE terminal state, which is exactly the
    # way a map gets "completed" dishonestly. The wrapper is one level up from
    # `ctrliface_lerobot.BACKBONE_PATH`; navigate there explicitly.
    PALI_PATH = "model.paligemma_with_expert.paligemma"
    assert CL.BACKBONE_PATH.startswith(PALI_PATH), \
        f"the house backbone path {CL.BACKBONE_PATH!r} no longer sits under {PALI_PATH!r}"
    pali = policy
    for part in PALI_PATH.split("."):
        pali = getattr(pali, part, None)
        if pali is None:
            break
    _bb, _layers, _d = CL.get_backbone(policy)
    rep["backbone_layers"] = len(_layers)
    rep["paligemma_type"] = type(pali).__name__ if pali is not None else None
    rep["paligemma_forward_accepts_labels"] = (
        None if pali is None else "labels" in __import__("inspect").signature(pali.forward).parameters)
    _w()
    if pali is None:
        rep["U3c_feasibility"] = {
            "state": "VOID", "gate": "cotrain_loader_infeasible",
            "why": f"no module at {PALI_PATH!r} on this policy; the VL supervision stream has no "
                   f"head to attach to",
            "pivot": "§9 pre-registered: recipe-causality UNTESTED, film proceeds"}
        _w()
        print("PLENUM_A3_PIVOT no paligemma submodule — U3c VOID-feasibility", flush=True)
        return

    # The policy's forward needs `observation.language.tokens`, which the raw LeRobotDataset batch
    # does not carry — lerobot's own trainer adds it through the checkpoint's processor pipeline
    # (`policy_preprocessor.json`). Dry-run 3 died on exactly that KeyError.
    #
    # This block REPORTS what the installed lerobot exposes rather than guessing an API: the
    # candidates are banked into metrics.json so the next run is chosen from evidence. It is a
    # diagnostic, not an error handler — nothing is swallowed, and the assert below still fails
    # loud if no entry point exists.
    import importlib
    import inspect as _insp
    cands = {}
    for modname in ("lerobot.processor", "lerobot.policies.factory", "lerobot.configs.policies"):
        m = importlib.import_module(modname) if importlib.util.find_spec(modname) else None
        if m is not None:
            cands[modname] = sorted(n for n in dir(m)
                                    if "process" in n.lower() or "make_" in n.lower())
    cands["policy_attrs"] = sorted(n for n in dir(policy)
                                   if "process" in n.lower() or "normal" in n.lower())
    rep["processor_entry_points"] = cands
    _w()
    print(f"[A3] processor candidates: {json.dumps(cands)[:400]}", flush=True)
    # Module chosen from the banked diagnostic, not guessed: `make_pre_post_processors` lives in
    # `lerobot.policies.factory`. `lerobot.processor` exposes only the `make_default_*` family,
    # which would build DEFAULT steps rather than the ones this checkpoint's
    # policy_preprocessor.json specifies — a silently different preprocessing pipeline.
    from lerobot.policies.factory import make_pre_post_processors
    pre, post = make_pre_post_processors(policy.config, pretrained_path=src)
    rep["preprocessor"] = type(pre).__name__
    _w()

    # OPTIMIZER, and an honest scope note. The U3c intervention (§2.3) is U3 + VL co-training under
    # the normal recipe, which means AdamW — ~36GB of optimizer state over a ~3B policy, so the full
    # run needs an 80GB card. AWS has returned InsufficientInstanceCapacity on every H100/H200
    # family for hours (the P-family QUOTA is 128 vCPU, so this is capacity, not a limit of ours).
    #
    # `--optim sgd` exists ONLY for the dry-run: SGD holds no moments, so the co-train LOOP —
    # LIBERO batch → action loss → backward → step, interleaved with a VL supervision step through
    # the policy's own PaliGemma head — can be verified end-to-end on a 44GB L40S. That verifies
    # MECHANICS, not the intervention: a U3c trained under SGD is a different recipe and would not
    # be a valid U3c. The feasibility record says which optimizer ran, so the two can never be
    # confused downstream.
    opt = (torch.optim.SGD(policy.parameters(), lr=1e-5) if a.optim == "sgd"
           else torch.optim.AdamW(policy.parameters(), lr=1e-5))
    rep["optimizer"] = a.optim
    if a.optim != "adamw":
        rep["scope_note"] = ("LOOP MECHANICS ONLY — SGD holds no optimizer moments. A U3c trained "
                             "this way is a DIFFERENT RECIPE and is not a valid U3c. The full "
                             "intervention requires AdamW, which fits a 44GB card in bf16 "
                             "(--dtype bf16); the 80GB figure was an fp32-master-copy assumption.")
    _w()
    if a.dtype == "bf16":
        # THE INTERNAL fp32 LEAK. Casting the batch got the forward past `state_proj` and straight
        # into `action_in_proj`, because `noisy_actions` is not in the batch at all -- lerobot's pi0
        # forward samples `noise` (and `time`) itself, with torch's DEFAULT dtype, and interpolates
        # them with the actions. No amount of batch casting can reach a tensor the model creates.
        #
        # Setting the default dtype is the lever that actually matches: every tensor the policy
        # constructs internally is then bf16 like its weights. This is what "the policy is a bf16
        # policy" has to mean end to end -- a half-bf16 forward is not a cheaper AdamW, it is a
        # crash. Set here rather than at load time so it covers the loop and nothing before it.
        # NOT set globally. An earlier version called this once before the loop, which broke π0.5:
        # its preprocessor (`processor_pi05.py`) creates a tensor and calls .numpy() on it, and numpy
        # has no bfloat16 — "TypeError: Got unsupported ScalarType BFloat16", raised inside
        # `pre(batch)` before the model is even reached. π0's processor does not do that, which is
        # why U1/U3/U3c never hit it. The default dtype is now scoped to the forward/backward only
        # (see the loop), so preprocessing runs under fp32 defaults as the library expects.
        # ...and the boundary cast alone was NOT enough. The failure stayed at `action_in_proj(noisy_actions)`, because
        # lerobot's flow-matching noise sampler passes an EXPLICIT dtype=float32 (the reference
        # openpi implementation does this deliberately, for noise stability). An explicit dtype
        # ignores the default, and `x_t = f(noise, actions, time)` is interpolated INSIDE
        # `model.forward`, upstream of every projection — so there is no batch-side or default-side
        # fix. The cast has to happen at the model.forward boundary.
        #
        # The signature is known from the traceback:
        #   model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        # Wrapping it and casting the floating args covers noise and time without depending on the
        # sampler's method name, and is robust to argument order.
        #
        # DISCLOSED PRECISION DEVIATION, and it travels with every U3c result: the reference
        # implementation interpolates x_t in fp32 and this interpolates in bf16. That is a precision
        # change, not a recipe change (the optimizer is still AdamW, the mixing ratio and data are
        # untouched), but it is a real difference from the reference and the fp32 alternative needs
        # an 80GB card that has not been obtainable. Recorded in the artifact, not just here.
        _model_fwd = policy.model.forward

        def _model_fwd_bf16(*args, **kw):
            cast = lambda x: (x.to(torch.bfloat16)
                              if torch.is_tensor(x) and x.is_floating_point() else x)
            out = _model_fwd(*[cast(x) for x in args],
                             **{k: cast(v) for k, v in kw.items()})
            # RETURN in fp32. With the input casts in place the forward and the loss both COMPUTE
            # fine in bf16 — the next thing to break was lerobot's own logging line,
            # `losses.mean(dim=[0,1]).detach().cpu().numpy()`, because numpy has no bfloat16
            # ("TypeError: Got unsupported ScalarType BFloat16"). Upcasting the returned losses
            # costs no precision (bf16 -> fp32 is exact) and keeps autograd intact, so the
            # optimizer still steps on the same values.
            up = lambda x: (x.to(torch.float32)
                            if torch.is_tensor(x) and x.dtype == torch.bfloat16 else x)
            if torch.is_tensor(out):
                return up(out)
            if isinstance(out, dict):
                return {k: up(v) for k, v in out.items()}
            if isinstance(out, (list, tuple)):
                return type(out)(up(v) for v in out)
            return out

        policy.model.forward = _model_fwd_bf16

        # ...and that reached the FOURTH dtype boundary: state_proj, then action_in_proj, then
        # action_out_proj on the way out (the forward now runs the whole PaliGemma+expert stack and
        # dies at line 639). Patching one call site per 45-minute run is the wrong strategy — the
        # class of bug is "an fp32 tensor produced inside library code meets a bf16 Linear", and the
        # sites are wherever lerobot chose to upcast.
        #
        # So kill the class instead of the instances: a forward_pre_hook on EVERY Linear and every
        # norm, casting its inputs to that module's own parameter dtype. Generic, covers the sites I
        # have hit and the ones I have not, and it is the same precision decision already disclosed
        # above rather than a new one — a cast that would otherwise happen inside the matmul.
        def _cast_to_param_dtype(mod, args):
            w = getattr(mod, "weight", None)
            if w is None:
                return None
            return tuple(x.to(w.dtype)
                         if torch.is_tensor(x) and x.is_floating_point() and x.dtype != w.dtype
                         else x for x in args)

        n_hooked = 0
        for m in policy.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.LayerNorm)):
                m.register_forward_pre_hook(_cast_to_param_dtype)
                n_hooked += 1
        rep["bf16_input_casts"] = {
            "n_modules_hooked": n_hooked,
            "why": "an fp32 tensor produced inside lerobot meets a bf16 Linear; four such sites were "
                   "hit one run at a time (state_proj, action_in_proj, action_out_proj) before "
                   "hooking the class instead of the instances"}
        rep["default_dtype"] = "bfloat16"
        rep["precision_deviation"] = (
            "x_t is interpolated in bf16, not fp32. lerobot/openpi samples flow-matching noise with "
            "an explicit dtype=float32; a bf16 policy cannot consume it, and the fp32 formulation of "
            "AdamW needs an 80GB card that has not been obtainable. Precision only -- optimizer, "
            "mixing ratio and data are unchanged. Travels with every U3c result.")
        _w()
    # Build the remap from the policy's OWN declared image features against what the loader supplies.
    # Derived, not hardcoded: a checkpoint that declares different camera names is covered, and the
    # mapping that was actually used is recorded in the artifact.
    cam_map = None            # derived from the FIRST batch (the loader's keys are only known then)
    cam_want = sorted(k for k, v in getattr(policy.config, "input_features", {}).items()
                      if "VISUAL" in str(getattr(v, "type", "")))

    # `--calib-eval-only` walks the ladder over banked checkpoints and trains nothing.
    n_dry = 0 if a.calib_eval_only else (2 if a.dry_run else a.steps)
    save_freq = max(1, a.steps // a.film_points)
    # ARCHIVE ANY EXISTING FILM CHECKPOINTS BEFORE OVERWRITING THEM. A re-run writes the same step
    # numbers to the same prefix, so a reader active during the overwrite (the held-out probe was, and
    # its instance died mid-read) can see a MIX of two runs' weights at different steps — which would
    # be a meaningless measurement that looks like a valid one.
    #
    # Moving the old set aside rather than deleting it keeps the superseded run auditable and makes the
    # prefix atomic from a reader's point of view: it holds one run's checkpoints or the other's.
    if not a.dry_run and not a.calib_eval_only:
        _pfx = f"{S3_FILM}/{a.unit}/seed{a.seed}"
        _existing = subprocess.run(["aws", "s3", "ls", f"{_pfx}/"], capture_output=True, text=True).stdout
        if _existing.strip():
            _arch = f"{S3_FILM}/_superseded/{a.unit}_seed{a.seed}_{int(t0)}"
            print(f"[A3] archiving existing film checkpoints -> {_arch}", flush=True)
            _sh(["aws", "s3", "mv", "--recursive", f"{_pfx}/", f"{_arch}/"])
            rep["superseded_checkpoints"] = {"moved_to": _arch,
                                            "why": "a re-run writes the same step numbers; a reader "
                                                   "active during the overwrite would see a mix"}
            _w()
    ckdir = pathlib.Path(a.work) / f"train_{a.unit}_{a.seed}" / "checkpoints"
    ckdir.mkdir(parents=True, exist_ok=True)
    rng = torch.Generator().manual_seed(a.seed)
    step = 0
    vl_losses, act_losses = [], []
    for batch in loader:
        if step >= n_dry:
            break
        batch = CL.to_device(pre(batch), "cuda")
        if cam_map is None:
            have = sorted(k for k in batch if k.startswith("observation.images."))
            if cam_want and have and not set(cam_want) & set(have):
                assert len(have) <= len(cam_want), (
                    f"the loader supplies more cameras ({have}) than the policy declares "
                    f"({cam_want}); dropping one would silently change the observation")
                cam_map = dict(zip(have, cam_want))
                rep["camera_remap"] = {
                    "map": cam_map, "policy_declares": cam_want, "loader_supplies": have,
                    "why": "the published policies declare their own training mix's cameras while "
                           "LIBERO supplies image/image2; the DATA is adapted to the policy's "
                           "declared inputs, which is not a recipe change"}
                print(f"[A3] camera remap {cam_map}", flush=True)
            else:
                cam_map = {}
                rep["camera_remap"] = {"map": {}, "why": "policy and loader agree on camera keys"}
            _w()
        if cam_map:
            # CAMERA-KEY REMAP, for U1/U2 only. The published π0/π0.5 declare their own training
            # mix's cameras — `observation.images.{base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb}` —
            # while the LIBERO dataset supplies `{image, image2}`. Without a remap lerobot raises
            # "All image features are missing from the batch", which is the SAME structural mismatch
            # that stops `lerobot-eval` pairing π0 with the LIBERO env.
            #
            # This is an adaptation of the DATA to the policy's declared inputs, not a change to the
            # policy or the recipe, and it is what the reference LIBERO fine-tuning recipe does
            # (agentview -> base_0_rgb, wrist -> left_wrist_0_rgb). §2.3's population defines U1 as π0
            # trained on LIBERO, which is not expressible at all without it. The banked U3/U4 need no
            # remap because their configs were built LIBERO-shaped, which is why they train as-is.
            for src_k, dst_k in cam_map.items():
                if src_k in batch:
                    batch[dst_k] = batch[src_k]
        if a.dtype == "bf16":
            # The loader and the policy preprocessor emit fp32 (state, actions, images); a bf16
            # policy's Linear then hits "mat1 and mat2 must have the same dtype". Cast the FLOATING
            # tensors only -- masks and token ids must keep their own dtypes or the attention and
            # embedding paths break in ways that are much harder to see than this error was.
            batch = {k: (v.to(torch.bfloat16)
                         if torch.is_tensor(v) and v.is_floating_point() else v)
                     for k, v in batch.items()}
        # bf16 default ONLY across the forward/backward: lerobot samples flow-matching noise with
        # torch's default dtype inside model.forward, so it must be bf16 there — but the processor
        # that ran above must see fp32 defaults or π0.5's .numpy() call dies.
        if a.dtype == "bf16":
            torch.set_default_dtype(torch.bfloat16)
        loss = loss_fn(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if a.dtype == "bf16":
            torch.set_default_dtype(torch.float32)
        act_losses.append(float(loss.detach()))
        if cotrain and step % a.cotrain_ratio == 0:
            idx = torch.randint(0, len(vqa_rows), (a.vl_batch_size,), generator=rng).tolist()
            enc, labels = vl_batch(proc, tasks, [vqa_rows[i] for i in idx], "cuda")
            vlo = pali(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                       token_type_ids=enc["token_type_ids"],
                       pixel_values=enc["pixel_values"].to(next(pali.parameters()).dtype),
                       labels=labels)
            vl_loss = vlo.loss if hasattr(vlo, "loss") else vlo["loss"]
            assert vl_loss is not None and torch.isfinite(vl_loss), \
                "VL co-training step produced no finite loss"
            opt.zero_grad(set_to_none=True)
            vl_loss.backward()
            opt.step()
            vl_losses.append(float(vl_loss.detach()))
        step += 1
        if not a.dry_run and step % save_freq == 0:
            cpath = ckdir / f"{step:08d}" / "pretrained_model"
            policy.save_pretrained(cpath)
            # SAVE THE PROCESSOR TOO, or the checkpoint is unusable. `policy.save_pretrained` writes
            # weights + config but NOT `policy_preprocessor.json` / `policy_postprocessor.json`, and
            # `lerobot-eval` refuses such a checkpoint with ProcessorMigrationError. The banked
            # lacuna_units carry them, which is why THOSE evaluate and these did not.
            #
            # Writing `pre`/`post` here is better than running lerobot's migration script after the
            # fact (which reported success and produced no files — the assertion in plenum_control
            # caught that): these are the processor objects the training loop ACTUALLY used, built from
            # the source unit and never recomputed, so saving them records what happened rather than
            # reconstructing a guess at it.
            pre.save_pretrained(cpath)
            post.save_pretrained(cpath)
            assert (cpath / "policy_preprocessor.json").exists(), (
                f"processor not written into {cpath} — the checkpoint would be unevaluable")
            # PUSH EVERY CHECKPOINT TO S3 AS IT IS WRITTEN. The instance self-terminates, so a
            # checkpoint that exists only on local disk is a checkpoint that does not exist. Two
            # consumers need these: the FILM (the Tier-1 battery at every checkpoint, §2.3) and
            # `plenum_units.POLICY_UNITS["u3c"]`, which is an S3 URI — u3c's 25 map cells are
            # unreachable until the final weights land there.
            dst = f"{S3_FILM}/{a.unit}/seed{a.seed}/{step:08d}/pretrained_model"
            _sh(["aws", "s3", "sync", str(cpath), dst])
            # PUBLISH THE UNIT FROM LOCAL DISK, not by an S3->S3 copy afterwards. Both `s3 sync` and
            # `s3 cp --recursive` failed on the instance for the 8GB weights file while succeeding from
            # the control plane, which points at the instance role lacking multipart server-side copy
            # (upload-part-copy) rights. Uploading the local file twice needs only PutObject, which the
            # instance demonstrably has — it just wrote the film copy.
            if a.unit == "U3c" and a.seed == 0 and step >= a.steps:
                _sh(["aws", "s3", "sync", str(cpath), S3_U3C])
                _chk = subprocess.run(["aws", "s3", "ls", f"{S3_U3C}/"], check=True,
                                      capture_output=True, text=True).stdout
                assert "model.safetensors" in _chk, f"u3c publish incomplete:\n{_chk}"
                rep["u3c_published"] = {"uri": S3_U3C, "from_step": step, "via": "local upload"}
                _w()
            # FREE THE LOCAL COPY. Each checkpoint is ~8GB and the loop kept every one on local disk,
            # so U1's 32-checkpoint calibration needed ~258GB and died at step 5632 with
            # "No space left on device" AFTER 11 checkpoints had already been written. U3's 8-point
            # film fit under the limit, which is why this only surfaced on the long run.
            # S3 is the store; local is scratch. The eval ladder below re-syncs the rung it needs.
            _sh(["rm", "-rf", str(cpath.parent)])
            cells.append({"unit": a.unit, "seed": a.seed, "step": step,
                          "action_loss": act_losses[-1],
                          "vl_loss": (vl_losses[-1] if vl_losses else None),
                          "elapsed_s": round(time.time() - t0, 1)})
            print(f"[A3] {a.unit} step={step} act={act_losses[-1]:.4f} "
                  f"vl={vl_losses[-1] if vl_losses else None}", flush=True)
            _w()

    if a.dry_run:
        rep["U3c_feasibility"] = {
            "state": "FEASIBLE", "n_action_steps_run": len(act_losses),
            "n_vl_steps_run": len(vl_losses), "action_losses": act_losses,
            "vl_losses": vl_losses,
            "note": "the co-train loop takes real LIBERO batches AND real VL supervision batches "
                    "through the policy's own PaliGemma half; both losses are finite"}
        _w()
        print(f"PLENUM_A3_DRYRUN_OK act={act_losses} vl={vl_losses}", flush=True)
        return

    # ---- the calibration ladder, on the checkpoints this loop just wrote ---------------------
    # BOTH modes run the ladder. `--calib-eval-only` previously set up the calibration metadata in the
    # early branch and then skipped this block, because only the early branch had been widened — so it
    # exited 0 with `rungs: []` and B_star None, reporting nothing while looking clean.
    if a.calibrate or a.calib_eval_only:
        bindir = pathlib.Path(sys.executable).parent
        binev = ([str(bindir / "lerobot-eval")] if (bindir / "lerobot-eval").exists()
                 else [sys.executable, "-m", "lerobot.scripts.eval"])
        b_star = None
        for nst in rep["calibration"]["ladder"]:
            # Re-sync from S3: local copies are deleted after upload (the disk-exhaustion fix).
            ck = pathlib.Path(a.work) / "calib_rungs" / f"{nst:08d}"
            ck.mkdir(parents=True, exist_ok=True)
            _sh(["aws", "s3", "sync",
                 f"{S3_FILM}/{a.unit}/seed{a.seed}/{nst:08d}/pretrained_model", str(ck)])
            if not list(ck.glob("*.safetensors")):
                rep["calibration"].setdefault("missing_rungs", []).append(nst)
                print(f"[A3] calib rung {nst}: no checkpoint on S3 — stopping the ladder here",
                      flush=True)
                _w()
                break
            ev = out / f"calib_eval_{nst}"
            _sh(binev + [f"--output_dir={ev}", "--env.type=libero",
                         f"--env.task={EVAL_SUITE}", f"--env.task_ids={a.probe_task_ids}",
                         "--eval.batch_size=1", f"--eval.n_episodes={a.eval_episodes}",
                         f"--policy.path={ck}", f"--policy.n_action_steps={a.n_action_steps}",
                         "--env.max_parallel_tasks=1"], env={**os.environ})
            import glob as _g
            pcs = []
            for f in sorted(_g.glob(str(ev / "**" / "*.json"), recursive=True)):
                d = json.loads(pathlib.Path(f).read_text())
                if isinstance(d, dict) and isinstance(d.get("overall"), dict) \
                        and "pc_success" in d["overall"]:
                    pcs.append(float(d["overall"]["pc_success"]))
            assert pcs, f"calibration rung {nst}: no overall.pc_success under {ev}"
            rep["calibration"]["rungs"].append({"steps": nst, "pc_success": pcs[-1]})
            print(f"[A3] calib {nst} steps pc_success={pcs[-1]:.4f}", flush=True)
            _w()
            if pcs[-1] >= 0.5:
                b_star = nst
                break
        rep["calibration"]["B_star"] = b_star
        rep["calibration"]["state"] = "CALIBRATED" if b_star else "VOID-budget"
        if not b_star:
            rep["calibration"]["why"] = (
                "the cap was reached without a rung at or above the frozen pc_success threshold; per "
                "the frozen procedure the learning curves are exploratory and no confirmatory savings "
                "verdict is available")
        _w()
        print(f"PLENUM_A3_CALIB_OK B_star={b_star} state={rep['calibration']['state']}", flush=True)

    # The canonical `u3c` UNIT (the one the battery installs) is the final checkpoint. Publish it
    # at the plain `lacuna_units/U3c` path that `plenum_units.POLICY_UNITS` already points at, so
    # the 25 u3c cells become reachable without touching the frozen unit table.
    assert cells or a.calib_eval_only, (
        f"no checkpoint was written — there is no {a.unit} to publish")
    if cells and a.unit == "U3c" and a.seed == 0:
        # Local copies are deleted after upload, so publish by copying S3 -> S3.
        # `aws s3 sync` failed on this 8GB S3->S3 copy (exit 1 after moving only config.json), which
        # is why the run exited 1 AFTER all 8 checkpoints were safely written — training complete,
        # publish broken. `cp --recursive` handles the multipart copy; the assertion then proves the
        # weights actually arrived rather than trusting the exit code, because a half-published unit
        # is worse than an unpublished one: the battery's "does it carry safetensors" gate would open
        # on a unit with no weights.
        # Published from local disk inside the loop (see the note at the final checkpoint) — an
        # S3->S3 copy of the 8GB weights fails on the instance role.
        assert rep.get("u3c_published"), "the final checkpoint did not publish the u3c unit"
        rep["u3c_published"] = {"uri": S3_U3C, "from_step": cells[-1]["step"], "seed": a.seed}
    rep["elapsed_s"] = round(time.time() - t0, 1)
    _w()
    print(f"PLENUM_A3_OK {a.unit} checkpoints={len(cells)}", flush=True)


if __name__ == "__main__":
    main()
