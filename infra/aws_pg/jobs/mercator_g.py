"""M-G — repair-π0 readers: can 32 numbers per layer give a lost faculty back? (standing W1-G).

π0's backbone is a Gemma-1 decoder that was trained for robot control. Installed back into the PG1
vision-language shell it produces a model that has LOST captioning and VQA — a documented lost
faculty, not a synthetic ablation. That is the setting this arm needs, and it is why the arm is
about REPAIR rather than about damage.

    THE TREATMENT. A rank-r READER per layer, confined to a bank:

        h_L  <-  h_L + U_L A_L (U_L^T h_L)            A_L is r x r, U_L is the bank

    It reads the residual stream ONLY through the bank's 32 coordinates, writes back only into
    them, and carries r^2 parameters per layer. Everything else — the shell, the backbone, the
    vision tower, the projector — stays frozen.

    REFUTER (pre-logged): core readers <= twin readers.

WHAT MAKES IT A MEASUREMENT:

  * EQUAL CAPACITY BY CONSTRUCTION. Every bank gets the same A_L shape, so the comparison across
    banks is a comparison of WHERE the reader looks, with the parameter count held fixed. That is
    the one thing a "the core is special" claim needs and the easiest thing to lose.
  * A FLOOR AND A CEILING, both measured in the same run. The floor is the un-repaired re-shell:
    whatever the readers achieve has to beat doing nothing. The ceiling is a small full fine-tune
    of q_proj at the same step budget — vastly more parameters — so "the readers recover X% of
    what is recoverable" is a statement with a denominator.
  * A CAPACITY DOSE LADDER. r in {4, 8, 16, 32}: a monotone knob, so a bank that wins only at one
    capacity is visible as such rather than as a headline.
  * TWO FACULTIES. COCO captioning (corpus CIDEr-D) and VQAv2 (per-item accuracy), because a
    repair that only moves a corpus metric is a different claim from one that also moves an
    item-level one.

The re-shell, the task battery, and the eval runner are all imported unchanged
(`lacuna_shell.build_row_policy`, `projector_ablation.build_tasks/eval_task`), so a difference
between this arm and LACUNA's numbers cannot be a difference of harness.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as WT
from infra.pg import mercator_core as M
from infra.pg import mercator_program as P

ARM = "M-G"
# FROZEN BEFORE THE RUN (defect M-D42). Uniform-random cross-entropy on this vocabulary is about
# 12.4 nats; a supervised loss on the INTACT shell above this bound means the targets or the label
# masking are wrong, not that the task is hard. The descent bound is deliberately modest — the
# probe asks whether the objective moves the model AT ALL, not whether it trains it well.
MAX_SANE_INTACT_LOSS = 10.0
MIN_PROBE_DESCENT = 0.02
BANKS = ("core32", "anti32", "complement32", "topsigma32",
         "haar32_s0", "haar32_s1", "haar32_s2")
RANKS = (4, 8, 16, 32)
TASKS = ("coco_captions", "vqav2")
# The re-shell's own key for the policy, not the repo id: `lacuna_shell.POLICY_REPOS` maps
# "pi0" -> "lerobot/pi0_base", and passing the repo id instead raised KeyError('pi0_base') after
# the arm had already evaluated both intact tasks (journal 2026-08-15 defect D29).
POLICY = "pi0"


class BankReader:
    """Per-layer rank-r reader confined to a bank: h <- h + U A (U^T h).

    Written as an additive correction inside the bank so that A = 0 is EXACTLY the un-repaired
    model — the floor and the r=0 treatment are the same forward, not two forwards that happen to
    agree. `A` is the only thing that trains.
    """

    def __init__(self, layers, device, basis_by_layer, rank: int):
        import torch
        self.layers = layers
        self.rank = int(rank)
        self.U = {}
        self.A = {}
        for L, U in basis_by_layer.items():
            assert U.shape[1] >= rank, f"bank at layer {L} has {U.shape[1]} < rank {rank}"
            self.U[L] = torch.as_tensor(U[:, :rank], device=device, dtype=torch.float32)
            # ZERO INIT, and there is no seed for that reason: at A = 0 the reader is EXACTLY the
            # un-repaired model, so every arm starts from the same forward and the floor is the
            # step-0 state of every treatment rather than a separately measured thing.
            self.A[L] = torch.zeros(rank, rank, device=device, dtype=torch.float32,
                                    requires_grad=True)
        self.keys = sorted(self.U)
        self._handles = []

    @property
    def n_params(self) -> int:
        return len(self.keys) * self.rank * self.rank

    def parameters(self):
        return [self.A[L] for L in self.keys]

    def _mk(self, L):
        import torch

        def pre_hook(_m, args):
            x = args[0]
            U, A = self.U[L], self.A[L]
            z = x.to(torch.float32) @ U
            return (x + ((z @ A) @ U.T).to(x.dtype),) + tuple(args[1:])
        return pre_hook

    def __enter__(self):
        for L in self.keys:
            self._handles.append(
                self.layers[L].self_attn.q_proj.register_forward_pre_hook(self._mk(L)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


def main():
    ap = argparse.ArgumentParser(description="M-G — repair-pi0 readers")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ranks", default=",".join(str(r) for r in RANKS))
    ap.add_argument("--banks", default=",".join(BANKS))
    ap.add_argument("--n-train", type=int, default=192)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ft-lr", type=float, default=1e-5)
    ap.add_argument("--probe-steps", type=int, default=40,
                    help="steps of the intact-shell objective probe (M-D42 instrument gate)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ranks = [int(r) for r in a.ranks.split(",") if r.strip()]
    banks_wanted = [b.strip() for b in a.banks.split(",") if b.strip()]
    M.check_dose_grid([float(r) for r in ranks], arm=ARM)
    meta = P.ARMS[ARM]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_w1") / "expectations.jsonl")
    expected_ref = exp.require(ARM)
    at = M.Atlas(out, arm=ARM, run_id="m_g", register=meta["register"], wave=meta["wave"],
                 expected_ref=expected_ref, queue_item=meta["queue_item"])
    sur = M.SurpriseLog(out)
    gates = H.Gates(out)

    import torch
    from contextlib import nullcontext
    from huggingface_hub import HfApi
    from transformers import PaliGemmaProcessor
    from infra.pg import lacuna_shell as LS
    from infra.pg import projector_ablation as PA
    from infra.pg import darwin_train as T
    from infra.aws_pg.jobs.invariant_core_func import _decoder_layers, _load_full_pg1

    assert torch.cuda.is_available(), "no CUDA"
    assert POLICY in LS.POLICY_REPOS, (
        f"{POLICY!r} is not a key of lacuna_shell.POLICY_REPOS "
        f"({sorted(LS.POLICY_REPOS)}); the re-shell takes the policy's KEY, not its repo id")
    # The corpus (CIDEr-D) task needs pycocoevalcap, which the AMI does not carry. Installed here,
    # BEFORE any scoring, so the failure mode is a loud install error at second zero rather than a
    # crash after the first task has already been evaluated.
    M.ensure_pycocoevalcap()
    api = HfApi()
    base_sha = api.model_info(WT.PG1_REPO).sha
    model = _load_full_pg1(base_sha, "cuda")
    processor = PaliGemmaProcessor.from_pretrained(WT.PG1_REPO, revision=base_sha)
    layers = _decoder_layers(model)
    bank_dir = M.fetch_all_banks(H.cache_dir(out, "banks"))
    basis = {b: {L: WT.load_bank(bank_dir, "q_proj", L, b) for L in range(len(layers))}
             for b in banks_wanted}

    tasks = {t.name: t for t in PA.build_tasks() if t.name in TASKS}
    assert set(tasks) == set(TASKS), f"battery gave {sorted(tasks)}, wanted {sorted(TASKS)}"
    items = {k: PA.load_eval_items(t, a.n_train + a.n_eval, seed=a.seed)
             for k, t in tasks.items()}
    train_rows = {k: v[: a.n_train] for k, v in items.items()}
    eval_rows = {k: v[a.n_train:] for k, v in items.items()}
    print(f"[M-G] battery {sorted(tasks)}; {a.n_train} train / {a.n_eval} eval items each",
          flush=True)

    def evaluate(label, ctx=None):
        scores = {}
        for k, t in tasks.items():
            with (ctx if ctx is not None else nullcontext()):
                r = PA.eval_task(model, processor, t, eval_rows[k], "cuda",
                                 batch_size=min(8, a.batch * 2))
            scores[k] = r
            print(f"[M-G] {label:24s} {k:15s} score={r['score']:.4f} "
                  f"CI={r['ci95']} n={r['n']}", flush=True)
        return scores

    def supervised_batches(k, rows):
        """Teacher-forced (prompt, target) batches for one task, batch 1.

        The training target is the task's own first reference — the same string the eval scorer
        compares against — so the objective and the readout are about the same thing.
        """
        t = tasks[k]
        out_b = []
        for r in rows:
            ref = M.battery_supervised_target(t, r)
            if not str(ref).strip():
                continue
            prompt, img = t.prompt_fn(r), t.image_fn(r).convert("RGB")
            inp = processor(text=[prompt], images=[img], suffix=[str(ref)],
                            return_tensors="pt", padding=False).to("cuda")
            inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
            out_b.append(inp)
        assert out_b, f"{k}: no usable supervised rows"
        return out_b

    # BUILD THE TRAINING BATCHES FIRST. They cost seconds and they are where the per-task target
    # rules can be wrong; evaluating two tasks before discovering that VQAv2 yielded zero rows is
    # what defect M-D36 cost.
    sup = {k: supervised_batches(k, train_rows[k]) for k in TASKS}
    print("[M-G] supervised batches: " +
          " ".join(f"{k}={len(v)}" for k, v in sup.items()), flush=True)

    def train_module(params, lr, ctx, label, steps=None):
        steps = a.steps if steps is None else steps
        opt = torch.optim.AdamW(params, lr=lr)
        rng = np.random.default_rng(H.cell_seed(a.seed, ARM, label, "train"))
        order = [(k, i) for k in TASKS for i in range(len(sup[k]))]
        model.train()
        hist = []
        with ctx:
            for step in range(steps):
                loss = 0.0
                for _ in range(a.batch):
                    k, i = order[int(rng.integers(0, len(order)))]
                    loss = loss + model(**sup[k][i]).loss
                loss = loss / a.batch
                opt.zero_grad()
                loss.backward()
                opt.step()
                hist.append(float(loss.detach()))
                if (step + 1) % 50 == 0:
                    print(f"    {label} step {step+1}/{steps} loss={hist[-1]:.5f}", flush=True)
        model.eval()
        return hist

    def snapshot():
        return {n: q.detach().clone() for n, q in model.named_parameters()}

    def restore(sd):
        with torch.no_grad():
            for n, q in model.named_parameters():
                q.copy_(sd[n])

    # ---- IS THE OBJECTIVE ABLE TO DESCEND AT ALL? (defect M-D42) -------------------------------
    # The first run of this arm read loss ~26, FLAT over 300 steps, for the ceiling and for every
    # reader alike, and reported a zero headroom that turned every recovered fraction into 0/0. A
    # flat loss on a broken model and a flat loss on a broken objective look identical, so the
    # objective is exercised HERE, on the intact shell that can already do both tasks, before
    # anything is re-shelled. Two readings, because they fail differently:
    #   * the intact loss MAGNITUDE — uniform-random cross-entropy on this vocabulary is ~12.4, so
    #     a loss far above that is a labelling/target bug, not a hard task;
    #   * the DESCENT over a short probe — an objective that cannot move a model that already
    #     works will not move a damaged one either.
    # Either failure is an instrument verdict, recorded and fatal, never a science result.
    intact_sd = snapshot()
    for q in model.parameters():
        q.requires_grad_(False)
    probe_mods = T.trainable_modules(layers, types=("q_proj",), layers=list(range(len(layers))))
    for m_ in probe_mods.values():
        m_.weight.requires_grad_(True)
    probe = train_module([m_.weight for m_ in probe_mods.values()], a.ft_lr, nullcontext(),
                         "objective_probe_intact", steps=a.probe_steps)
    for m_ in probe_mods.values():
        m_.weight.requires_grad_(False)
    restore(intact_sd)
    n_edge = max(1, a.probe_steps // 4)
    start, end = float(np.mean(probe[:n_edge])), float(np.mean(probe[-n_edge:]))
    drop = (start - end) / start if start > 0 else float("nan")
    sane = bool(start < MAX_SANE_INTACT_LOSS)
    descends = bool(drop >= MIN_PROBE_DESCENT)
    print(f"[M-G] objective probe on intact: loss {start:.4f} -> {end:.4f} "
          f"({drop:+.1%} over {a.probe_steps} steps)", flush=True)
    at.add("repair_objective_probe", value=float(drop), n=a.probe_steps,
           model_id=WT.PG1_REPO, model_sha=base_sha,
           flags={"start_loss": start, "end_loss": end, "lr": a.ft_lr,
                  "uniform_random_loss": float(np.log(model.config.text_config.vocab_size)),
                  "role": "does the training objective move an INTACT model"})
    gates.record("g_objective_descends_on_intact", bool(sane and descends),
                 start_loss=start, end_loss=end, relative_drop=drop,
                 max_sane_start=MAX_SANE_INTACT_LOSS, min_descent=MIN_PROBE_DESCENT,
                 loss_is_sane=sane, loss_descends=descends,
                 outcome=None if (sane and descends) else M.VOID_INSTRUMENT,
                 note=("measured on the intact shell before any re-shell. A flat loss on the "
                       "damaged model is only evidence about the damage if the objective is "
                       "known to move a model that works"))
    assert sane, (
        f"the supervised loss on the INTACT shell is {start:.3f}, above the sanity bound "
        f"{MAX_SANE_INTACT_LOSS} — uniform-random on this vocabulary is about "
        f"{float(np.log(model.config.text_config.vocab_size)):.2f}, so the targets or the label "
        f"masking are wrong and no repair number from this run would mean anything")
    assert descends, (
        f"the objective moved the intact shell by {drop:+.2%} over {a.probe_steps} steps "
        f"(< {MIN_PROBE_DESCENT:.0%}); it cannot be expected to repair a damaged one, so this is "
        f"an instrument failure and not a result about repair")

    # ---- the three reference points -----------------------------------------------------------
    pg1_scores = evaluate("pg1_shell (intact)")
    info_reshell = LS.build_row_policy(model, POLICY, embed="own")
    floor = evaluate("pi0_reshell (floor)")
    for k in TASKS:
        at.add("repair_task_score", value=float(pg1_scores[k]["score"]),
               ci_lo=pg1_scores[k]["ci95"][0], ci_hi=pg1_scores[k]["ci95"][1],
               n=pg1_scores[k]["n"], corpus_id=tasks[k].dataset, incumbent_id="pg1_intact",
               model_id=WT.PG1_REPO, model_sha=base_sha,
               flags={"role": "intact PG1 shell", "task": k, "rank": None})
        at.add("repair_task_score", value=float(floor[k]["score"]), ci_lo=floor[k]["ci95"][0],
               ci_hi=floor[k]["ci95"][1], n=floor[k]["n"], corpus_id=tasks[k].dataset,
               incumbent_id="unrepaired_reshell", model_id=WT.PG1_REPO, model_sha=base_sha,
               flags={"role": "un-repaired pi0 re-shell (the floor)", "task": k, "rank": 0,
                      **{f"install_{kk}": vv for kk, vv in info_reshell.items()
                         if isinstance(vv, (int, float, str, bool))}})
    lost = {k: float(pg1_scores[k]["score"] - floor[k]["score"]) for k in TASKS}
    faculty_lost = all(v > 0 for v in lost.values())
    gates.record("g_reshell_actually_lost_the_faculty", faculty_lost, lost=lost,
                 pg1={k: pg1_scores[k]["score"] for k in TASKS},
                 reshell={k: floor[k]["score"] for k in TASKS},
                 outcome=None if faculty_lost else M.VOID_INSTRUMENT,
                 note=("repair is only measurable if something was lost; a re-shell that scores "
                       "as well as PG1 leaves nothing to recover"))
    assert faculty_lost, (
        f"the pi0 re-shell did not lose either faculty (deltas {lost}); there is nothing to "
        f"repair and every recovery fraction below would have a zero denominator")

    # ---- the ceiling: a LADDER, because a fixed one was weaker than the treatments -------------
    # M-D48. The first two runs set the ceiling at q_proj / lr 1e-5 while the READERS train at lr
    # 1e-3 — a hundred times stronger. A "ceiling" weaker than the arm it is supposed to bound is
    # not a ceiling, and it recovered nothing, which turned every recovery fraction into 0/0 and
    # read as "the damage is unrepairable" when it only ever showed that this fine-tune was barely
    # trained. The ladder climbs until something beats the floor with separated intervals and stops
    # there; every rung is banked, so "how much repair capacity the damage needs" is a measurement
    # rather than an assumption. The weakest rung is kept FIRST and unchanged so the old point is
    # still on the record next to the ones that bound it.
    reshell_sd = snapshot()
    ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
    rungs = [(("q_proj",), a.ft_lr), (("q_proj",), a.lr), (ATTN, a.lr)]
    ceiling, n_ceiling, rung_used, ladder = None, 0, None, []
    for types_c, lr_c in rungs:
        restore(reshell_sd)
        for p in model.parameters():
            p.requires_grad_(False)
        ft_mods = T.trainable_modules(layers, types=types_c, layers=list(range(len(layers))))
        for m_ in ft_mods.values():
            m_.weight.requires_grad_(True)
        label = f"ceiling_{'+'.join(types_c)}@{lr_c:g}"
        hist = train_module([m_.weight for m_ in ft_mods.values()], lr_c, nullcontext(), label)
        c = evaluate(label)
        n_c = int(sum(m_.weight.numel() for m_ in ft_mods.values()))
        for m_ in ft_mods.values():
            m_.weight.requires_grad_(False)
        beats = {k: bool(c[k]["score"] > floor[k]["score"] and
                         c[k]["ci95"][0] > floor[k]["ci95"][1]) for k in TASKS}
        ladder.append({"types": list(types_c), "lr": lr_c, "n_params": n_c,
                       "scores": {k: float(c[k]["score"]) for k in TASKS},
                       "beats_floor": beats, "loss_first": hist[0], "loss_last": hist[-1]})
        for k in TASKS:
            at.add("repair_task_score", value=float(c[k]["score"]),
                   ci_lo=c[k]["ci95"][0], ci_hi=c[k]["ci95"][1], n=c[k]["n"],
                   corpus_id=tasks[k].dataset, incumbent_id="full_ft_ceiling",
                   model_id=WT.PG1_REPO, model_sha=base_sha,
                   flags={"role": "full-FT ceiling rung", "task": k, "n_params": n_c,
                          "steps": a.steps, "lr": lr_c, "types": list(types_c),
                          "beats_floor": beats[k], "loss_first": hist[0], "loss_last": hist[-1]})
        print(f"[M-G] ceiling rung {label}: loss {hist[0]:.4f} -> {hist[-1]:.4f}, "
              f"beats floor {beats}", flush=True)
        M.s3_sync_incremental(out)
        ceiling, n_ceiling, rung_used = c, n_c, label
        if any(beats.values()):
            break
    at.add("repair_ceiling_rungs", value=float(len(ladder)), model_id=WT.PG1_REPO,
           model_sha=base_sha,
           flags={"ladder": ladder, "rung_used": rung_used,
                  "definition": "how much full-fine-tune capacity the re-shell damage needs"})
    print(f"[M-G] ceiling ladder stopped at {rung_used} after {len(ladder)} rung(s)", flush=True)
    restore(reshell_sd)
    headroom = {k: float(ceiling[k]["score"] - floor[k]["score"]) for k in TASKS}
    print(f"[M-G] recoverable headroom {headroom} ({n_ceiling} ceiling params)", flush=True)
    # THE DENOMINATOR HAS TO EXIST BEFORE THE SWEEP SPENDS AN HOUR FILLING IT IN. Every reader's
    # recovered fraction is (score - floor) / headroom, so a ceiling that does not beat the floor
    # makes all of them 0/0 — which is how the first run died, deep inside the sweep, on a NaN.
    # The ceiling must beat the floor with SEPARATED intervals: a ceiling that merely edges ahead
    # inside the noise is not a scale to measure readers against either.
    repairable = {k: bool(headroom[k] > 0 and ceiling[k]["ci95"][0] > floor[k]["ci95"][1])
                  for k in TASKS}
    any_repairable = any(repairable.values())
    gates.record("g_damage_is_repairable_at_all", any_repairable,
                 headroom=headroom, repairable=repairable,
                 ceiling={k: ceiling[k]["score"] for k in TASKS},
                 ceiling_ci={k: ceiling[k]["ci95"] for k in TASKS},
                 floor={k: floor[k]["score"] for k in TASKS},
                 floor_ci={k: floor[k]["ci95"] for k in TASKS},
                 ceiling_params=n_ceiling, steps=a.steps, rung_used=rung_used,
                 ladder=ladder,
                 outcome=None if any_repairable else M.VOID_INSTRUMENT,
                 note=("the ceiling is what makes 'fraction recovered' a fraction OF something. "
                       "If a full q_proj fine-tune cannot beat doing nothing, no reader result "
                       "from this run is interpretable and the arm is void, not negative"))
    assert any_repairable, (
        f"the ceiling did not beat the floor on either task (headroom {headroom}) — a full "
        f"fine-tune of {n_ceiling} params at the top rung {rung_used} for {a.steps} steps "
        f"recovered "
        f"nothing, so there is no scale for the readers and every recovered fraction would be "
        f"0/0. VOID-instrument; the objective probe above says whether the objective or the "
        f"damage is what makes this so")
    # Only tasks with a real denominator are swept; a task whose ceiling failed is carried as void
    # rather than silently averaged in.
    scored_tasks = [k for k in TASKS if repairable[k]]
    print(f"[M-G] tasks with a usable denominator: {scored_tasks}", flush=True)

    # ---- the readers --------------------------------------------------------------------------
    results = {}
    for bid in banks_wanted:
        for r in ranks:
            reader = BankReader(layers, "cuda", basis[bid], r)
            hist = train_module(reader.parameters(), a.lr, reader, f"{bid}/r{r}")
            # BANK THE READER'S OWN FIT, not only the task score. Every reader in the first
            # decided run scored exactly 0.0000 on both tasks — an exact-match readout has no
            # resolution below "gets the answer right", so it cannot distinguish a reader that
            # nearly repairs from one that does nothing. The training loss can, and in that run it
            # was strongly ordered where the task scores were flat (M-D59).
            at.add("repair_reader_final_loss", value=float(hist[-1]), dose=float(r), bank_id=bid,
                   model_id=WT.PG1_REPO, model_sha=base_sha,
                   flags={"loss_first": float(hist[0]), "n_params": reader.n_params,
                          "steps": a.steps, "lr": a.lr,
                          "definition": ("the reader's own repair objective at the last step; a "
                                         "graded signal where the task readout is at its floor")})
            sc = evaluate(f"reader {bid}/r{r}", ctx=reader)
            cell = {"bank": bid, "rank": r, "n_params": reader.n_params,
                    "scores": {k: float(sc[k]["score"]) for k in TASKS},
                    "recovered_fraction": {k: ((sc[k]["score"] - floor[k]["score"])
                                               / headroom[k]) if repairable[k] else None
                                           for k in TASKS}}
            results[f"{bid}@r{r}"] = cell
            for k in TASKS:
                at.add("repair_task_score", value=float(sc[k]["score"]), dose=float(r),
                       bank_id=bid, ci_lo=sc[k]["ci95"][0], ci_hi=sc[k]["ci95"][1],
                       n=sc[k]["n"], corpus_id=tasks[k].dataset, model_id=WT.PG1_REPO,
                       model_sha=base_sha,
                       flags={**cell, "task": k, "role": "bank reader", "steps": a.steps,
                              "lr": a.lr, "floor": float(floor[k]["score"]),
                              "ceiling": float(ceiling[k]["score"]),
                              "denominator_exists": repairable[k]})
                # A task whose ceiling failed has no fraction to record. The SCORE above is still
                # banked — it is a measurement either way — but a fraction of nothing is not, and
                # writing it as NaN is what the atlas refuses.
                if repairable[k]:
                    at.add("repair_recovered_fraction", value=cell["recovered_fraction"][k],
                           dose=float(r), bank_id=bid, corpus_id=tasks[k].dataset,
                           model_id=WT.PG1_REPO, model_sha=base_sha,
                           flags={"task": k, "definition": "(score - floor) / (ceiling - floor)",
                                  "n_params": reader.n_params, "ceiling_params": n_ceiling})
            restore(reshell_sd)
            del reader
            torch.cuda.empty_cache()

    # ---- the refuter --------------------------------------------------------------------------
    at.check_readout_corpus(["repair_task_score", "repair_recovered_fraction"])
    summary = {}
    for k in TASKS:
        best = {b: max(results[f"{b}@r{r}"]["scores"][k] for r in ranks) for b in banks_wanted}
        twins = [best[b] for b in ("haar32_s0", "haar32_s1", "haar32_s2") if b in best]
        summary[k] = {"best_by_bank": best, "twin_band": [min(twins), max(twins)],
                      "floor": float(floor[k]["score"]),
                      "ceiling": float(ceiling[k]["score"]),
                      "core_beats_twins": bool(best["core32"] > max(twins)),
                      "core_beats_floor": bool(best["core32"] > floor[k]["score"])}
        # Both gates compare SCORES, so they still say something on a task whose ceiling failed —
        # but only alongside the fact that it did, or a reader "beating the floor" on a task where
        # nothing is repairable reads as a result.
        gates.record(f"g_core_readers_beat_twin_readers_{k}", summary[k]["core_beats_twins"],
                     core=best["core32"], twins=twins, floor=float(floor[k]["score"]),
                     ceiling=float(ceiling[k]["score"]), denominator_exists=repairable[k],
                     note="the pre-logged refuter: core readers <= twin readers")
        gates.record(f"g_readers_beat_the_floor_{k}", summary[k]["core_beats_floor"],
                     core=best["core32"], floor=float(floor[k]["score"]),
                     denominator_exists=repairable[k],
                     note="a reader that does not beat doing nothing repairs nothing")
        sur.add(arm=ARM, expected_ref=expected_ref, cell=k,
                statistic=f"repair_task_score/{k}/core32",
                s=M.surprise(best["core32"], float(np.mean(twins)), twins))

    info = at.to_parquet()
    sur.to_parquet()
    M.write_json(out, f"{ARM}_result.json",
                 {"arm": ARM, "queue_item": meta["queue_item"], "refuter": meta["refuter"],
                  "expected_ref": expected_ref, "provenance": M.provenance(), "utc": M.utc_now(),
                  "objective_probe": {"start_loss": start, "end_loss": end,
                                      "relative_drop": drop, "steps": a.probe_steps,
                                      "loss_is_sane": sane, "loss_descends": descends},
                  "repairable": repairable, "scored_tasks": scored_tasks,
                  "atlas": info, "shell_repo": WT.PG1_REPO, "shell_sha": base_sha,
                  "policy": POLICY, "reshell_install": info_reshell, "tasks": list(TASKS),
                  "ranks": ranks, "banks": banks_wanted, "steps": a.steps, "lr": a.lr,
                  "ft_lr": a.ft_lr, "n_train": a.n_train, "n_eval": a.n_eval,
                  "pg1_intact": {k: float(pg1_scores[k]["score"]) for k in TASKS},
                  "floor": {k: float(floor[k]["score"]) for k in TASKS},
                  "ceiling": {k: float(ceiling[k]["score"]) for k in TASKS},
                  "ceiling_params": n_ceiling, "headroom": headroom, "faculty_lost": lost,
                  "ceiling_ladder": ladder, "ceiling_rung_used": rung_used,
                  "cells": results, "summary": summary})
    print("G_OK " + json.dumps({"rows": info["n_rows"],
                                "core_beats_twins": {k: summary[k]["core_beats_twins"]
                                                     for k in TASKS}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
