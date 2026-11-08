"""E-Q1 — the DECISIVE trained low-rank adapter rank ladder (GraftLM trained-exploit prereg
research/2026-05-30_graftlm_trained_exploit_prereg.md).

QUESTION: a NO-RETRAIN rank-k SVD graft of the base->PG2-pt decoder delta closes ~58% of the KL gap
but installs 0% RefCOCO capability (the capability-readout cliff). Does a *trained* rank-k adapter on
the SAME decoder matrices escape that floor, and at what rank k*? H_topk (k*<=128) / H_threshold
(128<k*<=512) / H_floor (no k*<=512). Co-primary readout: RefCOCO acc@0.5 R_cap AND the <loc>-emission
policy flip (0 -> >50%).

DESIGN (apples-to-apples with the no-retrain baseline, all else frozen EXACTLY as E4):
  - Frozen bridge: build_pair("base-pt") -> teacher pg_after (PG2-pt full LM) + student pg_before
    (PG vision/connector/embed/lm_head + BASE gemma-2-2b decoder). Vision/connector/embed/lm_head/norms
    FROZEN; only a per-decoder-matrix rank-k adapter (W + A@B) trains.
  - Adapter init:
      * topk  : A,B from the rank-k SVD of dW = W_after - W_before -> at init the adapter EXACTLY
                reproduces the no-retrain rank-k SVD edit. So eval-at-init == the no-retrain baseline,
                computed IN THE SAME RUN (a free, perfectly-matched apples-to-apples contrast).
      * random: A ~ N(0, 1/in), B = 0 -> at init the student == FLOOR (base decoder). Standard LoRA.
  - Objective: KL( P_teacher || P_student ) on the GT-answer (suffix) tokens of a HELD-OUT multimodal
    install corpus (RefCOCO val + TextVQA train) -- distil PG2-pt's next-token distribution; NEVER
    trains on the eval-task labels (eval = RefCOCO testA / TextVQA val, disjoint by image content-hash).
  - Eval: the capability-readout generate-and-score path VERBATIM (_generate + battery_tasks scorers)
    -> trained R_cap + <loc>-emission. FLOOR (base) + CEILING (pg_after) computed once for R_cap.

CONTROLS (prereg §5): trainable-at-full positive control = --rank 1024 (must reach R_cap~1 BEFORE the
ladder is believed -- guards against under-training a false H_floor; the recipe is frozen on it).
--shuffle-teacher (label-permuted) and --depth-mask early/middle/late (E-Q3) are the other arms.

Reuses (not reinvented): build_pair (frozen bridge), _answer_batches (suffix->labels distil batches),
_eval_rows/_generate + battery_tasks scorers (eval), _decoder_keys (the 182 decoder matrices).
GPU job. hf_online_setup() first. No try/except around the science -- fails loud with a trace.

    # positive control / recipe calibration (prereg §9 step 2, BEFORE freeze):
    pg run infra.aws_pg.jobs.eq1_trained_adapter \
        --args="--rank=1024 --init=topk --seed=0 --steps=800 --eval-n=1975" \
        --type p4d.24xlarge,g6e.8xlarge --s3-prefix results/eq1_poscontrol --budget-min 240
    # a ladder cell (post-freeze):
    pg run infra.aws_pg.jobs.eq1_trained_adapter \
        --args="--rank=128 --init=topk --seed=0 --steps=<frozen> --eval-n=1975" \
        --type p4d.24xlarge,g6e.8xlarge --s3-prefix results/eq1_r128_topk_s0 --budget-min 240
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

from infra.aws_pg.jobs.e4_capability import _eval_rows, _generate, SEED
from infra.aws_pg.jobs.e4_compress_curve import _decoder_keys, N_LAYERS

# Install (distillation) corpus tasks -> (task_name, split). Held out from the eval by construction:
# RefCOCO val (eval = testA), TextVQA train (eval = validation). Detection prompts INCLUDED so the
# teacher emits <loc> targets the student can learn. (COCO captions deferrable: not needed for the
# decisive RefCOCO <loc> readout; add via --corpus once an AMI caption loader is wired.)
# Install corpora (held out from eval). 'refcoco' = RefCOCO-val only (distilling the teacher's <loc>
# detection — the gradient purely REINFORCES the RefCOCO-decisive readout). 'mix' adds TextVQA-train,
# but TextVQA answers carry NO <loc> → in the rank-1024 positive control the mix SUPPRESSED <loc>
# emission (0.61→0.48) and degraded R_cap (0.80→0.52); so 'refcoco' is the calibrated default for the
# RefCOCO-decisive experiment. (recipe-calibration finding 2026-05-30, pre-freeze.)
INSTALL_CORPORA = {"refcoco": (("refcoco", "val"),),
                   "mix": (("refcoco", "val"), ("textvqa", "train"))}
DEPTH_BANDS = {"all": range(N_LAYERS), "early": range(0, 7), "middle": range(10, 17), "late": range(19, 26)}
N_INSTALL_DEFAULT = 8000      # held-out install sequences (mixed across corpus tasks)
TRAIN_BATCH = 4               # KL forward materializes (b,T,V) logits x2 (teacher+student) -> keep small
EVAL_BATCH = 8
LR = 2e-5                     # transfer_a1 LoRA operating point (recipe FROZEN on the positive control)
WARMUP_RATIO = 0.03


# ===========================================================================
# Hand-rolled rank-k adapter (W + A@B). NOT peft: the top-k arm must init from the SVD of dW (peft's
# pissa is SVD-of-base-weight, the wrong object); a hand-rolled adapter makes the init exact and needs
# no peft on the AMI.
# ===========================================================================

def _adapter_module():
    import torch.nn as nn
    import torch.nn.functional as F

    class RankKAdapter(nn.Module):
        def __init__(self, base, A, B, scale):     # base: frozen nn.Linear; A:(out,k); B:(k,in)
            super().__init__()
            self.base = base
            self.A = nn.Parameter(A)
            self.B = nn.Parameter(B)
            self.scale = float(scale)               # LoRA-style alpha/r scale (effective-LR knob)

        def forward(self, x):
            return self.base(x) + self.scale * F.linear(F.linear(x, self.B), self.A)  # base + s·x@(A@B)^T

    return RankKAdapter


def _decoder_delta_svd(lm_after, lm_before, dkeys, device, q):
    """Truncated SVD of dW = W_after - W_before per decoder matrix (mirrors e4_capability
    _build_edit_family: torch.svd_lowrank q+256, niter=8). Seeded for reproducibility. Returns
    svd[key] = (U[:, :q], S[:q], V[:, :q]) on CPU, + the base state-dict slice."""
    import torch
    torch.manual_seed(SEED)                                   # reproducible randomized SVD
    after_sd = lm_after.state_dict()
    base_sd = lm_before.state_dict()
    svd = {}
    for k in dkeys:
        assert k in base_sd and k in after_sd, f"missing decoder key {k}"
        dW = (after_sd[k].float() - base_sd[k].float()).to(device)
        qq = min(q + 256, min(dW.shape))
        U, S, V = torch.svd_lowrank(dW, q=qq, niter=8)
        svd[k] = (U[:, :q].cpu(), S[:q].cpu(), V[:, :q].cpu())
        del dW, U, S, V
        torch.cuda.empty_cache()
    return svd


def _attach_adapters(lm, dkeys, svd, rank, init, device, gen, rand_scale):
    """Replace each target decoder Linear in `lm` with a rank-k adapter (W + scale·A@B).
    topk: A,B from the SVD of dW, scale=1 -> EXACTLY reproduces the no-retrain rank-k edit at init
      (= SVD of the base->pt CHANGE subspace; Si-endorsed task-specific directions).
    random: PEFT convention -> zero the OUT-side (A), random the IN-side (B), scale=rand_scale (alpha/r).
      Zeroing the out-side lets A grow FREELY toward the target out-directions; zeroing the in-side
      instead (the earlier bug) locks dW to a random out-subspace -> the rank-1024 plateau (eq1_pc_rand).
    Both start at student==no-retrain-edit (topk) / FLOOR (random, A=0 -> dW=0)."""
    import torch
    RankKAdapter = _adapter_module()
    n_params = 0
    for key in dkeys:
        modpath = key[: -len(".weight")]                      # "layers.5.self_attn.q_proj"
        lin = lm.get_submodule(modpath)
        out_d, in_d = lin.weight.shape
        kk = min(rank, min(out_d, in_d))
        if init == "topk":
            U, S, V = svd[key]
            sq = S[:kk].sqrt()
            A = U[:, :kk] * sq                                # (out, kk)
            Bm = sq[:, None] * V[:, :kk].T                    # (kk, in)
            scale = 1.0                                       # scale·A@B = dW_k exactly
        elif init == "random":
            A = torch.zeros(out_d, kk)                        # OUT-side zeroed -> grows freely (PEFT)
            Bm = torch.randn(kk, in_d, generator=gen) * (in_d ** -0.5)   # IN-side random (Kaiming-ish)
            scale = rand_scale                                # LoRA alpha/r effective-LR knob
        else:
            raise ValueError(f"bad init {init}")
        adapter = RankKAdapter(lin, A.to(device).to(torch.bfloat16), Bm.to(device).to(torch.bfloat16), scale)
        parent_path, child = modpath.rsplit(".", 1)
        setattr(lm.get_submodule(parent_path), child, adapter)
        n_params += A.numel() + Bm.numel()
    return n_params


# ===========================================================================
# Install corpus (held out from eval) + the image content-hash split audit (prereg §4 — guards the
# systemic W3/W4/W5 split-bug).
# ===========================================================================

def _img_hash(img):
    """Heuristic content hash: 32x32 grayscale bytes -> sha1. Catches the same source image appearing
    in both install and eval even across resizes/splits."""
    return hashlib.sha1(img.convert("L").resize((32, 32)).tobytes()).hexdigest()


def _install_items(B, n, seed, exclude_hashes, corpus):
    """Held-out (task, row) install items + their image hashes, EXCLUDING any item whose image
    content-hash is in `exclude_hashes` (the eval images) — enforces install<->eval disjointness BY
    CONSTRUCTION (the systemic W3/W4/W5 split-bug guard; lmms-lab/RefCOCO testA and val share images).
    `corpus` = tuple of (task_name, split). Logs how many were dropped (no silent truncation)."""
    from datasets import load_dataset
    items, hashes, dropped = [], set(), 0
    per_task = max(1, n // len(corpus))
    for tname, split in corpus:
        task = B.TASKS[tname]
        ds = load_dataset(task.dataset, split=split).shuffle(seed=seed)
        ds = ds.select(range(min(per_task, len(ds))))
        for i in range(len(ds)):
            r = ds[i]
            h = _img_hash(task.image_fn(r))
            if h in exclude_hashes:                            # image also in the eval set -> drop (leak)
                dropped += 1
                continue
            items.append({"t": tname, "row": r})
            hashes.add(h)
    print(f"[eq1] install corpus: kept {len(items)}, dropped {dropped} eval-overlapping items "
          f"(image content-hash disjointness)", flush=True)
    return items, hashes


def _split_audit(B, install_hashes, eval_rows, out):
    """HALT if any eval image content-hash appears in the install corpus. `eval_rows` is the preloaded
    {task_name: rows} dict (loaded once, before build_pair)."""
    audit = {"install_n_hashes": len(install_hashes), "tasks": {}}
    overlap_total = 0
    for tname, rows in eval_rows.items():
        task = B.TASKS[tname]
        eh = {_img_hash(task.image_fn(r)) for r in rows}
        ov = eh & install_hashes
        audit["tasks"][tname] = {"eval_n_hashes": len(eh), "overlap": len(ov)}
        overlap_total += len(ov)
    audit["overlap_total"] = overlap_total
    (out / "split_audit.json").write_text(json.dumps(audit, indent=2))
    assert overlap_total == 0, f"SPLIT LEAK: {overlap_total} eval image(s) in the install corpus — HALT. {audit}"
    print(f"[eq1] split_audit OK: 0 overlap ({len(install_hashes)} install hashes)", flush=True)


# ===========================================================================
# Eval (capability-readout path verbatim) + R_cap.
# ===========================================================================

def _refcoco_emission(preds):
    """Fraction of generations that parse >=4 <loc> tokens (the policy-flip indicator)."""
    from infra.pg import battery_tasks as B
    return float(sum(B._decode_loc_box(p) is not None for p in preds) / max(1, len(preds)))


def _score(model, proc, task, rows, device):
    import numpy as np
    preds = _generate(model, proc, task, rows, device, EVAL_BATCH)
    m = float(np.mean([task.score_fn(p, r) for r, p in zip(rows, preds)]))
    emit = _refcoco_emission(preds) if task.name == "refcoco" else None
    return m, emit, preds[:12]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--init", choices=["topk", "random"], default="topk")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--depth-mask", choices=list(DEPTH_BANDS), default="all")
    ap.add_argument("--corpus", choices=list(INSTALL_CORPORA), default="refcoco",
                    help="install corpus: 'refcoco' (val only — reinforces <loc>) or 'mix' (+TextVQA)")
    ap.add_argument("--corpus-n", type=int, default=N_INSTALL_DEFAULT)
    ap.add_argument("--eval-n", type=int, default=1975)
    ap.add_argument("--eval-tasks", default="refcoco")          # decisive task; "refcoco,textvqa" for both
    ap.add_argument("--train-batch", type=int, default=TRAIN_BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--rand-scale", type=float, default=2.0,
                    help="random-init adapter scale (LoRA alpha/r effective-LR knob); topk is fixed at 1.0")
    ap.add_argument("--shuffle-teacher", action="store_true", help="label-permute the teacher targets (null)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from infra.pg import battery_tasks as B
    from infra.pg import diff_harvest as H

    # hf_online_setup() MUST run before importing transformers/datasets: importing transformers pulls in
    # huggingface_hub, which reads HF_HUB_OFFLINE (run_job.sh sets it =1) into a MODULE CONSTANT at
    # import time — so flipping os.environ afterward is too late and load_dataset stays offline
    # (LocalEntryNotFoundError). E-Q2 works precisely because it calls this before any hub import.
    H.hf_online_setup()
    from torch.utils.data import DataLoader
    from transformers import get_scheduler
    device = "cuda"
    assert torch.cuda.is_available(), "E-Q1 needs a GPU"
    if a.smoke:
        a.corpus_n, a.steps, a.eval_n = 64, 20, 32
    torch.manual_seed(a.seed)
    eval_tasks = tuple(t.strip() for t in a.eval_tasks.split(","))
    print(f"[eq1] rank={a.rank} init={a.init} seed={a.seed} steps={a.steps} mask={a.depth_mask} "
          f"corpus_n={a.corpus_n} eval_n={a.eval_n} shuffle_teacher={a.shuffle_teacher} smoke={a.smoke}",
          flush=True)

    # Preload ALL datasets BEFORE build_pair. Model weights load from the S3-synced local path, so the
    # dataset load is the FIRST real HF-hub access; doing it up front matches the proven-working E-Q2
    # order and fail-fasts on data + the split audit before the expensive model build.
    corpus = INSTALL_CORPORA[a.corpus]
    eval_rows = {tname: _eval_rows(B.TASKS[tname], a.eval_n, SEED) for tname in eval_tasks}
    eval_hashes = {_img_hash(B.TASKS[t].image_fn(r)) for t in eval_tasks for r in eval_rows[t]}
    install_items, install_hashes = _install_items(B, a.corpus_n, SEED + a.seed, eval_hashes, corpus)
    _split_audit(B, install_hashes, eval_rows, out)   # verifies the filter -> expect 0 overlap
    print(f"[eq1] data preloaded: eval={{{', '.join(f'{t}:{len(eval_rows[t])}' for t in eval_tasks)}}} "
          f"install={len(install_items)}", flush=True)

    pg_after, pg_before, proc, paths, _img, gate = H.build_pair("base-pt", device)
    print(f"[eq1] swap_gate={json.dumps(gate, default=str)}", flush=True)
    lm = pg_before.get_submodule(paths["language_model"])     # student decoder host
    lm_after = pg_after.get_submodule(paths["language_model"])

    # depth-masked decoder keys
    band = set(DEPTH_BANDS[a.depth_mask])
    dkeys = [k for k in _decoder_keys() if int(k.split(".")[1]) in band]
    print(f"[eq1] depth-mask {a.depth_mask}: {len(dkeys)}/{len(_decoder_keys())} decoder matrices "
          f"(layers {sorted(band)[:3]}..{sorted(band)[-1]})", flush=True)

    # FLOOR + CEILING (computed once for R_cap), on the decisive eval task(s)
    anchors = {}
    for tname in eval_tasks:
        task = B.TASKS[tname]
        mf, ef, _ = _score(pg_before, proc, task, eval_rows[tname], device)   # FLOOR = base decoder, no adapter
        mc, ec, _ = _score(pg_after, proc, task, eval_rows[tname], device)    # CEILING = PG2-pt
        anchors[tname] = {"floor": mf, "ceiling": mc, "floor_emit": ef, "ceiling_emit": ec}
        print(f"[eq1] {tname} FLOOR={mf:.4f}(emit={ef}) CEILING={mc:.4f}(emit={ec})", flush=True)

    # attach the rank-k adapter, freeze everything else
    for p in pg_before.parameters():
        p.requires_grad_(False)
    svd = _decoder_delta_svd(lm_after, lm, dkeys, device, a.rank) if a.init == "topk" else None
    gen = torch.Generator(device="cpu").manual_seed(SEED + a.seed)
    n_adapter = _attach_adapters(lm, dkeys, svd, a.rank, a.init, device, gen, a.rand_scale)
    trainable = [p for p in pg_before.parameters() if p.requires_grad]
    n_train_params = int(sum(p.numel() for p in trainable))
    assert n_train_params == n_adapter, f"trainable {n_train_params} != adapter {n_adapter} (freeze leak)"
    print(f"[eq1] adapter attached: {n_adapter:,} trainable params (rank {a.rank} x {len(dkeys)} matrices)",
          flush=True)

    # eval-at-init: for topk this IS the no-retrain rank-k SVD edit (in-run apples-to-apples baseline);
    # for random it is FLOOR.
    init_metrics = {}
    for tname in eval_tasks:
        task = B.TASKS[tname]
        m0, e0, s0 = _score(pg_before, proc, task, eval_rows[tname], device)
        d = anchors[tname]["ceiling"] - anchors[tname]["floor"]
        init_metrics[tname] = {"metric": m0, "emit": e0,
                               "R_cap": ((m0 - anchors[tname]["floor"]) / d if d > 0 else float("nan")),
                               "label": "no_retrain_rank_k" if a.init == "topk" else "floor",
                               "samples": s0}
        print(f"[eq1] {tname} INIT ({init_metrics[tname]['label']}) metric={m0:.4f} emit={e0} "
              f"R_cap={init_metrics[tname]['R_cap']:.3f}", flush=True)

    # ---- distillation training (KL(teacher || student) on suffix/answer tokens) ----
    # install_items + split-audit already loaded/verified pre-build_pair (above).
    if a.shuffle_teacher:                                       # null: permute targets across items
        rng = np.random.default_rng(SEED + a.seed)
        perm = rng.permutation(len(install_items))
        rows_only = [install_items[i]["row"] for i in perm]
        for it, r in zip(install_items, rows_only):
            it["row_target"] = r                               # target from a DIFFERENT image
    print(f"[eq1] install corpus={a.corpus}: {len(install_items)} items across {corpus}", flush=True)

    def collate(items):
        imgs = [B.TASKS[it["t"]].image_fn(it["row"]).convert("RGB") for it in items]
        prompts = [B.TASKS[it["t"]].prompt_fn(it["row"]) for it in items]
        tgt_rows = [it.get("row_target", it["row"]) for it in items]
        suffixes = [B.TASKS[it["t"]].target_fn(tr) for it, tr in zip(items, tgt_rows)]
        enc = proc(text=prompts, images=imgs, suffix=suffixes, return_tensors="pt", padding="longest")
        enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
        return enc

    loader = DataLoader(install_items, batch_size=a.train_batch, shuffle=True, collate_fn=collate,
                        num_workers=2, drop_last=True)
    opt = torch.optim.AdamW(trainable, lr=a.lr)
    sched = get_scheduler("cosine", optimizer=opt, num_warmup_steps=int(WARMUP_RATIO * a.steps),
                          num_training_steps=a.steps)
    pg_before.train()
    step, losses, done = 0, [], False
    while not done:
        for enc in loader:
            enc = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in enc.items()}
            ii, am, pv, labels = enc["input_ids"], enc["attention_mask"], enc["pixel_values"], enc["labels"]
            with torch.no_grad():
                tl = pg_after(input_ids=ii, attention_mask=am, pixel_values=pv).logits[:, :-1]
            sl = pg_before(input_ids=ii, attention_mask=am, pixel_values=pv).logits[:, :-1]
            mask = (labels[:, 1:] != -100)
            tp = torch.log_softmax(tl[mask].float(), dim=-1)    # select answer positions in bf16 FIRST,
            sp = torch.log_softmax(sl[mask].float(), dim=-1)    # then float — avoids a full (b,T,V) fp32 copy
            loss = (tp.exp() * (tp - sp)).sum(-1).mean()       # KL(teacher || student) on answer tokens
            assert torch.isfinite(loss), f"non-finite loss step {step}: {loss.item()} (diverged — reported)"
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(float(loss.item()))
            step += 1
            if step % 25 == 0 or step == 1:
                print(f"[eq1] step {step}/{a.steps} kl_loss={losses[-1]:.4f}", flush=True)
            if step >= a.steps:
                done = True
                break

    # ---- eval-after-train ----
    pg_before.eval()
    trained = {}
    for tname in eval_tasks:
        task = B.TASKS[tname]
        m1, e1, s1 = _score(pg_before, proc, task, eval_rows[tname], device)
        d = anchors[tname]["ceiling"] - anchors[tname]["floor"]
        trained[tname] = {"metric": m1, "emit": e1,
                          "R_cap": ((m1 - anchors[tname]["floor"]) / d if d > 0 else float("nan")),
                          "delta_vs_init_R_cap": None, "samples": s1}
        trained[tname]["delta_vs_init_R_cap"] = (trained[tname]["R_cap"] - init_metrics[tname]["R_cap"])
        print(f"[eq1] {tname} TRAINED metric={m1:.4f} emit={e1} R_cap={trained[tname]['R_cap']:.3f} "
              f"(d_vs_init={trained[tname]['delta_vs_init_R_cap']:+.3f})", flush=True)

    results = {"experiment": "eq1_trained_adapter", "status": "confirmatory",
               "config": {"rank": a.rank, "init": a.init, "seed": a.seed, "steps": a.steps,
                          "depth_mask": a.depth_mask, "corpus": a.corpus,
                          "corpus_n": len(install_items), "eval_n": a.eval_n,
                          "eval_tasks": list(eval_tasks), "lr": a.lr, "rand_scale": a.rand_scale,
                          "train_batch": a.train_batch,
                          "shuffle_teacher": a.shuffle_teacher, "n_adapter_params": n_adapter},
               "provenance": {"axis": "base-pt", "teacher": "google/paligemma2-3b-pt-224",
                              "student_decoder": "google/gemma-2-2b", "swap_gate": gate,
                              "frozen_except": "rank-k decoder adapter (norms/embed/lm_head/connector frozen)",
                              "no_retrain_compare": "results/e4_capreadout_sweep (+ topk init-eval here)",
                              "install_corpus": list(corpus)},
               "anchors": anchors, "init_eval": init_metrics, "trained_eval": trained,
               "train_loss": {"start": (losses[0] if losses else None),
                              "end_mean10": (float(np.mean(losses[-10:])) if losses else None)}}
    (out / "eq1_trained_adapter.json").write_text(json.dumps(results, indent=2))
    print("EQ1_OK", json.dumps({t: {"init_R_cap": round(init_metrics[t]["R_cap"], 3),
                                     "trained_R_cap": round(trained[t]["R_cap"], 3),
                                     "trained_emit": trained[t]["emit"]} for t in eval_tasks}), flush=True)


if __name__ == "__main__":
    main()
