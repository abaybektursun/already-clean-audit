"""MENDELEEV recovery core (N8 helpers) — the pure, weight-free-testable half of
`infra/aws_pg/jobs/matrix_recovery.py` (contract `infra/pg/MATRIX_CONTRACT.md` §7; spec §1.2
Recovery/GB/SUB/IM, §6.6, §6.13).

Everything here runs on tiny CPU tensors in the dry-run: dose schedule, recipe binding (the
recipes file is the ONLY hyperparameter source), adapter + optimizer construction (LoRA through
`relearn` with alpha = r; full-FT through `Fp32AdamW` = exact fp32 master weights + moments, a
plain per-parameter loop, optional CPU offload), the FAIRSCRATCH decoder re-init (the
`diff_harvest.build_random_after` rule, generic), the 50/50 REH step mixer, the general
answer-token CE trainer (letter items reduce EXACTLY to `relearn.train_adapter_qa`'s loss), the
dose-0 identity check (D18), the GB helpers (sweep layers, no-cache grafted greedy decode,
O_restored), and the SUB statistics (top-k left-singular subspace of the install delta, Haar
null with the SAME numpy draws as `matrix_core.subspace_overlap`, unrelated-adapter null).

Verdicts (`n_half`, `recovery_verdict`, `graft_back_verdict`, contention) are the lead's, computed
offline from the banked inputs — nothing here decides. Heavy imports inside functions; fail loud.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

RECIPE_KEYS = {"r512": "LoRA-r512", "r64": "LoRA-r64", "full": "full-FT"}
ARMS = ("rec", "scratch", "healthy", "reh", "permuted")
IM_TASKS = ("mmlu_letter", "mmlu_cloze", "vqav2", "refcoco", "arith", "triviaqa_cloze")
IM_VLM_TASKS = ("vqav2", "refcoco")
VISION_TASKS = ("refcoco", "vqav2", "textvqa")
O_FLOOR = 0.0                    # task-native floor of the vision O tasks (e4 preflight: FLOOR = 0.00)
GB_TRIGGER_O_LOST = 0.3          # §1.2 GB: first dose where O_lost ≥ 0.3
IDENTITY_MIN_AGREE = 0.99        # D18: dose-0 per-item agreement with the behaviour cell
PERMUTED_DOSE = 4000             # §6.13: permuted-label control at the 4k dose only
IM_DOSE = 4000
SMOKE_DOSE, SMOKE_STEPS = 32, 4
N_NULL_CUTS = 5                  # D12: ≥ 5 mismatched-item cuts


class IdentityMismatch(AssertionError):
    """Dose-0 vectors disagree with the behaviour cell's — the recovery job is NOT measuring the
    same model/readout as the behaviour job (D18). Never caught in a job."""


# ---------------------------------------------------------------- recipes (the only source)
def load_recipes(path) -> tuple:
    """Return (recipes_dict, sha256 of the file text). Hash goes into every manifest."""
    text = open(path, encoding="utf-8").read()
    return json.loads(text), hashlib.sha256(text.encode()).hexdigest()


def recipe(recipes: dict, key: str) -> dict:
    """Bind one recipe (`r512` | `r64` | `full`) from the frozen file. LoRA: alpha == rank asserted
    (confound 13). Returns the flat dict the trainers consume."""
    assert key in RECIPE_KEYS, f"recipe {key!r} not in {sorted(RECIPE_KEYS)}"
    assert recipes["optimizer"] == "AdamW", recipes["optimizer"]
    r = recipes["recipes"][RECIPE_KEYS[key]]
    out = {"key": key, "name": RECIPE_KEYS[key], "lr": float(r["lr"]),
           "epochs": int(recipes["epochs_over_dose"]), "batch": int(recipes["batch_items"]),
           "warmup": float(recipes["warmup_frac"]), "betas": tuple(float(b) for b in recipes["betas"]),
           "weight_decay": float(recipes["weight_decay"]), "dtype": recipes["dtype"],
           "rank": None, "alpha": None, "min_gpu_gb": r.get("min_gpu_gb")}
    if key != "full":
        out["rank"], out["alpha"] = int(r["rank"]), float(r["alpha"])
        assert out["alpha"] == out["rank"], f"alpha must equal r (frozen): {r}"
    return out


def dose_schedule(doses, arm: str, smoke: bool) -> list:
    """The fixed ladder (§6.13: nothing promoted). `permuted` runs at 4k only; smoke = one tiny dose."""
    assert arm in ARMS, f"arm {arm!r} not in {ARMS}"
    d = sorted(int(x) for x in doses)
    assert d and all(x > 0 for x in d) and len(set(d)) == len(d), d
    if smoke:
        return [SMOKE_DOSE]
    if arm == "permuted":
        assert d == [PERMUTED_DOSE], f"permuted-label control is defined at {PERMUTED_DOSE} only, got {d}"
    return d


def n_steps(dose: int, batch: int, epochs: int) -> int:
    return int(epochs * math.ceil(dose / batch))


def lr_at(step: int, total: int, warmup_frac: float, lr: float) -> float:
    """EXACTLY relearn._cosine_warmup's factor (LambdaLR semantics: step 0 → factor 0 when warmup > 0)."""
    n_warm = int(warmup_frac * total)
    if step < n_warm:
        return lr * step / max(1, n_warm)
    p = (step - n_warm) / max(1, total - n_warm)
    return lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


# ---------------------------------------------------------------- optimizers
class Fp32AdamW:
    """Exact-fp32 AdamW for bf16 model parameters: fp32 master weights + fp32 first/second moments,
    decoupled weight decay, bias correction — the same arithmetic as torch.optim.AdamW (asserted in
    the dry-run) — as a plain per-parameter loop so the state can live on the CPU (`offload=True`:
    grads stream to pinned host memory, the update runs there, the bf16 copy streams back)."""

    def __init__(self, params, lr: float, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0, offload: bool = False):
        import torch
        self.params = [p for p in params]
        assert self.params, "no trainable params"
        self.lr, self.betas, self.eps, self.wd, self.offload = lr, tuple(betas), eps, weight_decay, offload
        dev = torch.device("cpu") if offload else self.params[0].device
        self.master = [p.detach().to(dev, torch.float32).clone() for p in self.params]
        self.m = [torch.zeros_like(w) for w in self.master]
        self.v = [torch.zeros_like(w) for w in self.master]
        if offload and torch.cuda.is_available():
            self.master = [w.pin_memory() for w in self.master]
            self.m = [w.pin_memory() for w in self.m]
            self.v = [w.pin_memory() for w in self.v]
        self.t = 0

    def state_gb(self) -> dict:
        n = sum(w.numel() for w in self.master)
        return {"params": int(n), "fp32_state_gb": 12 * n / 1e9,
                "where": "host" if self.offload else "device"}

    def step(self, lr_t: float):
        import torch
        self.t += 1
        b1, b2 = self.betas
        bc1, bc2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        with torch.no_grad():
            for p, w, m, v in zip(self.params, self.master, self.m, self.v):
                assert p.grad is not None, "param without grad in Fp32AdamW.step"
                g = p.grad.to(w.device, torch.float32)
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                denom = (v / bc2).sqrt_().add_(self.eps)
                if self.wd:
                    w.mul_(1 - lr_t * self.wd)
                w.addcdiv_(m, denom, value=-lr_t / bc1)
                p.data.copy_(w.to(p.dtype))
                p.grad = None

    def zero_grad(self):
        for p in self.params:
            p.grad = None


class TorchAdamW:
    """torch.optim.AdamW behind the same tiny interface (LoRA A/B are fp32 masters already)."""

    def __init__(self, params, lr: float, betas=(0.9, 0.999), weight_decay: float = 0.0):
        import torch
        self.opt = torch.optim.AdamW([p for p in params], lr=lr, betas=tuple(betas), weight_decay=weight_decay)

    def step(self, lr_t: float):
        for g in self.opt.param_groups:
            g["lr"] = lr_t
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)

    def zero_grad(self):
        self.opt.zero_grad(set_to_none=True)


def make_optimizer(trainable, rcp: dict, offload: bool):
    if rcp["key"] == "full":
        return Fp32AdamW(trainable, rcp["lr"], rcp["betas"], weight_decay=rcp["weight_decay"], offload=offload)
    return TorchAdamW(trainable, rcp["lr"], rcp["betas"], rcp["weight_decay"])


# ---------------------------------------------------------------- adapters (relearn, alpha = r)
def attach_lora(lm, rank: int, seed: int) -> dict:
    from infra.pg import relearn as R
    return R.attach_adapters(lm, rank=rank, seed=seed, alpha=float(rank))


def _is_adapter(m) -> bool:
    return type(m).__name__ == "RankRAdapter"


def detach_lora(lm) -> int:
    """Put every wrapped base Linear back (a fresh adapter per dose needs a pristine model)."""
    n = 0
    for name, m in list(lm.named_modules()):
        if _is_adapter(m):
            parent_name, _, leaf = name.rpartition(".")
            parent = lm.get_submodule(parent_name) if parent_name else lm
            setattr(parent, leaf, m.base)
            n += 1
    assert n > 0, "no adapters to detach"
    return n


def lora_state(lm) -> dict:
    """{'<module>.A': tensor, '<module>.B': tensor} on CPU — the per-dose artifact."""
    sd = {k: v.detach().cpu().clone() for k, v in lm.state_dict().items()
          if k.endswith(".A") or k.endswith(".B")}
    assert sd, "no adapter tensors in state_dict"
    return sd


def load_lora_state(lm, sd: dict) -> int:
    cur = {k for k in lm.state_dict() if k.endswith(".A") or k.endswith(".B")}
    assert set(sd) == cur, f"adapter key mismatch: {len(set(sd) ^ cur)} differ"
    missing, unexpected = lm.load_state_dict(sd, strict=False)
    assert not unexpected and set(missing) == set(lm.state_dict()) - cur
    return len(sd)


def lora_deltas(lm) -> dict:
    """Merged adapter delta per wrapped module: ΔW = (alpha/r)·A·B, (out × in), fp32 CPU."""
    import torch
    out = {}
    with torch.no_grad():
        for name, m in lm.named_modules():
            if _is_adapter(m):
                out[name] = (m.scale * (m.A.float() @ m.B.float())).cpu()
    assert out, "no adapters attached"
    return out


def lora_param_count(lm, rank: int) -> int:
    """Σ min(r, out, in)·(out + in) over `relearn.adapter_target_names` — no attach needed."""
    from infra.pg import relearn as R
    n = 0
    for name in R.adapter_target_names(lm):
        o, i = lm.get_submodule(name).weight.shape
        n += min(rank, o, i) * (o + i)
    return int(n)


# ---------------------------------------------------------------- full-FT scope + snapshots
def full_ft_params(lm) -> list:
    """(name, param) for every LM parameter except the embedding/lm_head interface (tied in Gemma):
    decoder layers + every norm — the LoRA scope plus the 1-D tensors."""
    ps = [(n, p) for n, p in lm.named_parameters() if "embed_tokens" not in n and "lm_head" not in n]
    assert ps, "no full-FT params"
    return ps


def snapshot(params) -> list:
    return [p.detach().cpu().clone() for p in params]


def restore(params, snap) -> None:
    import torch
    assert len(params) == len(snap)
    with torch.no_grad():
        for p, s in zip(params, snap):
            p.copy_(s.to(p.device, p.dtype))


def full_ft_deltas(lm, snap_by_name: dict) -> dict:
    """ΔW_rec for full-FT: trained − pre-dose over the 2-D adapter-scope Linears (fp32 CPU)."""
    from infra.pg import relearn as R
    out = {}
    for name in R.adapter_target_names(lm):
        w = lm.get_submodule(name).weight
        out[name] = (w.detach().float().cpu() - snap_by_name[name + ".weight"].float())
    return out


# ---------------------------------------------------------------- fair-scratch re-init
def reinit_decoder(lm, source_sd: dict, seed: int) -> dict:
    """The FAIRSCRATCH decoder: EXACTLY `diff_harvest.build_random_after`'s rule, generic over the
    LM module — load `source_sd` (the healthy BEFORE decoder) strictly, then every tensor with
    ndim ≥ 2 whose key carries no embed_tokens/lm_head becomes N(0, std of the source tensor)
    (one cpu Generator, state-dict order); 1-D tensors and the embedding interface keep the
    source values. Returns {n_random, n_kept}."""
    import torch
    keys = list(lm.state_dict().keys())
    assert keys == list(source_sd.keys()), "source state dict keys/order differ from the target LM"
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    new_sd, n_rand = {}, 0
    for k, v in source_sd.items():
        if "embed_tokens" in k or "lm_head" in k or v.ndim < 2:
            new_sd[k] = v
            continue
        vf = v.detach().cpu().float()
        std = vf.std().clamp_min(1e-8)
        new_sd[k] = (torch.randn(vf.shape, generator=g) * std).to(v.dtype)
        n_rand += 1
    dev = next(lm.parameters()).device
    miss, unexp = lm.load_state_dict({k: t.to(dev) for k, t in new_sd.items()}, strict=True)
    assert not miss and not unexp
    return {"n_random": n_rand, "n_kept": len(new_sd) - n_rand, "seed": int(seed)}


# ---------------------------------------------------------------- answer-token CE trainer
def encode_text(tok, examples: list, device):
    """Text examples {prompt, target_text} → (inputs, labels). Left-padded, BOS via
    add_special_tokens on the prompt, target tokens appended, content-relative position_ids —
    the `relearn.train_adapter_qa` conventions, so a one-token target gives its exact loss."""
    import torch
    seqs, tgts = [], []
    for e in examples:
        p = tok.encode(e["prompt"], add_special_tokens=True)
        t = tok.encode(e["target_text"], add_special_tokens=False)
        assert t, f"empty target {e['target_text']!r}"
        seqs.append(p + t)
        tgts.append(len(t))
    L = max(len(s) for s in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.full((len(seqs), L), int(pad), dtype=torch.long)
    am = torch.zeros((len(seqs), L), dtype=torch.long)
    lab = torch.full((len(seqs), L), -100, dtype=torch.long)
    for i, (s, nt) in enumerate(zip(seqs, tgts)):
        ids[i, L - len(s):] = torch.tensor(s)
        am[i, L - len(s):] = 1
        lab[i, L - nt:] = torch.tensor(s[-nt:])
    pos = (am.cumsum(-1) - 1).masked_fill(am == 0, 1)
    inputs = {"input_ids": ids.to(device), "attention_mask": am.to(device), "position_ids": pos.to(device)}
    return inputs, lab.to(device)


def encode_vlm(proc, examples: list, device):
    """Image examples {prompt, target_text, image} through the processor's suffix path (labels
    masked to the suffix — the battery/transfer training format verbatim). Left padding."""
    import torch
    tok = proc.tokenizer
    orig = tok.padding_side
    tok.padding_side = "left"
    batch = proc(text=[e["prompt"] for e in examples], images=[e["image"].convert("RGB") for e in examples],
                 suffix=[e["target_text"] for e in examples], return_tensors="pt", padding="longest")
    tok.padding_side = orig
    batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
    batch = {k: v.to(device) for k, v in batch.items()}
    labels = batch["labels"]
    assert (labels != -100).any(), "processor produced no supervised suffix labels"
    return batch, labels


def answer_ce(logits, labels):
    """CE over the labelled (answer) tokens, aligned from the right: logits (B, L, V) are the LAST L
    positions; position s is predicted by logits at s−1. Returns (sum_ce, n_tokens) so a micro-batch
    can be scaled to the step's token-mean."""
    import torch
    B, S = labels.shape
    L = logits.shape[1]
    k = min(L, S)
    pred = logits[:, -k:-1].float()
    tgt = labels[:, S - k + 1:]
    assert (labels[:, :S - k + 1] == -100).all(), "labelled tokens outside the kept logit window"
    ce = torch.nn.functional.cross_entropy(pred.reshape(-1, pred.shape[-1]), tgt.reshape(-1),
                                           ignore_index=-100, reduction="sum")
    return ce, int((tgt != -100).sum().item())


def step_plan(pools: list, per_pool: int, epochs: int, seed: int) -> list:
    """Deterministic step list over one or more example pools (REH: [mmlu, vqa] → 8 + 8 per step).
    Same RNG convention as relearn (one torch.Generator, randperm per epoch per pool). Every pool
    must have the same size (dose-matched). Returns [[(pool_idx, [indices]), ...], ...]."""
    import torch
    sizes = {len(p) for p in pools}
    assert len(sizes) == 1, f"pools differ in size: {[len(p) for p in pools]}"
    n = sizes.pop()
    g = torch.Generator().manual_seed(int(seed))
    steps = []
    for _ in range(epochs):
        orders = [torch.randperm(n, generator=g).tolist() for _ in pools]
        for s in range(0, n, per_pool):
            steps.append([(pi, orders[pi][s:s + per_pool]) for pi in range(len(pools))])
    return steps


def train_answer_ce(model, pools: list, encoders: list, opt, *, rcp: dict, per_pool: int, seed: int,
                    micro_batch: int, loss_log: list, max_steps=None, print_every: int = 10) -> dict:
    """The general trainer: per step, every pool contributes `per_pool` examples (a step = one
    optimizer update over ALL of them, token-mean CE via gradient accumulation), cosine-with-warmup
    LR from `lr_at`, `opt.step(lr_t)`. For a single text pool of letter items this is
    `relearn.train_adapter_qa` step-for-step (batching, RNG, schedule, loss) with a different
    optimizer object; it also carries REH (two pools), IM (suffix targets), full-FT."""
    import torch
    plan = step_plan(pools, per_pool, rcp["epochs"], seed)
    total = len(plan) if max_steps is None else min(len(plan), int(max_steps))
    model.train()
    step = 0
    tokens_seen = 0
    for groups in plan[:total]:
        encoded = []
        for pi, idx in groups:
            ex = [pools[pi][i] for i in idx]
            for ms in range(0, len(ex), micro_batch):
                encoded.append((encoders[pi], ex[ms:ms + micro_batch]))
        n_tok_step = 0
        micro = []
        for enc, ex in encoded:
            inputs, labels = enc(ex)
            micro.append((inputs, labels))
            n_tok_step += int((labels != -100).sum().item())
        assert n_tok_step > 0
        step_loss = 0.0
        for inputs, labels in micro:
            kw = dict(inputs)
            if "labels" not in kw:
                kw["logits_to_keep"] = int((labels != -100).sum(1).max().item()) + 1
            out = model(**kw)
            ce_sum, _ = answer_ce(out.logits, labels)
            loss = ce_sum / n_tok_step
            assert torch.isfinite(loss), f"non-finite loss at step {step} — HALT"
            loss.backward()
            step_loss += float(loss.item())
            tokens_seen += int(inputs["attention_mask"].sum().item())
        opt.step(lr_at(step, len(plan), rcp["warmup"], rcp["lr"]))
        loss_log.append(step_loss)
        step += 1
        if step % print_every == 0 or step == total:
            print(f"[REC][train] step {step}/{total} ce={step_loss:.4f} lr={lr_at(step - 1, len(plan), rcp['warmup'], rcp['lr']):.2e}",
                  flush=True)
    model.eval()
    ep = max(1, len(plan) // rcp["epochs"])
    return {"steps": step, "planned_steps": len(plan), "final_loss": loss_log[-1],
            "mean_loss_first_epoch": sum(loss_log[:ep]) / len(loss_log[:ep]),
            "mean_loss_last_epoch": sum(loss_log[-ep:]) / len(loss_log[-ep:]),
            "tokens_seen": tokens_seen, "n_examples": sum(len(p) for p in pools),
            "recipe": {"lr": rcp["lr"], "epochs": rcp["epochs"], "batch_size": per_pool * len(pools),
                       "warmup_ratio": rcp["warmup"], "seed": int(seed), "max_steps": max_steps,
                       "micro_batch": micro_batch, "weight_decay": rcp["weight_decay"],
                       "betas": list(rcp["betas"])}}


# ---------------------------------------------------------------- IM training examples (text tasks)
def im_text_examples(task: str, rows: list) -> list:
    """Supervised examples for the text IM tasks in the READOUT's own format (0-shot, like the MMLU
    letter ladder): mmlu_letter = relearn.format_qa; mmlu_cloze = text_eval._q_block_cloze prompt
    + the gold option TEXT; arith / triviaqa_cloze = question + 'Answer:' + the gold string."""
    from infra.pg import relearn as R
    from infra.pg.text_eval import _q_block_cloze
    out = []
    for r in rows:                       # `hash` is always the row's own matrix_items key, never re-derived
        if task == "mmlu_letter":
            e = R.format_qa(r["question"], r["choices"], r["answer"])
            out.append({"prompt": e["prompt"], "target_text": e["target_text"], "hash": r["hash"]})
        elif task == "mmlu_cloze":
            out.append({"prompt": _q_block_cloze(r["question"]),
                        "target_text": " " + str(r["choices"][int(r["answer"])]).strip(),
                        "hash": r["hash"]})
        elif task in ("arith", "triviaqa_cloze"):
            gold = r["answer"] if isinstance(r["answer"], str) else r["choices"][int(r["answer"])]
            out.append({"prompt": r["question"].strip() + "\nAnswer:", "target_text": " " + str(gold).strip(),
                        "hash": r["hash"]})
        else:
            raise ValueError(f"{task} is not a text IM task")
    return out


# ---------------------------------------------------------------- identity (D18) + fractions
def identity_check(ref: dict, cur: dict, min_agree: float = IDENTITY_MIN_AGREE) -> dict:
    """Per-item agreement of a dose-0 readout with the behaviour cell's vector for the SAME model
    (D18/D19: hash sequences identical, order included; matrix_core.preflight_match = ≥ 99 % item
    agreement AND |Δacc| ≤ 0.01). Failure raises — never caught."""
    from infra.pg import matrix_core as MC
    rh, ch = list(ref["per_item_hash"]), list(cur["per_item_hash"])
    assert rh == ch, f"item hash sequence differs (ref n={len(rh)}, cur n={len(ch)})"
    pm = MC.preflight_match(cur["per_item"], ref["per_item"], min_agree=min_agree)
    if not pm["pass"]:
        raise IdentityMismatch(f"dose-0 identity agree={pm['agree']:.4f} dacc={pm['dacc']:.4f} on n={len(rh)} (need ≥ {min_agree}, ≤ 0.01)")
    return {**pm, "n": len(rh), "min_agree": min_agree}


def subset_by_subject(items: list, rec: dict, subjects) -> dict:
    """Restrict a readout record to items whose subject ∈ subjects (held-out generalisation)."""
    keep = [i for i, it in enumerate(items) if it["subject"] in set(subjects)]
    assert len(items) == len(rec["per_item"])
    return {"per_item": [rec["per_item"][i] for i in keep], "per_item_hash": [rec["per_item_hash"][i] for i in keep],
            "n": len(keep), "acc": (sum(rec["per_item"][i] for i in keep) / len(keep)) if keep else float("nan")}


def rec_fraction(before, after, x, n_boot: int, seed: int = 1337) -> dict:
    """S_rec / K_rec inputs: matrix_core.bootstrap_fraction (ratio only when the lost-margin CI
    excludes 0) + the plain accuracies."""
    from infra.pg import matrix_core as MC
    import numpy as np
    bf = MC.bootstrap_fraction(before, after, x, n_boot=n_boot, seed=seed)
    return {"acc_before": float(np.mean(before)), "acc_after": float(np.mean(after)), "acc": float(np.mean(x)), **bf}


def lost_fraction(damaged, x, floor: float, n_boot: int, seed: int = 1337) -> dict:
    """O_lost(n) relative to the damaged model (§1.2): [(dam − floor) − (x − floor)]/(dam − floor),
    paired-item bootstrap, no draws dropped, clipped to 1 for the point as matrix_core.margin_lost."""
    from infra.pg import matrix_core as MC
    import numpy as np
    d, xx = np.asarray(damaged, float), np.asarray(x, float)
    assert d.size == xx.size >= 8, (d.size, xx.size)
    ml = MC.margin_lost(float(d.mean()), float(xx.mean()), floor)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    den = d[idx].mean(1) - floor
    draws = (d[idx].mean(1) - xx[idx].mean(1)) / np.where(np.abs(den) < 1e-12, np.nan, den)
    draws = draws[np.isfinite(draws)]
    return {"acc_damaged": float(d.mean()), "acc": float(xx.mean()), "floor": floor, **ml,
            "ci": (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))) if draws.size else None}


# ---------------------------------------------------------------- GB
def sweep_layers(n_layers: int, stride: int) -> list:
    ls = list(range(0, n_layers, stride))
    if ls[-1] != n_layers - 1:
        ls.append(n_layers - 1)
    return ls


def null_layers(n_layers: int, k: int = N_NULL_CUTS) -> list:
    """k evenly spread layers for the mismatched-item cuts (BROCA convention: spread, ≥ 5)."""
    import numpy as np
    ls = sorted({int(round(x)) for x in np.linspace(1, n_layers - 1, k)})
    assert len(ls) >= k, f"n_layers {n_layers} too small for {k} distinct null layers"
    return ls


def o_restored(o_graft: float, o_rec: float, o_dam: float) -> float:
    den = o_dam - o_rec
    assert den > 1e-9, f"GB needs O lost by recovery (damaged {o_dam}, recovered {o_rec})"
    return float((o_graft - o_rec) / den)


def greedy_generate(call, inputs: dict, max_new_tokens: int, eos_id: int, pad_id: int) -> list:
    """No-cache greedy decode through ANY callable model (a bare model or broca_sweep.PatchedLM,
    whose donor/recipient re-run on the growing sequence each step so the graft holds at every
    position). Left-padded batch; returns per-row new-token id lists truncated at the first EOS.
    Validated on the real cell against transformers' generate by the self-graft gate."""
    import torch
    ids, am = inputs["input_ids"], inputs["attention_mask"]
    extra = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    B = ids.shape[0]
    done = torch.zeros(B, dtype=torch.bool, device=ids.device)
    new = []
    for _ in range(max_new_tokens):
        out = call(input_ids=ids, attention_mask=am, use_cache=False, **extra)
        nxt = out.logits[:, -1, :].argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, pad_id), nxt)
        new.append(nxt)
        done = done | (nxt == eos_id)
        ids = torch.cat([ids, nxt[:, None]], dim=1)
        am = torch.cat([am, torch.ones_like(nxt[:, None])], dim=1)
        if bool(done.all()):
            break
    gen = torch.stack(new, dim=1).tolist()
    rows = []
    for r in gen:
        cut = r.index(eos_id) if eos_id in r else len(r)
        rows.append(r[:cut])
    return rows


# ---------------------------------------------------------------- SUB
def layer_of(name: str):
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def install_subspace(dW_inst, k: int):
    """Top-k left-singular subspace U_k (d_out × k) of the install delta, torch on dW's device."""
    import torch
    U, _, _ = torch.linalg.svd(dW_inst.float(), full_matrices=False)
    return U[:, :k].contiguous()


def projection_energy(Uk, dW_rec) -> tuple:
    """(‖U_kᵀ ΔW_rec‖²_F, ‖ΔW_rec‖²_F)."""
    import torch
    x = dW_rec.float().to(Uk.device)
    return float(torch.linalg.norm(Uk.T @ x) ** 2), float(torch.linalg.norm(x) ** 2)


def haar_null_energies(dW_rec, k: int, n_null: int = 200, seed: int = 1337) -> list:
    """‖Qᵀ ΔW_rec‖²_F for n_null Haar-random k-subspaces — the SAME numpy draws (default_rng(seed),
    standard_normal((d, k)) per draw, in order) as matrix_core.subspace_overlap, so its null_p95 is
    reproduced exactly; the QR + projection run in torch on dW's device."""
    import numpy as np
    import torch
    x = dW_rec.float()
    d = x.shape[0]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_null):
        G = torch.as_tensor(rng.standard_normal((d, k)), dtype=torch.float32, device=x.device)
        Q, _ = torch.linalg.qr(G)
        out.append(float(torch.linalg.norm(Q.T @ x) ** 2))
    return out


def sub_statistics(deltas_rec: dict, subspaces: dict, k: int, n_null: int, seed: int,
                   deltas_null: dict | None = None) -> dict:
    """Per-matrix and per-layer (energy-weighted) overlap of ΔW_rec with the install subspace, the
    Haar null (per matrix p95; per layer p95 of the per-draw energy sums), and the unrelated-adapter
    null (the same statistic for an IM adapter from another task).

    The per-layer records also carry `contention` and the `margin` that decided it, straight from
    `matrix_core.contention` — the frozen rule, called here, not re-expressed. Before 2026-09-03 this
    banked the three statistics and no boolean, so `matrix_core.sub_contention_frac` (which wants one
    boolean per layer) had no reachable input from a banked artifact and the offline reader restated
    the decision line by hand. `contention` is None exactly when a statistic is non-finite (a
    degenerate zero-energy layer): the rule REFUSES non-finite input by design, so the refusal is
    recorded here and stamped INPUTS-MISSING downstream rather than scored."""
    import numpy as np
    from infra.pg import matrix_core as MC
    per_matrix, by_layer = {}, {}
    for name, dW in deltas_rec.items():
        Uk = subspaces[name]
        num, den = projection_energy(Uk, dW)
        nulls = haar_null_energies(dW, k, n_null, seed)
        rec = {"overlap": num / den if den > 0 else float("nan"), "energy": den,
               "haar_p95": float(np.percentile(np.asarray(nulls) / den, 95)) if den > 0 else float("nan"),
               "haar_mean": float(np.mean(nulls) / den) if den > 0 else float("nan"), "k_over_d": k / Uk.shape[0]}
        if deltas_null is not None:
            nn_, nd = projection_energy(Uk, deltas_null[name])
            rec["overlap_unrelated"] = nn_ / nd if nd > 0 else float("nan")
        per_matrix[name] = rec
        L = layer_of(name)
        agg = by_layer.setdefault(L, {"num": 0.0, "den": 0.0, "null": np.zeros(n_null), "u_num": 0.0, "u_den": 0.0})
        agg["num"] += num
        agg["den"] += den
        agg["null"] += np.asarray(nulls)
        if deltas_null is not None:
            agg["u_num"] += nn_
            agg["u_den"] += nd
    layers = {}
    for L, a in sorted(by_layer.items(), key=lambda kv: (kv[0] is None, kv[0])):
        rec = {"overlap": a["num"] / a["den"] if a["den"] > 0 else float("nan"),
               "haar_p95": float(np.percentile(a["null"] / a["den"], 95)) if a["den"] > 0 else float("nan"),
               "energy": a["den"]}
        if deltas_null is not None:
            rec["overlap_unrelated"] = a["u_num"] / a["u_den"] if a["u_den"] > 0 else float("nan")
            vals = (rec["overlap"], rec["overlap_unrelated"], rec["haar_p95"])
            rec["margin"] = MC.CONTENTION_MARGIN
            rec["contention"] = MC.contention(*vals) if all(np.isfinite(v) for v in vals) else None
        layers[str(L)] = rec
    return {"k": k, "n_null": n_null, "seed": seed, "per_matrix": per_matrix, "per_layer": layers}


# ---------------------------------------------------------------- memory estimate
def memory_estimate_gb(n_model_params: int, n_trainable: int, recipe_key: str, offload: bool,
                       act_gb: float = 4.0, gpu_gb: float = 48.0) -> dict:
    """Weights bf16 (2 B/param) + optimizer state: LoRA A/B are fp32 params (4 master + 4 grad + 8
    moments = 16 B/param on the device); full-FT bf16 grads (2 B) + fp32 master/m/v (12 B) on the
    device or the host (offload). `act_gb` = activations + logits at micro-batch 2 (stated)."""
    weights = 2 * n_model_params / 1e9
    if recipe_key == "full":
        dev = weights + 2 * n_trainable / 1e9 + (0 if offload else 12 * n_trainable / 1e9) + act_gb
        host = 12 * n_trainable / 1e9 if offload else 0.0
    else:
        dev = weights + 16 * n_trainable / 1e9 + act_gb
        host = 0.0
    return {"weights_gb": weights, "device_gb": dev, "host_state_gb": host, "act_gb": act_gb,
            "fits_gpu": bool(dev <= gpu_gb), "gpu_gb": gpu_gb}


def atomic_write_json(path, obj) -> None:
    import os
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, str(path))


# ---------------------------------------------------------------- dry-run
def _dryrun():
    import numpy as np
    import torch
    from infra.pg import matrix_core as MC
    torch.manual_seed(0)
    rc, h = load_recipes("configs/matrix_recipes.json")
    assert len(h) == 64
    r512, r64, full = recipe(rc, "r512"), recipe(rc, "r64"), recipe(rc, "full")
    assert r512["rank"] == 512 and r512["alpha"] == 512.0 and r64["rank"] == 64 and full["rank"] is None
    assert r512["lr"] == 2e-4 and full["lr"] == 1e-5 and r512["batch"] == 16 and r512["epochs"] == 2
    assert dose_schedule([16000, 250, 1000, 4000], "rec", False) == [250, 1000, 4000, 16000]
    assert dose_schedule([4000], "permuted", False) == [4000] and dose_schedule([250], "rec", True) == [SMOKE_DOSE]
    assert n_steps(250, 16, 2) == 32 and n_steps(16000, 16, 2) == 2000
    # lr schedule == relearn's LambdaLR factor sequence
    from infra.pg import relearn as R
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1.0)
    n_warm, total = int(0.05 * 40), 40
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / max(1, n_warm) if s < n_warm else
                                              0.5 * (1 + math.cos(math.pi * min(1.0, (s - n_warm) / max(1, total - n_warm)))))
    for s in range(40):
        assert abs(opt.param_groups[0]["lr"] - lr_at(s, 40, 0.05, 1.0)) < 1e-12, s
        opt.step(); sched.step()
    # Fp32AdamW == torch AdamW (fp32 params, wd on/off, offload on/off)
    for wd in (0.0, 0.1):
        for off in (False, True):
            a = torch.nn.Parameter(torch.randn(5, 3)); b = torch.nn.Parameter(a.detach().clone())
            ref = torch.optim.AdamW([b], lr=1e-2, betas=(0.9, 0.999), weight_decay=wd)
            mine = Fp32AdamW([a], 1e-2, (0.9, 0.999), weight_decay=wd, offload=off)
            for _ in range(5):
                g = torch.randn(5, 3); a.grad = g.clone(); b.grad = g.clone()
                mine.step(1e-2); ref.step(); ref.zero_grad()
            assert torch.allclose(a, b, atol=1e-6), (wd, off, (a - b).abs().max())
    # bf16 params keep an exact fp32 master
    a16 = torch.nn.Parameter(torch.randn(4, 4).to(torch.bfloat16))
    o16 = Fp32AdamW([a16], 1e-3, offload=True)
    a16.grad = torch.ones_like(a16); o16.step(1e-3)
    assert o16.master[0].dtype == torch.float32 and a16.dtype == torch.bfloat16 and o16.state_gb()["where"] == "host"
    # answer_ce: one-token target == CE at the last prompt position (relearn's loss)
    logits = torch.randn(2, 3, 7); labels = torch.tensor([[-100, -100, 4], [-100, -100, 2]])
    ce, n = answer_ce(logits[:, -2:], labels)
    ref_ce = torch.nn.functional.cross_entropy(logits[:, 1], torch.tensor([4, 2]), reduction="sum")
    assert n == 2 and torch.allclose(ce, ref_ce)
    lab2 = torch.tensor([[-100, 5, 4], [-100, -100, 2]])
    ce2, n2 = answer_ce(logits, lab2)
    assert n2 == 3 and torch.allclose(ce2, torch.nn.functional.cross_entropy(logits[:, :2].reshape(-1, 7), lab2[:, 1:].reshape(-1), ignore_index=-100, reduction="sum"))
    # step plan: two pools, dose 20, 8 per pool, 2 epochs → 6 steps, both pools every step
    plan = step_plan([list(range(20)), list(range(20))], 8, 2, 0)
    assert len(plan) == 6 and all(len(g) == 2 for g in plan)
    assert sorted(i for st in plan[:3] for pi, idx in st for i in idx if pi == 0) == list(range(20))
    # identity pass / fail
    ref = {"per_item": [1, 0] * 12, "per_item_hash": [f"h{i}" for i in range(24)]}
    assert identity_check(ref, dict(ref))["agree"] == 1.0
    bad = {"per_item": [0, 0] + [1, 0] * 11, "per_item_hash": [f"h{i}" for i in range(24)]}
    raised = False
    try:
        identity_check(ref, bad)
    except IdentityMismatch:
        raised = True
    assert raised
    # fractions
    rng = np.random.default_rng(1)
    b = np.ones(100); a = (rng.random(100) < 0.3).astype(float); x = np.where(rng.random(100) < 0.6, 1.0, a)
    rf = rec_fraction(b, a, x, 200); assert rf["ratio_status"] == "reported" and 0 < rf["ratio_point"] < 1
    lf = lost_fraction(np.ones(50), np.r_[np.ones(25), np.zeros(25)], 0.0, 200)
    assert abs(lf["lost"] - 0.5) < 1e-9 and lf["ci"][0] <= 0.5 <= lf["ci"][1]
    # GB helpers
    assert sweep_layers(26, 2)[-1] == 25 and len(sweep_layers(26, 2)) == 14 and len(null_layers(26)) == 5
    assert abs(o_restored(0.5, 0.2, 0.8) - 0.5) < 1e-12
    # SUB: exact agreement with matrix_core.subspace_overlap (same draws)
    d = 40; inst = torch.as_tensor(rng.standard_normal((d, d)), dtype=torch.float32)
    U = install_subspace(inst, 4)
    rec_in = U[:, :4] @ torch.as_tensor(rng.standard_normal((4, d)), dtype=torch.float32)
    st = sub_statistics({"layers.0.q": rec_in}, {"layers.0.q": U}, 4, 50, 1337, {"layers.0.q": inst})
    ref_ov = MC.subspace_overlap(rec_in.numpy(), inst.numpy(), 4, inst.numpy(), n_null=50)
    pm = st["per_matrix"]["layers.0.q"]
    assert abs(pm["overlap"] - ref_ov["overlap"]) < 1e-4 and abs(pm["haar_p95"] - ref_ov["haar_p95"]) < 1e-3, (pm, ref_ov)
    assert abs(pm["overlap_unrelated"] - ref_ov["overlap_unrelated"]) < 1e-4
    assert pm["overlap"] > pm["haar_p95"] and "0" in st["per_layer"] and 0 <= pm["overlap_unrelated"] <= 1
    # the per-layer boolean `matrix_core.sub_contention_frac` consumes is now BANKED, and it is the
    # frozen rule's own answer over these same banked statistics — not a second opinion.
    pl = st["per_layer"]["0"]
    assert pl["margin"] == MC.CONTENTION_MARGIN
    assert pl["contention"] == MC.contention(pl["overlap"], pl["overlap_unrelated"], pl["haar_p95"])
    assert MC.sub_contention_frac([pl["contention"]])["frac"] in (0.0, 1.0)
    # memory table (stated in the report): row 2 = PaliGemma2-3b (3.03 B total, 2.02 B full-FT scope, r512 = 664 M)
    m = memory_estimate_gb(3_030_000_000, 664_000_000, "r512", False); assert m["fits_gpu"] and m["device_gb"] < 25
    mf = memory_estimate_gb(3_030_000_000, 2_020_000_000, "full", False)
    assert mf["fits_gpu"] and 36 < mf["device_gb"] < 40, mf          # nominal fit on 48 GB, ≈ 10 GB headroom
    assert memory_estimate_gb(3_030_000_000, 2_020_000_000, "full", True)["device_gb"] < 16
    assert not memory_estimate_gb(4_400_000_000, 3_600_000_000, "full", False)["fits_gpu"]   # row 3b: offload or 80 GB
    assert memory_estimate_gb(4_400_000_000, 3_600_000_000, "full", True)["fits_gpu"]
    assert not memory_estimate_gb(9_660_000_000, 1_730_000_000, "r512", False)["fits_gpu"]   # 2-10B r512 exceeds 48 GB
    print("matrix_recovery_core dry-run: ALL PASS")


if __name__ == "__main__":
    _dryrun()
