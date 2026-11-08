"""HELMHOLTZ-2 arm job — train ONE arm (B benign / D_know / D_skill) from gemma-3-4b-pt,
then score it. Ladder mode scores BEHAVIOR ONLY (MCQ+cloze margins + the dose gate) so the
diff statistics stay unseen until thresholds are frozen; --score-sites adds the full R0-style
site sweep (CKA + shipped-SAE diff + FVU + this-run plant validity) for qualifying-dose runs.

Parametrization is IDENTICAL across arms (relearn.RankRAdapter r=16 α=16 on every decoder
Linear, embeds/lm_head excluded) — the dose-matched-benign confound control. The whole
conditional model is frozen first (vision included), then adapters attach to the text module.

  # ladder rung (behavior only):
  pg run infra.aws_pg.jobs.helmholtz2_arm --args "--arm know --dose-tokens 5000000 \
      --bind-s3 s3://.../results/h2_seed/out/metrics.json" --type g6e.2xlarge \
      --s3-prefix results/h2_lad_know_5m --budget-min 300
  # qualifying-dose seed run (full scoring):
  ... --args "--arm skill --skill-n 8000 --skill-epochs 4 --score-sites \
      --thresholds-s3 s3://.../results/h2_r0/out/metrics_thresholds_draft.json ..."
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import time

DATASETS = {"know": "codeparrot/codeparrot-clean",       # ungated, field 'content'
            "benign": "allenai/tulu-3-sft-mixture"}      # ungated, field 'messages'
CHANCE = 0.25


def _fetch(uri, dest):
    """Presigned-HTTP (no AWS creds on rented boxes) or aws-cli S3 fetch."""
    if str(uri).startswith("http"):
        import urllib.request
        urllib.request.urlretrieve(uri, str(dest))
    else:
        subprocess.run(["aws", "s3", "cp", uri, str(dest), "--region", "us-east-1"], check=True)


def _text_module(model):
    from infra.aws_pg.jobs.helmholtz2_r0 import resolve_gemma3_layers
    layers = resolve_gemma3_layers(model)
    for _, mod in model.named_modules():
        if getattr(mod, "layers", None) is layers:
            return mod
    raise RuntimeError("text module owning .layers not found")


def _doc_stream(arm, dataset, seed):
    from datasets import load_dataset
    ds = load_dataset(dataset, split="train", streaming=True).shuffle(seed=seed, buffer_size=10_000)
    for r in ds:
        if arm == "know":
            yield r["content"]
        else:
            yield "\n\n".join(f"{m['role']}: {m['content']}" for m in r["messages"])


def _contamination_prefixes():
    from datasets import load_dataset
    from infra.pg import text_eval as TE
    ds = load_dataset(TE.MMLU_REPO, "all")
    return [r["question"][:64] for r in ds["test"] if len(r["question"]) >= 32]


def train_text_ce(model, tok, device, arm, dataset, dose_tokens, lr, seed,
                  seq_len=1024, micro_blocks=2, accum=8, warmup_ratio=0.03, sgd_master=False):
    """Full-sequence-CE LoRA training on packed blocks until dose_tokens are SEEN.
    Deterministic in seed. Returns loss trace + contamination report. Fail-loud."""
    import torch
    prefixes = _contamination_prefixes()
    dropped = 0
    buf: list = []
    blocks: list = []
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable, "no trainable params"
    total_steps = max(1, dose_tokens // (seq_len * micro_blocks * accum))
    n_warm = max(1, int(warmup_ratio * total_steps))
    def _lr_fac(st):
        return st / n_warm if st < n_warm else \
            0.5 * (1 + math.cos(math.pi * min(1.0, (st - n_warm) / max(1, total_steps - n_warm))))
    if sgd_master:
        # 48GB full-FT path: bf16 params + fp32 MASTER copies, plain SGD (no moment buffers).
        # AdamW on bf16 params underflows; AdamW fp32 states need 80GB (capacity drought).
        opt = sched = None
        masters = [p.detach().float().clone() for p in trainable]
    else:
        opt = torch.optim.AdamW(trainable, lr=lr)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_fac)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    losses: list = []
    tokens_seen = 0
    step = 0
    micro_i = 0
    stream = _doc_stream(arm, dataset, seed)
    t_last = time.time()
    while tokens_seen < dose_tokens:
        while len(blocks) < micro_blocks:
            doc = next(stream)
            if any(p in doc for p in prefixes):
                dropped += 1
                continue
            buf.extend(tok(doc, add_special_tokens=False)["input_ids"] + [tok.eos_token_id])
            while len(buf) >= seq_len:
                blocks.append(buf[:seq_len])
                buf = buf[seq_len:]
        import torch as t
        ids = t.tensor(blocks[:micro_blocks], dtype=t.long, device=device)
        blocks = blocks[micro_blocks:]
        out = model(input_ids=ids, labels=ids)
        loss = out.loss / accum
        assert t.isfinite(loss), f"non-finite loss at step {step}"
        loss.backward()
        tokens_seen += ids.numel()
        micro_i += 1
        if micro_i % accum == 0:
            if sgd_master:
                lr_t = lr * _lr_fac(step)
                with torch.no_grad():
                    for p_, m_ in zip(trainable, masters):
                        assert p_.grad is not None
                        m_.add_(p_.grad.float(), alpha=-lr_t)
                        p_.data.copy_(m_.to(p_.dtype))
                        p_.grad = None
            else:
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            losses.append(float(loss.item() * accum))
            if step % 50 == 0:
                print(f"[ARM] step {step}/{total_steps} ce={losses[-1]:.4f} "
                      f"tok={tokens_seen:,} ({time.time()-t_last:.0f}s/50)", flush=True)
                t_last = time.time()
    model.eval()
    model.gradient_checkpointing_disable()
    k = max(1, min(20, len(losses) // 5))
    return {"losses_head": losses[:50], "losses_tail": losses[-50:], "steps": len(losses),
            "tokens_seen": tokens_seen, "docs_dropped_contaminated": dropped,
            "ce_initial": sum(losses[:k]) / k, "ce_final": sum(losses[-k:]) / k,
            "recipe": {"dataset": dataset, "dose_tokens": dose_tokens, "lr": lr,
                       "optimizer": "sgd_master" if sgd_master else "adamw",
                       "seq_len": seq_len, "micro_blocks": micro_blocks, "accum": accum,
                       "seed": seed}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True, choices=("benign", "know", "skill"))
    ap.add_argument("--bind-s3", required=True)
    ap.add_argument("--thresholds-s3", default=None)
    ap.add_argument("--base-repo", default="google/gemma-3-4b-pt")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--dose-tokens", type=int, default=None, help="benign/know dose")
    ap.add_argument("--skill-n", type=int, default=8000)
    ap.add_argument("--skill-epochs", type=int, default=4)
    ap.add_argument("--skill-batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--skill-lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-per-subject", type=int, default=10)
    ap.add_argument("--score-sites", action="store_true")
    ap.add_argument("--load-adapter-s3", default=None,
                    help="skip training: attach adapters and load this adapter.pt (S3 URI) from a "
                         "prior run; REQUIRES --train-metrics-s3 for the CE provenance")
    ap.add_argument("--train-metrics-s3", default=None,
                    help="metrics.json URI of the run that trained the loaded adapter (ce fields + recipe)")
    ap.add_argument("--fullft-sgd", action="store_true",
                    help="48GB full-FT fallback: bf16 params + fp32-master plain SGD (lr default 2e-3). "
                         "G2's benign control must use the IDENTICAL config.")
    ap.add_argument("--full-ft", action="store_true",
                    help="K1 escalation: full-parameter SFT of the text decoder instead of LoRA. "
                         "Needs an 80GB GPU (p5.4xlarge/p4de): bf16 weights + fp32 AdamW states "
                         "~55GB. lr default drops to 1e-5.")
    ap.add_argument("--n-harvest-items", type=int, default=270)
    a = ap.parse_args()
    t0 = time.time()
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    bind_p = out / "bind.json"
    _fetch(a.bind_s3, bind_p)
    bind = json.loads(bind_p.read_text())
    thresholds = None
    if a.thresholds_s3:
        th_p = out / "thresholds.json"
        _fetch(a.thresholds_s3, th_p)
        thresholds = json.loads(th_p.read_text())
    assert not (a.score_sites and thresholds is None), "--score-sites requires --thresholds-s3 (frozen bands)"

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from infra.pg import helmholtz2_core as HC
    from infra.pg import relearn as Rl
    from infra.pg import text_eval as TE
    assert torch.cuda.is_available()
    device = "cuda"

    rep = {"object": f"HELMHOLTZ-2 arm={a.arm}", "arm": a.arm, "base": a.base_repo,
           "seed": a.seed, "rank": a.rank, "alpha": a.alpha, "score_sites": a.score_sites}
    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    _w()

    tok = AutoTokenizer.from_pretrained(a.base_repo)
    m_after = AutoModelForImageTextToText.from_pretrained(
        a.base_repo, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation="eager").eval()
    for p in m_after.parameters():
        p.requires_grad_(False)
    text_mod = _text_module(m_after)
    if a.full_ft:
        for prm in text_mod.parameters():
            prm.requires_grad_(True)
        n_tr = sum(p.numel() for p in text_mod.parameters() if p.requires_grad)
        rep["adapters"] = {"mode": "full_ft", "trainable_params": int(n_tr)}
        if a.lr == 2e-4:
            a.lr = 2e-3 if a.fullft_sgd else 1e-5   # explicit --lr overrides
        if not a.fullft_sgd:
            # 80GB path: fp32 train (bf16+AdamW@1e-5 underflows), cast back for scoring.
            m_after.to(torch.float32)
        print(f"[ARM] {a.arm}: FULL-FT text decoder "
              f"({'bf16+sgd_master' if a.fullft_sgd else 'fp32+adamw'}), "
              f"{n_tr:,} trainable, lr={a.lr}", flush=True)
    else:
        arep = Rl.attach_adapters(text_mod, rank=a.rank, seed=a.seed, alpha=a.alpha)
        assert arep["n_modules"] > 100, arep
        rep["adapters"] = {k: arep[k] for k in ("n_modules", "trainable_params", "rank", "alpha")}
        print(f"[ARM] {a.arm}: {arep['n_modules']} adapted modules, "
              f"{arep['trainable_params']:,} trainable", flush=True)
    m_after.to(device)
    _w()

    # ---- train per arm ----
    if a.load_adapter_s3:
        assert not a.full_ft and a.train_metrics_s3, "--load-adapter-s3 needs LoRA mode + --train-metrics-s3"
        ad_p = out / "adapter_loaded.pt"
        _fetch(a.load_adapter_s3, ad_p)
        tm_p = out / "train_metrics_prev.json"
        _fetch(a.train_metrics_s3, tm_p)
        prev = json.loads(tm_p.read_text())
        sd = torch.load(ad_p, map_location="cpu")
        missing, unexpected = text_mod.load_state_dict(sd, strict=False)
        n_loaded = len(sd)
        assert n_loaded > 400 and not unexpected, (n_loaded, unexpected[:5])
        rep["train"] = dict(prev["train"]) | {"loaded_adapter_from": a.load_adapter_s3,
                                              "train_metrics_source": a.train_metrics_s3}
        print(f"[ARM] adapters LOADED ({n_loaded} tensors) from {a.load_adapter_s3}; "
              f"CE provenance from {a.train_metrics_s3}", flush=True)
    elif a.arm == "skill":
        from datasets import load_dataset
        ds = load_dataset(TE.MMLU_REPO, "all")
        pool = [dict(r) for r in ds["auxiliary_train"]]
        ev_hashes = [Rl.qa_item_hash(r["question"]) for r in ds["test"]]
        kept = Rl.dedup_against_eval(pool, ev_hashes)["kept"]
        sub = Rl.shuffle_labels(Rl.ladder_subset(kept, a.skill_n), seed=a.seed)
        examples = [Rl.format_qa(r["question"], r["choices"], r["answer"]) for r in sub]
        tr = Rl.train_adapter_qa(m_after, tok, device, examples, lr=a.skill_lr,
                                 epochs=a.skill_epochs, batch_size=a.skill_batch,
                                 warmup_ratio=0.03, seed=a.seed, micro_batch=2)
        rep["train"] = {"steps": tr["steps"], "ce_initial": tr["mean_loss_first_epoch"],
                        "ce_final": tr["mean_loss_last_epoch"], "tokens_seen": tr["tokens_seen"],
                        "recipe": tr["recipe"] | {"n": a.skill_n, "epochs": a.skill_epochs},
                        "ln4": float(np.log(4))}
    else:
        assert a.dose_tokens, "--dose-tokens required for benign/know"
        dataset = a.dataset or DATASETS[a.arm]
        rep["dataset"] = dataset
        mb, acc_ = (1, 16) if a.full_ft else (2, 8)   # same effective batch; peak control
        tr = train_text_ce(m_after, tok, device, a.arm, dataset, a.dose_tokens, a.lr, a.seed,
                           micro_blocks=mb, accum=acc_, sgd_master=a.fullft_sgd)
        if a.full_ft and not a.fullft_sgd:
            m_after.to(torch.bfloat16)
            torch.cuda.empty_cache()
        rep["train"] = tr
    ce_ratio = rep["train"]["ce_final"] / max(rep["train"]["ce_initial"], 1e-9)
    rep["train"]["ce_final_over_initial"] = ce_ratio
    print(f"[ARM] trained: ce {rep['train']['ce_initial']:.4f} -> {rep['train']['ce_final']:.4f} "
          f"(ratio {ce_ratio:.3f})", flush=True)
    if not a.full_ft:   # adapters are small; a full-FT checkpoint (8GB) is deliberately not saved
        torch.save({k: v for k, v in text_mod.state_dict().items() if ".A" in k or ".B" in k},
                   out / "adapter.pt")
    torch.cuda.empty_cache()
    m_before = AutoModelForImageTextToText.from_pretrained(
        a.base_repo, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation="eager").eval()
    _w()

    # ---- behavior, both sides ----
    beh = {}
    for name, m in (("before", m_before), ("after", m_after)):
        mcq = TE.mmlu_accuracy(m, tok, device, n_per_subject=a.n_per_subject)
        clz = TE.mmlu_cloze(m, tok, device, processor=None, n_per_subject=a.n_per_subject)
        beh[name] = {"mcq": mcq, "clz": clz}
        rep[f"behavior_{name}"] = {"mcq_acc": mcq["acc"], "mcq_pct": mcq["acc_pct"],
                                   "cloze_acc": clz["acc"], "cloze_pct": clz["acc_pct"], "n": mcq["n"]}
        print(f"[ARM] {name}: MCQ {mcq['acc_pct']:.2f}% cloze {clz['acc_pct']:.2f}%", flush=True)
        _w()
    mb, ma = beh["before"]["mcq"]["acc"], beh["after"]["mcq"]["acc"]
    cb, ca = beh["before"]["clz"]["acc"], beh["after"]["clz"]["acc"]
    mcq_drop = float(max(0.0, (mb - ma) / max(mb - CHANCE, 1e-6)))
    cloze_drop = float(max(0.0, (cb - ca) / max(cb - CHANCE, 1e-6)))
    gate = {"benign": HC.gate_benign(mcq_drop),
            "know": HC.gate_dose_know(mcq_drop, ce_ratio),
            "skill": HC.gate_dose_skill(mcq_drop, ce_ratio, cloze_drop)}[a.arm]
    rep["margins"] = {"mcq_margin_drop": mcq_drop, "cloze_margin_drop": cloze_drop,
                      "ce_final_over_initial": ce_ratio, "gate_pass": bool(gate)}
    (out / "per_item.json").write_text(json.dumps(
        {"mcq_before": beh["before"]["mcq"]["per_item"], "mcq_after": beh["after"]["mcq"]["per_item"],
         "hash_before": beh["before"]["mcq"]["per_item_hash"],
         "hash_after": beh["after"]["mcq"]["per_item_hash"],
         "cloze_before": beh["before"]["clz"].get("per_item"),
         "cloze_after": beh["after"]["clz"].get("per_item")}))
    assert beh["before"]["mcq"]["per_item_hash"] == beh["after"]["mcq"]["per_item_hash"], \
        "item pairing broke"
    print(f"[ARM] margins: mcq_drop={mcq_drop:.3f} cloze_drop={cloze_drop:.3f} "
          f"ce_ratio={ce_ratio:.3f} GATE={'PASS' if gate else 'FAIL'}", flush=True)
    _w()

    # ---- optional full site sweep (qualifying doses only; frozen thresholds) ----
    if a.score_sites:
        from collections import defaultdict
        from datasets import load_dataset as ld
        from infra.pg import scope2_sae as S2
        from infra.aws_pg.jobs.helmholtz2_r0 import harvest_multi, resolve_gemma3_layers, sae_rates_torch
        import re
        d_model = bind["bind"]["d_model"]
        scope_repo = bind["scope_repo"]["repo"]
        chosen = bind["scope_repo"]["chosen_dirs"]
        lay_of = lambda d: int(re.search(r"layer_(\d+)", d).group(1))
        primary = sorted([d for d in chosen if "resid_post_all" not in d and "width_65k" in d], key=lay_of)
        final_dir = [d for d in chosen if "resid_post_all" in d]
        dir_by_layer = {lay_of(d): d for d in primary} | {lay_of(d): d for d in final_dir}
        mid_layer = bind["scope_repo"]["mid_layer"]
        ds = ld(TE.MMLU_REPO, "all")
        by_subj = defaultdict(list)
        for r in ds["test"]:
            by_subj[r["subject"]].append(r)
        items = []
        for subj in sorted(by_subj):
            items += by_subj[subj][: a.n_per_subject]
        rng = np.random.default_rng(1337)   # SAME harvest subset as R0 (frozen seed)
        idx = sorted(rng.choice(len(items), size=min(a.n_harvest_items, len(items)), replace=False).tolist())
        texts = [TE._q_block(items[i]["question"], items[i]["choices"]) for i in idx]
        lis = sorted(dir_by_layer)
        acts_b = harvest_multi(m_before, resolve_gemma3_layers(m_before), lis, texts, tok, device)
        acts_a = harvest_multi(m_after, resolve_gemma3_layers(m_after), lis, texts, tok, device)
        sites = {}
        bands = thresholds["per_site_identity_bands"]
        tau = thresholds["tau"]
        for li in lis:
            cka = HC.linear_cka(acts_b[li].numpy(), acts_a[li].numpy())
            sae = S2.load_scope2_sae(scope_repo, dir_by_layer[li], d_model)
            r_b, _, fvu_b, _ = sae_rates_torch(acts_b[li], sae, device)
            r_a, _, fvu_a, _ = sae_rates_torch(acts_a[li], sae, device)
            diff = HC.sae_diff_from_rates(r_b, r_a, tau)
            band = bands[str(li)]
            sites[str(li)] = {"cka": cka, "fvu_before": fvu_b, "fvu_after": fvu_a,
                              "diff": diff, "band": band,
                              "sae_fires": bool(diff["lost"] > band["lost"] * 2 + 1e-4
                                                or diff["dlog_rate"] > band["dlog_rate"] * 2)}
            print(f"[ARM][site L{li}] cka={cka:.4f} surv={diff['survival']:.4f} "
                  f"lost={diff['lost']:.4f} dlog={diff['dlog_rate']:.4f} "
                  f"fvu {fvu_b:.3f}->{fvu_a:.3f} fires={sites[str(li)]['sae_fires']}", flush=True)
            del sae; torch.cuda.empty_cache()
        # this-run plant validity at c=1 on the BEFORE side, mid layer
        sae_mid = S2.load_scope2_sae(scope_repo, dir_by_layer[mid_layer], d_model)
        A_mid = acts_b[mid_layer]
        alpha = HC.plant_alpha_from_spectrum(A_mid.numpy(), c=2.0)  # amended: CKA needs c=2 on this substrate
        g = torch.Generator().manual_seed(1337)
        u = torch.randn(d_model, generator=g); u = u / u.norm()
        planted = A_mid.clone()
        pidx = torch.randperm(A_mid.shape[0], generator=g)[: A_mid.shape[0] // 2]
        planted[pidx] += alpha * u
        cka_pl = HC.linear_cka(A_mid.numpy(), planted.numpy())
        r_c, _, _, _ = sae_rates_torch(A_mid, sae_mid, device)
        r_p, _, _, _ = sae_rates_torch(planted, sae_mid, device)
        pdiff = HC.sae_diff_from_rates(r_c, r_p, tau)
        band = bands[str(mid_layer)]
        rep["plant_validity"] = {"alpha": alpha, "cka_planted": cka_pl,
                                 "cka_fires": bool(cka_pl < 0.995),  # amended frozen rule 2026-09-01 (R0 plant calibration)
                                 "sae_fires": bool(pdiff["lost"] > band["lost"] * 2 + 1e-4
                                                   or pdiff["emergent"] > band["emergent"] * 2 + 1e-4
                                                   or pdiff["dlog_rate"] > band["dlog_rate"] * 2),
                                 "sae_diff_planted": pdiff}
        rep["sites"] = sites
        _w()

    rep["elapsed_s"] = time.time() - t0
    _w()
    print("HELMHOLTZ2_ARM_OK " + json.dumps(
        {"arm": a.arm, "gate_pass": bool(gate), "mcq_drop": mcq_drop,
         "cloze_drop": cloze_drop, "ce_ratio": ce_ratio}), flush=True)


if __name__ == "__main__":
    main()
