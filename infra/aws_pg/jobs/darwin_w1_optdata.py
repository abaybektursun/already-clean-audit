"""DARWIN W1-S / W1-V — the optimizer tail and the data that attacks the core. GPU.

  W1-S  SGD tail (product 1.5) + let-it-breathe (product 1.2), §4/W1-S. One fine-tune task under
        three update rules: stock exact Adam; Adam with the last 10% of steps run as SGD; and
        "let it breathe" — the update's core component deliberately scaled UP by a dose. Readouts:
        task, battery, core motion.

  W1-V  Data conflict scores (§4/W1-V, product 7.5b). Score every training example by how much of
        its gradient lands in the core, drop the top decile, retrain, and see whether retention
        improves. The comparison that makes it a result rather than an anecdote is the CONTROL
        SET: dropping a RANDOM decile and dropping the BOTTOM decile, at identical data size and
        identical step count. Without those, "retention improved after we removed 10% of the data"
        is a statement about data size.

WHY THE BREATHE ARM IS A GRADIENT-SIDE OPERATION. TENANT found the core deadlocked by Adam's
second-moment suppression, and REFEREE found the suppressed motion is not inert. "Let it breathe"
is therefore an intervention on what the optimizer is ALLOWED to move, and the cheapest faithful
form of that is to rescale the gradient's core component before the optimizer sees it:

    g <- g + (gain - 1) * g U Uᵀ

At gain = 1 this is exactly stock Adam, so the dose ladder contains its own null by construction —
which is worth more than a separately-run control that could differ in some other way.

REFEREE'S DOSE. §4/W1-S says to recompute the helpful dose from REFEREE's artifacts. The arm
FETCHES the banked REFEREE aggregate and inspects it for a dose-scale quantity; if it carries one,
that value is added to the ladder and the row names the object it came from. If it does not, the
arm runs its own ladder and records the miss with the keys it looked at. Either way no number
enters this arm from prose.

FAIL LOUD: no try/except, no silent fallback.

  pg run infra.aws_pg.jobs.darwin_w1_optdata --args "--arm W1-S" \\
      --type g6e.2xlarge,g6e.4xlarge,g6e.8xlarge,g6e.xlarge --no-weights ...
"""
import argparse
import json
import pathlib

from infra.pg import darwin_core as D
from infra.pg import darwin_train as T
from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as W

REFEREE_METRICS = "s3://pg-lineage-769063704696/results/referee_aggregate4/out/metrics.json"
DOSE_KEY_HINTS = ("dose", "helpful", "gain", "lam")
BREATHE_LADDER = (1.0, 2.0, 4.0, 8.0)     # gain = 1 IS stock Adam: the ladder contains its null
DROP_FRACTION = 0.1                        # §4/W1-V: "top-decile core-attackers"

# W1-V'S POSITIVE CONTROL, borrowed verbatim from W1-U because it is the same control. The arm
# asks whether removing core-attacking examples IMPROVES RETENTION. If the fine-tune does not
# damage retention in the first place, there is nothing to improve and every arm ties at "no
# forgetting" — the failure mode `feedback_signals_that_cannot_fail` names. The first full run
# measured exactly that: base ppl 24.561 and every arm landing at 23.6-23.8, i.e. the fine-tune
# IMPROVED wikitext perplexity, so the retention contrast was not measuring forgetting at all.
# Rule: the matched-size reference arm's retention LOSS must exceed DAMAGE_MIN_REL x base ppl, or
# the arm records VOID-NO-DAMAGE and makes no claim about core-attackers.
DAMAGE_MIN_REL = 0.05
SPEC_EXPECTATIONS = {
    "W1-S": ("no update rule differs from stock Adam beyond CI on task, battery or core motion",
             "an SGD tail or a deliberately breathing core changes core motion and battery "
             "retention relative to stock Adam at matched task score"),
    "W1-V": ("dropping top-decile core-attackers helps retention no more than dropping a random "
             "or bottom decile of the same size",
             "removing the examples whose gradients land hardest in the core improves retention "
             "beyond the matched random and bottom-decile controls"),
}


class BreatheScaler:
    """g <- g + (gain - 1) * g U Uᵀ, applied to the gradient before the optimizer step."""

    def __init__(self, basis_by_key, device, gain: float):
        import torch
        self.U = {k: torch.as_tensor(U, device=device, dtype=torch.float32)
                  for k, U in basis_by_key.items()}
        self.gain = float(gain)

    def apply(self, modules):
        import torch
        if self.gain == 1.0:
            return
        with torch.no_grad():
            for key, mod in modules.items():
                if mod.weight.grad is None or key not in self.U:
                    continue
                U = self.U[key]
                g = mod.weight.grad.to(torch.float32)
                mod.weight.grad = (g + (self.gain - 1.0) * ((g @ U) @ U.T)).to(mod.weight.grad.dtype)


def referee_dose(out):
    """(dose, provenance) — read from REFEREE's banked aggregate, never from prose."""
    import subprocess
    dest = pathlib.Path(out) / "referee_metrics.json"
    r = subprocess.run(["aws", "s3", "cp", REFEREE_METRICS, str(dest), "--region", "us-east-1"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None, {"source": "arm-local ladder", "uri": REFEREE_METRICS,
                      "why": f"REFEREE aggregate not readable: {r.stderr.strip()[:200]}"}
    d = json.loads(dest.read_text())
    hits = []

    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                p = f"{path}.{k}"
                if any(h in k.lower() for h in DOSE_KEY_HINTS) and isinstance(v, (int, float)):
                    hits.append((p, float(v)))
                walk(v, p)
        elif isinstance(o, list):
            for i, v in enumerate(o[:8]):
                walk(v, f"{path}[{i}]")
    walk(d)
    if hits:
        dose = max(v for _, v in hits)
        return dose, {"source": "REFEREE banked aggregate", "uri": REFEREE_METRICS,
                      "keys": [k for k, _ in hits][:16], "dose": dose,
                      "rule": "largest dose-scale quantity banked by REFEREE"}
    return None, {"source": "arm-local ladder", "uri": REFEREE_METRICS,
                  "top_level_keys": sorted(d)[:16],
                  "why": "REFEREE's banked aggregate carries no dose-scale quantity, so the arm "
                         "runs its own ladder rather than importing a number from prose"}


def motion_report(mods, pristine, banks, bank_ids):
    """{bank_id: mean motion fraction over cells} for the realized update."""
    import numpy as np
    import torch
    out = {}
    for bid in bank_ids:
        vals = []
        for key, m in mods.items():
            dW = (m.weight.detach().to(torch.float32) - pristine[key]).cpu().numpy().astype(
                np.float64)
            if float((dW ** 2).sum()) == 0:
                continue
            vals.append(T.motion_fraction(dW, banks[key][bid]))
        out[bid] = float(np.mean(vals)) if vals else 0.0
    return out


def train_arm(model, mods, batches, *, arm, steps, lr, kind, tail_frac=0.0, breathe=None):
    """One fine-tune under one update rule. Returns the per-step loss history."""
    import torch
    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for m in mods.values():
        m.weight.requires_grad_(True)
        params.append(m.weight)
    adam = T.ExactAdam(params, lr=lr)
    D.check_optimizer(arm, "exact_fp32_adam", offload_exact_fp32=False)
    sgd = torch.optim.SGD(params, lr=lr * 100.0, momentum=0.9)
    switch = int(steps * (1.0 - tail_frac)) if kind == "sgd_tail" else steps
    model.train()
    hist = []
    for step in range(steps):
        b = batches[step % len(batches)]
        loss = model(input_ids=b["input_ids"], labels=b["labels"]).loss
        opt = adam if step < switch else sgd
        opt.zero_grad()
        loss.backward()
        if breathe is not None:
            breathe.apply(mods)
        opt.step()
        hist.append(float(loss.detach()))
    model.eval()
    return hist, switch


def run_w1_s(model, processor, dl, bank_dir, out, atlas, gates, a, base_sha, ctx):
    import torch
    from infra.aws_pg.jobs.invariant_core_func import PG1_REPO, _ppl_blankimg
    from infra.aws_pg.jobs.darwin_w1_u import eval_task_loss

    mods, pristine, banks = ctx["mods"], ctx["pristine"], ctx["banks"]
    bank_ids = D.bank_ids_for("q_proj")
    dose, dose_prov = referee_dose(out)
    ladder = sorted(set(list(BREATHE_LADDER) + ([float(dose)] if dose else [])))
    D.check_dose_grid(ladder, arm="W1-S", min_doses=2 if a.smoke else 4, log_spaced=False)
    print(f"[W1-S] breathe ladder {ladder} (referee dose {dose}, {dose_prov['source']})",
          flush=True)

    arms = [("stock_adam", {"kind": "adam"}), ("sgd_tail", {"kind": "sgd_tail", "tail_frac": 0.1})]
    arms += [(f"breathe_g{g:g}", {"kind": "adam", "gain": g}) for g in ladder if g != 1.0]
    results, seen = {}, set()
    for name, cfg in arms:
        for k, m in mods.items():
            m.weight.data.copy_(pristine[k].to(m.weight.dtype))
        breathe = (BreatheScaler({k: banks[k]["core32"] for k in mods}, "cuda", cfg["gain"])
                   if "gain" in cfg else None)
        hist, switch = train_arm(model, mods, ctx["train_batches"], arm="W1-S", steps=a.steps,
                                 lr=a.lr, kind=cfg["kind"],
                                 tail_frac=cfg.get("tail_frac", 0.0), breathe=breathe)
        ppl = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])
        tl = eval_task_loss(model, ctx["eval_batches"])
        mot = motion_report(mods, pristine, banks, bank_ids)
        for bid, v in mot.items():
            atlas.add("motion_fraction", value=v, bank_id=bid, dose=float(cfg.get("gain", 1.0)),
                      incumbent_id="stock_adam", model_id=PG1_REPO, model_sha=base_sha,
                      flags={"arm_name": name, "kind": cfg["kind"], "steps": a.steps,
                             "switch_step": switch, "task_loss": tl, "retention_ppl": ppl,
                             "ppl_base": ctx["base_ppl"], "referee_dose": dose,
                             "referee_dose_source": dose_prov["source"],
                             "definition": "‖dW U‖²_F / ‖dW‖²_F, mean over cells"})
            seen.add(bid)
        atlas.add("retention_ppl", value=ppl, bank_id="none", dose=float(cfg.get("gain", 1.0)),
                  incumbent_id="stock_adam", model_id=PG1_REPO, model_sha=base_sha,
                  flags={"arm_name": name, "kind": cfg["kind"], "task_loss": tl,
                         "ppl_base": ctx["base_ppl"], "retention_loss": ppl - ctx["base_ppl"],
                         "train_loss_last": hist[-1]})
        atlas.add("task_loss", value=tl, bank_id="none", dose=float(cfg.get("gain", 1.0)),
                  incumbent_id="stock_adam", model_id=PG1_REPO, model_sha=base_sha,
                  flags={"arm_name": name, "kind": cfg["kind"], "retention_ppl": ppl,
                         "train_loss_history": hist[-8:]})
        results[name] = {"ppl": ppl, "task_loss": tl, "motion": mot,
                         "retention_loss": ppl - ctx["base_ppl"]}
        print(f"[W1-S] {name}: task={tl:.4f} ppl={ppl:.4f} core_motion={mot['core32']:.5f}",
              flush=True)
        torch.cuda.empty_cache()
        D.s3_sync(out, a.s3_prefix)

    for k, m in mods.items():
        m.weight.data.copy_(pristine[k].to(m.weight.dtype))
    D.check_controls(seen, arm="W1-S")
    gates.record("W1S_optimizer_arms_measured", True, arms=[n for n, _ in arms],
                 referee_dose=dose, referee_dose_source=dose_prov["source"],
                 results={k: {"ppl": v["ppl"], "task_loss": v["task_loss"],
                              "core_motion": v["motion"]["core32"]} for k, v in results.items()})
    return {"results": results, "breathe_ladder": ladder, "referee_dose": dose,
            "referee_dose_provenance": dose_prov, "base_ppl": ctx["base_ppl"]}


def run_w1_v(model, processor, dl, bank_dir, out, atlas, gates, a, base_sha, ctx):
    import numpy as np
    import torch
    from infra.aws_pg.jobs.invariant_core_func import PG1_REPO, _ppl_blankimg
    from infra.aws_pg.jobs.darwin_w1_u import eval_task_loss

    mods, pristine, banks = ctx["mods"], ctx["pristine"], ctx["banks"]
    batches = ctx["train_batches"]

    # Phase 1 — per-example core-attack score. Batch 1 by construction (build_batches already is),
    # so the gradient scored IS the example's gradient, not a batch average.
    for p in model.parameters():
        p.requires_grad_(False)
    for m in mods.values():
        m.weight.requires_grad_(True)
    model.train()
    scores = []
    for i, b in enumerate(batches):
        loss = model(input_ids=b["input_ids"], labels=b["labels"]).loss
        for m in mods.values():
            m.weight.grad = None
        loss.backward()
        num, den = 0.0, 0.0
        with torch.no_grad():
            for key, m in mods.items():
                if m.weight.grad is None:
                    continue
                g = m.weight.grad.to(torch.float32)
                U = torch.as_tensor(banks[key]["core32"], device=g.device, dtype=torch.float32)
                num += float(((g @ U) ** 2).sum())
                den += float((g ** 2).sum())
        assert den > 0, f"example {i} produced a zero gradient"
        scores.append(num / den)
        if (i + 1) % 25 == 0:
            print(f"[W1-V] scored {i+1}/{len(batches)}", flush=True)
    for m in mods.values():
        m.weight.grad = None
    scores = np.asarray(scores)
    order = np.argsort(scores)
    n_drop = max(1, int(len(batches) * DROP_FRACTION))
    atlas.add_ci("core_attack_score", scores.tolist(), bank_id="core32", incumbent_id="none",
                 model_id=PG1_REPO, model_sha=base_sha,
                 flags={"definition": "‖g U‖²_F/‖g‖²_F for a single example's gradient",
                        "n_examples": len(batches), "n_drop": n_drop,
                        "drop_fraction": DROP_FRACTION,
                        "score_min": float(scores.min()), "score_max": float(scores.max())})

    # Phase 2 — retrain under matched removals. Every arm trains on the SAME number of examples
    # for the SAME number of steps; only WHICH decile left differs. The random arm is the size
    # control and the bottom arm is §3's structured (top-k vs bottom-k) alternative.
    rng = np.random.default_rng(a.seed)
    # The random decile is drawn ONCE and then used. Inlining `rng.choice(...)` inside the
    # comprehension's `if` re-draws it for every element, which would leave a kept set of
    # essentially random size — a control that is not the control it is labelled as.
    rand_drop = set(rng.choice(len(batches), n_drop, replace=False).tolist())
    keep_sets = {
        "drop_top_decile": order[:-n_drop].tolist(),
        "drop_bottom_decile": order[n_drop:].tolist(),
        "drop_random_decile": [i for i in range(len(batches)) if i not in rand_drop],
    }
    n_keep = len(keep_sets["drop_top_decile"])
    for name, idx in keep_sets.items():
        assert len(idx) == n_keep, f"{name} keeps {len(idx)}, not the matched {n_keep}"
    # The honest "all data" arm is a MATCHED-SIZE random subset, not the full set: comparing a
    # 90%-data arm against a 100%-data arm measures data size, which is the confound §4/W1-V's
    # control set exists to remove.
    keep_sets["keep_matched_size"] = rng.choice(len(batches), n_keep, replace=False).tolist()

    results = {}
    for name, idx in keep_sets.items():
        for k, m in mods.items():
            m.weight.data.copy_(pristine[k].to(m.weight.dtype))
        sub = [batches[i] for i in idx]
        hist, _ = train_arm(model, mods, sub, arm="W1-V", steps=a.steps, lr=a.lr, kind="adam")
        ppl = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])
        tl = eval_task_loss(model, ctx["eval_batches"])
        mot = motion_report(mods, pristine, banks, ["core32"])
        atlas.add("retention_ppl", value=ppl, bank_id="core32", dose=float(len(idx)),
                  incumbent_id="drop_random_decile", model_id=PG1_REPO, model_sha=base_sha,
                  flags={"arm_name": name, "n_kept": len(idx), "n_dropped": len(batches) - len(idx),
                         "ppl_base": ctx["base_ppl"], "retention_loss": ppl - ctx["base_ppl"],
                         "task_loss": tl, "core_motion": mot["core32"],
                         "mean_kept_score": float(scores[idx].mean()),
                         "train_loss_last": hist[-1],
                         "matched": "identical example count and identical step count across arms"})
        results[name] = {"ppl": ppl, "task_loss": tl, "n_kept": len(idx),
                         "retention_loss": ppl - ctx["base_ppl"],
                         "core_motion": mot["core32"],
                         "mean_kept_score": float(scores[idx].mean())}
        print(f"[W1-V] {name}: n={len(idx)} task={tl:.4f} ppl={ppl:.4f}", flush=True)
        torch.cuda.empty_cache()
        D.s3_sync(out, a.s3_prefix)

    for k, m in mods.items():
        m.weight.data.copy_(pristine[k].to(m.weight.dtype))
    top = results["drop_top_decile"]["retention_loss"]
    ctrl = min(results["drop_random_decile"]["retention_loss"],
               results["drop_bottom_decile"]["retention_loss"])
    ref_loss = results["keep_matched_size"]["retention_loss"]
    damaged = bool(ref_loss > DAMAGE_MIN_REL * ctx["base_ppl"])
    atlas.add("damage_positive_control", value=ref_loss, bank_id="none",
              incumbent_id="none", model_id=PG1_REPO, model_sha=base_sha,
              flags={"base_ppl": ctx["base_ppl"], "required": DAMAGE_MIN_REL * ctx["base_ppl"],
                     "damage_min_rel": DAMAGE_MIN_REL, "damaged": damaged,
                     "outcome": "DAMAGE-PRESENT" if damaged else "VOID-NO-DAMAGE",
                     "why": "removing core-attacking examples can only IMPROVE retention if the "
                            "fine-tune damages retention; with no damage every arm ties at zero "
                            "forgetting and the contrast measures nothing"})
    gates.record("W1V_damage_present", damaged, reference_retention_loss=ref_loss,
                 required=DAMAGE_MIN_REL * ctx["base_ppl"], base_ppl=ctx["base_ppl"])
    gates.record("W1V_conflict_arms_measured", True, arms=sorted(results),
                 n_examples=len(batches), n_drop=n_drop,
                 top_retention_loss=top, best_control_retention_loss=ctrl,
                 damage_present=damaged,
                 refuter_fires=bool(damaged and top >= ctrl))
    print(f"[W1-V] damage control: reference retention loss {ref_loss:+.4f} vs required "
          f"{DAMAGE_MIN_REL * ctx['base_ppl']:+.4f} -> "
          f"{'DAMAGE-PRESENT' if damaged else 'VOID-NO-DAMAGE'}", flush=True)
    return {"results": results, "n_examples": len(batches), "n_drop": n_drop,
            "score_mean": float(scores.mean()), "score_max": float(scores.max()),
            "damage_present": damaged, "reference_retention_loss": ref_loss,
            "verdict": ("refuter_fires" if (damaged and top >= ctrl)
                        else "measured" if damaged else "VOID-NO-DAMAGE"),
            "refuter_fires": bool(damaged and top >= ctrl)}


RUNNERS = {"W1-S": run_w1_s, "W1-V": run_w1_v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True, choices=sorted(RUNNERS))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--s3-prefix", default=None)
    ap.add_argument("--types", default="q_proj,o_proj")
    ap.add_argument("--layers", default="")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=32)
    ap.add_argument("--n-seq", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    a.run_id = a.run_id or f"darwin_{a.arm.lower().replace('-', '_')}"
    a.s3_prefix = a.s3_prefix or f"{D.PROGRAM}/{a.run_id}"
    types = tuple(t.strip() for t in a.types.split(",") if t.strip())
    layers = [int(x) for x in a.layers.split(",") if x.strip()] or list(range(H.N_LAYERS))
    if a.smoke:
        a.steps, a.n_train, a.n_eval, a.n_seq = 4, 12, 4, 2
        layers = layers[:2]

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    H.hf_setup()

    import torch
    from huggingface_hub import HfApi
    from transformers import AutoProcessor
    from infra.aws_pg.jobs.invariant_core_func import (PG1_REPO, _decoder_layers, _load_full_pg1,
                                                       _ppl_blankimg)
    from infra.aws_pg.jobs.darwin_w1_u import build_batches, text_task_rows

    utc = D.utc_now()
    exp = D.Expectations(out.parent / "expectations.jsonl")
    if not exp.by_arm(a.arm):
        refuter, direction = SPEC_EXPECTATIONS[a.arm]
        exp.log(a.arm, refuter=refuter, direction=direction, utc=utc)
    ref = exp.require(a.arm)

    base_sha = HfApi().model_info(PG1_REPO).sha
    bank_dir = D.fetch_all_banks(H.cache_dir(out, "banks"))
    gates = H.Gates(out)
    atlas = D.Atlas(out, arm=a.arm, run_id=a.run_id, register="ATLAS", wave="W1",
                    expected_ref=ref)
    manifest = {"program": D.PROGRAM, "arm": a.arm, "run_id": a.run_id, "register": "ATLAS",
                "wave": "W1", "commit_sha": D.commit_sha(), "provenance": D.provenance(),
                "base_repo": PG1_REPO, "base_sha": base_sha, "types": list(types),
                "layers": layers, "steps": a.steps, "lr": a.lr,
                "breathe_ladder": list(BREATHE_LADDER), "drop_fraction": DROP_FRACTION,
                "optimizer": "exact fp32 Adam (§0.2.7)",
                "expected_ref": ref, "started_utc": utc, "seed": a.seed}
    D.write_json(out, "manifest.json", manifest)

    model = _load_full_pg1(base_sha, "cuda")
    processor = AutoProcessor.from_pretrained(PG1_REPO, revision=base_sha)
    dl = _decoder_layers(model)
    mods = T.trainable_modules(dl, types=types, layers=layers)
    pristine = {k: m.weight.detach().to(torch.float32).clone() for k, m in mods.items()}
    banks = {k: {bid: W.load_bank(bank_dir, k[0], k[1], bid) for bid in D.bank_ids_for(k[0])}
             for k in mods}
    train_rows, eval_rows = text_task_rows(a.n_train, a.n_eval, seed=a.seed)
    ctx = {"mods": mods, "pristine": pristine, "banks": banks,
           "train_batches": build_batches(processor, train_rows, "cuda"),
           "eval_batches": build_batches(processor, eval_rows, "cuda"),
           "base_ppl": float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])}
    print(f"[{a.arm}] base ppl={ctx['base_ppl']:.5f}", flush=True)

    result = RUNNERS[a.arm](model, processor, dl, bank_dir, out, atlas, gates, a, base_sha, ctx)

    atlas.close()
    rep = atlas.to_parquet()
    D.write_json(out, f"{a.arm.lower().replace('-', '_')}_summary.json",
                 {"manifest": manifest, "atlas": rep, "gates": gates.gates, "result": result,
                  "base_ppl": ctx["base_ppl"], "finished_utc": D.utc_now()})
    D.s3_sync(out, a.s3_prefix)
    print("=" * 78)
    print(f"DARWIN_{a.arm.replace('-', '_')}_OK rows={rep['n_rows']} "
          f"{json.dumps(result, default=str)[:300]}")


if __name__ == "__main__":
    main()
