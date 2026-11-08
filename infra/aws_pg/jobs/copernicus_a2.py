"""CO-A2 — is the release a mechanism, or optimizer degradation wearing a mechanism's name? (item 86)

Spec §2.2 / §6.2. The last named confound on MERCATOR's bedrock.

THE CLAIM UNDER TEST. Under a Hadamard gauge the core's delivery ratio RISES while the rest of the
battery falls. The deflationary alternative is that rotating simply damages optimization, and any
equally-damaged run would show the same thing. So this arm fine-tunes PG1 under degradation matched
to the hadamard arm's perplexity cost WITHOUT touching Adam's denominator, and asks whether the
hadamard replicas still sit ABOVE what that degradation predicts.

THE OUTCOME IS THE RELEASE SIGNATURE, S = log delivery(core32) − median log delivery(the other
seven banks), on NET delivery and on PATH delivery both. Raw core delivery would be satisfied by
any global rescaling; S is not, and the whole-battery-rise case is caught explicitly and voided.

THE LADDERS ARE NOT MATCHED, THEY ARE REGRESSED. Exact ppl matching across arms is unattainable and
would invite a fishing expedition; instead each ladder mechanism is fit separately (log S against
final ppl, linear, frozen) and the test is whether hadamard sits above the 95% PREDICTION interval
at its own ppl. If a ladder cannot bracket hadamard's ppl, or its interval is wider than the effect
it must adjudicate, that ladder is COVERAGE-IMPOSSIBLE — a named non-verdict, not a free pass.

WHAT IS NOT A CONTROL HERE, and why (§2.2, reviewed):
  * gradient noise — additive iid noise lifts v̂ by σ² everywhere, compressing the ratio structure
    of 1/√v̂. That is a diluted dose of the treatment, not a control for it.
  * weight decay — decoupled wd adds −lr·λ·W per step into the NET ΔW numerator with no counterpart
    in the gradient denominator (verified in ExactAdam.step), so its delivery reflects base-weight
    geometry. At the λ needed to match hadamard's ppl cost it dominates ΔW outright: a control that
    cannot show release.

RIDER: CO-B3 (item 24). Per-layer per-bank grad/motion rows are banked here at zero extra GPU —
the banked mq2 atlas carries layer=null on all 397 rows because the per-layer terms were summed
before banking, so the depth question needed exactly this three-line change and nothing more.

§0.2: ExactAdam (approximate Adam PROHIBITED — the second moment IS the object), fp32, TF32 off.

    pg run infra.aws_pg.jobs.copernicus_a2 \
        --args "--arms unrotated,hadamard,lr:2.5e-6,lr:5e-6,lr:2e-5,lr:4e-5,noise:1e-3,noise:3e-3,noise:1e-2" \
        --no-weights --type g6e.8xlarge,p5.4xlarge \
        --s3-prefix results/copernicus_a2 --budget-min 600
"""
import argparse
import json
import pathlib

import numpy as np

from infra.pg import copernicus_core as C
from infra.pg import humboldt_core as H
from infra.pg import darwin_train as T
from infra.pg import humboldt_weights as WT
from infra.pg import mercator_core as M
from infra.pg import mercator_gauge as G

ARM = "CO-A2"
QUEUE_ITEM = "86"
WAVE = "W0"
BANK = "core32"
MOTION_BANKS = ("core32", "anti32", "topsigma32", "bottomsigma32", "complement32",
                "haar32_s0", "haar32_s1", "haar32_s2")
PPL_CORPUS = "wikitext2:window512"
TASK_CORPUS = "codeparrot-clean-valid:completion200"
N_VHAT_CHECKPOINTS = 3


def orth(A):
    Q, _ = np.linalg.qr(np.asarray(A, dtype=np.float64))
    return Q[:, :A.shape[1]]


def parse_arms(spec: str):
    """'unrotated,hadamard,lr:5e-6,noise:1e-3' → ordered arm records. `unrotated` MUST come first:
    it is the reference every admissibility statistic is read against."""
    arms = []
    for tokspec in [t.strip() for t in spec.split(",") if t.strip()]:
        if ":" in tokspec:
            kind, val = tokspec.split(":", 1)
            assert kind in ("lr", "noise"), f"unknown ladder {kind!r}"
            arms.append({"name": tokspec, "kind": kind, "value": float(val)})
        else:
            assert tokspec in ("unrotated", "hadamard"), f"unknown arm {tokspec!r}"
            arms.append({"name": tokspec, "kind": tokspec, "value": None})
    assert arms and arms[0]["kind"] == "unrotated", "the unrotated reference must run first"
    assert any(a["kind"] == "hadamard" for a in arms), "no hadamard arm — nothing to adjudicate"
    return arms


def vhat_statistics(v_by_layer, U_by_layer, ref_cols=None):
    """The three AXIS-ALIGNED v̂ statistics of §2.2's denominator-intactness gate.

    The banks are input-space subspaces; v̂ is elementwise over (out, in). Those are different
    objects, which is why the round-1 review called the original gate ill-posed. These three are
    each well defined on v̂ as it actually exists:

      (i)   Spearman of the per-input-coordinate column profile v_col_j = Σ_i v̂_ij against the
            unrotated reference — does the denominator still rank coordinates the same way?
      (ii)  ELEMENTWISE σ²/μ² of v̂ restricted to each bank's coordinate support. Elementwise,
            never column-summed (M-D18): summing first hides exactly the dispersion being asked
            about.
      (iii) leverage-weighted share of v_col carried by the bank.
    """
    from scipy import stats
    out = {"spearman": {}, "dispersion": {}, "leverage_share": {}, "v_col": {}}
    for L, v in v_by_layer.items():
        v_col = np.asarray(v.sum(axis=0), dtype=np.float64)          # (n_in,)
        out["v_col"][L] = v_col
        if ref_cols is not None:
            out["spearman"][L] = float(stats.spearmanr(v_col, ref_cols[L]).statistic)
        for b, U in U_by_layer[L].items():
            lev = (np.asarray(U, dtype=np.float64) ** 2).sum(axis=1)  # (n_in,)
            k = U.shape[1]
            support = np.argsort(-lev)[:k]
            elems = np.asarray(v, dtype=np.float64)[:, support].ravel()
            mu = float(elems.mean())
            out["dispersion"].setdefault(L, {})[b] = float(elems.var() / (mu ** 2)) if mu > 0 else float("inf")
            tot = float(v_col.sum())
            out["leverage_share"].setdefault(L, {})[b] = float((lev * v_col).sum() / tot) if tot > 0 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default=("unrotated,hadamard,lr:2.5e-6,lr:5e-6,lr:2e-5,lr:4e-5,"
                                       "noise:1e-3,noise:3e-3,noise:1e-2"))
    ap.add_argument("--seeds", default="1337,1338,1339")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-5, help="the reference lr the ladder brackets")
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=32)
    ap.add_argument("--n-seq", type=int, default=32,
                    help="wikitext windows for final ppl; the ladder regression reads this number, "
                         "so it is raised well above mq2's 8")
    ap.add_argument("--n-gate-seq", type=int, default=4)
    ap.add_argument("--calibration", action="store_true",
                    help="v0.8 §A2 calibration sweep: 1 seed permitted, no thresholds written "
                         "(a single seed cannot form a band, and a calibration run must not be "
                         "mistakable for a scored one)")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = parse_arms(a.arms)
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    if a.calibration:
        assert len(seeds) >= 1, f"calibration needs at least one seed; got {seeds}"
    else:
        assert len(seeds) >= 3, f"the arm compares at matched seeds; got {seeds}"

    H.hf_setup()
    import torch
    from huggingface_hub import HfApi
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
    from infra.aws_pg.jobs.invariant_core_func import _blank_image, _decoder_layers
    # The final ppl is the regression's x-axis, so it must not be able to CRASH the run. The
    # anchor's `_ppl_blankimg` tokenises without the processor's appended newline and raises
    # whenever the tokenizer merges it (§15 v0.7 finding i); mq2 scored 8 windows and never hit
    # it, this arm scores several times more. CO-A1's corrected, dry-run-tested routine is reused
    # instead. Anchor continuity does not apply here: A2's anchor recompute is on delivery_ratio,
    # not on ppl.
    from infra.aws_pg.jobs.copernicus_a1 import build_corpus, ppl_windows
    from infra.aws_pg.jobs.darwin_w1_u import build_batches, eval_task_loss, text_task_rows
    from infra.aws_pg.jobs.mercator_mq import (_layers, _reference_logits, _untie_lm_head,
                                               fold_model, rotate_model, scale_gauge_model)

    assert torch.cuda.is_available(), "no CUDA"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    api = HfApi()
    base_sha = api.model_info(WT.PG1_REPO).sha
    processor = PaliGemmaProcessor.from_pretrained(WT.PG1_REPO, revision=base_sha)
    blank = _blank_image(128)
    gate_texts = ["The capital of France is Paris, and the capital of Italy is",
                  "In 1969 the first humans landed on the",
                  "Water boils at one hundred degrees",
                  "The quick brown fox jumps over the lazy"][: a.n_gate_seq]
    train_rows, eval_rows = text_task_rows(a.n_train, a.n_eval, seeds[0])
    ppl_all, corpus_info = build_corpus(processor, 512)
    ppl_wins = ppl_all[:a.n_seq]
    assert len(ppl_wins) == a.n_seq, f"corpus yielded {len(ppl_wins)} windows, need {a.n_seq}"

    def final_ppl(m):
        return float(C.ppl(ppl_windows(m, processor, "cuda", ppl_wins)))
    bank_dir = M.fetch_all_banks(H.cache_dir(out, "banks"))
    layers_idx = list(range(H.N_LAYERS))
    raw_banks = {L: {b: WT.load_bank(bank_dir, "q_proj", L, b) for b in MOTION_BANKS}
                 for L in layers_idx}
    optinfo = M.check_optimizer("M-Q", "darwin_train.ExactAdam", offload_exact_fp32=False)

    exp = C.Expectations(out / "copernicus_expectations.jsonl")
    exp.log(ARM, "primary", "director §11: MECHANISM-CLEAN — hadamard sits above both ladders' "
            "prediction intervals on the release signature S", confidence="mid",
            basis="spec §11 row CO-A2", family="mechanism")
    exp.log("CO-B3", "depth", "release strong in the interior (L1-L16), weak at L17",
            confidence="mid", basis="spec §11 row CO-B3; logged BEFORE the per-layer field exists",
            family="mechanism")
    at = C.Atlas(out, arm=ARM, run_id="copernicus_a2", wave=WAVE, queue_item=QUEUE_ITEM,
                 expected_ref=exp.require(ARM, "primary"))
    gates = C.Gates(out)

    def load():
        m = PaliGemmaForConditionalGeneration.from_pretrained(
            WT.PG1_REPO, revision=base_sha, torch_dtype=torch.float32, device_map="cuda",
            attn_implementation="eager")
        _untie_lm_head(m)
        return m

    # the ungauged reference logits + the MEASURED reassociation noise floor
    model = load()
    model.eval()
    ref_logits = _reference_logits(model, processor, "cuda", gate_texts, blank)
    _lm, mlayers = _layers(model)
    gains = {L: G.gemma_effective_gain(
        mlayers[L].input_layernorm.weight.detach().cpu().numpy()) for L in layers_idx}
    n_scaled = scale_gauge_model(model, G.REASSOCIATION_PROBE_SCALE)
    floor = G.function_preservation(
        ref_logits, _reference_logits(model, processor, "cuda", gate_texts, blank))
    assert floor["max_abs_diff"] > 0, "constant-scale gauge moved logits by exactly zero"
    del model, _lm, mlayers
    torch.cuda.empty_cache()
    print(f"[a2] reassociation noise floor = {floor['max_abs_diff']:.3e} ({n_scaled} norms)",
          flush=True)

    folded = {L: {b: orth(gains[L][:, None] * raw_banks[L][b]) for b in MOTION_BANKS}
              for L in layers_idx}
    d_model = folded[0][BANK].shape[0]

    rep = {"object": "copernicus_a2_v1", "arm": ARM, "program": C.PROGRAM,
           "queue_item": QUEUE_ITEM, "base_repo": WT.PG1_REPO, "base_sha": base_sha,
           "arms": [x["name"] for x in arms], "seeds": seeds, "steps": a.steps,
           "reference_lr": a.lr, "trainable": "q_proj x 18 layers", "optimizer": optinfo,
           "noise_floor": floor, "banks": list(MOTION_BANKS), "ppl_corpus": corpus_info,
           "calibration": bool(a.calibration),
           "excluded_controls": {
               "gradient_noise": "lifts v-hat by sigma^2 everywhere — a diluted dose of the treatment",
               "weight_decay": ("decoupled wd injects -lr*lambda*W into net dW with no gradient "
                                "counterpart (verified in ExactAdam.step); at the lambda needed to "
                                "match hadamard's ppl cost it dominates dW")},
           "provenance": C.provenance(), "utc": C.utc_now(), "rows": []}

    def save():
        (out / "copernicus_a2.json").write_text(json.dumps(rep, indent=2, sort_keys=True,
                                                           default=float))

    rows, per_layer_rows, ref_vcols, ref_stats, unrot_vcols = [], [], None, [], []
    for arec in arms:
        name, kind, val = arec["name"], arec["kind"], arec["value"]
        lr = val if kind == "lr" else a.lr
        sigma = val if kind == "noise" else 0.0
        model = load()
        model.eval()
        fold_info = fold_model(model)
        R = G.hadamard_rotation(d_model) if kind == "hadamard" else None
        rot_info = rotate_model(model, R) if R is not None else {"per_group": {}, "reader": 0,
                                                                 "writer": 0}
        fp = G.function_preservation(
            ref_logits, _reference_logits(model, processor, "cuda", gate_texts, blank))
        fp_gate = G.assert_function_preserved(fp, what=f"arm:{name}",
                                              noise_floor=floor["max_abs_diff"])
        # THE GAUGE GATE IS PER ARM. An arm whose function is not preserved is not a gauge, and a
        # fine-tune under it would be a fine-tune of a different model wearing this arm's label.
        # Scored against the MEASURED reassociation floor, never an absolute tolerance (M-Q lesson).
        gates.record(f"a2_function_preserved__{name}", bool(fp_gate.get("ok", True)),
                     arm_cell=name, logit_max_abs_diff=float(fp["max_abs_diff"]),
                     noise_floor=float(floor["max_abs_diff"]),
                     ratio_to_noise_floor=float(fp["max_abs_diff"] / floor["max_abs_diff"]),
                     rotation_applied=bool(R is not None),
                     n_norms_folded=int(fold_info["n_norms_folded"]),
                     rotated_groups=int(rot_info.get("reader", 0) + rot_info.get("writer", 0)),
                     band="measured constant-scale reassociation floor (§0.2 gauge rule)")
        print(f"[a2] {name}: fold ok ({fold_info['n_norms_folded']} norms) "
              f"logit max|d|={fp['max_abs_diff']:.3e} lr={lr:g} sigma={sigma:g}", flush=True)

        U_np = {L: {b: (folded[L][b] if R is None else orth(R @ folded[L][b]))
                    for b in MOTION_BANKS} for L in layers_idx}
        U = {L: {b: torch.as_tensor(U_np[L][b], device="cuda", dtype=torch.float32)
                 for b in MOTION_BANKS} for L in layers_idx}
        for p in model.parameters():
            p.requires_grad_(False)
        mods = T.trainable_modules(_decoder_layers(model), types=("q_proj",), layers=layers_idx)
        for mod in mods.values():
            mod.weight.requires_grad_(True)
        clean = {k: m.weight.detach().to(torch.float32).clone() for k, m in mods.items()}
        train_b = build_batches(processor, train_rows, "cuda")
        eval_b = build_batches(processor, eval_rows, "cuda")

        for seed in seeds:
            gen = torch.Generator(device="cuda").manual_seed(H.cell_seed(seed, ARM, name, "init"))
            for k, m in mods.items():
                m.weight.data.copy_(clean[k])
                if sigma > 0:
                    # INIT NOISE, applied ONCE, before the pristine snapshot — the perturbation is
                    # therefore OUTSIDE the measured dW, which is what keeps this a degradation
                    # control rather than an injected displacement.
                    w = m.weight.data
                    scale = sigma * float(w.norm()) / float(np.sqrt(w.numel()))
                    w.add_(torch.randn(w.shape, generator=gen, device=w.device,
                                       dtype=w.dtype) * scale)
            pristine = {k: m.weight.detach().to(torch.float32).clone() for k, m in mods.items()}
            keys = list(mods.keys())
            opt = T.ExactAdam([mods[k].weight for k in keys], lr=lr)
            rng = np.random.default_rng(H.cell_seed(seed, ARM, name, "stage2"))

            grad_in = {(L, b): 0.0 for L in layers_idx for b in MOTION_BANKS}
            grad_tot = {L: 0.0 for L in layers_idx}
            path_in = {(L, b): 0.0 for L in layers_idx for b in MOTION_BANKS}
            path_tot = {L: 0.0 for L in layers_idx}
            before = {k: torch.empty_like(mods[k].weight) for k in keys}
            vhat_ckpts = []
            ck_at = {max(1, int(a.steps * f)) for f in (0.34, 0.67, 1.0)}

            model.train()
            for step in range(1, a.steps + 1):
                batch = train_b[int(rng.integers(0, len(train_b)))]
                opt.zero_grad()
                model(input_ids=batch["input_ids"], labels=batch["labels"]).loss.backward()
                with torch.no_grad():
                    for (t, L) in keys:
                        g = mods[(t, L)].weight.grad.to(torch.float32)
                        grad_tot[L] += float((g * g).sum())
                        for b in MOTION_BANKS:
                            grad_in[(L, b)] += float((g @ U[L][b]).pow(2).sum())
                        before[(t, L)].copy_(mods[(t, L)].weight.detach())
                opt.step()
                with torch.no_grad():
                    # PATH share: the per-step displacement, accumulated. Net dW cancels the
                    # back-and-forth that this keeps, which is exactly item 54's distinction and
                    # why the verdict must hold on both.
                    for (t, L) in keys:
                        dstep = mods[(t, L)].weight.detach().to(torch.float32) - before[(t, L)]
                        path_tot[L] += float((dstep * dstep).sum())
                        for b in MOTION_BANKS:
                            path_in[(L, b)] += float((dstep @ U[L][b]).pow(2).sum())
                    if step in ck_at:
                        v_by_layer = {L: opt.v[i].detach().cpu().numpy()
                                      for i, (t, L) in enumerate(keys)}
                        vhat_ckpts.append(vhat_statistics(
                            v_by_layer, U_np,
                            ref_cols=(ref_vcols if ref_vcols is not None else None)))
            model.eval()
            ppl = final_ppl(model)
            task = eval_task_loss(model, eval_b)

            move_in = {(L, b): 0.0 for L in layers_idx for b in MOTION_BANKS}
            move_tot = {L: 0.0 for L in layers_idx}
            with torch.no_grad():
                for (t, L) in keys:
                    dW = mods[(t, L)].weight.detach().to(torch.float32) - pristine[(t, L)]
                    move_tot[L] += float((dW * dW).sum())
                    for b in MOTION_BANKS:
                        move_in[(L, b)] += float((dW @ U[L][b]).pow(2).sum())
            gt, mt, pt = sum(grad_tot.values()), sum(move_tot.values()), sum(path_tot.values())
            assert gt > 0 and mt > 0 and pt > 0, f"{name}/s{seed}: nothing moved"

            delivery_net, delivery_path = {}, {}
            for b in MOTION_BANKS:
                gs = sum(grad_in[(L, b)] for L in layers_idx) / gt
                ms = sum(move_in[(L, b)] for L in layers_idx) / mt
                ps = sum(path_in[(L, b)] for L in layers_idx) / pt
                delivery_net[b] = ms / gs
                delivery_path[b] = ps / gs
                rows.append({"arm": name, "kind": kind, "seed": seed, "bank": b, "lr": lr,
                             "sigma": sigma, "grad_share": gs, "motion_share": ms,
                             "path_share": ps, "delivery_ratio": ms / gs,
                             "delivery_path": ps / gs, "ppl": ppl, "task_loss": task})
                at.add("a2_delivery_ratio", value=ms / gs, bank_id=b, model_id=WT.PG1_REPO,
                       model_sha=base_sha, seed=seed, corpus_id=PPL_CORPUS,
                       flags={"arm_cell": name, "kind": kind, "lr": lr, "sigma": sigma,
                              "ppl": ppl, "task_loss": task, "grad_share": gs,
                              "motion_share": ms, "path_share": ps,
                              "delivery_path": ps / gs, "optimizer": optinfo,
                              "definition": "(|dW U|^2/|dW|^2) / (sum_t |g_t U|^2 / sum_t |g_t|^2)"})
                # CO-B3 rider: the same quantities, per layer, at zero extra GPU.
                for L in layers_idx:
                    if grad_tot[L] > 0 and move_tot[L] > 0:
                        gsl = grad_in[(L, b)] / grad_tot[L]
                        msl = move_in[(L, b)] / move_tot[L]
                        psl = path_in[(L, b)] / path_tot[L] if path_tot[L] > 0 else 0.0
                        per_layer_rows.append({"arm": name, "seed": seed, "bank": b, "layer": L,
                                               "grad_share": gsl, "motion_share": msl,
                                               "path_share": psl,
                                               "delivery_ratio": msl / gsl if gsl > 0 else None,
                                               "delivery_path": psl / gsl if gsl > 0 else None})
                        at.add("a2_delivery_ratio_by_layer",
                               value=(msl / gsl) if gsl > 0 else 0.0, bank_id=b, layer=L,
                               type="q_proj", model_id=WT.PG1_REPO, model_sha=base_sha, seed=seed,
                               flags={"arm_cell": name, "grad_share": gsl, "motion_share": msl,
                                      "path_share": psl, "rider": "CO-B3 (item 24)"})
            S_net = C.release_signature(delivery_net, BANK)
            S_path = C.release_signature(delivery_path, BANK)
            stat = {"arm": name, "kind": kind, "seed": seed, "lr": lr, "sigma": sigma, "ppl": ppl,
                    "task_loss": task, "S_net": S_net, "S_path": S_path,
                    "delivery_net": delivery_net, "delivery_path": delivery_path,
                    "vhat": {"n_checkpoints": len(vhat_ckpts),
                             "spearman_mean": (float(np.mean([np.mean(list(c["spearman"].values()))
                                                              for c in vhat_ckpts]))
                                               if vhat_ckpts and vhat_ckpts[-1]["spearman"] else None),
                             "dispersion_core_mean": float(np.mean(
                                 [np.mean([c["dispersion"][L][BANK] for L in layers_idx])
                                  for c in vhat_ckpts])),
                             "leverage_share_core_mean": float(np.mean(
                                 [np.mean([c["leverage_share"][L][BANK] for L in layers_idx])
                                  for c in vhat_ckpts]))}}
            if kind == "unrotated":
                ref_stats.append(stat)
            rep["rows"].append(stat)
            if kind == "unrotated":
                # Keep EVERY unrotated seed's column profile: the admissibility band for the
                # Spearman statistic is the reference arm's own seed-to-seed spread, and a band
                # needs the pairwise values, not one comparison against a single seed.
                unrot_vcols.append({L: vhat_ckpts[-1]["v_col"][L] for L in layers_idx})
                if ref_vcols is None:
                    ref_vcols = unrot_vcols[0]
            print(f"[a2] {name:14s} s{seed}: ppl={ppl:8.3f} task={task:.4f} "
                  f"S_net={S_net:+.4f} S_path={S_path:+.4f} "
                  f"core_delivery={delivery_net[BANK]:.4f}", flush=True)
            save()
        del model, mods, pristine, clean, train_b, eval_b, U, U_np, before
        torch.cuda.empty_cache()

        # §9.1b — A2 has no D/C split, so thresholds.json IS its blind. The unrotated reference
        # runs FIRST and its seed spread fixes the admissibility band; the file is written and
        # hashed HERE, before any ladder arm's delivery has been read.
        #
        # A CALIBRATION sweep (v0.8 §A2) writes no thresholds: one seed cannot form a band, and a
        # calibration artifact must not be mistakable for a scored one.
        if kind == "unrotated" and not a.calibration:
            # ρ_min is CALIBRATED, never invented: the reference band is the PAIRWISE
            # seed-to-seed Spearman of the unrotated replicas' column profiles. Three seeds give
            # three pairs, which is exactly the >=3 draws a band requires.
            from scipy import stats as _st
            pair_rho = []
            for i in range(len(unrot_vcols)):
                for j in range(i + 1, len(unrot_vcols)):
                    pair_rho.append(float(np.mean(
                        [_st.spearmanr(unrot_vcols[i][L], unrot_vcols[j][L]).statistic
                         for L in layers_idx])))
            thr = {
                "reference_arm": "unrotated", "seeds": seeds, "banks": list(MOTION_BANKS),
                "S_net": [s_["S_net"] for s_ in ref_stats],
                "S_path": [s_["S_path"] for s_ in ref_stats],
                "ppl": [s_["ppl"] for s_ in ref_stats],
                "delivery_net_core": [s_["delivery_net"][BANK] for s_ in ref_stats],
                "vhat_dispersion_core": [s_["vhat"]["dispersion_core_mean"] for s_ in ref_stats],
                "vhat_leverage_share_core": [s_["vhat"]["leverage_share_core_mean"]
                                             for s_ in ref_stats],
                "vhat_pairwise_spearman": pair_rho,
                "admissibility": {
                    "spearman_band": C.band(pair_rho),
                    "dispersion_band": C.band([s_["vhat"]["dispersion_core_mean"]
                                               for s_ in ref_stats]),
                    "leverage_share_band": C.band([s_["vhat"]["leverage_share_core_mean"]
                                                   for s_ in ref_stats])},
                "construction": ("every band is the unrotated replicas' OWN spread — pairwise "
                                 "seed-to-seed Spearman for rho, across-seed spread for the other "
                                 "two. No threshold is invented here. The composite gate MUST fail "
                                 "on hadamard (positive control, §6.2.3)")}
            rep["thresholds_sha256"] = C.write_thresholds(out / "thresholds.json", thr)
            rep["thresholds"] = thr
            save()
            print(f"[a2] thresholds FROZEN sha256={rep['thresholds_sha256'][:16]} "
                  f"(before any ladder delivery was read)", flush=True)

    (out / "a2_rows.json").write_text(json.dumps(rows, indent=1, default=float))
    (out / "a2_per_layer_rows.json").write_text(json.dumps(per_layer_rows, indent=1, default=float))
    rep["n_rows"] = len(rows)
    rep["n_per_layer_rows"] = len(per_layer_rows)
    save()
    at.close()
    at.to_parquet()
    print(f"COPERNICUS_A2_OK arms={len(arms)} seeds={len(seeds)} rows={len(rows)} "
          f"per_layer_rows={len(per_layer_rows)}", flush=True)


if __name__ == "__main__":
    main()
