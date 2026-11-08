"""M-AB stage ii — context burning (W1-A) and the ICL amplifier (W1-B), on the task that passed.

DARWIN recorded W1-A and W1-B **VOID-INSTRUMENT**: both were run on MMLU, which shows no real
in-context lift on this 2B decoder, so neither arm could have detected anything. The ICL-sink causal
question was never refuted — it was never tested. Stage i fixed the missing half by screening tasks
with power and found one that lifts (+0.406 CI [+0.302, +0.510] on in-context copy, against a
control task that does not lift). Stage ii is the arms themselves, re-run where they can see.

    BURN (W1-A). Train a per-layer bias confined to a 32-dimensional bank, b_L = U_L c_L, so the
    0-shot model reproduces the max-shot model's next-token distribution on UNLABELED queries. Then
    score 0-shot + burn against the shot ladder. If the sink is the channel in-context information
    travels on, writing into it should buy back some of the lift.
    REFUTER: the in-core burn is no better than its own twin burns.

    AMPLIFY (W1-B). Take the measured context shift at the mid rung, h_mid - h_0, over the query
    block, project it onto a bank and scale it by g. Does amplifying the core-projected component
    ALONE move the mid-shot model toward the max-shot model?
    REFUTER: twin gains do the same.

WHAT THIS RUN CARRIES THAT DARWIN'S COULD NOT:

  * A READOUT WITH HEADROOM, established in advance by stage i rather than assumed, and re-measured
    here. The shot-ladder gate is re-run in this job on this job's items, and a failure makes the
    arms VOID-INSTRUMENT again — explicitly, with the measured lift on the row, instead of silently.
  * THE MATCHED-CAPACITY INCUMBENT. `freecoord32` is a bias free to take any value on 32 randomly
    chosen residual coordinates per layer: the same 576 parameters, none of the bank geometry. A
    burn that beats twins but not freecoord32 has shown that 576 parameters help, not that the
    SUBSPACE does.
  * A DOSE. The burn is trained once per bank at unit strength and EVALUATED at lam in
    {0.25, 0.5, 1, 2, 4}, a monotone knob on a trained vector, which is what §3 asks for at the
    cost of five evaluations rather than five trainings.

THE INSTRUMENTS ARE DARWIN'S, IMPORTED NOT REWRITTEN — `ConfinedBias`, `ContextAmplifier`,
`banks_by_layer`, `freecoords_by_layer`, `paired_acc_ci`. Only the DATA LAYER differs, which is the
one thing stage i showed had to change. Re-implementing the hooks would have made a difference in
plumbing indistinguishable from a difference in result.

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

ARM = "M-AB"
FULL_BATTERY = ("core32", "anti32", "topsigma32", "bottomsigma32", "complement32",
                "haar32_s0", "haar32_s1", "haar32_s2")
INCUMBENT = "freecoord32"
SHOT_LADDER = (0, 2, 4)          # 0 = the burn's student, 2 = the amplifier's source, 4 = target
LAM_LADDER = (0.25, 0.5, 1.0, 2.0, 4.0)
GAIN_LADDER = (1.0, 1.5, 2.0, 4.0)
# Stage i's own threshold, reused so the two stages agree on what "a task that lifts" means.
MIN_USEFUL_LIFT = 0.10
CORPUS = "synthetic:copy"


def main():
    ap = argparse.ArgumentParser(description="M-AB stage ii — burn + amplifier on the copy task")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-items", type=int, default=96)
    ap.add_argument("--n-burn-queries", type=int, default=192)
    ap.add_argument("--burn-steps", type=int, default=150)
    ap.add_argument("--burn-batch", type=int, default=8)
    ap.add_argument("--burn-lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = P.ARMS[ARM]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_w1") / "expectations.jsonl")
    expected_ref = exp.require(ARM)
    at = M.Atlas(out, arm=ARM, run_id="m_ab_stage2", register=meta["register"], wave=meta["wave"],
                 expected_ref=expected_ref, queue_item=meta["queue_item"])
    sur = M.SurpriseLog(out)
    gates = H.Gates(out)

    import torch
    from huggingface_hub import HfApi
    from transformers import PaliGemmaProcessor
    from infra.pg import humboldt_act as ACT
    from infra.aws_pg.jobs.invariant_core_func import (_blank_image, _load_full_pg1,
                                                       _decoder_layers, _locate_text_block)
    from infra.aws_pg.jobs.darwin_w1_icl import (SITE, ConfinedBias, ContextAmplifier,
                                                 banks_by_layer, freecoords_by_layer,
                                                 paired_acc_ci)
    from infra.aws_pg.jobs.mercator_ab import build_prompt, score_exact

    assert torch.cuda.is_available(), "no CUDA"
    M.check_dose_grid(list(LAM_LADDER), arm=ARM)
    api = HfApi()
    base_sha = api.model_info(WT.PG1_REPO).sha
    model = _load_full_pg1(base_sha, "cuda")
    processor = PaliGemmaProcessor.from_pretrained(WT.PG1_REPO, revision=base_sha)
    tok = processor.tokenizer
    layers = _decoder_layers(model)
    bank_layers = list(range(len(layers)))
    blank = _blank_image(128)
    bank_dir = M.fetch_all_banks(H.cache_dir(out, "banks"))

    # Stage i's corpus, same seed, so "the task that passed" is literally the task that passed.
    n_pool = max(SHOT_LADDER)
    corpus = CP.copy_corpus(a.n_items + a.n_burn_queries + n_pool, seed=20260619)
    items = corpus[: a.n_items]
    burn_queries = corpus[a.n_items: a.n_items + a.n_burn_queries]
    pool = corpus[a.n_items + a.n_burn_queries:]
    rng = np.random.default_rng(a.seed)
    # ONE fixed shot draw per (item, n_shots), reused by every arm. Re-drawing per arm would put
    # demonstration variance into every contrast the arm reports.
    shots_for = {(i, n): [pool[j] for j in rng.choice(len(pool), size=n, replace=False)]
                 for n in SHOT_LADDER for i in range(max(len(items), len(burn_queries)))}

    def prompt(it, i, n):
        return build_prompt(it, shots_for[(i, n)] if n else [])

    def encode(text):
        """The three tensors PG1 is fed, and only those.

        The processor also emits `token_type_ids`, which PaliGemma's text-only forward must NOT
        receive — passing it changes which positions the model treats as suffix and silently moves
        the logits (house reference: the PaliGemma text-only logits gotcha).
        """
        inp = processor(text=[text], images=[blank], return_tensors="pt", padding=False).to("cuda")
        return {"input_ids": inp["input_ids"], "attention_mask": inp["attention_mask"],
                "pixel_values": inp["pixel_values"].to(next(model.parameters()).dtype)}

    def last_logits(inp, ctx=None):
        from contextlib import nullcontext
        with (ctx if ctx is not None else nullcontext()), torch.inference_mode():
            return model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                         pixel_values=inp["pixel_values"], logits_to_keep=1).logits[:, -1, :]

    # THE READOUT IS A CONSTRAINED ARGMAX over the corpus's own answer alphabet, which is DARWIN's
    # `choice_letter_ids` pattern. A free argmax at one position can land on whitespace and score 0
    # for an answer the model would in fact have produced — a systematic depression that would hit
    # the intervened arms and the reference ladder unequally, since the interventions perturb the
    # very distribution being read.
    #
    # A LETTER HAS MORE THAN ONE SPELLING. PaliGemma's processor APPENDS a trailing newline, so the
    # token after the prompt follows "->\n" and is the bare letter, while inside the prompt the
    # demonstrations spell it " a" with a leading space. Constraining to one spelling asks the model
    # for a token it may never emit and depresses accuracy — the first run of this stage did exactly
    # that and read 0.25 at 4 shots where stage i's free generation read far higher, tripping the
    # readout gate on the instrument rather than on the model (defect M-D24). A letter's score is
    # therefore the MAX over its single-token spellings.
    letters = list(CP._LETTERS)
    spellings = {}
    for ch in letters:
        ids = []
        for form in (ch, f" {ch}"):
            e = tok.encode(form, add_special_tokens=False)
            if len(e) == 1:
                ids.append(e[0])
        assert ids, f"no single-token spelling of answer {ch!r}"
        spellings[ch] = sorted(set(ids))
    flat = [i for ch in letters for i in spellings[ch]]
    assert len(set(flat)) == len(flat), "two letters share a token id"
    span = max(len(v) for v in spellings.values())
    # a (26, span) id matrix, padded by repetition so a max over axis 1 is well defined
    id_mat = torch.as_tensor([[v[min(j, len(v) - 1)] for j in range(span)]
                              for ch in letters for v in (spellings[ch],)], device="cuda")
    READOUT = ("constrained argmax over the answer alphabet, max over each letter's single-token "
               "spellings (bare and space-prefixed)")

    def score(it, i, n, ctx=None):
        """Correctness at the answer position under an optional intervention."""
        logits = last_logits(encode(prompt(it, i, n)), ctx)
        pred = letters[int(logits[0][id_mat].max(dim=1).values.argmax())]
        return score_exact(pred, it[1])

    def free_score(it, i, n):
        """Stage i's readout, verbatim: greedy generation and an exact match on the first token."""
        inp = encode(prompt(it, i, n))
        with torch.inference_mode():
            gen = model.generate(**inp, max_new_tokens=4, do_sample=False)
        text = tok.decode(gen[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        return score_exact(text, it[1])

    # ---- the readout must agree with the one stage i validated --------------------------------
    # Run BOTH readouts on the max-shot rung and compare. The constrained readout is used
    # everywhere below because the interventions perturb the distribution being read and free
    # generation would answer differently for reasons that are not the intervention; but a
    # constrained readout that DISAGREES with free generation is not a better version of stage i's
    # measurement, it is a different one, and that has to fail here rather than downstream.
    n_agree = max(SHOT_LADDER)
    constrained = np.array([score(it, i, n_agree) for i, it in enumerate(items)],
                           dtype=np.float64)
    free = np.array([free_score(it, i, n_agree) for i, it in enumerate(items)], dtype=np.float64)
    agreement = float((constrained == free).mean())
    at.add("readout_agreement", value=agreement, dose=float(n_agree), corpus_id=CORPUS,
           n=len(items), model_id=WT.PG1_REPO, model_sha=base_sha,
           flags={"constrained_acc": float(constrained.mean()), "free_acc": float(free.mean()),
                  "readout": READOUT, "reference": "stage i free generation"})
    gates.record("ab2_readout_agrees_with_stage_i", agreement >= 0.90, agreement=agreement,
                 constrained_acc=float(constrained.mean()), free_acc=float(free.mean()),
                 note=("a constrained readout that disagrees with free generation is a different "
                       "measurement, not a cleaner one"))
    assert agreement >= 0.90, (
        f"the constrained readout agrees with stage i's free generation on only "
        f"{agreement:.1%} of items (constrained {constrained.mean():.4f} vs free "
        f"{free.mean():.4f}); the readout is measuring something else and every contrast below "
        f"would inherit that")
    print(f"[M-AB2] readout agreement with stage i: {agreement:.3f} "
          f"(constrained {constrained.mean():.4f}, free {free.mean():.4f})", flush=True)

    # ---- the readout gate, re-measured on THIS job's items ------------------------------------
    ladder = {n: (constrained if n == n_agree else
                  np.array([score(it, i, n) for i, it in enumerate(items)], dtype=np.float64))
              for n in SHOT_LADDER}
    for n in SHOT_LADDER:
        s = at.add_ci("icl_accuracy", ladder[n], dose=float(max(n, 0.5)), corpus_id=CORPUS,
                      model_id=WT.PG1_REPO, model_sha=base_sha,
                      flags={"n_shot": n, "role": "reference ladder", "readout": READOUT})
        print(f"[M-AB2] {n}-shot acc={s['mean']:.4f}", flush=True)
    lift = paired_acc_ci(ladder[max(SHOT_LADDER)], ladder[0], seed=a.seed)
    readout_valid = bool(lift["ci_lo"] > MIN_USEFUL_LIFT)
    at.add("icl_ladder_lift", value=lift["contrast"], ci_lo=lift["ci_lo"], ci_hi=lift["ci_hi"],
           n=lift["n"], corpus_id=CORPUS, model_id=WT.PG1_REPO, model_sha=base_sha,
           flags={**lift, "min_useful_lift": MIN_USEFUL_LIFT, "valid": readout_valid,
                  "readout": READOUT})
    gates.record("ab2_readout_has_lift", readout_valid, **lift,
                 min_useful_lift=MIN_USEFUL_LIFT,
                 outcome=None if readout_valid else M.VOID_INSTRUMENT,
                 note=("both arms are DEFINED against the max-shot model; without a real ladder "
                       "their central quantities are ratios with a meaningless denominator, "
                       "which is exactly why DARWIN's W1-A/B were VOID-INSTRUMENT"))
    assert readout_valid, (
        f"the shot ladder does not lift on this job's items (lift {lift['contrast']:+.4f}, CI low "
        f"{lift['ci_lo']:+.4f} <= {MIN_USEFUL_LIFT}); running the burn and amplifier here would "
        f"repeat DARWIN's VOID-INSTRUMENT, and the gate row above records that it was measured")
    gap = float(ladder[max(SHOT_LADDER)].mean() - ladder[0].mean())

    # ---- ARM A: the burn ---------------------------------------------------------------------
    teacher = []
    for i, q in enumerate(burn_queries):
        lg = last_logits(encode(prompt(q, i, max(SHOT_LADDER))))
        teacher.append(torch.log_softmax(lg.float(), dim=-1)[0].to(torch.float16).cpu())
    teacher = torch.stack(teacher)
    print(f"[M-AB2] teacher distribution captured on {len(teacher)} unlabeled queries", flush=True)

    def train_burn(bias):
        """Fit `bias.c` so the 0-shot forward reproduces the teacher's next-token distribution.

        The model is frozen; the only tensors with gradients are the 576 coefficients. The
        objective never reads a gold answer — it matches the max-shot model's own behaviour, which
        is what 'unlabeled' means here.
        """
        opt = torch.optim.Adam(bias.parameters(), lr=a.burn_lr)
        hist = []
        with bias:
            for step in range(a.burn_steps):
                idx = [(step * a.burn_batch + j) % len(burn_queries) for j in range(a.burn_batch)]
                loss = 0.0
                for j in idx:
                    lg = model(**encode(prompt(burn_queries[j], j, 0)),
                               logits_to_keep=1).logits[:, -1, :]
                    student = torch.log_softmax(lg.float(), dim=-1)
                    t = teacher[j].to("cuda").float().unsqueeze(0)
                    loss = loss + torch.sum(torch.exp(t) * (t - student), dim=-1).mean()
                loss = loss / len(idx)
                opt.zero_grad()
                loss.backward()
                opt.step()
                hist.append(float(loss.detach()))
                if (step + 1) % 25 == 0:
                    print(f"    burn step {step+1}/{a.burn_steps} kl={hist[-1]:.5f}", flush=True)
        return hist

    burn_rows = {}
    for bid in FULL_BATTERY + (INCUMBENT,):
        if bid == INCUMBENT:
            bias = ConfinedBias(layers, "cuda", coords_by_layer=freecoords_by_layer(
                int(layers[0].self_attn.q_proj.weight.shape[1]), 32, bank_layers, a.seed))
        else:
            bias = ConfinedBias(layers, "cuda",
                                basis_by_layer=banks_by_layer(bank_dir, bid, bank_layers))
        hist = train_burn(bias)
        per_lam = {}
        for lam in LAM_LADDER:
            bias.scale = float(lam)
            acc = np.array([score(it, i, 0, ctx=bias) for i, it in enumerate(items)],
                           dtype=np.float64)
            c = paired_acc_ci(acc, ladder[0], seed=a.seed)
            per_lam[lam] = {"acc": float(acc.mean()), **c,
                            "recovered_fraction": c["contrast"] / gap if gap else float("nan")}
            at.add("burn_acc", value=float(acc.mean()), dose=float(lam), bank_id=bid,
                   corpus_id=CORPUS, n=len(items), ci_lo=c["ci_lo"], ci_hi=c["ci_hi"],
                   model_id=WT.PG1_REPO, model_sha=base_sha,
                   flags={**per_lam[lam], "bank": bid, "n_params": bias.n_params,
                          "final_kl": hist[-1], "site": SITE, "gap_to_max_shot": gap,
                          "readout": READOUT})
        bias.scale = 1.0
        best = max(per_lam, key=lambda L: per_lam[L]["contrast"])
        burn_rows[bid] = {"by_lam": per_lam, "best_lam": best, "kl_final": hist[-1],
                          "kl_first": hist[0], "n_params": bias.n_params}
        print(f"[M-AB2] burn {bid:14s} kl {hist[0]:.4f}->{hist[-1]:.4f} best lam={best} "
              f"contrast={per_lam[best]['contrast']:+.4f} "
              f"recovered={per_lam[best]['recovered_fraction']:+.3f}", flush=True)

    # ---- ARM B: the amplifier ----------------------------------------------------------------
    src, dst = SHOT_LADDER[1], max(SHOT_LADDER)
    h0_by_item, win_by_item = {}, {}
    for i, it in enumerate(items):
        block = f"{it[0]} ->"
        inp = encode(prompt(it, i, 0))
        win_ids = torch.as_tensor(tok.encode(block, add_special_tokens=False))
        s0 = _locate_text_block(inp["input_ids"][0], win_ids)
        n0 = int(win_ids.shape[0])
        slicer = ACT.SiteSlice(layers, "cuda", sites=(SITE,))
        with slicer, torch.inference_mode():
            slicer.set_window(s0, n0)
            model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                  pixel_values=inp["pixel_values"], logits_to_keep=1)
        h0_by_item[i] = {L: slicer.buf[(L, SITE)] for L in bank_layers}
        inp_s = encode(prompt(it, i, src))
        win_by_item[i] = (_locate_text_block(inp_s["input_ids"][0], win_ids), n0)
    print(f"[M-AB2] query-block activations captured for {len(items)} items", flush=True)

    amp_rows = {}
    for bid in FULL_BATTERY:
        basis = banks_by_layer(bank_dir, bid, bank_layers)
        per_gain = {}
        for g in GAIN_LADDER:
            amp = ContextAmplifier(layers, "cuda", basis, g)
            acc = []
            with amp:
                for i, it in enumerate(items):
                    s, n = win_by_item[i]
                    amp.set_item(h0_by_item[i], s, n)
                    acc.append(score(it, i, src))
            acc = np.array(acc, dtype=np.float64)
            c = paired_acc_ci(acc, ladder[src], seed=a.seed)
            room = float(ladder[dst].mean() - ladder[src].mean())
            per_gain[g] = {"acc": float(acc.mean()), **c,
                           "closed_fraction": c["contrast"] / room if room else float("nan")}
            at.add("amplify_acc", value=float(acc.mean()), dose=float(g), bank_id=bid,
                   corpus_id=CORPUS, n=len(items), ci_lo=c["ci_lo"], ci_hi=c["ci_hi"],
                   model_id=WT.PG1_REPO, model_sha=base_sha,
                   flags={**per_gain[g], "bank": bid, "gain": g, "source_shots": src,
                          "target_shots": dst, "room_to_target": room, "site": SITE,
                          "readout": READOUT})
        best = max(per_gain, key=lambda G_: per_gain[G_]["contrast"])
        amp_rows[bid] = {"by_gain": per_gain, "best_gain": best}
        print(f"[M-AB2] amplify {bid:14s} best g={best} contrast="
              f"{per_gain[best]['contrast']:+.4f} "
              f"closed={per_gain[best]['closed_fraction']:+.3f}", flush=True)

    # ---- the two refuters --------------------------------------------------------------------
    at.check_controls_per_statistic(["burn_acc", "amplify_acc"])
    at.check_readout_corpus(["icl_accuracy", "icl_ladder_lift", "burn_acc", "amplify_acc",
                             "readout_agreement"])

    def best_contrast(rows, bid):
        key = "by_lam" if "by_lam" in rows[bid] else "by_gain"
        return max(v["contrast"] for v in rows[bid][key].values())

    burn_twins = [best_contrast(burn_rows, b) for b in ("haar32_s0", "haar32_s1", "haar32_s2")]
    burn_core = best_contrast(burn_rows, "core32")
    gates.record("ab2_burn_core_beats_twins", burn_core > max(burn_twins),
                 core=burn_core, twin_band=[min(burn_twins), max(burn_twins)],
                 incumbent_freecoord32=best_contrast(burn_rows, INCUMBENT),
                 note=("beating the twins but not freecoord32 would show that 576 parameters "
                       "help, not that the SUBSPACE does"))
    gates.record("ab2_burn_core_beats_matched_incumbent",
                 burn_core > best_contrast(burn_rows, INCUMBENT),
                 core=burn_core, incumbent=best_contrast(burn_rows, INCUMBENT))
    amp_twins = [best_contrast(amp_rows, b) for b in ("haar32_s0", "haar32_s1", "haar32_s2")]
    amp_core = best_contrast(amp_rows, "core32")
    gates.record("ab2_amplifier_core_beats_twins", amp_core > max(amp_twins),
                 core=amp_core, twin_band=[min(amp_twins), max(amp_twins)],
                 note="W1-B's refuter is that twin gains do the same")
    for name, val, twins in (("burn_acc/core32", burn_core, burn_twins),
                             ("amplify_acc/core32", amp_core, amp_twins)):
        sur.add(arm=ARM, expected_ref=expected_ref, cell=name.split("/")[0],
                statistic=name, s=M.surprise(val, float(np.mean(twins)), twins))

    info = at.to_parquet()
    sur.to_parquet()
    M.write_json(out, f"{ARM}_stage2_result.json",
                 {"arm": ARM, "stage": "ii (context burning + ICL amplifier)",
                  "queue_item": meta["queue_item"], "refuter": meta["refuter"],
                  "expected_ref": expected_ref, "provenance": M.provenance(), "utc": M.utc_now(),
                  "atlas": info, "base_repo": WT.PG1_REPO, "base_sha": base_sha,
                  "corpus": CORPUS, "readout": READOUT, "n_items": len(items), "n_burn_queries": len(burn_queries),
                  "shot_ladder": list(SHOT_LADDER), "lam_ladder": list(LAM_LADDER),
                  "gain_ladder": list(GAIN_LADDER), "site": SITE,
                  "ladder_accuracy": {n: float(v.mean()) for n, v in ladder.items()},
                  "ladder_lift": lift, "readout_valid": readout_valid,
                  "readout_agreement": agreement,
                  "gap_to_max_shot": gap, "burn": burn_rows, "amplify": amp_rows,
                  "burn_core_vs_twins": {"core": burn_core, "twins": burn_twins},
                  "amplify_core_vs_twins": {"core": amp_core, "twins": amp_twins},
                  "instruments": ("darwin_w1_icl.ConfinedBias / ContextAmplifier / "
                                  "banks_by_layer / freecoords_by_layer / paired_acc_ci, "
                                  "imported unchanged; only the data layer differs")})
    print("AB2_OK " + json.dumps({"rows": info["n_rows"], "burn_core": burn_core,
                                  "burn_twins": burn_twins, "amp_core": amp_core,
                                  "amp_twins": amp_twins}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
