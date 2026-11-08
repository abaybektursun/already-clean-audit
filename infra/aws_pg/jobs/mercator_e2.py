"""M-E2 — distillation-32: is the sink a sufficient teaching signal? (queue E2). Wave 1.

    THE QUESTION. If the consensus subspace is where a model keeps whatever fine-tunes refuse to
    disturb, then matching a teacher's activations INSIDE that 32-dimensional subspace should teach
    a student more per bit than matching its full hidden state, and possibly as much as matching
    its logits. Three objectives, one student, one budget.

        logits    KL(teacher || student) on the next-token distribution — the standard, and the
                  instrument's positive control: if THIS does not teach, nothing measured here
                  means anything.
        hidden    MSE between the student's hidden state (through a learned linear adapter) and the
                  teacher's, at a matched layer — the dense baseline.
        sink32    the same MSE, but only after projecting BOTH onto the teacher's 32-dim bank.
        twin32    identical to sink32 on a matched-random bank. THE control: a 32-dimensional
                  bottleneck may teach well simply because it is a bottleneck.

    REFUTER (pre-logged): sink-activation matching teaches no better than the twin projections at
    equal budget.

EQUAL BUDGET MEANS EQUAL EVERYTHING EXCEPT THE LOSS. Same student architecture, same
initialisation seed, same token stream, same step count, same optimizer. The only difference
between arms is the target the loss is computed against, which is what makes the comparison about
the SIGNAL rather than about the setup.

TWO VALIDITY GATES, BOTH BEFORE ANY COMPARISON IS READ:
  * the logits arm must beat the student's own initialisation by a real margin. A student that
    learns nothing under the known-good objective cannot discriminate between the others, and the
    arm records VOID-INSTRUMENT rather than a null.
  * the teacher must be better than the student's ceiling on the readout, or "distillation" is
    imitation of something the student could reach alone.

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

ARM = "M-E2"
OBJECTIVES = ("logits", "hidden", "sink32", "topenergy32", "twin32_s0", "twin32_s1")
TEACH_LAYER = 12                 # the teacher layer the hidden/sink objectives read
# Batch 1 throughout: the teacher's text block sits at a different offset in every
# window, so padding a batch would either misalign the slice or need a mask nobody
# here has verified. There is no throughput argument at this scale that outweighs that.
MIN_LEARNING_MARGIN = 0.05       # the logits arm must cut its own init ppl by at least this much


def student_config(vocab_size: int, d_model: int, n_layers: int, n_heads: int):
    """A small Gemma-shaped student. Gemma-shaped so the student's residual stream is the same
    KIND of object as the teacher's, which is what makes a projection between them meaningful
    rather than a coincidence of dimensions."""
    from transformers import GemmaConfig
    return GemmaConfig(vocab_size=vocab_size, hidden_size=d_model,
                       intermediate_size=4 * d_model, num_hidden_layers=n_layers,
                       num_attention_heads=n_heads, num_key_value_heads=n_heads,
                       head_dim=d_model // n_heads, max_position_embeddings=1024,
                       attn_implementation="eager")


def main():
    ap = argparse.ArgumentParser(description="M-E2 — distillation-32")
    ap.add_argument("--out", required=True)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-docs", type=int, default=512)
    ap.add_argument("--n-eval", type=int, default=32)
    ap.add_argument("--teach-layer", type=int, default=TEACH_LAYER)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ppl_corpus = f"wikitext2:window{a.seq_len}"
    meta = P.ARMS[ARM]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_w1") / "expectations.jsonl")
    expected_ref = exp.require(ARM)
    at = M.Atlas(out, arm=ARM, run_id="m_e2", register=meta["register"], wave=meta["wave"],
                 expected_ref=expected_ref, queue_item=meta["queue_item"])
    sur = M.SurpriseLog(out)
    gates = H.Gates(out)

    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import GemmaForCausalLM, PaliGemmaProcessor
    from infra.aws_pg.jobs.invariant_core_func import (_blank_image, _decoder_layers,
                                                       _load_full_pg1, _locate_text_block,
                                                       _processor_text_ids)

    assert torch.cuda.is_available(), "no CUDA"
    api = HfApi()
    base_sha = api.model_info(WT.PG1_REPO).sha
    teacher = _load_full_pg1(base_sha, "cuda").eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    processor = PaliGemmaProcessor.from_pretrained(WT.PG1_REPO, revision=base_sha)
    tok = processor.tokenizer
    dec = _decoder_layers(teacher)
    blank = _blank_image(128)
    d_teacher = int(dec[0].self_attn.q_proj.weight.shape[1])
    bank_dir = M.fetch_all_banks(H.cache_dir(out, "banks"))
    U = {"sink32": WT.load_bank(bank_dir, "q_proj", a.teach_layer, "core32"),
         "twin32_s0": WT.load_bank(bank_dir, "q_proj", a.teach_layer, "haar32_s0"),
         "twin32_s1": WT.load_bank(bank_dir, "q_proj", a.teach_layer, "haar32_s1")}
    Ut = {k: torch.as_tensor(v, device="cuda", dtype=torch.float32) for k, v in U.items()}

    # ---- the windows, built as STRINGS and self-aligned -----------------------------------------
    # THE TEACHER IS RUN THROUGH ITS IN-DISTRIBUTION MULTIMODAL FORWARD with a blank image, which
    # is the only PG1 text-scoring path this program has verified. That forward wraps the text as
    # [256 image tokens .. <bos> .. text .. trailing newline], so the teacher's positions do not
    # line up with the student's by construction — the text block is LOCATED by exact token-block
    # match and both models are then read over the same tokens. Feeding the teacher a bare
    # `input_ids` would be a forward nobody here has checked.
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    raw = "\n\n".join(t for t in ds["text"] if t.strip())
    ids_all = tok(raw, return_tensors="pt", add_special_tokens=False).input_ids[0]
    n_win = min(a.n_docs + a.n_eval, ids_all.shape[0] // a.seq_len)
    assert n_win > a.n_eval + 8, f"corpus gives only {n_win} windows of {a.seq_len}"

    windows = []
    for w in range(n_win):
        piece = ids_all[w * a.seq_len:(w + 1) * a.seq_len]
        text = tok.decode(piece, skip_special_tokens=True)
        # TOKENISE THE STRING THE PROCESSOR ACTUALLY BUILDS, newline included. PaliGemma's
        # processor wraps text as [image .. <bos> .. text .. "\n"], and that trailing newline MERGES
        # into the final token — so `tok(text)` and the processor's text block differ in exactly one
        # id, and the exact-block anchor fails with the two lengths looking right (256 sought inside
        # 256 image + bos + 256). Appending the newline here makes the two tokenisations identical
        # by construction rather than by luck, and the merged token is a real token the model
        # predicts, so the readout still covers exactly the processor's text block.
        win_ids = _processor_text_ids(tok, text)
        if win_ids.shape[0] < 16:
            continue
        inp = processor(text=[text], images=[blank], return_tensors="pt", padding=False)
        st = _locate_text_block(inp["input_ids"][0], win_ids)
        windows.append({"ids": win_ids, "start": int(st), "n": int(win_ids.shape[0]),
                        "input_ids": inp["input_ids"], "attention_mask": inp["attention_mask"],
                        "pixel_values": inp["pixel_values"].to(torch.bfloat16)})
    assert len(windows) > a.n_eval + 8, f"only {len(windows)} usable windows"
    train_w, eval_w = windows[: len(windows) - a.n_eval], windows[len(windows) - a.n_eval:]
    print(f"[M-E2] {len(train_w)} train windows, {len(eval_w)} eval, teacher d={d_teacher}",
          flush=True)

    def teacher_forward(win, want_logits: bool):
        """Teacher logits and layer-`teach_layer` residual, sliced to the located text block."""
        buf = {}

        def hook(_m, args):
            buf["h"] = args[0].detach().to(torch.float32)

        hd = dec[a.teach_layer].self_attn.q_proj.register_forward_pre_hook(hook)
        with torch.inference_mode():
            o = teacher(input_ids=win["input_ids"].to("cuda"),
                        attention_mask=win["attention_mask"].to("cuda"),
                        pixel_values=win["pixel_values"].to("cuda"))
        hd.remove()
        s0, n0 = win["start"], win["n"]
        h = buf["h"][:, s0:s0 + n0, :]
        lg = o.logits[:, s0:s0 + n0, :].to(torch.float32) if want_logits else None
        assert h.shape[1] == n0, f"teacher hidden slice {h.shape} != block length {n0}"
        return lg, h

    # The hidden states are FIXED, so they are computed once and cached; the logits are not cached
    # because one window's logit tensor is vocab-sized and the corpus would not fit in memory.
    t_hidden = {}
    for i, win in enumerate(train_w + eval_w):
        _, h = teacher_forward(win, want_logits=False)
        t_hidden[id(win)] = h.to(torch.float16).cpu()
        if (i + 1) % 100 == 0:
            print(f"    cached teacher hidden {i+1}/{len(windows)}", flush=True)

    # THE STUDENT'S VOCABULARY IS THE TEACHER'S LOGIT WIDTH, not the tokenizer's length. PaliGemma
    # pads its output layer (257216) well past the tokens the tokenizer actually has (257153), and
    # a KL between two distributions is only defined when they are over the same index set — built
    # from len(tok) the student produced 257153 columns against the teacher's 257216 and the loss
    # failed to broadcast. The padded slots are real outputs the teacher assigns mass to; dropping
    # them would silently renormalise the target.
    v_teacher = int(teacher.config.text_config.vocab_size)
    assert v_teacher >= len(tok), (
        f"teacher logit width {v_teacher} is narrower than the tokenizer ({len(tok)}); the "
        f"student cannot be built over a vocabulary the teacher does not span")

    def fresh_student():
        torch.manual_seed(a.seed)
        cfg = student_config(v_teacher, a.d_model, a.n_layers, a.n_heads)
        return GemmaForCausalLM(cfg).to("cuda")

    # The student layer whose residual is matched: the same FRACTIONAL depth as the teacher's, so
    # a 4-layer student reading its layer 2 is at the same relative position as an 18-layer
    # teacher read at layer 12.
    S_LAYER = min(a.n_layers - 1, max(0, round(a.teach_layer * a.n_layers / len(dec))))

    def student_forward(model, win_ids):
        buf = {}

        def hook(_m, args):
            buf["h"] = args[0]

        hd = model.model.layers[S_LAYER].self_attn.q_proj.register_forward_pre_hook(hook)
        o = model(input_ids=win_ids)
        hd.remove()
        return o.logits, buf["h"]

    @torch.no_grad()
    def student_ppl(model):
        model.eval()
        tot, n = 0.0, 0
        for win in eval_w:
            b = win["ids"].unsqueeze(0).to("cuda")
            lg = model(input_ids=b).logits[:, :-1].to(torch.float32)
            tgt = b[:, 1:]
            tot += float(torch.nn.functional.cross_entropy(
                lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="sum"))
            n += int(tgt.numel())
        model.train()
        return float(np.exp(tot / n))

    @torch.no_grad()
    def teacher_ppl():
        tot, n = 0.0, 0
        for win in eval_w:
            lg, _ = teacher_forward(win, want_logits=True)
            tgt = win["ids"].unsqueeze(0)[:, 1:].to("cuda")
            tot += float(torch.nn.functional.cross_entropy(
                lg[:, :-1].reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="sum"))
            n += int(tgt.numel())
        return float(np.exp(tot / n))

    t_ppl = teacher_ppl()
    init_ppl = student_ppl(fresh_student())
    print(f"[M-E2] teacher ppl {t_ppl:.4f}, student-at-init ppl {init_ppl:.4f}, "
          f"student layer {S_LAYER} <- teacher layer {a.teach_layer}", flush=True)
    at.add("teacher_ppl", value=t_ppl, corpus_id=ppl_corpus, model_id=WT.PG1_REPO,
           model_sha=base_sha, flags={"role": "the ceiling the student is taught toward",
                                      "forward": "blank-image multimodal, text block located"})
    at.add("student_init_ppl", value=init_ppl, corpus_id=ppl_corpus,
           flags={"role": "the floor every objective must beat", "d_model": a.d_model,
                  "n_layers": a.n_layers, "seed": a.seed, "student_layer": S_LAYER})

    # THE STRUCTURED CONTROL, built from the teacher's own activations rather than loaded. Two
    # runs now show sink32 fitting its subspace ~7x better than a HAAR twin — but they also show
    # the sink holding 16.56 units of teacher energy against a twin's 0.80, twenty times more. A
    # Haar draw is not big anywhere, so it cannot separate "the sink is where the teacher keeps
    # what matters" from "any high-energy 32 coordinates would do". M-G met the same question and
    # answered it uncomfortably: its top-energy control fitted almost as well as the core. So the
    # top-32 coordinates by mean squared teacher activation at the teach layer enter as a
    # first-class objective (house rule: structured control, not just Haar).
    _e_coord = torch.stack([t_hidden[id(w)].to("cuda").float().pow(2).mean(dim=(0, 1))
                            for w in train_w[:64]]).mean(dim=0)
    _top = torch.topk(_e_coord, 32).indices.sort().values
    Ut["topenergy32"] = torch.zeros(d_teacher, 32, device="cuda", dtype=torch.float32)
    Ut["topenergy32"][_top, torch.arange(32, device="cuda")] = 1.0
    print(f"[M-E2] top-energy control: coords {_top[:8].tolist()}… "
          f"(energy {float(_e_coord[_top].mean()):.4f} vs mean {float(_e_coord.mean()):.4f})",
          flush=True)

    # THE QUANTITY THAT CONFOUNDED THE FIRST RUN, banked so the normalisation is auditable rather
    # than merely applied: how much of the teacher's activation energy each subspace holds. The
    # sink is selected FOR being high-energy, so this is expected to be large for sink32 and small
    # for the twins — which is exactly why an unnormalised MSE could not compare them.
    _e_probe = torch.stack([t_hidden[id(w)].to("cuda").float().pow(2).mean()
                            for w in train_w[:32]]).mean()
    for _obj in ("hidden", "sink32", "topenergy32", "twin32_s0", "twin32_s1"):
        if _obj == "hidden":
            e = float(_e_probe)
        else:
            B = Ut[_obj]
            e = float(torch.stack([(t_hidden[id(w)].to("cuda").float() @ B).pow(2).mean()
                                   for w in train_w[:32]]).mean())
        at.add("teacher_subspace_energy", value=e, corpus_id=ppl_corpus,
               flags={"objective": _obj, "signal_width": 2048 if _obj == "hidden" else 32,
                      "definition": ("mean squared teacher activation inside this objective's "
                                     "subspace; the per-objective loss is divided by it")})
        print(f"[M-E2] teacher energy in {_obj:10s} = {e:.5f}", flush=True)

    results = {}
    for obj in OBJECTIVES:
        student = fresh_student()
        # The adapter is part of the student for every hidden-space objective, and it exists for
        # all of them (including sink32) so the arms differ only in WHERE the loss is taken.
        adapter = torch.nn.Linear(a.d_model, d_teacher, bias=False).to("cuda")
        torch.manual_seed(a.seed + 1)
        torch.nn.init.normal_(adapter.weight, std=0.02)
        params = list(student.parameters()) + list(adapter.parameters())
        opt = torch.optim.AdamW(params, lr=a.lr)
        rng = np.random.default_rng(H.cell_seed(a.seed, ARM, obj, "train"))
        student.train()
        hist = []
        for step in range(a.steps):
            win = train_w[int(rng.integers(0, len(train_w)))]
            b = win["ids"].unsqueeze(0).to("cuda")
            s_lg, s_h = student_forward(student, b)
            if obj == "logits":
                t_lg, _ = teacher_forward(win, want_logits=True)
                t = torch.log_softmax(t_lg, dim=-1)
                sl = torch.log_softmax(s_lg.to(torch.float32), dim=-1)
                loss = (torch.exp(t) * (t - sl)).sum(dim=-1).mean()
            else:
                # RELATIVE error, not raw MSE (defect M-D58). The sink subspace is chosen for
                # carrying the teacher's activation energy, so a raw MSE inside it starts ~18x
                # larger than the same MSE inside a random 32-dim twin (16.59 against 0.87 in the
                # first run). With one learning rate for every arm, loss SCALE is effective
                # learning rate — so the arm would have been comparing how big each subspace is,
                # not how much each teaches, and "sink beats twin" would follow from the sink
                # being the high-energy subspace by construction.
                #
                # Dividing by the teacher's own energy in the SAME subspace makes every objective
                # "match the teacher to within x% of what is there to match". The denominator does
                # not depend on the student, so this rescales the loss without moving its minimum.
                t_h = t_hidden[id(win)].to("cuda").float()
                proj = adapter(s_h.to(torch.float32))
                if obj == "hidden":
                    tgt, out_ = t_h, proj
                else:
                    B = Ut[obj]
                    tgt, out_ = t_h @ B, proj @ B
                scale = tgt.pow(2).mean().clamp_min(1e-12)
                loss = torch.nn.functional.mse_loss(out_, tgt) / scale
            opt.zero_grad()
            loss.backward()
            opt.step()
            hist.append(float(loss.detach()))
            if (step + 1) % 100 == 0:
                print(f"    {obj:10s} step {step+1}/{a.steps} loss={hist[-1]:.5f}", flush=True)
        ppl = student_ppl(student)
        # Bits of teacher signal per step, so "teaches more per bit" is a number rather than a
        # figure of speech: logits carry vocab floats per token, hidden d_teacher, the projections
        # 32.
        width = {"logits": v_teacher, "hidden": d_teacher}.get(obj, 32)
        cell = {"objective": obj, "student_ppl": ppl, "loss_first": hist[0],
                "loss_final": hist[-1], "signal_width": width,
                "improvement_over_init": (init_ppl - ppl) / init_ppl,
                "gap_to_teacher": ppl / t_ppl, "student_layer": S_LAYER}
        results[obj] = cell
        at.add("distill_student_ppl", value=ppl, corpus_id=ppl_corpus,
               bank_id=("core32" if obj == "sink32" else
                        ("haar32_s0" if obj == "twin32_s0" else
                         ("haar32_s1" if obj == "twin32_s1" else None))),
               flags={**cell, "steps": a.steps, "lr": a.lr, "batch": 1,
                      "teacher_ppl": t_ppl, "init_ppl": init_ppl,
                      "teach_layer_teacher": a.teach_layer})
        print(f"[M-E2] {obj:10s} ppl={ppl:9.4f}  improvement over init "
              f"{cell['improvement_over_init']:+.4f}  width={width}", flush=True)
        del student, adapter, opt
        torch.cuda.empty_cache()

    # ---- validity first, comparison second ----------------------------------------------------
    learns = bool(results["logits"]["improvement_over_init"] >= MIN_LEARNING_MARGIN)
    gates.record("e2_logits_objective_actually_teaches", learns,
                 improvement=results["logits"]["improvement_over_init"],
                 required=MIN_LEARNING_MARGIN, init_ppl=init_ppl,
                 logits_ppl=results["logits"]["student_ppl"],
                 outcome=None if learns else M.VOID_INSTRUMENT,
                 note=("the positive control: a student that does not learn under the known-good "
                       "objective cannot discriminate between the others"))
    useful_teacher = bool(t_ppl < init_ppl)
    gates.record("e2_teacher_beats_student_floor", useful_teacher,
                 teacher_ppl=t_ppl, student_init_ppl=init_ppl)
    twins = [results[o]["student_ppl"] for o in ("twin32_s0", "twin32_s1")]
    sink = results["sink32"]["student_ppl"]
    beats_twins = bool(sink < min(twins))
    gates.record("e2_sink_projection_teaches_better_than_twin_projections", beats_twins,
                 sink32=sink, twins=twins, hidden=results["hidden"]["student_ppl"],
                 logits=results["logits"]["student_ppl"],
                 note=("a 32-dim bottleneck may teach well simply for being a bottleneck; the "
                       "twins are the same bottleneck without the sink"))
    gates.record("e2_sink_projection_matches_dense_hidden",
                 bool(sink <= results["hidden"]["student_ppl"]),
                 sink32=sink, hidden=results["hidden"]["student_ppl"],
                 signal_widths={o: results[o]["signal_width"] for o in OBJECTIVES},
                 note=("32 floats per token against d_teacher; equal or better at 1/64 the width "
                       "is the practical claim, and it is only readable if the twins fail"))
    # THE GATE THAT DECIDES WHETHER THIS IS ABOUT THE SINK. Beating a Haar twin only shows the
    # subspace is not arbitrary; the top-energy control has the sink's ENERGY without its identity,
    # so it is the one that separates "the consensus subspace" from "any 32 big coordinates". It is
    # recorded as its own gate rather than folded into the twin comparison, because the twin
    # comparison is what M-E2 pre-registered and this is an addition to it, not a reinterpretation.
    top_ppl = results["topenergy32"]["student_ppl"]
    gates.record("e2_sink_beats_the_top_energy_control", bool(sink < top_ppl),
                 sink32=sink, topenergy32=top_ppl, twins=twins,
                 note=("added after two runs showed the sink holds ~20x a Haar twin's teacher "
                       "energy; a control that is merely big tests whether energy alone explains "
                       "the result. NOT part of the pre-logged refuter — reported beside it"))
    sur.add(arm=ARM, expected_ref=expected_ref, cell="sink32_vs_twins",
            statistic="distill_student_ppl/core32",
            s=M.surprise(sink, float(np.mean(twins)), twins))

    info = at.to_parquet()
    sur.to_parquet()
    M.write_json(out, f"{ARM}_result.json",
                 {"arm": ARM, "queue_item": meta["queue_item"], "refuter": meta["refuter"],
                  "expected_ref": expected_ref, "provenance": M.provenance(), "utc": M.utc_now(),
                  "atlas": info, "teacher_repo": WT.PG1_REPO, "teacher_sha": base_sha,
                  "teacher_ppl": t_ppl, "student_init_ppl": init_ppl,
                  "student": {"d_model": a.d_model, "n_layers": a.n_layers,
                              "n_heads": a.n_heads, "seed": a.seed},
                  "steps": a.steps, "batch": 1, "seq_len": a.seq_len, "lr": a.lr,
                  "teach_layer_teacher": a.teach_layer, "objectives": list(OBJECTIVES),
                  "corpus": ppl_corpus,
                  "cells": results, "logits_objective_teaches": learns,
                  "sink_beats_twins": beats_twins,
                  "sink_beats_topenergy": bool(sink < results["topenergy32"]["student_ppl"]),
                  "outcome": None if learns else M.VOID_INSTRUMENT})
    print("E2_OK " + json.dumps({"rows": info["n_rows"],
                                 "ppl": {o: results[o]["student_ppl"] for o in OBJECTIVES},
                                 "teaches": learns, "sink_beats_twins": beats_twins}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
