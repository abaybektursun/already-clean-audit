"""latloc Arm 3 (EXPLORATORY, prereg §3.3) — the restoration-depth profile on a DEEP/erased deletion:
Gemma-1 π0 (MCQ AND cloze both at floor; genuine erasure), as a by-analogy companion to the Gemma-2
latent deletion of Arm 1. The two are compared ONLY at the verdict level as two INDEPENDENT categorical
labels {late|early|none}, each from that model's OWN decoder depth — NO weight / activation / number /
layer-index / normalized-depth ever crosses the Gemma-1 ↔ Gemma-2 generation firewall.

Head-sharing confound (§3.3), RESOLVED: both the intact ceiling (base Gemma-1 decoder) and the erased
subject (π0 decoder) are placed in the SAME base-embed swapback shell, so they share a bitwise-identical
embed+head and differ ONLY in the 18 decoder layers — exactly the Arm-1 PG2 construction, one generation
down. Text-only (π0's backbone has no image path).

    python -m infra.aws_pg.jobs.latloc_arm3        # (--out provided by run_job.sh)
"""
from __future__ import annotations

import argparse
import json
import pathlib

from infra.pg.latloc import GEMMA1_N_LAYERS


def assert_g1_layers(inner, expected=GEMMA1_N_LAYERS):
    from infra.pg.gemma_scope import resolve_gemma2_layers
    n = len(resolve_gemma2_layers(inner))
    assert n == expected, (f"GENERATION FIREWALL: expected {expected} Gemma-1 decoder layers, found {n}. "
                           f"A 26-layer Gemma-2 model here is a firewall breach.")
    return n


def _text_items(tok, split_pred, n_per_subject=None):
    """MMLU 5-shot TEXT-ONLY prompts (no image) under the base Gemma tokenizer. Mirrors the Arm-1
    item build but text-only. Returns (items, letter_ids)."""
    import hashlib
    from collections import defaultdict
    from datasets import load_dataset
    from infra.pg import text_eval as TE

    letter_ids = []
    for L in TE.CHOICES:
        ids = tok.encode(" " + L, add_special_tokens=False)
        assert len(ids) == 1, f"' {L}' → {len(ids)} tokens"
        letter_ids.append(ids[0])
    assert len(set(letter_ids)) == 4, f"letter ids collide: {letter_ids}"

    ds = load_dataset(TE.MMLU_REPO, "all")
    shots_by_subj, test_by_subj = defaultdict(list), defaultdict(list)
    for r in ds["dev"]:
        shots_by_subj[r["subject"]].append(r)
    for r in ds["test"]:
        test_by_subj[r["subject"]].append(r)
    items = []
    for subj in sorted(test_by_subj):
        shots = shots_by_subj[subj][:5]
        assert len(shots) == 5, f"{subj}: {len(shots)} dev shots"
        prefix = TE._subject_header(subj) + "".join(
            TE._q_block(s["question"], s["choices"], s["answer"]) for s in shots)
        rows = test_by_subj[subj]
        if n_per_subject:
            rows = rows[:n_per_subject]
        for r in rows:
            h = hashlib.sha1(r["question"][:64].encode()).hexdigest()[:12]
            if not split_pred(h):
                continue
            items.append({"prompt": prefix + TE._q_block(r["question"], r["choices"]),
                          "answer": int(r["answer"]), "hash": h})
    assert items, "no Arm-3 items selected"
    return items, letter_ids


def _text_sweep(intact, deleted, tok, items, letter_ids, device, donor_seeds, layers=None):
    """TEXT-ONLY intact→deleted residual restoration sweep (the Arm-1 sweep without pixel_values).
    Same-length buckets (Catch-5), correct + 5-seed wrong-donor per layer. Returns per-item floor,
    intact, correct (nL,n), wrong (nSeeds,nL,n)."""
    import numpy as np
    import torch
    from infra.pg.gemma_scope import resolve_gemma2_layers
    from infra.pg.latloc import donor_permutation, length_buckets
    from infra.pg.lm_patching import PatchHook
    from infra.aws_pg.jobs.latloc_cell import _capture, _chunks

    a_layers = resolve_gemma2_layers(deleted.model)     # erased subject (π0 decoder)
    b_layers = resolve_gemma2_layers(intact.model)      # ceiling (base Gemma-1 decoder)
    n_layers = len(a_layers)
    layer_list = list(range(n_layers)) if layers is None else list(layers)

    enc_per, lengths = [], []
    for it in items:
        e = tok(it["prompt"], return_tensors="pt", add_special_tokens=True)
        enc_per.append(e)
        lengths.append(int(e["input_ids"].shape[1]))
    buckets = length_buckets(lengths, min_bucket=2)
    n_used = sum(len(v) for v in buckets.values())
    print(f"[arm3] {len(items)} items → {len(buckets)} buckets covering {n_used} "
          f"(retention={n_used/max(1,len(items)):.2%})")

    letter_ids_t = torch.tensor(letter_ids, device=device)
    nS = len(donor_seeds)
    floor = np.full(len(items), np.nan)
    intact_a = np.full(len(items), np.nan)
    correct = np.full((n_layers, len(items)), np.nan)
    wrong = np.full((nS, n_layers, len(items)), np.nan)
    answers = np.array([int(it["answer"]) for it in items])

    def _acc(model, ids, am, ans):
        with torch.inference_mode():
            last = model(input_ids=ids, attention_mask=am, logits_to_keep=1).logits[:, -1, :]
        return (last[:, letter_ids_t].argmax(-1).cpu().numpy() == ans).astype(float)

    for _Llen, idxs in sorted(buckets.items()):
        for chunk in _chunks(list(idxs)):
            ids = torch.cat([enc_per[i]["input_ids"] for i in chunk], 0).to(device)
            am = torch.cat([enc_per[i]["attention_mask"] for i in chunk], 0).to(device)
            assert (am == 1).all(), "same-length bucket must have no padding"
            ans = answers[chunk]
            floor[chunk] = _acc(deleted, ids, am, ans)
            intact_a[chunk] = _acc(intact, ids, am, ans)
            intact_res = _capture(intact, b_layers, layer_list, ids, am)   # text-only (pv=None)
            donors = [donor_permutation(len(chunk), s) for s in donor_seeds]
            pm = torch.ones_like(ids, dtype=torch.bool)
            for L in layer_list:
                with PatchHook(a_layers[L], intact_res[L], pm, mode="resample"):
                    correct[L, chunk] = _acc(deleted, ids, am, ans)
                for si, dperm in enumerate(donors):
                    with PatchHook(a_layers[L], intact_res[L][dperm], pm, mode="resample"):
                        wrong[si, L, chunk] = _acc(deleted, ids, am, ans)
    return {"floor": floor, "intact": intact_a, "correct": correct, "wrong": wrong,
            "n_used": n_used, "n_buckets": len(buckets), "retention": float(n_used / max(1, len(items)))}


def _band_label(onset, n_layers):
    """Within-generation categorical band from THIS model's own depth (no cross-gen axis): late =
    upper third, early = lower third, else mid; none = no stable onset."""
    if onset is None:
        return "none"
    import math
    late0 = math.ceil(0.67 * n_layers)
    early1 = n_layers // 3
    if onset >= late0:
        return "late"
    if onset < early1:
        return "early"
    return "mid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--eval-split", choices=["bucket", "confirmation"], default="bucket")
    ap.add_argument("--n-per-subject", type=int, default=0)
    args = ap.parse_args()
    assert args.out, "--out required"
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from infra.pg import vlat_swapback as SB
    from infra.pg import latloc as LL, w16_probes as Wp
    from infra.pg.entities import ENTITIES
    from infra.aws_pg.jobs.vlat_retention import _bare_decoder
    from infra.aws_pg.jobs.policy_content_readout import _extract_own, vocab_parity_gate

    assert torch.cuda.is_available(), "arm3 needs GPU"
    device = "cuda"
    n_per = args.n_per_subject or None
    L0g1 = ENTITIES["L0_g1"]
    tok = AutoTokenizer.from_pretrained(L0g1.repo_id, revision=L0g1.revision)
    vp = vocab_parity_gate(tok, n_per_subject=(n_per or 2))
    print(f"[arm3] vocab-parity OK max_id={vp['max_token_id']}<256000")

    # intact = base Gemma-1 decoder in the base-embed shell; deleted = π0 decoder in the SAME shell.
    base = AutoModelForCausalLM.from_pretrained(
        L0g1.repo_id, revision=L0g1.revision, torch_dtype=torch.bfloat16, device_map="cpu")
    base_dec = {k: v.detach().clone() for k, v in _bare_decoder(base.state_dict()).items()}
    del base
    pi0_dec, _own_head, sha = _extract_own("pi0", device)
    # intact = base decoder in the base shell (UNCHANGED → must disable the "decoder changed" gate);
    # deleted = π0 decoder in the SAME shell (changed; default gate applies). Both keep base embed+head.
    intact, info_i = SB.load_swapback_model(base_dec, device, require_decoder_changed=False)
    deleted, info_d = SB.load_swapback_model(pi0_dec, device)
    intact.eval(); deleted.eval()
    n_dec = assert_g1_layers(intact.model, LL.GEMMA1_N_LAYERS)
    assert_g1_layers(deleted.model, LL.GEMMA1_N_LAYERS)
    print(f"[arm3] shells built; firewall {n_dec}==18; π0 sha={sha}")

    split_pred = Wp.is_confirmation if args.eval_split == "confirmation" else Wp.is_grid_subset
    donor_seeds = list(LL.WRONG_DONOR_SEEDS)
    items, letter_ids = _text_items(tok, split_pred, n_per_subject=n_per)

    # preflight: intact MCQ ceiling vs deleted (erased) floor — require dynamic range.
    sweep = _text_sweep(intact, deleted, tok, items, letter_ids, device, donor_seeds)
    used = np.isfinite(sweep["floor"])
    floor_i, intact_i = sweep["floor"][used], sweep["intact"][used]
    int_acc, del_acc = float(intact_i.mean()), float(floor_i.mean())
    gap = int_acc - del_acc
    print(f"[arm3] MCQ intact(base-G1)={int_acc:.3f} deleted(π0)={del_acc:.3f} gap={gap:.3f}")

    correct = sweep["correct"][:, used]
    wrong_mean = sweep["wrong"][:, :, used].mean(0)
    nL = correct.shape[0]
    if gap >= 0.10:    # dynamic range present → compute the within-G1 restoration onset
        points, ci_lo, ci_hi = LL.bootstrap_delta_rf_profile(correct, wrong_mean, floor_i, intact_i)
        profile = [float(x) for x in points]
        onset = LL.stable_onset_depth(profile)
        label = _band_label(onset, nL)
        adjudicable = True
    else:
        profile, ci_lo, ci_hi, onset, label, adjudicable = None, None, None, None, "cannot_adjudicate", False
        print(f"[arm3] gap {gap:.3f} < 0.10 → cannot adjudicate (no dynamic range)")

    row = {"study": "latloc", "arm": "3_pi0_EXPLORATORY", "generation": "gemma-1", "firewall_n_layers": n_dec,
           "eval_split": args.eval_split, "pi0_sha": sha, "donor_seeds": donor_seeds,
           "mcq_intact_base_g1": int_acc, "mcq_deleted_pi0": del_acc, "gap": gap,
           "n_items": len(items), "n_used": int(used.sum()), "retention": sweep["retention"],
           "adjudicable": adjudicable, "within_g1_stable_onset": onset, "band_label": label,
           "delta_rf_profile": profile, "delta_rf_ci_lo": (None if ci_lo is None else [float(x) for x in ci_lo]),
           "delta_rf_ci_hi": (None if ci_hi is None else [float(x) for x in ci_hi]),
           "delta": LL.DELTA, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
           "NOTE": "VERDICT-LEVEL categorical label ONLY; never numerically compared to the Gemma-2 Arm-1 onset."}
    (out / "latloc_arm3.json").write_text(json.dumps(row, indent=1, default=str))
    print(f"[arm3] banked; within-G1 onset={onset} band={label} adjudicable={adjudicable}")


if __name__ == "__main__":
    main()
