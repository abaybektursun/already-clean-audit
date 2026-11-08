"""Weight-free dry run of REUSE (studies/reuse/SPEC.md §9). Nothing is downloaded.

    python -m infra.aws_pg.jobs.reuse_dryrun --out <dir>              (parts 1-10)
    python -m infra.aws_pg.jobs.reuse_dryrun --out <dir> --part 8     (part 8 alone; it builds every bank it reads)
    python -m infra.aws_pg.jobs.reuse_dryrun --out <dir> --part 9     (part 9 alone; it builds every bank it reads)
    python -m infra.aws_pg.jobs.reuse_dryrun --out <dir> --part 10    (part 10 alone; it builds every bank it reads)

1. Pure logic on random tensors (`reuse_ablate.selfcheck`): fall, random_matched, top_m, linfit, the freeze rule, the cancel rule.
2. The sniper job end to end (`reuse_ablate.Job.run`) on a tiny random OLMo 3: S1 at "T_15" with every arm, planted pair and ceiling split, then
   S2 at "T_10" on a second random model that reads S1's foreign choice. Items in the real format, built from the models' own activations.
3. A constructed model in fp32 where one upstream neuron (l, k) is the only input of the pole: the residual coordinate the pole reads is written by
   (l, k) alone, through layer l's post-feedforward norm. Its fall under mean and zero replacement is predicted in closed form from the unablated
   forward (the MLP output with k's term replaced, then the norm), independently of the hooks.
4. The primary stage (SPEC §13.1 item 4): items in the primary format, 8 jobs with labels and ceiling splits, born at "T_8" and "T_11"; three loads,
   each on its own random model: "T_15" (every job, Q chosen there), then the two birth checkpoints (only the jobs born there, plus their ceiling
   splits, Q read from the T_15 step via --from-step). Two jobs share a pole and a ceiling c at one birth (one split read), and that split recurs
   at the other birth. Then s3_freeze.py on the dry S3 bank, and verdict.py on the three loads: green, then failing under each of its three
   failing inputs (own and random swapped, planted labels shuffled, cross and foreign swapped).
5. The amend2 stage (studies/reuse/SPEC2.md §3) on the same three random models. Items: part 4's, except that every job naming a split carries
   the first such job's sides, as in the real items (part 4 gives the split read at both births other sides at each), with four planted pairs
   whose side J is driven by neuron a and side O by neuron b on all three models. On them: the primary stage at the three loads and verdict.py (the primary bank verdict2 reads), an S3-format bank
   at "T_15" (s3_t15 on a setting file whose first split is a primary split, so the agreement check has a split to read twice) and
   s3_freeze.py on it; then amend2_t15 at "T_15" (R_J for every job, every distinct split on side 2) and amend2_birth at "T_8"
   and "T_11" (R_J for the jobs born there, the planted pairs with shuffled labels). Each load: identity exactly 0, every unit and arm banked,
   D(J) and D(O*) and a_P before equal the part 4 bank's, and R_J and its energies equal an independent recomputation from a fresh capture.
   Then verdict2.py: on the primary bank as the amend2 bank it must stop (exit 4, nothing written); on the amend2 bank green; under
   --shuffle-planted room-b fails and under --swap own rj the positive control fails (exit 1 each).

6. The heads stage (studies/reuse/SPEC_B_heads.md) and 7. the output stage (studies/reuse/SPEC_A_output.md), both per load as the pipeline runs them
   (reuse_ablate2.run_load), on fp32 tiny models with 48 heads of width 4 over 12 key-value heads: the sniper at "T_15" and "T_10" (h 10, 20, 40; r 10,
   20, 50) and a two-load primary at "T_15" and "T_11" (h 10, r 10), after REUSE's primary stage, an S3 bank and verdict.py on the same models (the
   I_MLP of R5h and R6o). Each load: identity exactly 0, the head or reader capture check passing and failing on the wrong layer, every job, split and
   planted pair read with every set, H(J) or R(J) equal to a fresh capture's. Part 6 also zeroes one key-value group's value rows (exactly its heads
   output 0), runs the head hooks in bf16, and prints the head capture check in bf16. Then verdict_heads.py and verdict_output.py: --sniper freezes h
   and r and fails under --swap own random and --shuffle-planted; the primary verdict stops on the sniper bank (exit 4), is green on the primary bank,
   and fails under each failing input (exit 1).

8. SPEC4 BIRTH (studies/reuse/SPEC4_birth.md §12 item 2): reuse_birth.selfcheck; the primary stage on part 5's items and models (the bank ranked
   without exclusion); the birth stage per load (jobs.reuse_birth.Job.run) at "T_15" (birth_t15), "T_7" (prebirth_load: the T_5 code path, on the
   "T_8" model with the T_8-born poles' gate and up rows scaled by 0.6), "T_8" (birth_load), "T_9" (prebirth_load), "T_10" (birth_load: fresh births
   there plus the pre-birth banks of the T_11 births) and "T_11" (birth_load), on birth items holding part 5's primary jobs, six fresh cases on new
   neurons (four with a ceiling, two sharing a split), part 5's four planted pairs and a known block per load. Each load: identity exactly 0, B_t the
   top 1% of |ā|, no member of any set in B_t, every case, split, pair and Part A bank with every arm, memory recorded. The known answers of the
   construction are checked in the verdict's Part A reading (rho_omega = rho_g = rho_u = 0.6, the rows' cosines 1, rho_P under 0.5, rho_a(O*)
   about 1). Then verdict_birth.py: exit 4 on a bank without t09, green on the full bank, exit 1 under --shuffle-planted, --swap own random,
   --failing-input exclusion (the primary bank's sets against B_t) and --failing-input gate_L, and exit 1 on items whose fresh case shares a primary
   neuron (build_items_birth.disjoint_or_stop, also called directly).

9. SPEC5 DATING (studies/reuse/SPEC5_dating.md §9 item 1): the dating job per load (jobs.reuse_dating.Job.run) at "T_12".."T_15" on one fp32 random
   model whose doubling poles' gate rows are scaled by 0.2, 0.25, 0.6, 1.0 (r_J doubles between T_13 and T_14 only) and whose flat poles sit below every
   scaled row, with a fake exemplar bank (the T_15 model's values at every checkpoint, so s_P is constant). Each load: the known number passing and its
   failing input failing, capture identity exactly 0, every case banked; T_15 adds the scale capture (within 1e-4 of the bank) and the 1M-token
   background pass (background_1m.npz's five arrays, the Jaccard of B_15 and the 1M top 1%). The known-number gate stops a load given its failing
   input. verdict_dating.per_case on three constructed cases with an invalid checkpoint (s_P under 0.01 s_P(T_15)), a negative r_J and a doubling under
   the floor, and d2_verdict on constructed intervals for each of its outcomes. Then verdict_dating.py: exit 4 on a bank without t13; green (exit 0)
   with the known answers (doubling pair T_13 on the growing cases and none on the flat ones, b_act, D1 0.5 on each set, every pair valid, D2 refuted
   by the O* ratio); exit 1 under each failing input with its target gates failed and every rule stamped not read.
   Amendment 5b: the scale stage (--stage scale) at each of the four loads on the same models (known number, capture identity exactly 0, no case,
   every pole's median within 1e-4 of the model's own reference median at that load); then verdict_dating.py --scale-run: exit 4 on a scale bank
   without t13; green (exit 0) with s_P(t) the scale run's medians, the flat cases' r_J unchanged and the growing cases' doubling gone (their scale
   moves with the pole); exit 1 under each failing input (the scale one now the next pole's dating-run T_15 median).

10. SPEC6 GENERIC (studies/reuse/SPEC6_generic.md §9 step 1): reuse_generic.selfcheck (the held norm against a float64 hand computation in fp32 and
   bf16, positions before the window start left to the model, the empty set exact, the shifted inv not, the recorded-shape assertion firing, the
   probe; the g1, g3 and per-layer cuts; the members-alone and single-largest arms); the held scale through the real decoder layer of a tiny fp32
   OLMo 3 (layer output = MLP input + weight y_ablated inv_clean by hand) and the bf16 empty set exact through the readout; then the generic job
   (jobs.reuse_generic.Job.run) on a tiny bf16 OLMo 3 against SPEC4's birth_t15 load on the same model (the free g1 arms' sets and a_P equal, so
   known_spec4_a and known_spec4_b read 0) and a T_14 bank from another model, every held gate passing with its failing input failing, every unit
   and (arm, exclusion, scale) banked; verdict_generic reads that bank (exit 0 or 1). Then verdict_generic on a constructed bank (jobs.reuse_generic
   .bank_arm, known falls, sets and Part A arrays): exit 4 without one npz, exit 0 with the known answers (W1 0.30, the unexcluded null 0.16, N1
   0.06), exit 1 under each failing input with its target failed.

Writes metrics.json and samples/arms.jsonl into --out. Prints REUSE_DRYRUN_OK only if: identity fall is exactly 0 on every context at every load;
ablating the sole driver gives the predicted fall; random_matched returns the right count and excludes the forbidden sets; the primary loads read
the right jobs and splits, bank every arm, and pass the known-number gate while its failing input fails; verdict.py passes on the dry bank and
fails on each failing input.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

from infra.aws_pg.jobs import reuse_ablate as JR
from infra.aws_pg.jobs import reuse_ablate2 as JR2
from infra.aws_pg.jobs.neuron_off_dryrun import WINDOW, fake_corpus, reference_a, tiny_model, tiny_tokenizer
from infra.pg import neurons_harvest as NH
from infra.pg import reuse_ablate as RA
from infra.pg import reuse_ablate2 as RA2
from infra.pg import unite_route as UR
from infra.pg import unite_stats as US
from infra.pg.neuron_rw_core import norm_eps

REPO = pathlib.Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "studies" / "reuse" / "scripts"

N_LAYERS = 6
N_WINDOWS = 60
S1_ARMS = 1 + 2 * 3 * 3 + 2 + 1
S2_ARMS = 1 + 2 * 3 + 2


# ------------------------------------------------------------------------------------------ items in the sniper format
def top_tok(ref, L, n, sign, k=50):
    a = np.asarray(ref[L])[:, :, n].astype(np.float64) * sign
    a[:, 0] = -np.inf                                   # window position 0 is not a context, as in the harvest
    return np.argsort(-a.ravel())[:k].astype(np.int64)  # window * WINDOW + position = the corpus index


def build_items(ref, ref2, tokens):
    jobs = []
    spec = [(3, 5, "+"), (4, 11, "+"), (4, 20, "-"), (5, 2, "+"), (5, 7, "+")]
    for i, (L, n, s) in enumerate(spec):
        sg = 1.0 if s == "+" else -1.0
        t = top_tok(ref, L, n, sg)
        others = [x for x in range(0, 40, 3) if x != n][:3]
        foreign = [{"pole": f"L{L:02d}N{x}{s}", "c": 0, "ranks": list(range(16)), "tok": top_tok(ref, L, x, sg)[:16].tolist()} for x in others]
        jobs.append({"job": f"L{L:02d}N{n}{s}:c1", "pole": f"L{L:02d}N{n}{s}", "layer": L, "neuron": n, "sign": s, "tercile": i % 3, "c": 1, "birth": 10,
                     "J_tok": t[:8].tolist(), "O_tok": t[8:24].tolist(),
                     "ceiling": {"c": 0, "sides_tok": [t[24:31].tolist(), t[31:38].tolist()], "in_s1": i in (0, 2, 4)}, "foreign_candidates": foreign})
    planted = []
    r4 = np.asarray(ref[4])[:, 1:, :].reshape(-1, np.asarray(ref[4]).shape[-1]).astype(np.float64)
    tok_rest = (np.arange(N_WINDOWS)[:, None] * WINDOW + np.arange(1, WINDOW)[None]).reshape(-1)
    for x, y in [(a, b) for a in range(0, 60, 7) for b in range(1, 60, 5) if a != b]:
        ta, tb = r4[:, x], r4[:, y]
        o = np.argsort(-(ta + tb))[:50]
        J, O = o[ta[o] > tb[o]], o[~(ta[o] > tb[o])]
        if len(J) >= 4 and len(O) >= 4:
            planted.append({"unit": "PL04_00", "layer": 4, "a": x, "b": y, "lam": 1.0, "J_tok": tok_rest[J][:12].tolist(), "O_tok": tok_rest[O].tolist()})
            break
    assert planted, "no planted pair with both sides on the tiny model"
    kp = ["L03N5+", "L04N11+", "L05N2+"]
    ktok = [top_tok(ref, NH.parse_pole(p)[0], NH.parse_pole(p)[1], 1.0) for p in kp]
    val = lambda r, p, t: np.asarray(r[NH.parse_pole(p)[0]])[t // WINDOW, t % WINDOW, NH.parse_pole(p)[1]].tolist()
    known = {"poles": kp, "tok": [t.tolist() for t in ktok], "wrong_rule": "the other checkpoint's model",
             "a_bank": {"T_15": [val(ref, p, t) for p, t in zip(kp, ktok)], "T_10": [val(ref2, p, t) for p, t in zip(kp, ktok)]},
             "a_wrong": {"T_15": [val(ref2, p, t) for p, t in zip(kp, ktok)], "T_10": [val(ref, p, t) for p, t in zip(kp, ktok)]}}
    return {"jobs": jobs, "planted": planted, "known": known}


def args_for(out, *extra, stage: str = "sniper", sha: str = "dry"):
    a = JR.parser().parse_args(["--out", str(out), "--stage", stage, "--items", "x", "--harvest-window", str(WINDOW), "--scan-tokens", str(N_WINDOWS * WINDOW),
                                "--batch", "8", "--full-batch", "4", "--grad-batch", "4", *extra])
    a.items_sha256 = sha
    return a


def check_step(d: pathlib.Path, items: dict, n_arms: int, spec: dict) -> dict:
    m = json.loads((d / "metrics.json").read_text())
    assert m["status"] == "done", m.get("status")
    assert m["known_number"]["pass"] and m["known_number"]["failing_input_failed"], m["known_number"]
    idn = m["identity"]
    assert idn["max_abs_fall"] == 0.0 and idn["max_abs_change"] == 0.0 and idn["n_over_tol"] == 0, f"identity is not exactly 0: {idn}"
    assert idn["failing_input_failed"] and idn["failing_input_n_over_tol"] > 0, f"identity's failing input did not fail: {idn}"
    n_lines = 0
    for j in items["jobs"]:
        r = m["jobs"][j["job"]]
        assert len(r["falls"]) == n_arms, (j["job"], len(r["falls"]), sorted(r["falls"]))
        z = np.load(d / r["falls_path"])
        assert z["a_after"].shape == (n_arms, len(j["J_tok"])) and (z["a_before"] == z["a_after"][0]).all(), "identity row differs from before"
        names = list(z["arm"])
        for rk in spec["rankings"]:
            for mm in spec["ms"]:
                sets = {}
                for x in ("own", "cross", "random"):
                    i = names.index(f"{x}_{rk}_m{mm}")
                    sets[x] = {(int(l), int(k)) for l, k in zip(z["set_layer"][i], z["set_neuron"][i]) if l >= 0}
                    assert len(sets[x]) == mm, (x, rk, mm, len(sets[x]))
                assert not sets["random"] & (sets["own"] | sets["cross"]), "random arm holds a forbidden neuron"
        n_lines += n_arms - 1
    for p in items["planted"] if spec["planted"] else []:
        n_lines += (4 - 1) + (3 - 1)
    for j in items["jobs"] if spec["ceiling"] else []:
        if j["ceiling"]["in_s1"]:
            n_lines += 2 * (3 - 1)
    lines = (d / "samples" / "arms.jsonl").read_text().splitlines()
    assert len(lines) == n_lines, (len(lines), n_lines)
    s = json.loads(lines[1])
    assert s["contexts"][0]["prompt"] and s["contexts"][0]["marked_token"] and len(s["contexts"][0]["top_ablated"]) == JR.TOP_SAMPLE, s["contexts"][0]
    assert m["linearity"]["n_rows"] > 0 and np.isfinite(m["linearity"]["d_a_vs_sum_A_c"]["slope"])
    return m


def part_job(out: pathlib.Path) -> dict:
    tok = tiny_tokenizer(bos=True)
    tokens = fake_corpus(tok, N_WINDOWS + 2)
    model = tiny_model("olmo3", seed=0, n_layers=N_LAYERS).requires_grad_(False)
    other = tiny_model("olmo3", seed=1, n_layers=N_LAYERS).requires_grad_(False)
    ref = reference_a(model, tok, tokens, list(range(N_WINDOWS)))
    ref2 = reference_a(other, tok, tokens, list(range(N_WINDOWS)))
    items = build_items(ref, ref2, tokens)
    offsets = np.arange(0, tokens.size + 37, 37)
    offsets[-1] = tokens.size
    corpus = {"tokens": tokens, "doc_offsets": offsets}
    d1, d2 = out / "s1", out / "s2"
    JR.Job(d1, args_for(d1, "--arms", "s1", "--t", "T_15"), items, model, tok, corpus, "cpu").run()
    m1 = check_step(d1, items, S1_ARMS, JR.ARMSETS["s1"])
    JR.Job(d2, args_for(d2, "--arms", "s2", "--t", "T_10", "--from-step", str(d1)), items, other, tok, corpus, "cpu").run()
    m2 = check_step(d2, items, S2_ARMS, JR.ARMSETS["s2"])
    assert json.loads((d1 / "foreign.json").read_text()) == json.loads((d2 / "foreign.json").read_text()), "S2 did not read S1's foreign choice"
    for u, v in m1["planted"].items():
        assert {"labels", "shuffled"} <= set(v) and len(v["labels"]["falls"]) == 4 and len(v["shuffled"]["falls"]) == 3, v.keys()
    assert len(m1["ceiling"]) == 3 and all("I_ceil" in v for v in m1["ceiling"].values())
    assert set(m1["summary"]) >= {"by_ranking_m", "freeze_rule", "zero_vs_mean_own_direct_m50", "planted_I", "planted_I_shuffled", "ceiling_I"}, set(m1["summary"])
    # a failing input through the job's own gate ends the job: the known number against the wrong bank
    wrong = json.loads(json.dumps(items))
    wrong["known"]["a_bank"]["T_15"] = wrong["known"]["a_wrong"]["T_15"]
    raised = False
    try:
        JR.Job(out / "wrong", args_for(out / "wrong", "--arms", "s1", "--t", "T_15"), wrong, model, tok, corpus, "cpu").run()
    except AssertionError:
        raised = True
    assert raised and json.loads((out / "wrong" / "metrics.json").read_text())["known_number"]["pass"] is False, "the wrong bank did not stop the job"
    s3 = part_s3(out, items, model, other, tok, corpus, d1)
    primary = part_primary(out, tok, tokens, corpus)
    shutil.copy(d1 / "samples" / "arms.jsonl", out / "samples" / "arms.jsonl")
    print("S1 identity", m1["identity"], "\nS1 summary", json.dumps(m1["summary"]["by_ranking_m"])[:600], "\nS1 freeze rule", m1["summary"]["freeze_rule"]["why"],
          "\nS1 linearity d_a vs sum_A_c", m1["linearity"]["d_a_vs_sum_A_c"], "\nS2 identity", m2["identity"]["max_abs_fall"])
    return {"s1_identity": m1["identity"], "s2_identity": m2["identity"], "s1_known": m1["known_number"]["ratio"], "s2_known": m2["known_number"]["ratio"],
            "s1_tokens_forwarded": m1["tokens_forwarded"], "s1_forwards": m1["forwards"], "s1_freeze_rule": m1["summary"]["freeze_rule"],
            "s1_linearity": m1["linearity"], "wrong_bank_stopped_the_job": raised, "s3": s3, "primary": primary}


# ------------------------------------------------------------------------------------------ S3: setting-set ceilings, planted pairs at two m, count-matched draws
S3_COUNTS = [[4, 6], [12, 12], [6, 12], [3, 9], [12, 5]]


def part_s3(out: pathlib.Path, items: dict, model, other, tok, corpus, d1: pathlib.Path) -> dict:
    """s3_t15 on the setting format (ceilings, planted, job_counts) and s3_t10 on the sniper format, reading S1's foreign choice. Both under out/s3/, the
    layout s3_freeze.py and verdict.py read; the setting items go to out/s3/setting_items.json (the first split marked as shared with a primary job)."""
    setting = {"ceilings": [{"split": f"{j['pole']}__c{j['ceiling']['c']}", "pole": j["pole"], "layer": j["layer"], "c": j["ceiling"]["c"],
                             "sides_tok": j["ceiling"]["sides_tok"], "shared_with_primary": i == 0} for i, j in enumerate(items["jobs"])],
               "planted": items["planted"], "known": items["known"], "job_counts": S3_COUNTS}
    NH.dump(out / "s3" / "setting_items.json", setting)
    d3 = out / "s3" / "t15"
    JR.Job(d3, args_for(d3, "--arms", "s3_t15", "--t", "T_15"), setting, model, tok, corpus, "cpu").run()
    m3 = json.loads((d3 / "metrics.json").read_text())
    assert m3["status"] == "done" and m3["known_number"]["pass"] and m3["known_number"]["failing_input_failed"], m3["known_number"]
    idn = m3["identity"]
    assert idn["max_abs_fall"] == 0.0 and idn["n_over_tol"] == 0 and idn["failing_input_failed"] and idn["failing_input"] == "own_direct_m50", idn
    spec = JR.ARMSETS["s3_t15"]
    K, cap = spec["draws"], spec["cap"]
    for p in setting["planted"]:
        r = m3["planted"][p["unit"]]
        assert len(r["labels"]["falls"]) == 1 + 3 * 2 and len(r["shuffled"]["falls"]) == 1 + 2 * 2, (len(r["labels"]["falls"]), len(r["shuffled"]["falls"]))
        assert set(r["labels"]["I_m"]) == {"m20", "m50"} and r["labels"]["n_O"] == min(len(p["O_tok"]), cap)
        cm = r["count_matched"]
        assert cm["n_draws"] == K and set(cm["I_median"]) == {"m20", "m50"} and len(cm["I_draws"]["m20"]) == K
        z = np.load(d3 / cm["falls_path"])
        lab = np.load(d3 / r["labels"]["falls_path"])
        nJ, nO = len(lab["tok"]), r["labels"]["n_O"]
        assert (z["a_before"] == lab["a_before"]).all() and z["j"].shape == (K, nJ) and z["o"].shape == (K, nO)
        assert (z["j"].sum(1) == np.minimum(z["count"][:, 0], nJ)).all() and (z["o"].sum(1) == np.minimum(z["count"][:, 1], nO)).all(), "a draw has the wrong count"
        names = list(lab["arm"])
        n_full = 0
        for mm in (20, 50):
            for d in range(K):
                if z["j"][d].all():   # the full J side: the draw's own set and readout are the planted unit's
                    assert (z[f"own_after_m{mm}"][d] == lab["a_after"][names.index(f"own_direct_m{mm}")]).all(), "a full-side draw differs from the planted unit's own arm"
                    n_full += 1
                if z["o"][d].all():
                    assert (z[f"cross_after_m{mm}"][d] == lab["a_after"][names.index(f"cross_direct_m{mm}")]).all(), "a full-side draw differs from the planted unit's cross arm"
                    n_full += 1
        assert n_full > 0, "no draw used a full side, so the reuse check did not run"
    for c in setting["ceilings"]:
        r = m3["ceiling"][c["split"]]
        assert set(r) >= {"I_ceil", "I_ceil_m", "side2"} and "side1" not in r and len(r["side2"]["falls"]) == 1 + 3 * 2, sorted(r)
    lines = (d3 / "samples" / "arms.jsonl").read_text().splitlines()
    assert len(lines) == len(setting["planted"]) * ((7 - 1) + (5 - 1)) + len(setting["ceilings"]) * (7 - 1), len(lines)
    assert set(m3["summary"]) >= {"planted_I_m", "planted_I_shuffled_m", "planted_I_count_matched", "ceiling_I_m"}, sorted(m3["summary"])

    d4 = out / "s3" / "t10"
    JR.Job(d4, args_for(d4, "--arms", "s3_t10", "--t", "T_10", "--from-step", str(d1)), items, other, tok, corpus, "cpu").run()
    m4 = check_step(d4, items, 5, JR.ARMSETS["s3_t10"])
    assert m4["identity"]["failing_input"] == "own_direct_m20"
    for j in items["jobs"]:
        r = m4["jobs"][j["job"]]
        assert r["n_O"] == min(len(j["O_tok"]), 12) and r["foreign"]["n"] <= 12 and "foreign_direct_m20" in r["falls"], (r["n_O"], r["foreign"])
    assert json.loads((d1 / "foreign.json").read_text()) == json.loads((d4 / "foreign.json").read_text()), "s3_t10 did not read S1's foreign choice"
    assert "direct_m20" in m4["summary"]["by_ranking_m"] and "foreign" in m4["summary"]["by_ranking_m"]["direct_m20"]
    print("S3 t15 identity", m3["identity"]["max_abs_fall"], "planted I count-matched", m3["summary"]["planted_I_count_matched"], "ceiling I", m3["summary"]["ceiling_I_m"],
          "\nS3 t10 identity", m4["identity"]["max_abs_fall"], "summary", m4["summary"]["by_ranking_m"])
    return {"t15_identity": m3["identity"], "t15_forwards": m3["forwards"], "t10_identity": m4["identity"], "t10_forwards": m4["forwards"]}


# ------------------------------------------------------------------------------------------ the primary stage and the verdict
# (layer, neuron, sign, job c, birth, label); every ceiling split has c 0. On these 128-wide random models the random arm (50 of 384-640 upstream
# neurons replaced) often moves a_P by about as much as own, so the cancel rule fires on some poles; this list holds poles it keeps, so the verdict's
# cancel-share gate can pass on the dry bank and its failing input can be seen to fail.
PRIMARY = [
    (5, 2, "-", 1, 8, "related"),
    (5, 2, "-", 2, 8, "unrelated"),    # same pole and ceiling c as the job above, same birth: one split read for both
    (5, 2, "-", 3, 11, "related"),     # the same split at the other birth: read again at that load
    (4, 11, "+", 1, 11, "unrelated"),
    (4, 2, "+", 1, 8, "related"),
    (4, 2, "-", 1, 11, "unrelated"),   # the other pole of neuron L04N2: resampled with it in the verdict's bootstrap
    (5, 7, "+", 1, 8, "unrelated"),
    (3, 5, "+", 1, 11, "related")]
PRIMARY_LOADS = ("T_15", "T_8", "T_11")
ARMS5 = {"identity", "own_direct_m50", "cross_direct_m50", "random_direct_m50", "foreign_direct_m50"}


def build_primary(refs: dict, spec: list = PRIMARY) -> dict:
    """Primary-format items. A job's J, O* and ceiling sides are its pole's top contexts on the model of its birth load, so the birth read is a real
    job's; Q's contexts and the known contexts are the T_15 model's. `spec`: (layer, neuron, sign, job c, birth, label) per job."""
    jobs = []
    for i, (L, n, s, c, birth, label) in enumerate(spec):
        sg = 1.0 if s == "+" else -1.0
        t = top_tok(refs[f"T_{birth}"], L, n, sg)
        k = sum(x[:3] == (L, n, s) and x[4] == birth for x in spec[:i])
        others = [x for x in range(0, 40, 3) if x != n][:3]
        foreign = [{"pole": f"L{L:02d}N{x}{s}", "c": 0, "ranks": list(range(12)), "tok": top_tok(refs["T_15"], L, x, sg)[:12].tolist()} for x in others]
        jobs.append({"job": f"L{L:02d}N{n}{s}:c{c}", "pole": f"L{L:02d}N{n}{s}", "layer": L, "neuron": n, "sign": s, "tercile": US.tercile(L), "c": c, "birth": birth,
                     "label": label, "J_tok": t[8 * k:8 * k + 8].tolist(), "O_tok": t[16:28].tolist(),
                     "ceiling": {"c": 0, "sides_tok": [t[28:35].tolist(), t[35:42].tolist()]}, "foreign_candidates": foreign})
    kp = ["L03N5+", "L04N11+", "L05N2+"]
    ktok = [top_tok(refs["T_15"], NH.parse_pole(p)[0], NH.parse_pole(p)[1], 1.0) for p in kp]
    val = lambda r, p, t: np.asarray(r[NH.parse_pole(p)[0]])[t // WINDOW, t % WINDOW, NH.parse_pole(p)[1]].tolist()
    wrong_of = {"T_15": "T_8", "T_8": "T_11", "T_11": "T_15"}
    known = {"poles": kp, "tok": [t.tolist() for t in ktok], "wrong_rule": "another load's model",
             "a_bank": {T: [val(refs[T], p, t) for p, t in zip(kp, ktok)] for T in PRIMARY_LOADS},
             "a_wrong": {T: [val(refs[wrong_of[T]], p, t) for p, t in zip(kp, ktok)] for T in PRIMARY_LOADS}}
    return {"jobs": jobs, "known": known, "job_counts": [[len(j["J_tok"]), len(j["O_tok"])] for j in jobs]}


def check_primary(d: pathlib.Path, items: dict, T: str) -> dict:
    m = json.loads((d / "metrics.json").read_text())
    assert m["status"] == "done" and (m["stage"], m["arms"], m["t"]) == ("primary", "primary", T), (m.get("status"), m["stage"], m["t"])
    assert m["known_number"]["pass"] and m["known_number"]["failing_input_failed"], m["known_number"]
    idn = m["identity"]
    assert idn["max_abs_fall"] == 0.0 and idn["max_abs_change"] == 0.0 and idn["n_over_tol"] == 0, f"{T}: identity is not exactly 0: {idn}"
    assert idn["failing_input"] == "own_direct_m50" and idn["failing_input_failed"] and idn["failing_input_n_over_tol"] > 0, idn
    by_job = {j["job"]: j for j in items["jobs"]}
    want = {x for x, j in by_job.items() if T == "T_15" or j["birth"] == int(T[2:])}
    assert set(m["jobs"]) == want and m["n_jobs_done"] == m["n_jobs_to_read"] == len(want), (T, sorted(m["jobs"]), sorted(want))
    splits = set()
    for x in want:
        r, j = m["jobs"][x], by_job[x]
        assert set(r["falls"]) == ARMS5 and r["label"] == j["label"], (x, sorted(r["falls"]), r.get("label"))
        z = np.load(d / r["falls_path"])
        assert z["a_after"].shape == (5, len(j["J_tok"])) and (z["a_before"] == z["a_after"][list(z["arm"]).index("identity")]).all(), x
        if T == "T_15":
            assert "ceiling_split" not in r, x
        else:
            key = f"{j['pole']}__c{j['ceiling']['c']}"
            assert r["ceiling_split"] == key and key in m["ceiling"], (x, r.get("ceiling_split"))
            splits.add(key)
    assert set(m["ceiling"]) == splits, (T, sorted(m["ceiling"]), sorted(splits))
    for key in splits:
        cr = m["ceiling"][key]
        assert "side1" not in cr and set(cr["side2"]["falls"]) == {"identity", "own_direct_m50", "cross_direct_m50"} and np.isfinite(cr["I_ceil"]), (key, sorted(cr))
        z = np.load(d / cr["side2"]["falls_path"])
        assert (z["a_before"] == z["a_after"][list(z["arm"]).index("identity")]).all(), key
    lines = [json.loads(x) for x in (d / "samples" / "arms.jsonl").read_text().splitlines()]
    got = collections.Counter((x["unit"], x["arm"]) for x in lines)
    expect = {(f"{by_job[x]['pole']}__c{by_job[x]['c']}", a) for x in want for a in sorted(ARMS5 - {"identity"})} \
        | {(f"{k}_ceiling_side2", a) for k in splits for a in ("own_direct_m50", "cross_direct_m50")}
    assert set(got) == expect and set(got.values()) == {1}, (T, sorted(set(got) ^ expect))
    for x in lines:
        c0 = x["contexts"][0]
        assert c0["prompt"] and c0["marked_token"] and len(c0["top_ablated"]) == JR.TOP_SAMPLE and "a_before" in c0 and "a_after" in c0, x["unit"]
    summ = m["summary"]
    assert "freeze_rule" not in summ and set(summ["by_ranking_m"]["direct_m50"]) >= {"own", "cross", "random", "foreign", "I_median", "s_median",
                                                                                     "cross_minus_foreign", "n_cancelled"}, sorted(summ)
    return {"t": T, "n_jobs": len(want), "n_splits": len(splits), "n_sample_lines": len(lines), "known": m["known_number"]["ratio"],
            "known_failing_input": m["known_number"]["wrong_ratio"], "identity_max_abs_fall": idn["max_abs_fall"], "identity_n": idn["n_contexts"],
            "medians": summ["by_ranking_m"]["direct_m50"]}


def part_primary(out: pathlib.Path, tok, tokens, corpus) -> dict:
    models = {T: tiny_model("olmo3", seed=s, n_layers=N_LAYERS).requires_grad_(False) for T, s in zip(PRIMARY_LOADS, (0, 1, 2))}
    refs = {T: reference_a(mdl, tok, tokens, list(range(N_WINDOWS))) for T, mdl in models.items()}
    items = build_primary(refs)
    dp = out / "primary"
    ip, mp = dp / "primary_items.json", dp / "manifest.json"
    NH.dump(ip, items)
    lab = collections.Counter(j["label"] for j in items["jobs"])
    NH.dump(mp, {"counts": {"primary": len(items["jobs"]), "primary_related": lab["related"], "primary_unrelated": lab["unrelated"]}})
    sha = hashlib.sha256(ip.read_bytes()).hexdigest()
    step = {"T_15": "t15", "T_8": "t08", "T_11": "t11"}
    res = {}
    for T in PRIMARY_LOADS:
        d = dp / step[T]
        extra = [] if T == "T_15" else ["--from-step", str(dp / "t15")]
        JR.Job(d, args_for(d, "--arms", "primary", "--t", T, *extra, stage="primary", sha=sha), items, models[T], tok, corpus, "cpu").run()
        res[T] = check_primary(d, items, T)
    f15 = json.loads((dp / "t15" / "foreign.json").read_text())
    assert set(f15) == {j["job"] for j in items["jobs"]}
    for T in PRIMARY_LOADS[1:]:
        assert json.loads((dp / step[T] / "foreign.json").read_text()) == f15, f"{T} did not read the T_15 foreign choice"
    assert res["T_8"]["n_splits"] == 3 and res["T_8"]["n_jobs"] == 4 and res["T_11"]["n_splits"] == 4, (res["T_8"], res["T_11"])
    # the known-number gate stops a birth load on its failing input
    wrong = json.loads(json.dumps(items))
    wrong["known"]["a_bank"]["T_8"] = wrong["known"]["a_wrong"]["T_8"]
    raised = False
    try:
        JR.Job(dp / "wrong_t08", args_for(dp / "wrong_t08", "--arms", "primary", "--t", "T_8", "--from-step", str(dp / "t15"), stage="primary", sha=sha),
               wrong, models["T_8"], tok, corpus, "cpu").run()
    except AssertionError:
        raised = True
    assert raised and json.loads((dp / "wrong_t08" / "metrics.json").read_text())["known_number"]["pass"] is False, "the wrong bank did not stop the birth load"
    shutil.rmtree(dp / "wrong_t08")

    # the S3 freeze numbers on the dry S3 bank, then the verdict on the three loads
    sp = out / "s3" / "setting_items.json"
    r = subprocess.run([sys.executable, str(SCRIPTS / "s3_freeze.py"), "--run", str(out / "s3"), "--s1", str(out / "s1" / "metrics.json"), "--primary", str(ip),
                        "--setting", str(sp), "--out", str(out / "s3")], capture_output=True, text=True)
    assert r.returncode == 0 and "S3_FREEZE_OK" in r.stdout, r.stdout + r.stderr
    p50 = json.loads((out / "s3" / "S3.json").read_text())["by_m"]["m50"]["planted"]
    vout = out / "verdict"
    base = [sys.executable, str(SCRIPTS / "verdict.py"), "--primary-run", str(dp), "--s3-run", str(out / "s3"), "--s3-json", str(out / "s3" / "S3.json"),
            "--items", str(ip), "--setting", str(sp), "--manifest", str(mp), "--out", str(vout),
            f"--frozen-planted={p50['weighted_median']!r},{p50['weighted_median_lo']!r},{p50['weighted_median_hi']!r}"]
    r = subprocess.run(base, capture_output=True, text=True)
    print(r.stdout, end="")
    assert r.returncode == 0 and "VERDICT_OK" in r.stdout, r.stdout + r.stderr
    v = json.loads((vout / "verdict.json").read_text())
    assert all(g["ok"] for g in v["gates"].values()) and v["count"]["run"] == len(items["jobs"]) == v["count"]["live_at_birth"] + v["count"]["cancelled_at_birth"], v["count"]
    assert v["rules"]["R2"]["verdict"].startswith("specific"), ("R2 is not specific on the dry bank, so its failing input could not show a change", v["rules"]["R2"])
    assert (vout / "GRID.md").read_text().startswith("Jobs over 8: run 8"), "GRID.md does not open with the count line"
    fail = {}
    for flag, mode in ((["--swap", "own", "random"], "swap_own_random"), (["--shuffle-planted"], "shuffle_planted"), (["--swap", "cross", "foreign"], "swap_cross_foreign")):
        r = subprocess.run(base + flag, capture_output=True, text=True)
        line = [x for x in r.stdout.splitlines() if x.startswith("FAILING_INPUT")]
        assert r.returncode == 1 and line, (mode, r.returncode, r.stdout + r.stderr)
        f = json.loads((vout / f"failing_input_{mode}" / "verdict.json").read_text())
        hit = {"swap_own_random": not f["gates"]["positive_control"]["ok"] and not f["gates"]["cancel_share"]["ok"],
               "shuffle_planted": not f["gates"]["room"]["ok"], "swap_cross_foreign": f["rules"]["R2"]["verdict"].startswith("not specific")}[mode]
        assert hit, (mode, "the failing input did not make the verdict fail")
        fail[mode] = line[0]
        print(line[0])
    return {"loads": res, "verdict_count": v["count"], "verdict_rules": {k: x["verdict"] for k, x in v["rules"].items()},
            "verdict_gates_ok": {k: g["ok"] for k, g in v["gates"].items()}, "failing_inputs": fail, "wrong_bank_stopped_the_birth_load": raised}


# ------------------------------------------------------------------------------------------ the amend2 stage and verdict2
A2_LOADS = ("T_15", "T_8", "T_11")
A2_STEP = {"T_15": "t15", "T_8": "t08", "T_11": "t11"}
# (layer, a, b) of the dry planted pairs. On these 128-wide random models two neurons of one layer often share half their top-50 drivers
# (s near 0.5), so a pair's I can sit near 1 at a load. A probe of 145 candidate pairs at "T_8" and "T_11" (amend2_birth,
# /private/tmp/claude-501/reuse_dry/amend2/probe/out/T_{8,11}/metrics.json) gave these four I under 0 at both loads and shuffled-label I of
# 0.73 to 1.22, so room-b can pass on the dry bank and its failing input can be seen to fail. The choice says nothing about the real model.
A2_PAIRS = [(3, 21, 6), (3, 21, 16), (4, 28, 1), (5, 49, 46)]


def amend2_planted(refs: dict) -> list:
    """The A2_PAIRS planted pairs a_A + a_B (layers 3-5, tercile 0, the dry jobs' tercile). The pairs are read at every load, as S3's are, so on
    these unrelated random models the sides are chosen on all three: side J the 12 contexts where A's term exceeds |B's| by the most on every
    model, side O the same with A and B swapped, so J is driven by neuron a and O by neuron b at each load (the planted construction)."""
    out = []
    tok_rest = (np.arange(N_WINDOWS)[:, None] * WINDOW + np.arange(1, WINDOW)[None]).reshape(-1)
    for i, (L, x, y) in enumerate(A2_PAIRS):
        rs = [np.asarray(refs[T][L])[:, 1:, :].reshape(-1, np.asarray(refs[T][L]).shape[-1]).astype(np.float64) for T in A2_LOADS]
        sJ = np.min([r[:, x] - np.abs(r[:, y]) for r in rs], 0)
        sO = np.min([r[:, y] - np.abs(r[:, x]) for r in rs], 0)
        J, O = np.argsort(-sJ)[:12], np.argsort(-sO)[:12]
        assert (sJ[J] > 0).all() and (sO[O] > 0).all(), (L, x, y, "a side is not driven by its own neuron on every model")
        out.append({"unit": f"PL{L:02d}_{i:02d}", "layer": L, "a": x, "b": y, "lam": 1.0, "J_tok": tok_rest[J].tolist(), "O_tok": tok_rest[O].tolist()})
    return out


def check_amend2(d: pathlib.Path, items: dict, T: str, model, tok, tokens, primary_dir: pathlib.Path) -> dict:
    m = json.loads((d / "metrics.json").read_text())
    assert m["status"] == "done" and (m["stage"], m["t"]) == ("amend2", T) and m["arms"] == ("amend2_t15" if T == "T_15" else "amend2_birth"), (m.get("status"), m["arms"])
    assert m["known_number"]["pass"] and m["known_number"]["failing_input_failed"], m["known_number"]
    idn = m["identity"]
    assert idn["max_abs_fall"] == 0.0 and idn["max_abs_change"] == 0.0 and idn["n_over_tol"] == 0 and idn["failing_input_failed"], f"{T}: identity: {idn}"
    pm = json.loads((primary_dir / A2_STEP[T] / "metrics.json").read_text())
    want = {j["job"] for j in items["jobs"] if T == "T_15" or j["birth"] == int(T[2:])}
    assert set(m["jobs"]) == want, (T, sorted(m["jobs"]), sorted(want))
    kw = dict(tokens=tokens, length=RA.LENGTH, bos=tok.bos_token_id, batch=8, hw=WINDOW, device="cpu")
    by_job = {j["job"]: j for j in items["jobs"]}
    n_recomputed = 0
    for x in sorted(want):
        r, j = m["jobs"][x], by_job[x]
        z = np.load(d / r["falls_path"])
        zp = np.load(primary_dir / A2_STEP[T] / pm["jobs"][x]["falls_path"])
        names = list(zp["arm"])
        own = {(int(l), int(k)) for l, k in zip(zp["set_layer"][names.index("own_direct_m50")], zp["set_neuron"][names.index("own_direct_m50")]) if l >= 0}
        cross = [(int(l), int(k)) for l, k in zip(zp["set_layer"][names.index("cross_direct_m50")], zp["set_neuron"][names.index("cross_direct_m50")]) if l >= 0]
        DJ, DO = list(zip(z["own_layer"].tolist(), z["own_neuron"].tolist())), list(zip(z["cross_layer"].tolist(), z["cross_neuron"].tolist()))
        assert list(z["arm"]) == ["rj_direct_m50"] and set(DJ) == own and DO == cross, (T, x, "D(J) or D(O*) differs from the part 4 bank's")
        assert (z["a_before"] == zp["a_before"]).all(), (T, x, "a_P before differs from the part 4 bank's")
        R = [(int(l), int(k)) for l, k in zip(z["set_layer"][0], z["set_neuron"][0]) if l >= 0]
        # independent recomputation: a fresh capture of J, mean |a| per neuron, the matching rule
        L = j["layer"]
        cap = UR.capture(model, np.asarray(j["J_tok"][:12]), L, {L}, **kw)
        act = torch.stack([cap["a"][l].abs().mean(0) for l in range(L)]).numpy().astype(np.float64)
        assert RA.random_matched(DO, DJ, act) == R, (T, x, "R_J differs from the matching rule on a fresh capture")
        assert np.allclose(z["rj_energy"], act[tuple(np.array(R).T)], rtol=1e-5) and np.allclose(z["match_energy"], act[tuple(np.array(DO).T)], rtol=1e-5), (T, x)
        assert not set(R) & (set(DJ) | set(DO)) and len(set(R)) == 50, (T, x)
        n_recomputed += 1
    splits = {f"{j['pole']}__c{j['ceiling']['c']}" for j in items["jobs"]}
    if T == "T_15":
        assert set(m["ceiling"]) == splits and not m["planted"], (sorted(m["ceiling"]), sorted(splits))
        for k in splits:
            assert set(m["ceiling"][k]["side2"]["falls"]) == {"identity", "own_direct_m50", "cross_direct_m50"} and "side1" not in m["ceiling"][k], k
    else:
        assert not m["ceiling"] and set(m["planted"]) == {p["unit"] for p in items["planted"]}, (sorted(m["ceiling"]), sorted(m["planted"]))
        for u, v in m["planted"].items():
            assert set(v["labels"]["falls"]) == {"identity", "own_direct_m50", "cross_direct_m50", "random_direct_m50"}, u
            assert set(v["shuffled"]["falls"]) == {"identity", "own_direct_m50", "cross_direct_m50"}, u
    lines = [json.loads(x) for x in (d / "samples" / "arms.jsonl").read_text().splitlines()]
    got = collections.Counter((x["unit"], x["arm"]) for x in lines)
    expect = {(f"{by_job[x]['pole']}__c{by_job[x]['c']}", "rj_direct_m50") for x in want}
    if T == "T_15":
        expect |= {(f"{k}_ceiling_side2", a) for k in splits for a in ("own_direct_m50", "cross_direct_m50")}
    else:
        expect |= {(f"{p['unit']}__planted", a) for p in items["planted"] for a in ("own_direct_m50", "cross_direct_m50", "random_direct_m50")}
        expect |= {(f"{p['unit']}__planted_shuffled", a) for p in items["planted"] for a in ("own_direct_m50", "cross_direct_m50")}
    assert set(got) == expect and set(got.values()) == {1}, (T, sorted(set(got) ^ expect))
    assert "rj_direct_m50" in m["summary"] and "freeze_rule" not in m["summary"], sorted(m["summary"])
    return {"t": T, "n_jobs": len(want), "n_recomputed": n_recomputed, "n_splits": len(m["ceiling"]), "n_planted": len(m["planted"]), "forwards": m["forwards"],
            "identity_n": idn["n_contexts"], "rj_median_fall": m["summary"]["rj_direct_m50"]["median_fall"]}


def part_amend2(out: pathlib.Path) -> dict:
    tok = tiny_tokenizer(bos=True)
    tokens = fake_corpus(tok, N_WINDOWS + 2)
    offsets = np.arange(0, tokens.size + 37, 37)
    offsets[-1] = tokens.size
    corpus = {"tokens": tokens, "doc_offsets": offsets}
    models = {T: tiny_model("olmo3", seed=s, n_layers=N_LAYERS).requires_grad_(False) for T, s in zip(A2_LOADS, (0, 1, 2))}
    refs = {T: reference_a(mdl, tok, tokens, list(range(N_WINDOWS))) for T, mdl in models.items()}
    da = out / "amend2"
    dp = da / "primary"
    items = build_primary(refs)
    by_key = {}
    for j in items["jobs"]:
        j["ceiling"] = by_key.setdefault(f"{j['pole']}__c{j['ceiling']['c']}", j)["ceiling"]
    assert len(by_key) < len(items["jobs"]), "no split is named by two jobs"
    ip, mp = dp / "primary_items.json", dp / "manifest.json"
    NH.dump(ip, items)
    lab = collections.Counter(j["label"] for j in items["jobs"])
    NH.dump(mp, {"counts": {"primary": len(items["jobs"]), "primary_related": lab["related"], "primary_unrelated": lab["unrelated"]}})
    sha = hashlib.sha256(ip.read_bytes()).hexdigest()
    for T in A2_LOADS:
        d = dp / A2_STEP[T]
        extra = [] if T == "T_15" else ["--from-step", str(dp / "t15")]
        JR.Job(d, args_for(d, "--arms", "primary", "--t", T, *extra, stage="primary", sha=sha), items, models[T], tok, corpus, "cpu").run()
        check_primary(d, items, T)
    planted = amend2_planted(refs)
    k0, k1 = sorted(by_key)[:2]
    setting = {"ceilings": [{"split": k, "pole": by_key[k]["pole"], "layer": by_key[k]["layer"], "c": by_key[k]["ceiling"]["c"], "sides_tok": by_key[k]["ceiling"]["sides_tok"],
                             "shared_with_primary": k == k0} for k in (k0, k1)],
               "planted": planted, "known": items["known"], "job_counts": items["job_counts"]}
    sp = da / "setting_items.json"
    NH.dump(sp, setting)
    ssha = hashlib.sha256(sp.read_bytes()).hexdigest()
    # the S3-format bank at "T_15" and its S3.json (the T_10 step is part 3's, copied: s3_freeze.py reads one)
    d3 = da / "s3" / "t15"
    JR.Job(d3, args_for(d3, "--arms", "s3_t15", "--t", "T_15"), setting, models["T_15"], tok, corpus, "cpu").run()
    shutil.copytree(out / "s3" / "t10", da / "s3" / "t10")
    r = subprocess.run([sys.executable, str(SCRIPTS / "s3_freeze.py"), "--run", str(da / "s3"), "--s1", str(out / "s1" / "metrics.json"), "--primary", str(ip),
                        "--setting", str(sp), "--out", str(da / "s3")], capture_output=True, text=True)
    assert r.returncode == 0 and "S3_FREEZE_OK" in r.stdout, r.stdout + r.stderr
    # verdict.py on this primary bank: its verdict.json (count, R1, R3) is what verdict2 checks its own reading against
    p50 = json.loads((da / "s3" / "S3.json").read_text())["by_m"]["m50"]["planted"]
    r = subprocess.run([sys.executable, str(SCRIPTS / "verdict.py"), "--primary-run", str(dp), "--s3-run", str(da / "s3"), "--s3-json", str(da / "s3" / "S3.json"),
                        "--items", str(ip), "--setting", str(sp), "--manifest", str(mp), "--out", str(da / "verdict"),
                        f"--frozen-planted={p50['weighted_median']!r},{p50['weighted_median_lo']!r},{p50['weighted_median_hi']!r}"], capture_output=True, text=True)
    assert r.returncode in (0, 1) and "Traceback" not in r.stderr and (da / "verdict" / "verdict.json").exists(), r.stdout + r.stderr
    print("verdict.py on the amend2 dry primary bank:", [x for x in r.stdout.splitlines() if x.startswith(("Jobs over", "VERDICT"))])
    # the amend2 stage, T_15 first
    a2items = {**items, "planted": planted}
    res = {}
    for T in A2_LOADS:
        d = da / "run" / A2_STEP[T]
        a = args_for(d, "--arms", "amend2_t15" if T == "T_15" else "amend2_birth", "--t", T, "--setting", str(sp), stage="amend2", sha=sha)
        a.setting_sha256 = ssha
        JR.Job(d, a, a2items, models[T], tok, corpus, "cpu").run()
        res[T] = check_amend2(d, a2items, T, models[T], tok, tokens, dp)
    assert res["T_15"]["n_jobs"] == len(items["jobs"]) and res["T_8"]["n_jobs"] + res["T_11"]["n_jobs"] == len(items["jobs"]), res
    # verdict2: stops on a bank without the new arms; green on the amend2 bank; fails under each failing input
    vout = da / "verdict2"
    base = [sys.executable, str(SCRIPTS / "verdict2.py"), "--primary-run", str(dp), "--primary-verdict", str(da / "verdict" / "verdict.json"), "--s3-run", str(da / "s3"),
            "--s3-json", str(da / "s3" / "S3.json"), "--items", str(ip), "--setting", str(sp), "--out", str(vout)]
    r = subprocess.run(base + ["--amend2-run", str(dp)], capture_output=True, text=True)
    assert r.returncode == 4 and "AMEND2_BANK_INCOMPLETE" in r.stderr and not vout.exists(), (r.returncode, r.stdout, r.stderr[-2000:])
    stopped = r.stderr.splitlines()[0]
    print("verdict2 on the primary bank as the amend2 bank:", stopped)
    r = subprocess.run(base + ["--amend2-run", str(da / "run")], capture_output=True, text=True)
    print(r.stdout, end="")
    assert r.returncode == 0 and "VERDICT2_OK" in r.stdout, r.stdout + r.stderr
    v = json.loads((vout / "verdict2.json").read_text())
    assert all(g["ok"] for g in v["gates"].values()), {k: g["ok"] for k, g in v["gates"].items()}
    fail = {}
    for flag, mode in ((["--shuffle-planted"], "shuffle_planted"), (["--swap", "own", "rj"], "swap_own_rj")):
        r = subprocess.run(base + ["--amend2-run", str(da / "run")] + flag, capture_output=True, text=True)
        line = [x for x in r.stdout.splitlines() if x.startswith("FAILING_INPUT")]
        assert r.returncode == 1 and line, (mode, r.returncode, r.stdout + r.stderr)
        f = json.loads((vout / f"failing_input_{mode}" / "verdict2.json").read_text())
        hit = {"shuffle_planted": not f["gates"]["room_b"]["ok"], "swap_own_rj": not f["gates"]["positive_control_rj"]["ok"]}[mode]
        assert hit, (mode, "the failing input did not make the verdict fail")
        fail[mode] = line[0]
        print(line[0])
    return {"loads": res, "stopped_without_new_arms": stopped, "gates_ok": {k: g["ok"] for k, g in v["gates"].items()},
            "rules": {k: x["verdict"] for k, x in v["rules"].items()}, "failing_inputs": fail}


# ------------------------------------------------------------------------------------------ SPEC B and SPEC A: the heads and output stages
# The models of parts 6 and 7 are fp32: the head capture check compares an fp32 sum over heads with the post-norm attention block unite_route captures,
# and in bf16 that block's own rounding puts the check at 1.2e-3 to 3.4e-3 of the pole's max on these 64-wide models, over SPEC B's 1e-3 line (part 6
# prints the bf16 values). 48 heads of width 4 over 12 key-value heads: 48 x 4 is not the hidden 64, so the head split must come from the config, and h
# 40 has room below L. Model seeds: "T_15" 0, "T_10" and "T_8" 1, "T_11" 2.
AB_SEED = {"T_15": 0, "T_10": 1, "T_8": 1, "T_11": 2}
AB_STEP = {"T_15": "t15", "T_10": "t10", "T_11": "t11"}
AB_LOADS = ("T_15", "T_11")          # the two-load primary: T_15 and the one birth checkpoint of AB_PRIMARY
H_DRY, R_DRY = 10, 10                # the dry primary's frozen h and r
# On these random models most poles fail at least one gate of SPEC A or B (the random-matched readers move about half as much as the own readers, so
# the output cancel rule fires on most jobs; most planted pairs share their heads). A probe of 156 poles and 120 planted pairs at "T_11" and of the
# planted pairs at "T_15" (/private/tmp/claude-501/reuse_dry/ab/probe*, scratch/probe.py) gave these: poles live under both stages with own >= 0.3,
# random under 0.2, cross above foreign and above activity-matched, and both ceilings >= 0.6; pairs with head I under 0.25 and reader I under 0.35 at
# h 10 and r 10, shuffled-label I over 0.6. The choice says nothing about the real model.
AB_PRIMARY = [(4, 0, "+", 1, 11, "related"), (4, 5, "+", 1, 11, "unrelated"), (4, 10, "-", 1, 11, "related"), (4, 50, "-", 1, 11, "unrelated"),
              (4, 55, "+", 1, 11, "related"), (4, 95, "+", 1, 11, "unrelated")]          # (layer, neuron, sign, job c, birth, label)
AB_PAIRS = [(4, 117, 30), (3, 90, 82), (4, 81, 56), (3, 99, 121), (2, 90, 95), (4, 63, 95)]   # (layer, a, b) on the "T_11" model
# The sniper's pairs: a second probe of 149 layer-3 pairs on the "T_15" model under the sniper arm sets (scratch/probe2.py, probe2/rows.json) gave
# these four: 95th percentile of head I 0.20 at h 10 and of reader I 0.56 at r 10 (the T_15 ceiling medians are 0.63 and 1.04, so h 10 and r 10 have room),
# and shuffled-label 95th percentiles above the room line at every h and r (so --shuffle-planted leaves no h and no r), each bound cleared by >= 0.228.
# Layer 3 only: a layer-4 pair has one 128-wide reader layer, which cannot hold its r-50 random matches outside two r-50 sets.
SNIPER_PAIRS = [(3, 0, 69), (3, 45, 82), (3, 72, 17), (3, 99, 69)]   # (layer, a, b) on the "T_15" model


def tiny_fp32(seed: int):
    """neuron_off_dryrun.tiny_model's OLMo 3 config in fp32 with 48 heads of width 4 over 12 key-value heads."""
    import transformers as T
    from infra.aws_pg.jobs.neuron_off_dryrun import VOCAB
    torch.manual_seed(seed)
    cfg = T.Olmo3Config(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=N_LAYERS, num_attention_heads=48, num_key_value_heads=12,
                        head_dim=4, max_position_embeddings=512, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    return T.Olmo3ForCausalLM(cfg).float().eval().requires_grad_(False)


def args2(out, *extra, sha: str = "dry", ssha=None):
    a = JR2.parser().parse_args(["--out", str(out), "--items", "x", "--harvest-window", str(WINDOW), "--scan-tokens", str(N_WINDOWS * WINDOW),
                                 "--batch", "8", "--full-batch", "4", *extra])
    a.items_sha256, a.setting_sha256 = sha, ssha
    return a


def foreign_choice(items: dict, model) -> dict:
    """REUSE's rule (reuse_ablate.Job.foreign at T_15): per job the candidate pole nearest in ||G||^2 + ||U||^2."""
    layers, out = NH.text_tower(model).layers, {}
    for j in items["jobs"]:
        mlp = layers[j["layer"]].mlp
        load = lambda n: float(mlp.gate_proj.weight[n].float().pow(2).sum() + mlp.up_proj.weight[n].float().pow(2).sum())
        best = min(j["foreign_candidates"], key=lambda c: (abs(load(NH.parse_pole(c["pole"])[1]) - load(j["neuron"])), c["pole"]))
        out[j["job"]] = {"pole": best["pole"], "c": best["c"], "tok": best["tok"]}
    return out


def ab_planted(ref, pairs: list) -> list:
    """Planted pairs a_A + a_B on one model: side J the 12 contexts where A's term exceeds |B's| by the most, side O the same with A and B swapped."""
    tok_rest = (np.arange(N_WINDOWS)[:, None] * WINDOW + np.arange(1, WINDOW)[None]).reshape(-1)
    out = []
    for i, (L, x, y) in enumerate(pairs):
        r = np.asarray(ref[L])[:, 1:, :].reshape(-1, np.asarray(ref[L]).shape[-1]).astype(np.float64)
        sJ, sO = r[:, x] - np.abs(r[:, y]), r[:, y] - np.abs(r[:, x])
        J, O = np.argsort(-sJ)[:12], np.argsort(-sO)[:12]
        assert (sJ[J] > 0).all() and (sO[O] > 0).all(), (L, x, y, "a side is not driven by its own neuron")
        out.append({"unit": f"PL{L:02d}_{i:02d}", "layer": L, "a": x, "b": y, "lam": 1.0, "J_tok": tok_rest[J].tolist(), "O_tok": tok_rest[O].tolist()})
    return out


def sha_of(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def part_ab_runs(out: pathlib.Path) -> dict:
    """Both stages per load, as the pipeline runs them: the sniper at "T_15" and "T_10" (S1's job format, SNIPER_PAIRS, Q by REUSE's rule), then the
    two-load primary at "T_15" and "T_11" (AB_PRIMARY, AB_PAIRS), after REUSE's primary stage, an S3 bank and verdict.py on the same models: the
    verdict.json whose I_MLP R5h and R6o read."""
    tok = tiny_tokenizer(bos=True)
    tokens = fake_corpus(tok, N_WINDOWS + 2)
    offsets = np.arange(0, tokens.size + 37, 37)
    offsets[-1] = tokens.size
    corpus = {"tokens": tokens, "doc_offsets": offsets}
    M = {s: tiny_fp32(s) for s in sorted(set(AB_SEED.values()))}
    refs = {s: reference_a(m, tok, tokens, list(range(N_WINDOWS))) for s, m in M.items()}
    da = out / "ab"
    sitems = build_items(refs[AB_SEED["T_15"]], refs[AB_SEED["T_10"]], tokens)
    sitems["planted"] = ab_planted(refs[AB_SEED["T_15"]], SNIPER_PAIRS)
    sip = da / "sniper_items.json"
    NH.dump(sip, sitems)
    s1 = da / "sniper_s1"
    NH.dump(s1 / "foreign.json", foreign_choice(sitems, M[AB_SEED["T_15"]]))
    for T in ("T_15", "T_10"):
        d = da / "sniper" / AB_STEP[T]
        JR2.run_load(d, args2(d, "--stage", "heads", "output", "--arms", "sniper", "--t", T, "--from-step", str(s1), sha=sha_of(sip)), sitems, M[AB_SEED[T]], tok, corpus, "cpu")
    # the primary items, REUSE's primary stage at the two loads, its S3 bank and verdict.py
    items = build_primary({T: refs[AB_SEED[T]] for T in PRIMARY_LOADS}, AB_PRIMARY)
    by_key = {}
    for j in items["jobs"]:
        j["ceiling"] = by_key.setdefault(f"{j['pole']}__c{j['ceiling']['c']}", j)["ceiling"]
    planted = ab_planted(refs[AB_SEED["T_11"]], AB_PAIRS)
    dp = da / "primary"
    ip, mp = dp / "primary_items.json", dp / "manifest.json"
    NH.dump(ip, items)
    lab = collections.Counter(j["label"] for j in items["jobs"])
    NH.dump(mp, {"counts": {"primary": len(items["jobs"]), "primary_related": lab["related"], "primary_unrelated": lab["unrelated"]}})
    isha = sha_of(ip)
    for T in AB_LOADS:
        d = da / "reuse" / AB_STEP[T]
        extra = [] if T == "T_15" else ["--from-step", str(da / "reuse" / "t15")]
        JR.Job(d, args_for(d, "--arms", "primary", "--t", T, *extra, stage="primary", sha=isha), items, M[AB_SEED[T]], tok, corpus, "cpu").run()
        check_primary(d, items, T)
    k0, k1 = sorted(by_key)[:2]
    setting = {"ceilings": [{"split": k, "pole": by_key[k]["pole"], "layer": by_key[k]["layer"], "c": by_key[k]["ceiling"]["c"], "sides_tok": by_key[k]["ceiling"]["sides_tok"],
                             "shared_with_primary": k == k0} for k in (k0, k1)], "planted": planted, "known": items["known"], "job_counts": items["job_counts"]}
    sp = da / "setting_items.json"
    NH.dump(sp, setting)
    d3 = da / "s3" / "t15"
    JR.Job(d3, args_for(d3, "--arms", "s3_t15", "--t", "T_15"), setting, M[AB_SEED["T_15"]], tok, corpus, "cpu").run()
    shutil.copytree(out / "s3" / "t10", da / "s3" / "t10")
    r = subprocess.run([sys.executable, str(SCRIPTS / "s3_freeze.py"), "--run", str(da / "s3"), "--s1", str(out / "s1" / "metrics.json"), "--primary", str(ip),
                        "--setting", str(sp), "--out", str(da / "s3")], capture_output=True, text=True)
    assert r.returncode == 0 and "S3_FREEZE_OK" in r.stdout, r.stdout + r.stderr
    p50 = json.loads((da / "s3" / "S3.json").read_text())["by_m"]["m50"]["planted"]
    r = subprocess.run([sys.executable, str(SCRIPTS / "verdict.py"), "--primary-run", str(da / "reuse"), "--s3-run", str(da / "s3"), "--s3-json", str(da / "s3" / "S3.json"),
                        "--items", str(ip), "--setting", str(sp), "--manifest", str(mp), "--out", str(da / "reuse_verdict"),
                        f"--frozen-planted={p50['weighted_median']!r},{p50['weighted_median_lo']!r},{p50['weighted_median_hi']!r}"], capture_output=True, text=True)
    assert r.returncode in (0, 1) and "Traceback" not in r.stderr and (da / "reuse_verdict" / "verdict.json").exists(), r.stdout + r.stderr
    print("verdict.py on the A/B dry REUSE bank (the I_MLP source):", [x for x in r.stdout.splitlines() if x.startswith(("Jobs over", "VERDICT"))])
    pitems = {**items, "planted": planted}
    for T in AB_LOADS:
        d = dp / AB_STEP[T]
        JR2.run_load(d, args2(d, "--stage", "heads", "output", "--arms", "primary", "--t", T, "--setting", str(sp), "--from-step", str(da / "reuse" / "t15"),
                              "--h", str(H_DRY), "--r", str(R_DRY), sha=isha, ssha=sha_of(sp)), pitems, M[AB_SEED[T]], tok, corpus, "cpu")
    return {"da": da, "items": items, "pitems": pitems, "ip": ip, "sp": sp, "sitems": sitems, "sip": sip, "pv": da / "reuse_verdict" / "verdict.json", "M": M,
            "tok": tok, "tokens": tokens}


def ab_units(items: dict, T: str, sniper: bool) -> tuple:
    """(jobs, ceiling names, planted units) a load of the A/B stages must read."""
    jobs = [j for j in items["jobs"] if sniper or T == "T_15" or j["birth"] == int(T[2:])]
    if sniper:
        return jobs, ({j["job"] for j in items["jobs"] if j["ceiling"]["in_s1"]} if T == "T_15" else set()), ({p["unit"] for p in items["planted"]} if T == "T_15" else set())
    return jobs, {f"{j['pole']}__c{j['ceiling']['c']}" for j in jobs}, (set() if T == "T_15" else {p["unit"] for p in items["planted"]})


def check_heads(d: pathlib.Path, items: dict, T: str, sniper: bool, model, tok, tokens) -> dict:
    m = json.loads((d / "metrics.json").read_text())
    xs = list(RA2.HS) if sniper else [H_DRY]
    assert m["status"] == "done" and (m["stage"], m["t"], m["xs"]) == ("heads", T, xs), (m.get("status"), m["stage"], m["t"], m["xs"])
    assert m["known_number"]["pass"] and m["known_number"]["failing_input_failed"] and m["completeness"]["pass"], m["known_number"]
    idn, hc = m["identity"], m["head_check"]
    assert idn["max_abs_fall"] == 0.0 and idn["max_abs_change"] == 0.0 and idn["n_over_tol"] == 0, f"{T}: identity is not exactly 0: {idn}"
    assert idn["failing_input"] == f"own_direct_m{max(xs)}" and idn["failing_input_failed"] and idn["failing_input_n_over_tol"] > 0, idn
    assert hc["pass"] and hc["max_err"] <= RA2.HEAD_CHECK_TOL and hc["failing_input_failed"] and hc["failing_input_min_err"] > RA2.HEAD_CHECK_TOL, hc
    jobs, ceil, units = ab_units(items, T, sniper)
    assert set(m["jobs"]) == {j["job"] for j in jobs} and set(m["ceiling"]) == ceil and set(m["planted"]) == units, (T, sorted(m["ceiling"]), sorted(m["planted"]))
    n_off = 0
    for j in jobs:
        r = m["jobs"][j["job"]]
        z = np.load(d / r["falls_path"])
        names = list(z["arm"])
        assert len(names) == 1 + 5 * len(xs) and (z["a_before"] == z["a_after"][names.index("identity")]).all(), (j["job"], names)
        for h in xs:
            S = {x: [(int(l), int(k)) for l, k in zip(z["set_layer"][names.index(f"{x}_direct_m{h}")], z["set_neuron"][names.index(f"{x}_direct_m{h}")]) if l >= 0]
                 for x in ("own", "cross", "random", "foreign", "active")}
            assert all(len(v) == len(set(v)) == h for v in S.values()), (j["job"], h, {k: len(v) for k, v in S.items()})
            assert not (set(S["random"]) | set(S["active"])) & (set(S["own"]) | set(S["cross"])), (j["job"], h, "a matched head is in H(J) or H(O*)")
            for x in ("random", "active"):
                off = sum(a[0] != b[0] for a, b in zip(S[x], S["cross"]))
                assert off == r["off_layer"][f"direct_m{h}"][x], (j["job"], h, x, off)
                n_off += off
    # independent recomputation on a fresh capture: H(J) at every h and the head capture check of the first job
    j = jobs[0]
    unit, Jt = UR.pole_unit(j["pole"]), np.asarray(j["J_tok"])[:JR2.CAP]
    cap = RA2.capture(model, Jt, unit["layer"], {unit["layer"]}, tokens=tokens, length=RA.LENGTH, bos=tok.bos_token_id, batch=8, hw=WINDOW, device="cpu")
    hrows, rows = RA2.HeadRows(model), UR.Rows(model)
    z = np.load(d / m["jobs"][j["job"]]["falls_path"])
    names = list(z["arm"])
    for h in xs:
        got = [(int(l), int(k)) for l, k in zip(z["set_layer"][names.index(f"own_direct_m{h}")], z["set_neuron"][names.index(f"own_direct_m{h}")]) if l >= 0]
        assert RA.top_m(RA2.direct_H(hrows, rows, cap, unit).sum(0), h, hrows.H) == got, (j["job"], h, "H(J) differs from a fresh capture")
    assert float(RA2.head_completeness(hrows, rows, cap, unit).max()) <= RA2.HEAD_CHECK_TOL
    lines = [json.loads(x) for x in (d / "samples" / "arms.jsonl").read_text().splitlines()]
    want = len(jobs) * 5 * len(xs) + len(ceil) * 2 * len(xs) + len(units) * 5 * len(xs)
    assert len(lines) == want, (T, len(lines), want)
    for x in lines:
        c0 = x["contexts"][0]
        assert c0["prompt"] and c0["marked_token"] and len(c0["top_ablated"]) == min(JR2.TOP_SAMPLE, x["set_size"]) and {"head", "A", "out_norm"} <= set(c0["top_ablated"][0]), x["unit"]
    return {"t": T, "n_jobs": len(jobs), "n_ceilings": len(ceil), "n_planted": len(units), "identity_n": idn["n_contexts"], "head_check_max": hc["max_err"],
            "head_check_wrong_layer_median": hc["failing_input_median_err"], "n_off_layer": n_off, "forwards": m["forwards"], "n_sample_lines": len(lines)}


def check_output(d: pathlib.Path, items: dict, T: str, sniper: bool, model, tok, tokens) -> dict:
    m = json.loads((d / "metrics.json").read_text())
    xs = list(RA2.RS) if sniper else [R_DRY]
    last = m["n_layers"] - 1
    assert m["status"] == "done" and (m["stage"], m["t"], m["xs"]) == ("output", T, xs), (m.get("status"), m["stage"], m["t"], m["xs"])
    assert m["known_number"]["pass"] and m["known_number"]["failing_input_failed"] and m["completeness"]["pass"], m["known_number"]
    idn, rc = m["identity"], m["reader_check"]
    assert idn["max_rel"] == 0.0 and idn["n_over_tol"] == 0 and idn["failing_input_failed"] and idn["failing_input_n_over_tol"] > 0, f"{T}: identity: {idn}"
    assert rc["pass"] and rc["max_err"] <= RA2.READER_CHECK_TOL and rc["failing_input_failed"] and rc["failing_input_min_err"] > RA2.READER_CHECK_TOL, rc
    jobs, ceil, units = ab_units(items, T, sniper)
    assert set(m["jobs"]) == {j["job"] for j in jobs} and set(m["ceiling"]) == ceil and set(m["planted"]) == units, T
    n_read, n_skip = 0, 0
    abar = torch.as_tensor(np.load(d / "abar.npz")["abar"])
    for j in jobs:
        r = m["jobs"][j["job"]]
        if j["layer"] == last:
            assert r == {**{k: r[k] for k in ("pole", "layer", "tercile", "birth")}, "skipped": "no MLP layer above L", "seconds": 0.0, **({"label": j["label"]} if "label" in j else {}),
                         **({"ceiling_split": r["ceiling_split"]} if "ceiling_split" in r else {})}, (j["job"], r)
            n_skip += 1
            continue
        z = np.load(d / r["falls_path"])
        names = list(z["arm"])
        assert len(names) == 4 * len(xs) and (z["set_a_id"] == z["set_a_cap"])[np.broadcast_to((z["set_layer"] >= 0)[:, None, :], z["set_a_id"].shape)].all(), j["job"]
        assert np.allclose(z["a_after"], float(abar[j["layer"], j["neuron"]]) * (1.0 if j["sign"] == "+" else -1.0)), (j["job"], "a_P after is not the held value")
        for rr in xs:
            S = {x: [(int(l), int(k)) for l, k in zip(z["set_layer"][names.index(f"{x}_r{rr}")], z["set_neuron"][names.index(f"{x}_r{rr}")]) if l >= 0] for x in ("own", "cross", "random", "foreign")}
            assert all(len(v) == len(set(v)) == rr for v in S.values()) and all(l > j["layer"] for v in S.values() for l, _ in v), (j["job"], rr)
            assert not set(S["random"]) & (set(S["own"]) | set(S["cross"])), (j["job"], rr)
            assert sum(a[0] != b[0] for a, b in zip(S["random"], S["cross"])) == r["off_layer"][f"r{rr}"], (j["job"], rr, "off-layer matches differ from the record")
            st = r["reads"][f"own_r{rr}"]
            assert st["n_pos"] == 0 or 0.0 <= st["share_mean"] <= 1.0 + 1e-9, (j["job"], rr, st)     # no context with a_P > 0: the effects are NaN, as REUSE's falls
        n_read += 1
    # independent recomputation on a fresh capture: R(J) at every r and the reader capture check of the first readable job
    j = next(j for j in jobs if j["layer"] < last)
    unit, Jt = UR.pole_unit(j["pole"]), np.asarray(j["J_tok"])[:JR2.CAP]
    cap = RA2.capture_down(model, Jt, unit, None, True, tokens=tokens, length=RA.LENGTH, bos=tok.bos_token_id, batch=8, hw=WINDOW, device="cpu")
    rows = UR.Rows(model)
    C = RA2.direct_C(rows, cap, unit).abs().sum(0)
    W = rows.layers[0].mlp.gate_proj.weight.shape[0]
    z = np.load(d / m["jobs"][j["job"]]["falls_path"])
    names = list(z["arm"])
    for rr in xs:
        got = [(int(l), int(k)) for l, k in zip(z["set_layer"][names.index(f"own_r{rr}")], z["set_neuron"][names.index(f"own_r{rr}")]) if l >= 0]
        assert [(unit["layer"] + 1 + l, k) for l, k in RA.top_m(C, rr, W)] == got, (j["job"], rr, "R(J) differs from a fresh capture")
    assert float(RA2.reader_completeness(rows, cap, unit).max()) <= RA2.READER_CHECK_TOL
    lines = [json.loads(x) for x in (d / "samples" / "arms.jsonl").read_text().splitlines()]
    n_ceil = sum("skipped" not in m["ceiling"][k] for k in ceil)
    n_pl = sum("skipped" not in m["planted"][u] for u in units)
    want = n_read * 4 * len(xs) + n_ceil * 2 * len(xs) + n_pl * 5 * len(xs)
    assert len(lines) == want, (T, len(lines), want)
    for x in lines:
        c0 = x["contexts"][0]
        assert c0["prompt"] and c0["marked_token"] and len(c0["top_readers"]) == min(JR2.TOP_SAMPLE, x["set_size"]) and "a_before" in c0 and "a_after" in c0, x["unit"]
    return {"t": T, "n_jobs_read": n_read, "n_jobs_skipped": n_skip, "n_ceilings_read": n_ceil, "n_planted_read": n_pl, "identity_n": idn["n_contexts"],
            "reader_check_max": rc["max_err"], "reader_check_wrong_layer_median": rc["failing_input_median_err"], "forwards": m["forwards"], "n_sample_lines": len(lines)}


def run_verdicts(script: str, ctx: dict, out: pathlib.Path, tag: str, fails: list, sniper_fail: list) -> dict:
    """The verdict on the sniper bank (freeze rule, then its failing input), on an incomplete bank (exit 4, nothing written), on the primary bank (green),
    then under each failing input (exit 1 with its FAILING_INPUT line)."""
    da = ctx["da"]
    res = {}
    vs = out / "ab" / f"verdict_{tag}_sniper"
    base_s = [sys.executable, str(SCRIPTS / script), "--sniper", "--run", str(da / "sniper"), "--items", str(ctx["sip"]), "--out", str(vs)]
    r = subprocess.run(base_s, capture_output=True, text=True)
    print(r.stdout, end="")
    assert r.returncode == 0 and f"{tag.upper()}_SNIPER_OK" in r.stdout, r.stdout + r.stderr
    res["sniper_choice"] = json.loads((vs / f"sniper_{tag}.json").read_text())["freeze_rule"]["choice"]
    for flag in sniper_fail:
        r = subprocess.run(base_s + flag, capture_output=True, text=True)
        line = [x for x in r.stdout.splitlines() if x.startswith("FAILING_INPUT")]
        assert r.returncode == 1 and line, (flag, r.returncode, r.stdout + r.stderr)
        print(line[0])
        res[f"sniper {' '.join(flag)}"] = line[0]
    vout = out / "ab" / f"verdict_{tag}"
    base = [sys.executable, str(SCRIPTS / script), "--run", str(da / "primary"), "--items", str(ctx["ip"]), "--setting", str(ctx["sp"]), "--primary-verdict", str(ctx["pv"]),
            "--out", str(vout)]
    r = subprocess.run(base[:3] + [str(da / "sniper")] + base[4:], capture_output=True, text=True)
    assert r.returncode == 4 and "BANK_INCOMPLETE" in r.stderr and not vout.exists(), (r.returncode, r.stdout, r.stderr[-2000:])
    res["stopped_on_the_sniper_bank"] = r.stderr.splitlines()[0]
    print(f"{script} on the sniper bank as the primary bank: {res['stopped_on_the_sniper_bank']} (exit 4)")
    r = subprocess.run(base, capture_output=True, text=True)
    print(r.stdout, end="")
    assert r.returncode == 0 and f"VERDICT_{tag.upper()}_OK" in r.stdout, r.stdout + r.stderr
    v = json.loads((vout / f"verdict_{tag}.json").read_text())
    assert all(g["ok"] for g in v["gates"].values()) and (vout / f"GRID_{tag}.md").read_text().startswith("Jobs over"), {k: g["ok"] for k, g in v["gates"].items()}
    res["rules"] = {k: x["verdict"] for k, x in v["rules"].items()}
    for flag in fails:
        r = subprocess.run(base + flag, capture_output=True, text=True)
        line = [x for x in r.stdout.splitlines() if x.startswith("FAILING_INPUT")]
        assert r.returncode == 1 and line, (flag, r.returncode, r.stdout + r.stderr)
        print(line[0])
        res[" ".join(flag)] = line[0]
    return res


def part_heads(out: pathlib.Path, ctx: dict) -> dict:
    da, M, tok, tokens = ctx["da"], ctx["M"], ctx["tok"], ctx["tokens"]
    loads = {f"sniper {T}": check_heads(da / "sniper" / AB_STEP[T] / "heads", ctx["sitems"], T, True, M[AB_SEED[T]], tok, tokens) for T in ("T_15", "T_10")}
    loads.update({f"primary {T}": check_heads(da / "primary" / AB_STEP[T] / "heads", ctx["pitems"], T, False, M[AB_SEED[T]], tok, tokens) for T in AB_LOADS})
    for k, v in loads.items():
        print(f"heads load {k}: {v}")
    kw = dict(tokens=tokens, length=RA.LENGTH, bos=tok.bos_token_id, batch=8, hw=WINDOW, device="cpu")
    # the head layout: with layer 1's value rows of key-value group g zeroed, exactly the heads of that group output 0
    m = tiny_fp32(3)
    H, hd, kv = RA2.heads_of(m)
    g = 5
    with torch.no_grad():
        NH.text_tower(m).layers[1].self_attn.v_proj.weight[g * hd:(g + 1) * hd] = 0
    cap = RA2.capture(m, np.arange(10) * WINDOW + 40, 2, set(), **kw)
    zero = (cap["z"][1].abs().amax((0, 2)) == 0).tolist()
    assert zero == [h // (H // kv) == g for h in range(H)] and not any((cap["z"][0].abs().amax((0, 2)) == 0).tolist()), zero
    layout = f"value rows of key-value group {g} zeroed at layer 1: heads {[h for h in range(H) if zero[h]]} output exactly 0, the other {H - sum(zero)} do not"
    print("head layout:", layout)
    # bf16 (part 4's model): the hooks write in bf16, identity leaves a_P exactly unchanged, an ablation moves it; the head capture check in bf16
    mb = tiny_model("olmo3", seed=0, n_layers=N_LAYERS).requires_grad_(False)
    refb = reference_a(mb, tok, tokens, list(range(N_WINDOWS)))
    bf = {}
    for L, n in ((3, 5), (4, 11), (5, 2)):
        unit, toks = UR.pole_unit(f"L{L:02d}N{n}+"), top_tok(refb, L, n, 1.0, 12)
        zb = RA2.background_mean_z(mb, UR.background_positions(15, N_WINDOWS * WINDOW, WINDOW)["e1"], L, **kw)
        before = RA.readout(mb, toks, unit, None, **kw)
        same = RA2.readout_heads(mb, toks, unit, RA2.head_ablation([], zb, L, "cpu", identity=True), **kw)
        moved = RA2.readout_heads(mb, toks, unit, RA2.head_ablation([(L - 1, 0), (L - 1, 1), (L - 2, 2)], zb, L, "cpu"), **kw)
        assert (same["a"] == before["a"]).all() and np.abs(moved["a"] - before["a"]).max() > 0, (L, n)
        capb = RA2.capture(mb, toks, L, {L}, **kw)
        e, w = RA2.head_completeness(RA2.HeadRows(mb), UR.Rows(mb), capb, unit), RA2.head_completeness(RA2.HeadRows(mb), UR.Rows(mb), capb, unit, shift=1)
        bf[f"L{L:02d}N{n}+"] = {"check_max": float(e.max()), "check_median": float(np.median(e)), "wrong_layer_median": float(np.median(w))}
    print("bf16 (4 heads of 16, 2 key-value heads): identity readout equals the plain readout exactly, 3 heads ablated move a_P; head capture check "
          + "; ".join(f"{u} max {v['check_max']:.2e} median {v['check_median']:.2e} (wrong layer {v['wrong_layer_median']:.3f})" for u, v in bf.items()))
    v = run_verdicts("verdict_heads.py", ctx, out, "heads", [["--swap", "own", "random"], ["--shuffle-planted"], ["--swap", "cross", "foreign"], ["--swap", "cross", "active"]],
                     [["--swap", "own", "random"], ["--shuffle-planted"]])
    return {"loads": loads, "layout": layout, "bf16": bf, "verdicts": v}


def part_output(out: pathlib.Path, ctx: dict) -> dict:
    da, M, tok, tokens = ctx["da"], ctx["M"], ctx["tok"], ctx["tokens"]
    loads = {f"sniper {T}": check_output(da / "sniper" / AB_STEP[T] / "output", ctx["sitems"], T, True, M[AB_SEED[T]], tok, tokens) for T in ("T_15", "T_10")}
    loads.update({f"primary {T}": check_output(da / "primary" / AB_STEP[T] / "output", ctx["pitems"], T, False, M[AB_SEED[T]], tok, tokens) for T in AB_LOADS})
    for k, x in loads.items():
        print(f"output load {k}: {x}")
    v = run_verdicts("verdict_output.py", ctx, out, "output", [["--swap", "own", "random"], ["--shuffle-planted"], ["--swap", "cross", "foreign"]],
                     [["--swap", "own", "random"], ["--shuffle-planted"]])
    return {"loads": loads, "verdicts": v}


# ------------------------------------------------------------------------------------------ SPEC4 BIRTH: the birth stage and verdict_birth (part 8)
# Loads and models. "T_15", "T_8", "T_11" are parts 4 and 5's models (seeds 0, 1, 2). A pre-birth model is its birth model with the gate and up rows of
# every pole born at that birth scaled by B8_SCALE: upstream neurons are unchanged, so rho_a(O*) reads about 1; omega is linear in the rows, so
# rho_omega reads B8_SCALE for every driver; g and u scale by B8_SCALE, so rho_g = rho_u = B8_SCALE and a_P falls by about B8_SCALE^2 (rho_P under 0.5).
# B8_SCALE is 0.6, not the 0.2 of the brief: at 0.2 rho_omega(O*) would read 0.2 and gate L, whose band is [0.5, 2.0], would fail on the green bank.
# "T_7" = "T_8" scaled on the T_8 births (a prebirth_load, the T_5 code path); "T_10" = "T_11" scaled on the T_11 births, and a birth load of its own
# for the fresh cases born at "T_10" (so one birth load also banks pre-birth captures); "T_9" = "T_10" scaled on the T_10 births (prebirth_load).
B8_SCALE = 0.6
B8_STEP = {"T_15": "t15", "T_7": "t07", "T_8": "t08", "T_9": "t09", "T_10": "t10", "T_11": "t11"}
B8_ORDER = ("T_15", "T_7", "T_8", "T_9", "T_10", "T_11")
B8_ARMS = {"T_15": "birth_t15", "T_7": "prebirth_load", "T_8": "birth_load", "T_9": "prebirth_load", "T_10": "birth_load", "T_11": "birth_load"}
B8_WRONG = {"T_15": "T_8", "T_8": "T_11", "T_11": "T_15", "T_7": "T_15", "T_9": "T_15", "T_10": "T_15"}
# fresh dry cases on neurons no primary dry case uses: (layer, neuron, sign, c, birth, label, ceiling); the two L05N15+ cases share a split
B8_FRESH = [(3, 9, "+", 1, 8, "related", True), (4, 30, "-", 1, 8, "unrelated", False), (4, 40, "+", 1, 10, "related", True),
            (5, 15, "+", 1, 11, "unrelated", True), (5, 15, "+", 2, 11, "related", True), (3, 20, "-", 1, 11, "unrelated", False)]
# planted pairs (layer, a, b): a probe of 269 pairs on these models with B_t excluded (scratchpad probe8/probe.py, 31 passing) gave I' under 0.25 at
# "T_15", "T_8", "T_10" and "T_11" and shuffled-label I' over 0.6 at each, so room-c and room-c15 can pass on the dry bank and fail on their failing
# input. The choice says nothing about the real model.
B8_PAIRS = [(3, 45, 81), (4, 45, 114), (4, 18, 59), (5, 9, 92)]
B8_ARMSET = {"primary": {"identity", "own_direct_m50", "cross_direct_m50", "random_direct_m50", "foreign_direct_m50", "rj_direct_m50"},
             "fresh": {"identity", "own_direct_m50", "cross_direct_m50", "random_direct_m50"}}


def scaled(model, poles: list, factor: float):
    """A copy of the model with the gate and up rows of each (layer, neuron) multiplied by factor."""
    import copy
    m = copy.deepcopy(model)
    layers = NH.text_tower(m).layers
    with torch.no_grad():
        for L, n in sorted(set(poles)):
            layers[L].mlp.gate_proj.weight[n] *= factor
            layers[L].mlp.up_proj.weight[n] *= factor
    return m.requires_grad_(False)


def top_both(refs: dict, Ts: tuple, L: int, n: int, sign: float, k: int = 50) -> np.ndarray:
    """top_tok on the smallest of the pole's signed activations over the models `Ts`: contexts the pole reads high at each of them."""
    a = np.min([np.asarray(refs[T][L])[:, :, n].astype(np.float64) * sign for T in Ts], 0)
    a[:, 0] = -np.inf
    return np.argsort(-a.ravel())[:k].astype(np.int64)


def b8_primary(refs: dict, known: dict) -> dict:
    """Part 4's PRIMARY jobs in the primary format, with contexts the pole reads high at its birth load and at "T_15" (part 4 takes the birth model's
    alone, so most T_15 reads of its items have a_P <= 0 and the T_15 room cannot be read), and every split's sides the first naming job's."""
    jobs, sides = [], {}
    for i, (L, n, s, c, birth, label) in enumerate(PRIMARY):
        sg = 1.0 if s == "+" else -1.0
        t = top_both(refs, (f"T_{birth}", "T_15"), L, n, sg)
        k = sum(x[:3] == (L, n, s) and x[4] == birth for x in PRIMARY[:i])
        others = [x for x in range(0, 40, 3) if x != n][:3]
        foreign = [{"pole": f"L{L:02d}N{x}{s}", "c": 0, "ranks": list(range(12)), "tok": top_tok(refs["T_15"], L, x, sg)[:12].tolist()} for x in others]
        ceil = sides.setdefault(f"L{L:02d}N{n}{s}__c0", {"c": 0, "sides_tok": [t[28:35].tolist(), t[35:42].tolist()]})
        jobs.append({"job": f"L{L:02d}N{n}{s}:c{c}", "pole": f"L{L:02d}N{n}{s}", "layer": L, "neuron": n, "sign": s, "tercile": US.tercile(L), "c": c, "birth": birth,
                     "label": label, "J_tok": t[8 * k:8 * k + 8].tolist(), "O_tok": t[16:28].tolist(), "ceiling": ceil, "foreign_candidates": foreign})
    return {"jobs": jobs, "known": known, "job_counts": [[len(j["J_tok"]), len(j["O_tok"])] for j in jobs]}


def b8_planted(refs: dict) -> list:
    """The B8_PAIRS planted pairs, sides chosen as amend2_planted chooses them but over the four loads that read them ("T_15", "T_8", "T_10", "T_11")."""
    out = []
    tok_rest = (np.arange(N_WINDOWS)[:, None] * WINDOW + np.arange(1, WINDOW)[None]).reshape(-1)
    for i, (L, x, y) in enumerate(B8_PAIRS):
        rs = [np.asarray(refs[T][L])[:, 1:, :].reshape(-1, np.asarray(refs[T][L]).shape[-1]).astype(np.float64) for T in ("T_15", "T_8", "T_10", "T_11")]
        sJ = np.min([r[:, x] - np.abs(r[:, y]) for r in rs], 0)
        sO = np.min([r[:, y] - np.abs(r[:, x]) for r in rs], 0)
        J, O = np.argsort(-sJ)[:12], np.argsort(-sO)[:12]
        assert (sJ[J] > 0).all() and (sO[O] > 0).all(), (L, x, y, "a side is not driven by its own neuron on every model")
        out.append({"unit": f"PL{L:02d}_{i:02d}", "layer": L, "a": x, "b": y, "lam": 1.0, "J_tok": tok_rest[J].tolist(), "O_tok": tok_rest[O].tolist()})
    return out


def b8_fresh(refs: dict) -> list:
    """Fresh dry cases in the birth items' format, contexts from the model of each case's birth load (build_primary's layout)."""
    out = []
    for i, (L, n, s, c, birth, label, ceil) in enumerate(B8_FRESH):
        sg = 1.0 if s == "+" else -1.0
        t = top_tok(refs[f"T_{birth}"], L, n, sg)
        k = sum(x[:3] == (L, n, s) and x[4] == birth for x in B8_FRESH[:i])
        others = [x for x in range(41, 80, 3) if x != n][:3]
        foreign = [{"pole": f"L{L:02d}N{x}{s}", "c": 0, "ranks": list(range(12)), "tok": top_tok(refs["T_15"], L, x, sg)[:12].tolist()} for x in others]
        out.append({"job": f"L{L:02d}N{n}{s}:c{c}", "pole": f"L{L:02d}N{n}{s}", "layer": L, "neuron": n, "sign": s, "tercile": US.tercile(L), "c": c, "birth": birth,
                    "t_before": birth - 1, "flagged": False, "o_star": 0, "label": label, "J_tok": t[8 * k:8 * k + 8].tolist(), "O_tok": t[16:28].tolist(),
                    "ceiling": {"c": 0, "sides_tok": [t[28:35].tolist(), t[35:42].tolist()]} if ceil else None, "foreign_candidates": foreign})
    return out


def args_birth(out, arms: str, T: str, *extra, sha: str):
    from infra.aws_pg.jobs import reuse_birth as JB
    a = JB.parser().parse_args(["--out", str(out), "--stage", "birth", "--arms", arms, "--t", T, "--items", "x", "--harvest-window", str(WINDOW),
                                "--scan-tokens", str(N_WINDOWS * WINDOW), "--batch", "8", "--full-batch", "4", *extra])
    a.items_sha256 = sha
    return a


def check_birth(d: pathlib.Path, items: dict, T: str) -> dict:
    """One birth-stage load: gates, B_t, every case, split, pair and Part A bank with every arm, no set member in B_t, identity exactly 0."""
    from infra.pg import reuse_birth as RB
    m = json.loads((d / "metrics.json").read_text())
    n, arms = int(T[2:]), B8_ARMS[T]
    assert m["status"] == "done" and (m["stage"], m["arms"], m["t"]) == ("birth", arms, T), (m.get("status"), m["arms"], m["t"])
    assert m["known_number"]["pass"] and m["known_number"]["failing_input_failed"] and m["completeness"]["pass"] and m["completeness"]["failing_input_failed"], T
    hb = json.loads((d / "high_background.json").read_text())
    assert hb["count"] == round(0.01 * hb["n_entries"]) == len(hb["members"]) and hb["n_entries"] == (N_LAYERS - 1) * m["width"], (T, hb["count"], hb["n_entries"])
    mask = RB.mask_of_members(hb["members"], hb["n_up"], m["width"])
    ab = np.abs(np.load(d / "abar.npz")["abar"][:hb["n_up"]])
    assert np.isclose(ab[mask].min(), hb["cut"]) and (ab[~mask] <= hb["cut"]).all(), (T, "B_t is not the top 1% of |ā|")
    assert m["memory_max_allocated_gb"] == 0.0 and m["exclusion"]["n_in_B"] == 0 and m["exclusion"]["n_members"] > 0, (T, m.get("memory_max_allocated_gb"), m["exclusion"])
    P, F = items["primary"], items["fresh"]
    readP = P if T == "T_15" else [j for j in P if j["birth"] == n] if arms == "birth_load" else []
    readF = [j for j in F if j["birth"] == n] if arms == "birth_load" else []
    assert set(m["jobs"]) == {j["job"] for j in readP} and set(m["fresh"]) == {j["job"] for j in readF}, (T, sorted(m["jobs"]), sorted(m["fresh"]))
    n_units, n_lines, n_sets = 0, 0, 0
    for tag, cases, rec in (("primary", readP, m["jobs"]), ("fresh", readF, m["fresh"])):
        for j in cases:
            r = rec[j["job"]]
            z = np.load(d / r["falls_path"])
            assert set(z["arm"]) == B8_ARMSET[tag] == set(r["falls"]), (T, j["job"], sorted(z["arm"]))
            assert (z["a_before"] == z["a_after"][list(z["arm"]).index("identity")]).all() and not z["set_in_B"].any(), (T, j["job"])
            n_units += 1
            n_lines += len(z["arm"]) - 1
    want_sp = {f"{j['pole']}__c{j['ceiling']['c']}" for j in readP if j["ceiling"]}
    want_sf = {f"{j['pole']}__c{j['ceiling']['c']}" for j in readF if j["ceiling"]}
    assert set(m["ceiling"]) == want_sp and set(m["fresh_ceiling"]) == want_sf, (T, sorted(m["ceiling"]), sorted(m["fresh_ceiling"]))
    for store in ("ceiling", "fresh_ceiling"):
        for k, v in m[store].items():
            assert set(v["side2"]["falls"]) == {"identity", "own_direct_m50", "cross_direct_m50"}, (T, k)
            n_units += 1
            n_lines += 2
    if arms != "prebirth_load":
        assert set(m["planted"]) == {p["unit"] for p in items["planted"]}, (T, sorted(m["planted"]))
        for u, v in m["planted"].items():
            assert set(v["labels"]["falls"]) == {"identity", "own_direct_m50", "cross_direct_m50", "random_direct_m50"} and set(v["shuffled"]["falls"]) == {"identity", "own_direct_m50", "cross_direct_m50"}, (T, u)
            n_lines += 3 + 2
        idn = m["identity"]
        assert idn["max_abs_fall"] == 0.0 and idn["max_abs_change"] == 0.0 and idn["n_over_tol"] == 0, f"{T}: identity is not exactly 0: {idn}"
        assert idn["failing_input"] == "own_direct_m50" and idn["failing_input_failed"], (T, idn)
    else:
        assert m["identity"]["read"] is False and not m["planted"] and not m["jobs"] and not m["fresh"], (T, m["identity"])
    for p in sorted((d / "falls").glob("*.npz")):
        z = np.load(p)
        for i in range(len(z["arm"])):
            S = [(int(l), int(k)) for l, k in zip(z["set_layer"][i], z["set_neuron"][i]) if l >= 0]
            assert not any(mask[l, k] for l, k in S) and not z["set_in_B"][i].any(), (T, p.name, z["arm"][i], "a set member is in B_t")
            n_sets += bool(S)
    born = [j for j in P + F if j["birth"] == n and T != "T_15"]
    pre = [j for j in P + F if j["t_before"] == n]
    assert set(m["part_a"]) == {j["job"] for j in born + pre}, (T, sorted(m["part_a"]))
    for read, cases in (("birth", born), ("prebirth", pre)):
        for j in cases:
            for side in ("J", "O"):
                z = np.load(d / "prebirth" / f"{j['pole']}__c{j['c']}__{side}.npz")
                L = j["layer"]
                assert str(z["read"]) == read and str(z["t"]) == T and z["a_mean_abs"].shape == (L, m["width"]) and ("loadsG" in z.files) == (side == "J"), (T, j["job"], side)
                assert not any(mask[l, k] for l, k in zip(z["D_layer"].tolist(), z["D_neuron"].tolist())) and len(z["D_layer"]) == 50, (T, j["job"], side)
    lines = (d / "samples" / "arms.jsonl").read_text().splitlines() if (d / "samples" / "arms.jsonl").exists() else []
    assert len(lines) == n_lines, (T, len(lines), n_lines)
    for x in lines[:3]:
        c0 = json.loads(x)["contexts"][0]
        assert c0["prompt"] and c0["marked_token"] and "a_before" in c0 and "a_after" in c0 and all(not mask[t["layer"], t["neuron"]] for t in c0["top_ablated"]), T
    return {"t": T, "arms": arms, "n_cases": len(readP) + len(readF), "n_splits": len(want_sp) + len(want_sf), "n_planted": len(m["planted"]), "n_part_a": len(m["part_a"]),
            "n_sets": n_sets, "B_count": hb["count"], "B_cut": hb["cut"], "identity_n": m["identity"].get("n_contexts"), "forwards": m["forwards"], "captures": m["captures"]}


def part_birth(out: pathlib.Path) -> dict:
    from infra.aws_pg.jobs import reuse_birth as JB
    from infra.pg import reuse_birth as RB
    sc = RB.selfcheck()
    print("PASS birth pure logic:", sc)
    tok = tiny_tokenizer(bos=True)
    tokens = fake_corpus(tok, N_WINDOWS + 2)
    offsets = np.arange(0, tokens.size + 37, 37)
    offsets[-1] = tokens.size
    corpus = {"tokens": tokens, "doc_offsets": offsets}
    db = out / "birth"
    models = {T: tiny_model("olmo3", seed=s, n_layers=N_LAYERS).requires_grad_(False) for T, s in zip(A2_LOADS, (0, 1, 2))}
    born = lambda b: [(L, n) for (L, n, s, c, birth, lab) in PRIMARY if birth == b] + [(L, n) for (L, n, s, c, birth, lab, ce) in B8_FRESH if birth == b]
    models["T_7"] = scaled(models["T_8"], born(8), B8_SCALE)
    models["T_10"] = scaled(models["T_11"], born(11), B8_SCALE)
    models["T_9"] = scaled(models["T_10"], born(10), B8_SCALE)
    refs = {T: reference_a(mdl, tok, tokens, list(range(N_WINDOWS))) for T, mdl in models.items()}
    kp = ["L03N5+", "L04N11+", "L05N2+"]
    ktok = [top_tok(refs["T_15"], NH.parse_pole(p)[0], NH.parse_pole(p)[1], 1.0).tolist() for p in kp]
    val = lambda r, p, t: np.asarray(r[NH.parse_pole(p)[0]])[np.asarray(t) // WINDOW, np.asarray(t) % WINDOW, NH.parse_pole(p)[1]].tolist()
    known = {"poles": kp, "tok": ktok, "wrong_rule": "another load's model",
             "a_bank": {T: [val(refs[T], p, t) for p, t in zip(kp, ktok)] for T in B8_ORDER},
             "a_wrong": {T: [val(refs[B8_WRONG[T]], p, t) for p, t in zip(kp, ktok)] for T in B8_ORDER}}
    # the primary items and the primary stage on them: the bank ranked without exclusion that the exclusion gate's failing input reads
    pitems = b8_primary(refs, known)
    dp = db / "primary"
    ip = dp / "primary_items.json"
    NH.dump(ip, pitems)
    psha = hashlib.sha256(ip.read_bytes()).hexdigest()
    for T in A2_LOADS:
        d = dp / A2_STEP[T]
        extra = [] if T == "T_15" else ["--from-step", str(dp / "t15")]
        JR.Job(d, args_for(d, "--arms", "primary", "--t", T, *extra, stage="primary", sha=psha), pitems, models[T], tok, corpus, "cpu").run()
        check_primary(d, pitems, T)
    planted = b8_planted(refs)
    fresh = b8_fresh(refs)
    pn = {j["pole"][:-1] for j in pitems["jobs"]}
    assert not {f["pole"][:-1] for f in fresh} & pn, "a dry fresh case is on a primary neuron"
    primary = [{**j, "t_before": j["birth"] - 1} for j in pitems["jobs"]]
    items = {"primary": primary, "fresh": fresh, "planted": planted, "known": known, "job_counts": pitems["job_counts"]}
    bi = db / "birth_items.json"
    NH.dump(bi, items)
    bsha = hashlib.sha256(bi.read_bytes()).hexdigest()
    run = db / "run"
    res = {}
    for T in B8_ORDER:
        d = run / B8_STEP[T]
        extra = {"birth_t15": ["--primary-foreign", str(dp / "t15" / "foreign.json")], "birth_load": ["--from-step", str(run / "t15")], "prebirth_load": []}[B8_ARMS[T]]
        a = args_birth(d, B8_ARMS[T], T, *extra, sha=bsha)
        JB.Job(d, a, items, models[T], tok, corpus, "cpu").run()
        res[T] = check_birth(d, items, T)
        print(f"birth {T} ({B8_ARMS[T]}): {res[T]}")
    f15 = json.loads((run / "t15" / "foreign.json").read_text())
    assert {k: f15[k] for k in json.loads((dp / "t15" / "foreign.json").read_text())} == json.loads((dp / "t15" / "foreign.json").read_text()), "T_15 did not keep the primary Q"
    assert set(f15) == {j["job"] for j in primary + fresh}
    for T in ("T_8", "T_10", "T_11"):
        if (run / B8_STEP[T] / "foreign.json").exists():
            assert json.loads((run / B8_STEP[T] / "foreign.json").read_text()) == f15, f"{T} did not read the T_15 foreign choice"
    # the known-number gate stops a birth-stage load on its failing input
    wrong = json.loads(json.dumps(items))
    wrong["known"]["a_bank"]["T_7"] = wrong["known"]["a_wrong"]["T_7"]
    raised = False
    try:
        JB.Job(db / "wrong_t07", args_birth(db / "wrong_t07", "prebirth_load", "T_7", sha=bsha), wrong, models["T_7"], tok, corpus, "cpu").run()
    except AssertionError:
        raised = True
    assert raised and json.loads((db / "wrong_t07" / "metrics.json").read_text())["known_number"]["pass"] is False, "the wrong bank did not stop the pre-birth load"
    shutil.rmtree(db / "wrong_t07")

    # the builder's stop on a fresh case that shares a neuron with a primary case
    sys.path.insert(0, str(SCRIPTS))
    import build_items_birth as BIB
    bad = json.loads(json.dumps(items))
    bad["fresh"][0] = {**bad["fresh"][0], "pole": primary[0]["pole"], "job": primary[0]["pole"] + ":c9"}
    stopped = False
    try:
        BIB.disjoint_or_stop(bad["primary"], bad["fresh"])
    except SystemExit as e:
        stopped = e.code == 1
    assert stopped, "the builder did not stop on a fresh case on a primary neuron"
    BIB.disjoint_or_stop(items["primary"], items["fresh"])
    bad_path = db / "birth_items_shared_neuron.json"
    NH.dump(bad_path, bad)

    # verdict_birth: incomplete bank, green, each failing input, the shared-neuron items
    vout = db / "verdict"
    base = [sys.executable, str(SCRIPTS / "verdict_birth.py"), "--items", str(bi), "--primary-run", str(dp), "--primary-items", str(ip), "--out", str(vout)]
    inc = db / "run_incomplete"
    shutil.copytree(run, inc)
    shutil.rmtree(inc / "t09")
    r = subprocess.run(base + ["--run", str(inc)], capture_output=True, text=True)
    assert r.returncode == 4 and "BIRTH_BANK_INCOMPLETE" in r.stderr and not vout.exists(), (r.returncode, r.stdout[-2000:], r.stderr[-3000:])
    exits = {"incomplete": r.returncode}
    print("verdict_birth on a bank without t09:", r.stderr.splitlines()[0], "exit", r.returncode)
    shutil.rmtree(inc)
    r = subprocess.run(base + ["--run", str(run)], capture_output=True, text=True)
    print(r.stdout, end="")
    assert r.returncode == 0 and "VERDICT_BIRTH_OK" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
    exits["green"] = r.returncode
    v = json.loads((vout / "verdict_birth.json").read_text())
    assert all(g["ok"] for g in v["gates"].values() if g.get("ok") is not None), {k: g.get("ok") for k, g in v["gates"].items()}
    assert all(x["verdict"] != "not read" for x in v["rules"].values()), {k: x["verdict"] for k, x in v["rules"].items()}
    assert (vout / "GRID4.md").read_text().startswith("Primary 8:"), "GRID4.md does not open with the count line"
    # known answers of the pre-birth construction: rho_omega = rho_g = rho_u = B8_SCALE up to bf16 rounding, the rows' cosines 1, rho_a(O*) about 1
    pa = v["part_a"]
    assert len(pa) == len(primary) + len(fresh), (len(pa), "Part A was not read for every case")
    for x, q in pa.items():
        for k in ("rho_w_J", "rho_w_O", "rho_g", "rho_u"):
            assert abs(q[k] - B8_SCALE) < 0.02, (x, k, q[k], f"the construction gives {B8_SCALE}")
        assert q["cos_G"] > 0.999 and q["cos_U"] > 0.999 and q["rho_P"] < 0.5, (x, q["cos_G"], q["cos_U"], q["rho_P"])
    rho_a_O = float(np.median([q["rho_a_O"] for q in pa.values()]))
    assert abs(rho_a_O - 1.0) < 0.1, ("rho_a(O*) median is not about 1 on unchanged upstream neurons", rho_a_O)
    fail = {}
    for flag, mode in ((["--shuffle-planted"], "shuffle_planted"), (["--swap", "own", "random"], "swap_own_random"), (["--failing-input", "exclusion"], "exclusion"),
                       (["--failing-input", "gate_L"], "gate_L")):
        r = subprocess.run(base + ["--run", str(run)] + flag, capture_output=True, text=True)
        line = [x for x in r.stdout.splitlines() if x.startswith("FAILING_INPUT")]
        assert r.returncode == 1 and line, (mode, r.returncode, r.stdout[-3000:] + r.stderr[-3000:])
        f = json.loads((vout / f"failing_input_{mode}" / "verdict_birth.json").read_text())
        hit = {"shuffle_planted": not f["gates"]["room_c"]["ok"] and not f["gates"]["room_c15"]["ok"],
               "swap_own_random": not f["gates"]["positive_control primary"]["ok"] and not f["gates"]["positive_control fresh"]["ok"],
               "exclusion": not f["gates"]["exclusion"]["ok"] and f["gates"]["exclusion"]["failing_input"]["value"] > 0,
               "gate_L": not f["gates"]["gate_L primary"]["ok"] and not f["gates"]["gate_L fresh"]["ok"]}[mode]
        assert hit, (mode, "the failing input did not make the verdict fail")
        assert all(x["verdict"] == "not read" for k, x in f["rules"].items() if k.startswith("L1")) or mode != "gate_L", (mode, "L1 was read with gate L failed")
        fail[mode] = line[0]
        exits[mode] = r.returncode
        print(line[0])
    r = subprocess.run([sys.executable, str(SCRIPTS / "verdict_birth.py"), "--items", str(bad_path), "--primary-run", str(dp), "--primary-items", str(ip),
                        "--out", str(db / "verdict_bad"), "--run", str(run)], capture_output=True, text=True)
    assert r.returncode == 1 and "BUILD_ITEMS_STOP" in r.stderr and not (db / "verdict_bad").exists(), (r.returncode, r.stdout[-2000:], r.stderr[-2000:])
    exits["fresh_on_a_primary_neuron"] = r.returncode
    print("verdict_birth on items with a fresh case on a primary neuron:", r.stderr.strip().splitlines()[-1], "exit", r.returncode)
    print("verdict_birth exit codes:", exits)
    return {"selfcheck": sc, "loads": res, "exits": exits, "failing_inputs": fail, "gates_ok": {k: g.get("ok") for k, g in v["gates"].items()},
            "rules": {k: x["verdict"] for k, x in v["rules"].items()}, "rho_a_O_median": rho_a_O, "known_answer_scale": B8_SCALE}


# ------------------------------------------------------------------------------------------ SPEC5 DATING: the dating job and verdict_dating (part 9)
# Four fp32 loads "T_12".."T_15", each the same random model with gate rows scaled: the doubling poles' (layer 4) by D9_GROW, so a_P on every context
# moves as silu(f g) u, about f: adjacent ratios 1.25, 2.4, 1.67, a doubling between T_13 and T_14 only; the known poles' (layer 5) by D9_KNOWN, so each
# load's known bank differs from its neighbours'; the flat poles (layer 3) sit below both and read the same at every load. The fake exemplar bank holds the
# T_15 model's values at every checkpoint, so s_P is constant and r_J follows mean a_P on J (a real bank would scale with the pole and keep r_J flat).
D9_TS = (12, 13, 14, 15)
D9_GROW = (0.2, 0.25, 0.6, 1.0)
D9_KNOWN = (0.4, 0.55, 0.75, 1.0)
D9_KP = ["L05N2+", "L05N11+", "L05N40+"]
D9_CASES = [("L04N9+", 14, "primary", True), ("L03N7+", 15, "primary", False), ("L04N21-", 14, "fresh", True), ("L03N30-", 15, "fresh", False)]   # pole, birth, set, doubles
D9_MODES = {"known_shifted": "known_number", "capture_wrong_layer": "capture_identity", "scale_other_pole": "scale", "anchor_other_case": "anchor"}
DOUBLE_5B_MAX = 2.0   # amendment 5b: with s_P(t) the pole's own median at each load, the growing cases' adjacent ratios stay under the doubling factor


def gate_scaled(model, rows: dict):
    """A copy of the model with the gate row of each (layer, neuron) multiplied by its factor."""
    import copy
    m = copy.deepcopy(model)
    layers = NH.text_tower(m).layers
    with torch.no_grad():
        for (L, n), f in rows.items():
            layers[L].mlp.gate_proj.weight[n] *= f
    return m.requires_grad_(False)


def part_dating(out: pathlib.Path) -> dict:
    from infra.aws_pg.jobs import reuse_dating as JD
    tok = tiny_tokenizer(bos=True)
    tokens = fake_corpus(tok, N_WINDOWS + 2)
    offsets = np.arange(0, tokens.size + 37, 37)
    offsets[-1] = tokens.size
    corpus = {"tokens": tokens, "doc_offsets": offsets}
    db = out / "dating"
    base = tiny_model("olmo3", seed=0, dtype=torch.float32, n_layers=N_LAYERS).requires_grad_(False)
    grow = [NH.parse_pole(p)[:2] for p, _, _, d in D9_CASES if d]
    known_rows = [NH.parse_pole(p)[:2] for p in D9_KP]
    models = {T: gate_scaled(base, {**{x: fg for x in grow}, **{x: fk for x in known_rows}}) for T, fg, fk in zip(D9_TS, D9_GROW, D9_KNOWN)}
    refs = {T: reference_a(m, tok, tokens, list(range(N_WINDOWS))) for T, m in models.items()}
    val = lambda T, p, t: np.asarray(refs[T][NH.parse_pole(p)[0]])[np.asarray(t) // WINDOW, np.asarray(t) % WINDOW, NH.parse_pole(p)[1]].astype(np.float64)
    ktok = [top_tok(refs[15], NH.parse_pole(p)[0], NH.parse_pole(p)[1], 1.0).tolist() for p in D9_KP]
    wrong = {12: 13, 13: 12, 14: 13, 15: 14}
    known = {"poles": D9_KP, "tok": ktok, "wrong_rule": "the previous load's model; at T_12 the next one's",
             "a_bank": {f"T_{T}": [val(T, p, t).tolist() for p, t in zip(D9_KP, ktok)] for T in D9_TS},
             "a_wrong": {f"T_{T}": [val(wrong[T], p, t).tolist() for p, t in zip(D9_KP, ktok)] for T in D9_TS}}
    cases, explore = [], []
    for p, birth, tag, _ in D9_CASES:
        L, n, s = NH.parse_pole(p)
        sg = 1.0 if s == 0 else -1.0
        t50 = top_tok(refs[15], L, n, sg)
        J, O = t50[20:32], t50[:12]
        s15 = float(np.median(sg * val(15, p, t50)))
        cases.append({"job": f"{p}:c1", "pole": p, "c": 1, "layer": L, "neuron": n, "sign": "+" if s == 0 else "-", "tercile": US.tercile(L), "birth": birth,
                      "t_before": birth - 1, "set": tag, "J_ranks": list(range(20, 32)), "J_tok": J.tolist(), "O_ranks": list(range(12)), "O_tok": O.tolist(),
                      "scale_tok": t50.tolist(), "s_P": {f"T_{T}": s15 for T in D9_TS}})
        aJb, aJp = float(np.mean(sg * val(birth, p, J))), float(np.mean(sg * val(birth - 1, p, J)))
        explore.append({"job": f"{p}:c1", "set": tag, "tercile": US.tercile(L), "birth": birth, "rho_P_J": aJp / aJb, "aJ_b": aJb, "aJ_p": aJp})
    items = {"cases": cases, "known": known, "grid_ts": list(D9_TS), "sources": {"grid": ["t12_step320000", "t13_step560000", "t14_step1000000", "t15_step1413814"]}}
    ip, ep, bp = db / "dating_items.json", db / "explore_rhoP_O.json", db / "verdict_birth.json"
    NH.dump(ip, items)
    NH.dump(ep, explore)
    NH.dump(bp, {"cases": {"primary_birth": [{"job": c["job"], "cancelled": False} for c in cases if c["set"] == "primary"],
                           "fresh_birth": [{"job": c["job"], "cancelled": False} for c in cases if c["set"] == "fresh"]}})
    isha = hashlib.sha256(ip.read_bytes()).hexdigest()
    run = db / "run"
    loads = {}
    for T in D9_TS:
        d = run / f"t{T:02d}"
        a = JD.parser().parse_args(["--out", str(d), "--stage", "dating", "--t", f"T_{T}", "--items", "x", "--harvest-window", str(WINDOW), "--scan-tokens",
                                    str(N_WINDOWS * WINDOW), "--batch", "8", "--full-batch", "4", "--bg-windows", "6", "--bg-length", "32", "--bg-batch", "4"])
        a.items_sha256 = isha
        JD.Job(d, a, items, models[T], tok, corpus, "cpu").run()
        m = json.loads((d / "metrics.json").read_text())
        assert m["status"] == "done" and set(m["cases"]) == {c["job"] for c in cases}, (T, m.get("status"), sorted(m["cases"]))
        assert m["known_number"]["pass"] and m["known_number"]["failing_input_failed"] and m["capture_identity"]["max_abs_diff"] == 0.0, (T, m["known_number"], m["capture_identity"])
        assert m["captures"] == 48 + 2 * 20 + 4 * 24 + (4 * 50 + 200 if T == 15 else 0), (T, m["captures"])
        loads[T] = {"known_ratio": m["known_number"]["ratio"], "known_wrong_ratio": m["known_number"]["wrong_ratio"], "identity_wrong_layer": m["capture_identity"]["failing_input_max_abs_diff"],
                    "captures": m["captures"], "mean_J": {x: v["mean_J"] for x, v in m["cases"].items()}}
        if T == 15:
            assert m["scale_summary"]["max_rel_diff"] < 1e-4 and m["background_1m"]["n_positions"] == 6 * 31 and 0 <= m["background_1m"]["jaccard_B15_top1pct_1m"] <= 1, (m["scale_summary"], m["background_1m"])
            z = np.load(d / "background_1m.npz")
            for k in ("abar_1m", "mabs_1m", "share_pos", "share_over_half", "share_over_twice"):
                assert z[k].shape == (m["background"]["n_layers"], m["width"]) and np.isfinite(z[k]).all(), (k, z[k].shape)
            loads[T].update(scale=m["scale_summary"], background_1m={k: m["background_1m"][k] for k in ("jaccard_B15_top1pct_1m", "B15_median_share_pos", "B15_median_share_over_half")})
        print(f"dating T_{T}: known {m['known_number']['ratio']:.2e} (failing input {m['known_number']['wrong_ratio']:.3f}), identity 0 (wrong layer "
              f"{m['capture_identity']['failing_input_max_abs_diff']:.3g}), {m['captures']} captures")
    # the known-number gate stops a dating load on its failing input
    bad = json.loads(json.dumps(items))
    bad["known"]["a_bank"]["T_13"] = bad["known"]["a_wrong"]["T_13"]
    raised = False
    try:
        a = JD.parser().parse_args(["--out", str(db / "wrong_t13"), "--stage", "dating", "--t", "T_13", "--items", "x", "--harvest-window", str(WINDOW)])
        a.items_sha256 = isha
        JD.Job(db / "wrong_t13", a, bad, models[13], tok, corpus, "cpu").run()
    except AssertionError:
        raised = True
    assert raised and json.loads((db / "wrong_t13" / "metrics.json").read_text())["known_number"]["pass"] is False, "the wrong bank did not stop the dating load"
    shutil.rmtree(db / "wrong_t13")

    # verdict_dating's per-case rules on constructed values (validity, r_J(t) > 0, the floor) and D2's decision on constructed intervals
    sys.path.insert(0, str(SCRIPTS))
    import verdict_dating as VD
    ts, one = list(D9_TS), np.ones(4)
    x = VD.per_case(np.array([1e-4, 0.3, 0.35, 0.4]), one, np.array([0.005, 1, 1, 1]), ts, 14)       # T_12 invalid; read raw, 0.02 -> 0.3 is a doubling
    assert x["valid"] == [False, True, True, True] and np.isnan(x["r_J"][0]) and x["doubling_pairs"] == [] and x["b_act"] == 13, x
    assert not x["valid_tb_minus_2"] and np.isnan(x["d2J"]) and abs(x["largest_adjacent_ratio"] - 0.35 / 0.3) < 1e-12, x
    x = VD.per_case(np.array([0.01, 0.1, 0.25, 0.3]), one, np.array([0.005, 1, 1, 1]), ts, 15)         # read raw, r_J(T_12) = 2 would set b_act at T_12
    assert x["b_act"] == 14 and x["doubling_pairs"] == [13] and x["valid_tb"] and x["valid_tb_minus_2"] and abs(x["d2J"] - 0.3 / 0.1) < 1e-12, x
    x = VD.per_case(np.array([-0.1, 0.02, 0.05, 0.3]), one, one, ts, 14)                                # r_J(T_12) < 0; 0.02 -> 0.05 under the floor
    assert x["doubling_pairs"] == [14] and x["b_act"] == 15 and abs(x["d2J"] + 0.5) < 1e-12, x
    iv = lambda v, lo, hi: {"value": v, "lo": lo, "hi": hi}
    for bJ, bO, want in [(iv(1.5, 1.3, 1.8), iv(0.5, 0.4, 0.6), "H_old-decline refuted"), (iv(1.0, 0.9, 1.1), iv(0.9, 0.8, 0.95), "H_old-decline refuted"),
                         (iv(1.0, 0.9, 1.1), iv(0.6, 0.5, 0.7), "H_old-decline not refuted"), (iv(1.2, 1.0, 1.4), iv(0.6, 0.5, 0.7), "not separated"),
                         (iv(None, None, None), iv(0.6, 0.5, 0.7), "cannot be read")]:
        assert VD.d2_verdict(bJ, bO) == want, (bJ, bO, VD.d2_verdict(bJ, bO), want)
    print("verdict_dating per_case and d2_verdict: validity, r_J(t) > 0, the floor and each D2 outcome as constructed")

    # verdict_dating: incomplete bank, green with the construction's known answers, each failing input
    vout = db / "verdict"
    cmd = [sys.executable, str(SCRIPTS / "verdict_dating.py"), "--items", str(ip), "--explore", str(ep), "--birth-verdict", str(bp), "--out", str(vout)]
    inc = db / "run_incomplete"
    shutil.copytree(run, inc)
    shutil.rmtree(inc / "t13")
    r = subprocess.run(cmd + ["--run", str(inc)], capture_output=True, text=True)
    assert r.returncode == 4 and "DATING_BANK_INCOMPLETE" in r.stderr and not vout.exists(), (r.returncode, r.stdout[-2000:], r.stderr[-3000:])
    exits = {"incomplete": r.returncode}
    print("verdict_dating on a bank without t13:", r.stderr.splitlines()[0], "exit", r.returncode)
    shutil.rmtree(inc)
    r = subprocess.run(cmd + ["--run", str(run)], capture_output=True, text=True)
    print(r.stdout, end="")
    assert r.returncode == 0 and "VERDICT_DATING_OK" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
    exits["green"] = r.returncode
    v = json.loads((vout / "verdict_dating.json").read_text())
    assert all(g["ok"] for g in v["gates"].values()) and all(x["verdict"] != "not read" for x in v["rules"].values()), {k: g["ok"] for k, g in v["gates"].items()}
    assert (vout / "GRID5.md").read_text().startswith("Cases 4 "), "GRID5.md does not open with the count line"
    rows = {r_["job"]: r_ for r_ in v["cases"]}
    for p, birth, tag, doubles in D9_CASES:
        x = rows[f"{p}:c1"]
        rJ = np.asarray(x["r_J"])
        if doubles:
            assert x["doubling_pairs"] == [13] and x["doubling_at_tb"] and x["b_act"] == 14 and x["within1"], (p, x["doubling_pairs"], x["b_act"], rJ)
            assert np.allclose(rJ[1:] / rJ[:-1], np.asarray(D9_GROW[1:]) / np.asarray(D9_GROW[:-1]), rtol=0.1), (p, rJ, "r_J does not follow the gate scale")
        else:
            assert x["doubling_pairs"] == [] and np.allclose(rJ, rJ[0], rtol=1e-12) and x["b_act"] == 12 and not x["within1"] and abs(x["d2J"] - 1) < 1e-12, (p, x["doubling_pairs"], rJ)
    for tag in ("primary", "fresh"):
        d1 = v["rules"][f"D1 {tag}"]
        assert d1["value"] == 0.5 and d1["n_with_doubling"] == 1 and d1["n_cases"] == 2 and d1["verdict"] == "not separated", (tag, d1)
        assert v["reported"]["doubling_pairs"][tag]["n_pairs"] == 1 and v["reported"]["doubling_pairs"][tag]["share_at_tb"] == 1.0, (tag, v["reported"]["doubling_pairs"][tag])
        d2 = v["rules"][f"D2 {tag}"]
        assert d2["verdict"] == "H_old-decline refuted" and d2["r_O_ratio"]["lo"] >= 0.8 and d2["n_cases"] == 2 and d2["n_without_valid_tb_minus_2"] == 0, (tag, d2)
        assert set(v["reported"]["valid_share"][tag]["all"].values()) == {1.0}, (tag, v["reported"]["valid_share"][tag])
    sc = v["gates"]["scale"]["value"]
    assert sc["n_nonpositive_T15"] == 0 and sc["min_valid_share_from_T6"] == 1.0 and v["gates"]["anchor"]["median_abs_diff"] <= 1e-2, (sc, v["gates"]["anchor"])
    fail = {}
    for mode, prefix in D9_MODES.items():
        r = subprocess.run(cmd + ["--run", str(run), "--failing-input", mode], capture_output=True, text=True)
        line = [x for x in r.stdout.splitlines() if x.startswith("FAILING_INPUT")]
        assert r.returncode == 1 and line, (mode, r.returncode, r.stdout[-3000:] + r.stderr[-3000:])
        f = json.loads((vout / f"failing_input_{mode}" / "verdict_dating.json").read_text())
        tg = {k: g for k, g in f["gates"].items() if k.split()[0] == prefix}
        assert tg and not any(g["ok"] for g in tg.values()) and all(x["verdict"] == "not read" for k, x in f["rules"].items() if k != "D1 decision"), (mode, {k: g["ok"] for k, g in tg.items()})
        fail[mode] = line[0]
        exits[mode] = r.returncode
        print(line[0])
    print("verdict_dating exit codes:", exits)

    # amendment 5b: the scale stage at every load on the same models, then verdict_dating --scale-run
    srun = db / "scale_run"
    ref_med = {(T, c["pole"]): float(np.median((1.0 if c["sign"] == "+" else -1.0) * val(T, c["pole"], c["scale_tok"]))) for T in D9_TS for c in cases}
    for T in D9_TS:
        d = srun / f"t{T:02d}"
        a = JD.parser().parse_args(["--out", str(d), "--stage", "scale", "--t", f"T_{T}", "--items", "x", "--harvest-window", str(WINDOW), "--scan-tokens",
                                    str(N_WINDOWS * WINDOW), "--batch", "8", "--full-batch", "4"])
        a.items_sha256 = isha
        JD.Job(d, a, items, models[T], tok, corpus, "cpu").run()
        m = json.loads((d / "metrics.json").read_text())
        assert m["status"] == "done" and m["stage"] == "scale" and m["cases"] == {} and not (d / "dating").exists(), (T, m.get("status"), sorted(m["cases"]))
        assert m["n_done"] == m["n_to_read"] == {"cases": 0, "scale_poles": 4} and m["captures"] == 48 + 2 * 20 + 4 * 50, (T, m["n_done"], m["captures"])
        assert m["known_number"]["pass"] and m["known_number"]["failing_input_failed"] and m["capture_identity"]["max_abs_diff"] == 0.0, (T, m["known_number"], m["capture_identity"])
        dev = {c["pole"]: abs(m["scale"][c["pole"]]["median_captured"] - ref_med[T, c["pole"]]) / abs(ref_med[T, c["pole"]]) for c in cases}
        assert max(dev.values()) < 1e-4, (T, "the scale stage's medians are not the model's own at this load", dev)
        loads[f"scale T_{T}"] = {"captures": m["captures"], "medians": {x: v["median_captured"] for x, v in m["scale"].items()}, "max_rel_dev_reference": max(dev.values())}
        print(f"scale T_{T}: known {m['known_number']['ratio']:.2e}, identity 0, {m['captures']} captures, medians "
              + ", ".join(f"{x} {v['median_captured']:.4g}" for x, v in m["scale"].items()))
    v5 = vout / "amend5b"
    cmd5 = cmd + ["--run", str(run), "--scale-run"]
    inc = db / "scale_run_incomplete"
    shutil.copytree(srun, inc)
    shutil.rmtree(inc / "t13")
    r = subprocess.run(cmd5 + [str(inc)], capture_output=True, text=True)
    assert r.returncode == 4 and "DATING_BANK_INCOMPLETE" in r.stderr and "scale_run_incomplete/t13" in r.stderr and not v5.exists(), (r.returncode, r.stdout[-2000:], r.stderr[-3000:])
    exits["5b incomplete"] = r.returncode
    print("verdict_dating --scale-run on a scale bank without t13:", r.stderr.splitlines()[1].strip(), "exit", r.returncode)
    shutil.rmtree(inc)
    r = subprocess.run(cmd5 + [str(srun)], capture_output=True, text=True)
    print(r.stdout, end="")
    assert r.returncode == 0 and "VERDICT_DATING_OK" in r.stdout and "AMENDMENT 5b" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
    exits["5b green"] = r.returncode
    w = json.loads((v5 / "verdict_dating.json").read_text())
    assert w["amendment"] == "5b" and w["inputs"]["scale_run"] == str(srun.resolve()), (w.get("amendment"), w["inputs"].get("scale_run"))
    assert all(g["ok"] for g in w["gates"].values()) and all(x["verdict"] != "not read" for x in w["rules"].values()), {k: g["ok"] for k, g in w["gates"].items()}
    assert {f"{g} scale t{T:02d}" for g in ("known_number", "completeness", "capture_identity") for T in D9_TS} <= set(w["gates"]), sorted(w["gates"])
    assert (v5 / "GRID5.md").read_text().splitlines()[2].startswith("# SPEC5 DATING verdict (amendment 5b)"), "GRID5.md is not labelled amendment 5b"
    sc5 = w["gates"]["scale"]
    assert sc5["value"]["max_abs_diff_T15"] == 0.0 and sc5["value"]["n_nonpositive_T15"] == 0 and sc5["value"]["min_valid_share_from_T9"] == 1.0, sc5["value"]
    assert sc5["failing_input"]["value"] > 1e-5 and set(sc5["bank_comparison"]["by_t"]) == {f"T_{T}" for T in D9_TS}, (sc5["failing_input"], sc5["bank_comparison"])
    an5 = w["gates"]["anchor"]
    assert an5["value"] == an5["median_abs_diff"] and an5["max_abs_diff"] >= an5["value"] and an5["failing_input"]["value"] > 1e-2, an5
    rows5 = {r_["job"]: r_ for r_ in w["cases"]}
    for p, birth, tag, doubles in D9_CASES:
        x, x0 = rows5[f"{p}:c1"], rows[f"{p}:c1"]
        assert np.allclose(x["s_P"], [ref_med[T, p] for T in D9_TS], rtol=1e-4), (p, x["s_P"], "s_P(t) is not the scale run's median at each load")
        if doubles:
            assert x0["doubling_pairs"] == [13] and x["doubling_pairs"] == [] and x["largest_adjacent_ratio"] < DOUBLE_5B_MAX, (p, x["r_J"], x["largest_adjacent_ratio"])
        else:
            assert np.allclose(x["r_J"], x0["r_J"], rtol=1e-4) and np.allclose(x["r_J"], x["r_J"][0], rtol=1e-12) and x["doubling_pairs"] == [], (p, x["r_J"], x0["r_J"])
    print("verdict_dating --scale-run: s_P(t) the scale run's medians; flat cases' r_J unchanged; growing cases' largest adjacent ratio "
          + ", ".join(f"{x} {rows5[x]['largest_adjacent_ratio']:.3f}" for x in rows5 if x.startswith("L04")) + " (no doubling: the scale moves with the pole)")
    fail5 = {}
    for mode, prefix in D9_MODES.items():
        r = subprocess.run(cmd5 + [str(srun), "--failing-input", mode], capture_output=True, text=True)
        line = [x for x in r.stdout.splitlines() if x.startswith("FAILING_INPUT")]
        assert r.returncode == 1 and line, (mode, r.returncode, r.stdout[-3000:] + r.stderr[-3000:])
        f = json.loads((v5 / f"failing_input_{mode}" / "verdict_dating.json").read_text())
        tg = {k: g for k, g in f["gates"].items() if k.split()[0] == prefix}
        assert tg and not any(g["ok"] for g in tg.values()) and all(x["verdict"] == "not read" for k, x in f["rules"].items() if k != "D1 decision"), (mode, {k: g["ok"] for k, g in tg.items()})
        fail5[mode] = line[0]
        exits[f"5b {mode}"] = r.returncode
        print(line[0])
    print("verdict_dating exit codes:", exits)
    return {"loads": loads, "exits": exits, "failing_inputs": fail, "failing_inputs_5b": fail5, "D1": {tag: v["rules"][f"D1 {tag}"]["value"] for tag in ("primary", "fresh")},
            "r_J": {x: r_["r_J"] for x, r_ in rows.items()}, "doubling_pairs": {x: r_["doubling_pairs"] for x, r_ in rows.items()},
            "r_J_5b": {x: r_["r_J"] for x, r_ in rows5.items()}, "s_P_5b": {x: r_["s_P"] for x, r_ in rows5.items()}}


# ------------------------------------------------------------------------------------------ SPEC6 GENERIC: held scale, cuts, arms, the generic job and verdict_generic (part 10)
# The tiny-model run checks the job against SPEC4's code on the same model: the free-scale arms under g1 must reproduce the birth_t15 load's sets and
# a_P exactly (known_spec4_a and known_spec4_b read 0). Its gates whose lines are set for the 7B model (the known null 0.163, the room) are not
# expected to pass on a 128-wide random model, so verdict_generic is run on it only to show it reads the job's bank (exit 0 or 1, never 3 or 4).
# The verdict's green run and its failing inputs use a constructed bank (G10 below) written through jobs.reuse_generic.bank_arm, whose falls,
# sets and Part A arrays are chosen so that every gate passes and each failing input makes its target fail.
G10_LAYERS, G10_W, G10_N = 32, 200, 10
G10_CASE_L = (4, 6, 8, 10, 12, 14, 16, 18, 24, 26, 28, 30)
G10_PAIR_L = (5, 5, 5, 15, 15, 15, 26, 26, 26)
G10_PAIR_I = {("none", "free"): 0.16, ("none", "held"): 0.10, ("g1", "free"): 0.06, ("g1", "held"): 0.05, ("g3", "free"): 0.045, ("g3", "held"): 0.04,
              ("pl1", "free"): 0.065, ("pl1", "held"): 0.055}
G10_W1_SHARED = 15        # members of D'(J) taken from the omega top 50: W1 reads 0.30 on every case


def held_decoder_check() -> dict:
    """A tiny fp32 OLMo 3 and the real decoder layer: with ten layer-1 neurons zeroed and layer 1 held, layer 1's output equals its MLP input plus
    weight * y_ablated / sqrt(mean(y_clean^2) + eps) at positions >= 1, by hand from captured tensors; free differs. On the bf16 model the
    held-scale readout of the empty set equals the clean a_P exactly, and the shifted inv does not."""
    from infra.pg import reuse_generic as RG
    tok = tiny_tokenizer(bos=True)
    tokens = fake_corpus(tok, N_WINDOWS + 2)
    toks = np.array([w * WINDOW + 60 + w for w in range(6)])
    kw = dict(tokens=tokens, length=RA.LENGTH, bos=tok.bos_token_id, batch=8, hw=WINDOW, device="cpu")
    out = {}
    model = tiny_model("olmo3", seed=3, dtype=torch.float32, n_layers=N_LAYERS).requires_grad_(False)
    tower = NH.text_tower(model)
    layers = list(tower.layers)
    L, l, start = 3, 1, 1
    abl = RA.ablation_dict([(l, k) for k in range(10)], "zero", torch.zeros(L, 128), L, "cpu")
    x, _ = UR.pack(tokens, toks, RA.LENGTH, tok.bos_token_id, WINDOW, "cpu")
    rec = {}
    hs_ = [layers[l].mlp.register_forward_pre_hook(lambda m, a: rec.__setitem__("h1", a[0].clone())),
           layers[l].mlp.register_forward_hook(lambda m, a, o: rec.__setitem__("y", o.clone())),
           layers[l].register_forward_hook(lambda m, a, o: rec.__setitem__("out", (o[0] if isinstance(o, tuple) else o).clone()))]
    with torch.no_grad(), RG.held_scale(layers, [l], start) as hs, UR.upto_layer(tower, L):
        hs.record()
        NH.forward(model, x, False)
        y_clean = rec["y"]
        hs.hold()
        with RA.replaced(layers, abl, start):
            NH.forward(model, x, False)
        hs.done()
        held = {k: v.clone() for k, v in rec.items()}
        hs.free()
        with RA.replaced(layers, abl, start):
            NH.forward(model, x, False)
        free_out = rec["out"].clone()
    for h in hs_:
        h.remove()
    norm = layers[l].post_feedforward_layernorm
    eps = norm.variance_epsilon
    hand = held["h1"].double() + norm.weight.detach().double() * held["y"].double() / torch.sqrt(y_clean.double().pow(2).mean(-1, keepdim=True) + eps)
    err = float((held["out"][:, start:].double() - hand[:, start:]).abs().max() / hand[:, start:].abs().max())
    moved = float((free_out[:, start:] - held["out"][:, start:]).abs().max())
    assert err <= 1e-5 and moved > 1e-3, (err, moved, "the held decoder layer is not residual + weight y_abl inv_clean, or free equals held")
    assert not torch.equal(held["y"], y_clean), "the zeroed neurons did not change layer 1's MLP output"
    out["decoder_fp32"] = {"max_rel_error": err, "free_minus_held_max": moved}
    mb = tiny_model("olmo3", seed=3, n_layers=N_LAYERS).requires_grad_(False)
    lb = list(NH.text_tower(mb).layers)
    unit = UR.pole_unit("L03N5+")
    ident = RA.ablation_dict([], "mean", torch.zeros(L, 128), L, "cpu", identity=True)
    with RG.held_scale(lb, range(L), start) as hs:
        hs.record()
        a0 = RA.readout(mb, toks, unit, None, **kw)["a"]
        hs.hold(set(ident))
        a1 = RA.readout(mb, toks, unit, ident, **kw)["a"]
        hs.done()
        hs.hold(set(ident), shift=1)
        a2 = RA.readout(mb, toks, unit, ident, **kw)["a"]
        hs.done()
    assert np.array_equal(a0, a1) and not np.array_equal(a0, a2), ("bf16: the held empty set is not exact, or the shifted inv is", a0, a1, a2)
    out["identity_bf16"] = {"max_abs_diff": float(np.abs(a0 - a1).max()), "shifted_max_abs_diff": float(np.abs(a0 - a2).max())}
    print(f"held decoder layer, fp32: rel error {err:.1e} against the hand computation, free - held {moved:.3f}; bf16 empty set exact, shifted inv "
          f"moves a_P by {out['identity_bf16']['shifted_max_abs_diff']:.3g}")
    return out


def check_generic(d: pathlib.Path, items: dict, d4: pathlib.Path) -> dict:
    """The tiny-model generic bank: gates, every unit and (arm, exclusion, scale) npz, the free g1 arms equal to SPEC4's birth_t15 bank, the cuts."""
    from infra.aws_pg.jobs import reuse_generic as JG
    m = json.loads((d / "metrics.json").read_text())
    assert m["status"] == "done" and (m["stage"], m["t"]) == ("generic", "T_15"), (m.get("status"), m["stage"])
    g = m["gates"]
    assert g["held_identity"]["ok"] and g["held_identity"]["value"] == 0.0 and g["held_identity"]["every_unit"]["max_abs_change"] == 0.0, g["held_identity"]
    assert g["held_completeness"]["ok"] and g["held_completeness"]["value"] <= 1e-4 < g["held_completeness"]["failing_input"]["value"], g["held_completeness"]
    assert g["known_spec4_a"]["value"] == 0.0 and g["known_spec4_a"]["failing_input_failed"], ("the clean a_P differ from SPEC4's on the same model", g["known_spec4_a"])
    assert g["known_spec4_b"]["value"] == 0.0 and g["known_spec4_b"]["failing_input_failed"], ("the free g1 falls differ from SPEC4's on the same model", g["known_spec4_b"])
    assert g["identity"]["ok"] and m["exclusion"]["pass"], (g["identity"], m["exclusion"])
    cuts = json.loads((d / "cuts.json").read_text())
    n = (N_LAYERS - 1) * m["width"]
    assert cuts["g1"]["count"] == round(0.01 * n) and cuts["g3"]["count"] == round(0.03 * n) and set(cuts["pl1"]["per_layer"]) == {round(0.01 * m["width"])}, cuts
    m4 = json.loads((d4 / "metrics.json").read_text())
    n_npz, n_moved = 0, 0
    for j in items["primary"]:
        r = m["jobs"][j["job"]]
        z4 = np.load(d4 / m4["jobs"][j["job"]]["falls_path"])
        names = list(z4["arm"])
        for x in ("own", "cross", "random", "foreign"):
            zf = np.load(d / r["falls"][f"{x}__g1__free"]["path"])
            zh = np.load(d / r["falls"][f"{x}__g1__held"]["path"])
            i = names.index(f"{x}_direct_m50")
            S4 = [(int(a), int(b)) for a, b in zip(z4["set_layer"][i], z4["set_neuron"][i]) if a >= 0]
            assert list(zip(zf["set_layer"].tolist(), zf["set_neuron"].tolist())) == S4, (j["job"], x, "the g1 set differs from SPEC4's")
            assert np.array_equal(zf["a_after"], z4["a_after"][i]) and np.array_equal(zf["a_before"], z4["a_before"]), (j["job"], x, "free a_P differs from SPEC4's")
            assert not zf["set_in_cut"].any() and set(zh["held_layers"].tolist()) == {l for l, _ in S4}, (j["job"], x)
            n_moved += int(not np.array_equal(zf["a_after"], zh["a_after"]))
        for k in JG_KEYS["case"]:
            assert (d / r["falls"][k]["path"]).exists(), (j["job"], k)
            n_npz += 1
        zi = np.load(d / r["falls"]["identity__none__held"]["path"])
        assert np.array_equal(zi["a_after"], zi["a_before"]) and set(zi["held_layers"].tolist()) == set(range(j["layer"])), (j["job"], "held identity")
        for s in ("J", "O"):
            z = np.load(d / r["prebirth"][s])
            assert z["a_mean_abs"].shape == (j["layer"], m["width"]) and ("loadsG" in z.files) == (s == "J") and str(z["read"]) == "generic_t15", (j["job"], s)
    for k, v in m["ceiling"].items():
        n_npz += sum((d / v["falls"][x]["path"]).exists() for x in JG_KEYS["split"])
    for u, v in m["planted"].items():
        keys = JG_KEYS["pair"] + (JG_KEYS["members"] if v["members_read"] else [])
        assert set(v["falls"]) == set(keys), (u, sorted(v["falls"]))
        for k in keys:
            z = np.load(d / v["falls"][k]["path"])
            assert not z["set_in_cut"].any() or k.split("__")[1] == "none", (u, k, "a set holds a member of its cut")
            n_npz += 1
    assert n_moved > 0, "no held arm differs from its free arm"
    for f in ("held_gates.json", "cases.json", "members_alone.json"):
        assert (d / "samples" / f).exists(), f
    assert len(json.loads((d / "samples" / "cases.json").read_text())) == JG.N_SAMPLE_CASES
    return {"n_npz": n_npz, "n_held_arms_moved": n_moved, "forwards": m["forwards"], "tokens_forwarded": m["tokens_forwarded"], "cuts": {e: c["count"] for e, c in cuts.items()},
            "held_completeness": g["held_completeness"]["value"], "held_completeness_failing_input": g["held_completeness"]["failing_input"]["value"],
            "held_identity_failing_input": g["held_identity"]["failing_input"]["value"], "known_t14": g["known_spec4_a"]["failing_input"]["value"],
            "known_b_next_case": g["known_spec4_b"]["failing_input"]["value"], "members_by_pair": {u: v["n_members"] for u, v in m["planted"].items()}}


JG_KEYS = {"case": [f"identity__none__{s}" for s in ("free", "held")] + [f"{a}__g1__{s}" for a in ("own", "cross", "random", "foreign") for s in ("free", "held")],
           "split": [f"identity__none__{s}" for s in ("free", "held")] + [f"{a}__g1__{s}" for a in ("own", "cross") for s in ("free", "held")],
           "pair": [f"identity__none__{s}" for s in ("free", "held")] + [f"{a}__{e}__{s}" for e in ("none", "g1", "g3", "pl1") for a in ("own", "cross") for s in ("free", "held")]
           + [f"single__none__{s}" for s in ("free", "held")],
           "members": [f"members__none__{s}" for s in ("free", "held")]}


def g10_pick(rng, L: int, n: int, forbid: np.ndarray, taken: set) -> list:
    """n distinct (layer, neuron) with layer < L, outside the mask and outside `taken`."""
    out = []
    while len(out) < n:
        x = (int(rng.integers(L)), int(rng.integers(forbid.shape[1])))
        if not forbid[x] and x not in taken and x not in out:
            out.append(x)
    return out


def synth_bank(root: pathlib.Path) -> tuple:
    """The constructed generic bank at root/t15, its items and before_nulls.json. Known answers: own'_held 0.40 + 0.03 i, cross'_held 0.10 + 0.02 i,
    random'_held 0.008 (positive control 0.5 against 0.008); ceiling'_held I 0.7; pair I per (exclusion, scale) G10_PAIR_I with noise under 0.008
    (the unexcluded free null about 0.16 against before_nulls 0.163, the 1%-excluded 0.06); W1 0.30 on every case (15 of D'(J) in the omega top 50),
    the shuffled omega under 0.06; SPEC4's a_P the bank's own, T_14's 0.6 of it."""
    from infra.aws_pg.jobs import reuse_generic as JG
    from infra.pg import reuse_birth as RB
    from infra.pg import reuse_generic as RG
    rng = np.random.default_rng(10)
    n_up, W, N = G10_LAYERS - 1, G10_W, G10_N
    d = root / "t15"
    for s in ("falls", "prebirth", "samples"):
        (d / s).mkdir(parents=True, exist_ok=True)
    ab_np = (rng.standard_cauchy((n_up, W)) * 0.01 * (1 + np.arange(n_up))[:, None]).astype(np.float32)
    abar = torch.tensor(ab_np)
    RB.write_high_background(d / "high_background.json", RB.high_background(abar, n_up), "T_15")
    np.savez_compressed(d / "abar.npz", abar=ab_np, e1=np.arange(10))
    cut = RG.masks(abar, n_up)
    cuts_rec = {e: {k: c[k] for k in ("count", "per_layer", "frac", "per_layer_rule", "cut", "n_entries", "n_up")} for e, c in cut.items()}
    NH.dump(d / "cuts.json", cuts_rec)
    mk = {e: c["mask"].numpy() for e, c in cut.items()}
    mt = {e: c["mask"] for e, c in cut.items()}
    med = np.median(np.abs(ab_np), 1)
    desc = lambda S: [[int(l), int(k), float(ab_np[l, k]), float(abs(ab_np[l, k]) / med[l])] for l, k in S]
    tok_of = lambda i, n_: (np.arange(n_) * 97 + 1000 * (i + 1)).astype(np.int64)
    rd = lambda a: {"a": a.astype(np.float32), "g": np.zeros((len(a), 1), np.float32), "u": np.zeros((len(a), 1), np.float32)}

    def read(key, unit, toks, arms, falls_of):
        """falls_of(arm, excl, scale) -> the fall on every context; returns the job-fall records banked through jobs.reuse_generic.bank_arm."""
        a0 = rng.uniform(1.0, 2.0, len(toks)).astype(np.float32)
        res = {}
        for arm in arms:
            for s in ("free", "held"):
                f = 0.0 if arm["arm"] == "identity" else falls_of(arm["arm"], arm["excl"], s)
                held = (set(range(unit["layer"])) if arm["arm"] == "identity" else {l for l, _ in arm["set"]}) if s == "held" else set()
                res[f"{arm['arm']}__{arm['excl']}__{s}"] = JG.bank_arm(d, key, unit, arm, s, toks, rd(a0), rd(a0 if f == 0.0 else a0 * (1 - f)), abar, mt["g1"],
                                                                        mt[arm["excl"]], held, len(held))
        return res, a0
    primary, planted, jobs, ceiling, prs = [], [], {}, {}, {}
    for i, L in enumerate(G10_CASE_L):
        pole = f"L{L:02d}N{7 + i}+"
        j = {"job": f"{pole}:c0", "pole": pole, "layer": L, "neuron": 7 + i, "sign": "+", "tercile": US.tercile(L), "c": 0, "birth": 10, "t_before": 9,
             "label": "related" if i % 2 else "unrelated", "J_tok": tok_of(i, N).tolist(), "O_tok": (tok_of(i, N) + 13).tolist(),
             "ceiling": {"c": 1, "sides_tok": [(tok_of(i, 6) + 29).tolist(), (tok_of(i, 6) + 31).tolist()]}, "foreign_candidates": []}
        primary.append(j)
        unit = UR.pole_unit(pole)
        LG = rng.uniform(0.1, 1.0, (L, W)).astype(np.float32)
        sc = LG.astype(np.float64).copy()
        sc[mk["g1"][:L]] = -np.inf
        T = [(int(q // W), int(q % W)) for q in np.argsort(-sc.ravel(), kind="stable")[:50]]
        own = T[:G10_W1_SHARED] + g10_pick(rng, L, 50 - G10_W1_SHARED, mk["g1"], set(T))
        cross = T[G10_W1_SHARED:G10_W1_SHARED + 5] + g10_pick(rng, L, 45, mk["g1"], set(T) | set(own))
        rand = g10_pick(rng, L, 50, mk["g1"], set(own) | set(cross))
        forg = g10_pick(rng, L, 50, mk["g1"], set())
        fo = {("own", "free"): 0.45 + 0.03 * i, ("own", "held"): 0.40 + 0.03 * i, ("cross", "free"): 0.12 + 0.02 * i, ("cross", "held"): 0.10 + 0.02 * i,
              ("random", "free"): 0.010, ("random", "held"): 0.008, ("foreign", "free"): 0.03, ("foreign", "held"): 0.025}
        arms = [{"arm": "identity", "excl": "none", "set": []}] + [{"arm": x, "excl": "g1", "set": S} for x, S in zip(("own", "cross", "random", "foreign"), (own, cross, rand, forg))]
        key = f"{pole}__c0"
        res, a0 = read(key, unit, np.asarray(j["J_tok"]), arms, lambda x, e, s: fo[(x, s)])
        gP, uP = (rng.uniform(0.5, 1.5, N).astype(np.float32) for _ in range(2))
        z0 = np.zeros(200)
        for side, D, extra in (("J", own, {"loadsG": LG, "loadsU": np.zeros_like(LG)}), ("O", cross, {})):
            np.savez_compressed(d / "prebirth" / f"{key}__{side}.npz", t="T_15", read="generic_t15", side=side, pole=pole, tok=np.asarray(j["J_tok"] if side == "J" else j["O_tok"]),
                                D_layer=np.array([l for l, _ in D], np.int16), D_neuron=np.array([k for _, k in D], np.int32), D_in_B=np.zeros(50, bool),
                                top_idx=z0.astype(np.int64), top_A=z0.astype(np.float32), top_abar=z0.astype(np.float32), a_mean_abs=np.ones((L, W), np.float32),
                                s_mean=np.ones(L, np.float32), gP=gP, uP=uP, GP=np.zeros(8, np.float32), UP=np.zeros(8, np.float32), **extra)
        f = lambda x, s: res[f"{x}__g1__{s}"]
        t14 = a0.astype(np.float64) * 0.6
        jobs[j["job"]] = {"pole": pole, "layer": L, "tercile": j["tercile"], "birth": 10, "label": j["label"], "n_J": N, "n_O": N, "foreign": {"pole": "synthetic", "c": 0, "n": N},
                          "falls": res, "I": {s: JR.ratio(f("cross", s)["mean"], f("own", s)["mean"]) for s in ("free", "held")},
                          "cancel": {s: RA.cancel(f("own", s)["mean"], f("random", s)["mean"], f("own", s)["n_pos"]) for s in ("free", "held")}, "s": 0.0,
                          "prebirth": {s: f"prebirth/{key}__{s}.npz" for s in ("J", "O")},
                          "known": {"tok": j["J_tok"], "spec4_a_before": a0.astype(np.float64).tolist(), "t14_a": t14.tolist(), "ratio_spec4": US.known_ratio(a0, a0.astype(np.float64)),
                                    "ratio_t14": US.known_ratio(a0, t14), "spec4_own": f("own", "free")["mean"], "spec4_cross": f("cross", "free")["mean"]}}
        D1, D2 = g10_pick(rng, L, 50, mk["g1"], set()), g10_pick(rng, L, 50, mk["g1"], set())
        sk = f"{pole}__c1_ceiling_side2"
        fc = {("own", "free"): 0.55, ("own", "held"): 0.50, ("cross", "free"): 0.40, ("cross", "held"): 0.35}
        rs, _ = read(sk, unit, np.asarray(j["ceiling"]["sides_tok"][1]), [{"arm": "identity", "excl": "none", "set": []}, {"arm": "own", "excl": "g1", "set": D2},
                                                                          {"arm": "cross", "excl": "g1", "set": D1}], lambda x, e, s: fc[(x, s)])
        ceiling[f"{pole}__c1"] = {"pole": pole, "c": 1, "key": sk, "n_sides": [6, 6], "s": 0.0, "falls": rs,
                                  "I_ceil": {s: JR.ratio(rs[f"cross__g1__{s}"]["mean"], rs[f"own__g1__{s}"]["mean"]) for s in ("free", "held")}}
    for i, L in enumerate(G10_PAIR_L):
        u = f"PL{L:02d}_{i:02d}"
        p = {"unit": u, "layer": L, "a": 3 + i, "b": 40 + i, "lam": 1.0, "J_tok": tok_of(50 + i, N).tolist(), "O_tok": (tok_of(50 + i, N) + 7).tolist(), "tercile": US.tercile(L)}
        planted.append(p)
        unit = UR.planted_unit(p)
        D = {}
        for e in ("none", "g1", "g3", "pl1"):
            own = g10_pick(rng, L, 50, mk[e], set())
            if e == "none":
                inB = [(int(a), int(b)) for a, b in zip(*np.nonzero(mk["g1"][:L]))][:3]
                cross = inB + g10_pick(rng, L, 50 - len(inB), mk["g1"], set(own) | set(inB))
            else:
                cross = g10_pick(rng, L, 50, mk[e], set(own))
            D[e] = (own, cross)
        mem = RG.members_alone(D["none"][1], mk["g1"], ab_np)
        one = RG.single_largest(D["none"][1], ab_np)
        noise = {k: float(rng.uniform(-0.008, 0.008)) for k in G10_PAIR_I}
        arms = [{"arm": "identity", "excl": "none", "set": []}] + [{"arm": x, "excl": e, "set": S} for e in ("none", "g1", "g3", "pl1") for x, S in zip(("own", "cross"), D[e])]
        arms += ([{"arm": "members", "excl": "none", "set": mem}] if mem else []) + [{"arm": "single", "excl": "none", "set": one}]
        fp = lambda x, e, s: {"own": 0.5, "cross": 0.5 * (G10_PAIR_I[(e, s)] + noise[(e, s)]) if (e, s) in G10_PAIR_I else None,
                              "members": 0.08 if s == "free" else 0.03, "single": 0.05 if s == "free" else 0.02}[x]
        res, _ = read(f"{u}__planted", unit, np.asarray(p["J_tok"]), arms, fp)
        prs[u] = {"layer": L, "tercile": US.tercile(L), "a": p["a"], "b": p["b"], "lam": 1.0, "n_J": N, "n_O": N, "falls": res,
                  "I": {f"{e}__{s}": JR.ratio(res[f"cross__{e}__{s}"]["mean"], res[f"own__{e}__{s}"]["mean"]) for e in ("none", "g1", "g3", "pl1") for s in ("free", "held")},
                  "members": desc(mem), "single": desc(one), "n_members": len(mem), "members_read": bool(mem),
                  "n_in_B": {f"{x}__{e}": int(RB.in_B(S, mk["g1"]).sum()) for e in D for x, S in zip(("own", "cross"), D[e])}, "s": {e: 0.0 for e in D}}
    items = {"primary": primary, "fresh": [], "planted": planted, "known": {}, "job_counts": []}
    ip = root / "items.json"
    NH.dump(ip, items)
    sha = hashlib.sha256(ip.read_bytes()).hexdigest()
    ids = [j["job"] for j in primary]
    va, wa = float(np.median([jobs[x]["known"]["ratio_spec4"] for x in ids])), float(np.median([jobs[x]["known"]["ratio_t14"] for x in ids]))
    b_ids = ids[:20]
    ours = [jobs[x]["falls"][f"{a}__g1__free"]["mean"] for x in b_ids for a in ("own", "cross")]
    nxt = [jobs[b_ids[(i + 1) % len(b_ids)]]["known"][f"spec4_{a}"] for i in range(len(b_ids)) for a in ("own", "cross")]
    gate = lambda v, fi, ok, **kw: {"value": v, "pass": ok, "failing_input": {"input": "synthetic", "value": fi}, "failing_input_failed": True, "ok": ok, **kw}
    m = {"stage": "generic", "t": "T_15", "status": "done", "items_sha256": sha, "width": W, "n_layers": G10_LAYERS, "pg_git_sha": "dry", "device_name": "synthetic",
         "known_number": {"ratio": 0.001, "wrong_ratio": 0.2, "pass": True, "failing_input_failed": True, "failing_input": "synthetic: the wrong bank"},
         "completeness": {"median_err_over_pole_max": 0.001, "wrong_layer": 0.3, "pass": True, "failing_input_failed": True},
         "gates": {"held_identity": gate(0.0, 0.37, True, identity={"value": 0.0, "reason": "the empty set"}, n_contexts=20),
                   "held_completeness": gate(0.0, 0.021, True, identity={"value": 0.0, "reason": "the same check on the empty set"}, case=ids[0], n_in_B=3),
                   "known_spec4_a": gate(va, wa, True), "known_spec4_b": gate(float(np.median(np.abs(np.subtract(ours, ours)))), float(np.median(np.abs(np.subtract(ours, nxt)))), True),
                   "identity": gate(0.0, 100, True)},
         "cuts": cuts_rec, "exclusion": {"pass": True, "rule": "synthetic"}, "memory_max_allocated_gb": 0.0, "gpu_memory_total_gb": None,
         "timing_s": {"wall_after_load": 3000.0}, "download_and_corpus_s": 300.0, "load_s": 10.0, "forwards": 0, "tokens_forwarded": 0,
         "jobs": jobs, "ceiling": ceiling, "planted": prs}
    NH.dump(d / "metrics.json", m)
    bn = root / "before_nulls.json"
    NH.dump(bn, {"T15_full_side_unexcluded": {"value": 0.163, "source": "synthetic: SPEC6 dry run part 10"}})
    return ip, bn


def part_generic(out: pathlib.Path) -> dict:
    from infra.aws_pg.jobs import reuse_birth as JB
    from infra.aws_pg.jobs import reuse_generic as JG
    from infra.aws_pg.jobs.neurons_harvest import fetch
    from infra.pg import reuse_generic as RG
    sc = RG.selfcheck()
    print("PASS generic pure logic:", sc)
    dec = held_decoder_check()
    # the job on a tiny bf16 OLMo 3, against SPEC4's birth_t15 load on the same model and a T_14 bank from another
    tok = tiny_tokenizer(bos=True)
    tokens = fake_corpus(tok, N_WINDOWS + 2)
    offsets = np.arange(0, tokens.size + 37, 37)
    offsets[-1] = tokens.size
    corpus = {"tokens": tokens, "doc_offsets": offsets}
    dg = out / "generic"
    model = tiny_model("olmo3", seed=0, n_layers=N_LAYERS).requires_grad_(False)
    other = tiny_model("olmo3", seed=1, n_layers=N_LAYERS).requires_grad_(False)
    ref, ref2 = (reference_a(x, tok, tokens, list(range(N_WINDOWS))) for x in (model, other))
    refs = {T: ref for T in ("T_15", "T_8", "T_10", "T_11")}
    kp = ["L03N5+", "L04N11+", "L05N2+"]
    ktok = [top_tok(ref, NH.parse_pole(p)[0], NH.parse_pole(p)[1], 1.0).tolist() for p in kp]
    val = lambda r, p, t: np.asarray(r[NH.parse_pole(p)[0]])[np.asarray(t) // WINDOW, np.asarray(t) % WINDOW, NH.parse_pole(p)[1]].tolist()
    known = {"poles": kp, "tok": ktok, "wrong_rule": "another model", "a_bank": {"T_15": [val(ref, p, t) for p, t in zip(kp, ktok)]},
             "a_wrong": {"T_15": [val(ref2, p, t) for p, t in zip(kp, ktok)]}}
    pitems = b8_primary(refs, known)
    items = {"primary": [{**j, "t_before": j["birth"] - 1} for j in pitems["jobs"]], "fresh": b8_fresh(refs), "planted": b8_planted(refs), "known": known,
             "job_counts": pitems["job_counts"], "sources": {"grid": []}}
    ip = dg / "birth_items.json"
    NH.dump(ip, items)
    sha = hashlib.sha256(ip.read_bytes()).hexdigest()
    layers = NH.text_tower(model).layers
    fpath = dg / "primary_foreign.json"
    NH.dump(fpath, {j["job"]: JB.choose_q(j, layers) for j in items["primary"]})
    d4 = dg / "spec4" / "t15"
    JB.Job(d4, args_birth(d4, "birth_t15", "T_15", "--primary-foreign", str(fpath), sha=sha), items, model, tok, corpus, "cpu").run()
    t14 = dg / "t14"
    (t14 / "dating").mkdir(parents=True, exist_ok=True)
    kw = dict(tokens=tokens, length=RA.LENGTH, bos=tok.bos_token_id, batch=8, hw=WINDOW, device="cpu")
    for j in items["primary"]:
        Jt, Ot = (np.asarray(x)[:JG.CAP] for x in (j["J_tok"], j["O_tok"]))
        unit = UR.pole_unit(j["pole"])
        aJ, aO = (RA.readout(other, x, unit, None, **kw)["a"] for x in (Jt, Ot))
        np.savez_compressed(t14 / "dating" / f"{j['pole']}__c{j['c']}.npz", t="T_14", pole=j["pole"], side=np.array(["J"] * len(Jt) + ["O"] * len(Ot)),
                            tok=np.concatenate([Jt, Ot]).astype(np.int64), aP=np.concatenate([aJ, aO]))
    gd = dg / "run" / "t15"
    a = JG.parser().parse_args(["--out", str(gd), "--stage", "generic", "--t", "T_15", "--items", str(ip), "--primary-foreign", str(fpath), "--spec4-bank", str(d4),
                                "--t14-bank", str(t14), "--harvest-window", str(WINDOW), "--scan-tokens", str(N_WINDOWS * WINDOW), "--batch", "8", "--full-batch", "4"])
    a.items_sha256 = sha
    a.primary_foreign_sha256 = hashlib.sha256(fpath.read_bytes()).hexdigest()
    JG.Job(gd, a, items, model, tok, corpus, "cpu", JG.fetch_refs(a, fetch)).run()
    res = check_generic(gd, items, d4)
    print("generic job on the tiny model:", json.dumps(res))
    v0 = subprocess.run([sys.executable, str(SCRIPTS / "verdict_generic.py"), "--run", str(dg / "run"), "--items", str(ip), "--out", str(dg / "verdict_tiny")],
                        capture_output=True, text=True)
    assert v0.returncode in (0, 1) and (dg / "verdict_tiny" / "verdict_generic.json").exists(), (v0.returncode, v0.stdout[-3000:], v0.stderr[-3000:])
    tiny_failed = [x for x in v0.stdout.splitlines() if x.startswith("VERDICT_GENERIC")]
    print("verdict_generic on the tiny-model bank reads it:", tiny_failed[0] if tiny_failed else "", "exit", v0.returncode)
    # the verdict on the constructed bank: incomplete, green, each failing input
    syn = out / "generic_synth"
    sip, sbn = synth_bank(syn)
    vout = syn / "verdict"
    base = [sys.executable, str(SCRIPTS / "verdict_generic.py"), "--items", str(sip), "--before-nulls", str(sbn), "--out", str(vout)]
    inc = syn / "incomplete"
    shutil.copytree(syn / "t15", inc / "t15")
    next((inc / "t15" / "falls").glob("*__members__none__held.npz")).unlink()
    r = subprocess.run(base + ["--run", str(inc)], capture_output=True, text=True)
    assert r.returncode == 4 and "GENERIC_BANK_INCOMPLETE" in r.stderr and not vout.exists(), (r.returncode, r.stdout[-2000:], r.stderr[-3000:])
    exits = {"incomplete": r.returncode}
    print("verdict_generic on a bank without one members npz:", r.stderr.splitlines()[0], "exit", r.returncode)
    shutil.rmtree(inc)
    r = subprocess.run(base + ["--run", str(syn)], capture_output=True, text=True)
    print(r.stdout, end="")
    assert r.returncode == 0 and "VERDICT_GENERIC_OK" in r.stdout, r.stdout[-4000:] + r.stderr[-4000:]
    exits["green"] = r.returncode
    v = json.loads((vout / "verdict_generic.json").read_text())
    assert all(g["ok"] for g in v["gates"].values() if g.get("ok") is not None) and all(x["verdict"] != "not read" for x in v["rules"].values())
    assert abs(v["rules"]["W1"]["value"] - G10_W1_SHARED / 50) < 1e-12 and v["rules"]["W1"]["verdict"] == "H_weights not refuted", v["rules"]["W1"]
    assert abs(v["reported"]["nulls"]["none_free"]["value"] - 0.16) < 0.01 and abs(v["reported"]["nulls"]["g1_held"]["value"] - 0.05) < 0.01, v["reported"]["nulls"]
    assert abs(v["rules"]["N1"]["value"] - 0.06) < 0.02 and abs(v["gates"]["positive_control"]["value"] - (0.40 + 0.03 * 5.5)) < 0.02, (v["rules"]["N1"], v["gates"]["positive_control"])
    assert (vout / "GRID6.md").read_text().startswith("Cases 12:"), "GRID6.md does not open with the count line"
    fail = {}
    for mode, target in (("known_t14", "known_a"), ("held_wrong_layer", "held_identity"), ("completeness_free", "held_completeness"), ("positive_random", "positive_control"),
                         ("room_shuffled", "room"), ("w1_shuffled", "W1")):
        r = subprocess.run(base + ["--run", str(syn), "--failing-input", mode], capture_output=True, text=True)
        line = [x for x in r.stdout.splitlines() if x.startswith("FAILING_INPUT")]
        assert r.returncode == 1 and line, (mode, r.returncode, r.stdout[-3000:] + r.stderr[-3000:])
        f = json.loads((vout / f"failing_input_{mode}" / "verdict_generic.json").read_text())
        if mode == "w1_shuffled":
            assert f["rules"]["W1"]["verdict"].startswith("H_weights refuted") and f["rules"]["W1"]["value"] < 0.06, f["rules"]["W1"]
        else:
            assert not f["gates"][target]["ok"] and all(x["verdict"] == "not read" for x in f["rules"].values()), (mode, f["gates"][target])
        fail[mode] = line[0]
        exits[mode] = r.returncode
        print(line[0])
    print("verdict_generic exit codes:", exits)
    return {"selfcheck": sc, "decoder": dec, "tiny_job": res, "tiny_verdict_exit": v0.returncode, "exits": exits, "failing_inputs": fail,
            "rules": {k: x["verdict"] for k, x in v["rules"].items()}}


# ------------------------------------------------------------------------------------------ the constructed sole driver
def part_sole_driver(out: pathlib.Path) -> dict:
    tok = tiny_tokenizer(bos=True)
    tokens = fake_corpus(tok, N_WINDOWS + 2)
    model = tiny_model("olmo3", seed=5, dtype=torch.float32, n_layers=N_LAYERS).requires_grad_(False)
    tower = NH.text_tower(model)
    layers = list(tower.layers)
    W = layers[0].mlp.down_proj.weight.shape[1]
    c, l, k, L, n = 0, 1, 7, 3, 5
    with torch.no_grad():
        tower.embed_tokens.weight[:, c] = 0
        for i, lyr in enumerate(layers):
            lyr.self_attn.o_proj.weight[c, :] = 0
            lyr.mlp.down_proj.weight[c, :] = 0
        layers[l].mlp.down_proj.weight[c, k] = 2.0
        for proj in (layers[L].mlp.gate_proj, layers[L].mlp.up_proj):
            proj.weight[n, :] = 0
            proj.weight[n, c] = 1.0
    ref = reference_a(model, tok, tokens, list(range(N_WINDOWS)))
    toks = top_tok(ref, L, n, 1.0, 12)
    unit = UR.pole_unit(f"L{L:02d}N{n}+")
    kw = dict(tokens=tokens, length=RA.LENGTH, bos=tok.bos_token_id, batch=8, hw=WINDOW, device="cpu")
    rows = UR.Rows(model)
    cap = UR.capture(model, toks, L, {L}, **kw)
    D, A = RA.drivers_direct(rows, cap, unit, W, ms=(1, 5))
    assert D[1] == [(l, k)], f"the sole driver is not ranked first: {D[1]}"
    flat = l * W + k
    others = torch.cat([A[:, :flat], A[:, flat + 1:]], 1)
    assert bool((others == 0).all()) and bool((A[:, flat] > 0).all()), "a neuron other than the sole driver has a direct attribution"
    Dg, Ag = RA.drivers_gradxact(model, toks, unit, W, ms=(1,), **kw)
    sets = UR.background_positions(15, N_WINDOWS * WINDOW, WINDOW)
    abar = RA.background_mean_a(model, sets["e1"], L, **kw)
    before = RA.readout(model, toks, unit, None, **kw)

    # the closed form, from an unablated forward: layer l's MLP output v and k's input a_k at the marked token
    rec = {}
    x, pos = UR.pack(tokens, toks, RA.LENGTH, tok.bos_token_id, WINDOW, "cpu")
    rr = torch.arange(len(toks))
    hs = [layers[l].mlp.register_forward_hook(lambda m, a, o: rec.__setitem__("v", o[rr, pos].clone())),
          layers[l].mlp.down_proj.register_forward_pre_hook(lambda m, a: rec.__setitem__("ak", a[0][rr, pos, k].clone()))]
    with torch.no_grad():
        NH.forward(model, x, False)
    for h in hs:
        h.remove()
    norm = layers[l].post_feedforward_layernorm
    eps, gam, col = norm_eps(norm), norm.weight.float(), layers[l].mlp.down_proj.weight[:, k].float()

    def a_of(v):
        h = gam[c] * v[:, c] / torch.sqrt(v.pow(2).mean(1) + eps)
        return (F.silu(h) * h).numpy()
    a0 = a_of(rec["v"])
    assert np.allclose(a0, before["a"], rtol=1e-5, atol=1e-6), ("the construction does not hold: the pole reads more than (l, k)", a0, before["a"])
    res = {"driver": [l, k], "layer": L, "neuron": n, "n_contexts": len(toks), "grad_top1": Dg[1]}
    # on a random model ā_k is near 0, so mean and zero replacement both send a_P to about 0; "half" puts r_k at half of a_k's mean on these
    # contexts, where the norm's rescaling matters, and a prediction that ignores it (h[c] scaled by r_k / a_k) must miss
    half = abar.clone()
    half[l, k] = 0.5 * rec["ak"].mean()
    for case, replace, ab in (("mean", "mean", abar), ("zero", "zero", abar), ("half", "mean", half)):
        r = float(ab[l, k]) if replace == "mean" else 0.0
        after = RA.readout(model, toks, unit, RA.ablation_dict([(l, k)], replace, ab, L, "cpu"), **kw)
        observed = RA.fall(before["a"], after["a"])
        predicted = RA.fall(before["a"], a_of(rec["v"] + (r - rec["ak"])[:, None] * col[None]))
        err = float(np.abs(observed - predicted).max())
        assert err <= 1e-4, (case, observed, predicted)
        h0 = gam[c] * rec["v"][:, c] / torch.sqrt(rec["v"].pow(2).mean(1) + eps)
        hn = h0 * r / rec["ak"]
        naive = RA.fall(before["a"], (F.silu(hn) * hn).numpy())
        naive_err = float(np.abs(observed - naive).max())
        if case == "half":
            assert naive_err > 10 * 1e-4 and 0.05 < float(np.median(observed)) < 0.95, ("the half case does not separate the closed form from the naive one", naive_err, observed)
        # a set at the layer between l and L does not move the pole at all: its MLP writes nothing into coordinate c. (Layers below l feed k's input,
        # and k's own layer shares the norm, so neither is a clean non-driver.)
        rest = [(2, j) for j in range(40)]
        untouched = RA.readout(model, toks, unit, RA.ablation_dict(rest, replace, abar, L, "cpu"), **kw)
        moved = float(np.abs(RA.fall(before["a"], untouched["a"])).max())
        assert moved == 0.0, f"a set without the driver moved the pole: {moved}"
        res[case] = {"r": r, "observed": observed.tolist(), "predicted": predicted.tolist(), "max_abs_error": err, "naive_max_abs_error": naive_err,
                     "non_driver_set_max_abs_fall": moved}
        print(f"sole driver, {case} r={r:.4g}: fall observed {np.round(observed, 4).tolist()[:4]} predicted {np.round(predicted, 4).tolist()[:4]} "
              f"max error {err:.2e} (naive, norm ignored: {naive_err:.2e}); 40 layer-2 neurons ablated: max |fall| {moved:.2e}")
    return res


PASS_GENERIC = ("PASS generic: held norm equal to the hand computation (tensors and the decoder layer), empty set exact, the cuts and the members arms; "
                "the generic job on a tiny model reproduces SPEC4's free arms exactly with every held gate passing; verdict_generic stops on an incomplete "
                "bank, green on the constructed bank, failing on each failing input")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", choices=["all", "8", "9", "10"], default="all",
                    help="8: the SPEC4 BIRTH part alone; 9: the SPEC5 DATING part alone; 10: the SPEC6 GENERIC part alone")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "samples").mkdir(parents=True)
    if a.part == "10":
        m = {"generic": part_generic(out)}
        print(PASS_GENERIC)
        NH.dump(out / "metrics.json", m)
        print("REUSE_DRYRUN_OK")
        return
    if a.part == "9":
        m = {"dating": part_dating(out)}
        print("PASS dating: four loads, known number and capture identity passing with their failing inputs failing, every case banked, scale and 1M "
              "background at T_15; verdict_dating stops on an incomplete bank, green with the construction's known answers, failing on each failing input; "
              "amendment 5b: the scale stage at four loads, verdict_dating --scale-run stops on an incomplete scale bank, green, failing on each failing input")
        NH.dump(out / "metrics.json", m)
        print("REUSE_DRYRUN_OK")
        return
    if a.part == "8":
        m = {"birth": part_birth(out)}
        print("PASS birth: six loads, identity exactly 0, no set member in B_t, every case, split, pair and Part A bank banked; verdict_birth stops on an "
              "incomplete bank, green, failing on each failing input and on a fresh case on a primary neuron")
        NH.dump(out / "metrics.json", m)
        print("REUSE_DRYRUN_OK")
        return
    m = {"selfcheck": RA.selfcheck()}
    print("PASS pure logic:", m["selfcheck"])
    m["sole_driver"] = part_sole_driver(out)
    print("PASS sole driver: the closed-form fall under mean and zero replacement")
    m["job"] = part_job(out)
    print("PASS job: S1 and S2 end to end, identity exactly 0, failing inputs fail")
    print("PASS primary: T_15 and two birth loads, identity exactly 0, every arm and split banked, verdict green and failing on each failing input")
    m["amend2"] = part_amend2(out)
    print("PASS amend2: T_15 and two birth loads, identity exactly 0, R_J recomputed, verdict2 stops without the new arms, green, and failing on each failing input")
    m["reuse_ablate2_selfcheck"] = RA2.selfcheck()
    print("PASS heads and output pure logic:", m["reuse_ablate2_selfcheck"])
    ctx = part_ab_runs(out)
    m["heads"] = part_heads(out, ctx)
    print("PASS heads: sniper and two-load primary, identity exactly 0, head capture check passing and failing on the wrong layer, H(J) recomputed, "
          "verdict_heads freezes h, stops on an incomplete bank, green, and failing on each failing input")
    m["output"] = part_output(out, ctx)
    print("PASS output: sniper and two-load primary, identity exactly 0, reader capture check passing and failing on the wrong layer, R(J) recomputed, "
          "verdict_output freezes r, stops on an incomplete bank, green, and failing on each failing input")
    m["birth"] = part_birth(out)
    print("PASS birth: six loads, identity exactly 0, no set member in B_t, every case, split, pair and Part A bank banked; verdict_birth stops on an "
          "incomplete bank, green, failing on each failing input and on a fresh case on a primary neuron")
    m["dating"] = part_dating(out)
    print("PASS dating: four loads, known number and capture identity passing with their failing inputs failing, every case banked, scale and 1M "
          "background at T_15; verdict_dating stops on an incomplete bank, green with the construction's known answers, failing on each failing input")
    m["generic"] = part_generic(out)
    print(PASS_GENERIC)
    NH.dump(out / "metrics.json", m)
    print("REUSE_DRYRUN_OK")


if __name__ == "__main__":
    main()
