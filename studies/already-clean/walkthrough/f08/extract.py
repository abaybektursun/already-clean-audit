"""f08: copying the mathematics model's changed weights into the base model; each condition's score on 500 competition
problems and three example problems with each condition's answer.

Reads only banked files in this repository (no network, no GPU) and writes data.js next to this file as
window.F08_DATA = {...}. Prints `path sha256` for every input.

    python3 extract.py
"""
import datetime
import glob
import hashlib
import json
import math
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]                       # walkthrough/<picture> -> already-clean -> studies -> repository root
ID = HERE.name.upper()
# Over the template's 300 KB: the plan's F8 row asks for three example problems with each condition's answer, for all
# 31 conditions; the answers run to 47,900 characters each. Every distinct answer text is stored once (56 distinct of 93).
MAX_BYTES = 420_000
SOURCES = []

N13 = "studies/next/n13"
VERDICT = f"{N13}/verdict_n13.json"
THRESH = f"{N13}/thresholds.json"
EXAMPLES = (0, 1, 2)
PROMPT_HEAD = ("Solve the following problem step by step. The last line of your response should be the answer to the problem "
               "in form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n")
PROMPT_TAIL = "\n\nRemember to put your answer on its own line after \"Answer:\""     # N13_SPEC.md §2.2

# Which tensors come from which model, per condition (N13_SPEC.md §3.2 lines 407-414; infra/pg/n13_arms.py arm_spec).
# "base" = allenai/Olmo-3-1025-7B main, the finished base model; "math" = allenai/Olmo-3.1-7B-RL-Zero-Math.
FROM_BASE = {"attention": "base", "norm_weights": "base", "embeddings_and_output_matrix": "base"}
FROM_MATH = {"attention": "math", "norm_weights": "math", "embeddings_and_output_matrix": "math"}
ATTN_NORM = {"attention": "math", "norm_weights": "math", "embeddings_and_output_matrix": "base"}
NON_MLP_CODE = {"before": FROM_BASE, "after": FROM_MATH, "attn_norm_after": ATTN_NORM}   # build.json spec.non_mlp


def pct(k):
    return {"k0.1": 0.1, "k0.3": 0.3, "k1": 1.0, "k3": 3.0}[k]


def condition(arm):
    """arm name -> (public id, plain name, family, percent of units or None, MLP part in words, non-MLP code, seed set)."""
    if arm == "base":
        return "base", "base model", "reference", None, "no change", "before", "first"
    if arm == "rl":
        return "math", "mathematics model", "reference", None, "the whole change", "after", "first"
    if arm == "zero":
        return "zeroed_change", "base model rebuilt by the copying code with the change set to zero", "reference", None, \
            "no change", "before", "second"
    if arm == "id":
        return "identity", "mathematics model rebuilt by the copying code (identity)", "reference", None, \
            "the whole change", "after", "second"
    if arm == "mlp":
        return "all_mlp", "all MLP", "half of the network", None, "the whole MLP change", "before", "first"
    if arm == "attn":
        return "attention_and_norms", "attention with every norm weight", "half of the network", None, \
            "no MLP change", "attn_norm_after", "first"
    if arm == "top_k1_B":
        return "most_changed_1_second_seeds", "most-changed 1 % of units, second seed set", "repeat reading", 1.0, \
            "the change of the 3,523 units with the largest change size", "before", "second"
    if arm == "topF_k1":
        return "most_changed_1_above_rounding", "most-changed 1 % of units, ranked on entries that moved more than one 16-bit step", \
            "repeat reading", 1.0, "the change of the 3,523 units with the largest change size counted over entries that moved " \
            "two or more 16-bit steps", "before", "first"
    if arm == "rndT_k1_d2":
        return "random_own_change_1_second_draw", "random units at their own change, 1 %, second random draw", "repeat reading", \
            1.0, "random units outside the most-changed 1 % at their own change, added until their energy reaches the " \
            "most-changed set's", "before", "first"
    if arm == "dare_k1_d2":
        return "scaled_sample_1_second_draw", "random sample of MLP entries scaled up, 1 %, second random draw", "repeat reading", \
            1.0, "each MLP entry's change kept with probability q = 0.0100 and multiplied by 1/q", "before", "first"
    if arm == "rnd0_k1":
        return "random_units_1_unscaled", "random 1 % of units at their own change, no energy matching", "repeat reading", 1.0, \
            "3,523 random units outside the most-changed 1 %, at their own change", "before", "first"
    fam, k = arm.split("_")
    p = pct(k)
    s = f"{p:g}".replace(".", "_")
    if fam == "top":
        return f"most_changed_{s}", f"most-changed {p:g} % of units", "most-changed units", p, \
            f"the change of the {p:g} % of units with the largest change size", "before", "first"
    if fam == "rndE":
        return f"random_equal_energy_scaled_{s}", f"random units of equal energy, one scale factor, {p:g} %", \
            "random units of equal energy", p, f"as many random units as the most-changed {p:g} %, drawn outside it, their " \
            "change multiplied by one scalar so the energy equals the most-changed set's", "before", "first"
    if fam == "rndT":
        return f"random_equal_energy_own_change_{s}", f"random units of equal energy, at their own change, {p:g} %", \
            "random units of equal energy", p, f"random units outside the most-changed {p:g} % at their own change, added " \
            "until their energy reaches the most-changed set's", "before", "first"
    if fam == "dare":
        return f"scaled_sample_{s}", f"random sample of MLP entries scaled up, {p:g} %", "random sample of MLP entries", p, \
            f"each MLP entry's change kept with probability q = (units in the {p:g} % set) / 352,256 and multiplied by 1/q", \
            "before", "first"
    if fam == "comp":
        return f"all_but_most_changed_{s}", f"mathematics model except its most-changed {p:g} % of units", \
            "mathematics model minus the most-changed units", p, f"the whole MLP change except the most-changed {p:g} % of " \
            "units, which stay at the base", "after", "first"
    raise ValueError(arm)


def src(rel):
    """Open a repository file by its path from the root, record its sha256 once, print `path sha256`."""
    p = ROOT / rel
    b = p.read_bytes()
    assert not b.startswith(b"version https://git-lfs"), f"{rel} is an LFS pointer: git lfs pull --include={rel}"
    if rel not in [s["path"] for s in SOURCES]:
        h = hashlib.sha256(b).hexdigest()
        SOURCES.append({"path": rel, "sha256": h})
        print(rel, h)
    return p


def compare(name, recomputed, banked, where, tol):
    """Store a recomputed summary beside the banked one; print both and whether they differ beyond rounding."""
    same = abs(recomputed - banked) <= tol
    print(f"CHECK {name}: recomputed {recomputed} banked {banked} ({where}) {'same' if same else 'DIFFERS'}")
    return {"recomputed": recomputed, "banked": banked, "banked_at": where, "same_within": tol, "same": same}


def write(data, notes):
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    data["meta"] = {"commit": commit, "sources": SOURCES,
                    "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                    "notes": notes}
    out = HERE / "data.js"
    out.write_text(f"window.{ID}_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    size = out.stat().st_size
    assert size <= MAX_BYTES, f"{out} is {size:,} bytes"
    print(f"wrote {out.relative_to(ROOT)}: {size:,} bytes")


def one(pattern):
    found = sorted(glob.glob(str(ROOT / pattern)))
    assert len(found) == 1, f"{pattern}: {found}"
    return str(Path(found[0]).relative_to(ROOT))


def main():
    v = json.loads(src(VERDICT).read_text())
    th = json.loads(src(THRESH).read_text())
    arms = v["arms"]
    assert all(a["status"] == "done" and a["n"] == 500 for a in arms.values())

    problems, texts, text_index, conditions, arm_names, checks = {}, [], {}, [], {}, {}
    for arm, a in arms.items():
        cid, name, family, p, mlp_part, non_mlp, seeds = condition(arm)
        arm_names[cid] = arm
        rel = one(f"{N13}/s3/*/{arm}/math500/samples.jsonl") if arm in ("base", "rl", "zero", "id") else \
            one(f"{N13}/s4/*/{arm}/{arm}/math500/samples.jsonl")
        rows = [json.loads(l) for l in src(rel).read_text().splitlines()]
        met = json.loads(src(str(Path(rel).parent / "metrics.json")).read_text())
        assert met["arm"] == arm and met["counts"]["finished"] == 500
        checks[cid] = compare(f"{arm} accuracy", 100 * met["counts"]["correct"] / 500, a["acc"],
                              f"verdict_n13.json arms.{arm}.acc", 1e-9)
        build = ROOT / Path(rel).parent.parent / "build.json"
        built = None
        if build.exists():
            b = json.loads(src(str(build.relative_to(ROOT))).read_text())
            assert b["arm"] == arm and b["spec"]["non_mlp"] == non_mlp, (arm, b["spec"]["non_mlp"], non_mlp)
            built = {"mlp_entries_in_mask": b["families"]["mlp"]["in_mask"],
                     "mlp_entries_changed": b["families"]["mlp"]["changed"],
                     "realized_mlp_change_energy": b["families"]["mlp"]["realized_sq"],
                     "realized_non_mlp_change_energy": sum(b["families"][f]["realized_sq"] for f in ("attn", "norm", "embed"))}
        examples = []
        for r in rows:
            if r["problem_index"] not in EXAMPLES:
                continue
            i = r["problem_index"]
            prob = r["prompt"]
            assert prob.startswith(PROMPT_HEAD) and prob.endswith(PROMPT_TAIL), (arm, i)
            if i in problems:
                assert problems[i]["prompt"] == prob and problems[i]["gold"] == r["gold"], (arm, i)
            else:
                problems[i] = {"index": i, "source_id": r["unique_id"], "prompt": prob,
                               "statement": prob[len(PROMPT_HEAD):-len(PROMPT_TAIL)], "gold": r["gold"]}
            if r["text"] not in text_index:
                text_index[r["text"]] = len(texts)
                texts.append(r["text"])
            examples.append({"problem": i, "text": text_index[r["text"]], "extracted": r["extracted"], "score": r["score"],
                             "new_tokens": r["n_new"], "hit_length_cap": r["cap_hit"], "finish_reason": r["finish_reason"],
                             "extraction": r["mode"], "sampling_seed": r["seed"]})
        assert sorted(e["problem"] for e in examples) == list(EXAMPLES), (arm, examples)
        conditions.append({"id": cid, "name": name, "family": family, "percent_of_units": p, "mlp_part": mlp_part,
                           "tensors_from": NON_MLP_CODE[non_mlp], "seed_set": seeds,
                           "accuracy": a["acc"], "range_consistent_with_data": a["wilson"], "n_problems": a["n"],
                           "share_hit_length_cap": a["cap_hit_share"], "share_with_answer_line": a["answer_line_share"],
                           "mean_new_tokens": a["mean_tokens"], "built": built,
                           "examples": sorted(examples, key=lambda e: e["problem"])})

    name_of = {c["id"]: c["name"] for c in conditions}
    id_of = {arm: cid for cid, arm in arm_names.items()}
    comparisons = []
    for key, c in v["comparisons"].items():
        if key.startswith("_"):
            continue
        a1, a2 = key.split("|")
        rec = compare(f"difference {key}", arms[a1]["acc"] - arms[a2]["acc"], c["diff"], f"verdict_n13.json comparisons.{key}.diff",
                      1e-9)
        comparisons.append({"first": id_of[a1], "second": id_of[a2], "difference": c["diff"], "range": c["bootstrap"],
                            "discordant_test_p": c["mcnemar_p"], "adjusted_p": c["bh_p"], "n_paired": c["n_paired"],
                            "check": rec})
    word = [{"first": id_of[k.split("|")[0]], "second": id_of[k.split("|")[1]], "difference": g["diff_gsm8k"], "range": g["ci"]}
            for k, g in v["gsm8k"].items()]

    T = v["thresholds"]
    gain = compare("gain = mathematics - base", T["L"] - T["B"], T["G"], "verdict_n13.json thresholds.G", 1e-9)
    delta = compare("threshold = 2 sqrt(2) sigma", 2 * math.sqrt(2) * T["sigma"], T["delta"], "verdict_n13.json thresholds.delta",
                    1e-12)
    bound = compare("bound = 2.5 x threshold", 2.5 * th["delta"], th["G-GATE"]["bound"], "thresholds.json G-GATE.bound", 1e-9)
    assert th["sigma"] == T["sigma"] and th["G"] == T["G"] and th["C"] == T["C"] == 16384

    top, dare = arms["top_k0.1"], arms["dare_k0.1"]
    bd = json.loads((ROOT / one(f"{N13}/s4/*/dare_k0.1/dare_k0.1/build.json")).read_text())
    data = {
        "problems": [problems[i] for i in EXAMPLES],
        "texts": texts,
        "conditions": conditions,
        "noise_of_one_reading": {"sigma_points": th["sigma"], "sigma_base_points": th["sigma_base"],
                                 "sigma_math_points": th["sigma_rl"], "difference_must_exceed_points": th["delta"],
                                 "check": delta},
        "gain": {"base": T["B"], "math": T["L"], "gain_points": T["G"], "required_above": th["G-GATE"]["bound"],
                 "halfway_line": T["H"], "quarter_line": T["Q"], "zeroed_change": T["B_prime"], "identity": T["L_prime"],
                 "check_gain": gain, "check_bound": bound},
        "length_cap_tokens": T["C"],
        "equal_count_comparison": {
            "most_changed": id_of["top_k0.1"], "most_changed_accuracy": top["acc"],
            "most_changed_entries": 352 * 12288,
            "scaled_sample": id_of["dare_k0.1"], "scaled_sample_accuracy": dare["acc"],
            "scaled_sample_entries_kept": bd["kept"], "scaled_sample_entries_expected": bd["kept_expected"],
            "comparison": next(c for c in comparisons if c["first"] == id_of["top_k0.1"] and c["second"] == id_of["dare_k0.1"])},
        "word_problems_1319": {"comparisons": word, "base": th["gsm8k"]["B"], "math": th["gsm8k"]["L"],
                               "gain_points": th["gsm8k"]["G"]},
        "comparisons": comparisons,
        "checks": checks,
    }
    for num, key in [("57.2", "conditions[base].accuracy"), ("92.0", "conditions[math].accuracy"),
                     ("57.6", "conditions[most_changed_0_1].accuracy"), ("61.4", "conditions[most_changed_0_3].accuracy"),
                     ("61.8", "conditions[most_changed_1].accuracy"), ("63.8", "conditions[most_changed_3].accuracy"),
                     ("58.8", "conditions[random_equal_energy_scaled_0_1].accuracy"),
                     ("63.8", "conditions[random_equal_energy_scaled_3].accuracy"),
                     ("85.2", "conditions[scaled_sample_0_1].accuracy"), ("86.4", "conditions[all_mlp].accuracy"),
                     ("83.0", "conditions[attention_and_norms].accuracy"), ("55.4", "conditions[zeroed_change].accuracy"),
                     ("91.2", "conditions[identity].accuracy"),
                     ("-23.7 (-26.5 to -21.1)", "word_problems_1319.comparisons[0].difference, .range"),
                     ("1.924", "noise_of_one_reading.sigma_points"), ("5.44", "noise_of_one_reading.difference_must_exceed_points"),
                     ("34.8", "gain.gain_points"), ("13.60", "gain.required_above"), ("16,384", "length_cap_tokens")]:
        print(f"PLAN {num} -> {key}")
    by_id = {c["id"]: c for c in conditions}
    for cid, val in [("base", 57.2), ("math", 92.0), ("most_changed_0_1", 57.6), ("most_changed_0_3", 61.4),
                     ("most_changed_1", 61.8), ("most_changed_3", 63.8), ("random_equal_energy_scaled_0_1", 58.8),
                     ("random_equal_energy_scaled_3", 63.8), ("scaled_sample_0_1", 85.2), ("all_mlp", 86.4),
                     ("attention_and_norms", 83.0), ("zeroed_change", 55.4), ("identity", 91.2)]:
        assert round(by_id[cid]["accuracy"], 1) == val, (cid, by_id[cid]["accuracy"], val)
    ee = [by_id[c]["accuracy"] for c in by_id if c.startswith("random_equal_energy")]
    print(f"NOTE equal-energy conditions span {min(ee):.1f} to {max(ee):.1f} (scaled: "
          f"{min(by_id[c]['accuracy'] for c in by_id if c.startswith('random_equal_energy_scaled')):.1f} to "
          f"{max(by_id[c]['accuracy'] for c in by_id if c.startswith('random_equal_energy_scaled')):.1f}); plan says 58.8 to 63.8")
    print(f"NOTE {len(conditions)} conditions in verdict_n13.json arms; plan says 33")

    notes = {
        "arm_names": "Public condition id -> banked arm name: " + "; ".join(f"{k} = {a}" for k, a in arm_names.items()) + ".",
        "conditions": f"All {len(conditions)} conditions scored on the 500 problems in verdict_n13.json arms (the plan's '33' is the "
                      "spec's planned count of model loads; 31 are banked).",
        "accuracy": "Percent of 500 competition problems answered right, one sample each, temperature 0.6, top-p 0.95, up to "
                    "16,384 new tokens; the answer is the text after the last 'Answer:' line, else the last boxed expression, "
                    "graded by math_verify (N13_SPEC.md §2.1-2.4).",
        "range_consistent_with_data": "Wilson 95 % interval of the accuracy, n = 500 (verdict_n13.json arms.<arm>.wilson).",
        "tensors_from": "For each condition, which model the attention matrices, all norm weights (including the norm that scales "
                        "each MLP's output) and the embedding and output matrices come from; 'base' = allenai/Olmo-3-1025-7B main, "
                        "'math' = allenai/Olmo-3.1-7B-RL-Zero-Math (N13_SPEC.md §3.2 lines 407-414, n13_arms.py arm_spec; "
                        "asserted against build.json spec.non_mlp where a build.json is banked).",
        "mlp_part": "Which part of the MLP change (mathematics minus base, per entry, 32-bit) is added to the base; a built model is "
                    "16-bit(base + c * M * change) per tensor (N13_SPEC.md §3.1).",
        "seed_set": "'first' = per-problem sampling seed 1,000,000 + index; 'second' = 2,000,000 + index.",
        "built": "From the condition's build.json: entries inside the mask, entries changed, and the realized squared change of the "
                 "MLP and of the other tensors; null where no build.json is banked (the four reference conditions, the four "
                 "'except its most-changed' conditions and random_equal_energy_own_change_3).",
        "examples": "Problems 0, 1 and 2 (file order) for every condition, copied from the condition's samples.jsonl: 'text' is an "
                    "index into texts (each distinct answer text is stored once, verbatim and complete); extracted, score, "
                    "new_tokens, hit_length_cap, finish_reason, extraction (answer_line / boxed / none) as banked.",
        "problems": "prompt = the exact rendered prompt banked with every answer; statement = the prompt with the fixed template's "
                    "first and last sentences removed (N13_SPEC.md §2.2); gold as banked.",
        "texts": "Model answers, verbatim and untruncated; up to 47,900 characters. They are the reason data.js exceeds 300 KB.",
        "noise_of_one_reading": "sigma = 100 * sqrt(sum (x_A - x_B)^2 / 2) / 500 over two seed sets of the same weights, the larger "
                                "of base and mathematics; a difference must exceed 2 * sqrt(2) * sigma (thresholds.json sigma, "
                                "delta).",
        "gain": "Base 57.2, mathematics 92.0, gain 34.8 points, required to exceed 2.5 x the threshold = 13.60 "
                "(thresholds.json G-GATE); halfway and quarter lines H and Q.",
        "equal_count_comparison": "The registered count-matched pair: most-changed 0.1 % of units (352 units x 12,288 = 4,325,376 "
                                  "entries) against the scaled random sample at q = 352 / 352,256 (4,325,381 entries kept, "
                                  "dare_k0.1 build.json kept). Matched in count, not in energy.",
        "comparisons": "The 14 registered paired comparisons on the 500 problems: difference in points (first minus second), "
                       "paired bootstrap 95 % range, exact discordant-pair test p, and the p adjusted over the family "
                       "(verdict_n13.json comparisons).",
        "word_problems_1319": "The 1,319 grade-school word problems at the 1 % size: three paired differences. 'REPLICATES on 3 of "
                              "3' in the record means these three comparisons, not three seeds; the head's read (READ_head.md "
                              "item 3) says the two random-unit labels are fragile and are to be quoted as consistent, not "
                              "replicated. The banked label (verdict_n13.json gsm8k.<pair>.label) is not carried.",
        "checks": "Each condition's accuracy recomputed as counts.correct / 500 from the metrics.json next to its samples.jsonl (both in meta.sources), stored beside verdict_n13.json arms.<arm>.acc.",
    }
    write(data, notes)


if __name__ == "__main__":
    main()
