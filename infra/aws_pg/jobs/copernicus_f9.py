"""CO-F9 (item 39) — W0-10 minority-recipe forensics on the five passing bases. CHARTER row.

Spec §3 Tier F: "W0-10 minority-recipe forensics on the five passing bases (metadata mining, no
GPU)." Item 39 names them: gemma-2b, Llama-3.2-3B, Qwen2.5-3B, Mistral-7B-v0.3, gemma-7b.

THE PRECONDITION, carried rather than glossed. W0-10 did not return a null — it returned
**VOID-underpowered**: held-out accuracy 0.333 against a 0.667 majority baseline with
n_minority_held = 1. An underpowered arm says nothing about whether recipe is readable from
weights, so this dossier asks the only question that can be asked from metadata: is there a roster
on these bases that WOULD be powered?

FROZEN CRITERIA, READ AT RUN TIME from W0-10's own manifest so the comparison is like-for-like and
no threshold is invented here. Their values are deliberately not restated in this docstring: a
number copied into prose beside a number read from an artifact is how the two drift apart, which
this program has already paid for once.

CHARTER: this produces a described artifact and a roster table. It scores no hypothesis.

    pg run infra.aws_pg.jobs.copernicus_f9 --args "" --no-weights \
        --type r7i.2xlarge --s3-prefix results/copernicus_f9 --budget-min 45
"""
import argparse
import json
import pathlib
import subprocess

from infra.pg import copernicus_core as C
from infra.pg import humboldt_core as H

ARM = "CO-F9"
QUEUE_ITEM = "39"
WAVE = "WF"
W010_S3 = "s3://pg-lineage-769063704696/results/darwin_w0_10/out"
# item 39's five bases, verbatim
BASES = ("google/gemma-2b", "meta-llama/Llama-3.2-3B", "Qwen/Qwen2.5-3B",
         "mistralai/Mistral-7B-v0.3", "google/gemma-7b")
# recipe classes and the tag/name evidence that declares each one
CLASS_MARKERS = {
    "dpo": ("dpo", "orpo", "kto", "simpo"),
    "rlhf": ("rlhf", "ppo", "rlaif", "reward"),
    "sft": ("sft", "instruct", "chat", "finetune", "fine-tune"),
}
# DISCLOSED SENSITIVITY, NOT PART OF THE FROZEN CLASSIFICATION. The rlhf markers above name the
# PPO-era vocabulary, and rlhf is the binding constraint at every base -- so if the marker set has
# simply gone stale, "no three-class roster exists" is an artifact of the instrument and would
# mis-charter the successor the same way the truncated census did. These are the post-PPO RL
# markers the frozen set cannot see. Counted and reported SEPARATELY: the frozen counts stay the
# statistic, and swapping markers after seeing rlhf come back near zero would be exactly the
# post-unblind rule change the program forbids. What the successor does with the gap is its call.
RL_MODERN_MARKERS = ("grpo", "rloo", "reinforce", "rlvr", "verl", "openrlhf", "ppo-max")


def declared_class(model_id: str, tags) -> str:
    """The DECLARED recipe, from tags and the repo name. Declaration is the point: item 40 records
    that only 11 of 192 smolVLA derivatives declare a recipe at all, and a classifier can only be
    trained on labels somebody wrote down. Checked most-specific first — 'orpo' before 'sft', or
    every preference-tuned repo would read as sft on the word 'instruct'."""
    # LINEAGE TAGS ARE NOT RECIPE DECLARATIONS. Every repo returned by `base_model:finetune:<base>`
    # carries that string as a literal tag, and it contains "finetune" — an sft marker. Left in the
    # haystack it makes `undeclared` UNREACHABLE and silently redefines sft as "everything that is
    # not dpo or rlhf": the second F9 run returned undeclared=0 at all five bases, with repos like
    # `mlabonne/OrcaGemma-2B` and `abhinand/gemma-2b-tamil` — which declare no recipe whatsoever —
    # counted as sft. The base_model:* family says what a repo was trained FROM, never how.
    tags = [str(t).lower() for t in (tags or [])]
    tags = [t for t in tags if not t.startswith("base_model:")]
    hay = " ".join([model_id.lower()] + tags)
    for cls in ("dpo", "rlhf", "sft"):
        if any(m in hay for m in CLASS_MARKERS[cls]):
            return cls
    return "undeclared"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--w010-s3", default=W010_S3)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    H.hf_setup()
    from huggingface_hub import HfApi
    api = HfApi()

    man = json.loads(subprocess.run(
        ["aws", "s3", "cp", f"{a.w010_s3}/manifest.json", "-", "--region", "us-east-1"],
        check=True, capture_output=True, text=True).stdout)
    atlas = [json.loads(l) for l in subprocess.run(
        ["aws", "s3", "cp", f"{a.w010_s3}/atlas.jsonl", "-", "--region", "us-east-1"],
        check=True, capture_output=True, text=True).stdout.splitlines() if l.strip()]
    verdict_row = next((r for r in atlas if r["statistic"] == "recipe_classifier_verdict"), None)
    assert verdict_row is not None, "W0-10 artifact carries no classifier verdict row"
    w010_verdict = json.loads(verdict_row["flags"])
    MIN_PER_CLASS = int(man["min_per_class"])
    MIN_MINORITY_HELD = int(man["min_minority_held"])
    print(f"[f9] W0-10 precondition: verdict={w010_verdict.get('verdict')} "
          f"acc={w010_verdict.get('heldout_accuracy'):.3f} vs majority "
          f"{w010_verdict.get('majority_baseline'):.3f}; thresholds MIN_PER_CLASS={MIN_PER_CLASS} "
          f"MIN_MINORITY_HELD={MIN_MINORITY_HELD}", flush=True)

    exp = C.Expectations(out / "copernicus_expectations.jsonl")
    exp.log(ARM, "f9_roster_dossier",
            "on the five item-39 bases the RLHF class is the binding constraint; a 2-class "
            "sft-vs-dpo roster is available where a 3-class one is not",
            confidence="low-mid", basis="item 39 + W0-10's banked base_selection probe",
            family="census")
    at = C.Atlas(out, arm=ARM, run_id="copernicus_f9", wave=WAVE, queue_item=QUEUE_ITEM,
                 expected_ref=exp.require(ARM))
    gates = C.Gates(out)

    rows = []
    for base in BASES:
        # LINEAGE, NOT NAME. The first cut of this census enumerated with `search=<base name>`,
        # which returns repos whose NAME contains the string and says nothing about what they were
        # trained from — and it capped at `limit`, which BOUND: `search="gemma-2b"` returns exactly
        # 1000 results at limit=1000 and exactly 5000 at limit=5000, so the population is larger
        # than any cap and the resulting counts were truncation artifacts, not a census. Both
        # defects pushed the same way (a roster question answered with a name-match sample), so the
        # replacement is exhaustive (limit=None) and lineage-scoped.
        #
        # RELATION-SCOPED, because "derivative" is not "fine-tune". For Mistral-7B-v0.3 the
        # declared derivatives split 371 finetune / 369 adapter / 85 quantized / 5 merge: the
        # unscoped filter over-counts the roster by ~2.2x, and the largest contaminant is exactly
        # the LoRA mass CO-X1a exists to reject. A delta is only defined against full weights, so
        # `finetune` is the roster and the rest is recorded as the mass that was excluded.
        counts = {"dpo": 0, "rlhf": 0, "sft": 0, "undeclared": 0}
        members = {"dpo": [], "rlhf": [], "sft": []}
        seen = set()
        excluded = {}
        for rel in ("adapter", "quantized", "merge"):
            excluded[rel] = len(list(api.list_models(
                filter=f"base_model:{rel}:{base}", limit=None)))
        rl_modern = []
        for m in api.list_models(filter=f"base_model:finetune:{base}", limit=None):
            if m.id in seen or m.id == base:
                continue
            seen.add(m.id)
            tags = [str(t).lower() for t in (getattr(m, "tags", None) or [])
                    if not str(t).lower().startswith("base_model:")]
            hay = " ".join([m.id.lower()] + tags)
            if any(k in hay for k in RL_MODERN_MARKERS):
                rl_modern.append(m.id)
            cls = declared_class(m.id, getattr(m, "tags", None))
            counts[cls] += 1
            if cls in members:
                members[cls].append(m.id)
        three_ok = all(counts[c] >= MIN_PER_CLASS for c in ("sft", "dpo", "rlhf"))
        two_ok = all(counts[c] >= MIN_PER_CLASS for c in ("sft", "dpo"))
        minority = min(counts["sft"], counts["dpo"], counts["rlhf"])
        row = {"base": base, "n_candidates": len(seen), **counts,
               "enumeration": f"base_model:finetune:{base} (exhaustive, no limit)",
               "rl_modern_sensitivity": len(rl_modern),
               "rl_modern_not_already_rlhf": len(
                   [r for r in rl_modern if r not in members["rlhf"]]),
               "excluded_by_relation": excluded,
               "minority_class_size": minority,
               "three_class_roster_ok": three_ok, "two_class_roster_ok": two_ok,
               "binding_constraint": min(("sft", "dpo", "rlhf"), key=lambda c: counts[c]),
               "declared_fraction": (sum(counts[c] for c in ("sft", "dpo", "rlhf"))
                                     / max(len(seen), 1))}
        rows.append(row)
        at.add("f9_recipe_roster", value=float(minority), model_id=base, n=len(seen),
               flags={**row, "register": "CHARTER — described artifact, not a verdict",
                      "members_sample": {c: members[c][:5] for c in members}})
        print(f"[f9] {base}: n={len(seen)} sft={counts['sft']} dpo={counts['dpo']} "
              f"rlhf={counts['rlhf']} undeclared={counts['undeclared']} "
              f"3class={three_ok} 2class={two_ok}", flush=True)
        (out / "copernicus_f9.json").write_text(json.dumps({"partial": True, "rows": rows},
                                                           indent=2))

    n3 = sum(1 for r in rows if r["three_class_roster_ok"])
    n2 = sum(1 for r in rows if r["two_class_roster_ok"])
    gates.record("f9_any_powered_roster_exists", bool(n3 > 0 or n2 > 0),
                 n_bases=len(rows), n_three_class=n3, n_two_class=n2,
                 min_per_class=MIN_PER_CLASS, min_minority_held=MIN_MINORITY_HELD,
                 w010_precondition=w010_verdict.get("verdict"),
                 band="W0-10's OWN frozen thresholds, read from its manifest; a roster is powered "
                      "iff every scored class clears MIN_PER_CLASS",
                 prediction="RLHF is the binding constraint; 2-class rosters exist where 3-class "
                            "ones do not (§F9 expectation, low-mid)")

    rep = {"object": "copernicus_f9_v1", "arm": ARM, "program": C.PROGRAM,
           "queue_item": QUEUE_ITEM, "register": "CHARTER (described artifact, no verdict)",
           "bases": list(BASES), "w010_precondition": w010_verdict,
           "w010_base_selection": man.get("base_selection"),
           "min_per_class": MIN_PER_CLASS, "min_minority_held": MIN_MINORITY_HELD,
           "n_bases_three_class_ok": n3, "n_bases_two_class_ok": n2, "rows": rows,
           "provenance": C.provenance(), "utc": C.utc_now()}
    (out / "copernicus_f9.json").write_text(json.dumps(rep, indent=2, default=str))
    at.close()
    at.to_parquet()
    print("=" * 78, flush=True)
    print(f"COPERNICUS_F9_OK bases={len(rows)} three_class_ok={n3} two_class_ok={n2} "
          f"w010_precondition={w010_verdict.get('verdict')}", flush=True)


if __name__ == "__main__":
    main()
