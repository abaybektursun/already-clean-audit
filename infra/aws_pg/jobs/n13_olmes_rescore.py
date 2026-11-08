"""N13 R0 rescorer (studies/next/07_PICKS/N13_SPEC.md §6.1 G-KNOWN failing input). Run under the olmes venv on the instance.

    python -m infra.aws_pg.jobs.n13_olmes_rescore --dir <olmes output dir> --seed 0 --out rescore.json [--scorer olmes|string] [--tokenizer DIR]

For each of the 7 subjects: read task-*-minerva_math_<subject>-predictions.jsonl (label = the normalized gold; model_output = the 4
continuations); score every continuation with olmes' own Minerva scorer, max(exact_match, exact_match_flex) of
GenericMATH.process_results under the task_config logged in the matching -metrics.json, so pass@1 = c/4 per document as PassAtK;
once with the true labels (the identity: it must equal olmes' stored pass_at_1 for every document) and once with the labels deranged
within the subject (n13_score.derangement, seed 0). Writes the macro means over the 7 subjects. --scorer string (normalized string
equality) exists only for the offline gate, where olmes and lm_eval are not installed. --tokenizer counts the prompt contexts longer
than max_length - max_gen_toks = 2048 - 1024 tokens (olmes' vLLM path keeps only the last 1,024 context tokens of those).
Fail-loud: no try/except, no fallback.
"""
from __future__ import annotations

import argparse
import json
import pathlib

from infra.pg.n13_score import derangement, normalise

SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus")


def olmes_scorer(task_config: dict):
    from oe_eval.tasks.oe_eval_tasks.minerva_math import GenericMATH

    class _Task:
        extract_answers = GenericMATH.extract_answers
        process_results = GenericMATH.process_results

    t = _Task()
    t.task_config = task_config

    def score(label: str, continuation: str) -> int:
        r = t.process_results({"answer": label}, [continuation])
        return max(r["exact_match"], r["exact_match_flex"])
    return score


def string_scorer(task_config: dict):
    def score(label: str, continuation: str) -> int:
        tail = continuation.split("The final answer is")[-1].split("I hope")[0].strip(" $.")
        return int(normalise(tail) == normalise(label))
    return score


def rescore(d: pathlib.Path, seed: int, scorer_name: str, tokenizer=None) -> dict:
    per_true, per_der, docs_equal, n_docs, ctx = {}, {}, True, 0, {"over_1024": 0, "n": 0, "max": 0}
    tok = None
    if tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer)
    for s in SUBJECTS:
        pf = sorted(d.glob(f"task-*-minerva_math_{s}-predictions.jsonl"))
        mf = sorted(d.glob(f"task-*-minerva_math_{s}-metrics.json"))
        assert len(pf) == 1 and len(mf) == 1, (s, pf, mf)
        cfg = json.loads(mf[0].read_text())["task_config"]
        score = (olmes_scorer if scorer_name == "olmes" else string_scorer)(cfg)
        rows = [json.loads(l) for l in pf[0].read_text().splitlines() if l.strip()]
        labels = [r["label"] for r in rows]
        der = derangement(len(rows), seed)
        pt, pd = [], []
        for i, r in enumerate(rows):
            conts = [x["continuation"] for x in r["model_output"]]
            assert len(conts) == 4, (s, r["doc_id"], len(conts))
            a = sum(score(labels[i], c) for c in conts) / 4
            b = sum(score(labels[der[i]], c) for c in conts) / 4
            pt.append(a)
            pd.append(b)
            docs_equal &= abs(a - r["metrics"]["pass_at_1"]) < 1e-12
        n_docs += len(rows)
        per_true[s], per_der[s] = sum(pt) / len(pt), sum(pd) / len(pd)
        if tok is not None:
            rf = sorted(d.glob(f"task-*-minerva_math_{s}-requests.jsonl"))
            seen = set()
            for l in rf[0].read_text().splitlines():
                r = json.loads(l)
                if r["doc_id"] in seen:
                    continue
                seen.add(r["doc_id"])
                n = len(tok(r["request"]["context"], add_special_tokens=False)["input_ids"])
                ctx["n"] += 1
                ctx["over_1024"] += int(n > 1024)
                ctx["max"] = max(ctx["max"], n)
    return {"scorer": scorer_name, "seed": seed, "per_subject_true": per_true, "per_subject_deranged": per_der,
            "identity_macro": sum(per_true.values()) / 7, "deranged_macro": sum(per_der.values()) / 7, "identity_docs_equal": bool(docs_equal),
            "n_docs": n_docs, "context_tokens": ctx if tok is not None else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scorer", default="olmes", choices=("olmes", "string"))
    ap.add_argument("--tokenizer", default="")
    a = ap.parse_args()
    r = rescore(pathlib.Path(a.dir), a.seed, a.scorer, a.tokenizer or None)
    pathlib.Path(a.out).write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
