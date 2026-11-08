"""N22 ten raw examples, the zero-GPU part (SPEC §10): the first ten freeze-half items exactly as the model will see
them, from studies/next/n22/items.json and the OLMo-3 tokenizer. No model output exists for these prompts in any
bank; the model's answer and its probability, the lens readout and the outputs after each edit are written by the
L1 job as samples/raw_examples_t15.jsonl (the first GPU step) and are marked PENDING here.

    PYTHONPATH=. python studies/next/n22/samples/raw_examples.py --tok <dir with tokenizer.json> --out studies/next/n22/samples
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from infra.pg import n22  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(a.tok)
    rec = json.loads((ROOT / "studies/next/n22/items.json").read_text())
    items = rec["items"]
    num = rec["number_ids"]
    ex = [it for it in items if it["half"] == "freeze"][:10]
    out, md = [], ["# N22 raw examples, zero-GPU part", "",
                   "The first ten freeze-half items of `studies/next/n22/items.json` (sha256 in `n22_score.ITEMS_SHA`), "
                   "as the model sees them. Every model output is PENDING: no bank holds a forward of these prompts. "
                   "The L1 job writes them first, to `samples/raw_examples_t15.jsonl`, before any statistic.", "",
                   "The prompt (BOS, then three solved lines, then the query), item 0:", "", "```",
                   "<|endoftext|>" + tok.decode(ex[0]["ids"][1:]) + "|", "```", "",
                   "| item | query | s1 | s2 | y (token id) | v | f1 = v c + d | f2 = v + d | partner's f1, f2 | slot-1 counterfactual | slot-2 counterfactual |",
                   "|---|---|---|---|---|---|---|---|---|---|---|"]
    for it in ex:
        p = items[it["partner"]]
        v = it["v"][0]
        q = n22.line(it["a"], it["b"], it["c"], it["d"], False).strip()
        toks = [tok.decode([t]) for t in it["ids"][n22.Q0:]]
        ic = it["ic"][0]
        r = {"item": it["i"], "half": it["half"], "query": q, "query_tokens": toks, "query_ids": it["ids"][n22.Q0:],
             "s1": it["s1"], "s2": it["s2"], "y": it["y"], "y_id": num[it["y"]],
             "v": it["v"], "targets": [{"v": x, "f1": x * it["c"] + it["d"], "f2": x + it["d"],
                                        "f1_id": num[x * it["c"] + it["d"]], "f2_id": num[x + it["d"]],
                                        "f1_partner": x * p["c"] + p["d"], "f2_partner": x + p["d"]} for x in it["v"]],
             "ic": [{"v": c["v"], "slot1": n22.line(*c["slot1"], False).strip(), "slot1_answer": (c["slot1"][0] + c["slot1"][1]) * c["slot1"][2] + c["slot1"][3],
                     "slot2": n22.line(*c["slot2"], False).strip(), "slot2_answer": (c["slot2"][0] + c["slot2"][1]) * c["slot2"][2] + c["slot2"][3]}
                    for c in it["ic"]],
             "PENDING": ["model top-1 and its probability", "p(y)", "lens readout top-10 at l1, l2, lm",
                         "outputs after each edit arm (v index 0)"]}
        out.append(r)
        md.append(f"| {it['i']} | `{q}` | {it['s1']} | {it['s2']} | {it['y']} ({num[it['y']]}) | {v} | "
                  f"{v * it['c'] + it['d']} | {v + it['d']} | {v * p['c'] + p['d']}, {v + p['d']} | "
                  f"`{n22.line(*ic['slot1'], False).strip()}` | `{n22.line(*ic['slot2'], False).strip()}` |")
    md += ["", "What they show at zero GPU: every number is one token; the query occupies positions 49-62 of a 63-token "
           "prompt (BOS at 0); the operands sit at 50, 53, 57, 60 and the answer is predicted at 62, the token `Ġ` after "
           "`Ġ=`. For every item, f1(v) and f2(v) differ from each other, from v and from every number in the item, "
           "so a raised target cannot be the injected token or a copied operand. The counterfactual of slot 1 keeps c "
           "and d and its answer is f1(v); that of slot 2 keeps d and its answer is f2(v).", "",
           "PENDING, written by the job: the model's top-1 and probability, p(y), the lens top-10 at the bands, and "
           "the outputs after LENS_U, LENS_M, DOM_M, ANS_U, ANS_M and their random floors for v index 0."]
    o = pathlib.Path(a.out)
    (o / "raw_examples_zero_gpu.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    (o / "raw_examples_zero_gpu.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[9:21]))


if __name__ == "__main__":
    main()
