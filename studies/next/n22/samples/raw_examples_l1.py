"""N22 raw examples at T_15 from L1's banks (spec §10, the PENDING part), freeze half only.

L1 answered 2 + 5 items, under N_MIN = 50 per half, so the job did not run the edits and did not write
samples/raw_examples_t15.jsonl (n22_score.run_point returns before it). This script fills, from the banked arrays,
every pending field that exists: the model's top-1 and its probability, p(y), the rank of y among the 1,000 number
tokens, and the ranks of s1, s2, y in the full vocabulary at every source block under the full lens and the logit
lens. The lens top-10 at the bands needs the residuals, which the job does not bank; the outputs after the edits do
not exist. The first ten freeze items are all wrong, so the first answered freeze item is added (spec §10).

    PYTHONPATH=. python studies/next/n22/samples/raw_examples_l1.py --tok <dir with tokenizer.json>
"""
import argparse
import json
import pathlib

import numpy as np
import transformers

D = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True)
    a = ap.parse_args()
    tok = transformers.AutoTokenizer.from_pretrained(a.tok)
    rec = json.loads((D / "items.json").read_text())
    items, num = rec["items"], rec["number_ids"]
    bank = D / "banks" / "l1-01" / "n22" / "t15"
    acc = json.loads((bank / "accuracy.json").read_text())
    lp = np.load(bank / "clean_lp_numbers.npy")
    rd = np.load(bank / "readout.npz")
    ranks, variants, names = rd["ranks"], list(rd["variants"]), list(rd["targets"])
    fz = [it for it in items if it["half"] == "freeze"]
    ex = fz[:10]
    if not any(acc["correct"][it["i"]] for it in ex):
        ex = ex + [it for it in fz if acc["correct"][it["i"]]][:1]
    out = []
    for it in ex:
        i = it["i"]
        assert it["half"] == "freeze"
        r = {"item": i, "prompt": tok.decode(it["ids"][1:]), "s1": it["s1"], "s2": it["s2"], "y": it["y"],
             "model_top1": tok.decode([acc["top1"][i]]), "model_top1_p": round(float(np.exp(acc["top1_lp"][i])), 4),
             "p_y": round(float(np.exp(lp[i, it["y"]])), 5),
             "rank_y_among_numbers": int((lp[i] > lp[i, it["y"]]).sum()) + 1,
             "correct": bool(acc["correct"][i]),
             "lens_top10_at_bands": "not produced: the job banks no residuals; edits not run (2 and 5 answered < 50)",
             "after_edits": "not produced: the edits did not run (fewer than N_MIN = 50 answered per half)"}
        for v in ("full", "logit"):
            for k in ("s1", "s2", "y"):
                r[f"rank_{k}_{v}_by_block"] = ranks[i, :, variants.index(v), names.index(k)].tolist()
        out.append(r)
    (D / "samples" / "raw_examples_t15.json").write_text(json.dumps(out, indent=1))
    L = ["# N22 raw examples at T_15 (L1 `l1-01`), freeze half", "",
         "Filled from `banks/l1-01/n22/t15/{accuracy.json, clean_lp_numbers.npy, readout.npz}` by "
         "`samples/raw_examples_l1.py`. The job wrote no `raw_examples_t15.jsonl`: 2 freeze and 5 test items were "
         "answered, under the 50 per half the edits need, so the lens top-10 at the bands and the outputs after the "
         "edits were not produced. The first ten freeze items are all wrong; the first answered freeze item (714) is "
         "added. Ranks are in the full vocabulary of 100,278 (1 = top); blocks are source blocks 0–29.", "",
         "| item | query | s1 | s2 | y | model top-1 (p) | p(y) | rank of y among 1,000 numbers | correct |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in out:
        q = r["prompt"].split("\n")[-1].strip()
        L.append(f"| {r['item']} | `{q}` | {r['s1']} | {r['s2']} | {r['y']} | {r['model_top1']} ({r['model_top1_p']}) | "
                 f"{r['p_y']} | {r['rank_y_among_numbers']} | {'yes' if r['correct'] else 'no'} |")
    L += ["", "Best (smallest) rank of s1, s2 and y over blocks 0–29, with the block, under the full lens and the "
          "logit lens:", "", "| item | full s1 | full s2 | full y | logit s1 | logit s2 | logit y |", "|---|---|---|---|---|---|---|"]
    for r in out:
        cells = []
        for v in ("full", "logit"):
            for k in ("s1", "s2", "y"):
                x = r[f"rank_{k}_{v}_by_block"]
                b = int(np.argmin(x))
                cells.append(f"{x[b]} (block {b})")
        L.append(f"| {r['item']} | " + " | ".join(cells) + " |")
    L.append("")
    (D / "samples" / "raw_examples_t15.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
