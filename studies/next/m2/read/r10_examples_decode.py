"""Reader of M2, phase 3: the job's raw-example exemplar texts against the banked T_15 contexts (the grid's
samples/contexts.jsonl, path as argv[1]): the marked token " [[token]] " must be the the banked token at that rank, and the
position must equal the banked tok. Output: read/r10_examples_decode.json.
"""
import json, sys
from pathlib import Path

M2 = Path(__file__).resolve().parents[1]


def main():
    ctx = {}
    for line in Path(sys.argv[1]).open():
        c = json.loads(line)
        ctx[c["pole"]] = c["topk"]
    n, ok, bad = 0, 0, []
    for h in ("freeze", "test"):
        for line in (M2 / f"banks/score-06/m2/samples/raw_examples_{h}.jsonl").open():
            r = json.loads(line)
            for e in r["exemplars"]:
                n += 1
                b = ctx[r["carrier"]][e["rank"]]
                t = e["text"]
                mark = " [[" + b["token"] + "]] "   # m2_score.decode_ctx: before + " [[" + token + "]] " + after
                if b["tok"] == e["tok"] and mark in t:
                    ok += 1
                else:
                    bad.append([r["unit"], e["rank"], e["tok"], b["tok"], mark, b["token"]])
    out = {"exemplars": n, "token_and_position_equal": ok, "differ": bad}
    (M2 / "read/r10_examples_decode.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()
