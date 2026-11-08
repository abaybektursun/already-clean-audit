"""Recount of G-GRADER's self-equality under the two rules, with the count of golds that parse to nothing (N13_SPEC.md amendment A1).
Run: PYTHONPATH=$PWD <n13 venv python> studies/next/n13/amendments/grader_recount.py <MATH-500 test.jsonl> <out.json>."""
import json, sys
from math_verify import parse, verify
from infra.pg import n13_score as S
golds = [json.loads(l)["answer"] for l in open(sys.argv[1])]
out = {"n_golds": len(golds)}
for rule in ("registered", "dollar_both"):
    pg = [parse(S.wrap(g, rule, "gold")) for g in golds]
    self_eq = [S.grade(g, g, rule)["score"] for g in golds]
    out[rule] = {"self_equal": int(sum(self_eq)), "self_unequal": len(golds) - int(sum(self_eq)),
                 "gold_parses_to_nothing": int(sum(1 for p in pg if len(p) == 0)),
                 "examples_unequal": [g for g, s in zip(golds, self_eq) if not s][:12]}
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps({r: {k: out[r][k] for k in ("self_equal", "self_unequal", "gold_parses_to_nothing")} for r in ("registered", "dollar_both")}))
