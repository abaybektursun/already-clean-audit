"""A26 and A25 on the attempt-9 smoke (N13_SPEC.md A25, A26): re-extract and re-grade every banked smoke answer with the code before
A26 (commit f55da49b6) and after it, and run G-COMPLETE before and after A25 on the same records.
Run: PYTHONPATH=$PWD <n13 venv python> studies/next/n13/amendments/a26_rescore.py <old n13_score.py> <dir with base/ and rl/
generations.jsonl> <out.json>. The old n13_score.py is `git show f55da49b6:infra/pg/n13_score.py`. The inputs are
s3://pg-lineage-769063704696/results/next_n13/s2smoke-09/out/smoke_{base,rl}/{base,rl}/math500/generations.jsonl.
Fail-loud: no try/except."""
import hashlib, importlib.util, json, pathlib, sys
from infra.pg import n13_score as NEW

spec = importlib.util.spec_from_file_location("n13_score_f55da49b6", sys.argv[1])
OLD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(OLD)
SHA = {"base": "4ab1220b7e9f220a779e25b69c9e21e797ea9fc06e01946988a46484928fd6ea",
       "rl": "a01c39653d02cd3282388ff8dc72e9b16ac7c475a3cf79ca8e10a6289b992bfb"}
out = {"old_code": "f55da49b6:infra/pg/n13_score.py", "old_sha256": hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest()}
for m in ("base", "rl"):
    p = pathlib.Path(sys.argv[2]) / m / "generations.jsonl"
    assert hashlib.sha256(p.read_bytes()).hexdigest() == SHA[m], f"{p} is not the banked S3 file"
    recs = [json.loads(l) for l in p.read_text().splitlines()]
    ident, rows = [], []
    for r in recs:
        mo, xo, _ = OLD.extract(r["text"])
        so = OLD.grade(r["gold"], xo)["score"]
        ident.append((mo, xo, so) == (r["mode"], r["extracted"], r["score"]))
        mn, xn, _ = NEW.extract(r["text"])
        sn = NEW.grade(r["gold"], xn)["score"]
        if (mn, xn, sn) != (mo, xo, so):
            rows.append({"problem_index": r["problem_index"], "gold": r["gold"], "old": [mo, xo, so], "new": [mn, xn, sn]})
    n = len(recs)
    old_c, new_c = sum(r["score"] for r in recs), sum(r["score"] for r in recs) + sum(x["new"][2] - x["old"][2] for x in rows)
    modes = lambda f: {k: sum(f(r) == k for r in recs) for k in ("answer_line", "boxed", "none")}
    n_sub = 300
    gc = {"old_rule": OLD.g_complete(recs, n_sub, "A"), "new_rule": NEW.g_complete(recs, n_sub, "A")}
    if len(recs) == n_sub:
        assert all(r["seed"] == NEW.SEED_BASE["A"] + r["problem_index"] for r in recs)
    out[m] = {"n_records": n, "identity_old_code_reproduces_recorded": int(sum(ident)), "correct_old": old_c, "correct_new": new_c,
              "acc_old": old_c / n, "acc_new": new_c / n,
              "wrong_to_right": sorted(x["problem_index"] for x in rows if x["old"][2] == 0 and x["new"][2] == 1),
              "right_to_wrong": sorted(x["problem_index"] for x in rows if x["old"][2] == 1 and x["new"][2] == 0),
              "changed_same_score": sorted(x["problem_index"] for x in rows if x["old"][2] == x["new"][2]),
              "modes_old": modes(lambda r: r["mode"]), "modes_new": modes(lambda r: NEW.extract(r["text"])[0]),
              "changes": sorted(rows, key=lambda x: x["problem_index"]), "G-COMPLETE_on_300_submitted": gc}
json.dump(out, open(sys.argv[3], "w"), indent=1, ensure_ascii=False)
for m in ("base", "rl"):
    o = out[m]
    print(m, {k: o[k] for k in ("n_records", "identity_old_code_reproduces_recorded", "correct_old", "correct_new", "wrong_to_right",
                                "right_to_wrong", "changed_same_score", "modes_old", "modes_new")})
    print("  G-COMPLETE old", o["G-COMPLETE_on_300_submitted"]["old_rule"], "\n  G-COMPLETE new", o["G-COMPLETE_on_300_submitted"]["new_rule"])
