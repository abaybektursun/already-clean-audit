"""S3 close-out instrument check: re-extract and re-grade every gate-arm answer on the laptop with the frozen code (n13_score.extract,
grade; math-verify 0.9.0) and compare with the recorded mode, extracted string and score; recount accuracy and G-COMPLETE.
Run: PYTHONPATH=$PWD <n13 venv python> studies/next/n13/s3/regrade.py <banks dir: <arm>/<arm>/<bench>/generations.jsonl> <out.json>"""
import hashlib, json, pathlib, sys
from infra.pg import n13_score as S
root, out = pathlib.Path(sys.argv[1]), {}
N = {"math500": 500, "gsm8k": 1319}
for arm in ("base", "rl", "zero", "id"):
    for b in ("math500", "gsm8k"):
        p = root / arm / arm / b / "generations.jsonl"
        if not p.exists():
            out[f"{arm}/{b}"] = "absent"
            continue
        recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        bad = []
        for r in recs:
            m, x, _ = S.extract(r["text"])
            g = S.grade(r["gold"], x)["score"]
            if (m, x, g) != (r["mode"], r["extracted"], r["score"]):
                bad.append({"problem_index": r["problem_index"], "recorded": [r["mode"], r["extracted"], r["score"]], "laptop": [m, x, g]})
        met = json.loads((p.parent / "metrics.json").read_text())
        out[f"{arm}/{b}"] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "n": len(recs), "correct": sum(r["score"] for r in recs),
                             "acc": S.acc([r["score"] for r in recs]), "metrics_correct": met["counts"]["correct"],
                             "regrade_mismatches": len(bad), "examples": bad[:5],
                             "G-COMPLETE": S.g_complete(recs, N[b], "B" if arm in ("zero", "id") else "A")}
pathlib.Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + "\n")
for k, v in out.items():
    print(k, v if v == "absent" else {x: v[x] for x in ("n", "correct", "acc", "metrics_correct", "regrade_mismatches")} | {"G-COMPLETE": v["G-COMPLETE"]["pass"]})
