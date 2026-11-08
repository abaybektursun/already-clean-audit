"""Step 5. Among the 1,238 late_primary jobs, the parent-test judge's flag (flagged = at most 1 of 4 random-decoy
judgments correct, SPEC_D1 §6.3; field `flagged` in q2_clusters.jsonl) and mean random-decoy accuracy (`acc_random`)
for jobs that pass (p < 0.05 in s04) against jobs that fail. p_u point estimate per group with a_ctrl from
metrics.json q2.p_u. Outputs: out/s06_judge_by_pass.json, out/s06_judge_by_pass.txt."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu

OUT = Path(__file__).resolve().parent / "out"
BANK = Path("/Users/abaybektursun/projects/interp_research_banks/neurons")
banked = {r["cluster"]: r for r in json.load(open(OUT / "q2_banked.json")) if not r["echo"]}
pc = json.load(open(OUT / "s04_per_cluster_p.json"))["per_cluster"]
a_ctrl = json.load(open(BANK / "w5/verdict/metrics.json"))["q2"]["p_u"]["a_ctrl"]
prim = [c for c, r in banked.items() if r["status"] == "late_primary"]
assert all("flagged" in banked[c] for c in prim)


def wilson(k, n, z=1.959963984540054):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [(c - h) / d, (c + h) / d]


out = {}
for name, sel in (("pass", [c for c in prim if pc[c]["p"] < 0.05]), ("fail", [c for c in prim if pc[c]["p"] >= 0.05])):
    k = sum(banked[c]["flagged"] for c in sel)
    acc = np.array([banked[c]["acc_random"] for c in sel])
    A = acc.mean()
    out[name] = {"n": len(sel), "flagged": k, "flag_rate": k / len(sel), "flag_rate_wilson95": wilson(k, len(sel)),
                 "mean_acc_random": A, "p_u_point": float(np.clip((a_ctrl - A) / (a_ctrl - 0.5), 0, 1)),
                 "mean_acc_hard": float(np.mean([banked[c]["acc_hard"] for c in sel]))}
    out[name]["_acc"] = acc
table = [[out["pass"]["flagged"], out["pass"]["n"] - out["pass"]["flagged"]],
         [out["fail"]["flagged"], out["fail"]["n"] - out["fail"]["flagged"]]]
odds, p_f = fisher_exact(table)
mw = mannwhitneyu(out["pass"].pop("_acc"), out["fail"].pop("_acc"))
out["fisher_flagged"] = {"table_[flagged,not]_pass_then_fail": table, "odds_ratio": odds, "p_two_sided": p_f}
out["mannwhitney_acc_random_p"] = float(mw.pvalue)
out["a_ctrl"] = a_ctrl
json.dump(out, open(OUT / "s06_judge_by_pass.json", "w"), indent=1, default=float)
L = [f"{g}: n {out[g]['n']}, flagged {out[g]['flagged']} ({out[g]['flag_rate']:.3f}, Wilson {[round(x,3) for x in out[g]['flag_rate_wilson95']]}), "
     f"mean acc_random {out[g]['mean_acc_random']:.4f}, p_u point {out[g]['p_u_point']:.3f}, mean acc_hard {out[g]['mean_acc_hard']:.4f}"
     for g in ("pass", "fail")]
L += [f"Fisher exact on flagged: odds ratio {odds:.3f}, p {p_f:.4f}; Mann-Whitney on acc_random: p {mw.pvalue:.4f}; a_ctrl {a_ctrl:.4f}"]
(OUT / "s06_judge_by_pass.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
