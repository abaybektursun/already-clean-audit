"""m04: groups, and the parent test with its control. The judge's question word for word, example calls with the
two records the judge saw, and the unrelated-share formula with its inputs.

Reads only banked files in this repository (no network, no GPU) and writes data.js next to this file as
window.<ID>_DATA = {...}. Prints `path sha256` for every input.

    python3 extract.py
"""
import datetime
import hashlib
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]                       # walkthrough/<picture> -> already-clean -> studies -> repository root
ID = HERE.name.upper()
MAX_BYTES = 300_000
SOURCES = []


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
    data["meta"] = {"commit": commit, "link": "studies/already-clean/results/RESULTS.md",  # the file the data link opens
                    "sources": SOURCES,
                    "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                    "notes": notes}
    out = HERE / "data.js"
    out.write_text(f"window.{ID}_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    size = out.stat().st_size
    assert size <= MAX_BYTES, f"{out} is {size:,} bytes"
    print(f"wrote {out.relative_to(ROOT)}: {size:,} bytes")


PROMPT = "infra/pg/neurons_prompts/judge.md"
RESULTS = "studies/already-clean/results/RESULTS.md"
CLOSEOUT = "studies/already-clean/results/reads/closeout_W5.md"
JUDGE_PAGE = "studies/already-clean/walkthrough/judge/data.js"
SPEC = "studies/already-clean/SPEC_D1.md"
PROMPT_SHA_PREFIX = "11b5fd81"   # the judge prompt sha recorded in RESULTS.md


def one(pattern, text, what):
    m = re.findall(pattern, text)
    assert len(m) == 1, f"{what}: {len(m)} matches of {pattern!r}"
    return m[0]


def main():
    prompt = src(PROMPT).read_text(encoding="utf-8")
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    assert prompt_sha.startswith(PROMPT_SHA_PREFIX), prompt_sha
    system = prompt.split("<!-- system -->\n")[1].split("<!-- user -->\n")[0].rstrip("\n")
    user = prompt.split("<!-- user -->\n")[1].rstrip("\n")

    res = src(RESULTS).read_text().splitlines()
    finding = next(s for s in res if s.startswith("Finding: On O7"))
    gate = next(s for s in res if s.startswith("Judge validity gate:"))
    p_u, lo, hi = map(float, one(r"p_u = ([\d.]+) \[([\d.]+), ([\d.]+)\], lower bound", finding, "p_u"))
    A_txt, n_late = one(r"from A = ([\d.]+) over ([\d,]+) late-born clusters", finding, "A")
    a_ctrl_txt = one(r"against a_ctrl = ([\d.]+) on S", finding, "a_ctrl")
    hard = float(one(r"control accuracy ([\d.]+), below 0\.80", finding, "hard decoy"))
    gate_acc, gate_poles, gate_thr = one(r"a_ctrl ([\d.]+) on (\d+) poles \(pass; threshold ([\d.]+)\)", gate, "gate")
    close = src(CLOSEOUT).read_text()
    A_full, a_ctrl_full = map(float, one(r"\| A ([\d.]+); n_clusters 1238; a_ctrl ([\d.]+);", close, "closeout A"))

    t = src(JUDGE_PAGE).read_text()
    jp = json.loads(t[t.index("=") + 1:].rstrip().rstrip(";"))
    n_calls_late, n_calls_ctrl = 4 * jp["n_late"], 4 * jp["n_control"]
    A_re = jp["right"]["late"] / n_calls_late
    a_ctrl_re = jp["right"]["control"] / n_calls_ctrl
    formula = {
        "judge_accuracy_late_born": compare("A", round(A_re, 5), A_full, f"{CLOSEOUT} line '| A ... a_ctrl ...|'", 1e-5),
        "control_accuracy": compare("a_ctrl", round(a_ctrl_re, 5), a_ctrl_full, f"{CLOSEOUT} line '| A ... a_ctrl ...|'", 1e-5),
        "unrelated_share": compare("p_u", round((a_ctrl_re - A_re) / (a_ctrl_re - 0.5), 4), p_u,
                                   f"{RESULTS} 'Finding: On O7' line", 5e-4),
        "unrelated_share_from_rounded_inputs": round((float(a_ctrl_txt) - float(A_txt)) / (float(a_ctrl_txt) - 0.5), 4),
        "inputs_as_reported": {"A": float(A_txt), "a_control": float(a_ctrl_txt), "chance": 0.5},
        "range": [lo, hi], "rule_line": 0.20,
        "calls_right": {"late_born": jp["right"]["late"], "late_born_calls": n_calls_late,
                        "control": jp["right"]["control"], "control_calls": n_calls_ctrl},
        "late_born_groups": int(n_late.replace(",", "")), "control_groups": jp["n_control"],
    }
    print(f"PLAN 0.924, 0.812, 0.265 -> unrelated_share.inputs_as_reported = {formula['inputs_as_reported']}, "
          f"unrelated_share.banked = {p_u}; from the rounded inputs the formula gives "
          f"{formula['unrelated_share_from_rounded_inputs']}")
    control_gate = {"threshold": float(gate_thr), "tuning_accuracy": float(gate_acc), "tuning_sides": int(gate_poles),
                    "hard_decoy_control_accuracy": hard}
    print(f"PLAN 0.80, 0.910 -> control_check = {control_gate}")

    spec = src(SPEC).read_text()
    assert "Four judgments per cluster per decoy type" in spec and "|m − m'| ≤ 1" in spec

    def call(e):
        true_first = e["order"] == "td"
        return {"call": e["id"], "outcome": e["kind"], "snapshot_before": e["t"], "step_before": jp["steps"][e["t"]],
                "group_contexts": [{"before": b, "token": k, "after": a} for b, k, a in e["ex"]],
                "record_a": [{"before": b, "token": k, "after": a} for b, k, a in e["a"]],
                "record_b": [{"before": b, "token": k, "after": a} for b, k, a in e["b"]],
                "own_side_record": "A" if true_first else "B", "judge_answer": e["answer"],
                "judge_right": e["answer"] == ("A" if true_first else "B")}
    picks = {}
    for e in jp["examples"]:
        if e["kind"] in ("late-right", "late-wrong", "control-right") and e["kind"] not in picks:
            picks[e["kind"]] = call(e)
    assert len(picks) == 3
    data = {"judge_model": "anthropic/claude-sonnet-5-5", "sorter_model": "Claude Sonnet 5.5",
            "prompt_sha256": prompt_sha, "prompt_system": system, "prompt_user": user,
            "group_contexts_shown_at_most": 10, "record_contexts": 20, "judgments_per_group": 4,
            "sorting_agreement_rule": "|m - m'| <= 1 between two independent clusterings",
            "formula": formula, "control_check": control_gate,
            "example_calls": [picks["late-right"], picks["late-wrong"], picks["control-right"]]}
    notes = {
        "prompt_system": "The judge's system prompt, verbatim from infra/pg/neurons_prompts/judge.md (its sha256 is the prompt "
                         "sha recorded in RESULTS.md); prompt_user is the user turn with its three slots.",
        "example_calls": "Three calls copied from the first edition's judge/data.js (built from the laptop bank's judge record): "
                         "the first late-born call the judge got right, the first it got wrong, and the first control call it "
                         "got right, in that file's order. Texts are as the judge saw them (line breaks as the ↵ character); the "
                         "marked token is `token`, shown to the judge between ⟦ and ⟧. own_side_record says which record was the "
                         "group's own side at snapshot_before (index 0 to 15; step_before is its training step, from judge/data.js "
                         "steps); the other record is a random side of the same layer.",
        "outcome": "late-right / late-wrong: a group born late, judge right or wrong; control-right: the side's main group "
                   "against its own record at snapshot 14 (the positive control).",
        "formula": "unrelated share = (a_control - A) / (a_control - 0.5). recomputed values use the call counts in judge/data.js; "
                   "banked values are the closeout's A and a_control (closeout_W5.md) and RESULTS.md's p_u. "
                   "unrelated_share_from_rounded_inputs applies the formula to the reported three-decimal inputs.",
        "control_check": "The control had to score at least the threshold on the tuning sides (internal S_gate) before the test "
                        "sides were judged; hard_decoy_control_accuracy is the control under the secondary (hard) decoy, below "
                        "the threshold, so that row could not be measured.",
        "sorter_model": "The model that sorted each side's fifty into groups and named them (internal: clusterer); echo "
                        "contexts (g < 0 and u < 0) form one flagged group excluded from every count.",
    }
    write(data, notes)


if __name__ == "__main__":
    main()
