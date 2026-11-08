"""Run the public repository's scripts/check_neuron.py on the real OLMo 3 checkpoints.

Frozen before launch (2026-11-04):
  K1, K2: the script's read() on the exact 2,048-token corpus window of L03N6372+'s rank-1 context
          at steps 60,000 and 1,413,814 reproduces the banked activation within 2 percent.
  F1, F2: the same read with the wrong neuron (6373) must miss that line, or the gate has no room.
  S1, S2: the command a stranger runs is executed as a subprocess at both steps on one sentence;
          stdout is banked. Expected, not a gate: " lithium" is the top token at step 60,000.
Fail-loud: no try/except.
"""
import argparse
import json
import pathlib
import subprocess
import sys

SCRIPT = "scripts/check_neuron_public.py"
TEXT = "Battery maker 24M received funding for its SemiSolid lithium-ion battery"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    sys.path.insert(0, "scripts")
    import check_neuron_public as C
    windows = json.load(open("scripts/check_neuron_public_windows.json"))
    rows = {}
    for step in (60000, 1413814):
        model, tok = C.load(step)
        w = windows[str(step)]
        ids = [tok.bos_token_id] + w["ids"]
        at = w["offset"] + 1
        right = C.read(model, ids, 3, 6372)[at]
        wrong = C.read(model, ids, 3, 6373)[at]
        rows[str(step)] = {"bank": w["a"], "read": right, "wrong_neuron": wrong,
                           "rel_err": abs(right - w["a"]) / abs(w["a"]),
                           "rel_err_wrong": abs(wrong - w["a"]) / abs(w["a"]),
                           "token": tok.decode([w["ids"][w["offset"]]]), "transformers_ok": True}
        del model
        r = subprocess.run([sys.executable, SCRIPT, "--step", str(step), "--pole", "L03N6372+", "--text", TEXT],
                           capture_output=True, text=True)
        (out / f"stranger_step{step}.txt").write_text(r.stdout + "\n--- stderr tail\n" + r.stderr[-1500:])
        rows[str(step)]["stranger_exit"] = r.returncode
        (out / "metrics.json").write_text(json.dumps(rows, indent=1))
    k = all(v["rel_err"] <= 0.02 for v in rows.values())
    f = all(v["rel_err_wrong"] > 0.02 for v in rows.values())
    s = all(v["stranger_exit"] == 0 for v in rows.values())
    rows["verdict"] = {"K": k, "F": f, "S": s}
    (out / "metrics.json").write_text(json.dumps(rows, indent=1))
    print(rows["verdict"])
    assert k and f and s, rows["verdict"]


if __name__ == "__main__":
    main()
