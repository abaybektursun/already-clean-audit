#!/usr/bin/env python3
"""Pull one AMUNDSEN arm's banked artifacts off S3 and assemble the scorer's input.

Scoring must never depend on what happens to be in a local directory, so this
downloads from the S3 prefix of record every time and writes exactly the files
`analysis_amundsen.py <arm> --dir <dir>` reads. Missing cells are reported, not
silently skipped: an arm whose cell crashed is a DIVERGED/absent cell, and that
is data.

  python3 collect_amundsen.py am1 --dir /tmp/am1 && \
  python3 analysis_amundsen.py am1 --dir /tmp/am1 --out /tmp/am1/verdict.json
"""
import argparse
import json
import os
import subprocess

S3 = "s3://pg-lineage-769063704696/results"
AWS = ["aws", "--profile", "fuelos"]
BANKED_V1A = f"{S3}/shackleton_r_v1a_14b/cells/v1a_full/v1abl_rows.json"
BANKED_V3 = f"{S3}/shackleton_r_v3/cells/v3_full/v3_rows.jsonl"


def cp(src, dst, required=True):
    r = subprocess.run([*AWS, "s3", "cp", src, dst], capture_output=True, text=True)
    ok = r.returncode == 0
    assert ok or not required, f"missing required artifact {src}: {r.stderr.strip()}"
    print(("[got] " if ok else "[ABSENT] ") + src)
    return ok


def ls(prefix):
    r = subprocess.run([*AWS, "s3", "ls", prefix], capture_output=True, text=True)
    return [l.split()[-1].rstrip("/") for l in r.stdout.splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["am1", "am2", "am3", "am4", "am5", "am6", "am7"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--rental", default=None, help="override the rental name")
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    D = a.dir

    if a.arm == "am1":
        base = f"{S3}/amundsen_{a.rental or 'am1_abl14b'}/cells/am1_full"
        for f in ("v1abl_rows.json", "v1abl_window_nll.json", "v1abl_arm_shares.json",
                  "v1abl_act_gate.json"):
            cp(f"{base}/{f}", f"{D}/{f}")
        cp(BANKED_V1A, f"{D}/banked_v1abl_rows.json")

    elif a.arm == "am2":
        pref = f"{S3}/amundsen_{a.rental or 'am2_lrsweep'}/cells/"
        cells, missing = [], []
        for cid in ls(pref):
            if not cid.startswith("am2_h"):
                continue
            p = f"{D}/{cid}_result.json"
            if not cp(f"{pref}{cid}/M-Q_stage2_result.json", p, required=False):
                missing.append(cid)
                continue
            res = json.load(open(p))
            seen = set()
            for r in res["rows"]:
                key = (r["gauge"], r["seed"])
                if key in seen:
                    continue
                seen.add(key)
                bad = not (r["ppl"] == r["ppl"] and r["task_loss"] == r["task_loss"])
                cells.append({"cell": cid, "lr": res["lr"], "steps": res["steps"],
                              "gauge": r["gauge"], "seed": r["seed"], "ppl": r["ppl"],
                              "task_loss": r["task_loss"], "base_ppl": r["base_ppl"],
                              "diverged": bool(bad)})
        json.dump(cells, open(f"{D}/am2_cells.json", "w"), indent=2)
        json.dump({"cells_without_a_result_json": missing},
                  open(f"{D}/am2_missing.json", "w"), indent=2)
        print(f"[am2] {len(cells)} (gauge,seed) rows; {len(missing)} cells absent: {missing}")

    elif a.arm == "am3":
        base = f"{S3}/amundsen_{a.rental or 'am3_softread'}/cells/am3_full"
        cp(f"{base}/softread_rows.jsonl", f"{D}/softread_rows.jsonl")
        cp(f"{base}/softread_config.json", f"{D}/softread_config.json")

    elif a.arm == "am4":
        base = f"{S3}/amundsen_{a.rental or 'am4_llama70b'}/cells"
        names = []
        for cid in ("am4_actstats_a", "am4_actstats_b"):
            cp(f"{base}/{cid}/actstats.npz", f"{D}/{cid}.npz")
            cp(f"{base}/{cid}/actstats_flags.json", f"{D}/{cid}_flags.json")
            names.append(f"{cid}.npz")
        fl = json.load(open(f"{D}/am4_actstats_a_flags.json"))
        json.dump({"batches": names, "n_layers": fl["n_layers"],
                   "window": fl["window"], "model_dir": fl["model_dir"]},
                  open(f"{D}/am4_meta.json", "w"), indent=2)

    elif a.arm == "am5":
        rental = a.rental or "am5_cont3000"
        # the continuation rental names its cell am5b_cont, not am5_cont
        cid = "am5b_cont" if rental.startswith("am5b") else "am5_cont"
        base = f"{S3}/amundsen_{rental}/cells/{cid}"
        cp(f"{base}/v2b_rows.jsonl", f"{D}/v2b_rows.jsonl")
        cp(f"{base}/gates.json", f"{D}/gates.json")
        cp(f"{base}/v2b_config.json", f"{D}/v2b_config.json")
        cfg = json.load(open(f"{D}/v2b_config.json"))
        json.dump({"step": cfg["steps"]}, open(f"{D}/am5_meta.json", "w"), indent=2)

    elif a.arm == "am6":
        base = f"{S3}/amundsen_{a.rental or 'am6_v3seeds'}/cells"
        rows = []
        for cid in ("am6_seed3001", "am6_seed3002"):
            p = f"{D}/{cid}.jsonl"
            cp(f"{base}/{cid}/v3_rows.jsonl", p)
            rows += [json.loads(l) for l in open(p) if l.strip()]
        with open(f"{D}/v3_rows_new.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        cp(BANKED_V3, f"{D}/v3_rows_banked.jsonl")

    else:  # am7
        base = f"{S3}/amundsen_{a.rental or 'am7_evals'}/cells/am7_full"
        cp(f"{base}/am7_rows.jsonl", f"{D}/am7_rows.jsonl")
        cp(f"{base}/am7_config.json", f"{D}/am7_config.json")
    print(f"[collect] {a.arm} ready in {D}")


if __name__ == "__main__":
    main()
