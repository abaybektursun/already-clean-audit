"""N2V: the ten raw examples at zero GPU (studies/next/07_PICKS/N2V_SPEC.md section 10).

The ten births are N12's freeze-half example births (M2's frozen rule, before N2V existed). For each, the four pairs
N12 ablated (the carrier and its top 3 competitors under B): N12's banked bf16 prediction -h G and its banked bf16
measured change, their ratio, the activation removed (sum |h| at the unit's positions, from N12's banked pull arrays
on S3, read once into the scratchpad), the loss terms the change can reach, and the sizes N2V's rows will have to
resolve (the 10 % finite difference is a tenth of the prediction). The fp32 values are PENDING: the job writes them
(n2v/samples/raw_examples_<half>.jsonl). One decoded exemplar and the first matched random pole's top context
(12_SHAPING_NOTE.md section 4 item 3) come from N12's banked raw examples.

    PYTHONPATH=. python studies/next/n2v/samples/raw_examples.py --cache <scratch dir> --out studies/next/n2v/samples
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import numpy as np

from infra.pg import m2, n2v

ROOT = pathlib.Path(__file__).resolve().parents[4]
S3 = "s3://pg-lineage-769063704696/results/next_n12/score-02/out/score/n12/arrays"
NOT_PRINTED = tuple(w[::-1] for w in ("diov", "regdel", "llun"))   # the three words this program does not print, stored reversed


def pick(texts: list) -> tuple:
    """The first text holding none of NOT_PRINTED (case-insensitive), with its index; the rule is stated in the output."""
    for i, x in enumerate(texts):
        if not any(w in x.lower() for w in NOT_PRINTED):
            return i, x
    return None, "(every text holds a word this program does not print)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cache = pathlib.Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    uj = json.loads((ROOT / "studies/next/n12/units.json").read_text())
    U = {u["unit"]: u for u in uj["units"]}
    recs = {r["unit"]: r for r in m2.read_jsonl(ROOT / "studies/next/n12/banks/score-02/n12/n12_freeze.jsonl")}
    exr = {e["birth"]: e for e in m2.read_jsonl(ROOT / "studies/next/n12/banks/score-02/n12/samples/raw_examples_freeze.jsonl")}
    out = []
    for b in uj["examples"]["freeze"]:
        u, r = U[b], recs[b]
        t = u["roles"]["pre"]
        f = cache / f"{b.replace(':', '_')}_pull.npz"
        if not f.exists():
            subprocess.run(["aws", "s3", "cp", f"{S3}/t{t:02d}/{b.replace(':', '_')}_pull.npz", str(f), "--profile", "fuelos",
                            "--region", "us-east-1", "--only-show-errors"], check=True)
        z = np.load(f)
        H, valid = z["H"], z["valid"]
        uw = n2v.unit_windows(u)
        pairs = []
        for i, (pole, pred, meas) in enumerate(r["abl"]):
            n = m2.parse_pole(pole)[1]
            pairs.append({"rank": i, "pole": pole, "pred_bf16": pred, "meas_bf16": meas,
                          "ratio_meas_to_pred": meas / pred if pred else None,
                          "sum_abs_h": float(np.abs(H[valid, n]).sum()), "fd_pred_bf16": (1 - n2v.FD_SCALE) * pred,
                          "PENDING": ["dL fp32", "dL_own", "dL_fd", "dL_fd_own", "pred_jvp", "pred_jvp_own", "probe"]})
        e = exr[b]["real"]
        mt = e.get("matched_top3") or {}
        out.append({"birth": b, "t_pre": t, "layer": u["layer"], "size": len(u["tok"]), "d": u["d"],
                    "windows": len(uw), "loss_terms_reachable": int(sum(m2.WINDOW - min(ix) for _, ix in uw)),
                    "exemplar_index_and_text": pick(e["exemplars"]), "pairs": pairs,
                    "matched_pole_context": {k: pick([c["text"] for c in v]) for k, v in mt.items()} if mt else None})
    (pathlib.Path(a.out) / "raw_examples_zero_gpu.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    lines = ["# N2V raw examples at zero GPU (freeze half; N12's ten example births)", "",
             "Banked bf16 values from N12 score-02 (`n12_freeze.jsonl` -> abl; `arrays/tNN/<unit>_pull.npz` -> H). The fp32 "
             "columns are PENDING until the job runs. pred = -h G summed over the unit's positions; meas = N12's bf16 window "
             "loss change; fd pred = 0.1 pred.", "",
             "| birth | t_pre | windows | loss terms | pole (rank) | sum abs h | pred bf16 | meas bf16 | meas / pred | fd pred |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for x in out:
        for p in x["pairs"]:
            lines.append(f"| {x['birth']} | {x['t_pre']} | {x['windows']} | {x['loss_terms_reachable']} | {p['pole']} "
                         f"({p['rank']}) | {p['sum_abs_h']:.3g} | {p['pred_bf16']:.4g} | {p['meas_bf16']:.4g} | "
                         f"{p['ratio_meas_to_pred']:.3g} | {p['fd_pred_bf16']:.3g} |")
    pr = np.array([p["pred_bf16"] for x in out for p in x["pairs"]])
    me = np.array([p["meas_bf16"] for x in out for p in x["pairs"]])
    lines += ["", f"Over these 40 pairs: median |pred| {np.median(np.abs(pr)):.4g} nats, median |meas| "
                  f"{np.median(np.abs(me)):.4g} nats, sign agreement {np.mean(np.sign(pr) == np.sign(me)):.2f}, "
                  f"median |0.1 pred| {np.median(np.abs(0.1 * pr)):.3g} nats.", ""]
    lines.append("Texts: the first exemplar and the first of the matched pole's top-3 contexts that holds none of the "
                 "words this program does not print (the index is given).")
    for x in out:
        i, ex = x["exemplar_index_and_text"]
        mc = list(x["matched_pole_context"].items())[0] if x["matched_pole_context"] else ("none", (None, ""))
        lines.append(f"- {x['birth']}: exemplar {i}: `{ex[-160:]}`; matched pole {mc[0]}, context {mc[1][0]}: "
                     f"`{mc[1][1][-120:]}`")
    (pathlib.Path(a.out) / "raw_examples_zero_gpu.md").write_text("\n".join(lines).replace("\n\n\n", "\n\n") + "\n")
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
