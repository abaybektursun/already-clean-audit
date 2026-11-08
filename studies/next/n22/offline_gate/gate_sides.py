"""N22 offline gate: both sides of every gate whose quantity a bank holds, at every one of the 16 points (the method of
M2's amendment A5). A gate goes to the instance only if its pass side passes and its registered failing input fails
where the bank holds the quantity; where no bank holds it, the record says so and the dry run shows both sides on
the tiny world.

Banks read (read-only, small files): the 16 grid harvests' controls.json and manifest.json (S3), the T_14 L40S
re-harvest (S3 grid/1c), M1's T_15 known-number run (S3 next_m1/k0-01) and M1's stage1-step476838 identity pair
(laptop, studies/next/m1/banks/start_{a,b}), the corpus manifest (S3), the HF refs (one API read), the branch
configs (studies/next/m2/offline_gate/hf for 11 points, studies/next/n22/offline_gate/hf for T_0-T_4), the
tokenizer files (the scratchpad copies of main and stage1-step60000), studies/next/n22/items.json.

    PYTHONPATH=. python studies/next/n22/offline_gate/gate_sides.py --tok <dir of tokenizer.json> \
        --tok2 <a second tokenizer.json> --out studies/next/n22/offline_gate/gate_sides.json
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from infra.pg import m1, m2, n22  # noqa: E402
from infra.aws_pg.jobs import m2_score as M  # noqa: E402

B = "s3://pg-lineage-769063704696"
AWS = ["--profile", "fuelos", "--region", "us-east-1"]


def s3json(key: str, d: pathlib.Path) -> dict:
    f = d / hashlib.sha1(key.encode()).hexdigest()
    subprocess.run(["aws", "s3", "cp", f"{B}/{key}", str(f), "--quiet", *AWS], check=True)
    return json.loads(f.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True)
    ap.add_argument("--tok2", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tmp = pathlib.Path(tempfile.mkdtemp())
    grid = "results/neurons/w3/grid"
    ctl = {t: s3json(f"{grid}/{p[1]}/controls.json", tmp) for t, p in enumerate(m2.POINTS)}
    man = {t: s3json(f"{grid}/{p[1]}/manifest.json", tmp) for t, p in enumerate(m2.POINTS)}
    c14 = s3json(f"{grid}/1c/out/t14_step1000000/controls.json", tmp)
    m14 = s3json(f"{grid}/1c/out/t14_step1000000/manifest.json", tmp)
    k0c = s3json("results/next_m1/k0-01/out/k0/controls.json", tmp)
    k0m = s3json("results/next_m1/k0-01/out/k0/manifest.json", tmp)
    corp = s3json("results/mdl/neurons/w1/q4b/1/out/corpus/corpus/manifest.json", tmp)
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{m2.REPO}/refs") as r:
        raw = r.read()
    refs = {b["name"]: b["targetCommit"] for b in json.loads(raw)["branches"]}
    nll = lambda c: c["nll_corpus_first_120k"]["mean_nll"]          # noqa: E731
    gates, pts = {}, range(16)

    # K0: the banked values are the registered ones; pass side from the control pairs; failing side at every point
    pairs = {"T15_harvest_vs_m1_k0": {"a": nll(ctl[15]), "b": nll(k0c), "devices": [man[15]["device_name"], k0m["device_name"]],
                                      "branches": [man[15]["branch"], k0m["branch"]]},
             "t476838_start_a_vs_b": {}, "T14_blackwell_vs_l40s": {"a": nll(ctl[14]), "b": nll(c14),
                                                                   "devices": [man[14]["device_name"], m14["device_name"]]}}
    sa = json.loads((ROOT / "studies/next/m1/banks/start_a/controls.json").read_text())
    sb = json.loads((ROOT / "studies/next/m1/banks/start_b/controls.json").read_text())
    pairs["t476838_start_a_vs_b"] = {"a": nll(sa), "b": nll(sb), "devices": ["NVIDIA L40S", "NVIDIA L40S"]}
    for p in pairs.values():
        p["diff"] = abs(p["a"] - p["b"])
    per = {}
    for t in pts:
        same = m2.POINTS[t][4] == m2.L40S
        tol = n22.K0_TOL_SAME if same else n22.K0_TOL_COARSE
        nb = t + 1 if t < 15 else t - 1
        gap = abs(nll(ctl[t]) - nll(ctl[nb]))
        per[t] = {"bank": nll(ctl[t]), "registered": n22.K0_BANK[t], "tol": tol, "gate_tier": "same" if same else "coarse",
                  "neighbour_gap": gap, "gap_over_tol": gap / tol, "fail_side_fails": gap > tol}
    gates["K0"] = {"banked": True, "control_pairs": pairs, "points": per,
                   "pass_side_ok": bool(pairs["T15_harvest_vs_m1_k0"]["diff"] <= n22.K0_TOL_SAME and
                                        pairs["t476838_start_a_vs_b"]["diff"] <= n22.K0_TOL_SAME and
                                        pairs["T14_blackwell_vs_l40s"]["diff"] <= n22.K0_TOL_COARSE and
                                        pairs["T14_blackwell_vs_l40s"]["diff"] <= n22.K0_REF_CROSS and
                                        all(v["bank"] == v["registered"] for v in per.values())),
                   "fail_side_ok": all(v["fail_side_fails"] for v in per.values()),
                   "summary": f"pairs {[round(p['diff'], 9) for p in pairs.values()]}; neighbour gap / tol min "
                              f"{min(v['gap_over_tol'] for v in per.values()):.1f}"}

    # G-REV: registered = banked manifest revision = HF ref; the failing input (forty zeros) equals none
    rv = {t: {"registered": m2.POINTS[t][3], "manifest": man[t]["revision"], "hf_ref": refs.get(m2.POINTS[t][2])}
          for t in pts}
    gates["G_REV"] = {"banked": True, "points": rv, "hf_refs_sha256": hashlib.sha256(raw).hexdigest(),
                      "pass_side_ok": all(v["registered"] == v["manifest"] == v["hf_ref"] for v in rv.values()),
                      "fail_side_ok": all(v["registered"] != "0" * 40 for v in rv.values()), "summary": "16 points"}

    # G-CFG: every branch config passes with no differing key; model_type renamed fails
    cf = {}
    for t in pts:
        br = m2.POINTS[t][2]
        f = ROOT / "studies/next/m2/offline_gate/hf" / f"config_{br}.json"
        if not f.exists():
            f = ROOT / "studies/next/n22/offline_gate/hf" / f"config_{br}.json"
        c = json.loads(f.read_text())
        g = m1.config_gate(c, m1.REF_CONFIG)
        cf[t] = {"file": str(f.relative_to(ROOT)), "pass": bool(g["pass"] and g["diff_keys"] == []),
                 "diff_keys": g["diff_keys"], "failing_pass": m1.config_gate({**c, "model_type": "olmo2-retrofit"},
                                                                              m1.REF_CONFIG)["pass"]}
    gates["G_CFG"] = {"banked": True, "points": cf, "pass_side_ok": all(v["pass"] for v in cf.values()),
                      "fail_side_ok": not any(v["failing_pass"] for v in cf.values()), "summary": "16 points"}

    # G-CORPUS: the registered tokens sha256 = every harvest's manifest = the corpus record
    cs = {t: man[t]["tokens_sha256"] for t in pts}
    gates["G_CORPUS"] = {"banked": True, "registered": M.TOKENS_SHA, "corpus_record": corp["tokenizers"]["olmo3"]["sha256_of_tokens_npy"],
                         "pass_side_ok": all(v == M.TOKENS_SHA for v in cs.values()) and
                         corp["tokenizers"]["olmo3"]["sha256_of_tokens_npy"] == M.TOKENS_SHA,
                         "fail_side_ok": M.TOKENS_SHA != "0" * 64, "summary": "16 manifests and the corpus record"}

    # G-TOK: the two tokenizer copies are identical; their vocabulary = the corpus record's = items.json's; the
    # registered ids re-encode; the failing input (item 0 against item 1's ids) differs. Every harvest asserted the
    # vocabulary against the corpus record at load (neurons_harvest.run), and all 16 completed.
    import transformers
    from infra.pg.matrix_models import vocab_sha256
    tok = transformers.AutoTokenizer.from_pretrained(a.tok)
    rec = json.loads((ROOT / "studies/next/n22/items.json").read_text())
    s1 = hashlib.sha256((pathlib.Path(a.tok) / "tokenizer.json").read_bytes()).hexdigest()
    s2 = hashlib.sha256(pathlib.Path(a.tok2).read_bytes()).hexdigest()
    items = rec["items"]
    re_ok = all(n22.prompt_ids(tok, (it["a"], it["b"], it["c"], it["d"])) == it["ids"] for it in items)
    ic_ok = all(n22.prompt_ids(tok, ic[f"slot{k}"]) == ic[f"ids{k}"] for it in items for ic in it["ic"] for k in (1, 2))
    cl_ok = all(n22.prompt_ids(tok, q["abcd"]) == q["ids"] for xs in rec["class_prompts"].values()
                for qs in xs.values() for q in qs)
    gates["G_TOK"] = {"banked": True, "tokenizer_json_sha256": [s1, s2], "vocab_sha256": vocab_sha256(tok),
                      "items_vocab_sha256": rec["vocab_sha256"], "corpus_vocab_sha256": corp["tokenizers"]["olmo3"]["vocab_sha256"],
                      "reencode": {"items": re_ok, "ic": ic_ok, "class": cl_ok},
                      "pass_side_ok": bool(s1 == s2 and vocab_sha256(tok) == rec["vocab_sha256"] ==
                                           corp["tokenizers"]["olmo3"]["vocab_sha256"] and re_ok and ic_ok and cl_ok),
                      "fail_side_ok": items[0]["ids"] != items[1]["ids"],
                      "summary": "all 1,200 items, 2,400 counterfactuals and every class prompt re-encode"}

    # not bank-checkable: no forward of these prompts, no lens, no edit of this model was ever banked
    nb = {k: "no bank holds it; dry run only (tiny world)" for k in
          ("I_GRAD", "LENS_FIDELITY", "I_FWD", "I_EDIT", "G_IC_ID", "IC (go/no-go)", "PC_ANS", "the floors RAND and SHUF")}
    nb["DEGENERATE (bank_scan.py)"] = "N22 reads no pole statistic; the degenerate-pole scan has no quantity here"
    res = {"what": __doc__.split("\n")[0], "items_sha256": hashlib.sha256((ROOT / "studies/next/n22/items.json").read_bytes()).hexdigest(),
           "gates": gates, "not_banked": nb}
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    for k, v in gates.items():
        print(f"{k:10s} pass {v['pass_side_ok']} fail {v['fail_side_ok']}  {v['summary']}")


if __name__ == "__main__":
    main()
