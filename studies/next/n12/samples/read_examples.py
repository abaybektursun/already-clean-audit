"""Print the job's raw examples of one half for the sample questions of N12_SPEC.md section 9.5 S3: for real units
scored right (u_S <= 0.25) and wrong (u_S >= 0.75) under H_signal, the cluster's name and decoded exemplars, the top
three competitors' own top contexts with their B, the loss-matched texts and the loss match, and the carrier's
ablation against its first-order prediction.

    python studies/next/n12/samples/read_examples.py --banks studies/next/n12/banks/score-02 --half freeze
"""
import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--banks", required=True)
    ap.add_argument("--half", required=True, choices=["freeze", "test"])
    a = ap.parse_args()
    births = {b["id"]: b for b in json.loads(pathlib.Path("studies/next/m2/births.json").read_text())["births"]}
    rows = [json.loads(x) for x in (pathlib.Path(a.banks) / "n12" / "samples" /
                                    f"raw_examples_{a.half}.jsonl").read_text().splitlines()]
    for ex in rows:
        r = ex.get("real")
        if r is None:
            print(f"== {ex['birth']}: no real record")
            continue
        p = r.get("pull", {})
        tag = "RIGHT" if p.get("u_S", 0.5) <= 0.25 else ("WRONG" if p.get("u_S", 0.5) >= 0.75 else "middle")
        b = births[ex["birth"]]
        print(f"== {ex['birth']} [{tag}] name: {b['name']!r}; k {r['k']}, size {r['size']}, d {r['d']}")
        print(f"   r0 {r['cf']['r0']} r1 {r['cf']['r1']} r_c {r['cf']['r_c']} r_o {r['cf']['r_o']} F {r['cf']['F']:+.3f}; "
              f"drop {r['drop']:+.3f}; B_pre {r['B_pre']:.5f} B_birth {r['B_birth']:.5f}")
        print(f"   pull: S {p.get('S')}, P {p.get('P')}, P' {p.get('Pp')}; u_S {p.get('u_S')} u_P {p.get('u_P')} "
              f"u_P' {p.get('u_Pp')} u_gain {p.get('u_gain')}; loss match mean |dl| {p.get('dell')}")
        for e in r.get("exemplars", [])[:5]:
            print(f"   exemplar: {e[:160]!r}")
        for c in r["comp"][:3]:
            print(f"   competitor {c['pole']}: B_pre {c['B_pre']:.5f} rank {c['rank_pre']} -> {c['rank_birth']}; S {c.get('S')}")
            for t in r.get("comp_top3", {}).get(c["pole"], []):
                print(f"      own top {t['rank']} a {t['a']:.3f}: {t['text'][:140]!r}")
        for lm in r.get("lm_text", []):
            print(f"   loss match: exemplar loss {lm['ell']}: {lm['exemplar'][:100]!r}")
            for t in lm["matched"]:
                print(f"      matched: {t[:120]!r}")
        for pole, pred, meas in r.get("abl", [])[:1]:
            print(f"   ablation of the carrier {pole}: measured dL {meas:+.5f}, predicted {pred:+.5f}, "
                  f"same sign {(meas > 0) == (pred > 0)}")
        for kind in ("pb", "dt"):
            if kind in ex:
                t = ex[kind]
                print(f"   {kind}: pole {t['pole']} r0 {t['cf']['r0']} r1 {t['cf']['r1']} F {t['cf']['F']:+.3f} drop {t['drop']:+.3f} "
                      f"u_S {t.get('pull', {}).get('u_S')}")


if __name__ == "__main__":
    main()
