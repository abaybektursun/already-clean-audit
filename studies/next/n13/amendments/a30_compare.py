"""A30: R1's outputs on the offline gate's stand-in, before A30 (the A29 dry run) and after it. sets.json must be equal byte for byte;
every arm hash must be equal except the rndE arms', which must differ; rndE's realized energy before and after, and rndT's ratio.
Run: python3 studies/next/n13/amendments/a30_compare.py <before dry-run out> <after dry-run out> <out.json>. Fail-loud."""
import hashlib, json, pathlib, sys
B, A = (pathlib.Path(x) / "e2e/out/r1_rank" for x in sys.argv[1:3])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
hb, ha = (json.loads((d / "arm_hashes.json").read_text()) for d in (B, A))
assert sorted(hb) == sorted(ha), (sorted(hb), sorted(ha))
same = {a: hb[a]["sha256"] == ha[a]["sha256"] for a in sorted(hb)}
sets_b, sets_a = json.loads((B / "sets.json").read_text()), json.loads((A / "sets.json").read_text())
out = {"sets_json_equal": (B / "sets.json").read_bytes() == (A / "sets.json").read_bytes(), "sets_sha256": [sha(B / "sets.json"), sha(A / "sets.json")],
       "arm_hash_equal": same, "changed": sorted(a for a, v in same.items() if not v),
       "rndE": {a: {"before_ratio": hb[a]["families"]["mlp"]["realized_sq"] / (sets_b["rndE"][a[5:]]["c"] ** 2 * sets_b["rndE"][a[5:]]["E_R"]),
                    "after_ratio": ha[a]["families"]["mlp"]["realized_sq"] / (sets_a["rndE"][a[5:]]["c"] ** 2 * sets_a["rndE"][a[5:]]["E_R"]),
                    "energy_passes": ha[a].get("energy_passes"), "c_built": ha[a].get("c_built")} for a in sorted(hb) if a.startswith("rndE_")},
       "rndT": {a: ha[a]["families"]["mlp"]["realized_sq"] / (sets_a["rndT"][a[5:]] if a != "rndT_k1_d2" else sets_a["rndT_d2"])["E_S"]
                for a in sorted(ha) if a.startswith("rndT_")}}
out["pass"] = out["sets_json_equal"] and out["changed"] == sorted(a for a in hb if a.startswith("rndE_")) \
    and all(abs(v["after_ratio"] - 1) <= 0.05 for v in out["rndE"].values())
pathlib.Path(sys.argv[3]).write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: out[k] for k in ("sets_json_equal", "changed", "pass")}), json.dumps({a: [round(v["before_ratio"], 4), round(v["after_ratio"], 4),
      len(v["energy_passes"])] for a, v in out["rndE"].items()}), json.dumps({a: round(v, 4) for a, v in out["rndT"].items()}))
assert out["pass"]
