"""Local summarizer for invariant_core_triv.json → the report numbers + one-line verdict.
Run: python infra/aws_pg/jobs/_triv_summarize.py <invariant_core_triv.json>
Pure-stdlib (json/statistics); no numpy needed. Throwaway analysis aid, not a job.
"""
import json
import statistics as st
import sys

d = json.load(open(sys.argv[1]))
n = 2048
ks = [str(k) for k in d["k_grid"]]
print(f"base={d['base_repo']}@{d['base_sha'][:8]} n_ft={d['n_ft']} types={d['types']} k={d['k_grid']}")
print()

# ---- TEST A summary across layers, per type, per k ----
print("=== TEST A: overlap(U_unt, base-BOTTOM) vs overlap(U_unt, base-TOP) vs random k/2048 ===")
for t in d["types"]:
    recs = [r for r in d["test_a"] if r["type"] == t]
    print(f"\n[{t}] ({len(recs)} layers)")
    for k in ks:
        ob = [r["k"][k]["overlap_base_bottom_norm"] for r in recs]
        ot = [r["k"][k]["overlap_base_top_norm"] for r in recs]
        sr = [r["k"][k]["mean_sigma_in_unt"] / r["base_sigma_global_mean"] for r in recs]
        rand = int(k) / n
        print(f"  k={k:>3}: base-BOTTOM ov={st.mean(ob):.3f} [{min(ob):.3f},{max(ob):.3f}]  "
              f"base-TOP ov={st.mean(ot):.4f} [max {max(ot):.4f}]  rand={rand:.4f}  "
              f"| σ̄_unt/σ̄_global={st.mean(sr):.3f} [{min(sr):.3f},{max(sr):.3f}]")

# ---- TEST B: shared residual subspace? ----
print("\n=== TEST B: untouched-subspace overlaps (q-vs-k, cross-layer) vs random k/2048 ===")
for r in d["test_b"]:
    row = f"  {r['a']:>10} vs {r['b']:<10}: " + "  ".join(
        f"k={k} ov={r['k'][k]['overlap_norm']:.3f}(rand {int(k)/n:.4f})" for k in ks)
    print(row)

# ---- VERDICT number: the σ-ratio is decisive (full-spectrum, not just endpoints) ----
print("\n=== VERDICT INPUTS ===")
allrec = d["test_a"]
for k in ks:
    sr = [r["k"][k]["mean_sigma_in_unt"] / r["base_sigma_global_mean"] for r in allrec]
    ob = [r["k"][k]["overlap_base_bottom_norm"] for r in allrec]
    ot = [r["k"][k]["overlap_base_top_norm"] for r in allrec]
    rand = int(k) / n
    print(f"k={k:>3} (all {len(allrec)} matrices): σ̄_unt/σ̄_global mean={st.mean(sr):.3f}  "
          f"base-BOTTOM ov mean={st.mean(ob):.3f}  base-TOP ov mean={st.mean(ot):.4f} (rand {rand:.4f}, "
          f"TOP/rand={st.mean(ot)/rand:.1f}x)")
