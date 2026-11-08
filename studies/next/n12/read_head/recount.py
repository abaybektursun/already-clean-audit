"""Head's recount of the N12 verdict from the job's unit records (own code; numpy only)."""
import json, math, numpy as np
from scipy.stats import spearmanr
def wilson(k, n, z=1.959963984540054):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return p,c-h,c+h
rng=np.random.default_rng(777)
def boot(v, n=10000):
    v=np.asarray(v,float); m=len(v); bs=np.array([v[rng.integers(0,m,m)].mean() for _ in range(n)]); return v.mean(), np.quantile(bs,.025), np.quantile(bs,.975)
for half in ("test","freeze"):
    R=[json.loads(l) for l in open(f"studies/next/n12/banks/score-02/n12/n12_{half}.jsonl") if l.strip()]
    real={r["birth"]:r for r in R if r["kind"]=="real"}; pb={r["birth"]:r for r in R if r["kind"]=="pb"}; dt={r["birth"]:r for r in R if r["kind"]=="dt"}
    print(f"\n== {half}: real {len(real)} pb {len(pb)} dt {len(dt)}")
    # identity on the decomposition
    assert all(abs(r["cf"]["F"]-(math.log10(r["cf"]["r_o"])-math.log10(r["cf"]["r_c"])))<1e-9 for r in real.values())
    F=[r["cf"]["F"] for r in real.values()]; m,lo,hi=boot(F); print(f"  H_cf: mean F {m:.4f} [{lo:.4f}, {hi:.4f}] -> {'OWN_RISE' if lo>0 else 'OTHERS_FALL' if hi<0 else 'MIXED'}; F>0 in {sum(1 for x in F if x>0)}/{len(F)}")
    for name,tw in (("pb",pb),("dt",dt)):
        Ft=[tw[b]["cf"]["F"] for b in tw]; mt,lt,ht=boot(Ft); print(f"    twins {name} F: {mt:.4f} [{lt:.4f}, {ht:.4f}]")
        units=[b for b in real if b in tw]; d=[real[b]["drop"]-tw[b]["drop"] for b in units]; md,ld,hd=boot(d)
        print(f"  H_away_sel_{name}: drop real {np.mean([real[b]['drop'] for b in units]):.3f} twin {np.mean([tw[b]['drop'] for b in units]):.3f}; paired {md:+.4f} [{ld:+.4f}, {hd:+.4f}] -> {'ABOVE' if ld>0 else 'BELOW' if hd<0 else 'AT'}")
    # hits under B at pre (rank r0 <= 10)
    hr={b:real[b]["cf"]["r0"]<=10 for b in real}; hp={b:pb[b]["cf"]["r0"]<=10 for b in pb}; hd_={b:dt[b]["cf"]["r0"]<=10 for b in dt}
    print(f"  hits at pre: real {sum(hr.values())}/{len(hr)} pb {sum(hp.values())}/{len(hp)} dt {sum(hd_.values())}/{len(hd_)}")
    u=[b for b in real if b in dt]; d=[float(hr[b])-float(hd_[b]) for b in u]; m,lo,hi=boot(d); print(f"  H_sel_dt real-dt: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] -> {'ABOVE' if lo>0 else 'BELOW' if hi<0 else 'AT'}")
    u2=[b for b in dt if b in pb]; d=[float(hd_[b])-float(hp[b]) for b in u2]; m,lo,hi=boot(d); print(f"  H_dt_pb dt-pb: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] -> {'HIGHER' if lo>0 else 'LOWER' if hi<0 else 'SAME'}")
    # G-VAL from abl records
    pairs=[(a["dL_hat"] if "dL_hat" in a else a.get("pred"), a["dL"] if "dL" in a else a.get("meas")) for r in real.values() for a in r["abl"]]
    if pairs and pairs[0][0] is not None:
        x=np.array([p[0] for p in pairs]); y=np.array([p[1] for p in pairs]); rho=spearmanr(x,y).correlation
        ratio=np.abs(y)/np.maximum(np.abs(x),1e-12); print(f"  G-VAL: pairs {len(pairs)}, Spearman {rho:.4f}; sign agreement {np.mean(np.sign(x)==np.sign(y)):.3f}; median |dL|/|dL_hat| {np.median(ratio):.1f}, p90 {np.quantile(ratio,.9):.1f}")
    else:
        print("  abl record keys:", list(real[next(iter(real))]["abl"][0].keys()))
    uS=[r["pull"]["u_S"] for r in real.values()]; m,lo,hi=boot(uS); print(f"  exploratory u_S: {m:.3f} [{lo:.3f}, {hi:.3f}]; u_gain mean {np.mean([r['pull']['u_gain'] for r in real.values()]):.3f}; H_bridge mean rho {np.mean([r['pull']['rho_S_gain'] for r in real.values() if r['pull']['rho_S_gain']==r['pull']['rho_S_gain']]):+.4f}")
