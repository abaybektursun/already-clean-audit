"""usage: PYTHONPATH=<worktree> <n13 venv python> studies/next/n13/s4/tools/manifests_02.py
The S4 B relaunch manifests after A31 (the head's brief, 2026-11-07): n13_plan.manifest(kind, 2, ...) with the assumptions of the
banked projection (studies/next/n13/r0/s2smoke-09/projection.json); s4armsB2-02 without rndT_k1 (banked from s4armsB2-01). Allowance
round(1.3 x projection) (A27). Checks first that manifest(kind, 1, ...) reproduces the committed -01 manifests."""
import json, pathlib
from infra.pg import n13_plan as P
R = pathlib.Path("/Users/abaybektursun/projects/interp_research-n13")
pj = json.loads((R / "studies/next/n13/r0/s2smoke-09/projection.json").read_text())
B = pj["assumptions"]
for k in ("s4armsB1", "s4armsB2"):
    old = json.loads((R / f"infra/aws_pg/steps/n13_{k}-01.json").read_text())
    assert old == P.manifest(k, 1, B, allowance_min=pj["launches"][k]["allowance_min"], tok_s=pj["tok_s"], mean_math=B["mean_math"], cap=16384), k
out = {}
for k, drop in (("s4armsB1", ()), ("s4armsB2", ("rndT_k1",))):
    arms = [x for x in P.S4_SPLIT[k] if x[0] not in drop]
    mn = B["setup_min"] + sum(P.gen_minutes(P.N_MATH, P.N_GSM if g else 0, pj["tok_s"], B["mean_math"], B["mean_gsm"], P.arm_load(a, B)) for a, _, g in arms)
    al = P.budget_min(mn)
    steps = [s for s in P.manifest(k, 2, B, allowance_min=al, tok_s=pj["tok_s"], mean_math=B["mean_math"], cap=16384) if s["name"] not in drop]
    c = P.check_manifest(steps)
    assert c["distinct"] and [s["name"] for s in steps[1:]] == [a for a, _, _ in arms]
    f = R / f"infra/aws_pg/steps/n13_{k}-02.json"
    f.write_text(json.dumps(steps, indent=1) + "\n")
    out[k] = {"file": str(f.relative_to(R)), "steps": c["names"], "projection_min": round(mn, 1), "gpu_h": round(mn / 60, 2), "allowance_min": al}
print(json.dumps(out, indent=1))
