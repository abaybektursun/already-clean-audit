"""BISPEC A3 — the manifold proxy Arm A's routing rule required and never got. Zero new forwards.

**Why this exists.** Arm A's frozen §6 routing sends a surviving asymmetry to `H_boundary` iff the
dose curve is kinked AND damage tracks a manifold proxy (`proxy_rho_cilo > 0`). The A3 proxy was
never implemented, so the job passed `proxy_rho_cilo = -1` and **`H_boundary` was structurally
unreachable**. The frozen rule then routed to `H_cubic` on two crude shape scores.

That is the fourth rule in this program that could not fire, and the pattern is the same one the
charter names: a branch whose enabling measurement was never taken is not a branch.

`H_cubic` is the **verdict of record** and is not swapped. What this job does is take the missing
measurement and report, as REGRAIN R16 did, whether the recorded label is supported by its own
evidence.

**The proxy the data points at.** Arm A measured, on 40k banked L0 rows:

    ⟨x, v₁⟩ :   mean −38.53,   std 14.63,   fraction positive 0.0028

So the model's operating point sits ~2.6σ on the NEGATIVE side of the hyperplane ⟨x, v₁⟩ = 0, and
the traversal `x + α·σ₁·v₁` walks it toward that hyperplane (σ₁ = 9.4337, so +3σ₁ shifts the
projection by +28.30 and the mean to −10.23 ≈ −0.70 std). The fraction of tokens that CROSS zero is
therefore ≈ 2 % at +1σ, ≈ 9 % at +2σ, ≈ 24 % at +3σ — and the measured damage is −0.017, −0.038,
**−0.496**. A one-sided boundary, approached and then crossed, predicts exactly this shape, and it
predicts the sign asymmetry for free: `−α` walks AWAY from the hyperplane and never crosses.

The proxy is the crossing fraction `f(α) = P[⟨x, v₁⟩ + α·σ₁ > 0]`, measured per α on the banked
rows. It costs nothing and it is a genuine manifold/gate statistic: it asks whether the perturbed
point has left the half-space the data occupies.

**Pre-declared, symmetric.** Spearman(f(α), |damage(α)|) over the dose grid, per direction class,
with a bootstrap CI.
  * CI excludes 0 for `pc1` AND contains 0 for the random/rotated controls ⇒ the proxy is specific,
    `H_boundary` is supported, and the recorded `H_cubic` is not supported by its own evidence.
  * CI contains 0 for `pc1` ⇒ the boundary reading fails; `H_cubic` stands unchallenged.
  * CI excludes 0 for the CONTROLS too ⇒ the proxy is not direction-specific and adjudicates nothing
    (it would just be tracking |α|). This is the failure mode that makes the proxy honest, and it is
    checked FIRST — `f(α)` is monotone in α by construction, so a spurious correlation with any
    monotone damage curve is the thing to rule out.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.bispec_a3 --type r7i.2xlarge --no-weights \
      --s3-prefix results/bispec_a3 --budget-min 30 \
      --args "--a-s3 s3://…/bispec_a/out/metrics.json --harvest-s3 s3://…/regrain_pipeA/out/harvest \
              --cdisc-s3 s3://…/regrain_c_disc/out"
"""
import argparse
import json
import pathlib

N_PCA_ROWS = 40000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--a-s3", required=True)
    ap.add_argument("--harvest-s3", required=True)
    ap.add_argument("--cdisc-s3", required=True)
    ap.add_argument("--site", default="L0")
    ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from scipy.stats import spearmanr

    from infra.aws_pg.jobs import _regrain_shared as RS
    from infra.pg import grain_bsf as GB
    from infra.pg import regrain_core as RC

    A = json.loads(RS.fetch(a.a_s3, "/tmp/bispec/a_metrics.json").read_text())
    L = int(a.site[1:])
    rep = {"object": "bispec_a3_v1", "site": a.site,
           "recorded_verdict_PRESERVED": A.get("adjudication"),
           "defect": "Arm A's §6 routing requires proxy_rho_cilo > 0 to reach H_boundary. A3 was "
                     "never implemented; the job passed -1, so H_boundary was structurally "
                     "unreachable and the rule routed to H_cubic by default. The recorded label is "
                     "NOT swapped; this job supplies the missing measurement.",
           "proxy": "crossing fraction f(alpha) = P[<x,v1> + alpha*sigma1 > 0]"}

    def _w():
        (out / "a3.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    # rebuild v1 exactly as Arm A did (regrain_f_steer construction)
    cd = json.loads(RS.fetch(f"{a.cdisc_s3}/c_disc.json", "/tmp/bispec/c_disc.json").read_text())
    bstar = [int(x) for x in cd["bstar"]]
    med_s = int(cd["median_seed"])
    nm = RS.dict_bank_name("primary", 1, med_s)
    src = f"{RS.GRAIN_DICTS_L0}/{nm}" if med_s == 0 else f"{a.cdisc_s3}/{nm}"
    ae, _c, _q, _m = GB.load_dict_bank(RS.fetch(src, f"/tmp/bispec/{nm}"))
    ae = ae.eval()
    Db = RC.decoder_matrix(ae)[:, bstar].float()

    meta = json.loads(RS.fetch(f"{a.harvest_s3}/harvest_meta.json",
                               "/tmp/bispec/harvest_meta.json").read_text())
    shard0 = meta["shards"][a.site][0]
    X = torch.load(RS.fetch(f"{a.harvest_s3}/{shard0}", f"/tmp/bispec/shards/{shard0}"),
                   map_location="cpu", weights_only=False)["X"][:N_PCA_ROWS].float()
    with torch.no_grad():
        fcode = ae.encode(X, use_threshold=True)
        contrib = fcode[:, bstar] @ Db.T
        fired = contrib.abs().sum(1) > 0
        Cc = contrib[fired].double()
        _U, S, Vh = torch.linalg.svd(Cc - Cc.mean(0), full_matrices=False)
        v1 = Vh[0]
        sigma1 = float(S[0] / max(1, Cc.shape[0] - 1) ** 0.5)
    ip = (X.double() @ v1)                                  # ⟨x, v₁⟩ over all rows
    rep["direction"] = {"sigma1": sigma1, "arm_a_sigma1": A["direction_source"]["sigma1"],
                        "inner_mean": float(ip.mean()), "inner_std": float(ip.std()),
                        "frac_positive": float((ip > 0).double().mean()),
                        "n_rows": int(X.shape[0])}
    assert abs(sigma1 - A["direction_source"]["sigma1"]) < 0.2, "v1 does not reproduce Arm A's"
    _w()

    alphas = sorted({c["alpha"] for c in A["cells"] if c["family"] == "A1_dose"})
    cross = {float(al): float(((ip + al * sigma1) > 0).double().mean()) for al in alphas}
    rep["crossing_fraction"] = cross
    _w()

    def dmg(direction):
        return {c["alpha"]: abs(c["dvqa"]) for c in A["cells"]
                if c["family"] == "A1_dose" and c["direction"] == direction}

    def corr(xs, ys):
        x, y = np.asarray(xs, float), np.asarray(ys, float)
        if len(x) < 4:
            return None
        rho = float(spearmanr(x, y).statistic)
        rng = np.random.default_rng(1337)
        b = []
        for _ in range(a.n_boot):
            i = rng.integers(0, len(x), len(x))
            if len(set(x[i])) < 3:
                continue
            v = spearmanr(x[i], y[i]).statistic
            if np.isfinite(v):
                b.append(v)
        b = np.asarray(b)
        return {"rho": rho, "n": int(len(x)),
                "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}

    dirs = sorted({c["direction"] for c in A["cells"] if c["family"] == "A1_dose"})
    res = {}
    for dn in dirs:
        dd = dmg(dn)
        als = sorted(dd)
        res[dn] = corr([cross[al] for al in als], [dd[al] for al in als])
    rep["proxy_correlation"] = res
    _w()

    pc1 = res.get("pc1")
    ctrls = [res[k] for k in res if k.startswith("rand") or k == "pc1_rotated"]
    ctrl_sig = [c for c in ctrls if c and c["ci95"][0] > 0]
    pc1_sig = bool(pc1 and pc1["ci95"][0] > 0)

    if ctrl_sig and pc1_sig:
        label = "PROXY_NOT_SPECIFIC"
        reading = ("the crossing fraction is monotone in α and correlates with damage for the "
                   "CONTROLS too, so it is tracking |α| and adjudicates nothing. Checked first, by "
                   "design.")
    elif pc1_sig and not ctrl_sig:
        label = "H_boundary_SUPPORTED"
        reading = ("damage along v₁ tracks the fraction of tokens that cross ⟨x, v₁⟩ = 0, and the "
                   "controls show no such tracking. The recorded H_cubic is the verdict of record "
                   "and is NOT supported by its own evidence: the branch the data points at was "
                   "unreachable because A3 was never taken. A confirmatory successor must route on "
                   "this proxy with sealed items.")
    else:
        label = "H_boundary_NOT_SUPPORTED"
        reading = ("damage along v₁ does not track the crossing fraction; the recorded H_cubic "
                   "stands unchallenged by this proxy.")
    rep["verdict"] = {"label": label, "reading": reading,
                      "pc1_rho": (pc1 or {}).get("rho"), "pc1_ci95": (pc1 or {}).get("ci95"),
                      "control_rhos": {k: (v or {}).get("rho") for k, v in res.items() if k != "pc1"}}
    _w()

    print("BISPEC_A3_OK", json.dumps({"verdict": label, "pc1": (pc1 or {}).get("rho"),
                                      "crossing_at_+3": cross.get(3.0),
                                      "crossing_at_-3": cross.get(-3.0)}, default=str))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
