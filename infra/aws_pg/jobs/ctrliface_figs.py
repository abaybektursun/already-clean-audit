"""CTRLIFACE figure suite (spec §10) — F1–F5, each backed by a number+CI+null (the frozen
`metrics.json`). Runs weight-free from the JSON (no model), so it lives on a CPU box (r7i) and
reads the confirmation metrics off S3. Conventions: CI bands / error bars, the NULL overlaid, a
plain-language finding in the title, a provenance footer (run·sha·input-hash·tag), colourblind-safe
palette, PNG (review) + PDF (paper), bbox tight. Defensive: a panel whose data is absent is skipped
(logged), never faked.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.ctrliface_figs --type r7i.2xlarge --no-weights \
      --s3-prefix results/ctrliface_figs --args "--metrics-s3 s3://.../ctrliface_confirm/out/metrics.json --sha <freeze>"

The frozen confirmation metrics.json schema this reads:
  floor, n_prefix, d, k_star, L_star, thresholds{tau_low,tau_high,delta}
  arm_a.rf_by_k{ "<k>": {method:{rf,ci:[lo,hi],success}} }          # sweep at L_star
  arm_a.rf_by_L{ "<L>": {"saction":{rf,ci}} }                       # depth at k_star
  arm_a.gap_at_kstar{gap,ci:[lo,hi],ci_lo}
  arm_b.overlap{value,ci}, arm_b.overlap_null{mean,p95}, arm_b.cos_spectrum[]
  arm_b.rf_control_stext{rf,ci}, arm_b.twoby2{saction_control,saction_text,stext_control,stext_text}
  proxy.rho, controls{...}, verdict{axis1,axis2}
"""
import argparse
import hashlib
import json
import pathlib

# colourblind-safe (Wong): blue, orange, green, vermillion, purple, grey
C = {"saction": "#0072B2", "random_energy": "#D55E00", "random_k": "#E69F00",
     "identity": "#009E73", "shuffle": "#999999", "floor": "#000000", "stext": "#CC79A7"}


def _foot(ax, run, sha, ih, tag):
    ax.figure.text(0.005, 0.005, f"run {run} · sha {sha} · in {ih} · {tag}", fontsize=6, color="#666")


def _save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")


def _band(ax, xs, pts, los, his, color, label):
    ax.plot(xs, pts, "-o", color=color, label=label, ms=4)
    ax.fill_between(xs, los, his, color=color, alpha=0.18)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import subprocess
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics", default=None, help="local metrics.json (else merge --confirm-s3 + --confirm-b-s3)")
    ap.add_argument("--confirm-s3", default=None)
    ap.add_argument("--confirm-b-s3", default=None)
    ap.add_argument("--run", default="ctrliface_confirm")
    ap.add_argument("--sha", default="unfrozen")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.metrics:
        m = json.loads(pathlib.Path(a.metrics).read_text())
    else:
        subprocess.run(["aws", "s3", "cp", a.confirm_s3, "/tmp/cf.json", "--region", "us-east-1"], check=True)
        m = json.loads(pathlib.Path("/tmp/cf.json").read_text())      # arm_a (F1,F2) + floor/thresholds
        if a.confirm_b_s3:
            subprocess.run(["aws", "s3", "cp", a.confirm_b_s3, "/tmp/cfb.json", "--region", "us-east-1"], check=True)
            m["arm_b"] = json.loads(pathlib.Path("/tmp/cfb.json").read_text()).get("arm_b", {})  # F3,F4
        pathlib.Path("/tmp/merged.json").write_text(json.dumps(m))
        a.metrics = "/tmp/merged.json"
    ih = hashlib.sha256(pathlib.Path(a.metrics).read_bytes()).hexdigest()[:10]
    floor = m.get("floor", 0.0)
    made = []

    # F1 — RF_control(k): S_action vs energy-matched random vs identity (headline).
    ra = m.get("arm_a", {})
    if ra.get("rf_by_k"):
        ks = sorted((int(k) for k in ra["rf_by_k"]))
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for method in ("saction", "random_energy", "random_k", "identity", "shuffle"):
            xs, pts, los, his = [], [], [], []
            for k in ks:
                cell = ra["rf_by_k"][str(k)].get(method)
                if not cell:
                    continue
                xs.append(k); pts.append(cell["rf"])
                lo, hi = cell.get("ci", [cell["rf"], cell["rf"]]); los.append(lo); his.append(hi)
            if xs:
                _band(ax, xs, pts, los, his, C.get(method, "#444"), method)
        ax.axhline(floor if isinstance(floor, (int, float)) else 0.0, ls=":", color=C["floor"], lw=1, label="floor")
        ax.set_xscale("log", base=2); ax.set_xlabel("k (ablated subspace rank)")
        ax.set_ylabel("RF_control  (1=harmless, 0=floored)")
        v = m.get("verdict", {}).get("axis1", "?")
        ax.set_title(f"F1 · Does ablating S_action crater control beyond an energy-matched null?  →  {v}")
        ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.25)
        _foot(ax, a.run, a.sha, ih, "CONFIRMATORY")
        _save(fig, out, "F1_rf_by_k"); plt.close(fig); made.append("F1")

    # F2 — depth profile RF_control(L) at k*.
    if ra.get("rf_by_L"):
        Ls = sorted((int(L) for L in ra["rf_by_L"]))
        xs, pts, los, his = [], [], [], []
        for L in Ls:
            cell = ra["rf_by_L"][str(L)].get("saction")
            if not cell:
                continue
            xs.append(L); pts.append(cell["rf"])
            lo, hi = cell.get("ci", [cell["rf"], cell["rf"]]); los.append(lo); his.append(hi)
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        _band(ax, xs, pts, los, his, C["saction"], "S_action")
        ax.axhline(floor if isinstance(floor, (int, float)) else 0.0, ls=":", color=C["floor"], lw=1, label="floor")
        ax.set_xlabel("backbone layer L"); ax.set_ylabel(f"RF_control at k*={m.get('k_star','?')}")
        ax.set_title("F2 · Depth signature of the control interface (localized vs uniform)")
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
        _foot(ax, a.run, a.sha, ih, "CONFIRMATORY")
        _save(fig, out, "F2_depth_profile"); plt.close(fig); made.append("F2")

    # F3 — S_action↔S_text overlap vs matched-random null.
    rb = m.get("arm_b", {})
    if rb.get("overlap"):
        ov = rb["overlap"]; nul = rb.get("overlap_null", {})
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        ax.bar([0], [ov["value"]], width=0.5, color=C["saction"], label="o(S_action,S_text)")
        lo, hi = ov.get("ci", [ov["value"], ov["value"]])
        ax.errorbar([0], [ov["value"]], yerr=[[ov["value"] - lo], [hi - ov["value"]]], color="k", capsize=4)
        if nul:
            ax.axhline(nul.get("p95", nul.get("mean", 0)), ls="--", color=C["random_k"], label="matched-random null p95")
        ax.set_xticks([]); ax.set_ylabel("principal-angle overlap  (0=orthogonal,1=nested)")
        v = m.get("verdict", {}).get("axis2", "?")
        ax.set_title(f"F3 · Is S_action geometrically disjoint from S_text?  →  {v}")
        ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
        _foot(ax, a.run, a.sha, ih, "CONFIRMATORY")
        _save(fig, out, "F3_overlap"); plt.close(fig); made.append("F3")

    # F4 — Arm-B 2×2 (ablate {S_action,S_text} × measure {control,text}).
    tb = rb.get("twoby2")
    if tb:
        import numpy as np
        def _f(x):
            return float(x) if isinstance(x, (int, float)) else float("nan")   # "N/A" text cells → nan
        M = np.array([[_f(tb.get("saction_control")), _f(tb.get("saction_text"))],
                      [_f(tb.get("stext_control")), _f(tb.get("stext_text"))]], float)
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        im = ax.imshow(M, cmap="cividis", vmin=0, vmax=1)
        ax.set_xticks([0, 1], ["control", "text"]); ax.set_yticks([0, 1], ["ablate S_action", "ablate S_text"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color="w" if M[i, j] < 0.5 else "k")
        ax.set_title("F4 · RF under crossed ablation (disjointness mechanism)")
        fig.colorbar(im, ax=ax, label="RF"); _foot(ax, a.run, a.sha, ih, "CONFIRMATORY")
        _save(fig, out, "F4_2x2"); plt.close(fig); made.append("F4")

    # F5 — proxy-vs-success validity scatter (ρ).
    pv = m.get("proxy", {})
    if pv.get("scatter"):
        xs = [p[0] for p in pv["scatter"]]; ys = [p[1] for p in pv["scatter"]]
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        ax.scatter(xs, ys, color=C["saction"], s=24)
        ax.set_xlabel("proxy: action-deviation D"); ax.set_ylabel("rollout success-drop")
        ax.set_title(f"F5 · Proxy validity  (ρ={pv.get('rho','?')})  [gate ρ≥0.6]")
        ax.grid(alpha=0.25); _foot(ax, a.run, a.sha, ih, "DISCOVERY")
        _save(fig, out, "F5_proxy_validity"); plt.close(fig); made.append("F5")

    (out / "figs_manifest.json").write_text(json.dumps(
        {"made": made, "metrics_hash": ih, "sha": a.sha}, indent=2))
    print("FIGS_OK", json.dumps({"made": made, "metrics_hash": ih}))


if __name__ == "__main__":
    main()
