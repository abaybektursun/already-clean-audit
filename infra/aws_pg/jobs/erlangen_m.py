"""ERLANGEN Arm M (+ the Arm B rider) — every cost cell INTO one destination task.

One instance per `dst`: the shell load, the item build and the criterion anchor (the dedicated
ft-dst checkpoint's own NLL on the very same held-out items) are computed ONCE and amortised over
all (src × seed) cells, which is what makes the 5×5×2 grid affordable. `rep["cells"]` is bound
BEFORE the loop, so a crash in cell k still banks cells 0..k−1
(`feedback_fleet_incremental_save`).

Estimand (spec §2-M, amendment A3):

    s_t  = (nll_0 − nll_t) / (nll_0 − nll_ref)
    d    = Σ_t Δexamples · (1 − clamp(s_t, 0, 1))   the area still to travel — LOWER is cheaper
    criterion : s_t ≥ 0.5 (half-way to what the dedicated fine-tune achieves)
    censored  : never reached inside the budget → d is a LOWER BOUND (A3c)

Cost is read with the SAME teacher-forced NLL instrument as the rest of the program, so no
per-task metric plumbing enters the comparison, and the budget ladder is read off ONE curve.

Arm B rides every run (spec §2-B): per-step gradient energy inside the consensus untouched core vs
a matched-random subspace of equal dimension, the end-of-run displacement ratio, and the curvature
quadratic form in both directions at 3 checkpoints. The random subspace is drawn with ONE fixed
seed across all cells, so cells are comparable.

    python -m infra.aws_pg.jobs.erlangen_m --out DIR --dst refcoco \
        --srcs cococap,refcoco,textvqa,vqav2,okvqa --seeds 0,1 \
        [--basis-s3 s3://…/erlangen_core_basis.npz]

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import time

QK_SUFFIX = ("self_attn.q_proj", "self_attn.k_proj")


def detach_adapters(lm, names):
    """Put every wrapped Linear back, so the next cell installs into clean parameter paths.
    Without this the shell's state-dict keys carry a `.base` segment and the next install_canon
    would silently fail to resolve."""
    for n in names:
        parent_name, _, leaf = n.rpartition(".")
        parent = lm.get_submodule(parent_name) if parent_name else lm
        mod = getattr(parent, leaf)
        assert hasattr(mod, "base"), f"{n} is not an adapter — detach called twice?"
        setattr(parent, leaf, mod.base)
    for p in lm.parameters():
        p.requires_grad_(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--srcs", required=True)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--budget", type=int, default=4000, help="training EXAMPLES (the spec ladder)")
    ap.add_argument("--ladder", default="1000,4000")
    ap.add_argument("--n-eval", type=int, default=128)
    ap.add_argument("--n-train", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--rank", type=int, default=512)
    ap.add_argument("--alpha", type=float, default=-1.0, help="-1 → α = rank (scale 1.0)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--basis-s3", default="")
    ap.add_argument("--n-grad-probe", type=int, default=6)
    ap.add_argument("--skip-curvature", action="store_true")
    ap.add_argument("--n-null-draws", type=int, default=1,
                    help="critic finding 12: independent matched-random Q draws per matrix. >1 "
                         "makes the CI unconditional on a single null realisation.")
    a = ap.parse_args()
    if a.alpha < 0:
        a.alpha = float(a.rank)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import numpy as np
    import torch
    from transformers import get_scheduler

    from infra.pg import erlangen_probe as P
    from infra.pg import erlangen_tasks as ET
    from infra.pg import relearn as R
    from infra.pg.lacuna_shell import canon, load_pg1, pg1_baseline_sd

    assert torch.cuda.is_available(), "Arm M needs a GPU"
    device = "cuda"
    t0 = time.time()
    ladder = [int(x) for x in a.ladder.split(",")]
    srcs = [s.strip() for s in a.srcs.split(",") if s.strip()]
    seeds = [int(s) for s in a.seeds.split(",")]
    rep = {"object": "erlangen_m_fleet_v1", "dst": a.dst, "srcs": srcs, "seeds": seeds,
           "recipe": {"rank": a.rank, "alpha": a.alpha, "lr": a.lr, "batch": a.batch,
                      "warmup": a.warmup, "budget_examples": a.budget, "ladder": ladder,
                      "n_train_pool": a.n_train, "n_eval": a.n_eval,
                      "scheduler": "cosine", "optimizer": "adamw",
                      "scope": "ONE frozen recipe; the estimand is the axiom PATTERN, not levels"},
           "ft_repos": {t: ET.FT_REPOS[t] for t in sorted(set(srcs + [a.dst]))}}
    cells = {}
    rep["cells"] = cells                     # bound BEFORE the loop

    def bank():
        (out / f"erlangen_m_into_{a.dst}.json").write_text(json.dumps(rep, indent=2))

    bank()
    model, proc = load_pg1(device=device)
    pg1_sd = pg1_baseline_sd(model)
    lm = model.get_submodule("model.language_model")

    ev = ET.items(a.dst, "eval", a.n_eval)
    tr = ET.items(a.dst, "train", a.n_train, n_eval_skip=a.n_eval)
    print(f"[M] dst={a.dst} items: eval={len(ev)} train={len(tr)} ({time.time()-t0:.0f}s)",
          flush=True)

    dst_sd = P.donor_repo(ET.FT_REPOS[a.dst])
    P.install_unit(model, "ft_dst", pg1_sd, sd=dst_sd)
    nll_ref = float(np.mean(P.score_pack(model, proc, device, ev, batch_size=a.batch)))
    del dst_sd
    rep["nll_ref"] = nll_ref
    print(f"[M] criterion anchor nll_ref={nll_ref:.4f} ({time.time()-t0:.0f}s)", flush=True)
    bank()

    # ---- Arm B basis (loaded once; the random twin uses ONE seed across cells) ----------------
    basis_np, rand_np = {}, {}
    if a.basis_s3:
        local = out / "erlangen_core_basis.npz"
        subprocess.run(["aws", "s3", "cp", a.basis_s3, str(local), "--region", "us-east-1"],
                       check=True)
        z = np.load(local)
        rng = np.random.default_rng(20260803)
        for k in z.files:
            U = torch.from_numpy(z[k]).float()
            basis_np[k] = U.to(device)
            # CRITIC FINDING 12: with n_null_draws=1 the CI is conditional on ONE null realisation
            # shared by every run. Drawing several per matrix makes the comparator a distribution.
            draws = []
            for _ in range(max(a.n_null_draws, 1)):
                A_ = torch.from_numpy(rng.standard_normal(tuple(U.shape))).float()
                Q, _ = torch.linalg.qr(A_)
                draws.append(Q.to(device))
            rand_np[k] = draws
        rep["arm_b_basis"] = {"n_matrices": len(basis_np),
                              "k": int(next(iter(basis_np.values())).shape[1]),
                              "n_null_draws": max(a.n_null_draws, 1),
                              "source": a.basis_s3}
        print(f"[M] Arm B basis: {len(basis_np)} q/k matrices", flush=True)
    else:
        rep["arm_b_basis"] = {"n_matrices": 0, "note": "no basis — B rows pending"}
    bank()

    def collate(rows):
        b = proc(text=[r["prompt"] for r in rows], images=[r["image"] for r in rows],
                 suffix=[r["target"] for r in rows], return_tensors="pt", padding="longest")
        b["pixel_values"] = b["pixel_values"].to(torch.bfloat16)
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}

    for src in srcs:
        src_sd = P.donor_repo(ET.FT_REPOS[src])
        for seed in seeds:
            key = f"{src}|{seed}"
            tcell = time.time()
            P.install_unit(model, "ft_src", pg1_sd, sd=src_sd)
            model.eval()
            nll_0 = float(np.mean(P.score_pack(model, proc, device, ev, batch_size=a.batch)))
            cell = {"src": src, "dst": a.dst, "seed": seed, "nll_0": nll_0, "nll_ref": nll_ref,
                    "headroom": nll_0 - nll_ref,
                    # RANGE GUARD both ways: no headroom makes the normalised curve undefined; a
                    # NEGATIVE headroom means the source already beats the dedicated fine-tune,
                    # which is data, not an error. Both are flagged, never silently divided.
                    "degenerate_headroom": bool(abs(nll_0 - nll_ref) < 1e-3),
                    "curve": [{"examples": 0, "nll": nll_0, "s": 0.0}],
                    "grad_probes": [], "curvature": [], "arm_b": {}}
            cells[key] = cell
            bank()

            info = R.attach_adapters(lm, rank=a.rank, seed=1337 + seed, alpha=a.alpha)
            # attach_adapters builds fresh nn.Parameters on the CPU; the shell is already on the
            # GPU, so the A/B matmuls would hit a device mismatch on the first forward. Move them
            # BEFORE the optimizer is constructed, or its state would reference stale CPU tensors.
            lm.to(device)
            names = info["names"]
            cell["adapters"] = {"n_modules": info["n_modules"],
                                "trainable_params": info["trainable_params"]}
            qk_mods = {n: m for n, m in lm.named_modules()
                       if any(n.endswith(s) for s in QK_SUFFIX) and hasattr(m, "base")}
            bmap = {n: canon(f"language_model.{n}.weight").replace(".", "__") for n in qk_mods}
            live_b = {n: k for n, k in bmap.items() if k in basis_np}

            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                    lr=a.lr, weight_decay=0.0)
            n_steps = a.budget // a.batch
            sched = get_scheduler("cosine", optimizer=opt,
                                  num_warmup_steps=int(a.warmup * n_steps),
                                  num_training_steps=n_steps)

            def grad_probe(step):
                if not live_b:
                    return None
                for n in live_b:
                    qk_mods[n].base.weight.requires_grad_(True)
                model.zero_grad(set_to_none=True)
                rows = [tr[(step * a.batch + j) % len(tr)] for j in range(a.batch)]
                loss = model(**collate(rows)).loss
                loss.backward()
                ec, er, per = 0.0, 0.0, {}
                for n, k in live_b.items():
                    g = qk_mods[n].base.weight.grad
                    assert g is not None, f"no gradient reached the base weight of {n}"
                    gf = g.float()
                    c = float((gf @ basis_np[k]).pow(2).sum())
                    # average over the null draws so a single unlucky Q cannot set the comparator
                    r_ = float(np.mean([float((gf @ Q).pow(2).sum()) for Q in rand_np[k]]))
                    ec += c
                    er += r_
                    # CRITIC FINDING 4: bank PER MATRIX so q_proj (a real low-eigenvalue band) can
                    # be separated from k_proj (rank-deficient under MQA — its "bottom-k" is an
                    # arbitrary selection from an exact kernel). Without this the pooled ratio is
                    # unattributable and the mechanism is unidentifiable.
                    per[n] = {"core": c, "rand": r_, "ratio": c / max(r_, 1e-30)}
                    qk_mods[n].base.weight.grad = None
                    qk_mods[n].base.weight.requires_grad_(False)
                model.zero_grad(set_to_none=True)
                return {"step": step, "loss": float(loss.item()), "energy_core": ec,
                        "energy_rand": er, "ratio": ec / max(er, 1e-30), "per_matrix": per}

            def curvature_probe(step):
                if not live_b or a.skip_curvature:
                    return None
                for n in live_b:
                    qk_mods[n].base.weight.requires_grad_(True)
                model.zero_grad(set_to_none=True)
                loss = model(**collate(tr[:1])).loss
                ws = [qk_mods[n].base.weight for n in live_b]
                gs = torch.autograd.grad(loss, ws, create_graph=True)
                res = {}
                for tag, B in (("core", basis_np), ("rand", rand_np)):
                    # rand_np holds a LIST of draws (finding 12); curvature needs ONE fixed
                    # direction, so it uses draw 0 — the energy statistics average over all draws.
                    def _b(k):
                        v = B[k]
                        return v[0] if isinstance(v, list) else v

                    dirs = [(g.float() @ _b(k) @ _b(k).T).detach()
                            for g, k in zip(gs, live_b.values())]
                    nrm = sum(float(d.pow(2).sum()) for d in dirs) ** 0.5
                    if nrm <= 0:
                        res[tag] = float("nan")
                        continue
                    dirs = [d / nrm for d in dirs]
                    gd = sum((g.float() * d).sum() for g, d in zip(gs, dirs))
                    hv = torch.autograd.grad(gd, ws, retain_graph=True)
                    res[tag] = float(sum((h.float() * d).sum() for h, d in zip(hv, dirs)))
                for n in live_b:
                    qk_mods[n].base.weight.grad = None
                    qk_mods[n].base.weight.requires_grad_(False)
                model.zero_grad(set_to_none=True)
                # CRITIC FINDING 11: a non-positive random denominator makes the ratio undefined
                # (one control cell was a SIGN FLIP, not merely small). Flagged, never silently
                # divided.
                return {"step": step, "vHv_core": res["core"], "vHv_rand": res["rand"],
                        "rand_denominator_nonpositive": bool(not (res["rand"] > 0))}

            eval_at = sorted({*ladder, a.budget})
            probe_at = sorted({int(round(i * n_steps / max(a.n_grad_probe, 1)))
                               for i in range(a.n_grad_probe)})
            curv_at = sorted({0, n_steps // 2, max(n_steps - 1, 0)})
            model.train()
            step, ex = 0, 0
            while step < n_steps:
                if step in probe_at:
                    g = grad_probe(step)
                    if g:
                        cell["grad_probes"].append(g)
                if step in curv_at:
                    c = curvature_probe(step)
                    if c:
                        cell["curvature"].append(c)
                rows = [tr[(step * a.batch + j) % len(tr)] for j in range(a.batch)]
                loss = model(**collate(rows)).loss
                assert torch.isfinite(loss), f"non-finite loss at {key} step {step}"
                loss.backward()
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                ex += a.batch
                if ex in eval_at or step == n_steps:
                    model.eval()
                    nll = float(np.mean(P.score_pack(model, proc, device, ev, batch_size=a.batch)))
                    model.train()
                    den = nll_0 - nll_ref
                    s = (nll_0 - nll) / den if abs(den) > 1e-9 else float("nan")
                    cell["curve"].append({"examples": ex, "nll": nll, "s": s,
                                          "train_loss": float(loss.item())})
                    print(f"[M] {src}→{a.dst} s{seed} ex={ex} nll={nll:.4f} s={s:+.3f} "
                          f"({(time.time()-tcell)/60:.1f}min)", flush=True)
                    bank()
            model.eval()

            if live_b:
                dc, dr, dper = 0.0, 0.0, {}
                for n, k in live_b.items():
                    m = qk_mods[n]
                    dW = m.scale * (m.A.float() @ m.B.float())
                    c = float((dW @ basis_np[k]).pow(2).sum())
                    r_ = float(np.mean([float((dW @ Q).pow(2).sum()) for Q in rand_np[k]]))
                    dc += c
                    dr += r_
                    dper[n] = {"core": c, "rand": r_, "ratio": c / max(r_, 1e-30)}
                cell["arm_b_displacement_per_matrix"] = dper
                cell["arm_b"] = {"displacement_core": dc, "displacement_rand": dr,
                                 "displacement_ratio": dc / max(dr, 1e-30),
                                 "gradient_ratio_mean": float(np.mean(
                                     [g["ratio"] for g in cell["grad_probes"]]))
                                 if cell["grad_probes"] else None,
                                 "n_matrices": len(live_b)}

            cur = cell["curve"]
            xs = [c["examples"] for c in cur]
            ss = [min(max(c["s"], 0.0), 1.0) if c["s"] == c["s"] else 0.0 for c in cur]
            cost = sum((xs[i] - xs[i - 1]) * (1.0 - (ss[i] + ss[i - 1]) / 2.0)
                       for i in range(1, len(xs)))
            reached = next((c["examples"] for c in cur if c["s"] == c["s"] and c["s"] >= 0.5), None)
            cell["cost_auc"] = cost
            cell["examples_to_criterion"] = reached
            cell["censored"] = bool(reached is None)
            cell["cost_by_budget"] = {}
            for b in ladder:
                sub = [c for c in cur if c["examples"] <= b]
                sv = [min(max(c["s"], 0.0), 1.0) if c["s"] == c["s"] else 0.0 for c in sub]
                cell["cost_by_budget"][str(b)] = sum(
                    (sub[i]["examples"] - sub[i - 1]["examples"]) * (1.0 - (sv[i] + sv[i - 1]) / 2)
                    for i in range(1, len(sub)))
            cell["secs"] = time.time() - tcell
            cell["DONE"] = True
            detach_adapters(lm, names)
            del opt, sched
            torch.cuda.empty_cache()
            bank()
            print(f"[M] CELL DONE {key}→{a.dst} cost={cost:.1f} censored={cell['censored']} "
                  f"({cell['secs']/60:.1f}min, total {(time.time()-t0)/60:.1f}min)", flush=True)
        del src_sd

    # ---- ANCHOR VALIDITY GATE ------------------------------------------------------------------
    # The criterion anchor is only meaningful if the DEDICATED fine-tune is actually the best model
    # on its own task under this prompt/target binding. When it is not, every s_t on this
    # destination is normalised by a negative headroom and the whole column is uninterpretable —
    # so the destination is flagged, not silently divided. (The first run failed exactly here, on a
    # target-tokenisation defect; the gate now makes that observable instead of inferable.)
    # OFF-DIAGONAL sources only: the diagonal cell's nll_0 IS nll_ref (identical weights), so
    # including it makes the strict inequality unsatisfiable and the gate would always read False.
    n0 = [c["nll_0"] for c in cells.values() if "nll_0" in c and c["src"] != a.dst]
    rep["anchor_gate"] = {
        "nll_ref": nll_ref, "min_nll_0": min(n0) if n0 else None,
        "valid": bool(n0 and nll_ref < min(n0)),
        "margin": (min(n0) - nll_ref) if n0 else None,
        "failure_means": "the published ft-dst is NOT the best model on its own task under this "
                         "binding — this destination's column is uninterpretable and is reported "
                         "as VOID-anchor"}
    print(f"[M] anchor gate: nll_ref={nll_ref:.4f} min_nll_0="
          f"{rep['anchor_gate']['min_nll_0']} valid={rep['anchor_gate']['valid']}", flush=True)
    # ---- CRITIC FINDING 4: the q_proj / k_proj split, computed from the per-matrix banking -----
    def _side(name):
        return "q_proj" if name.endswith("q_proj") else "k_proj"

    split = {"q_proj": {"grad": [], "disp": []}, "k_proj": {"grad": [], "disp": []}}
    for c in cells.values():
        for g in c.get("grad_probes", []):
            for nm, v in (g.get("per_matrix") or {}).items():
                split[_side(nm)]["grad"].append(v["ratio"])
        for nm, v in (c.get("arm_b_displacement_per_matrix") or {}).items():
            split[_side(nm)]["disp"].append(v["ratio"])
    rep["arm_b_qk_split"] = {
        side: {"n_grad": len(d["grad"]),
               "grad_ratio_median": float(np.median(d["grad"])) if d["grad"] else None,
               "grad_ratio_p5_p95": [float(np.percentile(d["grad"], 5)),
                                     float(np.percentile(d["grad"], 95))] if d["grad"] else None,
               "n_disp": len(d["disp"]),
               "disp_ratio_median": float(np.median(d["disp"])) if d["disp"] else None,
               "disp_ratio_p5_p95": [float(np.percentile(d["disp"], 5)),
                                     float(np.percentile(d["disp"], 95))] if d["disp"] else None}
        for side, d in split.items()}
    rep["arm_b_qk_split_note"] = (
        "k_proj is [256, 2048] under MQA, so each ΔWᵀΔW has rank ≤ 256 and the consensus Gram has "
        "an exact kernel — its 'bottom-k' is an arbitrary selection from that kernel and the "
        "below-null gate cannot fail there. The q_proj half carries a REAL low-eigenvalue band. "
        "H_repelled's scope is whatever this split says (critic finding 4).")
    for side, v in rep["arm_b_qk_split"].items():
        print(f"[M] Arm B {side}: grad median={v['grad_ratio_median']} "
              f"disp median={v['disp_ratio_median']} (n={v['n_grad']}/{v['n_disp']})", flush=True)
    rep["total_secs"] = time.time() - t0
    rep["DONE"] = True
    bank()
    print(f"[M] FLEET DONE dst={a.dst} {len(cells)} cells in {(time.time()-t0)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
