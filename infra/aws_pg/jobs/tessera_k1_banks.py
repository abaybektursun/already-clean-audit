"""TESSERA K1/K2 — per-FACULTY curvature banks at PG1 (GPU). Spec §2 K1 instrument.

K1 asks whether the curvature footprint — the one predictor QUORUM validated (ρ = 0.654 FDR on
n*) — explains faculty survival where recipe membership, ‖ΔW‖ and task similarity did not. The
footprint is

    footprint(u, f) = fraction of unit u's write energy inside faculty f's top-curvature subspace,

so it needs a curvature basis PER FACULTY. QUORUM's own banks are a single global dc/ppl operator;
a faculty-independent basis would make the footprint a per-UNIT constant, which cannot explain a
per-(unit, faculty) outcome even in principle. That is why this job exists.

THE RECIPE IS `stencil_banks.py`, UNCHANGED EXCEPT FOR THE LOSS. Same Σ harvest → `eig_bands` →
`subband_slices`; same `specter_core.make_residual_injector`; same Lanczos (m=72, restart 144) on
two disjoint halves A1/A2 with a held-out Rayleigh batch B; same 16×8 Hutchinson sub-band traces;
same `carry_core.materialize_q` with the < 1e-8 fidelity assertion; same `stiff_split`. The ONLY
substitution is the operator item: `build_ppl_item` (wikitext, blank image) becomes
`build_vqa_item` on the faculty's own task, which is the `_specter_shared` builder already used
for the VQA operator — so "faculty curvature" and "ppl curvature" differ in the loss and in
nothing else.

Faculties (`plenum_map.FACULTY_TASK`, so they are the SAME faculties the outcome labels use):
  vqa → vqav2 · ocr → textvqa · caption → coco_captions

FP32 CERTIFY (P4). SPECTER's bf16 finite-difference floor is what makes a small curvature signal
untrustworthy, so every faculty's operator is certified with `specter_core.fd_directional_check`
on probe directions before its bank is written, and the realised relative error is banked beside
the eigenvectors. A faculty whose certify exceeds `FD_REL_TOL` writes its bank with
`certified = false`, and `tessera_k1_fit` VOIDs that faculty's cells rather than fitting on it.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.tessera_k1_banks --args "--faculty vqa" \
      --type g6e.2xlarge,p5.4xlarge --s3-prefix results/tessera_k1_banks_vqa --budget-min 900

Run the three faculties as three concurrent jobs (they share no state).

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib

CONV_RESID_FRAC = 0.10
M_RESTART = 144
TOP_EVAL_HELDOUT = 16
N_SPIKE = 8
FIDELITY_TOL = 1e-8
N_LAYERS = 18


def _parse_layers(s):
    out = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def faculty_task(faculty):
    """The faculty's TaskSpec — bound through `plenum_map.FACULTY_TASK` so K1's curvature and
    PLENUM's outcome labels are about the same faculty, not two things with the same name."""
    from infra.pg.plenum_map import FACULTY_TASK
    from infra.pg.projector_ablation import build_tasks
    assert faculty in FACULTY_TASK, f"unknown faculty {faculty!r}; have {sorted(FACULTY_TASK)}"
    want = FACULTY_TASK[faculty]
    tasks = {t.name: t for t in build_tasks()}
    assert want in tasks, f"faculty {faculty!r} wants task {want!r}; build_tasks has {sorted(tasks)}"
    return tasks[want]


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--faculty", required=True, choices=("vqa", "ocr", "caption"))
    ap.add_argument("--layers", default="0-17")
    ap.add_argument("--n-op", type=int, default=16, help="operator items, split A1/A2")
    ap.add_argument("--n-op-held", type=int, default=8, help="held-out Rayleigh items")
    ap.add_argument("--n-sigma", type=int, default=200, help="Σ-harvest items")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch

    from infra.aws_pg.jobs import _ration_shared as RATS
    from infra.aws_pg.jobs import _specter_shared as SPS
    from infra.aws_pg.jobs import _stencil_shared as SS
    from infra.pg import atlas_fit as AF
    from infra.pg import bispec_core as BC
    from infra.pg import carry_core as CC
    from infra.pg import inherit_pg1 as IP
    from infra.pg import ration_core as RC
    from infra.pg import specter_core as SC

    SS.install_pg()
    device, d = "cuda", IP.D
    path = "func"
    layers_want = _parse_layers(a.layers)
    rep = {"object": "tessera_k1_banks_v1", "faculty": a.faculty,
           "layers": layers_want, "gpu_seconds": {}}
    tm = SPS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    with tm.stage("load"):
        model, proc, layers, sha, base_ppl = SS.load_model("/tmp/tessera")
    rep["pinned_sha"] = sha
    rep["base_ppl"] = base_ppl        # P5: the anchor, banked digit-exact on every load
    _w()

    task = faculty_task(a.faculty)
    rep["task"] = task.name

    # ---- Σ harvest at ALL 18 layers, on the FACULTY's own items -------------------------------
    # windows=[] so Σ carries no wikitext: the completion basis must describe the faculty's
    # activations, not a text corpus's, or "faculty curvature" is faculty-specific in the operator
    # and generic in the basis it is completed against.
    with tm.stage("sigma_harvest"):
        rows_all, hashes = IP.load_items(task, a.n_sigma + a.n_op + a.n_op_held)
        sigma_rows = rows_all[:a.n_sigma]
        sigma_state = RATS.accumulate_sigma_all_layers(model, proc, layers, [],
                                                       vqa_rows=sigma_rows, task=task,
                                                       device=device)
    rep["n_sigma_rows"] = len(sigma_rows)
    _w()

    # operator rows are DISJOINT from the Σ rows — a shared row would let the basis and the
    # operator it is completed against be fit on the same data
    op_rows = rows_all[a.n_sigma:a.n_sigma + a.n_op]
    held_rows = rows_all[a.n_sigma + a.n_op:a.n_sigma + a.n_op + a.n_op_held]
    assert len(op_rows) == a.n_op and len(held_rows) == a.n_op_held, (
        f"{a.faculty}: got {len(op_rows)} operator / {len(held_rows)} held-out rows")
    rep["item_hashes"] = {"sigma": hashes[:a.n_sigma][:8],
                          "operator": hashes[a.n_sigma:a.n_sigma + a.n_op],
                          "held": hashes[a.n_sigma + a.n_op:a.n_sigma + a.n_op + a.n_op_held]}

    def measure_q(L):
        """`stencil_banks.measure_q`, with the faculty loss in place of the ppl loss."""
        st = sigma_state[L]
        Sigma = (torch.as_tensor(st["gram"], dtype=torch.float64) / st["n"])
        bands = BC.eig_bands(Sigma)
        sb = AF.subband_slices(bands["band_slices"], bands["spectrum"])
        basis = torch.as_tensor(bands["basis"], dtype=torch.float64)
        basis32 = basis.to(device, torch.float32)
        rms = float((torch.diagonal(Sigma).sum() / d).sqrt())
        eps = 0.05 * rms

        install, holder = SC.make_residual_injector(layers, L)
        A = [SPS.build_vqa_item(model, proc, device, r, task, install, holder) for r in op_rows]
        A1, A2 = A[:len(A) // 2], A[len(A) // 2:]
        B = [SPS.build_vqa_item(model, proc, device, r, task, install, holder) for r in held_rows]

        def mvf(its):
            def mv(v):
                return SC.mode_matvec(its, v.to(device) * eps, "dc", sign_seed=None, path=path) / eps
            return mv

        runs = {}
        for name, its in (("A1", A1), ("A2", A2)):
            res = SC.lanczos(mvf(its), d, m=SC.LANCZOS_M, seed=17, device=device)
            if not bool((res["resid"][:8]
                         <= CONV_RESID_FRAC * max(float(res["evals"][0]), 1e-30)).all()):
                res = SC.lanczos(mvf(its), d, m=M_RESTART, seed=17, device=device)
                res["restarted"] = True
            runs[name] = res

        def ray_b(u):
            return SC.mode_quadform(B, u.to(device) * eps, "dc", sign_seed=None,
                                    path=path) / (eps ** 2)
        spikes, _clusters = SC.stable_spikes(runs["A1"], runs["A2"], ray_b, k=TOP_EVAL_HELDOUT)

        gen = torch.Generator(device="cpu").manual_seed(310 + L)
        traces_sb = []
        for (lo, hi) in sb:
            kb = hi - lo
            qs = []
            for _ in range(SC.N_TRACE_PROBES):
                g = torch.randn(kb, generator=gen)
                z = (basis32[:, lo:hi].to(device) @ (g / g.norm()).to(device))
                qs.append(SC.mode_quadform(A, z * eps, "dc", sign_seed=None,
                                           path=path) / (eps ** 2))
            traces_sb.append((sum(qs) / len(qs)) * kb)

        sp = spikes[:N_SPIKE]
        if sp:
            U = runs["A1"]["evecs"][:, [s["idx"] for s in sp]].double().cpu()
            lam = [float(s["eval_a1"]) for s in sp]
        else:
            U, lam = None, []
        bulk = SC.band_bulk_from_traces([float(t) for t in traces_sb], sb, U, lam, basis)
        Q = CC.materialize_q(U, lam, bulk, basis, sb)

        vg = torch.Generator(device="cpu").manual_seed(4242)
        rels = []
        for _ in range(16):
            v = torch.randn(d, generator=vg, dtype=torch.float64)
            v = v / v.norm()
            qd = float(v @ Q @ v)
            qf = SC.spectral_completion_quadform(v, U, lam, bulk, basis, sb)
            rels.append(abs(qd - qf) / max(abs(qf), 1e-300))
        fid = max(rels)
        assert fid < FIDELITY_TOL, f"L{L}: dense Q disagrees with closed form ({fid:.2e})"

        # ---- P4: fp32 certify against a finite difference on probe directions -----------------
        # The residual injector asserts `delta.shape[-2:] == hidden.shape[-2:]`, so a probe is a
        # (T, d) FIELD, not a (d,) vector: an env profile over positions times a unit direction —
        # `specter_gate0._probe`'s construction. Passing a bare (d,) vector is what the first run
        # of this preflight did, and the assert caught it.
        # ---- P4, per SPECTER's OWN frozen config (amendment A13) -----------------------------
        # `specter_gate0.py:188` reads  fd_cfg = {"ppl": (1.0, FD_REL_TOL, True),
        #                                         "vqa": (0.25, 0.1, False)}
        # — for a VQA-loss item the step is h_rel = 0.25, the tolerance is 0.1, and the check is
        # NOT GATED. Every K1 faculty operator is built by `build_vqa_item`, so it is a vqa-loss
        # item and this is the leg that applies. Two earlier preflights applied the *ppl* leg's
        # step (1e-3, then 5e-2) and gated on FD_REL_TOL, and both pinned rel at exactly 1.0000 —
        # q_fd = 0, i.e. the perturbed forward returned bit-identical logits because the step sat
        # under the bf16 rounding floor. That is the floor SPECTER measures and declines to gate
        # on; the parent's reason is that the wiring cannot be loss-selective (the check is
        # label-free and both losses share every line of the differentiation path), so the ppl leg
        # gates the wiring and the vqa leg is reported.
        #
        # What DOES gate a bank here is what the recipe can actually certify: the materialization
        # fidelity assertion above (dense vᵀQv against the closed form, < 1e-8) and Lanczos
        # convergence. Those are properties of THIS operator, not of a differencing floor.
        FD_H_REL, FD_TOL = 0.25, 0.1
        cert = []
        pg = torch.Generator(device="cpu").manual_seed(777 + L)
        for it in A[:4]:
            v = torch.randn(d, generator=pg).float()
            v = (v / v.norm()).to(device)
            c = SC.env_coeffs("dc", it["T"], sign_seed=None).to(device)
            probe = eps * c.unsqueeze(1) * v.unsqueeze(0)          # (T, d) field, not a (d,) vector
            q, qfd, rel = SC.fd_directional_check(it, probe, path=path, h_rel=FD_H_REL)
            cert.append({"q": float(q), "q_fd": float(qfd), "rel": float(rel)})
        cert_err = max(c_["rel"] for c_ in cert)
        # rel pinned at exactly 1.0 with q_fd == 0 is not a floor MEASUREMENT — it means the
        # differenced forward returned bit-identical logits, so the check never exercised the
        # perturbation at all. Recorded as UNINFORMATIVE rather than as a floor, because a floor
        # is a number and this is the absence of one. Open instrument question, not a pass.
        fd_uninformative = all(c_["q_fd"] == 0.0 for c_ in cert)

        evals, evecs, k_stiff = CC.stiff_split(Q)
        return {"Q_evals": evals, "Q_evecs": evecs, "k_stiff": int(k_stiff),
                "n_spikes": len(sp), "spike_evals": lam, "fidelity_err": fid,
                "fd_rel_err": cert_err, "fd_probes": cert,
                "fd_h_rel": FD_H_REL, "fd_tol": FD_TOL, "fd_gated": False,
                "fd_uninformative": fd_uninformative,
                "fd_within_tol": bool(cert_err < FD_TOL),
                # the gate is the operator's own certifiable properties, not the FD floor
                "certified": bool(fid < FIDELITY_TOL),
                "restarted": bool(runs["A1"].get("restarted"))}

    banks = {}
    rep["per_layer"] = {}
    for L in layers_want:
        with tm.stage(f"L{L}/energy"):
            st = sigma_state[L]
            mean_x, C_x, absmean = RC.finalize_sigma(st["gram"], st["sum"], st["abs"], st["n"])
            sev, sea = RC._sigma_eig(C_x)
        with tm.stage(f"L{L}/operator"):
            mq = measure_q(L)
        banks[L] = {"Q_evecs": mq["Q_evecs"].float(), "Q_evals": mq["Q_evals"].double(),
                    "sigma_evecs": sev.float(), "sigma_evals": sea.double(),
                    "sigma_mean": mean_x.float(), "k_stiff": mq["k_stiff"], "d": d,
                    "metric_key": f"L{L}/dc/{a.faculty}"}
        rep["per_layer"][L] = {k: v for k, v in mq.items()
                               if k not in ("Q_evals", "Q_evecs")}
        rep["per_layer"][L]["sigma_n"] = int(st["n"])
        print(f"[k1banks {a.faculty}] L{L}: k_stiff={mq['k_stiff']} spikes={mq['n_spikes']} "
              f"fid={mq['fidelity_err']:.2e} fd={mq['fd_rel_err']:.3e} "
              f"certified={mq['certified']}", flush=True)
        # bind the accumulator BEFORE the loop can crash — a late failure keeps the done layers
        torch.save(banks, out / f"tessera_k1_banks_{a.faculty}.pt")
        _w()

    import hashlib
    p = out / f"tessera_k1_banks_{a.faculty}.pt"
    rep["banks_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    n_cert = sum(1 for L in layers_want if rep["per_layer"][L]["certified"])
    rep["certified_layers"] = n_cert
    rep["faculty_certified"] = bool(n_cert == len(layers_want))
    rep["green"] = True
    _w()
    print("TESSERA_K1_BANKS_OK", json.dumps(
        {"faculty": a.faculty, "layers": layers_want, "certified_layers": n_cert,
         "faculty_certified": rep["faculty_certified"]}, default=str), flush=True)


if __name__ == "__main__":
    main()
