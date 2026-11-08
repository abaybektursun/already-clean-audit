"""CLEAVE offline gate (CPU) — spec §9 steps 1–2, BEFORE the freeze / any GPU minute.

`feedback_offline_gate_before_gpu`: run the whole pipeline MINUS the model forward, on real data and
real dimensions, so the GPU only ever validates what ONLY the GPU can.

  A. the weight-free dry-run logic gate (`test_cleave_core.main`) — build_decoder("residual") IS
     ΔW_off; identity is bitwise base; the rank-d annihilation (A-CLEAVE-5); the grad partition; the
     resolution gate certifies/refuses/raises; the null-derived selectors; every §6 branch.
  B. the same pipeline at the REAL residual width d=2048 over a REAL-SHAPED 18×7 decoder grid
     (synthetic weights, tiny non-residual side): the surgery round-trip, the off-projection energy
     bookkeeping, `off_bases` across the full PROTECT ladder, resolution certification, and the
     mechanism census — plus the real MMLU roster/splits/dedup firewall and a battery-data smoke.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.cleave_offline --type r7i.4xlarge \
      --no-weights --s3-prefix results/cleave_offline --budget-min 60
"""
import argparse
import json
import os
import pathlib

INNER = 64          # the non-residual side of the synthetic grid (real d, tiny cost)


def _synth_decoder(d, inner, n_layers, seed=3):
    """A real-SHAPED (in the residual dimension) synthetic 18×7 decoder pair (base, ft)."""
    import torch

    from infra.pg import comwrite_core as CW
    from infra.pg.gauge_common import PROJ
    g = torch.Generator().manual_seed(seed)
    base, ft = {}, {}
    for L in range(n_layers):
        for t in PROJ:
            scope = "self_attn" if t in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
            k = f"model.layers.{L}.{scope}.{t}.weight"
            sh = (d, inner) if CW.residual_side(t) == "output" else (inner, d)
            Wb = torch.randn(*sh, generator=g, dtype=torch.float32) * 0.02
            base[k] = Wb
            ft[k] = Wb + 0.002 * torch.randn(*sh, generator=g, dtype=torch.float32)
    base["model.norm.weight"] = torch.ones(d)               # a non-target key
    ft["model.norm.weight"] = torch.ones(d)
    return base, ft


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch

    from infra.aws_pg.jobs import _cleave_shared as CS
    from infra.pg import cleave_core as CL
    from infra.pg import comwrite_surgery as SURG

    CS.install_pg()
    rep = {"object": "cleave_offline_v1", "gpu_seconds": {"total": 0.0, "note": "CPU job"}}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    # ---- A. weight-free dry-run logic gate ----------------------------------------------
    from infra.pg.tests import test_cleave_core as T
    n_pass = T.main()
    rep["dryrun"] = {"n_pass": int(n_pass), "n_total": len(T.ALL)}
    assert n_pass == len(T.ALL), "dry-run logic gate incomplete"
    _w()

    # ---- B1. MMLU adaptation roster + splits + dedup firewall (substrate = STENCIL's, A-CLEAVE-1)
    mmlu = CS.mmlu_by_subject()
    roster = CS.build_roster(mmlu)
    assert roster["discovery"] and roster["confirm"] and roster["broad_mmlu"], "empty roster partition"
    assert not (set(roster["adapt"]) & set(roster["broad_mmlu"])), "broad-MMLU overlaps adaptation"
    usable = [s for s in roster["adapt"] if CS.splittable(mmlu[s]["test"])]
    assert len(usable) >= 4, f"only {len(usable)} splittable adaptation subjects: {usable}"
    subj = usable[0]
    train, evl, ev_hashes = CS.subject_train_eval(mmlu[subj]["test"])
    examples, dedup = CS.train_examples(train, ev_hashes)
    assert examples and all("prompt" in e and "answer_idx" in e for e in examples), "format_qa shape"
    from infra.pg.relearn import qa_item_hash
    ev = set(ev_hashes)
    assert not any(qa_item_hash(r["question"]) in ev for r in dedup["kept"]), "train↔eval leak"
    # A-CLEAVE-14: the firewall-safe screen pool = eligible MINUS the 8 adapt subjects MINUS the
    # broad-MMLU forgetting probe. Never touched by discovery OR confirmation, so range-screening
    # these leaks nothing about the sealed contrast.
    from infra.aws_pg.jobs._stencil_shared import BROAD_MMLU_SUBJECTS, MIN_TEST, N_ADAPT
    eligible = sorted([x for x in mmlu if len(mmlu[x]["test"]) >= MIN_TEST and len(mmlu[x]["dev"]) >= 5],
                      key=lambda x: (-len(mmlu[x]["test"]), x))
    used = set(roster["adapt"]) | set(roster["broad_mmlu"])
    pool = [x for x in eligible if x not in used]
    pool_split = [x for x in pool if CS.splittable(mmlu[x]["test"])]
    rep["roster"] = {"discovery": roster["discovery"], "confirm": roster["confirm"],
                     "broad_mmlu": roster["broad_mmlu"], "n_eligible": roster["n_eligible"],
                     "splittable_adapt": usable, "example_subject": subj,
                     "n_train_examples": len(examples), "n_eval": len(evl),
                     "eligible_ordered": eligible,
                     "screen_pool_firewall_safe": pool,
                     "screen_pool_splittable": pool_split,
                     "screen_pool_sizes": {x: len(mmlu[x]["test"]) for x in pool_split}}
    _w()

    # ---- B2. the SURGERY at the real d, over a real-shaped 18×7 grid ---------------------
    d, n_layers = CS.D, CS.N_LAYERS
    g = torch.Generator().manual_seed(0)
    banks = {}
    for L in range(n_layers):
        Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
        S, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
        banks[L] = {"Q_evecs": Q, "sigma_evecs": S, "d": d}
    base_sd, ft_sd = _synth_decoder(d, INNER, n_layers)
    grid = CS.assert_grid(base_sd)
    assert len(grid) == n_layers * 7, f"grid {len(grid)} != {n_layers * 7}"

    # inertness through the SAME pipeline the treatment uses
    rep["identity_gate"] = CS.assert_identity_inert(base_sd, banks)

    # the ft cell reproduces ΔW exactly (the wiring gate's weight-free half)
    bases5 = CL.off_bases(banks, "random", 5, n_layers=n_layers)
    ft_cell, ft_book = SURG.build_decoder("ft", base_sd, ft_sd, bases5, n_layers=n_layers)
    assert all(torch.allclose(ft_cell[k].float(), ft_sd[k].float(), atol=1e-5) for k in ft_sd), \
        "ft cell != natural ΔW at real d"
    del ft_cell

    # the residual cell IS ΔW_off, and its removed-energy bookkeeping tracks the rank
    energy_by_rank = {}
    for r in (5, 50, 679, 2048):
        bs = CL.off_bases(banks, "susceptibility", r, n_layers=n_layers)
        cell, book = SURG.build_decoder("residual", base_sd, ft_sd, bs, n_layers=n_layers)
        frac = book["total_captured_fro2"] / book["total_fro2"]
        energy_by_rank[r] = frac
        k0 = "model.layers.0.self_attn.o_proj.weight"
        dW = ft_sd[k0].float() - base_sd[k0].float()
        want = dW - SURG.project_delta(dW, bs[(0, "o_proj")], "o_proj")
        got = cell[k0].float() - base_sd[k0].float()
        assert torch.allclose(got, want, atol=1e-5), f"r={r}: residual cell != ΔW_off at real d"
        if r == d:      # A-CLEAVE-5: at rank d the whole residual-facing update is removed
            assert frac > 0.999, f"rank-d off-projection removed only {frac:.4f} of the energy"
            assert all(torch.allclose(cell[k].float(), base_sd[k].float(), atol=1e-4)
                       for k in ft_sd), "rank-d residual cell is not the base (retention would not reach 0)"
        del cell, bs
    # removed energy must RISE with rank and bracket the Haar floor at the top
    ranks_sorted = sorted(energy_by_rank)
    assert all(energy_by_rank[ranks_sorted[i]] <= energy_by_rank[ranks_sorted[i + 1]] + 1e-6
               for i in range(len(ranks_sorted) - 1)), f"removed energy not monotone: {energy_by_rank}"
    rep["surgery_gate"] = {"grid": len(grid), "ft_cell": "reproduces_dW",
                           "removed_energy_by_rank": energy_by_rank,
                           "total_fro2": ft_book["total_fro2"]}
    _w()

    # ---- B3. off_bases across the FULL protect ladder + certification + census at real d --
    for r in CL.PROTECT_LADDER:
        bs = CL.off_bases(banks, "random", r, n_layers=n_layers)
        assert len(bs) == n_layers * 7 and tuple(bs[(0, "q_proj")].shape) == (d, r), (r, len(bs))
        del bs
    # certification: independent Q draws must NOT certify; near-identical draws must.
    reps_indep = [banks[L]["Q_evecs"].numpy() for L in (0, 1, 2)]
    ev_indep = [torch.linspace(10, 1, d).numpy() * (1 + 0.1 * i) for i in range(3)]
    cert_noise = CL.certify_resolution(reps_indep, ranks=(5, 50), evals_reps=ev_indep, d=d)
    assert cert_noise["certified_max"] == 0, f"independent Q draws certified: {cert_noise['per_rank']}"
    U0 = banks[0]["Q_evecs"]
    reps_stab = [U0.numpy()] + [torch.linalg.qr(U0 + 1e-5 * torch.randn(d, d, generator=g,
                                dtype=torch.float64))[0].numpy() for _ in range(2)]
    ev_stab = [torch.linspace(10, 1, d).numpy() * (1 + 1e-3 * i) for i in range(3)]
    cert_ok = CL.certify_resolution(reps_stab, ranks=(5, 50, 679), evals_reps=ev_stab, d=d)
    assert cert_ok["certified_max"] == 679, f"stable Q draws failed to certify: {cert_ok['per_rank']}"
    # mechanism census: mass planted in U_energy must read energy≈1, susceptibility≈floor
    r0 = 50
    U_en = banks[0]["sigma_evecs"][:, :r0]
    M = (U_en @ torch.randn(r0, INNER, generator=g, dtype=torch.float64)).numpy()
    cen = CL.mechanism_census(M, banks[0], r0, "o_proj")
    assert abs(cen["energy"] - 1.0) < 1e-8, cen
    assert CL.census_contrast(cen) > 0.5, cen
    rep["basis_gate"] = {"ladder_ok": list(CL.PROTECT_LADDER),
                         "cert_noise_certified_max": cert_noise["certified_max"],
                         "cert_stable_certified_max": cert_ok["certified_max"],
                         "cert_stable_per_rank": cert_ok["per_rank"],
                         "census_energy_planted": cen}
    _w()

    # ---- B4. selectors on a synthetic frontier (the ρ*-reachability contract, T2) ---------
    null_curve = [{"rank": r, "retention": max(0.0, 1.0 - (r / d) ** 0.5),
                   "reduction": min(1.0, (r / d) ** 0.5)} for r in CL.PROTECT_LADDER]
    r_star = CL.knee_rank([p["rank"] for p in null_curve], [p["reduction"] for p in null_curve])
    rho_star = CL.rho_star_from_null(null_curve, r_star)
    curves = {p: [{"rank": c["rank"], "retention": c["retention"]} for c in null_curve]
              for p in CL.POLICIES}
    reach = CL.rho_reachable(curves, rho_star)
    assert reach["reachable"], f"ρ*={rho_star} unreachable on a ladder that reaches d — T2 broken"
    rep["selector_gate"] = {"r_star": r_star, "rho_star": rho_star, "reachable": reach["reachable"]}
    _w()

    # ---- B5. processor + battery-data smoke ----------------------------------------------
    from transformers import AutoProcessor

    from infra.pg import inherit_pg1 as IP
    from infra.pg.relearn import letter_token_ids
    proc = AutoProcessor.from_pretrained(IP.PG1_REPO)
    letters = letter_token_ids(proc.tokenizer)
    assert len(set(letters)) == 4, letters
    vqa_task, coco_task = IP.pg1_tasks()
    vqa_rows, _ = IP.load_items(vqa_task, 4)
    coco_rows, _ = IP.load_items(coco_task, 4)
    assert len(vqa_rows) == 4 and len(coco_rows) == 4, "battery loaders"
    rep["battery_smoke"] = {"letter_token_ids": [int(x) for x in letters],
                            "vqa_rows": len(vqa_rows), "coco_rows": len(coco_rows),
                            "vqa_task": vqa_task.name, "coco_task": coco_task.name,
                            "mm_components": list(CS.MM_COMPONENTS)}
    rep["green"] = True
    _w()
    print("CLEAVE_OFFLINE_OK", json.dumps(
        {"dryrun": rep["dryrun"], "surgery": rep["surgery_gate"]["removed_energy_by_rank"],
         "splittable_adapt": usable, "r_star": r_star, "rho_star": rho_star}, default=str), flush=True)


if __name__ == "__main__":
    main()
