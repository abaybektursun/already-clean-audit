"""HANDLE offline gate (CPU) — spec §9.1. Two halves, both BEFORE the freeze / any GPU minute:

  A. the synthetic D-suite (test_handle_core.main) — every §6 branch, the degenerate completions,
     the δ-mixture marginal matching, the binomial reachability, and the historical replays.
  B. the full ladder pipeline minus model-forward on REAL data — real Q bank, real base + one real
     FT decoder; build EVERY ladder cell (core_k per rung, band_k ×3, band_k^stiff ×3, ft) +
     strict-load + vision-tower-leak-assert + energy-book; the REAL-evec nesting asserts; and the
     ladder identities only the offline gate can check end-to-end:
       * core_k injected energy is monotone non-decreasing in k (nested projections)
       * core_{k_stiff} == the CARRY sealed `stiff` cell's banked injected energy for this FT
       * band_k / band_k^stiff ranks == k; identity cell bitwise base

  ./infra/aws_pg/pg run infra.aws_pg.jobs.handle_offline --type r7i.4xlarge \
      --no-weights --s3-prefix results/handle_offline --budget-min 90
"""
import argparse
import json
import os
import pathlib

OFFLINE_FT_SLUG = "vqav2"          # a scored-sealed FT with a banked CARRY stiff cell


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

    from infra.aws_pg.jobs import _handle_shared as HS
    from infra.pg import comwrite_surgery as SURG
    from infra.pg import handle_core as HC
    from infra.pg import inherit_pg1 as IP
    from infra.pg.entities import resolve_submodule_paths

    HS.install_pg()
    rep = {"object": "handle_offline_v1", "gpu_seconds": {"total": 0.0, "note": "CPU job"}}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    # ---- A. synthetic D-suite ------------------------------------------------------------
    from infra.pg.tests import test_handle_core as T
    n_pass = T.main()
    rep["dsuite"] = {"n_pass": int(n_pass), "n_total": len(T.ALL)}
    assert n_pass == len(T.ALL), "D-suite incomplete"
    _w()

    # ---- B. real-data ladder gate --------------------------------------------------------
    wd = "/tmp/handle_offline"
    fts = HS.load_carry_fts(wd)
    qb = HS.load_q_bank(wd)
    k_stiff = int(qb["k_stiff"])
    evecs = torch.as_tensor(qb["evecs"])
    ks = HC.ladder_ks(k_stiff)
    rep["k_stiff"], rep["ladder"] = k_stiff, list(ks)

    # REAL-evec nesting (§0.5-2) on the banked eigvectors
    stiff_proj = evecs[:, :k_stiff].double() @ evecs[:, :k_stiff].double().T
    rep["nesting"] = HC.assert_nesting(evecs, ks, k_stiff, stiff_projector=stiff_proj)
    _w()

    assert OFFLINE_FT_SLUG in fts["sealed_scored"], f"{OFFLINE_FT_SLUG} not a scored-sealed FT"
    repo = fts["repo_of"][OFFLINE_FT_SLUG]
    sha = fts["shas"][repo]
    carry_sealed = json.loads(HS.fetch(f"{HS.CARRY_C_SEALED}/metrics.json",
                                       pathlib.Path(wd) / "carry_sealed.json").read_text())
    banked_stiff_inj = carry_sealed["cells"][OFFLINE_FT_SLUG]["stiff"]["injected_fro2"]

    model, proc = IP.load_pg1(device="cpu", sha=fts["base_sha"])
    paths = resolve_submodule_paths(model)
    lm = model.get_submodule(paths["language_model"])
    base_lm_sd = {k: v.detach().clone() for k, v in lm.state_dict().items()}
    vt_orig = {k: v.detach().clone()
               for k, v in model.get_submodule(paths["vision_tower"]).state_dict().items()}
    ft_sd = HS.ft_decoder_sd(repo, sha)
    grid = SURG.grid_of(base_lm_sd)
    SURG.assert_full_grid(grid)

    # identity inertness
    ident_sd, _ = SURG.build_decoder("identity", base_lm_sd, ft_sd, HS.rung_bases(evecs[:, :k_stiff].contiguous(), grid))
    assert all(torch.equal(ident_sd[k], base_lm_sd[k]) for k in base_lm_sd), "identity not inert"
    del ident_sd
    rep["identity_gate"] = "bitwise_base_ok"

    def build_load_book(cellname, U, tag):
        sd, book = SURG.build_decoder(cellname, base_lm_sd, ft_sd, HS.rung_bases(U, grid))
        missing, unexpected = lm.load_state_dict(sd, strict=True)
        assert not missing and not unexpected, f"{tag}: LM load not clean"
        vt = model.get_submodule(paths["vision_tower"]).state_dict()
        assert all(torch.equal(vt[k].detach(), vt_orig[k]) for k in vt_orig), f"{tag}: LEAK"
        return book

    # ft (full write) once
    ft_book = build_load_book("ft", evecs[:, :k_stiff].contiguous(), "ft")
    rep["ft_injected_fro2"] = ft_book["total_injected_fro2"]
    rep["rungs"] = {}
    prev_core_inj = -1.0
    for k in ks:
        core_book = build_load_book("shared", evecs[:, :k].contiguous(), f"core@{k}")
        bmax = []
        for btype, pool in (("band", None), ("band_stiff", k_stiff)):
            for s in range(HC.N_RAND_SEEDS):
                U, idx = HC.band_basis(evecs, k, 1000 * k + s + (0 if pool is None else 7),
                                       within_stiff=pool)
                assert U.shape[1] == k, f"{btype}@{k} rank parity broken"
                bk = build_load_book("shared", U, f"{btype}@{k}.{s}")
                bmax.append(bk["total_injected_fro2"])
        rep["rungs"][str(k)] = {"core_injected_fro2": core_book["total_injected_fro2"],
                                "core_captured_fraction":
                                    core_book["total_captured_fro2"] / core_book["total_fro2"]}
        # nested projections capture monotonically more energy as k grows
        assert core_book["total_injected_fro2"] >= prev_core_inj - 1e-3, \
            f"core energy non-monotone at k={k}"
        prev_core_inj = core_book["total_injected_fro2"]
        print(f"[handle-offline] k={k}: core_inj={core_book['total_injected_fro2']:.4e} "
              f"cap_frac={rep['rungs'][str(k)]['core_captured_fraction']:.3f}", flush=True)
        _w()

    # core_{k_stiff} reproduces the CARRY sealed stiff cell's banked energy
    top_inj = rep["rungs"][str(k_stiff)]["core_injected_fro2"]
    rel = abs(top_inj - banked_stiff_inj) / max(banked_stiff_inj, 1e-30)
    rep["stiff_energy_reproduction"] = {"reproduced": top_inj, "banked": banked_stiff_inj,
                                        "rel_err": rel}
    assert rel < 1e-2, f"core_{{k_stiff}} energy {top_inj:.4e} != CARRY banked {banked_stiff_inj:.4e}"

    lm.load_state_dict(base_lm_sd, strict=True)
    HS.drop_snapshot(repo)
    print("HANDLE_OFFLINE_OK", json.dumps({"dsuite": rep["dsuite"], "k_stiff": k_stiff,
                                           "stiff_rel_err": rel}))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
