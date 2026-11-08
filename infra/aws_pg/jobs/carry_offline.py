"""CARRY offline gate (CPU) — §9.1: the full Arm-C pipeline minus model-forward, on a CPU
instance with REAL data: real Q bank, real base + one real FT decoder, every one of the 9
cells BUILT + strict-loaded + leak-asserted + energy-booked, and the split identities that
only the offline gate can check end-to-end:

  * stiff_sd + flat_sd − base ≡ ft_sd per key (the A5 additivity identity on REAL ΔW)
  * per-key injected energies: stiff + flat = full; randA + randB = full (rank parity)
  * identity cell bitwise base

Run AFTER carry_scope (needs the Q bank), BEFORE the freeze.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.carry_offline --type r7i.4xlarge \
      --no-weights --s3-prefix results/carry_offline --budget-min 90
"""
import argparse
import json
import os
import pathlib

OFFLINE_FT = "google/paligemma-3b-ft-vqav2-224"


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
    from transformers import PaliGemmaForConditionalGeneration

    from infra.aws_pg.jobs import _carry_shared as CS
    from infra.pg import carry_core as CC
    from infra.pg import comwrite_surgery as SURG
    from infra.pg import inherit_pg1 as IP
    from infra.pg.entities import resolve_submodule_paths

    rep = {"object": "carry_offline_v1", "ft": OFFLINE_FT,
           "gpu_seconds": {"total": 0.0, "note": "CPU job"}}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    wd = "/tmp/carry_offline"
    anat = CS.load_roster(wd)
    qb = CS.load_q_bank(wd)
    k_stiff = int(qb["k_stiff"])
    U_stiff = qb["evecs"][:, :k_stiff].contiguous()
    rep["k_stiff"] = k_stiff

    model, proc = IP.load_pg1(device="cpu", sha=anat["base_sha"])
    paths = resolve_submodule_paths(model)
    lm = model.get_submodule(paths["language_model"])
    base_lm_sd = {k: v.detach().clone() for k, v in lm.state_dict().items()}
    vt_orig = {k: v.detach().clone()
               for k, v in model.get_submodule(paths["vision_tower"]).state_dict().items()}
    ft = PaliGemmaForConditionalGeneration.from_pretrained(
        OFFLINE_FT, revision=anat["shas"][OFFLINE_FT], torch_dtype=torch.bfloat16,
        device_map="cpu", attn_implementation="eager").eval()
    ft_sd = {k: v.detach().clone()
             for k, v in ft.get_submodule(
                 resolve_submodule_paths(ft)["language_model"]).state_dict().items()}
    del ft

    grid = SURG.grid_of(base_lm_sd)
    SURG.assert_full_grid(grid)
    bases_stiff = {lt: U_stiff for lt in grid}

    ident_sd, _ = SURG.build_decoder("identity", base_lm_sd, ft_sd, bases_stiff)
    assert all(torch.equal(ident_sd[k], base_lm_sd[k]) for k in base_lm_sd)
    rep["identity_gate"] = "bitwise_base_ok"
    del ident_sd

    built = {}
    cells = [("ft", "ft", bases_stiff), ("stiff", "shared", bases_stiff),
             ("flat", "residual", bases_stiff)]
    for s in range(CC.N_RAND_SEEDS):
        import hashlib
        seed = int(hashlib.sha1(f"carry_rand|vqav2|{s}".encode()).hexdigest()[:8], 16)
        U_rand, _idx = CC.random_eigsplit(qb["evecs"], k_stiff, seed)
        br = {lt: U_rand for lt in grid}
        cells += [(f"randA@{s}", "shared", br), (f"randB@{s}", "residual", br)]
    for tag, cellname, br in cells:
        sd, book = SURG.build_decoder(cellname, base_lm_sd, ft_sd, br)
        missing, unexpected = lm.load_state_dict(sd, strict=True)
        assert not missing and not unexpected, f"{tag}: LM load not clean"
        vt_now = model.get_submodule(paths["vision_tower"]).state_dict()
        assert all(torch.equal(vt_now[k].detach(), vt_orig[k]) for k in vt_orig), \
            f"{tag}: LEAK — vision tower changed"
        if tag in ("ft", "stiff", "flat"):
            built[tag] = sd            # kept for the additivity identity below
        rep.setdefault("cells", {})[tag] = {
            "injected_fro2": book["total_injected_fro2"],
            "captured_fro2": book["total_captured_fro2"],
            "total_fro2": book["total_fro2"]}
        print(f"[carry-offline] {tag}: built+loaded+leak-asserted "
              f"inj={book['total_injected_fro2']:.4e}", flush=True)
        _w()

    # ---- the A5 identities on REAL ΔW ---------------------------------------------------
    max_add = 0.0
    for key, w in base_lm_sd.items():
        if not SURG.is_target(key, w):
            continue
        b = w.float()
        recon = built["stiff"][key].float() + built["flat"][key].float() - b
        ftv = built["ft"][key].float()
        denom = float((ftv - b).norm()) or 1.0
        max_add = max(max_add, float((recon - ftv).norm()) / denom)
    rep["additivity_max_rel_err"] = max_add
    assert max_add < 5e-2, \
        f"stiff+flat != ft at {max_add:.3f} rel — bf16 cast noise exceeded the gate"
    e_ft = rep["cells"]["ft"]["injected_fro2"]
    e_sf = rep["cells"]["stiff"]["injected_fro2"] + rep["cells"]["flat"]["injected_fro2"]
    assert abs(e_sf - e_ft) / e_ft < 1e-3, f"energy partition broken: {e_sf} vs {e_ft}"
    for s in range(CC.N_RAND_SEEDS):
        e_ab = (rep["cells"][f"randA@{s}"]["injected_fro2"]
                + rep["cells"][f"randB@{s}"]["injected_fro2"])
        assert abs(e_ab - e_ft) / e_ft < 1e-3, f"rand split {s} energy partition broken"
    rep["energy_partition"] = "ok"

    lm.load_state_dict(base_lm_sd, strict=True)
    print("CARRY_OFFLINE_OK", json.dumps({"additivity_max_rel_err": max_add,
                                          "k_stiff": k_stiff}))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
