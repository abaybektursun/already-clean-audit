"""HUMBOLDT A3 — initialization fossil (spec §3/A3). CPU.

Instantiate the PG1 decoder config with seeded random init (3 seeds) and record A1's statistic set
on the init twin's weight matrices, using SVD spaces in place of consensus banks (there is no
fine-tune cloud at init). Then record principal angles between the REAL banks and
  (a) the init twin's corresponding singular subspaces, and
  (b) the BASE model's own top/bottom singular subspaces
for every type and layer — the complete recorded table that extends TRIV's σ-band observation.

  pg run infra.aws_pg.jobs.humboldt_a3 --type r7i.2xlarge --no-weights \
      --s3-prefix results/humboldt_a3 --budget-min 180
"""
import argparse
import itertools
import json
import pathlib

from infra.pg import humboldt_core as H
from infra.pg import humboldt_smoke as S
from infra.pg import humboldt_weights as W

INIT_SEEDS = (0, 1, 2)                 # §3/A3 "3 seeds, logged"
REAL_BANKS = ("core32", "anti32")      # the real objects the fossil table is about
SIGMA_BANKS = ("bottomsigma32", "topsigma32")


def _init_decoder(text_cfg, seed):
    """Seeded random init of the decoder architecture. Returns {(type, layer): W fp32}."""
    import torch
    from transformers import AutoModelForCausalLM
    torch.manual_seed(seed)
    m = AutoModelForCausalLM.from_config(text_cfg)
    sd = m.model.state_dict()
    out = {}
    for t in H.TYPES:
        for L in range(H.N_LAYERS):
            k = W.matrix_key(t, L)
            assert k in sd, f"init model missing {k} (keys e.g. {sorted(sd)[:3]})"
            out[(t, L)] = sd[k].detach().to(torch.float32)
    del m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--types", default=",".join(H.TYPES))
    ap.add_argument("--layers", default="")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--run-id", default="humboldt_a3")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    types = tuple(t.strip() for t in a.types.split(",") if t.strip())
    assert all(t in H.TYPES for t in types), f"unknown type in {types}"
    layers = ([int(x) for x in a.layers.split(",") if x.strip()] if a.layers.strip()
              else list(range(H.N_LAYERS)))
    init_seeds = INIT_SEEDS
    if a.smoke:
        layers, types, init_seeds = layers[:2], tuple(t for t in types if t != "down_proj"), (0,)

    H.hf_setup()                       # FIRST — XET off + gated-repo auth (§0.2.7)
    from huggingface_hub import HfApi
    from transformers import AutoConfig

    gates = H.Gates(out)
    S.run_all_offline(gates)

    api = HfApi()
    base_sha = api.model_info(W.PG1_REPO).sha
    text_cfg = AutoConfig.from_pretrained(W.PG1_REPO, revision=base_sha).text_config
    bank_dir = W.fetch_banks(H.cache_dir(out, "banks"))
    w_base = W._load_decoder(W.PG1_REPO, base_sha)

    manifest = {"arm": "A3", "run_id": a.run_id, "commit_sha": H.commit_sha(),
                "base_repo": W.PG1_REPO, "base_sha": base_sha, "types": list(types),
                "layers": layers, "init_seeds": list(init_seeds), "seed": a.seed,
                "init_recipe": "AutoModelForCausalLM.from_config(PaliGemmaConfig.text_config)",
                "initializer_range": float(getattr(text_cfg, "initializer_range")),
                "note": ("at init there is no fine-tune cloud, so the init twin's 'banks' are its "
                         "bottom-32 / top-32 ECONOMY right-singular subspaces")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    atlas = H.Atlas(out, arm="A3", run_id=a.run_id)      # bound BEFORE any loop (§0.2.4)

    # ---- the BASE model's own σ-subspaces + the real-vs-base angle table (b).
    base_sig = {}
    for t in types:
        for L in layers:
            Wm = w_base[W.matrix_key(t, L)].numpy()
            bots, tops, sig = H.sigma_subspaces(Wm, H.BANK_K)
            base_sig[(t, L)] = {"bottomsigma32": bots, "topsigma32": tops}
            fl = H.type_flags(t)
            atlas.add("base_singular_spectrum",
                      vector_ref=atlas.vector(f"A3__base_sigma__{t}__L{L}", sig),
                      layer=L, type=t, model_id=W.PG1_REPO, model_sha=base_sha, n=int(sig.size),
                      flags=fl)
            # §5: the statistic is recorded for the WHOLE battery against each σ reference, not
            # for the real banks with a couple of twins bolted on. `bottomsigma32 | base_bottomsigma32`
            # is degenerately zero by construction — that is exactly what a twin column is for: it
            # calibrates the scale of the comparison the real banks are read against.
            for rb in H.BANK_IDS:
                U = W.load_bank(bank_dir, t, L, rb)
                for sb in SIGMA_BANKS:
                    ang = H.principal_angles(U, base_sig[(t, L)][sb])
                    # Both subspaces live in the SAME type's input space, so the comparison is
                    # never cross-space — but the flag must SAY so rather than be absent, or a
                    # reader cannot distinguish "same space" from "nobody checked".
                    f2 = {**fl, "real_bank": rb, "reference": f"base_{sb}", "model_role": "base",
                          "is_real_bank": rb in REAL_BANKS, "cross_space": False,
                          "site_a": H.SITE[t], "site_b": H.SITE[t]}
                    atlas.add("bank_vs_base_sigma_angles",
                              vector_ref=atlas.vector(f"A3__ang_base__{t}__L{L}__{rb}__{sb}", ang),
                              layer=L, type=t, bank_id=f"{rb}|base_{sb}", model_id=W.PG1_REPO,
                              model_sha=base_sha, n=int(ang.size), flags=f2)
                    atlas.add_ci("bank_vs_base_sigma_angle_mean", ang, layer=L, type=t,
                                 bank_id=f"{rb}|base_{sb}", model_id=W.PG1_REPO,
                                 model_sha=base_sha, seed=a.seed, flags=f2)
            del Wm
        print(f"[A3] base σ table: {t} done ({atlas.n_rows} rows)", flush=True)
    H.s3_sync(out, f"humboldt/{a.run_id}")

    # ---- the init twins.
    band = H.haar_angle_band(2048, H.BANK_K, n_draw=200, seed=a.seed)
    atlas.add("haar_angle_band_mean",
              vector_ref=atlas.vector("A3__haar_angle_band_mean", band["mean"]), n=H.BANK_K,
              seed=a.seed, flags={"n": 2048, "k": H.BANK_K, "n_draw": band["n_draw"],
                                  "role": "chance column for every A3 angle cell"})
    for s in init_seeds:
        init = _init_decoder(text_cfg, s)
        mid = f"init_seed{s}"
        init_sig = {}
        for t in types:
            for L in layers:
                Wi = init[(t, L)].numpy()
                bots, tops, sig = H.sigma_subspaces(Wi, H.BANK_K)
                init_sig[(t, L)] = {"bottomsigma32": bots, "topsigma32": tops}
                fl = {**H.type_flags(t), "init_seed": s}
                atlas.add("init_singular_spectrum",
                          vector_ref=atlas.vector(f"A3__init{s}_sigma__{t}__L{L}", sig),
                          layer=L, type=t, model_id=mid, model_sha=f"seed{s}", n=int(sig.size),
                          seed=s, flags=fl)
                # A1's per-bank statistic set, on the init twin's own subspaces. At init there is
                # no fine-tune cloud, so bottomsigma32 stands in for the real bank and topsigma32
                # for its structured opposite; haar/complement supply the chance and generic
                # columns so the init sub-arm carries a battery too (§5).
                # The chance column is the Haar bank, which depends only on (n, k, seed) and is
                # therefore the SAME object for every model. core32/anti32/complement32/bottomsigma32
                # from the library are PG1 objects; recording them under an init model_id would
                # imply an init-dependence they do not have, so they are deliberately absent here.
                init_battery = dict(init_sig[(t, L)])
                for si2 in H.HAAR_SEEDS:
                    init_battery[f"haar32_s{si2}"] = W.load_bank(bank_dir, t, L, f"haar32_s{si2}")
                for sb, Ub in init_battery.items():
                    bfl = {**fl, "bank_is_model_independent": sb.startswith("haar32_s"),
                           "bank_source": ("haar (model-independent chance column)"
                                           if sb.startswith("haar32_s")
                                           else "the init twin's own economy σ subspace")}
                    lev = H.leverage(Ub)
                    atlas.add("leverage_profile",
                              vector_ref=atlas.vector(f"A3__init{s}_lev__{t}__L{L}__{sb}", lev),
                              layer=L, type=t, bank_id=sb, model_id=mid, model_sha=f"seed{s}",
                              n=int(lev.size), seed=s, flags=bfl)
                    atlas.add("leverage_max", value=float(lev.max()), layer=L, type=t, bank_id=sb,
                              model_id=mid, model_sha=f"seed{s}", seed=s,
                              flags={**bfl, "haar_expectation": H.BANK_K / H.IN_DIM[t]})
                    atlas.add("participation_ratio", value=H.participation_ratio(lev), layer=L,
                              type=t, bank_id=sb, model_id=mid, model_sha=f"seed{s}", seed=s,
                              flags={**bfl, "n_coords": int(lev.size)})
                # (a) real banks vs the init twin's singular subspaces.
                for rb in H.BANK_IDS:
                    U = W.load_bank(bank_dir, t, L, rb)
                    for sb in SIGMA_BANKS:
                        ang = H.principal_angles(U, init_sig[(t, L)][sb])
                        f2 = {**fl, "real_bank": rb, "reference": f"init_{sb}",
                              "model_role": "init", "is_real_bank": rb in REAL_BANKS,
                              "cross_space": False, "site_a": H.SITE[t], "site_b": H.SITE[t]}
                        atlas.add("bank_vs_init_sigma_angles",
                                  vector_ref=atlas.vector(
                                      f"A3__ang_init{s}__{t}__L{L}__{rb}__{sb}", ang),
                                  layer=L, type=t, bank_id=f"{rb}|init_{sb}", model_id=mid,
                                  model_sha=f"seed{s}", n=int(ang.size), seed=s, flags=f2)
                        atlas.add_ci("bank_vs_init_sigma_angle_mean", ang, layer=L, type=t,
                                     bank_id=f"{rb}|init_{sb}", model_id=mid,
                                     model_sha=f"seed{s}", seed=a.seed, flags=f2)
                del Wi
            print(f"[A3] init seed {s}: {t} done ({atlas.n_rows} rows)", flush=True)
        # cross-type angles among the init twin's own subspaces (A1's cross-type statistic, on init).
        for L in layers:
            for ta, tb in itertools.combinations([t for t in types if t in H.TYPES_RESID], 2):
                for sb in SIGMA_BANKS:
                    ang = H.principal_angles(init_sig[(ta, L)][sb], init_sig[(tb, L)][sb])
                    atlas.add_ci("init_cross_type_angle_mean", ang, layer=L, type=f"{ta}|{tb}",
                                 bank_id=f"{sb}|{sb}", model_id=mid, model_sha=f"seed{s}",
                                 seed=a.seed,
                                 flags={"init_seed": s, "type_a": ta, "type_b": tb,
                                        "cross_space": H.SITE[ta] != H.SITE[tb]})
        del init, init_sig
        H.s3_sync(out, f"humboldt/{a.run_id}")

    atlas.close()
    rep = atlas.to_parquet()
    (out / "a3_summary.json").write_text(json.dumps(
        {"manifest": manifest, "atlas": rep, "gates": gates.gates,
         "haar_band_first_last": [band["mean"][0], band["mean"][-1]]}, indent=2, default=str))
    H.s3_sync(out, f"humboldt/{a.run_id}")
    print("=" * 78)
    print(f"HUMBOLDT_A3_OK rows={rep['n_rows']} vectors={rep['n_vectors']} seeds={list(init_seeds)}")


if __name__ == "__main__":
    main()
