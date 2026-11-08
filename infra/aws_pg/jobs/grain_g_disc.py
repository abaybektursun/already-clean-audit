"""GRAIN G1 (GPU, DISCOVERY half only) — screening + threshold calibration for one read.
Everything the sealed pass consumes is FROZEN here and journaled before any sealed statistic
(spec §5.1/§5.2): the per-seed B* unions + the seed-median union, the grain-1 comparator, the
random-block-union draws + δ_G, the em-span construction, the equal-rank read-span cell, the
G3 support spread, and the promoted sealed cell list.

Screening recipe (frozen):
  a. alignment ranking (‖Q_bᵀV‖²_F/m) → top-16 blocks per m* seed;
  b. causal screening: those 16 blocks ablated singly on the screen slice;
  c. B*_s = cumulative union in descending single-block-effect order (≤8 blocks), stopped at
     ≥ρ·ΔRF(full read); seed-median union = the B*_s with median discovery effect (3 seeds);
  d. grain-1 comparator: m=1 dict, top-16 latents by alignment → single-latent screening →
     cumulative unions in chunks of m* latents up to rank(B*) (equal total rank).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.grain_g_disc --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/grain_g_disc_saction --budget-min 300 --args "\
      --read saction --dicts-s3 .../grain_train_primary/out \
      --harvest-s3 .../grain_harvest/out --gate0-s3 .../grain_gate0/out"
"""
import argparse
import json
import pathlib

RHO = 0.5
TOP_ALIGN = 16
MAX_BLOCKS = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--read", required=True, choices=("saction", "svl"))
    ap.add_argument("--dicts-s3", required=True)
    ap.add_argument("--gates-s3", default=None, help="gates.json location override (T3-v2 re-adjudication); defaults to --dicts-s3")
    ap.add_argument("--harvest-s3", required=True)
    ap.add_argument("--gate0-s3", required=True)
    ap.add_argument("--n-disc-items", type=int, default=400)
    ap.add_argument("--n-screen-items", type=int, default=160)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from infra.aws_pg.jobs import _grain_shared as GS
    from infra.pg import grain_bsf as GB
    from infra.pg import inherit_pg1 as IP
    from infra.pg.ctrliface_core import energy_matched_random, removed_energy
    from infra.pg.inherit_core import slice_k

    GS.install_pg()
    banks = GS.load_read_banks()
    L, V = banks[a.read]["L"], banks[a.read]["V"]
    key = f"L{L}"
    gates = json.loads(GS.fetch(f"{a.gates_s3 or a.dicts_s3}/gates.json", "/tmp/grain/gates.json").read_text())
    assert gates["L"] == L, f"dicts trained at L{gates['L']} but read {a.read} lives at L{L}"
    rep = {"object": "grain_g_disc_v1", "read": a.read, "L": L, "rho": RHO,
           "pivot": gates.get("pivot"), "m_star": gates.get("m_star")}
    (out / "g_disc.json").write_text(json.dumps(rep, indent=2))
    if gates.get("pivot") == "instrument_null":
        rep["mode"] = "UNRUN"
        (out / "g_disc.json").write_text(json.dumps(rep, indent=2, default=str))
        print("G_DISC_UNRUN (instrument_null pivot)")
        return
    m_star = gates.get("m_star")
    mode = "full" if m_star else "grain1_descriptive"
    rep["mode"] = mode

    def _w():
        (out / "g_disc.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    # --- dictionaries (banked; load, never retrain) ---
    def load_dict(m, seed):
        name = GS.dict_bank_name(gates["site"], m, seed)
        cell = gates["configs"].get(f"m{m}_s{seed}", {})
        assert cell.get("gate_pass"), f"dict m{m}_s{seed} did not pass T3 — not loadable for G"
        ae, cfg, q, sha = GB.load_dict_bank(GS.fetch(f"{a.dicts_s3}/{name}", f"/tmp/grain/{name}"))
        assert sha == gates["manifest_sha"], "dictionary trained on a different data manifest"
        return ae

    seeds = [s for s in (0, 1, 2) if gates["configs"].get(f"m{m_star}_s{s}", {}).get("gate_pass")] \
        if mode == "full" else []
    dicts_mstar = {s: load_dict(m_star, s) for s in seeds}
    ae_m1 = load_dict(1, 0) if gates["configs"].get("m1_s0", {}).get("gate_pass") else None

    # --- model + items + site stats ---
    sha, _ = GS.pinned_pg1_sha()
    model, proc = IP.load_pg1(sha=sha)
    layers = IP.decoder_layers(model)
    vqa, _coco = IP.pg1_tasks()
    it = json.loads(GS.fetch(f"{a.gate0_s3}/items_vqa.json", "/tmp/grain/items_vqa.json").read_text())
    rows, hashes = IP.load_items(vqa, it["stream_window"])
    assert hashes == it["hashes"], "item-stream drift vs the gate0 bank — halt"
    split = GS.grain_split(hashes, name="grain_vqa")
    disc_rows = [rows[i] for i in split["disc_idx"]][: a.n_disc_items]
    screen_rows = disc_rows[: a.n_screen_items]
    meta = json.loads(GS.fetch(f"{a.harvest_s3}/harvest_meta.json",
                               "/tmp/grain/harvest_meta.json").read_text())
    stats = torch.load(GS.fetch(f"{a.harvest_s3}/site_stats.pt", "/tmp/grain/site_stats.pt"),
                       map_location="cpu", weights_only=False)
    hbar = stats[L]["hbar"]
    Xs, labs = GS.load_shards(a.harvest_s3, meta, key)
    Xnull = Xs[:20000].double()                    # em-null energy inputs (fixed sample)

    # --- reference cells on the discovery slice ---
    intact = IP.eval_pg1(model, proc, vqa, disc_rows, "cuda")
    ii = np.asarray(intact["per_item"], float)
    floor_eval = IP.eval_pg1(model, proc, vqa, disc_rows, "cuda",
                             ctx=IP.replace_at(layers, L, hbar))
    floor = float(np.asarray(floor_eval["per_item"], float).mean())
    full = IP.eval_pg1(model, proc, vqa, disc_rows, "cuda",
                       ctx=IP.ablate_at(layers, L, slice_k(V, GS.K_READ), hbar))
    fi = np.asarray(full["per_item"], float)
    denom = ii.mean() - floor
    assert denom > 1e-3, "no dynamic range vs dead floor on discovery — halt"

    def effect(per_item, n=None):
        """RF drop 1 − RF on (optionally the first n) paired items."""
        pi = per_item[:n] if n else per_item
        iref = ii[:n] if n else ii
        return float(1.0 - (pi.mean() - floor) / (iref.mean() - floor))

    eff_full_screen = effect(fi, a.n_screen_items)
    rep["reference"] = {"intact": intact["score"], "floor": floor,
                        "full_read": full["score"], "effect_full_disc": effect(fi),
                        "effect_full_screen": eff_full_screen}
    _w()

    def eval_Q(rows_, Q):
        r = IP.eval_pg1(model, proc, vqa, rows_, "cuda", ctx=IP.ablate_at(layers, L, Q, hbar))
        return np.asarray(r["per_item"], float)

    # --- per-seed screening + greedy B* ---
    per_seed = {}
    if mode == "full":
        for s, ae in dicts_mstar.items():
            Wd = ae.decoder.weight.detach()
            align = GB.block_alignment(Wd, m_star, V)
            top = torch.topk(align, TOP_ALIGN).indices.tolist()
            singles = {}
            for b in top:
                pi = eval_Q(screen_rows, GB.block_subspace(Wd, m_star, b))
                singles[b] = effect(pi, a.n_screen_items)
            order = sorted(top, key=lambda b: -singles[b])
            union, union_curve = [], []
            for b in order[:MAX_BLOCKS]:
                union.append(b)
                pi = eval_Q(screen_rows, GB.union_Q(Wd, m_star, union))
                e = effect(pi, a.n_screen_items)
                union_curve.append({"blocks": list(union), "effect": e})
                if e >= RHO * eff_full_screen:
                    break
            per_seed[s] = {"align_top": top, "singles": singles,
                           "union_curve": union_curve, "bstar": list(union),
                           "bstar_effect": union_curve[-1]["effect"]}
            rep["per_seed"] = per_seed
            _w()
        med_s = sorted(seeds, key=lambda s: per_seed[s]["bstar_effect"])[len(seeds) // 2]
        bstar = per_seed[med_s]["bstar"]
        rep["median_seed"] = med_s
        rep["bstar"] = bstar
        rep["bstar_rank"] = len(bstar) * m_star
        if len(seeds) >= 2:
            jac, _ = GB.pairwise_jaccard([per_seed[s]["bstar"] for s in seeds])
            rep["bstar_seed_jaccard"] = jac
        _w()
        ae_med = dicts_mstar[med_s]
        Wmed = ae_med.decoder.weight.detach()
        r_bstar = len(bstar) * m_star

        # --- grain-1 comparator at equal total rank ---
        if ae_m1 is not None:
            W1 = ae_m1.decoder.weight.detach()
            score = ((W1.double() / W1.double().norm(dim=0, keepdim=True)).T @ V).pow(2).sum(1)
            top_l = torch.topk(score, TOP_ALIGN).indices.tolist()
            singles1 = {}
            for lat in top_l:
                pi = eval_Q(screen_rows, GB.union_Q(W1, 1, [lat]))
                singles1[lat] = effect(pi, a.n_screen_items)
            order1 = sorted(top_l, key=lambda x: -singles1[x])
            # extend the candidate pool by alignment if 16 < r_bstar latents are needed
            if r_bstar > len(order1):
                extra = [x for x in torch.topk(score, r_bstar).indices.tolist()
                         if x not in order1]
                order1 = order1 + extra
            g1_curve = []
            for n in range(m_star, r_bstar + 1, m_star):
                pi = eval_Q(screen_rows, GB.union_Q(W1, 1, order1[:n]))
                g1_curve.append({"n_latents": n, "effect": effect(pi, a.n_screen_items)})
            rep["grain1"] = {"singles": singles1, "curve": g1_curve,
                             "union": order1[:r_bstar]}
            _w()

        # --- δ_G calibration: random-block unions from the SAME dictionary (active blocks) ---
        usage = GB.block_usage(ae_med, Xs[:40000], labs[:40000], "cuda")
        active = torch.nonzero(usage["freq"] > 0).flatten().tolist()
        rep["n_active_blocks"] = len(active)
        g = torch.Generator().manual_seed(7)
        rand_draws, rand_rfs = [], []
        for dnum in range(3):
            perm = torch.randperm(len(active), generator=g)[: len(bstar)]
            blocks = [active[i] for i in perm.tolist()]
            pi = eval_Q(disc_rows, GB.union_Q(Wmed, m_star, blocks))
            rand_draws.append(blocks)
            rand_rfs.append(1.0 - effect(pi))
        delta_G = (max(rand_rfs) - min(rand_rfs)) / 2
        rep["random_unions"] = {"draws": rand_draws, "rf": rand_rfs, "delta_G": delta_G}
        _w()

        # --- em-span construction (energy matched to span(B*)'s removed energy) ---
        Qb = GB.union_Q(Wmed, m_star, bstar)
        E_target = removed_energy(Xnull, Qb, hbar)
        trace = float(((Xnull - hbar) ** 2).sum().item())
        Qem, r_e = energy_matched_random(IP.D, E_target, trace, 900, "cpu", torch.float64)
        rep["em_span"] = {"E_target": E_target, "trace": trace, "r_e": int(r_e), "seed": 900}
        _w()

        # --- G3 support spread (banks only) ---
        align_med = GB.block_alignment(Wmed, m_star, V)
        rep["support_spread"] = GB.support_spread(align_med, n_active=len(active))
        rep["align_energies"] = [float(x) for x in align_med]
        _w()

        # --- the frozen promoted sealed cells ---
        rep["promoted_cells"] = {
            "intact": {}, "floor_replace": {}, "identity_hook": {},
            "full_read": {"k": GS.K_READ},
            "bstar": {"blocks": bstar, "m": m_star, "seed": med_s},
            "rand_union_0": {"blocks": rand_draws[0]},
            "rand_union_1": {"blocks": rand_draws[1]},
            "rand_union_2": {"blocks": rand_draws[2]},
            "em_span": {"seed": 900, "r_e": int(r_e)},
            "equal_rank_span": {"k": r_bstar},
            "grain1_union": {"latents": rep.get("grain1", {}).get("union")},
        }
    else:
        # §11.3 pivot: grain-m unfaithful here → 1-D-only DESCRIPTIVE screening (no sealed
        # adjudication for this read; chunk size frozen at 8, ceiling at 64 latents).
        assert ae_m1 is not None, "grain-1 also unfaithful — should have been instrument_null"
        W1 = ae_m1.decoder.weight.detach()
        score = ((W1.double() / W1.double().norm(dim=0, keepdim=True)).T @ V).pow(2).sum(1)
        top_l = torch.topk(score, TOP_ALIGN).indices.tolist()
        singles1 = {}
        for lat in top_l:
            pi = eval_Q(screen_rows, GB.union_Q(W1, 1, [lat]))
            singles1[lat] = effect(pi, a.n_screen_items)
        order1 = sorted(top_l, key=lambda x: -singles1[x])
        extra = [x for x in torch.topk(score, 64).indices.tolist() if x not in order1]
        order1 = order1 + extra
        g1_curve = []
        for n in range(8, 65, 8):
            pi = eval_Q(screen_rows, GB.union_Q(W1, 1, order1[:n]))
            g1_curve.append({"n_latents": n, "effect": effect(pi, a.n_screen_items)})
        rep["grain1_descriptive"] = {"singles": singles1, "curve": g1_curve}
        rep["note"] = ("grain-m unfaithful at this site (§11.3 pivot) — descriptive grain-1 "
                       "screening only; no sealed adjudication for this read")

    rep["green"] = True
    _w()
    print("G_DISC_OK", json.dumps({"read": a.read, "mode": mode, "bstar": rep.get("bstar"),
                                   "delta_G": rep.get("random_unions", {}).get("delta_G"),
                                   "eff_full": rep["reference"]["effect_full_disc"]
                                   if "reference" in rep else None}, default=str))


if __name__ == "__main__":
    main()
