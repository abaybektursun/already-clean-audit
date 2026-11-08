"""N13 R1, weight-only (studies/next/07_PICKS/N13_SPEC.md §9.4 S2 step 3; §0.5, §3.2, §3.4, §5 PC-RANK and PC-ART, §6.1 G-INIT,
G-PARTITION, G-HASH, §6.3 H_artifact).

    python -m infra.aws_pg.jobs.n13_rank --setup-dir <setup out with base, stage2, rl> --out <dir>

Order (metrics.json rewritten after every stage):
  1. G-PARTITION on the RL index and config (failing input: the index with one MLP tensor removed); the shape fields of the base and RL
     configs equal.
  2. G-INIT: (a) from the setup step's shard hashes; (b) the unchanged MLP share base main vs RL against stage2 vs RL, margin 0.10
     (failing input: the two candidates swapped).
  3. Weight statistics, s and s' per neuron (ranking.npz), the sets of §3.2 (sets.json), J_floor at all k; the per-block relative
     weight change (A20: block_profile in metrics.json and sets.json; informational, no rule reads it).
  4. PC-RANK (n_plant planted below the median, k_ref = k at 1%) and its failing input (rows of down_proj only).
  5. PC-ART (N13_SPEC.md A2): the H_artifact rule on P+ (a synthetic one-step change of the base) must return SUPPORTED and on P-
     (P+ with 1% of neurons moved 8 bf16 steps) REFUTED; J_round and J_sp of the real change at all k (J_sp exploratory).
  6. The per-tensor sha256 of every arm of the roster (built in memory, not written) and G-HASH of zero and id.
controls.json holds every gate with its failing input. Then stage2 is deleted from disk: its snapshot, its refs entries and the blobs no
other revision links to (delete_snapshot, A23; until A23 the whole models-- folder went, base main with it).
Fail-loud: no try/except, no fallback.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import time

import numpy as np

from infra.pg import n13_arms as A
from infra.pg import n13_plan as P

TOTAL = 7_298_011_136
SHAPE_FIELDS = ("num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads", "vocab_size",
                "tie_word_embeddings", "sliding_window", "layer_types", "rope_scaling")


def dump(path, obj):
    path = pathlib.Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def links(d: pathlib.Path, blobs: pathlib.Path) -> dict:
    """The snapshot's files that are symlinks, each with the blob it resolves to (asserted inside the repo's blobs/)."""
    out = {}
    for p in d.rglob("*"):
        if p.is_symlink():
            t = p.resolve()
            assert t.parent == blobs, (f"A23: {p} links outside {blobs}", t)
            out[p] = t
    return out


def delete_snapshot(snap) -> dict:
    """A23 (N13_SPEC.md): remove one revision from the Hugging Face cache and nothing of another revision. The setup step's
    snapshot_download lays a repo out as models--<org>--<name>/ with blobs/<hash> (the bytes), snapshots/<commit>/<file> (symlinks
    ../../blobs/<hash>) and refs/<branch> (a commit; written only for a branch or tag, not for a commit revision). Base main and
    stage2-step47684 are two snapshots of one repo folder, and files of equal content share a blob. Removed: snapshots/<commit>, the
    refs entries that name the commit, and the blobs no other snapshot links to. Every other snapshot's links must still resolve."""
    snap = pathlib.Path(snap)
    repo, rev = snap.parents[1], snap.name
    assert snap.parent.name == "snapshots" and repo.name.startswith("models--") and snap.is_dir(), f"A23: not a cache snapshot: {snap}"
    blobs = (repo / "blobs").resolve()
    others = [d for d in (repo / "snapshots").iterdir() if d.name != rev]
    mine = set(links(snap, blobs).values())
    kept = set().union(*[links(d, blobs).values() for d in others])
    refs = [r for r in (repo / "refs").rglob("*") if r.is_file() and r.read_text().strip() == rev] if (repo / "refs").exists() else []
    gone = sorted(mine - kept)
    freed = sum(b.stat().st_size for b in gone)
    shutil.rmtree(snap)
    for r in refs:
        r.unlink()
    for b in gone:
        b.unlink()
    broken = [str(p) for d in others for p, t in links(d, blobs).items() if not t.exists()]
    assert not broken, ("A23: a link of another revision no longer resolves", broken[:5])
    return {"snapshot": str(snap), "refs_removed": [str(r) for r in refs], "blobs_removed": len(gone), "blobs_shared_kept": len(mine & kept),
            "bytes_freed": freed, "other_snapshots": sorted(d.name for d in others), "amendment": "N13_SPEC.md A23"}


def run(a) -> dict:
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    setup = json.loads((pathlib.Path(a.setup_dir) / "setup.json").read_text())
    sn = setup["snapshots"]
    m = {"step": "r1_rank", "status": "running", "stages": {}}
    w = lambda: dump(out / "metrics.json", m)
    w()
    cfg = json.loads((pathlib.Path(sn["rl"]) / "config.json").read_text())
    cfg_b = json.loads((pathlib.Path(sn["base"]) / "config.json").read_text())
    total = a.total or TOTAL
    base, rl, st2 = A.Weights(sn["base"]), A.Weights(sn["rl"]), A.Weights(sn["stage2"])
    t0 = time.time()

    # 1. G-PARTITION ------------------------------------------------------------------------------------------------------------
    names = rl.names()
    gp = A.partition(names, cfg, total)
    drop = next(n for n in names if A.family(n) == "mlp")
    gp["failing_input"] = {"what": f"the index without {drop}", "failed": not A.partition([n for n in names if n != drop], cfg, total)["pass"]}
    gp["config_shape_fields_equal"] = {k: cfg.get(k) == cfg_b.get(k) for k in SHAPE_FIELDS}
    gp["pass"] = gp["pass"] and all(gp["config_shape_fields_equal"].values()) and base.names() == names
    m["stages"]["G-PARTITION"] = gp
    w()
    assert gp["pass"] and gp["failing_input"]["failed"], ("G-PARTITION", gp)

    # 2. G-INIT -----------------------------------------------------------------------------------------------------------------
    sm, so = A.unchanged_mlp_share(base, rl, cfg), A.unchanged_mlp_share(st2, rl, cfg)
    gi = {"a": json.loads((pathlib.Path(a.setup_dir) / "metrics.json").read_text())["stages"]["G-INIT_a"], "b": A.g_init_share(sm, so)}
    gi["b"]["failing_input"] = {"what": "the two candidates swapped", "failed": not A.g_init_share(so, sm)["pass"]}
    gi["pass"] = gi["a"]["pass"] and gi["b"]["pass"]
    gi["failing_input"] = {"failed": gi["a"]["failing_input"]["failed"] and gi["b"]["failing_input"]["failed"]}
    m["stages"]["G-INIT"] = gi
    w()

    # 3. statistics, ranking, sets ----------------------------------------------------------------------------------------------
    s2, s2f, stats = A.weight_stats(base, rl, cfg)
    nn = A.neuron_count(cfg)
    ks = a.ks or A.k_grid(nn)
    np.savez(out / "ranking.npz", s2=s2, s2f=s2f, order=A.order(s2), order_floor=A.order(s2f))
    sets = A.make_sets(s2, s2f, ks, nn)
    sets["block_profile"] = m["block_profile"] = stats["block_profile"]      # A20: informational, no rule reads it
    dump(out / "sets.json", sets)
    dump(out / "weight_stats.json", {**stats, "ks": ks, "n_neurons": nn, "J_floor": sets["J_floor"],
                                     "rndT_m": {t: sets["rndT"][t]["m"] for t in A.K_TAGS}, "rndT_short": {t: sets["rndT"][t]["short"] for t in A.K_TAGS},
                                     "rndE_c": {t: sets["rndE"][t]["c"] for t in A.K_TAGS},
                                     "layer_hist_k1": np.bincount(np.array(sets["top"][A.K_TAGS[A.PRIMARY]]) // cfg["intermediate_size"],
                                                                  minlength=cfg["num_hidden_layers"]).tolist()})
    m["stages"]["stats"] = {"mlp_unchanged_share": stats["mlp_unchanged_share"], "ks": ks, "seconds": round(time.time() - t0, 1)}
    w()

    # 4. PC-RANK ----------------------------------------------------------------------------------------------------------------
    kref = ks[A.PRIMARY]
    pc = A.pc_rank(base, rl, cfg, s2, kref, a.n_plant)
    fi = A.pc_rank(base, rl, cfg, s2, kref, a.n_plant, along="down_rows")
    pc["pass"] = pc["recovery"] >= 0.99
    pc["failing_input"] = {"what": "planted along rows of down_proj (h = n mod hidden), gate and up untouched", "value": fi["recovery"],
                           "failed": fi["recovery"] <= 0.05, "chance": fi["chance"]}
    m["stages"]["PC-RANK"] = pc
    w()

    # 5. PC-ART (A2: the H_artifact rule on P+ and P-), J_round and J_sp ----------------------------------------------------------
    tag = A.K_TAGS[A.PRIMARY]
    o_sp = A.order(A.spacing_sq(base, cfg))

    def j_all(after, s2x, s2fx, with_sp):
        o = A.order(s2x)
        r = {"J_floor": {t: A.jaccard(o[:k], A.order(s2fx)[:k]) for t, k in zip(A.K_TAGS, ks)},
             "J_round": A.j_round(base, after, cfg, s2x, ks, a.n_redraw)["J_round"]}
        if with_sp:
            r["J_sp"] = {t: A.jaccard(o[:k], o_sp[:k]) for t, k in zip(A.K_TAGS, ks)}
        r["rule_k1"] = A.artifact_rule(r["J_floor"][tag], r["J_round"][tag])
        return r

    syn_t = A.synthetic_rounding(base, cfg)
    syn = A.Overlay(base, syn_t)
    s2_p, s2f_p, _ = A.weight_stats(base, syn, cfg)
    plus = j_all(syn, s2_p, s2f_p, True)
    planted_m = A.plant_multistep(base, cfg, syn_t, ks[A.PRIMARY])           # P+ becomes P- in place
    s2_m, s2f_m, _ = A.weight_stats(base, syn, cfg)
    minus = j_all(syn, s2_m, s2f_m, False)
    del syn, syn_t
    art = {"P_plus": plus, "P_minus": minus, "rule_P_plus": plus["rule_k1"], "rule_P_minus": minus["rule_k1"],
           "pass": plus["rule_k1"] == "SUPPORTED" and minus["rule_k1"] == "REFUTED",
           "rule": "the H_artifact rule at k1 returns SUPPORTED on P+ and REFUTED on P- (N13_SPEC.md A2)",
           "failing_input": {"what": f"P-: P+ with {len(planted_m)} neurons moved 8 bf16 steps on every entry (default_rng(41), signs 42)",
                             "rule": minus["rule_k1"], "failed": minus["rule_k1"] == "REFUTED"},
           "n_planted_P_minus": len(planted_m), "J_round_P_plus_reference": plus["J_round"]}
    jr = A.j_round(base, rl, cfg, s2, ks, a.n_redraw)
    o_rl = A.order(s2)
    j_sp = {t: A.jaccard(o_rl[:k], o_sp[:k]) for t, k in zip(A.K_TAGS, ks)}
    m["stages"]["PC-ART"] = art
    m["stages"]["H_artifact"] = {"J_floor": sets["J_floor"], "J_round": jr["J_round"], "J_sp": j_sp}
    w()

    # 6. arm hashes, G-HASH -----------------------------------------------------------------------------------------------------
    arms = [r["arm"] for r in P.roster()] if a.hash_arms == "roster" else [x for x in a.hash_arms.split(",") if x]
    hashes = {}
    for arm in arms:
        if arm in ("base", "rl"):
            continue
        aud = A.build_arm(base, rl, cfg, A.arm_spec(arm, sets), kept_sd=a.kept_sd)
        hashes[arm] = {"sha256": aud["sha256"], "mask_pass": aud["mask"].get("pass"), "kept_rel_err": aud.get("kept_rel_err"),
                       "kept_sd_units": aud.get("kept_sd_units"), "kept_pass": aud.get("kept_pass"),
                       "energy_rel_err": aud.get("energy_rel_err"), "energy_passes": aud.get("energy_passes"), "c_built": aud.get("c_built"),
                       "families": aud["families"]}
    dump(out / "arm_hashes.json", hashes)
    gh = {}
    if "zero" in hashes and "id" in hashes:
        hb, hr = A.snapshot_hashes(base), A.snapshot_hashes(rl)
        rev = A.build_arm(base, rl, cfg, {"name": "id_rev00", "kind": "complement", "neurons": [0], "c": 1.0, "non_mlp": "after"})
        gh = {"zero": A.g_hash(hashes["zero"]["sha256"], hb), "id": A.g_hash(hashes["id"]["sha256"], hr),
              "failing_input": {"what": "id with neuron (0, 0) reverted", "failed": not A.g_hash(rev["sha256"], hr)["pass"]}}
        gh["pass"] = gh["zero"]["pass"] and gh["id"]["pass"]
    m["stages"]["G-HASH"] = gh
    tk = f"top_{A.K_TAGS[A.PRIMARY]}"
    gm_ok = all(h["mask_pass"] for arm, h in hashes.items() if h["mask_pass"] is not None)
    bad = A.build_arm(base, rl, cfg, A.arm_spec(tk, sets), down_axis="rows")
    gmask = {"pass": gm_ok, "n_mask_arms": sum(h["mask_pass"] is not None for h in hashes.values()),
             "failing_input": {"what": f"{tk} built with down_proj rows instead of columns", "failed": not bad["mask"]["pass"]}}
    m["stages"]["G-MASK"] = gmask
    controls = {"G-PARTITION": gp, "G-INIT": gi, "PC-RANK": pc, "PC-ART": art, "G-HASH": gh, "G-MASK": gmask, "J_floor": sets["J_floor"],
                "J_round": jr["J_round"], "J_round_per_redraw": jr["per_redraw"], "J_sp": j_sp}
    dump(out / "controls.json", controls)
    if a.delete_stage2:
        m["stage2_deleted"] = delete_snapshot(sn["stage2"])
    m["status"] = "done"
    m["seconds"] = round(time.time() - t0, 1)
    w()
    return controls


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--setup-dir", required=True)
    ap.add_argument("--n-plant", type=int, default=1000)
    ap.add_argument("--n-redraw", type=int, default=A.N_REDRAW)
    ap.add_argument("--ks", type=lambda s: [int(x) for x in s.split(",")], default=None)
    ap.add_argument("--total", type=int, default=None)
    ap.add_argument("--hash-arms", default="roster")
    ap.add_argument("--kept-sd", type=float, default=4.0, help="DARE kept-count tolerance in binomial sd (N13_SPEC.md A4)")
    ap.add_argument("--delete-stage2", action="store_true")
    return ap


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
