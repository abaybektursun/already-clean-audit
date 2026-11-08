"""LACUNA G0 — the offline gate. Spec §6.1-G0 / §9 step 1: the ENTIRE decision surface minus the
model forward, on real dimensions, before a single GPU minute (`feedback_offline_gate_before_gpu`).

Sections
  A. S / CI / null-band / label logic on synthetic scores, INCLUDING the two replayed known events
     the spec names: a "policy ≈ null" row must emit `consumed`, and a "Δ_perception > 0, S ≈ null"
     row must emit `perception-alive + consumed`.
  B. every gate's FAILURE branch actually fires: under-powered G3, INVERTED G3, saturated θ_null,
     G5 scope cap on/off, BH with a VOID member excluded from m
     (`feedback_signals_that_cannot_fail`, `feedback_one_sided_thresholds`).
  C. the hash-parity splitter on real md5 content hashes.
  D. (torch) key canonicalisation across BOTH transformers namespaces, `install_canon`'s
     unknown-key / shape-mismatch raises, and the matched-ΔW + random-decoder surgeries on tiny
     tensors; the census statistics on a planted low-rank ΔW.
  E. (torch/net) the real item loaders + the real policy key inventory gate, no weights downloaded.

  local:  .pgvenv/bin/python -m infra.aws_pg.jobs.lacuna_offline --out /tmp/lac --no-torch
  pg:     ./infra/aws_pg/pg run infra.aws_pg.jobs.lacuna_offline --type r7i.2xlarge --no-weights \
              --s3-prefix results/lacuna_offline --budget-min 45
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-torch", action="store_true", help="control-plane run (sections A–C only)")
    ap.add_argument("--no-net", action="store_true", help="skip section E (loaders / repo inventory)")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from infra.pg import lacuna_core as LC

    rep = {"object": "lacuna_offline_v1", "sections": {}}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    rng = np.random.default_rng(0)
    N = 2000

    def bern(p, n=N, r=None):
        return (r or rng).binomial(1, p, n).astype(float)

    # =====================================================================
    # A. the S surface + the two replayed known events
    # =====================================================================
    ceiling = bern(0.62)                 # a PG1-like VQA ceiling
    floor = bern(0.18)                   # the matched-random-ΔW floor
    theta = LC.theta_null(floor, ceiling, n_boot=2000, seed=1)
    assert not theta["saturated"], theta
    A = {"theta_null": theta}

    # event 1 — "policy ≈ null" MUST read `consumed`
    pol_null = bern(0.185)
    s1 = LC.boot_S_paired(pol_null, floor, ceiling, n_boot=2000, seed=2)
    lab1 = LC.label_row(s1, theta["theta_null"],
                        LC.delta_perception(pol_null, bern(0.183), n_boot=2000, seed=3))
    assert lab1["label"] == "consumed", f"replayed event 1 emitted {lab1}"
    assert lab1["perception_alive"] is False, lab1
    A["event1_policy_at_null"] = lab1

    # event 2 — "Δ_perception > 0 while S ≈ null" MUST read `perception-alive + consumed`.
    # The gray row sits AT the floor (S ≈ 0) while the real-image row is measurably above it, i.e.
    # image information demonstrably flows even though no task score survives — the two-axis case
    # RFP could not express.
    gray = bern(0.181)
    real = np.array(gray)
    lift = rng.choice(N, size=int(0.05 * N), replace=False)
    real[lift] = 1.0
    s2 = LC.boot_S_paired(gray, floor, ceiling, n_boot=2000, seed=4)
    d2 = LC.delta_perception(real, gray, n_boot=2000, seed=5)
    lab2 = LC.label_row(s2, theta["theta_null"], d2)
    assert lab2["perception_alive"] is True, lab2
    assert lab2["label"] == "consumed", f"replayed event 2 emitted {lab2}"
    A["event2_perception_alive_and_consumed"] = lab2

    # a genuinely retained row must read `retained`
    ret = bern(0.62)
    s3 = LC.boot_S_paired(ret, floor, ceiling, n_boot=2000, seed=6)
    lab3 = LC.label_row(s3, theta["theta_null"], None)
    assert lab3["label"] == "retained", lab3
    A["event3_retained"] = lab3

    # a mid row must read `partial` WITH magnitude
    mid = bern(0.40)
    s4 = LC.boot_S_paired(mid, floor, ceiling, n_boot=2000, seed=7)
    lab4 = LC.label_row(s4, theta["theta_null"], None)
    assert lab4["label"] == "partial" and "S" in lab4 and lab4["S_lo95"] < lab4["S_hi95"], lab4
    A["event4_partial"] = lab4
    rep["sections"]["A"] = A
    _w()

    # =====================================================================
    # B. every failure branch fires
    # =====================================================================
    B = {}
    B["power_ok"] = LC.power_gate(floor, ceiling, n_boot=500)
    assert B["power_ok"]["passes"]
    B["power_underpowered"] = LC.power_gate(bern(0.55), ceiling, n_boot=500)
    assert not B["power_underpowered"]["passes"] and not B["power_underpowered"]["inverted"]
    B["power_inverted"] = LC.power_gate(bern(0.80), bern(0.20), n_boot=500)
    assert B["power_inverted"]["inverted"] and not B["power_inverted"]["passes"]

    # a floor that nearly touches the ceiling saturates the null band → labels VOID
    near = bern(0.60)
    B["theta_saturated"] = LC.theta_null(near, ceiling, n_boot=2000, seed=8)
    assert B["theta_saturated"]["saturated"], B["theta_saturated"]

    B["g5_fires"] = LC.scope_cap_g5({"pi0": {"vqav2": "consumed", "coco": "consumed"},
                                     "pi05": {"vqav2": "consumed", "coco": "consumed"}},
                                    {"pi0": -0.001, "pi05": 0.0})
    assert B["g5_fires"]["g5_fired"] and B["g5_fires"]["capped_verdict"]
    B["g5_silent_on_perception"] = LC.scope_cap_g5(
        {"pi0": {"vqav2": "consumed"}, "pi05": {"vqav2": "consumed"}},
        {"pi0": 0.05, "pi05": -0.01})
    assert not B["g5_silent_on_perception"]["g5_fired"]
    B["g5_silent_on_retained"] = LC.scope_cap_g5(
        {"pi0": {"vqav2": "partial"}, "pi05": {"vqav2": "consumed"}}, {"pi0": -0.01, "pi05": -0.01})
    assert not B["g5_silent_on_retained"]["g5_fired"]

    bh = LC.bh_reject({"T1": 0.001, "T2": 0.04, "T3": 0.6, "T4": None}, q=0.05)
    assert bh["m"] == 3 and bh["void"] == ["T4"] and "T1" in bh["rejected"] and "T3" not in bh["rejected"]
    B["bh"] = bh
    B["bh_all_void"] = LC.bh_reject({"T1": None}, q=0.05)
    assert B["bh_all_void"]["m"] == 0
    B["range_guard_saturated"] = LC.range_guard({"U1": 0.9, "U2": 0.85, "U3": 0.95, "U4": 0.82})
    B["range_guard_collapsed"] = LC.range_guard({"U1": 0.0, "U2": 0.0, "U3": 0.01, "U4": 0.0})
    B["range_guard_ok"] = LC.range_guard({"U1": 0.6, "U2": 0.5, "U3": 0.4, "U4": 0.05})
    assert B["range_guard_saturated"]["void_range"] and B["range_guard_collapsed"]["void_range"]
    assert not B["range_guard_ok"]["void_range"]
    sv = LC.savings_ci(bern(0.60, 200), bern(0.40, 200), n_boot=2000, seed=9)
    assert sv["lo95"] > 0 and sv["p_le0"] < 0.05, sv
    B["savings_positive"] = sv
    B["savings_null"] = LC.savings_ci(bern(0.40, 200), bern(0.40, 200), n_boot=2000, seed=10)
    assert B["savings_null"]["lo95"] < 0 < B["savings_null"]["hi95"], B["savings_null"]
    rep["sections"]["B"] = B
    _w()

    # =====================================================================
    # C. the hash-parity splitter on real md5 content hashes
    # =====================================================================
    hashes = [hashlib.md5(f"lacuna-item-{i}".encode()).hexdigest() for i in range(4000)]
    disc, conf = LC.parity_split(hashes)
    frac = sum(disc) / len(disc)
    assert 0.45 < frac < 0.55, frac
    rep["sections"]["C"] = {"n": len(hashes), "discovery_frac": frac,
                            "partition_ok": all(d ^ c for d, c in zip(disc, conf)),
                            "collision_check": len(set(hashes)) == len(hashes)}
    _w()

    if a.no_torch:
        rep["green_partial"] = True
        _w()
        print("LACUNA_OFFLINE_OK (sections A–C; --no-torch)",
              json.dumps({"theta_null": theta["theta_null"],
                          "labels": [lab1["label"], lab2["label"], lab3["label"], lab4["label"]]}),
              flush=True)
        return

    # =====================================================================
    # D. surgery + census logic on tiny tensors
    # =====================================================================
    import torch
    import torch.nn as nn
    from infra.pg import lacuna_shell as LS

    D = {}
    # canonicalisation unifies old and new transformers namespaces, and never eats `vision_model`
    assert LS.canon("language_model.model.layers.0.self_attn.q_proj.weight") == \
           LS.canon("model.language_model.layers.0.self_attn.q_proj.weight")
    assert LS.canon("model.vision_tower.vision_model.encoder.layers.3.mlp.fc1.weight") == \
           "vision_tower.vision_model.encoder.layers.3.mlp.fc1.weight"
    assert LS.canon("lm_head.weight") == "lm_head.weight"
    D["canon"] = "ok"

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.ModuleDict({"language_model": nn.ModuleDict(
                {"a": nn.Linear(4, 4, bias=False)})})
            self.lm_head = nn.Linear(4, 6, bias=False)

    t = Tiny()
    before = t.model["language_model"]["a"].weight.detach().clone()
    info = LS.install_canon(t, {"language_model.a.weight": torch.ones(4, 4)}, "tiny",
                            allow_missing=("lm_head",))
    assert info["n_installed"] == 1
    assert torch.equal(t.model["language_model"]["a"].weight.detach(), torch.ones(4, 4))
    assert not torch.equal(t.model["language_model"]["a"].weight.detach(), before)
    D["install"] = info
    for bad, why in (({"language_model.nope.weight": torch.ones(4, 4)}, "unknown key"),
                     ({"language_model.a.weight": torch.ones(3, 4)}, "shape mismatch")):
        try:
            LS.install_canon(t, bad, "tiny")
        except AssertionError as e:
            D[f"raises_{why.replace(' ', '_')}"] = str(e)[:90]
        else:
            raise AssertionError(f"install_canon did NOT raise on {why}")

    from infra.aws_pg.jobs.policy_content_readout import matched_dw_decoder
    from infra.aws_pg.jobs.vlat_retention import _random_decoder_sd
    g = torch.Generator().manual_seed(0)
    base = {"layers.0.self_attn.q_proj.weight": torch.randn(8, 8, generator=g),
            "layers.0.input_layernorm.weight": torch.randn(8, generator=g)}
    tgt = {k: (v + 0.3 * torch.randn(v.shape, generator=g)) for k, v in base.items()}
    md, frob = matched_dw_decoder(base, tgt, 1234)
    k2 = "layers.0.self_attn.q_proj.weight"
    got = float((md[k2] - base[k2]).norm())
    assert abs(got - frob[k2]) / frob[k2] < 1e-4, (got, frob[k2])
    assert torch.equal(md["layers.0.input_layernorm.weight"], base["layers.0.input_layernorm.weight"])
    D["matched_dw"] = {"target_frob": frob[k2], "achieved_frob": got, "norms_kept": True}
    rd = _random_decoder_sd(base, 7)
    assert not torch.equal(rd[k2], base[k2]) and torch.equal(
        rd["layers.0.input_layernorm.weight"], base["layers.0.input_layernorm.weight"])
    D["random_decoder"] = "ok"

    # census statistics: a planted low-rank ΔW must read BELOW the shape-matched random band
    W = torch.randn(64, 32, generator=g).numpy()
    lowrank = np.outer(np.random.default_rng(1).standard_normal(64),
                       np.random.default_rng(2).standard_normal(32))
    band = LC.srank_random_band((64, 32), n_draws=64, seed=3)
    # the §6.2 rule is one-sided ("low-rank iff BELOW the band"), so the gate tests exactly that:
    # a planted rank-1 ΔW must fall below p2.5, and fresh full-rank draws must not.
    assert LC.stable_rank(lowrank) < band["p2_5"], (LC.stable_rank(lowrank), band)
    fresh = [LC.stable_rank(np.random.default_rng(100 + i).standard_normal((64, 32)))
             for i in range(8)]
    assert min(fresh) >= band["p2_5"], (fresh, band)
    D["srank_band"] = {"lowrank": LC.stable_rank(lowrank), "band": band,
                       "fresh_draws_min": float(min(fresh)), "fresh_draws_max": float(max(fresh)),
                       "real_matrix_srank": LC.stable_rank(W)}
    # subspace overlap: identical bases → 1, independent bases → inside the null band
    Q = np.linalg.qr(np.random.default_rng(5).standard_normal((64, 8)))[0]
    same = LC.overlap_vs_rotation_null(Q, Q, n_draw=32, seed=6)
    diff = LC.overlap_vs_rotation_null(
        Q, np.linalg.qr(np.random.default_rng(7).standard_normal((64, 8)))[0], n_draw=32, seed=8)
    assert same["overlap"] > 0.99 and same["above_null"], same
    assert not diff["above_null"], diff
    D["overlap"] = {"identical": same["overlap"], "independent": diff["overlap"],
                    "null_p95": diff["null_p95"]}
    rep["sections"]["D"] = D
    _w()

    if a.no_net:
        rep["green_partial"] = True
        _w()
        print("LACUNA_OFFLINE_OK (A–D; --no-net)", flush=True)
        return

    # =====================================================================
    # E. real loaders + the real policy key inventory (headers only, no weights)
    # =====================================================================
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    E = {}
    tasks = LS.mm_tasks()
    assert set(tasks) == {"vqav2", "coco_captions", "textvqa", "refcoco"}, sorted(tasks)
    from infra.pg import inherit_pg1 as IP
    for name in ("vqav2", "coco_captions", "textvqa", "refcoco"):
        rows, hs = IP.load_items(tasks[name], 32)
        gt = LS.gray_task(tasks[name])
        im = gt.image_fn(rows[0])
        assert im.size == (224, 224), im.size
        d, c = LC.parity_split(hs)
        E[name] = {"n": len(rows), "uniq_hashes": len(set(hs)), "discovery": sum(d),
                   "prompt_example": tasks[name].prompt_fn(rows[0])[:80],
                   "gray_is_constant": bool(min(im.convert("L").getdata()) ==
                                            max(im.convert("L").getdata()) == 128)}
        print(f"[G0-E] {name}: {E[name]}", flush=True)
        _w()
    from huggingface_hub import HfApi
    api = HfApi()
    E["repo_shas"] = {k: api.model_info(v, revision="main").sha
                      for k, v in {**LS.POLICY_REPOS, "pg1": LS.PG1_REPO,
                                   "gemma1": LS.GEMMA1_REPO, "siglip": LS.SIGLIP_REPO}.items()}
    rep["sections"]["E"] = E
    rep["green"] = True
    _w()
    print("LACUNA_OFFLINE_OK", json.dumps({"theta_null": theta["theta_null"],
                                           "shas": E["repo_shas"]}, default=str), flush=True)


if __name__ == "__main__":
    main()
