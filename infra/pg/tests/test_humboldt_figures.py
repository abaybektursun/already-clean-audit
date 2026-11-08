"""Weight-free test of the figure/coverage/anomaly assembly on a SYNTHETIC atlas.

The figures job is the last thing that runs and the first thing a reader sees; it must not be the
first place its pandas/matplotlib paths are ever executed. This builds a miniature but
schema-faithful atlas for every arm, runs the assembly end to end, and checks the outputs.

Run: PYTHONPATH=. python3 infra/pg/tests/test_humboldt_figures.py
"""
import importlib.util
import json
import pathlib
import tempfile

import numpy as np

from infra.pg import humboldt_act as A
from infra.pg import humboldt_core as H

_spec = importlib.util.spec_from_file_location(
    "humboldt_figures", pathlib.Path(__file__).resolve().parents[2] / "aws_pg/jobs/humboldt_figures.py")
FIG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FIG)

LAYERS = [0, 1, 2]
TYPES = ("q_proj", "k_proj", "gate_proj")


def _write_arm(root, prefix, arm, emit):
    d = root / prefix
    d.mkdir(parents=True, exist_ok=True)
    at = H.Atlas(d, arm=arm, run_id=prefix)
    emit(at)
    at.close()
    (d / "gates.json").write_text(json.dumps({f"G_{arm}_demo": {"ok": True, "evidence": 1}}))
    return at


def _build(root):
    def a1(at):
        for t in TYPES:
            for L in LAYERS:
                n = H.IN_DIM[t]
                at.add("gram_spectrum", vector_ref=at.vector(f"A1__s__{t}__L{L}", np.arange(8)),
                       layer=L, type=t, n=8, flags=H.type_flags(t))
                at.add("reconstruction_residual", value=1e-15, layer=L, type=t,
                       flags=H.type_flags(t))
                for bid in H.BANK_IDS:
                    at.add("leverage_max", value=0.7 if bid == "core32" else 0.02 + 0.001 * L,
                           layer=L, type=t, bank_id=bid,
                           flags={**H.type_flags(t), "haar_expectation": 32 / n})
                    at.add("participation_ratio", value=100.0, layer=L, type=t, bank_id=bid,
                           flags=H.type_flags(t))
                    at.add("bank_energy_fraction_of_W", value=0.01, layer=L, type=t, bank_id=bid,
                           flags={**H.type_flags(t), "chance": 32 / n})
        for L in LAYERS:
            for ba, bb in (("core32", "core32"), ("anti32", "anti32"),
                           ("haar32_s0", "haar32_s0"), ("bottomsigma32", "bottomsigma32"),
                           ("complement32", "complement32"), ("core32", "anti32")):
                ang = np.full(32, 0.2 if ba == "core32" == bb else 1.4)
                at.add("cross_type_principal_angles",
                       vector_ref=at.vector(f"A1__a__L{L}__{ba}__{bb}", ang), layer=L,
                       type="q_proj|k_proj", bank_id=f"{ba}|{bb}", n=32,
                       flags={"cross_space": False})
                at.add_ci("cross_type_principal_angle_mean", ang, layer=L, type="q_proj|k_proj",
                          bank_id=f"{ba}|{bb}", flags={"cross_space": False})

    def a2(at):
        for t in ("q_proj", "k_proj"):
            for L in LAYERS:
                for bid in H.BANK_IDS:
                    c = np.random.default_rng(L).dirichlet(np.ones(128))
                    at.add("rope_band_coupling", vector_ref=at.vector(f"A2__c__{t}__L{L}__{bid}", c),
                           layer=L, type=t, bank_id=bid, n=128, flags=H.type_flags(t))
                    at.add("rope_coupling_participation_ratio", value=float(H.participation_ratio(c)),
                           layer=L, type=t, bank_id=bid, flags=H.type_flags(t))
                    at.add("rope_coupling_max", value=float(c.max()), layer=L, type=t,
                           bank_id=bid, flags=H.type_flags(t))

    def a3(at):
        for t in TYPES:
            for L in LAYERS:
                # angle rows must declare cross_space — the critic found A3 omitting it on 14 112
                # real rows, so the fixture models the corrected arm.
                afl = {**H.type_flags(t), "cross_space": False}
                for rb in H.BANK_IDS:
                    at.add_ci("bank_vs_base_sigma_angle_mean", np.full(32, 1.2), layer=L, type=t,
                              bank_id=f"{rb}|base_bottomsigma32", flags=afl)
                    at.add_ci("bank_vs_init_sigma_angle_mean", np.full(32, 1.3), layer=L, type=t,
                              bank_id=f"{rb}|init_bottomsigma32", model_id="init_seed0", flags=afl)

    def a4(at):
        for mid in ("org/ext-one", "org/ext-two"):
            for t in TYPES:
                for L in LAYERS:
                    at.add("ext_dW_fro2", value=1.0, layer=L, type=t, model_id=mid,
                           flags=H.type_flags(t))
                    for bid in H.BANK_IDS:
                        at.add("ext_dW_bank_energy_fraction",
                               value=0.001 if bid == "core32" else 0.02, layer=L, type=t,
                               bank_id=bid, model_id=mid,
                               flags={**H.type_flags(t), "chance": 32 / H.IN_DIM[t]})

    def b1(at):
        for mid in ("gemma-2b", "pg1-pt224"):
            for cid in ("wikitext", "parallel"):
                for L in LAYERS:
                    for site in H.SITES_PHASE_B:
                        order = [(t, bid) for t in A.BANK_TYPES_AT_SITE[site] for bid in H.BANK_IDS]
                        blk = np.random.default_rng(L).random((64, len(order))).astype("float32")
                        ref = at.vector(f"B1__eU__{mid}__{cid}__L{L}__{site}", blk, dtype="float32")
                        for col, (t, bid) in enumerate(order):
                            fl = {**H.type_flags(t), "chance": 32 / H.IN_DIM[t], "site": site}
                            at.add("e_U_series", vector_ref=f"{ref}#{col}", layer=L, type=t,
                                   bank_id=bid, model_id=mid, corpus_id=cid, n=64, flags=fl)
                            at.add_ci("e_U_mean", blk[:, col], layer=L, type=t, bank_id=bid,
                                      model_id=mid, corpus_id=cid, flags=fl)

    def b2(at):
        for mid in ("pg1-pt224",):
            for L in LAYERS:
                for t in TYPES:
                    for bid in H.BANK_IDS:
                        at.add_ci("parallel_paired_delta_vs_reference", np.random.default_rng(1).
                                  random(20), layer=L, type=t, bank_id=bid, model_id=mid,
                                  corpus_id="parallel", token_class="zh", flags=H.type_flags(t))
                        at.add("repetition_period_response", value=0.01, layer=L, type=t,
                               bank_id=bid, model_id=mid, corpus_id="repetition",
                               token_class="period16", flags=H.type_flags(t))

    def b3(at):
        for L in LAYERS:
            for t in ("q_proj",):
                for bid in H.BANK_IDS:
                    for k in (1, 4, 16):
                        at.add_ci("icl_shift_bank_fraction", np.random.default_rng(k).random(16),
                                  layer=L, type=t, bank_id=bid, model_id="gemma-2b",
                                  corpus_id="mmlu", token_class=f"{k}shot",
                                  flags={**H.type_flags(t), "shots": k, "chance": 32 / 2048})
                    at.add_ci("cross_query_shift_bank_fraction", np.random.default_rng(9).random(16),
                              layer=L, type=t, bank_id=bid, model_id="gemma-2b", corpus_id="mmlu",
                              token_class="cross_query_control",
                              flags={**H.type_flags(t), "chance": 32 / 2048})

    def b4(at):
        for L in LAYERS:
            for t in ("q_proj",):
                for bid in H.BANK_IDS:
                    for i, f in enumerate((0.001, 0.01, 0.3)):
                        fl = {**H.type_flags(t), "eps_frac": f, "eps_abs": f * 10}
                        at.add("perturb_attn_js", value=f * 0.5, layer=L, type=t, bank_id=bid,
                               model_id="pg1-pt224", token_class=f"eps{i}", flags=fl)
                        at.add("perturb_next_token_kl", value=f, layer=L, type=t, bank_id=bid,
                               model_id="pg1-pt224", token_class=f"eps{i}", flags=fl)
                        at.add("perturb_dppl", value=f * 2, layer=L, type=t, bank_id=bid,
                               model_id="pg1-pt224", token_class=f"eps{i}", flags=fl)

    for pref, arm, fn in (("humboldt_a1", "A1", a1), ("humboldt_a2", "A2", a2),
                          ("humboldt_a3", "A3", a3), ("humboldt_a4", "A4", a4),
                          ("humboldt_b1", "B1", b1), ("humboldt_b2", "B2", b2),
                          ("humboldt_b3", "B3", b3), ("humboldt_b4", "B4", b4)):
        _write_arm(root, pref, arm, fn)


def test_assembly_end_to_end():
    import pandas as pd
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        src = root / "arms"
        _build(src)
        out = root / "out"
        (out / "figures").mkdir(parents=True, exist_ok=True)

        frames = []
        present = {}
        for pref in FIG.ARM_PREFIXES:
            ok = (src / pref / "atlas.jsonl").exists()
            present[pref] = ok
            if ok:
                df = pd.read_json(src / pref / "atlas.jsonl", lines=True)
                df["_source"] = pref
                frames.append(df)
        atlas = pd.concat(frames, ignore_index=True)
        fl = atlas["flags"].apply(lambda s: json.loads(s) if isinstance(s, str) else {})
        atlas["chance"] = fl.apply(lambda d: d.get("chance"))
        atlas["site"] = fl.apply(lambda d: d.get("site"))
        atlas["eps_frac"] = fl.apply(lambda d: d.get("eps_frac"))

        anom = FIG._anomaly_log(atlas)
        assert len(anom) > 0, "the synthetic atlas plants deviations; none were detected"
        assert set(anom["bank_id"]) <= set(FIG.REAL_BANKS)
        assert (anom["deviation_multiple"] > FIG.DEVIATION_K).all()
        # the planted A1 leverage_max spike (core32 = 0.70 vs twins ≈ 0.02) must be caught
        assert (anom["statistic"] == "leverage_max").any()
        # the planted A1 cross-type diagonal deviation must be caught through the composite id
        assert (anom["statistic"] == "cross_type_principal_angle_mean").any()
        # the planted A4 avoidance must be caught
        assert (anom["statistic"] == "ext_dW_bank_energy_fraction").any()

        prov = "test"
        n_fig = 0
        for d, kw in (
            (atlas[atlas["statistic"] == "leverage_max"],
             dict(x="layer", series="bank_id", facet="type")),
            (atlas[atlas["statistic"] == "cross_type_principal_angle_mean"],
             dict(x="layer", series="bank_id", facet="type")),
            (atlas[atlas["statistic"] == "e_U_mean"],
             dict(x="layer", series="bank_id", facet="type", logy=True, chance="chance")),
            (atlas[atlas["statistic"] == "perturb_attn_js"],
             dict(x="eps_frac", series="bank_id", facet="layer", logx=True)),
            (atlas[atlas["statistic"] == "rope_coupling_max"],
             dict(x="layer", series="bank_id", facet="type")),
        ):
            p = FIG._sweep_figure(d, title="t", xlabel="x", ylabel="y",
                                  path=out / "figures" / f"f{n_fig}", provenance=prov, **kw)
            assert p and pathlib.Path(p).exists(), kw
            n_fig += 1
        print(f"  anomalies={len(anom)} figures={n_fig} rows={len(atlas)}")


if __name__ == "__main__":
    test_assembly_end_to_end()
    print("figures assembly test passed")
