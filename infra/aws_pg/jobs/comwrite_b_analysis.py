"""COMWRITE Arm-B adjudication (spec §5.3/§7; CPU) — banked B metrics → calibrated null
bands → predicates → ONE primary identity label per donor (+ the full flag dict), the
interaction term per readout family, and the sealed `confirm` when both halves are given.

FROZEN calibration recipe (recorded here + in the journal BEFORE any sealed unblind):
  * effects are vs the identity cell (bitwise base — the §5.2 zero anchor);
  * the em-random band for a readout = [min_d(eff_d − hw_d), max_d(eff_d + hw_d)] over the
    ≥3 em-random draws, hw_d = the draw's own per-item bootstrap CI half-width (bands from
    DRAWS, never the best draw, §2; ppl has no per-item CI → hw=0);
  * damage = capped log₁₀ ppl, cap = 1e4 (above the 4-digit dead-model ppl the
    forgetting_locus instrument saw; cap hits disclosed per cell);
  * energy_median_ok := the shared cell's total capture fraction ≥ max(k*/d chance floor,
    0.5 × the A3 pooled sealed-ladder value at k*) — the §5.3 'carries ≥ the family-median
    captured energy' bookkeeping check, read from the Arm-A artifact.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.comwrite_b_analysis --type r7i.2xlarge --no-weights \
      --s3-prefix results/comwrite_b_verdicts --budget-min 20 --args "\
      --a-s3 s3://.../comwrite_a_anatomy/out \
      --disc-s3 s3://.../comwrite_b_vqav2_disc/out,s3://.../comwrite_b_textvqa_disc/out,... \
      --sealed-s3 s3://.../comwrite_b_vqav2_sealed/out,..."
"""
import argparse
import json
import pathlib

PPL_CAP = 1e4       # frozen (module docstring)


def _fetch(uri, dst):
    from infra.aws_pg.jobs._inherit_shared import fetch
    return json.loads(fetch(uri, dst).read_text())


def _hw(readout):
    ci = readout.get("ci95")
    return (float(ci[1]) - float(ci[0])) / 2 if ci else 0.0


def band_from_draws(effs_hws):
    """[min(eff−hw), max(eff+hw)] over the em draws — the frozen envelope."""
    los = [e - h for e, h in effs_hws]
    his = [e + h for e, h in effs_hws]
    return (min(los), max(his))


def donor_verdict(m, a_metrics, k_star):
    """One donor job's metrics.json → adjudicate() kwargs → verdict dict."""
    from infra.pg import comwrite_adjudicate as CA
    rd = m["readouts"]
    ident = rd["identity@0"]
    ft = rd["ft@0"]
    shared = rd["shared@0"]
    resid = rd["residual@0"]
    ems = [rd[t] for t in rd if t.startswith("em_random")]
    assert len(ems) >= 3, f"need ≥3 em draws, got {len(ems)} (§2 band-not-best-draw)"
    cross = sorted(k[6:] for k in ident if k.startswith("cross_"))

    def dppl(cell):
        return CA.d_ppl(cell["ppl"]["ppl"], ident["ppl"]["ppl"], PPL_CAP)

    score_band = band_from_draws([(e["donor"]["score"] - ident["donor"]["score"],
                                   _hw(e["donor"])) for e in ems])
    ppl_band = band_from_draws([(dppl(e), 0.0) for e in ems])
    cross_effs = [shared[f"cross_{s}"]["score"] - ident[f"cross_{s}"]["score"]
                  for s in cross]
    cross_bands = [band_from_draws([(e[f"cross_{s}"]["score"]
                                     - ident[f"cross_{s}"]["score"],
                                     _hw(e[f"cross_{s}"])) for e in ems]) for s in cross]
    nll_effs = [shared[f"nll_{s}"]["nll"] - ident[f"nll_{s}"]["nll"] for s in cross]
    nll_bands = [band_from_draws([(e[f"nll_{s}"]["nll"] - ident[f"nll_{s}"]["nll"],
                                   _hw(e[f"nll_{s}"])) for e in ems]) for s in cross]
    resid_cross = [resid[f"cross_{s}"]["score"] - ident[f"cross_{s}"]["score"]
                   for s in cross]

    cap_frac = (shared["book_totals"]["total_captured_fro2"]
                / shared["book_totals"]["total_fro2"])
    pooled = float(a_metrics["a3_pooled_ladder"][str(k_star)])
    d = int(a_metrics["per_matrix"][0]["d"])
    energy_ok = bool(cap_frac >= max(k_star / d, 0.5 * pooled))

    verdict = CA.adjudicate(
        s_base=ident["donor"]["score"], s_ft=ft["donor"]["score"],
        s_shared=shared["donor"]["score"], s_resid=resid["donor"]["score"],
        score_band=score_band,
        dppl_shared=dppl(shared), dppl_resid=dppl(resid), f_ppl=dppl(ft),
        ppl_band=ppl_band,
        cross_score_effects=cross_effs, cross_score_bands=cross_bands,
        cross_nll_effects=nll_effs, cross_nll_bands=nll_bands,
        resid_cross_effects=resid_cross, resid_cross_bands=cross_bands,
        energy_median_ok=energy_ok)
    inter = {fam: CA.interaction(shared[key]["score"] if fam != "ppl" else dppl(shared),
                                 resid[key]["score"] if fam != "ppl" else dppl(resid),
                                 ident[key]["score"] if fam != "ppl" else 0.0,
                                 ft[key]["score"] if fam != "ppl" else dppl(ft))
             for fam, key in (("donor_score", "donor"), ("ppl", "ppl"))}
    return {"donor": m["donor"], "half": m["half"], "verdict": verdict,
            "interaction": inter, "bands": {"score": score_band, "ppl": ppl_band,
                                            "cross": cross_bands, "nll": nll_bands},
            "cap_frac": cap_frac, "energy_ok": energy_ok,
            "ppl_by_cell": {t: rd[t]["ppl"]["ppl"] for t in rd if "ppl" in rd[t]},
            "cap_hits": {t: bool(rd[t]["ppl"]["ppl"] >= PPL_CAP)
                         for t in rd if "ppl" in rd[t]}}


def boot_p(cell_items, ident_items, edge, rng, n_boot=2000):
    """One-sided per-item bootstrap p that the (cell − identity) mean effect does NOT clear
    `edge` (the band's upper edge). Small p ⇒ the effect confidently clears the band."""
    import numpy as np
    c = np.asarray(cell_items, float)
    b = np.asarray(ident_items, float)
    assert c.size == b.size > 0, "per-item misalignment in bootstrap p"
    idx = rng.integers(0, c.size, size=(n_boot, c.size))
    effs = c[idx].mean(1) - b[idx].mean(1)
    return float((np.sum(effs <= edge) + 1) / (n_boot + 1))


def apply_bh_and_confirm(rep, seed=1337):
    """Sealed confirmation (§5.3) + BH-FDR q=0.10 over the promoted sealed family (§7):
    every SCORE-legged load-bearing inequality of each sealed label gets a per-item
    one-sided bootstrap p; relaw.bh_fdr rejects at q=0.10; a donor's label stands only if
    its discovery label reproduces AND every score leg survives BH. ppl legs have no
    per-item data (corpus instrument) → threshold checks only, disclosed. Consumes
    rep['sealed'] + rep['_sealed_raw'] (popped), fills rep['bh'] + rep['confirm']."""
    import numpy as np
    from infra.pg.comwrite_adjudicate import confirm
    from infra.pg.relaw import bh_fdr
    rng = np.random.default_rng(seed)
    bh_tests = []          # (donor, name, p)
    for donor, sv in rep["sealed"].items():
        rd = rep["_sealed_raw"][donor]["readouts"]
        ident, ft, shared, resid = (rd["identity@0"], rd["ft@0"], rd["shared@0"],
                                    rd["residual@0"])
        lab = sv["verdict"]["label"]
        sb_hi = sv["bands"]["score"][1]
        if lab == "H_damage_carrier" and ident["donor"]["per_item"]:
            # task-carried-by-residual's score leg: gain_resid beyond the band's upper
            # edge (the ≥frac·gain_ft conjunct is a point inequality, already in flags).
            bh_tests.append((donor, "task_carried_score",
                             boot_p(resid["donor"]["per_item"],
                                    ident["donor"]["per_item"], sb_hi, rng)))
        if lab == "H_shared_skill":
            cross_names = sorted(k[6:] for k in ident if k.startswith("cross_"))
            for s, b in zip(cross_names, sv["bands"]["cross"]):   # SAME sorted order as
                e = f"cross_{s}"                                  # donor_verdict's bands
                if shared[e].get("per_item") and ident[e].get("per_item"):
                    bh_tests.append((donor, f"cross_gain_{s}",
                                     boot_p(shared[e]["per_item"],
                                            ident[e]["per_item"], b[1], rng)))
        if lab == "H_entangled" and ident["donor"]["per_item"]:
            bh_tests.append((donor, "ft_differs_score",
                             boot_p(ft["donor"]["per_item"],
                                    ident["donor"]["per_item"], sb_hi, rng)))
    if bh_tests:
        rej = bh_fdr([p for _d, _n, p in bh_tests], q=0.10)
        rep["bh"] = [{"donor": d, "test": n, "p": p, "bh_pass": bool(r)}
                     for (d, n, p), r in zip(bh_tests, rej)]
        # a donor's label stands only if EVERY load-bearing score leg survives BH.
        bh_fail = {d for (d, _n, _p), r in zip(bh_tests, rej) if not r}
    else:
        rep["bh"], bh_fail = [], set()
    for donor, dv in rep["disc"].items():
        if donor in rep["sealed"]:
            c = confirm(dv["verdict"], rep["sealed"][donor]["verdict"])
            if donor in bh_fail:
                c["stands"] = False
                c["result_label"] = c["sealed_label"] + " (BH-failed score leg)"
            rep["confirm"][donor] = c
    rep.pop("_sealed_raw", None)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--a-s3", required=True)
    ap.add_argument("--disc-s3", required=True, help="comma list of B discovery prefixes")
    ap.add_argument("--sealed-s3", default="", help="comma list of B sealed prefixes")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg import comwrite_adjudicate as CA
    A = _fetch(f"{a.a_s3}/metrics.json", "/tmp/comwrite/ba_a.json")
    k_star = int(A["k_star"])

    rep = {"object": "comwrite_b_analysis_v1", "k_star": k_star, "ppl_cap": PPL_CAP,
           "disc": {}, "sealed": {}, "confirm": {}}
    rep["_sealed_raw"] = {}
    for tag, uris in (("disc", a.disc_s3), ("sealed", a.sealed_s3)):
        for i, uri in enumerate(x for x in uris.split(",") if x):
            m = _fetch(f"{uri}/metrics.json", f"/tmp/comwrite/ba_{tag}{i}.json")
            v = donor_verdict(m, A, k_star)
            rep[tag][m["donor"]] = v
            if tag == "sealed":
                rep["_sealed_raw"][m["donor"]] = m
            print(f"[b-analysis] {tag} {m['donor']}: {v['verdict']['label']} "
                  f"(energy_ok={v['energy_ok']}, cap_frac={v['cap_frac']:.3f})", flush=True)

    apply_bh_and_confirm(rep)
    labels = [v["verdict"]["label"] for v in (rep["sealed"] or rep["disc"]).values()]
    from infra.pg.comwrite_adjudicate import family_label
    rep["family"] = family_label(labels)
    (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    print("BANALYSIS_OK", json.dumps({"family": rep["family"],
                                      "confirm": {k: v["stands"]
                                                  for k, v in rep["confirm"].items()}}))


if __name__ == "__main__":
    main()
