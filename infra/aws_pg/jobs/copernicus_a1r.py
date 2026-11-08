"""CO-A1r — the payload arm re-specified (v0.8 amendment §A1). Items 27 + 87.

WHAT CHANGED FROM v0.7, and nothing else changed:
  §A1.1  MEASUREMENT-VALIDITY replaces the ppl-scale saturation cap. A cell is valid iff its
         paired-bootstrap ΔNLL CI half-width ≤ 0.10 nats AND its ΔNLL < ln(1000). The old cap_sat
         flag is banked as context and gates nothing. A devastated model is information; an
         imprecise or out-of-range readout is not.
  §A1.2  A twin BATTERY replaces the single permutation twin: 5 twin draws per structured bank
         against ≥5 same-construction reference draws, two-sided rank-sum at α = 0.05. One hot
         draw can no longer void the instrument; a displaced twin POPULATION still can.
  §A1.3  The verdict scores on C2 — wikitext windows DISJOINT from v0.7's D ∪ C. The v0.7 C-half
         aggregates are published and therefore seen; nothing scored on them could claim a blind.

WHAT IS INHERITED VERBATIM: every v0.7 branch, margin (m_sup/m_eq), band, precedence row, and the
thresholds file itself — loaded BY HASH, never recalculated. The topenergy banks are CONSUMED from
the published v0.7 artifact under its do-not-regenerate contract, not rebuilt.

REGISTER: CONFIRMATORY-WITH-DISCLOSURE. The disclosure travels with the verdict — the analyst had
seen the old C-half aggregates when the amendment was written; C2 is unseen.

    pg run infra.aws_pg.jobs.copernicus_a1r \
        --args "--bank-s3 s3://.../results/copernicus_a1_banks/out \
                --v07-s3 s3://.../results/copernicus_a1/out --thresholds-sha256 <sha>" \
        --no-weights --type g6e.2xlarge --s3-prefix results/copernicus_a1r --budget-min 240
"""
import argparse
import json
import pathlib
import subprocess

import numpy as np

from infra.pg import copernicus_core as C
from infra.pg import humboldt_core as H

from infra.aws_pg.jobs.invariant_core import PG1_REPO, N_LAYERS
from infra.aws_pg.jobs.invariant_core_func import _decoder_layers, _load_full_pg1
from infra.aws_pg.jobs.copernicus_a1 import (
    build_corpus, split_dc, encode_windows, ppl_encoded, fetch_banks, by_layer,
    _InputProject, _ResidualProject, SPLIT_SALT, DECISIVE_K, SMALL_K_GATE, N_RAND_DRAWS, N_IN,
)

ARM = "CO-A1r"
QUEUE_ITEM = "27"
WAVE = "W0"                       # v0.8 calls it W-0r; the Atlas vocabulary keeps W0
N_TWIN_DRAWS = 5                  # §A1.2 — twin draws per structured bank
N_TWIN_REF = 5                    # §A1.2 — reference draws per structured bank
ADJUDICATED = ("core", "anti", "topenergy")


def fetch_v07(v07_s3: str, dest: pathlib.Path):
    """Consume the published v0.7 artifacts: thresholds (by hash) and the topenergy banks (by the
    manifest's per-bank hashes, honouring the do-not-regenerate contract)."""
    assert v07_s3.startswith("s3://"), f"--v07-s3 must be an s3 uri, got {v07_s3!r}"
    dest.mkdir(parents=True, exist_ok=True)
    for fn in ("thresholds.json", "topenergy_banks.npz", "topenergy_manifest.json",
               "copernicus_a1.json"):
        p = dest / fn
        if not p.exists():
            subprocess.run(["aws", "s3", "cp", f"{v07_s3}/{fn}", str(p),
                            "--region", "us-east-1", "--quiet"], check=True)
        assert p.stat().st_size > 0, f"empty v0.7 dependency {v07_s3}/{fn}"
    man = json.loads((dest / "topenergy_manifest.json").read_text())
    z = np.load(dest / "topenergy_banks.npz")
    topen, topen_g = {}, {}
    for k in man["k_grid"]:
        topen[k] = {L: np.asarray(z[f"topenergy/{k}/{L}"], dtype=np.float64) for L in man["layers"]}
        topen_g[k] = {L: np.asarray(z[f"topenergy_gamma/{k}/{L}"], dtype=np.float64)
                      for L in man["layers"]}
        for L in man["layers"]:
            got = C.bank_hash(topen[k][L])
            want = man["hashes"]["topenergy"][str(k)][str(L)]
            assert got == want, (
                f"topenergy/{k}/{L} hash {got[:12]} != published {want[:12]} — the consumed bank is "
                f"not the published object, and the do-not-regenerate contract exists to stop "
                f"exactly this")
    print(f"[a1r] consumed {sum(len(v) for v in topen.values())} published topenergy banks, "
          f"all hashes verified against the v0.7 manifest", flush=True)
    return topen, topen_g, man, json.loads((dest / "copernicus_a1.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bank-s3", required=True)
    ap.add_argument("--v07-s3", required=True)
    ap.add_argument("--thresholds-sha256", required=True)
    ap.add_argument("--k-grid", default="2,4,8,16,32,64,128")
    ap.add_argument("--ppl-window", type=int, default=512)
    ap.add_argument("--v07-scanned", type=int, default=400,
                    help="windows the v0.7 run consumed as D u C; C2 starts after them")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    k_grid = sorted(int(x) for x in a.k_grid.split(","))
    layers = list(range(N_LAYERS))

    H.hf_setup()
    import torch
    from transformers import AutoProcessor
    assert torch.cuda.is_available(), "GPU required"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda"

    banks, bmeta = fetch_banks(a.bank_s3, pathlib.Path(H.cache_dir(out, "a1_banks")))
    base_sha = bmeta["base_sha"]
    v07dir = pathlib.Path(H.cache_dir(out, "v07"))
    topen, topen_g, tman, v07rep = fetch_v07(a.v07_s3, v07dir)
    thresholds = C.load_thresholds(v07dir / "thresholds.json", a.thresholds_sha256)
    print(f"[a1r] inherited v0.7 thresholds by hash {a.thresholds_sha256[:16]} — nothing "
          f"recalibrated", flush=True)

    processor = AutoProcessor.from_pretrained(PG1_REPO, revision=base_sha)
    model = _load_full_pg1(base_sha, device)
    mlayers = _decoder_layers(model)

    # ---- C2: windows DISJOINT from v0.7's D ∪ C -----------------------------------------
    windows, cinfo = build_corpus(processor, a.ppl_window)
    seen = windows[:a.v07_scanned]
    C2 = windows[a.v07_scanned:]
    seen_D, seen_C = split_dc(seen, SPLIT_SALT)
    seen_idx = {w["idx"] for w in seen_D} | {w["idx"] for w in seen_C}
    assert not (seen_idx & {w["idx"] for w in C2}), "C2 overlaps the v0.7 D u C — not a fresh slice"
    assert len(C2) >= 120, (
        f"C2 holds {len(C2)} windows, under §2.1's floor of 120 — extend the corpus slice")
    print(f"[a1r] C2 = {len(C2)} windows, disjoint from v0.7's D({len(seen_D)}) u C({len(seen_C)})",
          flush=True)

    exp = C.Expectations(out / "copernicus_expectations.jsonl")
    exp.log(ARM, "a1r_primary",
            "director §A1.3: the repaired instrument PASSES its gates and the verdict lands "
            "TRAFFIC on the inherited v0.7 branch rules", confidence="mid",
            basis="v0.8 amendment §A1.3 prediction row", family="payload")
    exp.log(ARM, "a1r_twin_battery",
            "both twin batteries PASS — twin and reference are the same construction, so the "
            "populations should not separate", confidence="mid",
            basis="v0.8 §A1.2", family="payload")
    exp.log(ARM, "a1r_small_k",
            "the small-k residual (core-minus-topenergy at k=8) replicates on C2 and fires",
            confidence="low-mid", basis="v0.8 §A1.3 (genuinely open cell)", family="payload")
    at = C.Atlas(out, arm=ARM, run_id="copernicus_a1r", wave=WAVE, queue_item=QUEUE_ITEM,
                 expected_ref=exp.require(ARM, "a1r_primary"))
    gates = C.Gates(out)

    rep = {"object": "copernicus_a1r_v1", "arm": ARM, "program": C.PROGRAM,
           "queue_item": QUEUE_ITEM, "amendment": "v0.8 §A1",
           "register": "CONFIRMATORY-WITH-DISCLOSURE",
           "disclosure": ("the analyst had seen the v0.7 C-half aggregates when the amendment was "
                          "written; C2 is unseen and disjoint from v0.7's D u C"),
           "base_repo": PG1_REPO, "base_sha": base_sha, "bank_s3": a.bank_s3,
           "v07_s3": a.v07_s3, "thresholds_sha256": a.thresholds_sha256,
           "k_grid": k_grid, "layers": layers, "seed": a.seed,
           "corpus": {**cinfo, "v07_scanned": a.v07_scanned, "n_C2": len(C2)},
           "provenance": C.provenance(), "utc": C.utc_now()}

    def save():
        (out / "copernicus_a1r.json").write_text(
            json.dumps(rep, indent=2, sort_keys=True, default=float))
    save()

    # ---- the arm table -------------------------------------------------------------------
    rand_draws = {}
    for d in range(N_RAND_DRAWS):
        full = {L: C.nested_random_subspaces(N_IN, max(k_grid), a.seed * 7919 + 1000 * d + L)
                for L in layers}
        for k in k_grid:
            rand_draws.setdefault(k, []).append({L: np.ascontiguousarray(full[L][:, :k])
                                                 for L in layers})
    arms = {}
    for k in k_grid:
        arms[f"core/{k}"] = by_layer(banks, "core", k, layers)
        arms[f"anti/{k}"] = by_layer(banks, "anti", k, layers)
        arms[f"topenergy/{k}"] = topen[k]
        for d in range(N_RAND_DRAWS):
            arms[f"random{d}/{k}"] = rand_draws[k][d]
    for bid in ("topsigma", "bottomsigma"):
        arms[f"{bid}/{DECISIVE_K}"] = by_layer(banks, bid, DECISIVE_K, layers)

    # §A1.2 TWIN BATTERY: 5 twin draws + 5 same-construction reference draws, per structured bank.
    # SEED SPACING IS PART OF THE GATE. The battery's draws must be INDEPENDENT, so every
    # (draw, layer) cell needs its own seed. Spacing of 11 and 23 collided across draws — 28 of 90
    # twin cells shared a seed with another cell while the references had none — which makes the
    # five twin draws correlated: one unlucky shared layer-permutation lifts several draws
    # together. The first CO-A1r run's topenergy battery failed on exactly that. Spacing is now
    # 10000/20000, far wider than the 18 layers, giving zero collisions (asserted in the dry-run).
    for r in range(N_TWIN_DRAWS):
        arms[f"twin_core{r}/{DECISIVE_K}"] = {
            L: C.coordinate_permutation_twin(banks[f"core/{DECISIVE_K}/{L}"],
                                             a.seed + 10000 * (r + 1) + L)[0] for L in layers}
        arms[f"twin_topenergy{r}/{DECISIVE_K}"] = {
            L: C.coordinate_permutation_twin(topen[DECISIVE_K][L],
                                             a.seed + 20000 * (r + 1) + L)[0] for L in layers}
    for r in range(N_TWIN_REF):
        arms[f"twinref_dense{r}/{DECISIVE_K}"] = {
            L: C.coordinate_permutation_twin(banks[f"core/{DECISIVE_K}/{L}"],
                                             a.seed + 500 * (r + 1) + L)[0] for L in layers}
        cell = {}
        for L in layers:
            rng = np.random.default_rng(a.seed + 900 * (r + 1) + L)
            idx = rng.choice(N_IN, DECISIVE_K, replace=False)
            Q = np.zeros((N_IN, DECISIVE_K)); Q[idx, np.arange(DECISIVE_K)] = 1.0
            cell[L] = Q
        arms[f"twinref_onehot{r}/{DECISIVE_K}"] = cell

    # stage-2 banks reconstructed from the SAME generating objects and checked against the frozen
    # hashes — they were hashed into thresholds.json in v0.7 but never published as arrays.
    st2 = thresholds["stage2"]
    kb, sd_dim, small_dim = st2["k_b"], st2["set_difference_dim"], st2["small_k_set_difference_dim"]
    stage2 = {f"kb_topenergy/{kb}": topen[kb] if kb in topen else None}
    assert stage2[f"kb_topenergy/{kb}"] is not None, f"k_b={kb} is not in the published topenergy grid"
    sd_core, sd_te, sd_small = {}, {}, {}
    for L in layers:
        A, _, _ = C.set_difference_bank(banks[f"core/{DECISIVE_K}/{L}"], topen[DECISIVE_K][L])
        B, _, _ = C.set_difference_bank(topen[DECISIVE_K][L], banks[f"core/{DECISIVE_K}/{L}"])
        sd_core[L] = np.ascontiguousarray(A[:, :sd_dim])
        sd_te[L] = np.ascontiguousarray(B[:, :sd_dim])
        S, _, _ = C.set_difference_bank(banks[f"core/{SMALL_K_GATE}/{L}"], topen[SMALL_K_GATE][L])
        sd_small[L] = np.ascontiguousarray(S[:, :small_dim])
    stage2[f"core_minus_topenergy/{sd_dim}"] = sd_core
    stage2[f"topenergy_minus_core/{sd_dim}"] = sd_te
    stage2[f"core_minus_topenergy_smallk/{small_dim}"] = sd_small
    for dim, tag in ((sd_dim, "sd"), (small_dim, "sd_small")):
        for d in range(N_RAND_DRAWS):
            stage2[f"randdim_{tag}{d}/{dim}"] = {
                L: np.ascontiguousarray(
                    C.nested_random_subspaces(N_IN, max(k_grid),
                                              a.seed * 7919 + 1000 * d + L)[:, :dim])
                for L in layers}
    mismatch = []
    for name, cell in stage2.items():
        want = st2["hashes"].get(name)
        if not want:
            continue
        for L, Q in cell.items():
            if C.bank_hash(Q) != want[str(L)]:
                mismatch.append(f"{name}/L{L}")
    assert not mismatch, (
        f"reconstructed stage-2 banks do not match the v0.7 frozen hashes: {mismatch[:6]} — the "
        f"inherited thresholds and these banks describe different objects")
    print(f"[a1r] stage-2 banks reconstructed and hash-matched to the frozen v0.7 set "
          f"(k_b={kb}, sd_dim={sd_dim}, small_dim={small_dim})", flush=True)

    scope = [(f"scope_core/{DECISIVE_K}", by_layer(banks, "core", DECISIVE_K, layers)),
             (f"scope_topenergy/{DECISIVE_K}", topen[DECISIVE_K])]

    # ---- score C2, once ------------------------------------------------------------------
    enc = encode_windows(processor, device, C2)
    print(f"[a1r] encoded {len(C2)} C2 windows (processor run once)", flush=True)
    pw_clean = ppl_encoded(model, enc)
    clean_ppl = C.ppl(pw_clean)
    print(f"[a1r][C2] clean ppl = {clean_ppl:.4f}", flush=True)
    res = {}
    table = {**arms, **stage2}
    for i, (name, Qby) in enumerate(table.items()):
        with _InputProject(mlayers, Qby, device):
            pw = ppl_encoded(model, enc)
        ci = C.paired_log_ratio_ci(pw, pw_clean, seed=a.seed)
        mv = C.measurement_valid(ci["point"], ci["half_width"])
        res[name] = {"dnll": ci["point"], "ci": ci, "ppl": C.ppl(pw),
                     "measurement_valid": mv,
                     "cap_sat_context_only": bool(C.ppl(pw) > C.CAP_SAT_MULT * clean_ppl),
                     "per_window": pw.tolist()}
        print(f"[a1r][C2] {i + 1}/{len(table)} {name}: dNLL={ci['point']:+.4f} "
              f"hw={ci['half_width']:.4f} valid={mv['valid']}", flush=True)
    for name, Qby in scope:
        with _ResidualProject(mlayers, Qby, device):
            pw = ppl_encoded(model, enc)
        ci = C.paired_log_ratio_ci(pw, pw_clean, seed=a.seed)
        res[name] = {"dnll": ci["point"], "ci": ci, "ppl": C.ppl(pw), "scope": "full residual",
                     "measurement_valid": C.measurement_valid(ci["point"], ci["half_width"]),
                     "per_window": pw.tolist()}
        print(f"[a1r][C2] SCOPE {name}: dNLL={ci['point']:+.4f}", flush=True)
    rep["C2"] = {"clean_ppl": clean_ppl, "n_windows": len(C2),
                 "arms": {k: {kk: vv for kk, vv in v.items() if kk != "per_window"}
                          for k, v in res.items()}}
    rep["per_window"] = {k: v["per_window"] for k, v in res.items() if "per_window" in v}
    save()

    # ---- §A1.2 twin batteries ------------------------------------------------------------
    tb = {}
    for bank, twin_pref, ref_pref in (("core", "twin_core", "twinref_dense"),
                                      ("topenergy", "twin_topenergy", "twinref_onehot")):
        tw = [res[f"{twin_pref}{r}/{DECISIVE_K}"]["dnll"] for r in range(N_TWIN_DRAWS)]
        rf = [res[f"{ref_pref}{r}/{DECISIVE_K}"]["dnll"] for r in range(N_TWIN_REF)]
        g = C.twin_battery_gate(tw, rf)
        tb[bank] = g
        gates.record(f"a1r_twin_battery__{bank}", g["passes"], **{
            k: v for k, v in g.items() if k not in ("construction",)},
            band=g["construction"], prediction="both batteries PASS (§A1.2)")
    rep["twin_batteries"] = tb
    twins_ok = all(g["passes"] for g in tb.values())

    # ---- instrument gates (v0.7 §6.1.6, inherited; twin clause replaced by §A1.2) ---------
    bandD = {k: thresholds["bands"][str(k)] for k in k_grid}
    w_rand_dec = thresholds["bands"][str(DECISIVE_K)]["width"]
    anchor_clean = float(v07rep["anchor"]["clean_ppl"])
    g_dyn = bool(res[f"topsigma/{DECISIVE_K}"]["dnll"]
                 > bandD[DECISIVE_K]["hi"] + C.TOPSIGMA_DYNAMIC_RANGE * w_rand_dec)
    g_floor = bool(res[f"bottomsigma/{DECISIVE_K}"]["dnll"] <= bandD[DECISIVE_K]["hi"])
    g_sane = bool(clean_ppl < C.SANE_PPL_MULT * anchor_clean)
    gates.record("a1r_topsigma_dynamic_range", g_dyn,
                 topsigma_dnll=res[f"topsigma/{DECISIVE_K}"]["dnll"],
                 random_band_hi=bandD[DECISIVE_K]["hi"], w_rand=w_rand_dec,
                 band="inherited frozen D-half band (v0.7 thresholds.json)")
    gates.record("a1r_bottomsigma_floor", g_floor,
                 bottomsigma_dnll=res[f"bottomsigma/{DECISIVE_K}"]["dnll"],
                 random_band_hi=bandD[DECISIVE_K]["hi"],
                 band="inherited frozen D-half band (v0.7 thresholds.json)")
    gates.record("a1r_sane_ppl", g_sane, clean_ppl=clean_ppl, anchor_clean=anchor_clean,
                 bound=C.SANE_PPL_MULT * anchor_clean, band="frozen 2x the anchor's clean value")
    instrument_ok = bool(g_dyn and g_floor and g_sane and twins_ok)

    # ---- §A1.1 decisive k from MEASUREMENT-VALIDITY --------------------------------------
    valid_k = [k for k in k_grid
               if all(res[f"{b}/{k}"]["measurement_valid"]["valid"] for b in ADJUDICATED)]
    dec_k = DECISIVE_K if DECISIVE_K in valid_k else (max(valid_k) if valid_k else None)
    rep["measurement_validity"] = {
        f"{b}/{k}": res[f"{b}/{k}"]["measurement_valid"] for b in ADJUDICATED for k in k_grid}
    rep["valid_k"] = valid_k
    rep["decisive_k_used"] = dec_k
    gates.record("a1r_decisive_k_exists", dec_k is not None, valid_k=valid_k, decisive_k=dec_k,
                 pinned_at=DECISIVE_K,
                 band=("§A1.1: half-width <= 0.10 nats AND dNLL < ln(1000) on every adjudicated "
                       "bank; k falls to the largest all-valid width"))
    print(f"[a1r] MEASUREMENT-VALID k = {valid_k} -> decisive k = {dec_k}", flush=True)

    verdict, fired, detail = None, [], {}
    if not instrument_ok:
        verdict, detail = C.VOID_INSTRUMENT, {"reason": "an instrument gate did not pass"}
    elif dec_k is None:
        verdict, detail = C.VOID_INSTRUMENT, {"reason": "no k is MEASUREMENT-VALID on all banks"}
    else:
        pw = {n: np.asarray(v["per_window"]) for n, v in res.items() if "per_window" in v}
        m = thresholds["margins"][str(dec_k)]
        m_sup, m_eq = m["m_sup"], m["m_eq"]
        rand_avg = np.mean([pw[f"random{d}/{dec_k}"] for d in range(N_RAND_DRAWS)], axis=0)
        ci_core_rand = C.paired_log_ratio_ci(pw[f"core/{dec_k}"], rand_avg, seed=a.seed)
        ci_core_te = C.paired_log_ratio_ci(pw[f"core/{dec_k}"], pw[f"topenergy/{dec_k}"], seed=a.seed)
        ci_core_anti = C.paired_log_ratio_ci(pw[f"core/{dec_k}"], pw[f"anti/{dec_k}"], seed=a.seed)
        ci_anti_core = C.paired_log_ratio_ci(pw[f"anti/{dec_k}"], pw[f"core/{dec_k}"], seed=a.seed)
        ci_eqE = C.paired_log_ratio_ci(pw[f"core/{dec_k}"], pw[f"kb_topenergy/{kb}"], seed=a.seed)
        sd_band = thresholds["bands"][f"sd_dim{sd_dim}"]
        sd_m = thresholds["margins"][f"sd_dim{sd_dim}"]
        randdim_avg = np.mean([pw[f"randdim_sd{d}/{sd_dim}"] for d in range(N_RAND_DRAWS)], axis=0)
        ci_sd_rand = C.paired_log_ratio_ci(pw[f"core_minus_topenergy/{sd_dim}"], randdim_avg,
                                           seed=a.seed)
        ci_sd_vs = C.paired_log_ratio_ci(pw[f"core_minus_topenergy/{sd_dim}"],
                                         pw[f"topenergy_minus_core/{sd_dim}"], seed=a.seed)
        te_sd_out = bool(res[f"topenergy_minus_core/{sd_dim}"]["dnll"] > sd_band["hi"])
        overlap = float(np.mean([C.mean_cos2(arms[f"core/{dec_k}"][L], topen[dec_k][L])
                                 for L in layers]))
        non_adj = bool(overlap > C.BANK_OVERLAP_CSTAR)
        anti_overlap = float(np.mean([C.mean_cos2(arms[f"core/{dec_k}"][L], arms[f"anti/{dec_k}"][L])
                                      for L in layers]))
        anti_non_adj = bool(anti_overlap > C.BANK_OVERLAP_CSTAR)
        gates.record("a1r_equal_k_adjudicating", not non_adj, mean_cos2_core_topenergy=overlap,
                     mean_cos2_core_anti=anti_overlap, c_star=C.BANK_OVERLAP_CSTAR, k=dec_k,
                     band="frozen ceiling c*=0.25 (§2.1, inherited)")
        vd = C.a1_verdict({
            "instrument_ok": instrument_ok, "unflagged_k": valid_k, "dec_k": dec_k,
            "dnll": {b: {k: res[f"{b}/{k}"]["dnll"] for k in k_grid} for b in ADJUDICATED},
            "bands": bandD, "m_sup": m_sup, "m_eq": m_eq,
            "non_adjudicating": non_adj, "anti_non_adjudicating": anti_non_adj,
            "sd_dnll_core": res[f"core_minus_topenergy/{sd_dim}"]["dnll"],
            "sd_band": sd_band, "sd_m_sup": sd_m["m_sup"], "te_sd_outside": te_sd_out,
            "ci_core_rand": ci_core_rand, "ci_core_te": ci_core_te, "ci_core_anti": ci_core_anti,
            "ci_anti_core": ci_anti_core, "ci_eqE": ci_eqE, "ci_sd_rand": ci_sd_rand,
            "ci_sd_vs": ci_sd_vs})
        verdict, fired = vd["verdict"], vd["fired"]
        detail = {"legs": vd["legs"], "m_sup": m_sup, "m_eq": m_eq, "k": dec_k, "k_b": kb,
                  "ci_core_vs_random": ci_core_rand, "ci_core_vs_topenergy": ci_core_te,
                  "ci_core_vs_anti": ci_core_anti, "ci_anti_minus_core": ci_anti_core,
                  "ci_equal_energy_core_minus_kb": ci_eqE, "ci_setdiff_vs_random": ci_sd_rand,
                  "ci_setdiff_core_vs_te": ci_sd_vs, "set_difference_dim": sd_dim,
                  "stage2_pinned_at_k": DECISIVE_K,
                  "stage2_matches_decisive_k": bool(dec_k == DECISIVE_K)}
        # §6.1.7 small-k gate, inherited
        sband = thresholds["bands"][f"sd_small_dim{small_dim}"]
        smallrand = np.mean([pw[f"randdim_sd_small{d}/{small_dim}"] for d in range(N_RAND_DRAWS)],
                            axis=0)
        ci_small = C.paired_log_ratio_ci(pw[f"core_minus_topenergy_smallk/{small_dim}"],
                                         smallrand, seed=a.seed)
        sm_sup = thresholds["margins"][f"sd_small_dim{small_dim}"]["m_sup"]
        small_fired = C.ci_separated(ci_small, sm_sup)
        gates.record("a1r_small_k_residual_sink_specific", bool(small_fired), k=SMALL_K_GATE,
                     dim=small_dim, dnll=res[f"core_minus_topenergy_smallk/{small_dim}"]["dnll"],
                     ci_lo_one_sided=ci_small["lo_one_sided"], m_sup=sm_sup,
                     band=sband["construction"], prediction="fires, replicating v0.7's C half")
        detail["small_k_gate"] = {"fired": bool(small_fired), "ci": ci_small}

    gates.record("a1r_primary", verdict == C.A1_SINK_SPECIFIC, verdict=verdict, fired=fired,
                 decisive_k=dec_k, legs=detail.get("legs"),
                 twin_batteries={b: g["passes"] for b, g in tb.items()},
                 band="§6.1.5 branch set + precedence (inherited verbatim), scored on C2",
                 prediction="TRAFFIC (§A1.3 director row)")
    rep.update({"verdict": verdict, "verdict_branches_fired": fired, "verdict_detail": detail,
                "instrument_gates": {"topsigma_dynamic_range": g_dyn, "bottomsigma_floor": g_floor,
                                     "sane_ppl": g_sane, "twin_batteries": twins_ok,
                                     "all_ok": instrument_ok}})
    save()
    at.close()
    at.to_parquet()
    print("=" * 78, flush=True)
    print(f"COPERNICUS_A1R_OK verdict={verdict} fired={fired} decisive_k={dec_k} "
          f"valid_k={valid_k} twins_ok={twins_ok} n_C2={len(C2)}", flush=True)


if __name__ == "__main__":
    main()
