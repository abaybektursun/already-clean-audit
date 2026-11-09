"""f12: the data mixes. Per layer band, the share of a side's fifty strongest corpus positions that fall in the five
code-and-math directories, at the mixes' start and after 100 billion tokens of each mix; the drift between neighbouring
stage-1 snapshots; per-side histograms.

Reads only banked files in this repository (no network, no GPU) and writes data.js next to this file as
window.F12_DATA = {...}. Prints `path sha256` for every input.

    python3 extract.py
"""
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]                       # walkthrough/<picture> -> already-clean -> studies -> repository root
ID = HERE.name.upper()
MAX_BYTES = 300_000
SOURCES = []

M1 = "studies/next/m1"
SLOTS = 50
# Harvest keys -> public names (M1_SPEC.md §0.5; M1_ASSETS.md line 41: gen-mc-only 0 %, round5 40 %,
# math-code-reasoning-web 65 % code and math).
HARVESTS = {"start_a": "the start, step 476,838 of stage 1 (first harvest)",
            "start_b": "the start, step 476,838 of stage 1 (second harvest of the same weights)",
            "G": "after 100 billion tokens with no code or math (0 %)",
            "R": "after 100 billion tokens with 40 % code and math",
            "M": "after 100 billion tokens with 65 % code and math"}
PUBLIC = {"start_a": "start", "start_b": "start_second_harvest", "G": "no_code_or_math", "R": "code_and_math_40",
          "M": "code_and_math_65"}
MIXES = ("G", "R", "M")
# Neighbouring stage-1 snapshots that set the drift bar (M1_SPEC.md §3.2: pairs (t12, t13), (t13, t14b), (t14c, t15)).
REF_PAIRS = (("t12", "t13"), ("t13", "t14b"), ("t14c", "t15"))
REFS = ("t12", "t13", "t14b", "t14c", "t15")


def src(rel):
    """Open a repository file by its path from the root, record its sha256 once, print `path sha256`."""
    p = ROOT / rel
    b = p.read_bytes()
    assert not b.startswith(b"version https://git-lfs"), f"{rel} is an LFS pointer: git lfs pull --include={rel}"
    if rel not in [s["path"] for s in SOURCES]:
        h = hashlib.sha256(b).hexdigest()
        SOURCES.append({"path": rel, "sha256": h})
        print(rel, h)
    return p


def compare(name, recomputed, banked, where, tol):
    """Store a recomputed summary beside the banked one; print both and whether they differ beyond rounding."""
    same = abs(recomputed - banked) <= tol
    print(f"CHECK {name}: recomputed {recomputed} banked {banked} ({where}) {'same' if same else 'DIFFERS'}")
    return {"recomputed": recomputed, "banked": banked, "banked_at": where, "same_within": tol, "same": same}


def write(data, notes):
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    data["meta"] = {"commit": commit, "sources": SOURCES,
                    "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                    "notes": notes}
    out = HERE / "data.js"
    out.write_text(f"window.{ID}_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    size = out.stat().st_size
    assert size <= MAX_BYTES, f"{out} is {size:,} bytes"
    print(f"wrote {out.relative_to(ROOT)}: {size:,} bytes")


def load(rel):
    """counts.npz -> (counts [32, 11008, 2, 50 directories], code-and-math slots per side [32, 11008, 2], filled slots per
    side, directory names)."""
    z = np.load(src(rel))
    c, f, cm = z["counts"], z["filled"], z["cm_mask"]
    assert (f > 0).all() and (c.sum(-1) == f).all() and cm.sum() == 5
    return c, c[..., cm].sum(-1).astype(np.int64), f.astype(np.int64), [str(d) for d in z["dirs"][cm]]


def main():
    v = json.loads(src(f"{M1}/verdict_m1.json").read_text())
    th = json.loads(src(f"{M1}/thresholds.json").read_text())
    k0 = json.loads(src(f"{M1}/banks/k0-01/k0/m1/metrics.json").read_text())["K0"]
    C, shift = v["exploratory"]["C"], v["exploratory"]["shift_from_start"]

    raw, slots, cm_dirs = {}, {}, None
    for h in HARVESTS:
        raw[h], slots[h], filled, d = load(f"{M1}/banks/{h}/m1/counts.npz")
        assert (filled == SLOTS).all() and cm_dirs in (None, d)
        cm_dirs = d
    L = slots["start_a"].shape[0]
    band = (np.arange(L) * 4) // L                       # m1.band_index: layers 0-7, 8-15, 16-23, 24-31
    sides_per_band = [int((band == b).sum()) * slots["start_a"].shape[1] * 2 for b in range(4)]

    def bmeans(x):                                       # m1.band_means: mean over the band's sides
        return [float(x[band == b].mean()) for b in range(4)]

    harvests = {}
    for h, name in HARVESTS.items():
        share = slots[h] / SLOTS
        means = bmeans(share)
        harvests[PUBLIC[h]] = {
            "name": name,
            "band_mean_share": means,
            "check": [compare(f"band mean {h} band {b}", means[b], C[h][b], f"verdict_m1.json exploratory.C.{h}[{b}]", 1e-12)
                      for b in range(4)],
            "sides_by_code_and_math_slots": [np.bincount(slots[h][band == b].ravel(), minlength=SLOTS + 1).tolist()
                                             for b in range(4)]}

    shifts = {}
    for m in MIXES:
        d = slots[m] - slots["start_a"]
        means = bmeans(d / SLOTS)
        shifts[PUBLIC[m]] = {
            "band_mean_shift": means,
            "band_mean_shift_in_slots": [x * SLOTS for x in means],
            "range_banked": [shift[m][b][1:] for b in range(4)],
            "check": [compare(f"shift {m} - start band {b}", means[b], shift[m][b][0],
                              f"verdict_m1.json exploratory.shift_from_start.{m}[{b}][0]", 1e-12) for b in range(4)],
            "sides_by_change_in_slots": {"from": -SLOTS, "counts": [
                np.bincount(d[band == b].ravel() + SLOTS, minlength=2 * SLOTS + 1).tolist() for b in range(4)]}}

    # the identity pair: the start harvested twice
    differ = int((raw["start_a"] != raw["start_b"]).any(-1).sum())
    n_sides = int(np.prod(slots["start_a"].shape))
    identity = {"sides": n_sides, "sides_whose_counts_differ": differ,
                "check": compare("identity pair: share of sides whose counts differ in any directory", differ / n_sides,
                                 v["G_ID"]["changed_pole_share"], "verdict_m1.json G_ID.changed_pole_share", 0)}

    # the drift between neighbouring stage-1 snapshots, and the bar
    refs = {}
    for r in REFS:
        _, s_r, f_r, d = load(f"{M1}/banks/refs/{r}/m1/counts.npz")
        assert d == cm_dirs
        refs[r] = bmeans(s_r / f_r)                      # m1.share: slots / filled (t15 has two sides with fewer than 50)
    ref_checks = [compare(f"reference {r} band {b}", refs[r][b], th["C_refs"][r][b], f"thresholds.json C_refs.{r}[{b}]", 1e-12)
                  for r in REFS for b in range(4)]
    tau = [max(abs(refs[y][b] - refs[x][b]) for x, y in REF_PAIRS) for b in range(4)]
    bar = {"band_bar": v["bar"],
           "check": [compare(f"drift bar band {b}", tau[b], th["tau"][b], f"thresholds.json tau[{b}]", 1e-12) for b in range(4)],
           "snapshot_pairs": [list(p) for p in REF_PAIRS],
           "band_mean_share_at_snapshots": {r: refs[r] for r in REFS},
           "check_snapshots": ref_checks,
           "cross_device_spread": th["eps_dev"]}
    assert v["bar"] == th["tau"] == v["thresholds"]["tau"]

    lead = [{"difference_65_minus_none": x["mean"], "range": [x["lo"], x["hi"]]} for x in v["lead"]]
    between = [{"difference_40_minus_none": x["R_minus_G"], "difference_65_minus_40": x["M_minus_R"]} for x in v["between"]]

    g_slots = shifts["no_code_or_math"]["band_mean_shift_in_slots"]
    m_slots = shifts["code_and_math_65"]["band_mean_shift_in_slots"]
    data = {
        "bands": [{"layers": [8 * b, 8 * b + 7], "sides": sides_per_band[b]} for b in range(4)],
        "slots_per_side": SLOTS,
        "code_and_math_directories": cm_dirs,
        "harvests": harvests,
        "shift_from_start": shifts,
        "drift_bar": bar,
        "identity_pair": identity,
        "difference_between_mixes": {"code_and_math_65_minus_none": lead, "code_and_math_40_between": between},
        "earlier_snapshot_reproduced": {"activations_checked": k0["values"]["n"], "exact_share": k0["values"]["exact_share"],
                                        "positions_in_bank_share": k0["subset"]["share"]},
    }
    assert sides_per_band == [176128] * 4 and n_sides == 704512 and differ == 0
    print(f"NOTE no code or math, in slots of fifty: {[round(-x, 4) for x in g_slots]}")
    print(f"NOTE 65 % mix, in slots of fifty: {[round(x, 4) for x in m_slots]}; largest |.| {max(abs(x) for x in m_slots):.4f}")
    for num, key in [("0.0082, 0.0066, 0.0023, 0.0035", "drift_bar.band_bar"),
                     ("-0.0113, -0.0170, -0.0173, -0.0161", "shift_from_start.no_code_or_math.band_mean_shift"),
                     ("0.56-0.87 of one", "shift_from_start.no_code_or_math.band_mean_shift_in_slots"),
                     ("within 0.36 of one", "shift_from_start.code_and_math_65.band_mean_shift_in_slots"),
                     ("176,128 sides per band", "bands[*].sides"), ("0 of 704,512 differ", "identity_pair"),
                     ("1,535 of 1,535", "earlier_snapshot_reproduced.activations_checked")]:
        print(f"PLAN {num} -> {key}")

    notes = {
        "harvest_names": "Banked harvest keys -> public keys: " + "; ".join(f"{k} -> {PUBLIC[k]}" for k in HARVESTS) +
                         ". Mixes (M1_ASSETS.md line 41): gen-mc-only = 0 %, round5 = 40 %, math-code-reasoning-web = 65 % "
                         "code and math, each 100 billion tokens (47,684 steps x 2,097,152 tokens) from stage1-step476838.",
        "side": "A unit's positive side or negative side; 32 layers x 11,008 units x 2 = 704,512 sides (banked as 'poles').",
        "band_mean_share": "Per band of 8 layers, the mean over its 176,128 sides of (the side's fifty strongest corpus positions "
                           "that fall in the five code-and-math directories) / 50 (M1_SPEC.md §0.5), recomputed from "
                           "counts.npz and stored beside verdict_m1.json exploratory.C.",
        "sides_by_code_and_math_slots": "Per band, how many sides have 0, 1, ..., 50 of their fifty positions in code-and-math "
                                        "directories (index = slots). Every side has all 50 slots filled.",
        "band_mean_shift": "Per band, mean over sides of (share after the mix - share at the start, first harvest), paired by "
                           "side; band_mean_shift_in_slots is the same times 50, in slots of fifty.",
        "range_banked": "The 95 % bootstrap range over units (both sides of a unit resampled together, 2,000 resamples) as banked "
                        "in verdict_m1.json exploratory.shift_from_start; not recomputed here.",
        "sides_by_change_in_slots": "Per band, how many sides changed by -50, ..., +50 code-and-math slots from the start "
                                    "(index 0 = -50).",
        "drift_bar": "The bar per band = the largest band-mean change of the share between neighbouring stage-1 snapshots "
                     "(t12-t13, t13-t14b, t14c-t15; banked as 'tau', M1_SPEC.md §3.2); recomputed from the snapshots' "
                     "counts.npz. cross_device_spread is the same weights read on two devices (banked 'eps_dev').",
        "identity_pair": "The start's weights harvested twice on two instances; sides whose counts differ in any of the 50 "
                         "directories (banked as G_ID.changed_pole_share).",
        "difference_between_mixes": "As banked in verdict_m1.json: 'lead' = 65 % mix minus no code or math per band (mean, range); "
                                    "'between' = 40 % minus none and 65 % minus 40 % per band, each [mean, low, high]. "
                                    "Result labels are not carried.",
        "earlier_snapshot_reproduced": "The check run before the harvests: 1,535 banked activations of the end-of-stage-1 "
                                       "snapshot recomputed, share bit-identical, and share of positions found in the bank "
                                       "(banks/k0-01/k0/m1/metrics.json K0).",
        "plan_rounding": "The plan's 'within 0.36 of one' is a bound: the largest band shift of the 65 % mix is 0.3519 slots "
                         "(20_ABSTRACT_CHECK.md E2: 0.35 is nearer).",
    }
    write(data, notes)


if __name__ == "__main__":
    main()
