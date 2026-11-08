"""CARREL pure core — weight-transplant bookkeeping + frozen decision rules (no weights, no torch).

Spec: studies/lineage/cross-lineage/research/2026-09-02_carrel_spec.md. Pure python/numpy;
asserts; `python -m infra.pg.carrel_core` is the weight-free dry-run gate (must pass before
launch). The job (infra/aws_pg/jobs/carrel_transplant.py) owns the tensor copies; this module
owns WHICH parameter names a config touches and HOW the numbers map to verdicts.
"""
import re

import numpy as np

from infra.pg.broca_core import restoration_fraction, threshold

# ---- Gemma-2 decoder layer anatomy (relative to `layers.{i}.`) ----
LAYER_ATTN = ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
              "self_attn.o_proj.weight", "input_layernorm.weight",
              "post_attention_layernorm.weight")
LAYER_MLP = ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
             "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight")
N_LAYERS = 26
_LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")

# ---- frozen roster (order = execution order: sanity -> controls -> contrast -> resolution) ----
# spec: layers = inclusive layer list; modules = which halves of each layer; head = embed +
# lm_head + final norm. `all` = every LM parameter from healthy (must equal BEFORE exactly).
CONFIGS = [
    ("none",           {"layers": [],                "modules": ("attn", "mlp"), "head": False}),
    ("all",            {"layers": list(range(26)),   "modules": ("attn", "mlp"), "head": True}),
    ("ctrl_0_15",      {"layers": list(range(0, 16)), "modules": ("attn", "mlp"), "head": False}),
    ("ctrl_2_8",       {"layers": list(range(2, 9)),  "modules": ("attn", "mlp"), "head": False}),
    ("ctrl_6_12",      {"layers": list(range(6, 13)), "modules": ("attn", "mlp"), "head": False}),
    ("band_16_22",     {"layers": list(range(16, 23)), "modules": ("attn", "mlp"), "head": False}),
    ("band_16_25",     {"layers": list(range(16, 26)), "modules": ("attn", "mlp"), "head": False}),
    ("band_19_23",     {"layers": list(range(19, 24)), "modules": ("attn", "mlp"), "head": False}),
    ("band_14_24",     {"layers": list(range(14, 25)), "modules": ("attn", "mlp"), "head": False}),
] + [
    (f"solo_{l}",      {"layers": [l],                "modules": ("attn", "mlp"), "head": False})
    for l in range(16, 24)
] + [
    ("attn_16_22",     {"layers": list(range(16, 23)), "modules": ("attn",),      "head": False}),
    ("mlp_16_22",      {"layers": list(range(16, 23)), "modules": ("mlp",),       "head": False}),
    ("head",           {"layers": [],                "modules": (),              "head": True}),
    # exploratory (not in any verdict): [16,25] + final norm == BROCA's converse graft at L15,
    # so R(tail_norm_16_25) must equal 1 - D(15) (D(14)=D(16)=1.02 in the BROCA artifact).
    ("tail_norm_16_25", {"layers": list(range(16, 26)), "modules": ("attn", "mlp"), "head": True}),
]
CONTROL_NAMES = ("ctrl_0_15", "ctrl_2_8", "ctrl_6_12")
BAND_NAMES = ("band_16_22", "band_16_25", "band_19_23", "band_14_24")
SOLO_NAMES = tuple(f"solo_{l}" for l in range(16, 24))
NAILED_BAR = 0.6          # frozen: R >= 0.6 = "restored"
CONTROL_CEILING = 0.30    # frozen: a control that restores > half the bar voids "nailed"
VISION_MIN_RANGE = 0.15   # frozen (e4_capability GATE_MIN_RANGE): AFTER-BEFORE below this = VOID
RETENTION_BAR = 0.8       # frozen: "repair without loss" iff retained fraction >= 0.8
CI_MAX_DEGENERATE = 0.10  # frozen: CI VOID when > 10% of bootstrap draws have lost margin <= 0
# Replication gate: healthy layers [0,15] inside the broken model IS BROCA's prefix-graft at L15
# (same weights, same inputs, same kernels), so R(ctrl_0_15) must reproduce the artifact value
# broca_metrics.json curve["15"].acc_patched = 91/285 -> R = 11/54, within +-2 items.
BROCA_R15_ITEMS, BROCA_LOST_ITEMS, REPL_TOL_ITEMS = 11, 54, 2


def replication_check(R_ctrl_0_15: float) -> dict:
    target = BROCA_R15_ITEMS / BROCA_LOST_ITEMS
    tol = REPL_TOL_ITEMS / BROCA_LOST_ITEMS
    return {"target_R": target, "observed_R": float(R_ctrl_0_15), "tol_R": tol,
            "ok": bool(abs(R_ctrl_0_15 - target) <= tol + 1e-9)}


def partition_keys(full_keys, lm_prefix: str) -> dict:
    """Classify every state_dict key of the FULL VLM. Keys under `lm_prefix` (e.g.
    'model.language_model.') are the LM; each must be a layer key, the final norm, or the
    embedding — anything else under the prefix is a FAIL (unknown anatomy). `lm_head.weight`
    is found anywhere (top-level in transformers>=4.52, under the LM before). Returns
    {'layers': {i: {'attn': [...], 'mlp': [...]}}, 'final_norm': [k], 'embed': [k],
    'lm_head': [k...], 'other': [...]} with every layer complete (6 attn + 5 mlp keys)."""
    assert lm_prefix.endswith(".")
    layers: dict = {}
    final_norm, embed, lm_head, other = [], [], [], []
    for k in full_keys:
        if k.endswith("lm_head.weight"):
            lm_head.append(k)
            continue
        if not k.startswith(lm_prefix):
            other.append(k)
            continue
        rel = k[len(lm_prefix):]
        if rel == "norm.weight":
            final_norm.append(k)
            continue
        if rel == "embed_tokens.weight":
            embed.append(k)
            continue
        m = _LAYER_RE.match(rel)
        assert m, f"unknown LM key (anatomy drift): {k}"
        i, sub = int(m.group(1)), m.group(2)
        d = layers.setdefault(i, {"attn": [], "mlp": []})
        if sub in LAYER_ATTN:
            d["attn"].append(k)
        elif sub in LAYER_MLP:
            d["mlp"].append(k)
        else:
            raise AssertionError(f"unknown layer sub-key (anatomy drift): {k}")
    assert sorted(layers) == list(range(N_LAYERS)), f"layers found: {sorted(layers)}"
    for i, d in layers.items():
        assert len(d["attn"]) == len(LAYER_ATTN) and len(d["mlp"]) == len(LAYER_MLP), \
            f"layer {i} incomplete: attn={d['attn']} mlp={d['mlp']}"
    assert len(final_norm) == 1, final_norm
    assert len(embed) == 1, embed
    assert len(lm_head) <= 1, lm_head
    return {"layers": layers, "final_norm": final_norm, "embed": embed, "lm_head": lm_head,
            "other": other}


def config_keys(part: dict, spec: dict) -> list:
    """The sorted parameter-name set one config copies healthy -> broken."""
    ks = []
    for l in spec["layers"]:
        for mod in spec["modules"]:
            ks += part["layers"][l][mod]
    if spec["head"]:
        ks += part["embed"] + part["lm_head"] + part["final_norm"]
    assert len(ks) == len(set(ks))
    return sorted(ks)


def lm_keys(part: dict) -> list:
    """Every LM parameter name (layers + final norm + embed + lm_head) — the snapshot universe."""
    ks = []
    for l in sorted(part["layers"]):
        ks += part["layers"][l]["attn"] + part["layers"][l]["mlp"]
    return sorted(ks + part["final_norm"] + part["embed"] + part["lm_head"])


# ---- statistics ----
def boot_R(per_item_patched, per_item_after, per_item_before, n_boot=2000, seed=1337) -> dict:
    """Paired item-bootstrap of the restoration fraction. All three per-item vectors are
    aligned by position (identical item order asserted upstream via per_item_hash). The POINT
    estimate asserts a positive lost margin; bootstrap draws whose resampled lost margin is
    <= 1e-9 are DEGENERATE (R undefined) — dropped and counted; the CI is VOID when more than
    CI_MAX_DEGENERATE of the draws are degenerate (instrument resolution too coarse for a CI)."""
    p = np.asarray(per_item_patched, float); a = np.asarray(per_item_after, float)
    b = np.asarray(per_item_before, float)
    assert p.shape == a.shape == b.shape and p.ndim == 1 and p.size > 0
    R = restoration_fraction(p.mean(), a.mean(), b.mean())
    rng = np.random.default_rng(seed)
    n = p.size
    draws, n_degen = [], 0
    for t in range(n_boot):
        idx = rng.integers(0, n, n)
        lost = b[idx].mean() - a[idx].mean()
        if lost <= 1e-9:
            n_degen += 1
            continue
        draws.append((p[idx].mean() - a[idx].mean()) / lost)
    frac = n_degen / n_boot
    void = bool(frac > CI_MAX_DEGENERATE) or not draws
    return {"R": float(R), "n": int(n), "n_boot": int(n_boot), "n_boot_degenerate": int(n_degen),
            "frac_degenerate": float(frac), "ci_void": void,
            "lo": None if void else float(np.percentile(draws, 2.5)),
            "hi": None if void else float(np.percentile(draws, 97.5))}


def null_band(R_by_name: dict) -> float:
    """Frozen: band = max |R| over the three control configs."""
    assert all(c in R_by_name for c in CONTROL_NAMES), sorted(R_by_name)
    return float(max(abs(R_by_name[c]) for c in CONTROL_NAMES))


def best_band(R_by_name: dict) -> str:
    """Frozen selector for the cloze/vision follow-ups: max R over the four band configs;
    ties -> narrowest, then earliest start."""
    def width(name):
        lo, hi = name.split("_")[1:]
        return int(hi) - int(lo), int(lo)
    return max(BAND_NAMES, key=lambda n: (R_by_name[n], -width(n)[0], -width(n)[1]))


def carriers(R_by_name: dict, T: float) -> list:
    """Frozen: a single layer is a carrier iff its solo R >= T."""
    return [int(n.split("_")[1]) for n in SOLO_NAMES if R_by_name[n] >= T]


def carrel_verdict(R_by_name: dict, sanity_ok: bool) -> dict:
    """Frozen mapping (spec §6). The literal instruction 'all three controls <= T' cannot fail
    (T >= band >= every control), so it is reported but the binding control clause is the
    ceiling: band <= CONTROL_CEILING AND R(band_16_22) >= T. VOID if the sanity identities
    (none == AFTER, all == BEFORE) failed."""
    band = null_band(R_by_name)
    T = threshold(band)
    r_band = R_by_name["band_16_22"]; r_tail = R_by_name["band_16_25"]
    literal_controls_ok = all(abs(R_by_name[c]) <= T for c in CONTROL_NAMES)   # vacuous by construction
    controls_ok = band <= CONTROL_CEILING
    if not sanity_ok:
        v = "VOID (sanity identity failed: none != AFTER or all != BEFORE)"
    elif r_band >= NAILED_BAR and controls_ok and r_band >= T:
        v = "lesion nailed at weight level (band [16,22] restores >= 0.6; controls quiet)"
    elif r_band >= NAILED_BAR and not controls_ok:
        v = "VOID-CONTROLS (band restores but an outside-lesion control restores > ceiling)"
    elif r_band < NAILED_BAR and r_tail >= NAILED_BAR:
        v = "band-plus-tail needed ([16,22] < 0.6 but [16,25] >= 0.6)"
    else:
        v = "distributed (neither [16,22] nor [16,25] reaches 0.6)"
    return {"verdict": v, "null_band": band, "T": T, "R_band_16_22": r_band,
            "R_band_16_25": r_tail, "controls_within_ceiling": controls_ok,
            "literal_controls_le_T": literal_controls_ok, "carriers": carriers(R_by_name, T),
            "best_band": best_band(R_by_name)}


def retention(v_hybrid: float, v_after: float, v_before: float) -> dict:
    """Skill retention: hybrid's vision metric as a fraction of (AFTER - BEFORE). VOID when the
    readout has no dynamic range (AFTER - BEFORE < VISION_MIN_RANGE)."""
    rng = v_after - v_before
    if rng < VISION_MIN_RANGE:
        return {"fraction": None, "void": True, "range": float(rng)}
    f = (v_hybrid - v_before) / rng
    return {"fraction": float(f), "void": False, "range": float(rng),
            "repair_without_loss": bool(f >= RETENTION_BAR)}


def boot_retention(s_hybrid, s_after, s_before, n_boot=2000, seed=1337) -> dict:
    h = np.asarray(s_hybrid, float); a = np.asarray(s_after, float); b = np.asarray(s_before, float)
    assert h.shape == a.shape == b.shape and h.ndim == 1
    rng = np.random.default_rng(seed); n = h.size
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r = a[idx].mean() - b[idx].mean()
        if r >= VISION_MIN_RANGE:
            draws.append((h[idx].mean() - b[idx].mean()) / r)
    out = retention(h.mean(), a.mean(), b.mean())
    out["n_boot_valid"] = len(draws)
    if draws:
        out["lo"] = float(np.percentile(draws, 2.5)); out["hi"] = float(np.percentile(draws, 97.5))
    return out


# ---- dry-run ----
def _fake_keys(layout: str):
    """Synthetic full-VLM key lists for both transformers layouts."""
    if layout == "new":     # transformers >= 4.52: model.language_model = Gemma2Model, lm_head top-level
        p, head = "model.language_model.", "lm_head.weight"
        other = ["model.vision_tower.x.weight", "model.multi_modal_projector.linear.weight"]
    else:                   # older: language_model = Gemma2ForCausalLM
        p, head = "language_model.model.", "language_model.lm_head.weight"
        other = ["vision_tower.x.weight", "multi_modal_projector.linear.weight"]
    ks = list(other) + [head, p + "norm.weight", p + "embed_tokens.weight"]
    for i in range(N_LAYERS):
        ks += [f"{p}layers.{i}.{s}" for s in LAYER_ATTN + LAYER_MLP]
    return ks, p


def _dryrun():
    for layout in ("new", "old"):
        ks, p = _fake_keys(layout)
        part = partition_keys(ks, p)
        cfg = dict(CONFIGS)
        assert len(config_keys(part, cfg["none"])) == 0
        assert len(config_keys(part, cfg["all"])) == 26 * 11 + 3
        assert len(config_keys(part, cfg["band_16_22"])) == 7 * 11
        assert len(config_keys(part, cfg["ctrl_0_15"])) == 16 * 11
        assert len(config_keys(part, cfg["ctrl_2_8"])) == len(config_keys(part, cfg["ctrl_6_12"])) == 7 * 11
        assert len(config_keys(part, cfg["attn_16_22"])) == 7 * 6
        assert len(config_keys(part, cfg["mlp_16_22"])) == 7 * 5
        assert len(config_keys(part, cfg["head"])) == 3
        assert len(config_keys(part, cfg["solo_21"])) == 11
        assert len(config_keys(part, cfg["tail_norm_16_25"])) == 10 * 11 + 3
        assert set(config_keys(part, cfg["attn_16_22"])) | set(config_keys(part, cfg["mlp_16_22"])) \
            == set(config_keys(part, cfg["band_16_22"]))
        assert len(lm_keys(part)) == 26 * 11 + 3 and len(part["other"]) == 2
    # anatomy drift must fail loud
    ks, p = _fake_keys("new")
    try:
        partition_keys(ks + [p + "layers.3.self_attn.q_norm.weight"], p); raise SystemExit("no raise")
    except AssertionError:
        pass
    try:
        partition_keys([k for k in ks if "layers.7.mlp.up_proj" not in k], p); raise SystemExit("no raise")
    except AssertionError:
        pass
    # bootstrap arithmetic on the real pair sizes (285 items, 134 vs 80 correct)
    b = np.r_[np.ones(134), np.zeros(151)]; a = np.r_[np.ones(80), np.zeros(205)]
    r = boot_R(b, a, b, n_boot=200); assert abs(r["R"] - 1.0) < 1e-12 and r["lo"] <= 1.0 <= r["hi"]
    r = boot_R(a, a, b, n_boot=200); assert abs(r["R"]) < 1e-12 and r["frac_degenerate"] == 0.0 and not r["ci_void"]
    # cloze-like margins: 21 lost items (126 vs 105) -> runs, degenerate draws counted; 1 lost item
    # (81 vs 80) -> many degenerate draws -> CI VOID, point estimate kept; 0 lost -> point asserts
    b21 = np.r_[np.ones(126), np.zeros(159)]; a21 = np.r_[np.ones(105), np.zeros(180)]
    r = boot_R(a21, a21, b21, n_boot=500); assert abs(r["R"]) < 1e-12 and 0 <= r["frac_degenerate"] < 1
    b1 = np.r_[np.ones(81), np.zeros(204)]; a1 = np.r_[np.ones(80), np.zeros(205)]
    r = boot_R(b1, a1, b1, n_boot=500); assert abs(r["R"] - 1.0) < 1e-12 and r["ci_void"] and r["lo"] is None
    try:
        boot_R(a1, a1, a1, n_boot=10); raise SystemExit("no raise")
    except AssertionError:
        pass
    # verdict replays: every branch reachable
    base = {c: 0.02 for c in CONTROL_NAMES}; base.update({n: 0.0 for n in SOLO_NAMES})
    nailed = dict(base, band_16_22=0.9, band_16_25=1.0, band_19_23=0.5, band_14_24=0.95, solo_21=0.3)
    v = carrel_verdict(nailed, True); assert v["verdict"].startswith("lesion nailed"), v
    assert v["T"] == 0.10 and v["carriers"] == [21] and v["best_band"] == "band_16_25", v
    tail = dict(base, band_16_22=0.3, band_16_25=0.8, band_19_23=0.1, band_14_24=0.4)
    v = carrel_verdict(tail, True); assert v["verdict"].startswith("band-plus-tail"), v
    dist = dict(base, band_16_22=0.1, band_16_25=0.2, band_19_23=0.0, band_14_24=0.3)
    v = carrel_verdict(dist, True); assert v["verdict"].startswith("distributed"), v
    leaky = dict(nailed, ctrl_0_15=0.5)     # control above ceiling -> VOID-CONTROLS, literal rule still 'ok'
    v = carrel_verdict(leaky, True); assert v["verdict"].startswith("VOID-CONTROLS") and v["literal_controls_le_T"], v
    v = carrel_verdict(nailed, False); assert v["verdict"].startswith("VOID (sanity"), v
    # inflated-but-legal band: ctrl_0_15 at BROCA's R(15)=0.204 -> T=0.407, still nailed at 0.9
    infl = dict(nailed, ctrl_0_15=0.204); v = carrel_verdict(infl, True)
    assert v["verdict"].startswith("lesion nailed") and abs(v["T"] - 0.408) < 1e-9 and v["carriers"] == [], v
    # replication gate: 11/54 passes, 13/54 passes (edge), 14/54 fails
    assert replication_check(11 / 54)["ok"] and replication_check(13 / 54)["ok"]
    assert not replication_check(14 / 54)["ok"] and not replication_check(0.0)["ok"]
    # best_band tie-break: equal R -> narrowest, then earliest
    tie = dict(base, band_16_22=0.7, band_16_25=0.7, band_19_23=0.7, band_14_24=0.7)
    assert best_band(tie) == "band_19_23"
    # retention: VOID without range; fraction otherwise; bootstrap
    assert retention(0.2, 0.28, 0.0)["fraction"] == 0.2 / 0.28
    assert retention(0.2, 0.1, 0.0)["void"]
    h = np.r_[np.ones(50), np.zeros(150)]; af = np.r_[np.ones(56), np.zeros(144)]; bf = np.zeros(200)
    br = boot_retention(h, af, bf, n_boot=200); assert not br["void"] and "lo" in br and br["repair_without_loss"]
    print("CARREL core dry-run: ALL PASS")


if __name__ == "__main__":
    _dryrun()
