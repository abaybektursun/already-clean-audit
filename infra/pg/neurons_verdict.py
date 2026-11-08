"""D1 verdict (SPEC_D1 §2.3, §2.4, §5, §6, §7, §11; DATA_CONTRACT §4): pure functions plus a CLI.

Reads only banked files: harvest `out/` directories (DATA_CONTRACT §2), explain directories (§3) and
probe scores. Writes `<out>/verdict/`.

Subcommands:
    theta       θ_s from the discovery set's shuffled-pole scores (§6.1) -> verdict/theta.json.
    draw-s      S and S_gate for the W3 grid (§4.4) from the audit and extra pole lists, excluding RARE
                poles by the harvest's stats only -> draw/S.json, S_gate.json, grid_poles.json, draw.json.
    discovery   the discovery set's report for one prompt version (§9.2 W2-D): pass rate, classes,
                strong rates, X3, X7, prompt SHA-256s, models and batch counts -> verdict/discovery.json
                and verdict/discovery_samples.jsonl. No timing gates (θ_s is set on this same set).
                Exit 3 when the θ_s gate fails (θ_s > 0.5: W1-C does not start; θ_s <= 0: no readout).
    judge-gate  positive-control accuracy on the development discovery set (§6.3 validity gate)
                -> verdict/judge_gate.json. Requires every planned control judgment.
    unblind     Q1 classes per model and the Q2 readouts on S -> verdict/metrics.json,
                verdict/grid.md (§11), verdict/samples/. Refuses (exit 2, naming the file) unless the
                PREDICTIONS gate passes (`predictions_gate`).

Rules implemented here, by function:
    score, theta_s, theta_gate, explanation_passes, essay_rule, is_rare, job_supported,
    assign_class, wilson, cohen_kappa, single_token_suppressor            (Q1, §2.3, §6.1, §6.2)
    signed_magnitude, presence, birth, late_flags, cluster_table           (Q2 births, §2.4, B3, B4)
    p_u_value, p_u_bootstrap, p_u_rule, pooled_share, birth_order_test,
    vestigial_counts, vestigial, stratum_weights, holm                     (§6.3-§6.6, §7, §4.4)
    predictions_gate                                                       (§9.3)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

from infra.pg.neurons_llm import (backend_sidecar_path, decline_text, declined_path, declined_rows, read_jsonl,
                                  rejected_path, requests_path, response_models, responses_path, utc_now,
                                  write_jsonl_new)

GRID_STEPS = (0, 1000, 2000, 3000, 5000, 8000, 12000, 20000, 35000, 60000, 100000, 180000,
              320000, 560000, 1000000, 1413814)                       # §4.3, T_0 .. T_15
GRID_LABELS = tuple(f"T_{i}" for i in range(len(GRID_STEPS)))
T_FIRST_TRAINED = 1        # births at T_1 have only T_0 (random init) before them (§2.4)
T_LATE_HALF = 12           # T_12 = step 320k, start of the growth window (§2.4)
T_CONTROL_BEFORE = 14      # T_14, the positive control's earlier record (§2.4)
K_TOPK = 50                # DATA_CONTRACT §2: top-K, and the denominator of the double-negative share
RARE_DENSITY = 50          # §2.3
THETA_PERCENTILE = 99      # §6.1
THETA_CEILING = 0.5        # §6.1: above it the corpus is too small; W1-C does not start
THETA_MIN_SCORES = 100     # a 99th percentile needs at least 100 values
THETA_BAND = 0.1           # §5 width control: classes at θ_s ± 0.1
ESSAY_STRONG = 0.30        # the essay's strong threshold (essay.md 637), reported only
RAW_PRESENCE = 0.3         # §2.4 raw-ratio presence reading
CTRL_GATE = 0.80           # §5 / §6.3 validity gate
CTRL_PLAN = {"random": 4, "hard": 2}   # judgments per control and per late-born cluster (§2.4)
PU_RG_UPPER = 0.10         # §6.3
PU_SP_LOWER = 0.20         # §6.3
VEST_LOWER = 0.5           # §6.5
G2_LINE = 0.10             # §5 G2: confident explanations of T_0 poles in at most 10% of poles
DOUBLE_NEGATIVE_SPLIT = 0.8  # §7 stability split
MIN_CLUSTERS_POWERED = 200   # §7
S_SIZE = 600               # §4.4 (as amended at W0): the development sample S
CENTER_NOT_COMPUTED = "not computed: W3 scores T_15 only"   # §2.4 center birth curve needs probe scores per grid point
CENTER_RATIO = 0.3         # §2.4 center birth curve: the first grid point at which the blind maximum reaches 0.3 of its T_15 score
CENTER_MONOTONE_TOL = 0.05  # exploratory: a ratio curve counts as non-decreasing when no step drops by more than this
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)   # F3 distributions over the grid
GALLERY_GROUPS = (("ONE",), ("TWO-REL",), ("UNREL", "UNK"))   # §10 F6: one pole per group, by its Q1 class
UNTESTABLE_PHRASE = "cannot be reproduced in a short text"   # the wording explainer.md asks for an untestable job
N_BOOT = 10_000
DEGENERATE_LIMIT = 0.01    # p_u: redrawn draws allowed, as a share of the bootstrap draws
N_PERM = 10_000
SENSITIVITY_EXCLUDED_BIRTHS = (7, 15)   # checker F2 item 3 (checks/W1-W3.md): births at the device boundaries T_7 and T_15
CLASSES = ("ONE", "TWO-REL", "UNREL", "UNK", "RARE")
UNK_PARTS = ("UNK-N", "UNK-F")   # §2.3 as amended: explainer said UNK / an explanation was given and failed
BLIND_MAX_CENTER = 1.0           # X3: a blind maximum at or above the corpus max
TEST_KINDS = ("pos", "near", "distant")
STATUSES = ("echo", "never_born", "earliest", "formed_before_step_1000", "readings_disagree",
            "unmatched_in_sample_1", "late_primary")
READOUTS = ("p_u", "b", "v", "H_MIG", "hard-decoy")          # §11 rows the PREDICTIONS entry must name
JOURNAL_SUFFIX = ("studies", "already-clean", "JOURNAL.md")
HARVEST_FILES = {"manifest": "manifest.json", "poles": "poles/poles.json", "vocab": "poles/vocab.json",
                 "exemplar": "poles/exemplar_a.npz", "probe": "poles/probe/scores.npz"}
POLE_RE = re.compile(r"^L(\d{2,})N(\d+)([+-])$")
UTC = timezone.utc


# --- banked files ----------------------------------------------------------------------------

def parse_pole(pole: str) -> tuple[int, int, int]:
    """'L12N4051+' -> (12, 4051, 0); sign index 0 = positive pole, 1 = negative pole."""
    m = POLE_RE.match(pole)
    if m is None:
        raise ValueError(f"not a pole id: {pole!r}")
    return int(m.group(1)), int(m.group(2)), 0 if m.group(3) == "+" else 1


def pole_id(layer: int, neuron: int, sign_index: int) -> str:
    return f"L{layer:02d}N{neuron}{'+-'[sign_index]}"


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_npz(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@lru_cache(maxsize=64)
def layer_file(harvest: str, kind: str, layer: int) -> dict:
    """`<harvest>/<kind>/layer_XX.npz` for kind in topk, stats, snapshots/band1/stats, ... At most 64 files
    are kept: an O7 layer is 34.6 MiB of stats and 21.0 MiB of topk in memory, so a 16-point grid of 32
    layers held whole would need about 28 GiB. Callers walk poles in id order (layer by layer), which
    needs at most 2 x 16 files at a time."""
    return load_npz(Path(harvest) / kind / f"layer_{layer:02d}.npz")


def pole_stats(harvest, pole: str, kind: str = "stats") -> dict:
    layer, neuron, s = parse_pole(pole)
    st = layer_file(str(harvest), kind, layer)
    sc = st["sign_counts"][neuron, s].astype(int).tolist()
    return {"max_abs": float(st["max_abs"][neuron, s]), "q99": float(st["q99"][neuron, s]),
            "q999": float(st["q999"][neuron, s]), "q9999": float(st["q9999"][neuron, s]),
            "density": int(st["density"][neuron, s]), "n_pole_tokens": int(st["n_pole_tokens"][neuron, s]),
            "same_token_frac": float(st["top_same_token_frac"][neuron, s]), "sign_counts": sc,
            "double_negative_share": sc[3] / K_TOPK}


def pole_pos0_share(harvest, pole: str) -> float:
    """DATA_CONTRACT §8: the pre-exclusion share of the pole's top-50 at window position 0. A stats
    file without the field raises, naming the file."""
    layer, neuron, s = parse_pole(pole)
    st = layer_file(str(harvest), "stats", layer)
    if "pos0_share" not in st:
        raise ValueError(f"{Path(harvest) / 'stats' / f'layer_{layer:02d}.npz'}: no pos0_share (DATA_CONTRACT §8)")
    return float(st["pos0_share"][neuron, s])


def harvest_poles(harvest) -> list[str]:
    return read_json(Path(harvest) / HARVEST_FILES["poles"])


def probe_scores(harvest) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    """(text index, pole index, max_a [T, P], min_a [T, P]) from `poles/probe/scores.npz`."""
    z = load_npz(Path(harvest) / HARVEST_FILES["probe"])
    poles = harvest_poles(harvest)
    if z["max_a"].shape[1] != len(poles):
        raise ValueError(f"{harvest}: probe has {z['max_a'].shape[1]} poles, poles.json {len(poles)}")
    return ({str(t): i for i, t in enumerate(z["text_ids"])}, {p: j for j, p in enumerate(poles)},
            z["max_a"], z["min_a"])


def pole_vocab(vocab: dict, pole: str) -> dict:
    """The pole's own output view from `poles/vocab.json`. The harvest writes the file per pole: the "+"
    entry is W_U @ (gamma ⊙ W_down[:, i]) and the "-" entry is already its sign flip (promoted and
    suppressed swapped, signs negated). Until 2026-10-24 this function flipped the "-" entry a second
    time, so every negative pole's explainer and second-reader prompt in W2-D, W2-C and the T_0 control
    showed the positive pole's view (JOURNAL 2026-10-24, F41). The clusterer and judge prompts carry no
    vocabulary line. The file's own entry is returned; a "-" entry that is not the flip of its "+" entry
    is refused."""
    entry = vocab[pole]
    layer, neuron, sign = parse_pole(pole)
    if sign == 1:
        plus = vocab[pole_id(layer, neuron, 0)]
        if entry["promoted"][:3] != [[t, -s] for t, s in plus["suppressed"][:3]]:
            raise ValueError(f"{pole}: vocab.json '-' entry is not the sign flip of its '+' entry; the file's layout changed")
    return entry


PROBE_EXCLUDED = "text position 0"     # harvest neurons/harvest 62d865d3: the probe scores text positions >= 1


def probe_rule(probe_dir) -> dict:
    """The scoring rule a probe directory declares. Its scores.npz must carry `probe_rule` and its
    manifest `probe_excluded_positions == "text position 0"`; probe outputs from before that rule are
    refused, naming the file."""
    path = Path(probe_dir) / HARVEST_FILES["probe"]
    with np.load(path, allow_pickle=False) as z:
        if "probe_rule" not in z.files:
            raise ValueError(f"{path}: no probe_rule; probe outputs from before the text-position-0 rule are refused")
        rule = [str(x) for x in np.atleast_1d(z["probe_rule"])]
    manifest = Path(probe_dir) / HARVEST_FILES["manifest"]
    excluded = read_json(manifest).get("probe_excluded_positions")
    if excluded != PROBE_EXCLUDED:
        raise ValueError(f"{manifest}: probe_excluded_positions is {excluded!r}, need {PROBE_EXCLUDED!r}")
    return {"probe_rule": rule, "probe_excluded_positions": excluded}


def input_hashes(paths: dict) -> dict:
    return {name: {"path": str(p), "sha256": file_sha256(p)} for name, p in paths.items()}


# --- Q1: scores, θ_s, legibility, classes (§2.3, §6.1, §6.2) ---------------------------------

def score(max_a: float, min_a: float, max_abs: float, sign_index: int) -> float:
    """§2.2: max over the text of a / corpus max|a| (positive pole); min / -max|a| (negative)."""
    if not (np.isfinite(max_a) and np.isfinite(min_a)):
        raise ValueError(f"non-finite probe value (max_a {max_a}, min_a {min_a})")
    if not max_abs > 0:
        raise ValueError(f"max_abs {max_abs}: a dead pole has no score")
    return float(max_a / max_abs) if sign_index == 0 else float(min_a / -max_abs)


def theta_s(shuffled_scores) -> float:
    """§6.1: the 99th percentile (numpy linear interpolation) of the shuffled-pole scores."""
    x = np.asarray(shuffled_scores, dtype=float)
    if x.size < THETA_MIN_SCORES:
        raise ValueError(f"{x.size} shuffled scores; θ_s needs at least {THETA_MIN_SCORES}")
    if not np.all(np.isfinite(x)):
        raise ValueError("non-finite shuffled score")
    return float(np.percentile(x, THETA_PERCENTILE))


def theta_gate(theta: float) -> str:
    """pass iff 0 < θ_s <= 0.5. Above 0.5 the corpus is too small (§6.1, §14). At or below 0 a
    zero-activation text would count as strong, so the readout does not count."""
    if theta <= 0:
        return "fail_floor"
    if theta > THETA_CEILING:
        return "fail_ceiling"
    return "pass"


def n_strong(scores, threshold: float) -> int:
    return int(sum(s >= threshold for s in scores))


def _five(name, scores):
    if len(scores) != 5:
        raise ValueError(f"{name}: {len(scores)} scores, the rule needs exactly 5")


def explanation_passes(pos, near, distant, theta: float) -> bool:
    """§6.1: >= 3/5 positives strong, <= 2/5 near misses strong, <= 1/5 distant texts strong."""
    for name, x in (("pos", pos), ("near", near), ("distant", distant)):
        _five(name, x)
    return (n_strong(pos, theta) >= 3 and n_strong(near, theta) <= 2
            and n_strong(distant, theta) <= 1)


def essay_rule(pos, near) -> str:
    """The essay's reading, reported beside ours and never used to decide: strong = 0.30; win if
    strong positives outnumber strong near misses, tie if equal."""
    _five("pos", pos)
    _five("near", near)
    p, q = n_strong(pos, ESSAY_STRONG), n_strong(near, ESSAY_STRONG)
    return "win" if p > q else "tie" if p == q else "lose"


def is_rare(density: int, max_abs: float) -> bool:
    """§2.3: fewer than 50 tokens with |a| >= 0.01 max|a|, or max|a| = 0."""
    return max_abs == 0 or density < RARE_DENSITY


def job_supported(job_scores, theta: float) -> bool:
    """§2.3 UNREL support clause: >= 3 of the 5 positives written from the job's sentence alone are strong."""
    _five("job_pos", job_scores)
    return n_strong(job_scores, theta) >= 3


def assign_class(rare: bool, label: str, jobs: list[dict], passes: bool,
                 job_support: dict[int, bool]) -> str:
    """§2.3. `label`, `jobs` are Agent 1's answer; `passes` is §6.1 on its explanation;
    `job_support[k]` is the support clause for job k (needed for every unrelated job of an UNREL label).
    Unrelated jobs are the jobs with related_to_others = false."""
    if rare:
        return "RARE"
    if label == "ONE" and len(jobs) == 1 and passes:
        return "ONE"
    if (label == "TWO-REL" and 2 <= len(jobs) <= 3 and all(j["related_to_others"] for j in jobs)
            and passes):
        return "TWO-REL"
    if label == "UNREL":
        unrelated = [k for k, j in enumerate(jobs) if not j["related_to_others"]]
        if len(unrelated) >= 2 and sum(job_support[k] for k in unrelated) >= 2:
            return "UNREL"
    return "UNK"


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson 95% interval for k successes in n."""
    if n <= 0 or not 0 <= k <= n:
        raise ValueError(f"wilson({k}, {n})")
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(center - half), float(center + half)


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    """Cohen's kappa between two label lists; None when chance agreement is 1 (undefined)."""
    if len(a) != len(b) or not a:
        raise ValueError("kappa needs two equal, non-empty label lists")
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(lab) / n) * (b.count(lab) / n) for lab in labels)
    return None if pe == 1 else float((po - pe) / (1 - pe))


def single_token_suppressor(vocab: dict) -> bool:
    """§2.3 banked flag: the strongest W_out·W_U entry is negative and at least twice the second.
    `vocab` is the pole's own view (`pole_vocab`)."""
    entries = sorted([s for _, s in vocab["promoted"]] + [s for _, s in vocab["suppressed"]],
                     key=abs, reverse=True)
    return entries[0] < 0 and abs(entries[0]) >= 2 * abs(entries[1])


# --- Q2: presence, births, clusters (§2.4) ---------------------------------------------------

def signed_magnitude(a, sign_index: int) -> np.ndarray:
    """a⁺ = activation in the pole's own direction, clipped at 0 (§2.4 as amended 2026-10-19):
    max(a, 0) on the positive pole, max(-a, 0) on the negative pole."""
    s = 1.0 if sign_index == 0 else -1.0
    return np.maximum(s * np.asarray(a, dtype=float), 0.0)


def presence(med, q999, max_abs) -> tuple[np.ndarray, np.ndarray]:
    """§2.4 per checkpoint: (percentile reading med >= q99.9_t, raw reading med >= 0.3 max_t).
    Both require med > 0, so a zero median is never present (a dead pole has q999 = max = 0)."""
    med, q999, max_abs = (np.asarray(x, dtype=float) for x in (med, q999, max_abs))
    alive = med > 0
    return alive & (med >= q999), alive & (med >= RAW_PRESENCE * max_abs)


def birth(present) -> int | None:
    """§2.4: the first grid index t at which the cluster is present and stays present at the next
    grid point. At T_15 there is no next point; presence there is birth = T_15 (amended 2026-10-19)."""
    p = [bool(x) for x in present]
    for t in range(len(p)):
        if p[t] and (t == len(p) - 1 or p[t + 1]):
            return t
    return None


def late_flags(births: list[int | None]) -> list[bool]:
    """§2.4: late-born iff birth > the pole's earliest birth. An unborn cluster is not late-born."""
    defined = [b for b in births if b is not None]
    if not defined:
        return [False] * len(births)
    first = min(defined)
    return [b is not None and b > first for b in births]


def majority_matched(ranks: list[int], others: list[list[int]]) -> bool:
    """A cluster is present in the other clustering sample when one cluster there holds more than
    half of its exemplars (B4 exclusion, §8)."""
    return any(2 * len(set(ranks) & set(o)) > len(ranks) for o in others)


def cluster_table(pole: str, clusters0: list[dict], clusters1: list[dict], mag: np.ndarray,
                  q999, max_abs, q999_alt=None, max_abs_alt=None) -> list[dict]:
    """One row per sample-0 cluster of `pole`. `mag` [T, E] is the pole-direction activation at the
    pole's exemplar columns (column = rank in the T_15 top-50) at every grid point; q999 and max_abs
    are the pole's stats per grid point; `*_alt` the same on the 10M prefix (B2), optional.
    Echo clusters (§2.4 as amended: the double-negative regime) get status `echo` and take no part in
    main, late-born, matching or any count. A pole whose clusters are all echoes has no main."""
    if mag.shape[0] != len(GRID_STEPS):
        raise ValueError(f"{pole}: {mag.shape[0]} grid points, expected {len(GRID_STEPS)}")
    max_abs = np.asarray(max_abs, dtype=float)
    others = [d["exemplar_ranks"] for d in clusters1 if not d["echo"]]
    rows = []
    for c in clusters0:
        ranks = list(c["exemplar_ranks"])
        sub = mag[:, ranks]
        if not np.all(np.isfinite(sub)):
            raise ValueError(f"{pole}:c{c['c']}: non-finite exemplar activation")
        med = np.median(sub, axis=1)
        pct, raw = presence(med, q999, max_abs)
        row = {"cluster": f"{pole}:c{c['c']}", "pole": pole, "c": c["c"], "name": c["name"],
               "echo": c["echo"], "ranks": ranks, "med": med.tolist(), "present_pct": pct.tolist(),
               "present_raw": raw.tolist(), "birth_pct": birth(pct), "birth_raw": birth(raw),
               "matched": majority_matched(ranks, others), "peak_end": float(sub[-1].max()),
               "is_main": False, "late_pct": False, "late_raw": False, "r_at_birth": None,
               "growth_last_half": None, "status": "echo", "t_before": None}
        if q999_alt is not None:
            row["birth_pct_alt"] = birth(presence(med, q999_alt, max_abs_alt)[0])
        rows.append(row)
    jobs = [r for r in rows if not r["echo"]]
    if not jobs:
        return rows
    max(jobs, key=lambda r: (r["peak_end"], -r["c"]))["is_main"] = True
    late_pct = late_flags([r["birth_pct"] for r in jobs])
    late_raw = late_flags([r["birth_raw"] for r in jobs])
    late_alt = late_flags([r["birth_pct_alt"] for r in jobs]) if q999_alt is not None else None
    for i, r in enumerate(jobs):
        b = r["birth_pct"]
        r["late_pct"], r["late_raw"] = late_pct[i], late_raw[i]
        if late_alt is not None:
            r["split_changed_10m"] = late_alt[i] != late_pct[i]
        if b is not None:
            if not max_abs[b] > 0:
                raise ValueError(f"{r['cluster']}: exemplars present at {GRID_LABELS[b]} but max|a| there "
                                 f"is {max_abs[b]}; the stats and exemplar files disagree")
            r["r_at_birth"] = float(r["med"][b] / max_abs[b])
        if max_abs[T_LATE_HALF] > 0 and max_abs[-1] > 0:      # else growth is undefined (§2.4)
            r["growth_last_half"] = float(r["med"][-1] / max_abs[-1] - r["med"][T_LATE_HALF] / max_abs[T_LATE_HALF])
        if b is None:
            r["status"] = "never_born"
        elif not late_pct[i]:
            r["status"] = "earliest"
        elif b == T_FIRST_TRAINED:
            r["status"] = "formed_before_step_1000"
        elif not late_raw[i]:
            r["status"] = "readings_disagree"
        elif not r["matched"]:
            r["status"] = "unmatched_in_sample_1"
        else:
            r["status"] = "late_primary"
        r["t_before"] = b - 1 if r["status"] == "late_primary" else None
    return rows


def check_grid(grid_dirs: list) -> None:
    """The grid directories must be T_0 .. T_15 in order (each manifest's branch is stage1-step<n>)
    and scored on one corpus (equal tokens_sha256 and n_tokens_scored, §4.2)."""
    if len(grid_dirs) != len(GRID_STEPS):
        raise ValueError(f"{len(grid_dirs)} grid directories, expected {len(GRID_STEPS)}")
    first = read_json(Path(grid_dirs[0]) / HARVEST_FILES["manifest"])
    for g, step in zip(grid_dirs, GRID_STEPS):
        m = read_json(Path(g) / HARVEST_FILES["manifest"])
        if m["branch"] != f"stage1-step{step}":
            raise ValueError(f"{g}: branch {m['branch']!r}, grid position expects stage1-step{step}")
        for key in ("tokens_sha256", "n_tokens_scored"):
            if m[key] != first[key]:
                raise ValueError(f"{g}: {key} {m[key]!r} differs from {grid_dirs[0]} ({first[key]!r})")


def grid_cluster_tables(grid_dirs: list, explain: Path, b2: bool = False) -> dict[str, list[dict]]:
    """cluster_table for every pole in `explain/clusters.jsonl`, reading exemplar activations and
    stats at every grid point. Exemplar columns are the T_15 top-50 ranks (`explain/positions.npz`).
    The grid may hold more poles than this explain directory (one harvest serves S and the judge-gate
    set); every grid directory must hold the same pole list, containing all of
    `positions_poles.json`, and a pole's exemplar row is its index in that list."""
    check_grid(grid_dirs)
    order = read_json(explain / "positions_poles.json")
    grid_poles = harvest_poles(grid_dirs[0])
    for g in grid_dirs:
        here = harvest_poles(g)
        missing = [p for p in order if p not in here]
        if missing:
            raise ValueError(f"{g}: {HARVEST_FILES['poles']} lacks {missing} of {explain / 'positions_poles.json'}")
        if here != grid_poles:
            raise ValueError(f"{g}: {HARVEST_FILES['poles']} differs from {Path(grid_dirs[0]) / HARVEST_FILES['poles']}")
    row = {p: i for i, p in enumerate(grid_poles)}
    a = np.stack([load_npz(Path(g) / HARVEST_FILES["exemplar"])["a"] for g in grid_dirs])  # [T, P, E]
    if a.shape[1] != len(grid_poles):
        raise ValueError(f"exemplar_a has {a.shape[1]} rows, {HARVEST_FILES['poles']} lists {len(grid_poles)} poles")
    tables = {}
    for pole, samples in sorted(clusters_by_pole(explain).items()):
        _, _, s = parse_pole(pole)
        st = [pole_stats(g, pole) for g in grid_dirs]
        alt = [pole_stats(g, pole, "snapshots/band1/stats") for g in grid_dirs] if b2 else None
        tables[pole] = cluster_table(
            pole, samples[0], samples[1], signed_magnitude(a[:, row[pole], :], s),
            [x["q999"] for x in st], [x["max_abs"] for x in st],
            [x["q999"] for x in alt] if b2 else None, [x["max_abs"] for x in alt] if b2 else None)
    return tables


def declined_clusterer(explain: Path) -> dict[str, str]:
    """{pole: reason} for every pole with a clustering sample declined after the attempts cap
    (collect-clusters records it with `declined`). Such a pole leaves Q2 and is reported (declined rule,
    2026-10-21)."""
    out: dict[str, list[str]] = {}
    for row in read_jsonl(explain / "clusters.jsonl"):
        if "declined" in row:
            out.setdefault(row["pole"], []).append(f"sample {row['sample']}: {row['declined']}")
    return {pole: "; ".join(v) for pole, v in sorted(out.items())}


def clusters_by_pole(explain: Path) -> dict[str, dict[int, list[dict]]]:
    """{pole: {sample: clusters}} for every pole clustered twice; poles in declined_clusterer are left out."""
    gone = declined_clusterer(explain)
    by_pole: dict[str, dict[int, list]] = {}
    for row in read_jsonl(explain / "clusters.jsonl"):
        if row["pole"] not in gone:
            by_pole.setdefault(row["pole"], {})[row["sample"]] = row["clusters"]
    for pole, samples in by_pole.items():
        if set(samples) != {0, 1}:
            raise ValueError(f"{pole}: clustering samples {sorted(samples)}, need 0 and 1")
    return by_pole


# --- Q2: parent test, birth order, vestigial (§6.3-§6.6) -------------------------------------

def p_u_value(A: float, a_ctrl: float) -> float:
    """§6.3: p_u = (a_ctrl - A) / (a_ctrl - 0.5), clipped to [0, 1]."""
    if not a_ctrl > 0.5:
        raise ValueError(f"a_ctrl {a_ctrl} <= 0.5: p_u is undefined")
    return float(np.clip((a_ctrl - A) / (a_ctrl - 0.5), 0.0, 1.0))


def p_u_bootstrap(cluster_acc: list[list[float]], ctrl_acc: list[float | None],
                  n_boot: int = N_BOOT, seed: int = 0, weights=None) -> dict:
    """§6.3 with its 95% interval; p_u carries no p-value (§7 as amended). One entry per pole of S:
    cluster_acc[i] lists the per-cluster accuracies (random decoy) of pole i's late-born clusters;
    ctrl_acc[i] is pole i's positive-control accuracy. Each draw resamples poles with replacement;
    clusters move with their pole and a_ctrl is recomputed over the same resampled poles. `weights`
    (one per pole) gives the population-weighted estimate (§4.4); None weighs poles equally.
    The bootstrap is conditional: a draw that holds no late-born cluster or no control (possible
    when some poles of S have no late-born primary) is redrawn, and the number of redrawn draws is
    returned as `degenerate_draws`. More than 1% of n_boot redrawn (DEGENERATE_LIMIT) refuses."""
    if len(cluster_acc) != len(ctrl_acc) or not cluster_acc:
        raise ValueError("one entry per pole in both lists")
    w = np.ones(len(cluster_acc)) if weights is None else np.asarray(weights, dtype=float)
    S = w * np.array([sum(x) for x in cluster_acc], dtype=float)
    n = w * np.array([len(x) for x in cluster_acc], dtype=float)
    C = w * np.array([0.0 if c is None else c for c in ctrl_acc])
    m = w * np.array([c is not None for c in ctrl_acc], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(S), size=(n_boot, len(S)))
    degenerate = 0
    while True:
        bad = np.flatnonzero((n[idx].sum(1) == 0) | (m[idx].sum(1) == 0))
        if bad.size == 0:
            break
        degenerate += bad.size
        if degenerate > DEGENERATE_LIMIT * n_boot:
            raise ValueError(f"a bootstrap draw holds no late-born cluster or no control: {degenerate} "
                             f"degenerate draws exceed {DEGENERATE_LIMIT:.0%} of {n_boot}")
        idx[bad] = rng.integers(0, len(S), size=(bad.size, len(S)))
    A, a_ctrl = S.sum() / n.sum(), C.sum() / m.sum()
    n_b, m_b = n[idx].sum(1), m[idx].sum(1)
    A_b, a_b = S[idx].sum(1) / n_b, C[idx].sum(1) / m_b
    if (a_b <= 0.5).any():
        raise ValueError(f"{int((a_b <= 0.5).sum())} draws have a_ctrl <= 0.5: p_u is undefined")
    pu = np.clip((a_b - A_b) / (a_b - 0.5), 0.0, 1.0)
    lo, hi = np.percentile(pu, [2.5, 97.5])
    return {"A": float(A), "a_ctrl": float(a_ctrl), "point": p_u_value(A, a_ctrl),
            "ci95": [float(lo), float(hi)], "n_clusters": int(sum(len(x) for x in cluster_acc)),
            "n_poles": len(S), "n_boot": n_boot, "seed": seed, "degenerate_draws": degenerate}


def p_u_rule(ci95) -> str:
    """§6.3: H_RG if the upper bound < 0.10; H_SP if the lower bound > 0.20; mixed otherwise."""
    lo, hi = ci95
    if hi < PU_RG_UPPER:
        return "H_RG"
    if lo > PU_SP_LOWER:
        return "H_SP"
    return "mixed"


def pooled_share(k, n, weights=None, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, list, np.ndarray]:
    """sum(w k) / sum(w n) over poles with a 95% interval from resampling poles. Returns
    (point, [lo, hi], bootstrap draws)."""
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    w = np.ones(len(k)) if weights is None else np.asarray(weights, dtype=float)
    if (w * n).sum() == 0:
        raise ValueError("pooled share over zero items")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(k), size=(n_boot, len(k)))
    den = (w * n)[idx].sum(1)
    if (den == 0).any():
        raise ValueError("a bootstrap draw holds no item")
    draws = (w * k)[idx].sum(1) / den
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float((w * k).sum() / (w * n).sum()), [float(lo), float(hi)], draws


def birth_order_test(poles: list[tuple[list[int], int]], n_perm: int = N_PERM, seed: int = 0,
                     n_boot: int = N_BOOT) -> dict:
    """§6.4. poles = [(births of the pole's born clusters, index of main)], poles with >= 2 births.
    b = share of poles whose main has the earliest birth (ties count as earliest), with a 95% interval
    from resampling poles (§7). Null: births permuted within each pole, so main's birth is a uniform
    draw from that pole's births."""
    eligible = [(np.asarray(b), mi) for b, mi in poles if len(b) >= 2]
    if not eligible:
        raise ValueError("no pole with two or more born clusters")
    hits = np.array([b[mi] == b.min() for b, mi in eligible], dtype=float)
    b_obs, ci, _ = pooled_share(hits, np.ones(len(hits)), None, n_boot, seed)
    rng = np.random.default_rng(seed)
    null = np.zeros(n_perm)
    for b, _ in eligible:
        null += b[rng.integers(0, len(b), size=n_perm)] == b.min()
    null /= len(eligible)
    band_hi = float(np.percentile(null, 97.5))
    return {"b": b_obs, "ci95": ci, "n_poles": len(eligible), "null_mean": float(null.mean()),
            "null_97_5": band_hi, "gap_to_null_mean": b_obs - float(null.mean()),
            "supports_main_oldest": b_obs > band_hi, "drift_share": 1 - b_obs,
            "p": float((1 + (null >= b_obs).sum()) / (1 + n_perm)), "n_perm": n_perm,
            "n_boot": n_boot, "seed": seed}


def sensitivity_excluding_births(tables: dict, poles: list[str], primary: dict, ctrl_acc: list,
                                 gate_pass: bool, seed: int, excluded=SENSITIVITY_EXCLUDED_BIRTHS) -> dict:
    """Checker F2 item 3: p_u (§6.3) and b (§6.4) recomputed with clusters born at the device boundaries left out,
    reported as a labelled sensitivity beside the primary readings. p_u drops the excluded late-born clusters and keeps
    every pole's control; b drops the excluded clusters from each pole's births and leaves out a pole whose main is
    excluded. Same seed, draws and permutations as the primary readings."""
    excl = set(excluded)
    out = {"excluded_births": [GRID_LABELS[t] for t in excluded], "p_u": None}
    if gate_pass:
        cluster_acc = [[primary[r["cluster"]]["acc_random"] for r in tables[p]
                        if r["cluster"] in primary and r["birth_pct"] not in excl] for p in poles]
        pu = p_u_bootstrap(cluster_acc, ctrl_acc, N_BOOT, seed)
        out["p_u"] = {**pu, "rule": p_u_rule(pu["ci95"]),
                      "n_clusters_left_out": sum(r["birth_pct"] in excl for r in primary.values())}
    order, main_excluded, left_out = [], 0, 0
    for p in poles:
        main = next(r for r in tables[p] if r["is_main"])
        if main["birth_pct"] is None:
            continue
        if main["birth_pct"] in excl:
            main_excluded += 1
            continue
        born = [r for r in tables[p] if not r["echo"] and r["birth_pct"] is not None]
        kept = [r for r in born if r["birth_pct"] not in excl]
        left_out += len(born) - len(kept)
        if len(kept) >= 2:
            order.append(([r["birth_pct"] for r in kept], kept.index(main)))
    out["birth_order"] = {**birth_order_test(order, N_PERM, seed), "n_clusters_left_out": left_out,
                          "n_poles_left_out_main_excluded": main_excluded}
    return out


def vestigial_counts(rows: list[dict]) -> tuple[int, int, int] | None:
    """§6.5 per pole: (non-main clusters born at or before main and shrinking, non-main born clusters
    with a defined growth, non-main born clusters whose growth is undefined because max|a| is 0 at
    T_12 or T_15). None when the pole has no born main."""
    main = next((r for r in rows if r["is_main"]), None)
    if main is None or main["birth_pct"] is None:
        return None
    nonmain = [r for r in rows if not r["is_main"] and not r["echo"] and r["birth_pct"] is not None]
    defined = [r for r in nonmain if r["growth_last_half"] is not None]
    k = sum(r["birth_pct"] <= main["birth_pct"] and r["growth_last_half"] < 0 for r in defined)
    return k, len(defined), len(nonmain) - len(defined)


def vestigial(per_pole: list[tuple[int, int]], n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """§6.5. per_pole = [(vestigial non-main clusters, non-main clusters)]. v pools over poles; the
    interval resamples poles. H_VEST iff the lower bound > 0.5."""
    k = [x for x, _ in per_pole]
    n = [y for _, y in per_pole]
    v, ci, draws = pooled_share(k, n, None, n_boot, seed)
    return {"v": v, "ci95": ci, "n_clusters": int(sum(n)), "label": "H_VEST" if ci[0] > VEST_LOWER else "not H_VEST",
            "p": float((1 + (draws <= VEST_LOWER).sum()) / (1 + n_boot)), "n_boot": n_boot, "seed": seed}


def stratum_weights(poles: list[str], strata: dict[str, str], population: dict[str, int]) -> tuple[np.ndarray, dict]:
    """§4.4 population weights: a pole of class k weighs N_k / n_k, where N_k is the class count over
    the classified audit poles and n_k the count in S."""
    missing = [p for p in poles if p not in strata or strata[p] == "RARE"]
    if missing:
        raise ValueError(f"S poles without a Q1 class (or RARE): {missing[:10]}")
    n_s = Counter(strata[p] for p in poles)
    w = np.array([population[strata[p]] / n_s[strata[p]] for p in poles])
    return w, {"population": population, "sample": dict(n_s),
               "unrepresented_strata": sorted(k for k, N in population.items() if N > 0 and n_s.get(k, 0) == 0)}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    out, running = {}, 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (len(items) - i) * p))
        out[name] = running
    return out


# --- §9.3 blind gate -------------------------------------------------------------------------

STAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)(Z|[+-]\d{2}:\d{2})?")


def predictions_section(lines: list[str]) -> tuple[int, int] | None:
    """(first line, end) of the first `## PREDICTIONS` section; it ends at the next level-1 or
    level-2 heading."""
    start = next((i for i, line in enumerate(lines) if line.startswith("## PREDICTIONS")), None)
    if start is None:
        return None
    end = next((j for j in range(start + 1, len(lines)) if re.match(r"#{1,2} ", lines[j])), len(lines))
    return start, end


def heading_time(heading: str) -> datetime | None:
    """The heading's timestamp in UTC; None unless it carries an explicit zone (Z or ±hh:mm)."""
    m = STAMP_RE.search(heading)
    if m is None or m.group(3) is None:
        return None
    zone = "+00:00" if m.group(3) == "Z" else m.group(3)
    return datetime.fromisoformat(f"{m.group(1)}T{m.group(2)}{zone}").astimezone(UTC)


def missing_readouts(body: list[str]) -> list[str]:
    """§11 readouts with no line of their own: a line names a readout when it starts (after an
    optional list marker and backtick) with the readout's name followed by a space, colon, '=' or '('."""
    return [name for name in READOUTS
            if not any(re.match(rf"\s*(?:[-*]\s+)?`?{re.escape(name)}`?(?=[\s:=(]|$)", line) for line in body)]


def git(repo, *args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def committed_time(journal: Path, lines: list[str], start: int, end: int) -> datetime:
    """Latest commit time (committer date, UTC) among the commits that changed the count of the exact
    heading line (`git log -S`) and the commits that last touched each line of the section
    (`git blame`). Refuses when the section differs from HEAD or the heading is in no commit."""
    top = git(journal.parent, "rev-parse", "--show-toplevel")
    if top.returncode:
        refuse(f"{journal}: not inside a git repository ({top.stderr.strip()})")
    repo = Path(top.stdout.strip())
    rel = journal.resolve().relative_to(repo.resolve()).as_posix()
    head = git(repo, "show", f"HEAD:{rel}")
    if head.returncode:
        refuse(f"{journal}: not in any commit")
    committed = head.stdout.splitlines()
    span = predictions_section(committed)
    if span is None or committed[span[0]:span[1]] != lines[start:end]:
        refuse(f"{journal}: the PREDICTIONS section has uncommitted changes")
    log = git(repo, "log", "--format=%cI", "-S", lines[start], "--", rel)
    if log.returncode or not log.stdout.strip():
        refuse(f"{journal}: the PREDICTIONS heading is in no commit")
    times = [datetime.fromisoformat(t).astimezone(UTC) for t in log.stdout.split()]
    blame = git(repo, "blame", "--porcelain", "-L", f"{span[0] + 1},{span[1]}", "HEAD", "--", rel)
    if blame.returncode:
        refuse(f"{journal}: git blame failed ({blame.stderr.strip()})")
    times += [datetime.fromtimestamp(int(line.split()[1]), UTC) for line in blame.stdout.splitlines()
              if line.startswith("committer-time ")]
    return max(times)


def parse_utc(stamp: str) -> datetime:
    dt = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def refuse(message: str) -> None:
    print(f"neurons_verdict: refusing to run: {message}", file=sys.stderr)
    sys.exit(2)


def predictions_gate(journal, manifests: list) -> dict:
    """Exit 2 unless all hold for the first `## PREDICTIONS` section of
    `.../studies/already-clean/JOURNAL.md` (a later section never stands in for it):
    (a) its heading carries a timestamp with an explicit zone; (b) it has a line for each §11
    readout; (c) it equals HEAD's version and its heading and lines are committed; and both the
    heading timestamp and the latest commit time precede the started_utc of every manifest.
    Returns {"heading_utc", "commit_utc"}."""
    journal = Path(journal)
    if journal.resolve().parts[-3:] != JOURNAL_SUFFIX:
        refuse(f"{journal}: the journal must be {'/'.join(JOURNAL_SUFFIX)}")
    if not journal.is_file():
        refuse(f"{journal}: journal not found")
    lines = journal.read_text(encoding="utf-8").splitlines()
    span = predictions_section(lines)
    if span is None:
        refuse(f"{journal}: no '## PREDICTIONS' heading")
    start, end = span
    heading_utc = heading_time(lines[start])
    if heading_utc is None:
        refuse(f"{journal}: the PREDICTIONS heading needs a timestamp with an explicit zone "
               f"(YYYY-MM-DDTHH:MMZ or ±hh:mm): {lines[start]!r}")
    missing = missing_readouts(lines[start + 1:end])
    if missing:
        refuse(f"{journal}: the PREDICTIONS section has no line for {missing} (each line starts with the readout name)")
    commit_utc = committed_time(journal, lines, start, end)
    if not manifests:
        refuse("no confirmation manifests given")
    for path in manifests:
        path = Path(path)
        if not path.is_file():
            refuse(f"{path}: manifest not found")
        started = read_json(path).get("started_utc")
        if started is None:
            refuse(f"{path}: no started_utc")
        for name, when in (("heading", heading_utc), ("commit", commit_utc)):
            if not when < parse_utc(started):
                refuse(f"{path}: started_utc {started} does not follow the PREDICTIONS {name} time "
                       f"({when.isoformat()}) in {journal}")
    return {"heading_utc": heading_utc.isoformat(), "commit_utc": commit_utc.isoformat()}


# --- CLI: θ_s (discovery) --------------------------------------------------------------------

def texts_by_pole(explain: Path) -> dict[str, dict[str, list[dict]]]:
    """{pole: {kind: [text rows in k order]}}; job_pos rows are keyed 'job_pos:<job>'."""
    out: dict[str, dict[str, list[dict]]] = {}
    for row in read_jsonl(explain / "texts.jsonl"):
        kind = row["kind"] if row["kind"] != "job_pos" else f"job_pos:{row['job']}"
        out.setdefault(row["pole"], {}).setdefault(kind, []).append(row)
    return out


def scorer(stats_dir, probe_dir):
    """score(text_id, pole) from the probe file and the stats harvest; a missing or non-finite probe
    value raises, naming the text and the pole."""
    if read_json(Path(probe_dir) / HARVEST_FILES["manifest"])["revision"] != \
            read_json(Path(stats_dir) / HARVEST_FILES["manifest"])["revision"]:
        raise ValueError(f"{probe_dir} and {stats_dir} are different model revisions")
    probe_rule(probe_dir)
    tix, pix, max_a, min_a = probe_scores(probe_dir)

    def sc(text_id: str, pole: str) -> float:
        if text_id not in tix or pole not in pix:
            raise ValueError(f"{probe_dir}: no probe value for text {text_id} at pole {pole}")
        hi, lo = max_a[tix[text_id], pix[pole]], min_a[tix[text_id], pix[pole]]
        if not (np.isfinite(hi) and np.isfinite(lo)):
            raise ValueError(f"{probe_dir}: non-finite probe value for text {text_id} at pole {pole}")
        _, _, s = parse_pole(pole)
        return score(hi, lo, pole_stats(stats_dir, pole)["max_abs"], s)
    return sc


def shuffled_scores(explain: Path, sc) -> list[float]:
    shuffle = read_json(explain / "shuffle.json")["pairs"]
    return [sc(t["text_id"], shuffle[pole])
            for pole, kinds in sorted(texts_by_pole(explain).items())
            for kind in TEST_KINDS for t in kinds[kind]]


def cmd_theta(a) -> None:
    explain = Path(a.explain) / "explain"
    vals = shuffled_scores(explain, scorer(a.stats, a.probe))
    theta = theta_s(vals)
    gate = theta_gate(theta)
    out = Path(a.out) / "verdict"
    out.mkdir(parents=True, exist_ok=True)
    record = {"theta_s": theta, "gate": gate, "created_utc": utc_now(), "percentile": THETA_PERCENTILE,
              "method": "numpy.percentile linear", "n_scores": len(vals),
              "shuffled_scores_quantiles": {q: float(np.percentile(vals, q)) for q in (50, 90, 99)},
              "inputs": input_hashes({
                  "texts": explain / "texts.jsonl", "shuffle": explain / "shuffle.json",
                  "explain_manifest": explain / "manifest.json",
                  "scores": Path(a.probe) / HARVEST_FILES["probe"], "probe_poles": Path(a.probe) / HARVEST_FILES["poles"],
                  "probe_manifest": Path(a.probe) / HARVEST_FILES["manifest"],
                  "stats_manifest": Path(a.stats) / HARVEST_FILES["manifest"]}),
              "probe": probe_rule(a.probe)}
    path = out / "theta.json"
    if path.exists():
        raise FileExistsError(f"{path} exists; θ_s is frozen once")
    path.write_text(json.dumps(record, indent=1))
    print(json.dumps({"theta_s": theta, "gate": gate, "file": str(path)}))
    if gate != "pass":
        sys.exit(3)


# --- CLI: draw S and S_gate (W3 input) -------------------------------------------------------

DRAW_RULE = ("S = every audit pole that is not RARE; S_gate = every extra pole that is not RARE; "
             "grid_poles = S ∪ S_gate. RARE (§2.3) is read from the harvest's stats/ only (max|a| = 0, or "
             "fewer than 50 tokens with |a| >= 0.01 max|a|). The audit sample has 600 poles and §4.4 sets "
             "|S| = 600 with RARE excluded, so the class order of §4.4 (UNREL and UNK first, then TWO-REL, "
             "then ONE at random) selects every non-RARE pole and needs no explanation or class; no random "
             "step is taken and the seed is recorded only. Population weights stay with unblind "
             "(stratum_weights, §4.4).")


def cmd_draw_s(a) -> None:
    """§4.4 development sample S and gate set S_gate, blind by construction (DRAW_RULE): no
    explanation, class or score is read, only stats/ through pole_stats and is_rare. More than S_SIZE
    non-RARE audit poles would need the class order, so it refuses instead of inventing one."""
    out = Path(a.out) / "draw"
    if out.exists():
        raise FileExistsError(f"{out} exists; S is drawn once")
    audit, extra = read_json(a.audit), read_json(a.extra)
    for name, poles in (("audit", audit), ("extra", extra)):
        if len(set(poles)) != len(poles):
            raise ValueError(f"{name} pole list repeats a pole")
    if set(audit) & set(extra):
        raise ValueError(f"audit and extra share poles: {sorted(set(audit) & set(extra))[:5]}")

    def split(poles):
        stats = {p: pole_stats(a.stats, p) for p in poles}
        rare = {p: {"density": stats[p]["density"], "max_abs": stats[p]["max_abs"]}
                for p in poles if is_rare(stats[p]["density"], stats[p]["max_abs"])}
        return sorted((p for p in poles if p not in rare), key=parse_pole), rare
    S, rare_audit = split(audit)
    S_gate, rare_extra = split(extra)
    if len(S) > S_SIZE:
        raise ValueError(f"{len(S)} non-RARE audit poles > |S| = {S_SIZE}: the §4.4 class order would decide, "
                         f"and draw-s reads no class")
    grid = sorted(S + S_gate, key=parse_pole)
    counts = {"audit": len(audit), "extra": len(extra), "rare_excluded_audit": len(rare_audit),
              "rare_excluded_extra": len(rare_extra), "S": len(S), "S_gate": len(S_gate), "grid_poles": len(grid)}
    record = {"rule": DRAW_RULE, "seed": a.seed, "created_utc": utc_now(), "counts": counts,
              "rare_excluded": {"audit": rare_audit, "extra": rare_extra},
              "inputs": {"stats": str(a.stats), **input_hashes({"audit": Path(a.audit), "extra": Path(a.extra),
                                                                "stats_manifest": Path(a.stats) / HARVEST_FILES["manifest"]})}}
    out.mkdir(parents=True)
    for name, obj in (("S.json", S), ("S_gate.json", S_gate), ("grid_poles.json", grid), ("draw.json", record)):
        (out / name).write_text(json.dumps(obj, indent=1))
    print(json.dumps(counts))


# --- CLI: discovery report (W2-D prompt iteration) -------------------------------------------

def batch_counts(explain: Path) -> dict:
    """Per batch: request, response and rejected-line counts, the requested model ids, and the backend
    and reported model ids from the backend sidecar (manual batches have none)."""
    out = {}
    for path in sorted((explain / "requests").glob("*.jsonl")):
        batch = path.stem
        reqs = read_jsonl(requests_path(explain, batch))
        side = backend_sidecar_path(explain, batch)
        sidecar = read_json(side) if side.exists() else None
        out[batch] = {"requests": len(reqs),
                      "responses": len(read_jsonl(responses_path(explain, batch))) if responses_path(explain, batch).exists() else 0,
                      "rejected_lines": len(read_jsonl(rejected_path(explain, batch))) if rejected_path(explain, batch).exists() else 0,
                      "declined": len(read_jsonl(declined_path(explain, batch))) if declined_path(explain, batch).exists() else 0,
                      "requested_models": sorted({r["model"] for r in reqs}),
                      "backend": sidecar["backend"] if sidecar else "manual (no backend sidecar)",
                      "reported_models": sidecar["reported_models"] if sidecar else None}
    return out


def discovery_sample(rec: dict) -> dict:
    return {"pole": rec["pole"], "class": rec["class"], "unk_part": rec.get("unk_part"), "label": rec["label"],
            "passes": rec.get("passes"), "explanation": rec["explanation"], "declined": rec["declined"],
            "unk_reason": rec["unk_reason"],
            "strong": {k: sum(t["strong"] for t in v) for k, v in rec.get("texts", {}).items()},
            "blind_max_score": rec.get("blind_max_score")}


def cmd_discovery(a) -> None:
    """One prompt version's discovery report: the same records and metrics unblind computes
    (q1_records, q1_metrics), with θ_s from the given theta file and none of unblind's gates."""
    explain = Path(a.explain) / "explain"
    theta_rec = read_json(a.theta)
    if theta_rec["gate"] != "pass":
        raise ValueError(f"{a.theta}: θ_s gate is {theta_rec['gate']}")
    recs = q1_records(a.stats, a.probe, explain, theta_rec["theta_s"])
    q1 = q1_metrics(recs, theta_rec["theta_s"])
    manifest = read_json(explain / "manifest.json")
    report = {"set": "discovery", "created_utc": utc_now(),
              "theta": {"theta_s": theta_rec["theta_s"], "gate": theta_rec["gate"],
                        "created_utc": theta_rec["created_utc"], "file": str(a.theta)},
              "pass_rate": pass_rate(recs),
              "fractions": q1["fractions"],
              "agreement": {k: q1[k] for k in ("kappa_label", "job_count_agreement", "n_rated_twice", "kappa_models")
                            if k in q1},
              "by_explainer_model": q1["by_explainer_model"],
              "declined": q1["declined"],
              "X7_pass_rate_by_encoding_flag": q1["X7_pass_rate_by_encoding_flag"],
              "strong_rates": q1["strong_rates"],
              "X3_blind_max_ge_corpus_max": q1["X3_blind_max_ge_corpus_max"],
              "prompts": {"version": manifest["prompt_version"], "source": str(explain / "manifest.json"),
                          "sha256": {r: manifest["prompts"][r] for r in ("explainer", "second_reader", "writer")}},
              "batches": batch_counts(explain),
              "inputs": {"stats": str(a.stats), "probe": str(a.probe), "explain": str(explain), **probe_rule(a.probe)}}
    out = Path(a.out) / "verdict"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "discovery.json"
    if path.exists():
        raise FileExistsError(f"{path} exists; one report per run")
    write_jsonl_new(out / "discovery_samples.jsonl", [discovery_sample(r) for r in recs])
    path.write_text(json.dumps(report, indent=1))
    print(json.dumps({"pass_rate": report["pass_rate"]["rate"], "n": report["pass_rate"]["n"], "file": str(path)}))


# --- judge plans, judge validity gate (development discovery) --------------------------------

def judge_plans(explain: Path) -> list[dict]:
    paths = sorted((explain / "plans").glob("*-judge.jsonl"))
    if not paths:
        raise ValueError(f"{explain / 'plans'}: no judge plan")
    return [row for p in paths for row in read_jsonl(p)]


def check_judgments_match_plans(explain: Path, judgments: list[dict], control_only: bool = False) -> None:
    """Every planned judgment (or every planned control judgment) has exactly one judgment row."""
    planned = {p["id"] for p in judge_plans(explain) if p["control"] or not control_only}
    got = [j["id"] for j in judgments if j["control"] or not control_only]
    missing = sorted(planned - set(got))
    extra = sorted(set(got) - planned)
    repeated = sorted({i for i in got if got.count(i) > 1})
    if missing or extra or repeated:
        raise ValueError(f"{explain}: judgments do not match plans/: missing {missing[:20]} (n={len(missing)}), "
                         f"unplanned {extra[:20]}, repeated {repeated[:20]}")


def judge_identity(explain: Path) -> dict:
    """The judge an explain directory's judgments came from: every model that answered a judge line
    (manifest models_by_role, written by collect-judge) and the judge prompt's SHA-256."""
    m = read_json(explain / "manifest.json")
    return {"models": sorted(m["models_by_role"]["answered"]["judge"]), "prompt_sha256": m["prompts"]["judge"]}


def cmd_judge_gate(a) -> None:
    """§6.3 validity gate on the development discovery set. Every pole with a non-echo cluster must
    have its full control plan (4 random-decoy and 2 hard-decoy judgments) planned and judged or declined;
    a pole with a declined control judgment leaves a_ctrl, and a pole with a declined clustering sample
    is not in the set (declined rule); a_ctrl is computed over the rest."""
    explain = Path(a.explain) / "explain"
    judgments = read_jsonl(explain / "judgments.jsonl")
    poles = sorted(p for p, s in clusters_by_pole(explain).items() if any(not c["echo"] for c in s[0]))
    plans = judge_plans(explain)
    problems = []
    for pole in poles:
        planned = Counter(p["decoy_type"] for p in plans if p["control"] and p["cluster"].split(":")[0] == pole)
        if dict(planned) != CTRL_PLAN:
            problems.append(f"{pole}: planned controls {dict(planned)}, need {CTRL_PLAN}")
    if problems:
        raise ValueError(f"{explain}: incomplete control plan:\n  " + "\n  ".join(problems))
    check_judgments_match_plans(explain, judgments, control_only=True)
    _, ctrls, declined = judged([j for j in judgments if j["control"]])
    kept = [p for p in poles if p not in declined["control_poles"]]
    if not kept:
        raise ValueError(f"{explain}: no pole has a complete control; a_ctrl is undefined")
    a_ctrl = float(np.mean([np.mean(ctrls[p]["random"]) for p in kept]))
    gone = declined_clusterer(explain)
    record = {"a_ctrl": a_ctrl, "n_poles": len(kept), "gate": "pass" if a_ctrl >= CTRL_GATE else "fail",
              "threshold": CTRL_GATE, "created_utc": utc_now(),
              "f_ctrl": float(np.mean([sum(ctrls[p]["random"]) <= 1 for p in kept])),
              "hard_ctrl": float(np.mean([np.mean(ctrls[p]["hard"]) for p in kept])), "judge": judge_identity(explain),
              "declined_judgments": declined, "declined_clusterer": {"n": len(gone), "poles": gone},
              "inputs": input_hashes({"judgments": explain / "judgments.jsonl", "clusters": explain / "clusters.jsonl",
                                      "explain_manifest": explain / "manifest.json",
                                      **{f"plan:{p.name}": p for p in sorted((explain / "plans").glob("*-judge.jsonl"))}})}
    out = Path(a.out) / "verdict"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "judge_gate.json"
    if path.exists():
        raise FileExistsError(f"{path} exists; the gate is read once")
    path.write_text(json.dumps(record, indent=1))
    print(json.dumps({k: record[k] for k in ("a_ctrl", "n_poles", "gate", "f_ctrl")}))


# --- CLI: unblind, Q1 ------------------------------------------------------------------------

def requests_by_id(explain: Path) -> dict[str, dict]:
    return {r["id"]: r for p in sorted((explain / "requests").glob("*.jsonl")) for r in read_jsonl(p)}


def answering_models(explain: Path) -> dict[str, str]:
    """{request id: model} over every batch: the model the response line reports; a declined id (no
    response) carries the model its request names."""
    out = {}
    for path in sorted((explain / "requests").glob("*.jsonl")):
        answered = response_models(explain, path.stem)
        out.update({r["id"]: answered[r["id"]] if r["id"] in answered else r["model"] for r in read_jsonl(path)})
    return out


def q1_records(stats_dir, probe_dir, explain: Path, theta: float) -> list[dict]:
    sc = scorer(stats_dir, probe_dir)
    models = answering_models(explain)
    texts = texts_by_pole(explain)
    shuffle = read_json(explain / "shuffle.json")["pairs"]
    agent = {0: {}, 1: {}}
    for row in read_jsonl(explain / "explanations.jsonl"):
        agent[row["sample"]][row["pole"]] = row
    writer_gave_up = {r["id"].split(":")[1]: decline_text(r) for r in declined_rows(explain)
                      if r["role"] in ("writer", "writer_job")}
    n_layers = read_json(Path(stats_dir) / HARVEST_FILES["manifest"])["n_layers"]
    vocab = read_json(Path(stats_dir) / HARVEST_FILES["vocab"])
    bands = {"class": theta, "class_theta_plus": theta + THETA_BAND}
    if theta - THETA_BAND > 0:                      # at or below 0 every non-negative text is strong
        bands["class_theta_minus"] = theta - THETA_BAND
    records = []
    for pole, a1 in sorted(agent[0].items()):
        layer, _, _ = parse_pole(pole)
        st = pole_stats(stats_dir, pole)
        rec = {"pole": pole, "layer": layer, "tercile": min(2, 3 * layer // n_layers),
               "rare": is_rare(st["density"], st["max_abs"]), "label": a1["class"],
               "n_jobs": len(a1["jobs"]), "explanation": a1["explanation"], "jobs": a1["jobs"],
               "double_negative_share": st["double_negative_share"], "pos0_share": pole_pos0_share(stats_dir, pole),
               "encoding_flag": a1["encoding_flag"], "declined": a1.get("declined") or writer_gave_up.get(pole),
               "explainer_model": models[f"explainer:{pole}:0"],
               "suppressor": single_token_suppressor(pole_vocab(vocab, pole)) if layer == n_layers - 1 else None}
        if pole in agent[1] and "declined" in agent[1][pole]:
            rec["second_reader_declined"] = agent[1][pole]["declined"]
        elif pole in agent[1]:
            rec["label_2"], rec["n_jobs_2"] = agent[1][pole]["class"], len(agent[1][pole]["jobs"])
            rec["second_reader_model"] = models[f"second_reader:{pole}:1"]
        if st["max_abs"] > 0 and not rec["declined"]:
            kinds = texts[pole]
            rec["texts"] = {k: [{"text_id": t["text_id"], "text": t["text"], "score": sc(t["text_id"], pole)}
                                for t in rows] for k, rows in kinds.items()}
            for rows in rec["texts"].values():
                for t in rows:
                    t["strong"] = t["score"] >= theta
            rec["shuffle_pole"] = shuffle[pole]
            rec["shuffled"] = [{"text_id": t["text_id"], "score": sc(t["text_id"], shuffle[pole])}
                               for k in TEST_KINDS for t in kinds[k]]
            for t in rec["shuffled"]:
                t["strong"] = t["score"] >= theta
        for name, th in bands.items():
            if st["max_abs"] == 0 or rec["declined"]:       # a declined pole has no texts: RARE or UNK-N
                rec[name] = "RARE" if rec["rare"] else "UNK"
                continue
            s = {k: [t["score"] for t in v] for k, v in rec["texts"].items()}
            passes = explanation_passes(s["pos"], s["near"], s["distant"], th)
            support = {int(k.split(":")[1]): job_supported(v, th) for k, v in s.items()
                       if k.startswith("job_pos:")}
            rec[name] = assign_class(rec["rare"], rec["label"], rec["jobs"], passes, support)
            if name == "class":
                rec["passes"], rec["job_support"] = passes, support
                rec["essay_rule"] = essay_rule(s["pos"], s["near"])
                rec["blind_max_score"] = s["max"][0]
        rec["unk_part"] = part_of(rec, rec["class"])
        rec["unk_reason"] = ((rec["declined"] or unk_reason(rec, rec["passes"], rec["job_support"]))
                             if rec["class"] == "UNK" else None)
        records.append(rec)
    return records


def part_of(rec: dict, cls: str) -> str | None:
    """UNK-N or UNK-F for a record's class; a pole whose explainer or writer declined is UNK-N."""
    if cls == "UNK" and rec["declined"]:
        return "UNK-N"
    return unk_part(cls, rec["label"])


def unk_part(cls: str, label: str) -> str | None:
    """§2.3 as amended: UNK-N when the explainer's own class is UNK, UNK-F for every other UNK (an
    explanation was given and did not survive the rules)."""
    if cls != "UNK":
        return None
    return "UNK-N" if label == "UNK" else "UNK-F"


def unk_reason(rec: dict, passes: bool, support: dict) -> str:
    """Why an UNK-F pole is UNK: its explanation failed §6.1, an UNREL label lacked two supported
    unrelated jobs, or the label disagrees with the job list."""
    if rec["label"] == "UNK":
        return "explainer said UNK"
    if rec["label"] in ("ONE", "TWO-REL") and not passes:
        return "failed §6.1"
    if rec["label"] == "UNREL":
        return "UNREL support clause failed"
    return "label and job list disagree"


def fractions(classes: list[str], parts: list[str | None]) -> dict:
    """Class fractions with Wilson intervals; UNK is also split into UNK-N and UNK-F (§2.3), and the
    two parts add up to UNK."""
    n = len(classes)
    if len(parts) != n:
        raise ValueError("one UNK part per class")
    return {"n": n, **{c: {"k": classes.count(c), "p": classes.count(c) / n,
                           "wilson95": list(wilson(classes.count(c), n))} for c in CLASSES},
            **{u: {"k": parts.count(u), "p": parts.count(u) / n, "wilson95": list(wilson(parts.count(u), n))}
               for u in UNK_PARTS}}


def pos0_by_class(records: list[dict]) -> dict:
    """DATA_CONTRACT §8: the distribution of pos0_share per class (min, quartiles, max, mean)."""
    out = {}
    for c in CLASSES + UNK_PARTS:
        x = [r["pos0_share"] for r in records if r["class"] == c or r.get("unk_part") == c]
        out[c] = ({"n": len(x), "mean": float(np.mean(x)),
                   "quantiles_0_25_50_75_100": [float(q) for q in np.percentile(x, [0, 25, 50, 75, 100])]}
                  if x else {"n": 0})
    return out


def pass_rate_by_encoding_flag(records: list[dict]) -> dict:
    """X7 (§0c): §6.1 pass rate of poles whose explainer view carried a decoded or transliterated line,
    against poles whose view did not."""
    out = {}
    for name, flag in (("flagged", True), ("unflagged", False)):
        hits = [r["passes"] for r in records if "passes" in r and r["encoding_flag"] is flag]
        out[name] = ({"k": sum(hits), "n": len(hits), "rate": sum(hits) / len(hits),
                      "wilson95": list(wilson(sum(hits), len(hits)))} if hits else {"n": 0})
    return out


def blind_max_by_class(records: list[dict]) -> dict:
    """X3 (§0c, exploratory): per class, the share of scored poles whose blind-maximum sentence
    scores at least 1.0 of the corpus max."""
    out = {}
    for c in CLASSES + UNK_PARTS:
        hits = [r["blind_max_score"] >= BLIND_MAX_CENTER for r in records if "blind_max_score" in r
                and (r["class"] == c or r.get("unk_part") == c)]
        out[c] = ({"k": sum(hits), "n": len(hits), "share": sum(hits) / len(hits),
                   "wilson95": list(wilson(sum(hits), len(hits)))} if hits else {"n": 0})
    return out


def q1_metrics(records: list[dict], theta: float) -> dict:
    def sub(rows):
        return fractions([r["class"] for r in rows], [r["unk_part"] for r in rows])

    def band(name):
        return fractions([r[name] for r in records], [part_of(r, r[name]) for r in records])
    out = {"fractions": sub(records),
           "by_depth_tercile": {str(t): sub([r for r in records if r["tercile"] == t])
                                for t in range(3) if any(r["tercile"] == t for r in records)},
           "band_theta_minus_0.1": (band("class_theta_minus")
                                    if theta - THETA_BAND > 0 else f"not computed: θ_s - {THETA_BAND} <= 0"),
           "band_theta_plus_0.1": band("class_theta_plus"),
           "declined": {"n": sum(bool(r["declined"]) for r in records),
                        "poles": {r["pole"]: r["declined"] for r in records if r["declined"]},
                        "second_reader": sum("second_reader_declined" in r for r in records)},
           "unk_reasons": dict(Counter(r["unk_reason"] for r in records if r["class"] == "UNK")),
           "X3_blind_max_ge_corpus_max": blind_max_by_class(records),
           "X7_pass_rate_by_encoding_flag": pass_rate_by_encoding_flag(records),
           "pos0_share_by_class": pos0_by_class(records),
           "essay_rule": {k: sum(r.get("essay_rule") == k for r in records) for k in ("win", "tie", "lose")},
           "single_token_suppressors": sum(bool(r["suppressor"]) for r in records),
           "untestable_phrase": {"k": sum(UNTESTABLE_PHRASE in r["explanation"].lower() for r in records),
                                 "n": len(records), "phrase": UNTESTABLE_PHRASE,
                                 "reading": "explanations containing the phrase explainer.md asks for when a job "
                                            "cannot be carried by a short text (string match; no field records it)"},
           "strong_rates": strong_rates([r for r in records if not r["rare"]]),
           "strong_rates_including_rare": strong_rates(records)}
    dn = [r for r in records if r["double_negative_share"] >= DOUBLE_NEGATIVE_SPLIT]
    rest = [r for r in records if r["double_negative_share"] < DOUBLE_NEGATIVE_SPLIT]
    out["double_negative_ge_0.8"] = sub(dn) if dn else {"n": 0}
    out["double_negative_lt_0.8"] = sub(rest) if rest else {"n": 0}
    out.update(agreement(records))
    out["by_explainer_model"] = {
        model: {"n": len(rows), "fractions": sub(rows), "pass_rate": pass_rate(rows),
                "strong_rates": (strong_rates([r for r in rows if not r["rare"]])
                                 if any("texts" in r and not r["rare"] for r in rows) else {"n": 0}),
                **agreement(rows)}
        for model, rows in sorted(group_by(records, "explainer_model").items())}
    return out


def group_by(records: list[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in records:
        out.setdefault(r[key], []).append(r)
    return out


def pass_rate(records: list[dict]) -> dict:
    """§6.1 pass rate over the scored poles that are not RARE, with a Wilson interval."""
    scored = [r["passes"] for r in records if "passes" in r and not r["rare"]]
    if not scored:
        return {"n": 0}
    return {"k": sum(scored), "n": len(scored), "rate": sum(scored) / len(scored),
            "wilson95": list(wilson(sum(scored), len(scored))), "over": "scored poles that are not RARE"}


def agreement(records: list[dict]) -> dict:
    """§6.2 inter-rater kappa between the explainer's label and the second reader's, labelled with both
    models; pairs answered by different models are counted as cross-model (2026-10-20 amendment)."""
    rated = [r for r in records if "label_2" in r]
    if not rated:
        return {"n_rated_twice": 0}
    return {"kappa_label": cohen_kappa([r["label"] for r in rated], [r["label_2"] for r in rated]),
            "job_count_agreement": float(np.mean([r["n_jobs"] == r["n_jobs_2"] for r in rated])),
            "n_rated_twice": len(rated),
            "kappa_models": {"explainer": dict(Counter(r["explainer_model"] for r in rated)),
                             "second_reader": dict(Counter(r["second_reader_model"] for r in rated)),
                             "cross_model_pairs": sum(r["explainer_model"] != r["second_reader_model"] for r in rated)}}


def strong_rates(records: list[dict]) -> dict:
    """F2: strong rates of positives, near misses, distant and shuffled texts, with Wilson intervals."""
    def rate(flags):
        return {"k": sum(flags), "n": len(flags), "rate": sum(flags) / len(flags),
                "wilson95": list(wilson(sum(flags), len(flags)))}
    scored = [r for r in records if "texts" in r]
    out = {kind: rate([t["strong"] for r in scored for t in r["texts"][kind]]) for kind in TEST_KINDS}
    out["shuffled"] = rate([t["strong"] for r in scored for t in r["shuffled"]])
    out["packing_rate_distant"] = out["distant"]["rate"]
    return out


# --- CLI: unblind, Q2 ------------------------------------------------------------------------

def judged(judgments: list[dict]) -> tuple[dict, dict, dict]:
    """(tests {cluster: {decoy_type: [correct]}}, controls {pole: {decoy_type: [correct]}}, declined).
    A judge line declined after the attempts cap (collect-judge records it with `declined`) takes its
    cluster out of tests, or its pole out of controls, so every judged cluster and control keeps the
    full plan (declined rule, 2026-10-21). `declined` lists the lines, clusters and control poles."""
    lines = [j for j in judgments if j.get("declined")]
    out_tests = sorted({j["cluster"] for j in lines if not j["control"]})
    out_ctrls = sorted({j["cluster"].split(":")[0] for j in lines if j["control"]})
    tests: dict[str, dict[str, list[bool]]] = {}
    ctrls: dict[str, dict[str, list[bool]]] = {}
    for j in judgments:
        pole = j["cluster"].split(":")[0]
        if (pole in out_ctrls) if j["control"] else (j["cluster"] in out_tests):
            continue
        target = ctrls.setdefault(pole, {}) if j["control"] else tests.setdefault(j["cluster"], {})
        target.setdefault(j["decoy_type"], []).append(j["correct"])
    for name, group in (("cluster", tests), ("control of", ctrls)):
        for key, d in group.items():
            got = {t: len(d.get(t, [])) for t in CTRL_PLAN}
            if got != CTRL_PLAN:
                raise ValueError(f"{name} {key}: judgments {got}; the plan makes {CTRL_PLAN}")
    declined = {"n": len(lines), "ids": {j["id"]: j["declined"] for j in sorted(lines, key=lambda j: j["id"])},
                "clusters": out_tests, "control_poles": out_ctrls}
    return tests, ctrls, declined


def g2_readout(t0_explain: Path) -> dict:
    """§5 G2: share of T_0 sampled poles whose explainer confidence exceeds 0.5, against the 10% line."""
    conf = [r["confidence"] for r in read_jsonl(t0_explain / "explanations.jsonl") if r["sample"] == 0]
    if not conf:
        raise ValueError(f"{t0_explain}: no explainer rows")
    k = sum(c > 0.5 for c in conf)
    return {"n_poles": len(conf), "k_confidence_gt_0.5": k, "share": k / len(conf),
            "wilson95": list(wilson(k, len(conf))), "line": G2_LINE, "within_line": k / len(conf) <= G2_LINE,
            "input": str(t0_explain / "explanations.jsonl")}


def center_birth_curve(center_dir: Path, grid_dirs: list, texts_explain: Path, poles: list[str],
                       strata: dict[str, str] | None) -> dict:
    """§2.4 center birth curve (exploratory, §11). The blind-maximum sentence and the five positives of every pole of
    S, scored at every grid point by a per-checkpoint probe run (`<center_dir>/<step>/poles/probe/scores.npz`, one
    harvest out directory per grid step with the grid directories' names, T_15 last), each raw probe value divided by that checkpoint's own max|a|
    from the W3 grid stats (`score`). Birth = the first t at which the blind maximum reaches CENTER_RATIO of its own
    T_15 score in the same run; the mean of the five positives is read the same way beside it. A pole whose max|a| is
    0 at a checkpoint scores 0 there (counted in `dead_pole_checkpoints`); each step's manifest revision must equal the
    grid step's and its probe rule is checked, as `scorer` does; a half-integer median is written as both labels; a pole with no positive T_15 score is
    undefined; a pole without a blind-maximum text and five positives (UNK-N, or declined at the explainer or writer)
    is left out and counted in `n_poles_without_texts`. Reported by the pole's Q1 class when `strata` is given."""
    steps = [Path(g).name for g in grid_dirs]
    texts = texts_by_pole(texts_explain)
    ids = {}
    for p in poles:
        kinds = texts.get(p, {})
        if kinds.get("max") and len(kinds.get("pos", [])) == 5:
            ids[p] = (kinds["max"][0]["text_id"], [r["text_id"] for r in kinds["pos"]])
    no_texts = [p for p in poles if p not in ids]   # UNK-N or declined at the explainer/writer: no texts were written
    poles = [p for p in poles if p in ids]
    s_max = {p: [] for p in poles}
    s_pos = {p: [] for p in poles}
    dead = 0
    hashes = {}
    for step, g in zip(steps, grid_dirs):
        probe = Path(center_dir) / step
        if read_json(probe / HARVEST_FILES["manifest"])["revision"] != read_json(Path(g) / HARVEST_FILES["manifest"])["revision"]:
            raise ValueError(f"{probe} and {g} are different model revisions")
        probe_rule(probe)
        hashes[step] = file_sha256(probe / HARVEST_FILES["probe"])
        tix, pix, max_a, min_a = probe_scores(probe)
        for p in poles:
            mab = pole_stats(g, p)["max_abs"]
            sign = parse_pole(p)[2]
            if not mab > 0:
                dead += 1
                s_max[p].append(0.0)
                s_pos[p].append(0.0)
                continue
            sc = lambda tid: score(float(max_a[tix[tid], pix[p]]), float(min_a[tix[tid], pix[p]]), mab, sign)
            s_max[p].append(sc(ids[p][0]))
            s_pos[p].append(float(np.mean([sc(tid) for tid in ids[p][1]])))

    def first_t(series):
        end = series[-1]
        if not end > 0:
            return None
        return next(t for t, v in enumerate(series) if v >= CENTER_RATIO * end)

    def t_label(x: float) -> str:
        lo = int(np.floor(x))
        return GRID_LABELS[lo] if x == lo else f"{GRID_LABELS[lo]}/{GRID_LABELS[lo + 1]}"

    def summary(ps, series):
        ft = {p: first_t(series[p]) for p in ps}
        defined = [p for p in ps if ft[p] is not None]
        ratios = np.array([[v / series[p][-1] for v in series[p]] for p in defined]) if defined else np.zeros((0, len(steps)))
        nondec = [bool(np.all(np.diff(r) >= -CENTER_MONOTONE_TOL)) for r in ratios]
        births = [ft[p] for p in defined]
        return {"n_poles": len(ps), "n_undefined_no_T15_score": len(ps) - len(defined),
                "median_first_t": (t_label(float(np.median(births))) if births else None),
                "median_first_t_index": (float(np.median(births)) if births else None),
                "first_t_counts": {GRID_LABELS[t]: int(sum(b == t for b in births)) for t in range(len(steps))},
                "median_ratio_curve": [float(x) for x in np.median(ratios, axis=0)] if defined else None,
                "share_nondecreasing": (float(np.mean(nondec)) if nondec else None)}

    out = {"rule": f"first t at which the blind maximum reaches {CENTER_RATIO} of its T_15 score (same per-checkpoint probe run)",
           "grid": steps, "dead_pole_checkpoints": dead, "n_poles_without_texts": len(no_texts),
           "poles_without_texts": no_texts, "blind_max": summary(poles, s_max),
           "positives_mean": summary(poles, s_pos), "by_class": {}, "inputs": {"center_probe": str(center_dir), "scores_sha256": hashes}}
    if strata is not None:
        for cls in sorted({strata[p] for p in poles if p in strata}):
            ps = [p for p in poles if strata.get(p) == cls]
            out["by_class"][cls] = {"blind_max": summary(ps, s_max), "positives_mean": summary(ps, s_pos)}
    bm = out["blind_max"]
    by = "; ".join(f"{cls} {v['blind_max']['median_first_t']} (n = {v['blind_max']['n_poles']})" for cls, v in out["by_class"].items())
    out["grid_text"] = (f"blind maximum reaches {CENTER_RATIO} of its T_15 score first at median {bm['median_first_t']} over "
                        f"{bm['n_poles'] - bm['n_undefined_no_T15_score']} poles" + (f" ({by})" if by else "")
                        + f"; positives' mean median {out['positives_mean']['median_first_t']}; ratio curve non-decreasing "
                        f"(tolerance {CENTER_MONOTONE_TOL}) for {bm['share_nondecreasing']:.3f} of poles; "
                        f"{bm['n_undefined_no_T15_score']} poles without a positive T_15 score; {len(no_texts)} poles without "
                        f"texts left out; {dead} dead pole-checkpoints")
    return out


def birth_histogram(rows: list[dict]) -> dict:
    """F4 (§10): births (percentile reading) per grid index, main and non-main clusters apart, echo clusters
    left out; clusters never present are counted as never_born."""
    out = {}
    for name, sel in (("main", [r for r in rows if r["is_main"]]), ("non_main", [r for r in rows if not r["is_main"]])):
        b = [r["birth_pct"] for r in sel]
        out[name] = {"counts": [b.count(t) for t in range(len(GRID_STEPS))], "never_born": b.count(None), "n": len(b)}
    return out


def quantiles(x) -> list[float]:
    return [float(v) for v in np.quantile(np.asarray(x, dtype=float), QUANTILES)]


def grid_population(grid_dirs: list) -> dict:
    """F3 (exploratory, §10): at each grid point, over every pole of every layer, the dead share (max_abs == 0,
    DATA_CONTRACT §2) and, over the live poles, quantiles of the double-negative share (sign_counts[..., 3] / K)
    and of the density (tokens with |a| ≥ 0.01·max_abs). Reads three arrays of each stats file, not the histograms."""
    points = []
    for g, step in zip(grid_dirs, GRID_STEPS):
        m = read_json(Path(g) / HARVEST_FILES["manifest"])
        cols: dict[str, list] = {"max_abs": [], "sign_counts": [], "density": []}
        for layer in range(m["n_layers"]):
            with np.load(Path(g) / "stats" / f"layer_{layer:02d}.npz", allow_pickle=False) as z:
                for k in cols:
                    cols[k].append(z[k].reshape(-1, *z[k].shape[2:]))
        mx = np.concatenate(cols["max_abs"])
        live = mx > 0
        dn = np.concatenate(cols["sign_counts"])[:, 3][live] / K_TOPK
        points.append({"step": step, "n_poles": int(mx.size), "dead_share": float(1 - live.mean()),
                       "double_negative_share": quantiles(dn),
                       "double_negative_ge_0.8": float((dn >= DOUBLE_NEGATIVE_SPLIT).mean()),
                       "density": quantiles(np.concatenate(cols["density"])[live]),
                       "n_tokens_scored": m["n_tokens_scored"]})
    return {"quantiles": list(QUANTILES), "per_grid_point": points}


def clusterer_contexts(explain: Path) -> dict[str, dict[int, str]]:
    """{pole: {rank: marked context}} as the clusterer saw them in its sample-0 requests."""
    reqs = requests_by_id(explain)
    out = {}
    for path in sorted((explain / "plans").glob("*-clusterer.jsonl")):
        for p in read_jsonl(path):
            if p["sample"] == 0:
                found = (re.match(r"#(\d+) \(.*?\) \| (.*)$", line) for line in reqs[p["id"]]["user"].splitlines())
                out[p["pole"]] = {int(m.group(1)): m.group(2) for m in found if m}
    return out


def gallery(tables: dict, poles: list[str], strata: dict[str, str], explain: Path, seed: int) -> list[dict]:
    """F6 (§10): per group (ONE; TWO-REL; UNREL or UNK, by the pole's Q1 class) one pole of S drawn with `seed`
    among those with at least two born non-echo clusters; per cluster its name, birth, r(c), presence per grid
    point and its first two exemplars as the clusterer saw them."""
    rng = np.random.default_rng(seed)
    seen = clusterer_contexts(explain)
    out = []
    for group in GALLERY_GROUPS:
        name = " or ".join(group)
        cands = [p for p in poles if strata[p] in group
                 and sum(not r["echo"] and r["birth_pct"] is not None for r in tables[p]) >= 2]
        if not cands:
            out.append({"group": name, "pole": None, "note": "no pole of S in this group has two born clusters"})
            continue
        pole = cands[int(rng.integers(len(cands)))]
        out.append({"group": name, "pole": pole, "class": strata[pole], "clusters": [
            {**{k: r[k] for k in ("cluster", "name", "is_main", "echo", "status", "birth_pct", "r_at_birth",
                                  "present_pct")},
             "exemplars": [seen[pole][k] for k in sorted(r["ranks"])[:2]]} for r in tables[pole]]})
    return out


def q2_readouts(grid_dirs: list, explain: Path, gate: dict, seed: int, b2: bool,
                strata: dict[str, str] | None = None,
                population: dict[str, int] | None = None) -> tuple[dict, dict]:
    tables = grid_cluster_tables(grid_dirs, explain, b2)
    poles = sorted(p for p, t in tables.items() if any(r["is_main"] for r in t))
    rows = [r for p in poles for r in tables[p] if not r["echo"]]
    judgments = read_jsonl(explain / "judgments.jsonl")
    check_judgments_match_plans(explain, judgments)
    tests, ctrls, declined = judged(judgments)
    primary = {r["cluster"]: r for r in rows if r["status"] == "late_primary" and r["cluster"] not in declined["clusters"]}
    if set(tests) != set(primary):
        raise ValueError(f"judged clusters != late-born primary clusters: missing "
                         f"{sorted(set(primary) - set(tests))[:5]}, extra {sorted(set(tests) - set(primary))[:5]}")
    ctrl_poles = [p for p in poles if p not in declined["control_poles"]]
    if set(ctrls) != set(ctrl_poles):
        raise ValueError(f"control poles != poles with a main: {sorted(set(ctrl_poles) ^ set(ctrls))[:5]}")
    for key, r in primary.items():
        r["acc_random"] = float(np.mean(tests[key]["random"]))
        r["acc_hard"] = float(np.mean(tests[key]["hard"]))
        r["flagged"] = sum(tests[key]["random"]) <= 1
    born = [r for r in rows if r["birth_pct"] is not None]
    m = {p: {s: sum(not c["echo"] for c in cs) for s, cs in samples.items()}
         for p, samples in clusters_by_pole(explain).items() if p in poles}
    echo = [r for t in tables.values() for r in t if r["echo"]]
    gone = declined_clusterer(explain)
    out = {"n_positions": len(read_json(explain / "positions_poles.json")), "n_poles": len(poles), "n_clusters": len(rows),
           "declined_clusterer": {"n": len(gone), "poles": gone}, "declined_judgments": declined,
           "poles_all_echo": sorted(set(tables) - set(poles)), "echo_clusters": len(echo),
           "status_counts": {s: sum(r["status"] == s for r in rows) for s in STATUSES if s != "echo"},
           "share_born_T_1": sum(r["birth_pct"] == T_FIRST_TRAINED for r in born) / len(born),
           "B3_late_split_agreement": float(np.mean([r["late_pct"] == r["late_raw"] for r in born])),
           "B4_m_within_1": float(np.mean([abs(v[0] - v[1]) <= 1 for v in m.values()])),
           "B4_unmatched_share": float(np.mean([not r["matched"] for r in rows])),
           "B2_split_changed_10m": (float(np.mean([r["split_changed_10m"] for r in born])) if b2
                                    else "PENDING (run with --b2)")}
    out["pivot_more_than_half_born_T_1"] = out["share_born_T_1"] > 0.5
    ctrl_acc = [float(np.mean(ctrls[p]["random"])) if p in ctrls else None for p in poles]
    cluster_acc = [[primary[r["cluster"]]["acc_random"] for r in tables[p] if r["cluster"] in primary] for p in poles]
    out["f_ctrl"] = float(np.mean([sum(ctrls[p]["random"]) <= 1 for p in ctrl_poles]))
    if gate["gate"] == "pass":
        pu = p_u_bootstrap(cluster_acc, ctrl_acc, N_BOOT, seed)
        pu.update(rule=p_u_rule(pu["ci95"]), valid=True, underpowered=pu["n_clusters"] < MIN_CLUSTERS_POWERED)
        flagged = [r for r in primary.values() if r["flagged"]]
        hard_ctrl = float(np.mean([np.mean(ctrls[p]["hard"]) for p in ctrl_poles]))
        out["p_u"] = pu
        out["flags"] = {"n_flagged": len(flagged), "share": len(flagged) / len(primary)}
        out["hard"] = {"accuracy": float(np.mean([r["acc_hard"] for r in primary.values()])),
                       "accuracy_on_flagged": (float(np.mean([r["acc_hard"] for r in flagged]))
                                               if flagged else None),
                       "ctrl_accuracy": hard_ctrl, "measurable": hard_ctrl >= CTRL_GATE}
        mig = [p for p in poles for r in tables[p] if r["is_main"] and r.get("flagged")]
        out["H_MIG"] = {"count": len(mig), "share_of_S": len(mig) / len(poles), "poles": mig}
    else:
        reason = f"judge validity gate failed: a_ctrl {gate['a_ctrl']:.3f} < {CTRL_GATE} on the development discovery set"
        out["p_u"] = out["hard"] = out["H_MIG"] = out["flags"] = {"valid": False, "reason": reason}
    order, order_poles, vest, vest_poles, undefined = [], [], [], [], 0
    for p in poles:
        main = next(r for r in tables[p] if r["is_main"])
        if main["birth_pct"] is not None:
            b = [r for r in tables[p] if not r["echo"] and r["birth_pct"] is not None]
            if len(b) >= 2:
                order.append(([r["birth_pct"] for r in b], b.index(main)))
                order_poles.append(p)
        counts = vestigial_counts(tables[p])
        if counts is not None:
            undefined += counts[2]
            if counts[1]:
                vest.append(counts[:2])
                vest_poles.append(p)
    out["birth_order"] = birth_order_test(order, N_PERM, seed)
    out["vestigial"] = vestigial(vest, N_BOOT, seed)
    out["vestigial"]["n_excluded_growth_undefined"] = undefined
    out["sensitivity_F2_excl_T7_T15"] = sensitivity_excluding_births(tables, poles, primary, ctrl_acc,
                                                                     gate["gate"] == "pass", seed)
    late_r = [primary[k]["r_at_birth"] for k in primary]
    out["r_at_birth"] = {"median_late_primary": float(np.median(late_r)) if late_r else None,
                         "median_all_born": float(np.median([r["r_at_birth"] for r in born]))}
    out["center_birth_curve"] = CENTER_NOT_COMPUTED
    out["births"] = birth_histogram(rows)
    out["grid_population"] = grid_population(grid_dirs)
    out["gallery"] = gallery(tables, poles, strata, explain, seed) if strata is not None else "PENDING (run with --q2-strata)"
    if strata is None:
        out["population_weighted"] = "PENDING (run with --q2-strata)"
    else:
        w, info = stratum_weights(poles, strata, population)
        wpole = dict(zip(poles, w))
        weighted = {"strata": info}
        if gate["gate"] == "pass":
            pw = p_u_bootstrap(cluster_acc, ctrl_acc, N_BOOT, seed, weights=w)
            weighted["p_u"] = {**pw, "rule": p_u_rule(pw["ci95"])}
        hits = [float(b[i] == min(b)) for b, i in order]
        bw, bci, _ = pooled_share(hits, [1.0] * len(hits), [wpole[p] for p in order_poles], N_BOOT, seed)
        vw, vci, _ = pooled_share([k for k, _ in vest], [n for _, n in vest], [wpole[p] for p in vest_poles], N_BOOT, seed)
        weighted["b"] = {"b": bw, "ci95": bci}
        weighted["v"] = {"v": vw, "ci95": vci, "label": "H_VEST" if vci[0] > VEST_LOWER else "not H_VEST"}
        out["population_weighted"] = weighted
    return out, tables


# --- CLI: unblind, samples and grid ----------------------------------------------------------

def q1_samples(records: list[dict], requests: dict) -> list[dict]:
    """The gallery: one passing and one failing pole per class (§10), or a line saying none exists."""
    out = []
    for cls in CLASSES:
        for which, want in (("passing", True), ("failing", False)):
            hit = next((r for r in records if r["class"] == cls and r.get("passes") is want), None)
            if hit is None:
                out.append({"class": cls, "which": which, "pole": None, "note": "no such pole in the data"})
                continue
            out.append({"class": cls, "which": which, **hit,
                        "explainer_saw": requests[f"explainer:{hit['pole']}:0"]["user"]})
    return out


def q2_samples(explain: Path, tables: dict) -> list[dict]:
    """The gallery: one correct and one wrong judgment, for tests and for controls, with the judge's
    text, the truth and the cluster's birth table."""
    requests = requests_by_id(explain)
    rows = {r["cluster"]: r for t in tables.values() for r in t}
    out = []
    judgments = read_jsonl(explain / "judgments.jsonl")
    for control in (False, True):
        for correct in (True, False):
            j = next((x for x in judgments if x["control"] is control and x["correct"] is correct), None)
            if j is None:
                out.append({"control": control, "correct": correct, "note": "no such judgment in the data"})
                continue
            out.append({**j, "judge_saw": requests[j["id"]]["user"], "birth_table": rows[j["cluster"]]})
    return out


def fmt_ci(ci) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def grid_markdown(metrics: dict) -> str:
    q1, q2 = metrics.get("q1", {}), metrics.get("q2")
    pend = "PENDING (no Q2 input)"
    if q2 is None:
        pu = bo = ve = mig = rc = hard = cb = (pend, pend)
    else:
        p = q2["p_u"]
        pu = ((f"p_u = {p['point']:.3f} {fmt_ci(p['ci95'])}; A = {p['A']:.3f}, a_ctrl = {p['a_ctrl']:.3f}, "
               f"{p['n_clusters']} clusters" + (" (underpowered)" if p["underpowered"] else "")
               + f"; pole bootstrap conditional on a late-born cluster and a control in each draw, "
               f"{p['degenerate_draws']} of {p['n_boot']} draws redrawn (refused above 1%)", p["rule"])
              if p.get("valid") else (p["reason"], "not valid"))
        b = q2["birth_order"]
        bo = (f"b = {b['b']:.3f} {fmt_ci(b['ci95'])}; null mean {b['null_mean']:.3f}, 97.5th "
              f"{b['null_97_5']:.3f}; p = {b['p']:.4f} (Holm {metrics['holm']['b']:.4f}); {b['n_poles']} poles",
              "main oldest (above band)" if b["supports_main_oldest"] else "drift (inside band)")
        v = q2["vestigial"]
        ve = (f"v = {v['v']:.3f} {fmt_ci(v['ci95'])}; p = {v['p']:.4f} (Holm {metrics['holm']['v']:.4f}); "
              f"{v['n_clusters']} clusters, {v['n_excluded_growth_undefined']} excluded (growth undefined)", v["label"])
        m = q2["H_MIG"]
        mig = ((f"{m['count']} poles ({m['share_of_S']:.3f} of S)", "count") if "count" in m
               else (m["reason"], "not valid"))
        r = q2["r_at_birth"]
        rc = (f"median r: late-born {r['median_late_primary']}, all born {r['median_all_born']:.3f}",
              "exploratory")
        h = q2["hard"]
        hard = ((f"accuracy {h['accuracy']:.3f}; on flagged {h['accuracy_on_flagged']}; "
                 f"control {h['ctrl_accuracy']:.3f}", "measurable" if h["measurable"] else "not measurable")
                if "accuracy" in h else (h["reason"], "not valid"))
        cb = ((q2["center_birth_curve"]["grid_text"], "exploratory") if isinstance(q2["center_birth_curve"], dict)
              else (q2["center_birth_curve"], "not computed"))
    q1_text = "; ".join(f"{k}: " + ", ".join(f"{cls} {v['fractions'][cls]['p']:.3f} "
                                               f"{fmt_ci(v['fractions'][cls]['wilson95'])}" for cls in CLASSES)
                        + f" (UNK-N {v['fractions']['UNK-N']['k']}, UNK-F {v['fractions']['UNK-F']['k']})"
                        for k, v in sorted(q1.items())) or "PENDING (no Q1 input)"
    pred = metrics["predictions"]
    lines = ["# D1 prediction grid (SPEC_D1 §11), filled once", "",
             f"PREDICTIONS entry: heading {pred['heading_utc']}, committed {pred['commit_utc']}. "
             f"θ_s = {metrics['theta']['theta_s']:.4f} (gate {metrics['theta']['gate']}). Judge validity gate: "
             f"{metrics['judge_gate']['gate'] if metrics['judge_gate'] else 'not given'}. "
             f"G2: {metrics['G2'] if isinstance(metrics['G2'], str) else round(metrics['G2']['share'], 3)}. "
             f"Population-weighted Q2 estimates: metrics.json q2.population_weighted."
             + ("" if q2 is None else
                f" Left out of Q2 (declined after the attempts cap): {q2['declined_clusterer']['n']} poles at the "
                f"clusterer; {q2['declined_judgments']['n']} judge lines, taking out "
                f"{len(q2['declined_judgments']['clusters'])} clusters and "
                f"{len(q2['declined_judgments']['control_poles'])} controls."), "",
             "| readout | H_RG predicts | H_SP predicts | H_VEST predicts | observed | label |",
             "|---|---|---|---|---|---|",
             f"| `p_u` (§6.3) | upper bound < 0.10 | lower bound > 0.20 | — | {pu[0]} | {pu[1]} |",
             f"| `b` birth order (§6.4) | above the permutation band | inside it | main not earliest in many poles | {bo[0]} | {bo[1]} |",
             f"| `v` vestigial (§6.5) | low | — | lower bound > 0.5 | {ve[0]} | {ve[1]} |",
             f"| H_MIG count (§6.6) | ≈ 0 | some | — | {mig[0]} | {mig[1]} |",
             f"| `r(c)` at birth (exploratory) | median well below 1 | no prediction | — | {rc[0]} | {rc[1]} |",
             f"| hard-decoy accuracy (secondary) | far above 0.5 | near 0.5 on flagged clusters | — | {hard[0]} | {hard[1]} |",
             f"| Q1 fractions (descriptive) | in the essay's regime with stated thresholds | — | — | {q1_text} | descriptive |",
             f"| center birth curve (exploratory) | monotone, early for ONE poles | — | — | {cb[0]} | {cb[1]} |",
             ""]
    return "\n".join(lines)


def kappa_text(k) -> str:
    return "undefined" if k is None else f"{k:.3f}"


def ci_text(x: dict) -> str:
    return f"{x['p']:.3f} {fmt_ci(x['wilson95'])}"


def results_markdown(metrics: dict, source: Path) -> str:
    """The RESULTS.md template (§6.2, §11): every number copied from `source` (verdict/metrics.json); the prose
    slots and the cost lines without a machine-readable source are labelled blanks."""
    cls = CLASSES + UNK_PARTS
    out = ["# D1 RESULTS", "",
           f"Numbers from `{source}` (sha256 {file_sha256(source)}, unblind {metrics['created_utc']}). "
           "Blanks marked ____ are written by hand; nothing below is re-derived.", "",
           "Finding: ____", "", "## Q1 audit (§6.2)", "",
           f"θ_s = {metrics['theta']['theta_s']:.4f} (gate {metrics['theta']['gate']}). The essay's 90 / 7 / 2–4 is "
           "its own estimate under its own thresholds, not ours.", ""]
    for key, q in sorted(metrics["q1"].items()):
        f = q["fractions"]
        out += [f"### {key}", "",
                f"On model {key}, {100 * f['ONE']['p']:.1f}% of poles ({100 * f['ONE']['wilson95'][0]:.1f}–"
                f"{100 * f['ONE']['wilson95'][1]:.1f}%, Wilson 95%) have one job under explanations that pass §6.1 "
                f"(n = {f['n']}).", "",
                (f"Poles: the {q['poles_list']['n']} listed in `{q['poles_list']['path']}` (sha256 "
                 f"{q['poles_list']['sha256'][:16]}); {q['poles_list']['excluded_by_list']['n']} poles of the explain "
                 f"directory left out by the list: {', '.join(q['poles_list']['excluded_by_list']['poles']) or 'none'}."
                 if q["poles_list"] else "Poles: every pole of the explain directory (no list given)."), "",
                "| set | n | " + " | ".join(cls) + " |", "|---|---|" + "---|" * len(cls),
                f"| all | {f['n']} | " + " | ".join(ci_text(f[c]) for c in cls) + " |"]
        for t, ft in sorted(q["by_depth_tercile"].items()):
            out.append(f"| depth tercile {t} | {ft['n']} | " + " | ".join(ci_text(ft[c]) for c in cls) + " |")
        for key, label in (("double_negative_ge_0.8", "double-negative share ≥ 0.8"),
                           ("double_negative_lt_0.8", "double-negative share < 0.8")):
            if q[key]["n"]:
                out.append(f"| {label} | {q[key]['n']} | " + " | ".join(ci_text(q[key][c]) for c in cls) + " |")
        for model, m in q["by_explainer_model"].items():
            fm = m["fractions"]
            out.append(f"| explainer {model} | {fm['n']} | " + " | ".join(ci_text(fm[c]) for c in cls) + " |")
        km = q.get("kappa_models")
        out += ["", (f"Label agreement: kappa {kappa_text(q['kappa_label'])} over {q['n_rated_twice']} poles rated twice; "
                     f"explainer {km['explainer']}, second reader {km['second_reader']}, {km['cross_model_pairs']} "
                     "cross-model pairs." if km else "Label agreement: no pole rated twice."),
                "Kappa by explainer model: " + "; ".join(
                    f"{model} {kappa_text(m.get('kappa_label'))} (n = {m['n_rated_twice']})"
                    for model, m in q["by_explainer_model"].items()) + ".",
                f"Declined: {q['declined']['n']} poles (explainer or writer; UNK-N), {q['declined']['second_reader']} "
                "second-reader answers.", ""]
    q2 = metrics.get("q2")
    out += ["## Q2 parent test and the prediction grid (§6.3–§6.6, §11)", ""]
    if q2 is None:
        out += ["PENDING (no Q2 input).", ""]
    else:
        gate = metrics["judge_gate"]
        dj = q2["declined_judgments"]
        out += [f"Judge validity gate: a_ctrl {gate['a_ctrl']:.3f} on {gate['n_poles']} poles ({gate['gate']}; "
                f"threshold {gate['threshold']}), judge {gate['judge']['models']}.",
                f"Declined at W4: {q2['declined_clusterer']['n']} poles at the clusterer "
                f"({', '.join(q2['declined_clusterer']['poles']) or 'none'}); {dj['n']} judge lines, taking out "
                f"clusters {dj['clusters'] or 'none'} and controls {dj['control_poles'] or 'none'}. "
                f"S: {q2['n_positions']} poles, {q2['n_poles']} with a main cluster.", ""]
        s = q2.get("sensitivity_F2_excl_T7_T15")
        if s:
            sp, sb = s["p_u"], s["birth_order"]
            out += [f"Sensitivity (checker F2 item 3; clusters born at {' or '.join(s['excluded_births'])} left out): "
                    + (f"p_u = {sp['point']:.3f} {fmt_ci(sp['ci95'])} ({sp['rule']}; {sp['n_clusters']} clusters, "
                       f"{sp['n_clusters_left_out']} left out)" if sp else "p_u not valid (gate failed)")
                    + f"; b = {sb['b']:.3f} {fmt_ci(sb['ci95'])}, null 97.5th {sb['null_97_5']:.3f}, p = {sb['p']:.4f} "
                    f"({sb['n_poles']} poles; {sb['n_clusters_left_out']} clusters and {sb['n_poles_left_out_main_excluded']} "
                    "poles whose main is excluded left out). A labelled sensitivity; the §6.3 and §6.4 labels are not reapplied.", ""]
    out += grid_markdown(metrics).split("\n")[2:]
    cbc = q2.get("center_birth_curve") if q2 else None
    if isinstance(cbc, dict):
        out += ["## Center birth curve (§2.4, exploratory)", "",
                f"{cbc['rule']}; a pole with max|a| = 0 at a checkpoint scores 0 there ({cbc['dead_pole_checkpoints']} "
                "pole-checkpoints). Per-checkpoint probe run: `" + cbc["inputs"]["center_probe"] + "`.", "",
                "| set | n | undefined | median first t (blind max) | median first t (positives' mean) | non-decreasing share |",
                "|---|---|---|---|---|---|"]
        rows = [("all", cbc["blind_max"], cbc["positives_mean"])] + [(cls, v["blind_max"], v["positives_mean"]) for cls, v in cbc["by_class"].items()]
        for name, bm, pm in rows:
            out.append(f"| {name} | {bm['n_poles']} | {bm['n_undefined_no_T15_score']} | {bm['median_first_t']} | {pm['median_first_t']} | "
                       + (f"{bm['share_nondecreasing']:.3f}" if bm["share_nondecreasing"] is not None else "—") + " |")
        out += ["", "Median ratio curve (blind maximum's score over its T_15 score) by grid point: "
                + ", ".join(f"{g} {r:.3f}" for g, r in zip(GRID_LABELS, cbc["blind_max"]["median_ratio_curve"] or [])) + ".",
                "First-t counts (blind maximum): " + ", ".join(f"{k} {v}" for k, v in cbc["blind_max"]["first_t_counts"].items() if v) + ".", ""]
    out += ["## Instrument readouts", ""]
    for key, q in sorted(metrics["q1"].items()):
        x3, x7, u = q["X3_blind_max_ge_corpus_max"], q["X7_pass_rate_by_encoding_flag"], q["untestable_phrase"]
        out += [f"- {key} X3 (blind maximum at or above the corpus maximum): "
                + ", ".join(f"{c} {v['k']}/{v['n']}" for c, v in x3.items() if "k" in v) + ".",
                f"- {key} X7 (pass rate by encoding flag): "
                + ", ".join(f"{flag} {v['k']}/{v['n']}" if "k" in v else f"{flag} n = 0" for flag, v in x7.items()) + ".",
                f"- {key} explanations naming an untestable job: {u['k']}/{u['n']} ({u['reading']})."]
    out += ["", "## Cost", "",
            "- GPU: ____ (no machine-readable source; JOURNAL and SPEC §12 hold the record).",
            "- API: ____ (no machine-readable dollar source; responses/*.usage.jsonl hold token counts per call).", ""]
    return "\n".join(out)


def cmd_results_template(a) -> None:
    source = Path(a.unblind) / "verdict" / "metrics.json"
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(f"{out} exists")
    out.write_text(results_markdown(read_json(source), source))
    print(out)


def first_judge_batch_utc(explain: Path) -> str:
    utcs = [b["utc"] for b in read_json(explain / "manifest.json")["batches"] if "judge" in b["roles"]]
    if not utcs:
        raise ValueError(f"{explain}: no judge batch in the manifest")
    return min(utcs, key=parse_utc)


def q1_arg(text: str) -> tuple[str, str, str, str, str | None]:
    """--q1 KEY=STATS,PROBE,EXPLAIN[,poles=PATH] -> (key, stats, probe, explain, pole list path or None)."""
    key, value = text.split("=", 1)
    parts = value.split(",")
    if len(parts) == 3:
        return key, *parts, None
    if len(parts) == 4 and parts[3].startswith("poles="):
        return key, *parts[:3], parts[3][len("poles="):]
    raise ValueError(f"--q1 {text!r}: expected KEY=STATS,PROBE,EXPLAIN[,poles=PATH]")


def listed_records(recs: list[dict], poles_path: str) -> tuple[list[dict], dict]:
    """§4.4/§6.2: a Q1 cell's readout over its listed poles only (the audit sample). Returns the listed records
    and the record of the list: path, SHA-256, count, and the explain directory's other poles as excluded_by_list.
    A listed pole without a Q1 record is refused."""
    listed = read_json(poles_path)
    have = {r["pole"] for r in recs}
    missing = sorted(set(listed) - have)
    if missing:
        raise ValueError(f"{poles_path}: {len(missing)} listed poles have no Q1 record: {missing[:10]}")
    excluded = sorted(have - set(listed))
    return ([r for r in recs if r["pole"] in set(listed)],
            {"path": str(poles_path), "sha256": file_sha256(poles_path), "n": len(listed),
             "excluded_by_list": {"n": len(excluded), "poles": excluded}})


def cmd_unblind(a) -> None:
    q1_args = [q1_arg(x) for x in a.q1]
    grid = a.q2_grid.split(",") if a.q2_grid else None
    if not q1_args and grid is None:
        raise ValueError("give --q1 and/or --q2-grid")
    if (grid is None) != (a.q2_explain is None) or (grid is not None and a.judge_gate is None):
        raise ValueError("--q2-grid, --q2-explain and --judge-gate go together")
    if a.q2_strata is not None and a.q2_strata not in {k for k, *_ in q1_args}:
        raise ValueError(f"--q2-strata {a.q2_strata} names no --q1 model")
    if a.center_probe and (grid is None or a.q2_strata is None):
        raise ValueError("--center-probe needs --q2-grid and --q2-strata")
    manifests = [Path(probe) / HARVEST_FILES["manifest"] for _, _, probe, _, _ in q1_args]
    manifests += [Path(e) / "explain" / "manifest.json" for _, _, _, e, _ in q1_args]
    if grid is not None:
        manifests += [Path(g) / HARVEST_FILES["manifest"] for g in grid]
        manifests.append(Path(a.q2_explain) / "explain" / "manifest.json")
    predictions = predictions_gate(a.journal, manifests)
    theta_rec = read_json(a.theta)
    if theta_rec["gate"] != "pass":
        refuse(f"{a.theta}: θ_s gate is {theta_rec['gate']}; the instrument does not run")
    for _, _, _, e, _ in q1_args:
        started = read_json(Path(e) / "explain" / "manifest.json")["started_utc"]
        if not parse_utc(theta_rec["created_utc"]) < parse_utc(started):
            refuse(f"{a.theta}: θ_s was written {theta_rec['created_utc']}, after {e} started ({started})")
    gate = read_json(a.judge_gate) if a.judge_gate else None
    if gate is not None:
        first = first_judge_batch_utc(Path(a.q2_explain) / "explain")
        if not parse_utc(gate["created_utc"]) < parse_utc(first):
            refuse(f"{a.judge_gate}: written {gate['created_utc']}, after the first S judge batch ({first})")
        s_judge = judge_identity(Path(a.q2_explain) / "explain")
        if s_judge != gate["judge"]:
            refuse(f"{a.judge_gate}: the gate measured the judge {gate['judge']}; the S judgments come from "
                   f"{s_judge}. The validity gate holds only for the judge that runs on S (§2.4, §5)")
    out = Path(a.out) / "verdict"
    if (out / "metrics.json").exists():
        raise FileExistsError(f"{out / 'metrics.json'} exists; the grid is filled once")
    metrics = {"created_utc": utc_now(), "predictions": predictions,
               "gated_manifests": [str(m) for m in manifests], "theta": theta_rec, "judge_gate": gate,
               "q1": {}, "G2": g2_readout(Path(a.t0_explanations) / "explain") if a.t0_explanations
               else "PENDING (run with --t0-explanations)"}
    samples, q1_classes = {}, {}
    for key, stats, probe, explain, poles_path in q1_args:
        e = Path(explain) / "explain"
        recs = q1_records(stats, probe, e, theta_rec["theta_s"])
        listing = None
        if poles_path is not None:
            recs, listing = listed_records(recs, poles_path)
        q1_classes[key] = {r["pole"]: r["class"] for r in recs}
        metrics["q1"][key] = {**q1_metrics(recs, theta_rec["theta_s"]), "poles_list": listing,
                              "inputs": {**input_hashes({"explanations": e / "explanations.jsonl",
                                                         "texts": e / "texts.jsonl",
                                                         "scores": Path(probe) / HARVEST_FILES["probe"]}),
                                         **probe_rule(probe)}}
        reqs = requests_by_id(e)
        samples[f"q1_{key}"] = q1_samples(recs, reqs)
        samples[f"q1_{key}_poles"] = [{**r, "explainer_saw": reqs[f"explainer:{r['pole']}:0"]["user"]} for r in recs]   # §10: per pole, the contexts as the explainer saw them
    if grid is not None:
        e = Path(a.q2_explain) / "explain"
        strata = q1_classes[a.q2_strata] if a.q2_strata else None
        population = (dict(Counter(c for c in strata.values() if c != "RARE")) if strata is not None else None)
        q2, tables = q2_readouts(grid, e, gate, a.seed, a.b2, strata, population)
        q2["inputs"] = {"grid": grid, **input_hashes({"judgments": e / "judgments.jsonl",
                                                      "clusters": e / "clusters.jsonl"})}
        if a.center_probe:
            texts_explain = next(Path(ex) / "explain" for k, _, _, ex, _ in q1_args if k == a.q2_strata)
            s_poles = sorted(p for p, tb in tables.items() if any(r["is_main"] for r in tb))
            q2["center_birth_curve"] = center_birth_curve(Path(a.center_probe), grid, texts_explain, s_poles, strata)
        metrics["q2"] = q2
        metrics["holm"] = holm({"b": q2["birth_order"]["p"], "v": q2["vestigial"]["p"]})
        samples["q2_parent"] = q2_samples(e, tables)
        judge_reqs = requests_by_id(e)
        by_cluster: dict[str, list] = {}
        for j in read_jsonl(e / "judgments.jsonl"):             # §10: per cluster, the two records as the judge saw them, answer and truth
            by_cluster.setdefault(j["cluster"], []).append({**j, "judge_saw": judge_reqs[j["id"]]["user"]})
        samples["q2_clusters"] = [{**r, "judgments": by_cluster.get(r["cluster"], [])}
                                  for t in tables.values() for r in t]
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in samples.items():
        write_jsonl_new(out / "samples" / f"{name}.jsonl", rows)
    (out / "grid.md").write_text(grid_markdown(metrics))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    print(out / "grid.md")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="neurons_verdict")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("theta", help="θ_s from the discovery set (§6.1)")
    t.add_argument("--stats", required=True, help="harvest out/ with stats/ (corpus max|a|)")
    t.add_argument("--probe", required=True, help="harvest out/ with poles/probe/scores.npz")
    t.add_argument("--explain", required=True, help="directory holding explain/")
    t.add_argument("--out", required=True)
    w = sub.add_parser("draw-s", help="S and S_gate for the W3 grid (§4.4), RARE excluded by stats only")
    w.add_argument("--stats", required=True, help="the W1 harvest out/ with stats/ (O7 T_15, 50M)")
    w.add_argument("--audit", required=True, help="audit_poles.json")
    w.add_argument("--extra", required=True, help="extra_poles.json (the development discovery set)")
    w.add_argument("--seed", type=int, required=True, help="recorded; the draw takes no random step")
    w.add_argument("--out", required=True)
    d = sub.add_parser("discovery", help="the discovery set's report for one prompt version (§9.2 W2-D)")
    d.add_argument("--theta", required=True, help="verdict/theta.json set on this discovery set")
    d.add_argument("--stats", required=True, help="harvest out/ with stats/ (corpus max|a|)")
    d.add_argument("--probe", required=True, help="harvest out/ with poles/probe/scores.npz")
    d.add_argument("--explain", required=True, help="directory holding explain/")
    d.add_argument("--out", required=True)
    j = sub.add_parser("judge-gate", help="positive-control accuracy on the development discovery set")
    j.add_argument("--explain", required=True)
    j.add_argument("--out", required=True)
    u = sub.add_parser("unblind", help="classes, p_u, b, v, H_MIG, grid (§6, §11)")
    u.add_argument("--journal", required=True, help="studies/already-clean/JOURNAL.md in its git repository")
    u.add_argument("--theta", required=True, help="verdict/theta.json from the discovery set")
    u.add_argument("--judge-gate", help="verdict/judge_gate.json (required with --q2-grid)")
    u.add_argument("--q1", action="append", default=[], metavar="KEY=STATS,PROBE,EXPLAIN[,poles=PATH]",
                   help="a Q1 cell; with poles=PATH its readout is over the listed poles only (§6.2)")
    u.add_argument("--q2-grid", help="comma-separated harvest out/ dirs T_0..T_15")
    u.add_argument("--q2-explain", help="directory holding explain/ for S")
    u.add_argument("--q2-strata", help="the --q1 KEY whose classes weight S to the population (§4.4)")
    u.add_argument("--b2", action="store_true", help="recompute births on snapshots/band1 (10M)")
    u.add_argument("--t0-explanations", help="directory holding explain/explanations.jsonl from a T_0 harvest (G2)")
    u.add_argument("--center-probe", help="per-checkpoint probe run: <dir>/<grid step name>/ (a harvest out dir with "
                   "poles/probe/scores.npz) for every --q2-grid step (§2.4 center birth curve; needs --q2-strata)")
    u.add_argument("--seed", type=int, required=True)
    u.add_argument("--out", required=True)
    r = sub.add_parser("results-template", help="RESULTS.md laid out from an unblind's metrics.json")
    r.add_argument("--unblind", required=True, help="the unblind --out directory (holding verdict/metrics.json)")
    r.add_argument("--out", required=True, help="the RESULTS.md to write (refused if it exists)")
    a = ap.parse_args(argv)
    {"theta": cmd_theta, "draw-s": cmd_draw_s, "discovery": cmd_discovery, "judge-gate": cmd_judge_gate,
     "unblind": cmd_unblind, "results-template": cmd_results_template}[a.cmd](a)


if __name__ == "__main__":
    main()
