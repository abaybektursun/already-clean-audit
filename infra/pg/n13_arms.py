"""N13 weights instrument (studies/next/07_PICKS/N13_SPEC.md §0.5, §3.1-3.4, §5 PC-RANK and PC-ART, §6.1 G-INIT, G-PARTITION,
G-HASH, G-MASK, §6.3 H_artifact). NumPy and the standard library only, so the same code runs on the laptop (stand-in weights) and on
the instance (real weights, memory-mapped).

A bf16 tensor is handled as its uint16 bit pattern. f32(u) is exact; bf16(x) rounds to nearest even, as torch's .to(torch.bfloat16).
Safetensors files are read by memory map (Weights.get) and written by stream (write_arm), one tensor at a time.

Conventions fixed here where the spec leaves them open (each is listed in N13_OFFLINE_GATE.md "Spec issues"):
  - a random stream per tensor: default_rng([seed, layer, w]) with w = 0, 1, 2 for gate, up, down, so a draw does not depend on the
    order in which tensors are visited;
  - the builder takes the source bits where M = 1 and c = 1 (and asserts that the §3.1 formula gives the same bits wherever the change
    is non-zero), because the formula maps a +0/-0 pair to +0 and would break the bitwise G-HASH;
  - PC-RANK's failing input multiplies row h = n mod hidden of down_proj only (gate and up untouched);
  - PC-ART's synthetic change P+ uses default_rng([40, 1, layer, w]); P- (A2) plants neurons with default_rng(41), signs default_rng(42).
Fail-loud: no try/except, no fallback.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import struct

import numpy as np

MLP = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
ATTN = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
NORM = ("self_attn.q_norm", "self_attn.k_norm", "post_attention_layernorm", "post_feedforward_layernorm")
FAMILIES = ("mlp", "attn", "norm", "embed")
K_FRACTIONS = (0.001, 0.003, 0.01, 0.03)
K_TAGS = ("k0.1", "k0.3", "k1", "k3")
PRIMARY = 2                                   # k = 1%
SEED = {"rndE": 10, "rndT": 20, "dare": 30, "plant": 40, "rnd0": 50, "redraw": 60, "rndT_d2": 120, "dare_d2": 130,
        "rndE_round": 140}                             # A30: the rndE stochastic-rounding streams, default_rng([140 + j, layer, w])
ENERGY_PASSES = 3                                      # A30: at most three builds to bring a rescaled arm within the energy band
A31_REL_TOL = 1e-12                                    # A31: written against matched MLP energy, relative; the float64 bound for
                                                       # two orders of 96 non-negative terms is 2 x 95 x 2^-53 = 2.1e-14
N_REDRAW = 20
COPY_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "chat_template.jinja")


# ------------------------------------------------------------------------------------------------ names, shapes, partition
def expected_shapes(cfg: dict) -> dict:
    L, H, W, V = cfg["num_hidden_layers"], cfg["hidden_size"], cfg["intermediate_size"], cfg["vocab_size"]
    hd = H // cfg["num_attention_heads"]
    q, kv = cfg["num_attention_heads"] * hd, cfg["num_key_value_heads"] * hd
    s = {"model.embed_tokens.weight": (V, H), "lm_head.weight": (V, H), "model.norm.weight": (H,)}
    for l in range(L):
        p = f"model.layers.{l}."
        s |= {p + "mlp.gate_proj.weight": (W, H), p + "mlp.up_proj.weight": (W, H), p + "mlp.down_proj.weight": (H, W),
              p + "self_attn.q_proj.weight": (q, H), p + "self_attn.k_proj.weight": (kv, H), p + "self_attn.v_proj.weight": (kv, H),
              p + "self_attn.o_proj.weight": (H, q), p + "self_attn.q_norm.weight": (q,), p + "self_attn.k_norm.weight": (kv,),
              p + "post_attention_layernorm.weight": (H,), p + "post_feedforward_layernorm.weight": (H,)}
    return s


def family(name: str) -> str:
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        return "embed"
    if name == "model.norm.weight":
        return "norm"
    assert name.startswith("model.layers.") and name.endswith(".weight"), name
    mid = name.split(".", 3)[3][: -len(".weight")]
    for fam, keys in (("mlp", MLP), ("attn", ATTN), ("norm", NORM)):
        if mid in keys:
            return fam
    raise ValueError(f"parameter outside the four families: {name}")


def layer_of(name: str) -> int:
    return int(name.split(".")[2])


def which_mlp(name: str) -> int:
    return [k in name for k in MLP].index(True)


def partition(names: list, cfg: dict, total: int) -> dict:
    """G-PARTITION: every name in one family, the names equal the config's, the four families sum to `total`."""
    exp = expected_shapes(cfg)
    fam = {n: family(n) for n in names}
    per = {f: int(sum(int(np.prod(exp[n])) for n in names if n in exp and fam[n] == f)) for f in FAMILIES}
    v = {"n_names": len(names), "n_expected": len(exp), "missing": sorted(set(exp) - set(names)), "extra": sorted(set(names) - set(exp)),
         "per_family": per, "sum": sum(per.values()), "total": total}
    v["pass"] = bool(not v["missing"] and not v["extra"] and v["sum"] == total)
    return v


def neuron_count(cfg: dict) -> int:
    return cfg["num_hidden_layers"] * cfg["intermediate_size"]


def k_grid(n_neurons: int, fractions=K_FRACTIONS) -> list:
    return [int(round(f * n_neurons)) for f in fractions]


# ------------------------------------------------------------------------------------------------ safetensors by memory map
def st_header(path) -> tuple:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


class Weights:
    """One safetensors snapshot directory: model.safetensors.index.json plus its shards. get() returns a read-only uint16 memmap."""

    def __init__(self, d):
        self.dir = pathlib.Path(d)
        self.index = json.loads((self.dir / "model.safetensors.index.json").read_text())
        self.weight_map = self.index["weight_map"]
        self._h = {}

    def names(self) -> list:
        return sorted(self.weight_map)

    def get(self, name: str) -> np.ndarray:
        f = self.dir / self.weight_map[name]
        if f not in self._h:
            self._h[f] = st_header(f)
        h, off = self._h[f]
        e = h[name]
        assert e["dtype"] == "BF16", (name, e["dtype"])
        a, b = e["data_offsets"]
        shape = tuple(e["shape"])
        assert b - a == 2 * int(np.prod(shape)), (name, a, b, shape)
        return np.memmap(f, dtype="<u2", mode="r", offset=off + a, shape=shape)


def tensor_sha256(u16: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(u16, dtype="<u2").tobytes()).hexdigest()


def file_sha256(path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def write_safetensors(path, names: list, shapes: dict, produce):
    """Stream one file: the header from shapes, then each tensor from produce(name) (uint16), in `names` order."""
    header, off = {}, 0
    for n in names:
        nb = 2 * int(np.prod(shapes[n]))
        header[n] = {"dtype": "BF16", "shape": list(shapes[n]), "data_offsets": [off, off + nb]}
        off += nb
    header["__metadata__"] = {"format": "pt"}
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    tmp = pathlib.Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for n in names:
            t = produce(n)
            assert t.dtype == np.uint16 and tuple(t.shape) == tuple(shapes[n]), (n, t.dtype, t.shape)
            f.write(np.ascontiguousarray(t, dtype="<u2").tobytes())
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------ bf16 arithmetic
def f32(u16) -> np.ndarray:
    return (np.asarray(u16).astype(np.uint32) << 16).view(np.float32)


def bf16_stochastic(x, rng) -> np.ndarray:
    """A30: x (float32) to bf16, to the neighbour away from zero with probability equal to x's fractional position between its two
    bf16 neighbours (the low 16 bits / 2^16), to the neighbour toward zero otherwise; exact bf16 values are kept. Unbiased in value."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    assert np.isfinite(x).all(), "non-finite value before the bf16 cast"
    u = x.view(np.uint32)
    p = (u & np.uint32(0xFFFF)).astype(np.float64) / 65536.0
    up = rng.random(x.shape) < p
    out = ((u >> 16) + up.astype(np.uint32)).astype(np.uint16)
    assert np.isfinite(f32(out)).all(), "non-finite value after the stochastic bf16 cast"
    return out


def bf16(x) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    assert np.isfinite(x).all(), "non-finite value before the bf16 cast"
    u = x.view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) >> 16).astype(np.uint16)


def ordered(u16) -> np.ndarray:
    i = np.asarray(u16).view(np.int16).astype(np.int32)
    return np.where(i >= 0, i, -(i & 0x7FFF))


def from_ordered(o) -> np.ndarray:
    o = np.asarray(o, dtype=np.int32)
    return np.where(o >= 0, o, (-o) | 0x8000).astype(np.uint16)


def bit_distance(b16, a16) -> np.ndarray:
    return np.abs(ordered(a16) - ordered(b16))


def exponent(u16) -> np.ndarray:
    return np.maximum((np.asarray(u16) >> 7) & 0xFF, 1).astype(np.int32)


def spacing(u16) -> np.ndarray:
    """2^(floor(log2|w|) - 7) for normal w; 2^-133 for zero and subnormal w."""
    return np.ldexp(np.float32(1.0), exponent(u16) - 134).astype(np.float32)


def delta(b16, a16) -> np.ndarray:
    return f32(a16) - f32(b16)


def inexact(b16, a16) -> np.ndarray:
    """Entries whose float32 difference is not exact: both non-zero and exponents more than 16 apart (§0.5)."""
    nz = ((np.asarray(b16) & 0x7FFF) != 0) & ((np.asarray(a16) & 0x7FFF) != 0)
    return nz & (np.abs(exponent(a16) - exponent(b16)) > 16)


def inexact_count(b16, a16) -> int:
    return int(inexact(b16, a16).sum())


def signed_zero_swaps(b16, a16) -> int:
    b, a = np.asarray(b16), np.asarray(a16)
    return int((((b & 0x7FFF) == 0) & ((a & 0x7FFF) == 0) & (a != b)).sum())


# ------------------------------------------------------------------------------------------------ neuron scores
def neuron_sq(dg, du, dd) -> np.ndarray:
    """Per-neuron squared change: row n of gate and up, column n of down; float64 sums."""
    return (np.square(dg, dtype=np.float64).sum(1) + np.square(du, dtype=np.float64).sum(1)
            + np.square(dd, dtype=np.float64).sum(0))


def layer_triplet(w: Weights, l: int) -> tuple:
    p = f"model.layers.{l}."
    return tuple(w.get(p + k + ".weight") for k in MLP)


def weight_stats(before: Weights, after: Weights, cfg: dict) -> tuple:
    """s^2 and s'^2 per neuron [L, W]; per-family counts of unchanged, one-step and above-floor entries, energy, inexact entries,
    signed-zero swaps; the unchanged share of MLP entries."""
    L, W = cfg["num_hidden_layers"], cfg["intermediate_size"]
    s2, s2f = np.zeros((L, W)), np.zeros((L, W))
    fam = {f: {"entries": 0, "unchanged": 0, "one_step": 0, "above_floor": 0, "energy": 0.0, "inexact": 0, "signed_zero_swaps": 0}
           for f in FAMILIES}
    sq = {}
    for name in before.names():
        b, a = before.get(name), after.get(name)
        d = bit_distance(b, a)
        D = delta(b, a)
        r = fam[family(name)]
        r["entries"] += int(d.size)
        r["unchanged"] += int((d == 0).sum())
        r["one_step"] += int((d == 1).sum())
        r["above_floor"] += int((d >= 2).sum())
        sq[name] = (float(np.square(D, dtype=np.float64).sum()), float(np.square(f32(b), dtype=np.float64).sum()))
        r["energy"] += sq[name][0]
        r["inexact"] += inexact_count(b, a)
        r["signed_zero_swaps"] += signed_zero_swaps(b, a)
    for l in range(L):
        bt, at = layer_triplet(before, l), layer_triplet(after, l)
        Ds = [delta(b, a) for b, a in zip(bt, at)]
        s2[l] = neuron_sq(*Ds)
        Df = [np.where(bit_distance(b, a) >= 2, D, np.float32(0)) for b, a, D in zip(bt, at, Ds)]
        s2f[l] = neuron_sq(*Df)
    m = fam["mlp"]
    total_e = sum(r["energy"] for r in fam.values())
    stats = {"families": fam, "mlp_unchanged_share": m["unchanged"] / m["entries"],
             "energy_share": {f: (r["energy"] / total_e if total_e else None) for f, r in fam.items()},
             "block_profile": block_profile(sq, cfg)}
    return s2, s2f, stats


def block_profile(sq: dict, cfg: dict) -> dict:
    """N13_SPEC.md A20 (informational; no rule reads it): the relative weight change ||after - before|| / ||before|| (Frobenius norm
    over the group's tensors, float64 sums of squares) per transformer block 0..L-1, for the attention projections (q, k, v, o) and the
    MLP (gate, up, down) separately, and for the embedding and the output head. sq[name] = (sum of squared change, sum of squared
    before-weights). The norm weights (q_norm, k_norm, the two block layer norms, the final norm) are in no group."""
    L = cfg["num_hidden_layers"]
    groups = {"attn": [[f"model.layers.{l}.{k}.weight" for k in ATTN] for l in range(L)],
              "mlp": [[f"model.layers.{l}.{k}.weight" for k in MLP] for l in range(L)]}
    out = {"what": "||theta_after - theta_before|| / ||theta_before|| per group (N13_SPEC.md A20); attn = q, k, v, o projections, "
                   "mlp = gate, up, down; norms in no group", "n_blocks": L}
    for g, blocks in groups.items():
        d = [sum(sq[n][0] for n in names) for names in blocks]
        b = [sum(sq[n][1] for n in names) for names in blocks]
        out[g] = [float(np.sqrt(x / y)) for x, y in zip(d, b)]
        out[g + "_sq"] = {"delta": d, "before": b}
    for g, n in (("embed", "model.embed_tokens.weight"), ("lm_head", "lm_head.weight")):
        out[g] = float(np.sqrt(sq[n][0] / sq[n][1]))
        out[g + "_sq"] = {"delta": sq[n][0], "before": sq[n][1]}
    return out


def unchanged_mlp_share(before: Weights, after: Weights, cfg: dict) -> float:
    n = u = 0
    for l in range(cfg["num_hidden_layers"]):
        for b, a in zip(layer_triplet(before, l), layer_triplet(after, l)):
            d = bit_distance(b, a)
            n += d.size
            u += int((d == 0).sum())
    return u / n


# ------------------------------------------------------------------------------------------------ ranking and sets
def order(s2: np.ndarray) -> np.ndarray:
    """Flat neuron indices (l * W + n) by s descending, ties by (layer, index) ascending."""
    flat = np.asarray(s2, dtype=np.float64).ravel()
    return np.lexsort((np.arange(flat.size), -flat))


def jaccard(a, b) -> float:
    a, b = set(map(int, a)), set(map(int, b))
    return len(a & b) / len(a | b) if (a | b) else 1.0


def energy(s2_flat, idx) -> float:
    return float(np.asarray(s2_flat, dtype=np.float64)[np.asarray(idx, dtype=np.int64)].sum())


def rnd_e(s2_flat, top, tail, seed) -> dict:
    R = np.sort(np.random.default_rng(seed).choice(tail, size=len(top), replace=False))
    eS, eR = energy(s2_flat, top), energy(s2_flat, R)
    assert eR > 0, "rndE draw holds no change"
    return {"neurons": R.tolist(), "c": float(np.sqrt(eS / eR)), "E_S": eS, "E_R": eR, "seed": seed}


def rnd_t(s2_flat, top, tail, seed) -> dict:
    perm = np.random.default_rng(seed).permutation(tail)
    cum = np.cumsum(np.asarray(s2_flat, dtype=np.float64)[perm])
    eS = energy(s2_flat, top)
    short = bool(cum[-1] < eS)
    m = len(perm) if short else int(np.searchsorted(cum, eS, side="left")) + 1
    sel = np.sort(perm[:m])
    return {"neurons": sel.tolist(), "m": m, "E_S": eS, "E": float(cum[m - 1]), "short": short,
            "shortfall": float(eS - cum[-1]) if short else 0.0, "seed": seed}


def rnd_plain(tail, k, seed) -> dict:
    return {"neurons": np.sort(np.random.default_rng(seed).choice(tail, size=k, replace=False)).tolist(), "seed": seed}


def dare_mask(seed: int, l: int, w: int, shape, q: float) -> np.ndarray:
    return np.random.default_rng([seed, l, w]).random(shape, dtype=np.float32) < np.float32(q)


def make_sets(s2: np.ndarray, s2f: np.ndarray, ks: list, n_neurons: int) -> dict:
    """Every neuron set of §3.2 from the scores: top, tail energy, rndE, rndT, rnd0, second draws, topF, DARE q."""
    flat = s2.ravel()
    o = order(s2)
    of = order(s2f)
    sets = {"n_neurons": n_neurons, "ks": ks, "k_tags": list(K_TAGS), "top": {}, "rndE": {}, "rndT": {}, "dare_q": {}, "E_total": float(flat.sum())}
    for j, (tag, k) in enumerate(zip(K_TAGS, ks)):
        top, tail = np.sort(o[:k]), o[k:]
        sets["top"][tag] = top.tolist()
        sets["rndE"][tag] = rnd_e(flat, top, tail, SEED["rndE"] + j)
        sets["rndT"][tag] = rnd_t(flat, top, tail, SEED["rndT"] + j)
        sets["dare_q"][tag] = k / n_neurons
        sets.setdefault("E_top", {})[tag] = energy(flat, top)
        sets.setdefault("E_tail", {})[tag] = float(flat.sum() - energy(flat, top))
    j, tag, k = PRIMARY, K_TAGS[PRIMARY], ks[PRIMARY]
    tail = o[k:]
    sets["rnd0"] = rnd_plain(tail, k, SEED["rnd0"])
    sets["rndT_d2"] = rnd_t(flat, np.array(sets["top"][tag]), tail, SEED["rndT_d2"] + j)
    sets["topF"] = np.sort(of[:k]).tolist()
    sets["J_floor"] = {t: jaccard(np.sort(o[:kk]), np.sort(of[:kk])) for t, kk in zip(K_TAGS, ks)}
    return sets


# ------------------------------------------------------------------------------------------------ positive controls
def pc_rank(before: Weights, after: Weights, cfg: dict, s2: np.ndarray, k_ref: int, n_plant: int, seed: int = SEED["plant"],
            along: str = "neurons") -> dict:
    """§5 PC-RANK: draw n_plant neurons below the median s, multiply their gate row, up row and down column by
    f = 1.5 s_(k_ref) / s_n in an independent copy of the change, rank the copy; recovery = share of planted in the top k_ref + n_plant.
    along="down_rows" is the failing input: row h = n mod hidden of down_proj only."""
    L, W, H = cfg["num_hidden_layers"], cfg["intermediate_size"], cfg["hidden_size"]
    flat = np.sqrt(s2.ravel())
    o = order(s2)
    s_ref = flat[o[k_ref - 1]]
    below = np.flatnonzero(flat < np.median(flat))
    planted = np.sort(np.random.default_rng(seed).choice(below, size=n_plant, replace=False))
    assert (flat[planted] > 0).all(), "a planted neuron has no change; f would be infinite"
    f = 1.5 * s_ref / flat[planted]
    s2c = np.zeros((L, W))
    for l in range(L):
        dg, du, dd = (delta(b, a).copy() for b, a in zip(layer_triplet(before, l), layer_triplet(after, l)))
        for p, fp in zip(planted, f):
            if p // W != l:
                continue
            n = int(p % W)
            if along == "neurons":
                dg[n, :] *= fp
                du[n, :] *= fp
                dd[:, n] *= fp
            elif along == "down_rows":
                dd[n % H, :] *= fp
            else:
                raise ValueError(along)
        s2c[l] = neuron_sq(dg, du, dd)
    top = order(s2c)[: k_ref + n_plant]
    rec = len(set(planted.tolist()) & set(top.tolist())) / n_plant
    return {"recovery": rec, "n_plant": n_plant, "k_ref": k_ref, "top_n": k_ref + n_plant, "along": along, "seed": seed,
            "s_ref": float(s_ref), "chance": (k_ref + n_plant) / flat.size}


def synthetic_rounding(before: Weights, cfg: dict, p: float = 0.3, seed: int = SEED["plant"]) -> dict:
    """§5 PC-ART: every MLP entry of the base moves one bf16 step, random sign, with probability p. Returns {name: uint16}."""
    out = {}
    for l in range(cfg["num_hidden_layers"]):
        for w, b in enumerate(layer_triplet(before, l)):
            rng = np.random.default_rng([seed, 1, l, w])
            move = rng.random(b.shape, dtype=np.float32) < np.float32(p)
            sign = np.where(rng.random(b.shape, dtype=np.float32) < np.float32(0.5), -1, 1)
            out[f"model.layers.{l}.{MLP[w]}.weight"] = from_ordered(ordered(b) + move * sign)
    return out


def plant_multistep(before: Weights, cfg: dict, replaced: dict, n_plant: int, steps: int = 8, seed_n: int = 41, seed_s: int = 42) -> list:
    """PC-ART P- (N13_SPEC.md A2): in `replaced` (P+'s MLP tensors, modified in place), n_plant neurons drawn uniformly over all
    (default_rng(seed_n)) have every entry of their gate row, up row and down column moved `steps` bf16 steps from the base value, with a
    random sign per entry (one default_rng(seed_s) stream, consumed in ascending neuron order, gate, up, down). Returns the neurons."""
    W, H = cfg["intermediate_size"], cfg["hidden_size"]
    planted = np.sort(np.random.default_rng(seed_n).choice(neuron_count(cfg), n_plant, replace=False))
    rng = np.random.default_rng(seed_s)
    for f in planted:
        l, n = divmod(int(f), W)
        for w, b in enumerate(layer_triplet(before, l)):
            t = replaced[f"model.layers.{l}.{MLP[w]}.weight"]
            sg = np.where(rng.random(H) < 0.5, -1, 1)
            if w < 2:
                t[n, :] = from_ordered(ordered(b[n, :]) + steps * sg)
            else:
                t[:, n] = from_ordered(ordered(b[:, n]) + steps * sg)
    return planted.tolist()


def spacing_sq(before: Weights, cfg: dict) -> np.ndarray:
    """sp_n^2 per neuron [L, W]: the sum of sp(w_before)^2 over the neuron's gate row, up row and down column (J_sp, A2)."""
    return np.stack([neuron_sq(*(spacing(x) for x in layer_triplet(before, l))) for l in range(cfg["num_hidden_layers"])])


def artifact_rule(j_floor: float, j_round: float) -> str:
    """The H_artifact rule of N13_SPEC.md §6.3 as amended (A2), at one k."""
    if j_floor >= 0.7 and j_round >= 0.7:
        return "REFUTED"
    if j_floor < 0.4 or j_round < 0.4:
        return "SUPPORTED"
    return "INDETERMINATE"


class Overlay:
    """A Weights view whose MLP tensors are replaced by the given arrays (PC-ART's synthetic 'after')."""

    def __init__(self, base: Weights, replaced: dict):
        self.base, self.replaced = base, replaced

    def names(self):
        return self.base.names()

    def get(self, name):
        return self.replaced[name] if name in self.replaced else self.base.get(name)


def j_round(before, after, cfg, s2, ks, n_redraw: int = N_REDRAW, seed0: int = SEED["redraw"]) -> dict:
    """J_round(k): mean over redraws of Jaccard(S_k by s, S_k by s*), s* from U* = delta + uniform(-sp/2, sp/2) of the before value."""
    L = cfg["num_hidden_layers"]
    o = order(s2)
    J = {t: [] for t in K_TAGS[: len(ks)]}
    for r in range(n_redraw):
        sd = seed0 + r
        s2r = np.zeros_like(s2)
        for l in range(L):
            U = []
            for w, (b, a) in enumerate(zip(layer_triplet(before, l), layer_triplet(after, l))):
                sp = spacing(b)
                u = np.random.default_rng([sd, l, w]).random(b.shape, dtype=np.float32) - np.float32(0.5)
                U.append(delta(b, a) + u * sp)
            s2r[l] = neuron_sq(*U)
        orr = order(s2r)
        for t, k in zip(J, ks):
            J[t].append(jaccard(o[:k], orr[:k]))
    return {"J_round": {t: float(np.mean(v)) for t, v in J.items()}, "per_redraw": J, "seeds": [seed0, seed0 + n_redraw - 1]}


# ------------------------------------------------------------------------------------------------ the builder
def arm_spec(name: str, sets: dict) -> dict:
    """§3.2 roster entry -> mask kind, neurons, scale, non-MLP side. Names: base rl zero id top_<k> comp_<k> rndE_<k> rndT_<k> dare_<k>
    mlp attn rnd0_k1 top_k1_B rndT_k1_d2 dare_k1_d2 topF_k1."""
    tag = name.split("_")[1] if "_" in name else None
    if name in ("base", "zero"):
        return {"name": name, "kind": "none", "non_mlp": "before"}
    if name in ("rl", "id"):
        return {"name": name, "kind": "all", "non_mlp": "after"}
    if name == "mlp":
        return {"name": name, "kind": "all", "non_mlp": "before"}
    if name == "attn":
        return {"name": name, "kind": "none", "non_mlp": "attn_norm_after"}
    if name in ("rnd0_k1",):
        return {"name": name, "kind": "neurons", "neurons": sets["rnd0"]["neurons"], "c": 1.0, "non_mlp": "before"}
    if name == "topF_k1":
        return {"name": name, "kind": "neurons", "neurons": sets["topF"], "c": 1.0, "non_mlp": "before"}
    if name == "rndT_k1_d2":
        r = sets["rndT_d2"]
        return {"name": name, "kind": "neurons", "neurons": r["neurons"], "c": 1.0, "non_mlp": "before",
                "intended_mlp_energy": r["E_S"], "energy_short": bool(r["short"])}              # A30
    if name.startswith("top_"):
        return {"name": name, "kind": "neurons", "neurons": sets["top"][tag], "c": 1.0, "non_mlp": "before"}
    if name.startswith("comp_"):
        return {"name": name, "kind": "complement", "neurons": sets["top"][tag], "c": 1.0, "non_mlp": "after"}
    if name.startswith("rndE_"):
        r = sets["rndE"][tag]
        return {"name": name, "kind": "neurons", "neurons": r["neurons"], "c": r["c"], "non_mlp": "before",
                "intended_mlp_energy": r["c"] ** 2 * r["E_R"], "rounding": "stochastic",
                "round_seed": SEED["rndE_round"] + K_TAGS.index(tag)}                          # A30
    if name.startswith("rndT_"):
        r = sets["rndT"][tag]
        return {"name": name, "kind": "neurons", "neurons": r["neurons"], "c": 1.0, "non_mlp": "before",
                "intended_mlp_energy": r["E_S"], "energy_short": bool(r["short"])}              # A30: the energy ratio, reported
    if name.startswith("dare_"):
        j = K_TAGS.index(tag)
        seed = SEED["dare_d2"] + j if name.endswith("_d2") else SEED["dare"] + j
        q = sets["dare_q"][tag]
        return {"name": name, "kind": "dare", "q": q, "seed": seed, "c": 1.0 / q, "non_mlp": "before"}
    raise ValueError(f"unknown arm {name}")


def _neuron_mask(kind, neurons_in_layer, shape, w, down_axis):
    W = shape[0] if w < 2 else shape[1]
    sel = np.zeros(W, dtype=bool)
    sel[neurons_in_layer] = True
    if kind == "complement":
        sel = ~sel
    if w < 2:
        return np.broadcast_to(sel[:, None], shape)
    if down_axis == "columns":
        return np.broadcast_to(sel[None, :], shape)
    rows = np.zeros(shape[0], dtype=bool)            # the G-MASK failing input: rows of down_proj (h = n mod hidden)
    rows[np.flatnonzero(sel) % shape[0]] = True
    return np.broadcast_to(rows[:, None], shape)


def build_tensor(name, b16, a16, spec, W, down_axis="columns") -> tuple:
    """Returns (theta_T as uint16, mask M or None for all-zero)."""
    fam = family(name)
    if fam != "mlp":
        side = spec["non_mlp"]
        take = side == "after" or (side == "attn_norm_after" and fam in ("attn", "norm"))
        return (np.array(a16) if take else np.array(b16)), (np.ones(b16.shape, bool) if take else None)
    l, w = layer_of(name), which_mlp(name)
    kind = spec["kind"]
    if kind == "none":
        return np.array(b16), None
    if kind == "all":
        M = np.ones(b16.shape, dtype=bool)
    elif kind in ("neurons", "complement"):
        nl = [int(p % W) for p in spec["neurons"] if p // W == l]
        M = _neuron_mask(kind, nl, b16.shape, w, down_axis)
    elif kind == "dare":
        M = dare_mask(spec["seed"], l, w, b16.shape, spec["q"])
    else:
        raise ValueError(kind)
    c = float(spec.get("c", 1.0))
    if c == 1.0:
        T = np.where(M, a16, b16).astype(np.uint16)
        D = delta(b16, a16)
        # A29 (N13_SPEC.md): the check holds where §3.1 says it holds, on entries whose float32 difference is exact (§0.5), as
        # values (+0 equals -0, A6). T copies the after bits everywhere M = 1, so the arm does not depend on this check.
        nz = M & (D != 0) & ~inexact(b16, a16)
        assert np.array_equal(f32(bf16(f32(b16) + D))[nz], f32(a16)[nz]), f"{name}: the formula does not return theta_a"
        return T, M
    D = delta(b16, a16)
    nz = M & (D != 0)
    x = f32(b16) + np.float32(c) * D
    if spec.get("rounding") == "stochastic":         # A30: one stream per tensor, independent of the order tensors are visited
        r = bf16_stochastic(x, np.random.default_rng([spec["round_seed"], l, w]))
    else:
        r = bf16(x)
    T = np.where(nz, r, b16).astype(np.uint16)
    return T, M


def energy_match(build_once, spec: dict, energy_tol=0.05, passes=ENERGY_PASSES) -> tuple:
    """A30: build_once(c) -> (audit, realized MLP energy). Builds at the spec's c; if the realized energy is outside energy_tol of the
    intended, c is multiplied by sqrt(intended / realized) and the arm is built again, at most `passes` builds; else AssertionError."""
    ie, c, rec = spec["intended_mlp_energy"], float(spec["c"]), []
    for _ in range(passes):
        audit, re_ = build_once(c)
        rec.append({"c": c, "realized": re_, "ratio": re_ / ie})
        if abs(re_ / ie - 1) <= energy_tol:
            return audit, c, rec
        c *= float(np.sqrt(ie / re_))
    raise AssertionError(f"{spec['name']}: realized MLP energy not within {energy_tol} of the intended after {passes} builds: {rec}")


def a31_compare(written: dict, matched: dict, tol: float = A31_REL_TOL) -> dict:
    """A31 (N13_SPEC.md): the written arm against the in-memory build that met the energy band (A30). Every tensor must be equal bit for
    bit (per-tensor sha256). The MLP energy is summed over the tensors in sorted-name order in memory and shard by shard when written, so
    the two float64 sums can differ in their last bits; they must agree within tol, relative. Both sums are recorded."""
    w, m = written["families"]["mlp"]["realized_sq"], matched["families"]["mlp"]["realized_sq"]
    names = sorted(set(written["sha256"]) | set(matched["sha256"]))
    differ = [n for n in names if written["sha256"].get(n) != matched["sha256"].get(n)]
    rel = abs(w - m) / m
    return {"written_sq": w, "matched_sq": m, "rel_diff": rel, "tol": tol, "energy_pass": bool(rel <= tol), "n_tensors": len(names),
            "n_tensors_differ": len(differ), "tensors_differ": differ[:10], "pass": bool(rel <= tol and not differ)}


def build_arm(before: Weights, after: Weights, cfg: dict, spec: dict, out_dir=None, down_axis="columns", kept_sd=4.0,
              energy_tol=0.05) -> dict:
    """§3.1 and §3.4. A30: an arm with stochastic rounding (rndE) is brought within energy_tol of its intended MLP energy by
    energy_match (in memory), then written at the c found; every other arm is built once. An arm with an intended energy (rndE, rndT)
    must end within energy_tol of it; an rndT arm (c = 1, the shortest tail prefix with E >= E(S_k), §3.2) has its ratio recorded and
    flagged, not asserted: its overshoot is bounded by one neuron's energy, not by the 5 % of rescaled arms.
    A31: the written rndE arm is compared with the in-memory build that met the band (a31_compare)."""
    if spec.get("rounding") == "stochastic":
        def once(c):
            a = _build_arm(before, after, cfg, dict(spec, c=c), None, down_axis, kept_sd, energy_tol)
            return a, a["families"]["mlp"]["realized_sq"]
        audit, c, rec = energy_match(once, spec, energy_tol)
        if out_dir is not None:
            matched = audit
            audit = _build_arm(before, after, cfg, dict(spec, c=c), out_dir, down_axis, kept_sd, energy_tol)
            audit["a31"] = a31_compare(audit, matched)
            assert audit["a31"]["pass"], ("A31: the written arm differs from the matched build", audit["a31"])
        audit["energy_passes"], audit["c_built"] = rec, c
    else:
        audit = _build_arm(before, after, cfg, spec, out_dir, down_axis, kept_sd, energy_tol)
    if spec.get("rounding") == "stochastic":
        assert not audit["energy_flag"], (f"A30: {spec['name']} realized MLP energy off by {audit['energy_rel_err']:.4f}", energy_tol)
    return audit


def _build_arm(before: Weights, after: Weights, cfg: dict, spec: dict, out_dir=None, down_axis="columns", kept_sd=4.0,
               energy_tol=0.05) -> dict:
    """Build one arm tensor by tensor; audit it (§3.4); write it when out_dir is given (shards as the after snapshot's).
    DARE kept count (N13_SPEC.md A4): |K - P_k| <= kept_sd * sqrt(P_k (1 - q_k)), four binomial standard deviations."""
    W = cfg["intermediate_size"]
    names = after.names()
    shapes = {n: tuple(after.get(n).shape) for n in names}
    audit = {"arm": spec["name"], "spec": {k: v for k, v in spec.items() if k != "neurons"}, "n_neurons_in_set": len(spec.get("neurons", [])),
             "sha256": {}, "families": {f: {"realized_sq": 0.0, "changed": 0, "in_mask": 0} for f in FAMILIES},
             "mask": {"gate_rows": {}, "up_rows": {}, "down_cols": {}, "expected": {}}, "kept": 0}
    want = set(spec.get("neurons", [])) if spec["kind"] in ("neurons", "complement") else None

    def produce(n):
        b, a = before.get(n), after.get(n)
        T, M = build_tensor(n, b, a, spec, W, down_axis)
        fam = family(n)
        r = audit["families"][fam]
        dT = f32(T).astype(np.float64) - f32(b)
        r["realized_sq"] += float(np.square(dT).sum())
        ch = (T != b) & ~(((T & 0x7FFF) == 0) & ((np.asarray(b) & 0x7FFF) == 0))
        r["changed"] += int(ch.sum())
        r["in_mask"] += int(M.sum()) if M is not None else 0
        if fam == "mlp":
            l, w = layer_of(n), which_mlp(n)
            if spec["kind"] == "dare":
                audit["kept"] += int(M.sum())
            if want is not None:
                axis = 1 if w < 2 else 0
                idx = np.flatnonzero(ch.any(axis=axis))
                key = ("gate_rows", "up_rows", "down_cols")[w]
                audit["mask"][key][str(l)] = (l * W + idx).tolist()
                nzc = np.flatnonzero((delta(b, a) != 0).any(axis=axis))
                inset = {p for p in (l * W + nzc).tolist()}
                exp = sorted(inset & want) if spec["kind"] == "neurons" else sorted(inset - want)
                audit["mask"]["expected"].setdefault(key, {})[str(l)] = exp
        audit["sha256"][n] = tensor_sha256(T)
        return T

    if out_dir is None:
        for n in names:
            produce(n)
    else:
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        by_shard = {}
        for n in names:
            by_shard.setdefault(after.weight_map[n], []).append(n)
        for shard, ns in sorted(by_shard.items()):
            write_safetensors(out / shard, ns, shapes, produce)
        (out / "model.safetensors.index.json").write_text(json.dumps(after.index, indent=1))
        for f in COPY_FILES:
            if (after.dir / f).exists():
                shutil.copyfile(after.dir / f, out / f)
    m = audit["mask"]
    if want is not None:
        m["pass"] = all(m[k] == m["expected"][k] for k in ("gate_rows", "up_rows", "down_cols"))
    if spec["kind"] == "dare":
        P = spec["q"] * sum(int(np.prod(shapes[n])) for n in names if family(n) == "mlp")
        audit["kept_expected"] = P
        sd = float(np.sqrt(P * (1 - spec["q"])))
        audit["kept_rel_err"] = abs(audit["kept"] - P) / P
        audit["kept_sd_units"] = (audit["kept"] - P) / sd
        audit["kept_tol_entries"] = kept_sd * sd
        audit["kept_pass"] = abs(audit["kept"] - P) <= kept_sd * sd
    if "intended_mlp_energy" in spec or spec["kind"] == "dare":
        if spec["kind"] == "dare":
            spec_e = 0.0
            for n in names:
                if family(n) == "mlp":
                    b, a = before.get(n), after.get(n)
                    M = dare_mask(spec["seed"], layer_of(n), which_mlp(n), b.shape, spec["q"])
                    spec_e += float(np.square(delta(b, a)[M], dtype=np.float64).sum()) * spec["c"] ** 2
            audit["intended_mlp_energy"] = spec_e
        else:
            audit["intended_mlp_energy"] = spec["intended_mlp_energy"]
        ie, re_ = audit["intended_mlp_energy"], audit["families"]["mlp"]["realized_sq"]
        audit["energy_rel_err"] = abs(re_ - ie) / ie if ie else None
        audit["energy_flag"] = bool(ie and audit["energy_rel_err"] > energy_tol)
    return audit


def snapshot_hashes(w: Weights) -> dict:
    return {n: tensor_sha256(w.get(n)) for n in w.names()}


def g_hash(built: dict, reference: dict) -> dict:
    bad = sorted(n for n in reference if built.get(n) != reference[n])
    return {"pass": not bad and set(built) == set(reference), "mismatched": bad[:10], "n_mismatched": len(bad)}


# ------------------------------------------------------------------------------------------------ G-INIT
def g_init_hash(downloaded: dict, logged_init_oids: dict) -> dict:
    """(a) every downloaded base shard's sha256 equals the LFS oid at the logged init revision."""
    keys = sorted(logged_init_oids)
    ok = {k: downloaded.get(k) == logged_init_oids[k] for k in keys}
    return {"pass": bool(keys) and all(ok.values()) and sorted(downloaded) == keys, "per_shard": ok}


def g_init_share(share_main: float, share_other: float, margin: float = 0.10) -> dict:
    """(b) unchanged MLP share (base main vs RL) exceeds (stage2 vs RL) by at least margin."""
    return {"pass": bool(share_main - share_other >= margin), "share_main": share_main, "share_other": share_other,
            "difference": share_main - share_other, "margin": margin}


# ------------------------------------------------------------------------------------------------ selfcheck
A29_PLANTED = [(0.0123, 0.0125, "exact"), (0.0123, 1.3e-8, "inexact, big to tiny"), (1.3e-8, 0.0123, "inexact, tiny to big"),
               (0.0123, -0.0, "to -0"), (0.5, 2.9e-6, "inexact, gap 18"), (0.0, 0.25, "from +0"), (-0.75, -0.7, "exact, negative")]


def a29_planted() -> dict:
    """A29: a 1 x 7 down_proj tensor of planted (before, after) pairs. The c = 1 arm built from it must equal the after bits; the
    pre-A29 assertion (bits on every entry with Delta != 0) must fail on it, as it failed on the real weights at s3gate-01."""
    b16 = bf16(np.array([[x[0] for x in A29_PLANTED]], np.float32))
    a16 = bf16(np.array([[x[1] for x in A29_PLANTED]], np.float32))
    T, M = build_tensor("model.layers.0.mlp.down_proj.weight", b16, a16, {"kind": "all", "c": 1.0, "non_mlp": "after"}, 7)
    D = delta(b16, a16)
    old_nz = M & (D != 0)
    formula = bf16(f32(b16) + D)
    return {"cases": [{"case": x[2], "before": hex(int(b)), "after": hex(int(a)), "formula": hex(int(f)), "built": hex(int(t)),
                       "inexact": bool(i)} for x, b, a, f, t, i in zip(A29_PLANTED, b16[0], a16[0], formula[0], T[0], inexact(b16, a16)[0])],
            "built_equals_after": bool(np.array_equal(T, a16)),
            "old_assertion_holds": bool(np.array_equal(formula[old_nz], np.asarray(a16)[old_nz])),
            "formula_off_entries": int((formula != a16)[old_nz].sum()), "inexact_entries": inexact_count(b16, a16)}


def a30_tensor(rounding, n=64, w=96, seed=7):
    """A30's planted arm: one down_proj tensor whose every change is one bf16 step up (b in [0.01, 0.02)), all its columns in the set,
    c = 1.3, intended energy 1.69 E_R. Returns (build_once(c) for energy_match, spec, E_R)."""
    b16 = bf16(np.random.default_rng(seed).uniform(0.01, 0.02, (n, w)).astype(np.float32))
    a16 = (b16 + 1).astype(np.uint16)
    E_R = float(np.square(delta(b16, a16), dtype=np.float64).sum())
    spec = {"name": f"a30_{rounding}", "kind": "neurons", "neurons": list(range(w)), "c": 1.3, "non_mlp": "before",
            "intended_mlp_energy": 1.69 * E_R, **({"rounding": "stochastic", "round_seed": SEED["rndE_round"]} if rounding == "stochastic" else {})}

    def once(c):
        T, _ = build_tensor("model.layers.0.mlp.down_proj.weight", b16, a16, dict(spec, c=c), w)
        return None, float(np.square(f32(T).astype(np.float64) - f32(b16)).sum())
    return once, spec, E_R


def a30_planted() -> dict:
    """A30: deterministic rounding at c = 1.3 realizes E_R, 1 / 1.69 of the intended (outside 5 %); stochastic rounding with energy_match
    ends within 5 %."""
    det_once, det_spec, E_R = a30_tensor("deterministic")
    det = det_once(1.3)[1] / det_spec["intended_mlp_energy"]
    st_once, st_spec, _ = a30_tensor("stochastic")
    _, c, rec = energy_match(st_once, st_spec)
    return {"E_R": E_R, "intended": st_spec["intended_mlp_energy"], "deterministic_ratio_at_1.3": det,
            "stochastic_passes": rec, "stochastic_c_final": c, "stochastic_final_ratio": rec[-1]["ratio"]}


def selfcheck():
    q = a30_planted()
    assert abs(q["deterministic_ratio_at_1.3"] - 1 / 1.69) < 1e-9 and abs(q["stochastic_final_ratio"] - 1) <= 0.05, q
    r = a29_planted()
    assert r["built_equals_after"] and not r["old_assertion_holds"] and r["formula_off_entries"] == 3 and r["inexact_entries"] == 3, r
    x = np.array([0.0, -0.0, 1.0, -1.5, 3.14159, 1e-30, -2.0 ** -130, 65504.0, 1.0 + 2 ** -8, 1.0 + 3 * 2 ** -8], np.float32)
    u = bf16(x)
    assert np.array_equal(bf16(f32(u)), u), "bf16(f32(u)) != u"
    assert u[8] == 0x3F80 and u[9] == 0x3F82, (hex(u[8]), hex(u[9]))          # ties to even: 1+2^-8 -> 1.0; 1+3*2^-8 -> 1+2^-6
    o = ordered(u)
    assert o[0] == o[1] == 0 and np.array_equal(from_ordered(ordered(u))[2:], u[2:]), o
    one = np.array([0x3F80], np.uint16)
    up = from_ordered(ordered(one) + 1)
    assert f32(up)[0] - 1.0 == spacing(one)[0] == 2.0 ** -7, (f32(up), spacing(one))
    assert bit_distance(one, up)[0] == 1 and bit_distance(np.array([0x8000], np.uint16), np.array([0], np.uint16))[0] == 0
    big, tiny = bf16(np.array([2.0 ** 20], np.float32)), bf16(np.array([1.0], np.float32))
    assert inexact_count(big, tiny) == 1 and inexact_count(tiny, tiny) == 0
    s = np.array([[3.0, 1.0, 3.0], [0.5, 9.0, 1.0]])
    assert order(s).tolist() == [4, 0, 2, 1, 5, 3], order(s)
    dg = np.zeros((3, 2), np.float32)
    dg[1, :] = 1
    dd = np.zeros((2, 3), np.float32)
    dd[:, 2] = 2
    assert neuron_sq(dg, np.zeros((3, 2), np.float32), dd).tolist() == [0.0, 2.0, 8.0]
    return True
