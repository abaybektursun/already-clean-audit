"""m01: the model and the snapshots (OLMo-3 7B layer structure, tensor shapes, the sixteen released snapshots, and
the three points outside the sixteen that later chapters use).

Reads only banked files in this repository (no network, no GPU) and writes data.js next to this file as
window.<ID>_DATA = {...}. Prints `path sha256` for every input.

    python3 extract.py
"""
import ast
import datetime
import hashlib
import json
import math
import importlib.util
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]                       # walkthrough/<picture> -> already-clean -> studies -> repository root
ID = HERE.name.upper()
MAX_BYTES = 300_000
SOURCES = []


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
    data["meta"] = {"commit": commit, "link": "studies/next/00_ASSETS.md",  # the file the data link opens
                    "sources": SOURCES,
                    "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                    "notes": notes}
    out = HERE / "data.js"
    out.write_text(f"window.{ID}_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    size = out.stat().st_size
    assert size <= MAX_BYTES, f"{out} is {size:,} bytes"
    print(f"wrote {out.relative_to(ROOT)}: {size:,} bytes")


def literal(rel, name):
    """The value of a module-level assignment NAME in a python file, read with ast (the module is not run)."""
    tree = ast.parse(src(rel).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == name:
            return eval(compile(ast.Expression(node.value), rel, "eval"), {"json": json})
    raise KeyError(f"{name} not in {rel}")


def line_with(rel, needle):
    """The one line of a text file that contains NEEDLE; fails unless exactly one line does."""
    hits = [s for s in src(rel).read_text().splitlines() if needle in s]
    assert len(hits) == 1, f"{needle!r} in {rel}: {len(hits)} lines"
    return hits[0]


def main():
    cfg = literal("infra/pg/m1.py", "REF_CONFIG")
    same = literal("infra/pg/m1.py", "SAME")
    spec = importlib.util.spec_from_file_location("n13_arms", src("infra/pg/n13_arms.py"))
    arms = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arms)
    shapes = arms.expected_shapes(cfg)

    # one layer's tensors, in the order of the computation
    L = cfg["num_hidden_layers"]
    layer0 = {k[len("model.layers.0."):]: list(v) for k, v in shapes.items() if k.startswith("model.layers.0.")}
    assert "input_layernorm.weight" not in layer0, "an input norm would contradict the post-norm structure"
    order = ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
             "self_attn.q_norm.weight", "self_attn.k_norm.weight", "self_attn.o_proj.weight",
             "post_attention_layernorm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
             "post_feedforward_layernorm.weight"]
    assert sorted(order) == sorted(layer0), (sorted(order), sorted(layer0))
    n_params = sum(math.prod(v) for v in shapes.values())
    spec_line = line_with("studies/next/07_PICKS/M1_SPEC.md", "| mix index |")
    banked_params = int(re.search(r"`total_parameters` ([\d,]+)", spec_line)[1].replace(",", ""))
    params = compare("parameters", n_params, banked_params, "studies/next/07_PICKS/M1_SPEC.md line '| mix index |'", 0)
    print(f"PLAN 7,298,011,136 -> parameters.recomputed = {n_params:,}")

    # the sixteen snapshots (00_ASSETS.md table: branch and tokens seen; tokens = step x 4,194,304)
    TPS = 4_194_304
    rows = [s for s in src("studies/next/00_ASSETS.md").read_text().splitlines() if re.match(r"\| T_\d+ \| `stage1-step", s)]
    assert len(rows) == 16, len(rows)
    snapshots = []
    for s in rows:
        c = [x.strip() for x in s.strip("|").split("|")]
        step = int(re.fullmatch(r"`stage1-step(\d+)`", c[1])[1])
        tokens = int(c[3].replace(",", ""))
        assert tokens == step * TPS, (c[0], tokens, step * TPS)
        snapshots.append({"snapshot": int(c[0][2:]), "branch": c[1].strip("`"), "step": step, "tokens": tokens})
    assert [x["snapshot"] for x in snapshots] == list(range(16))
    # the same sixteen steps in SPEC_D1.md §4.3
    d1 = [s for s in src("studies/already-clean/SPEC_D1.md").read_text().splitlines() if re.match(r"\| T_\d+ \| [\d,]+ \|", s)]
    d1_steps = [int(s.split("|")[2].strip().replace(",", "")) for s in d1]
    assert d1_steps == [x["step"] for x in snapshots], d1_steps
    print(f"PLAN step 1,413,814 -> snapshots[15].step = {snapshots[15]['step']:,}; tokens {snapshots[15]['tokens']:,}")

    # the three points outside the sixteen
    start = line_with("studies/next/07_PICKS/M1_SPEC.md", "| shared start |")
    m = re.search(r"`stage1-step(\d+)`, commit `([0-9a-f]+)`.*= ([\d,]+) tokens", start)
    start_step, start_commit, start_tokens = int(m[1]), m[2], int(m[3].replace(",", ""))
    assert start_tokens == start_step * TPS
    before = max((x for x in snapshots if x["step"] < start_step), key=lambda x: x["step"])
    after = min((x for x in snapshots if x["step"] > start_step), key=lambda x: x["step"])
    print(f"PLAN step 476,838 (2.0 trillion tokens, between snapshots 12 and 13) -> off_grid[0]: {start_step:,}, "
          f"{start_tokens:,}, between {before['snapshot']} and {after['snapshot']}")
    mixes = []
    for tag, share in (("G (0 % code+math)", 0), ("R (40 %)", 40), ("M (65 %)", 65)):
        s = line_with("studies/next/07_PICKS/M1_SPEC.md", f"| {tag} |")
        b = re.search(r"`(stage2-step\d+-mix-[A-Za-z0-9-]+)`, commit `([0-9a-f]+)`", s)
        mixes.append({"code_and_math_percent": share, "branch": b[1], "commit": b[2]})
    steps_line = line_with("studies/next/07_PICKS/M1_ASSETS.md", "Every mix run is named step")
    ms = re.search(r"step ([\d,]+)\. That is 100B tokens at 2\^21 = ([\d,]+) tokens per step", steps_line)
    mix_steps, mix_tps = int(ms[1].replace(",", "")), int(ms[2].replace(",", ""))
    base = line_with("studies/next/07_PICKS/N13_SPEC.md", "| pre-RL base, pairs 1 and 1b |")
    base_commit = re.search(r"branch `main`, commit `([0-9a-f]+)`", base)[1]
    s3 = line_with("studies/next/07_PICKS/N13_SPEC.md", "| `main` equals the end of stage 3 |")
    s3_branch = re.search(r"`(stage3-step\d+)`", s3)[1]
    rl = line_with("studies/next/07_PICKS/N13_SPEC.md", "| RL model, pair 1 (registered) |")
    rl_m = re.search(r"`(allenai/Olmo-3\.1-7B-RL-Zero-Math)`, `main`, commit `([0-9a-f]+)`", rl)
    off_grid = [
        {"name": "the start of the three data mixes", "branch": f"stage1-step{start_step}", "commit": start_commit,
         "stage": 1, "step": start_step, "tokens": start_tokens,
         "between_snapshots": [before["snapshot"], after["snapshot"]],
         "mixes": mixes, "mix_steps": mix_steps, "mix_tokens_per_step": mix_tps, "mix_tokens": mix_steps * mix_tps},
        {"name": "the finished base model (after stages 2 and 3)", "repo": "allenai/Olmo-3-1025-7B", "branch": "main",
         "commit": base_commit, "equals_branch": s3_branch},
        {"name": "the model trained further on mathematics", "repo": rl_m[1], "branch": "main", "commit": rl_m[2]},
    ]

    data = {
        "model": "allenai/Olmo-3-1025-7B",
        "layers": L, "hidden_size": cfg["hidden_size"], "mlp_units_per_layer": cfg["intermediate_size"],
        "mlp_units_total": L * cfg["intermediate_size"], "sides_per_layer": 2 * cfg["intermediate_size"],
        "sides_total": 2 * L * cfg["intermediate_size"],
        "attention_heads": cfg["num_attention_heads"], "key_value_heads": cfg["num_key_value_heads"],
        "head_dim": cfg["hidden_size"] // cfg["num_attention_heads"],
        "layer_types": cfg["layer_types"], "sliding_window": cfg["sliding_window"],
        "vocab_size": cfg["vocab_size"], "tied_embeddings": cfg["tie_word_embeddings"],
        "rms_norm_eps": cfg["rms_norm_eps"], "activation": cfg["hidden_act"],
        "input_tokens_per_window": same["window"] + 1, "corpus_windows": same["n_windows"],
        "scored_tokens": same["n_tokens_scored"],
        "layer_tensors_in_order": [{"name": k, "shape": layer0[k]} for k in order],
        "outside_layers": [{"name": k, "shape": list(v)} for k, v in shapes.items() if not k.startswith("model.layers.")],
        "parameters": params,
        "tokens_per_step": TPS,
        "snapshots": snapshots,
        "off_grid": off_grid,
    }
    notes = {
        "layer_tensors_in_order": "The tensors of one layer (the same in all 32), from infra/pg/n13_arms.py expected_shapes on the "
            "registered config; there is no input norm: each layer computes h <- h + post_attention_layernorm(Attention(h)), "
            "then h <- h + post_feedforward_layernorm(MLP(h)), so the norm is on a sublayer's output, before the add.",
        "outside_layers": "Embedding matrix, final norm and output matrix (untied).",
        "layer_types": "Per layer, sliding-window attention (window sliding_window) or full attention; three sliding then one full, "
            "repeated 8 times. Inputs are input_tokens_per_window = 2,049 tokens, shorter than the window, so every layer sees the whole input.",
        "sides_per_layer": "Each MLP unit has a positive side (tokens with activation > 0) and a negative side (activation < 0).",
        "input_tokens_per_window": "A corpus window: the start-of-text token plus 2,048 tokens (infra/pg/m1.py SAME window).",
        "parameters": "recomputed = the sum of the products of every tensor shape; banked = total_parameters in the mix index "
            "as recorded in M1_SPEC.md §0.2.",
        "snapshots": "The sixteen released stage-1 snapshots (internal names T_0 to T_15), branch and tokens seen from "
            "studies/next/00_ASSETS.md; tokens = step x 4,194,304 is asserted for each, and the steps equal SPEC_D1.md §4.3.",
        "off_grid": "The three model points outside the sixteen: the stage-1 point the three data mixes start from (each mix "
            "continues 47,684 steps of 2,097,152 tokens = 100 billion tokens; internal mix codes G, R, M = 0, 40, 65 % code and "
            "math), the finished base model used as the starting point of the mathematics study, and the model trained "
            "further on mathematics (internal: RL model).",
    }
    write(data, notes)


if __name__ == "__main__":
    main()
