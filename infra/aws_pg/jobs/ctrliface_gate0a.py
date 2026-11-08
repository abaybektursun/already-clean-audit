"""CTRLIFACE Gate 0a — INTROSPECTION (§9.0 preflight, part 1: resolve the wiring).

Purpose: learn the lerobot π0.5 API + confirm the backbone hook target on the REAL checkpoint,
WITHOUT guessing an API. Prints (does not assert, except the compartment gate which pi0_port
already defines):
  * lerobot version + every policy class it exposes (pkgutil walk) → the class to load
  * the checkpoint's config.json (`type`, architecture, action/obs specs)
  * the checkpoint safetensors KEY set (via safe_open headers — no 7GB tensor load) →
    the runtime module naming for language_model.layers[L] / gemma_expert / vision_tower
  * the pi0_port compartment gate on the real keys: remap_backbone_lm + assert_18x7 (18×7 backbone
    spine, NO gemma_expert leak) — the one HARD assertion (fail-loud).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.ctrliface_gate0a --type r7i.2xlarge --no-weights \
      --s3-prefix results/ctrliface_gate0a --budget-min 40

CPU-only (key/shape headers, no forward) → cheap. GPU rollout + hook + grad live in Gate 0b.
"""
import argparse
import importlib
import inspect
import json
import os
import pathlib
import pkgutil
import subprocess
import sys

POLICY_REPO = "lerobot/pi05_libero_finetuned"


def _install():
    subprocess.run(["bash", "-lc",
                    "apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y "
                    "git python3-dev build-essential"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "lerobot[libero,pi] @ git+https://github.com/huggingface/lerobot.git"], check=True)


def _enumerate_policy_classes():
    """Walk lerobot.* and collect every class whose name ends in 'Policy' + its module."""
    import lerobot
    found = {}
    for mod in pkgutil.walk_packages(lerobot.__path__, prefix="lerobot."):
        name = mod.name
        if "polic" not in name.lower():
            continue
        m = importlib.import_module(name)          # fail-loud: a broken submodule traces
        for cn, cobj in inspect.getmembers(m, inspect.isclass):
            if cn.endswith("Policy") and cobj.__module__ == name:
                found[f"{name}.{cn}"] = [x for x in ("from_pretrained", "forward", "select_action",
                                                     "compute_loss", "get_optim_params")
                                        if hasattr(cobj, x)]
    return found


def _safetensors_keys(repo):
    """Key set + shapes of the checkpoint safetensors, read from headers (no tensor load)."""
    from huggingface_hub import HfApi, hf_hub_download
    from safetensors import safe_open
    files = [f for f in HfApi().list_repo_files(repo) if f.endswith(".safetensors")]
    keyshapes = {}
    for f in files:
        p = hf_hub_download(repo, f)
        with safe_open(p, framework="pt") as sf:
            for k in sf.keys():
                keyshapes[k] = list(sf.get_slice(k).get_shape())
    return files, keyshapes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    _install()
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import lerobot
    from huggingface_hub import hf_hub_download

    rep = {"lerobot": getattr(lerobot, "__version__", "?"), "repo": POLICY_REPO}

    # 1. policy classes
    rep["policy_classes"] = _enumerate_policy_classes()

    # 2. checkpoint config
    cfg_path = hf_hub_download(POLICY_REPO, "config.json")
    rep["config"] = json.loads(pathlib.Path(cfg_path).read_text())

    # 3. checkpoint key set (headers only)
    files, keyshapes = _safetensors_keys(POLICY_REPO)
    rep["safetensors_files"] = files
    rep["n_keys"] = len(keyshapes)
    # prefix histogram (2-level) → the module tree from the weights
    from collections import Counter
    pref = Counter(".".join(k.split(".")[:4]) for k in keyshapes)
    rep["prefix_histogram_top40"] = pref.most_common(40)
    # sample keys per compartment (broad substrings — the real naming is what we're learning)
    def _sample(sub, n=12):
        return [k for k in keyshapes if sub in k][:n]
    rep["sample_language_model"] = _sample("language_model")
    rep["sample_gemma_expert"] = _sample("gemma_expert")
    rep["sample_vision_tower"] = _sample("vision_tower")
    rep["sample_layers0"] = [k for k in keyshapes if ".layers.0." in k][:20]
    rep["sample_layers1"] = [k for k in keyshapes if ".layers.1." in k][:6]
    # every distinct layer-index seen under each *.layers.N.* container (→ #decoder layers)
    import re
    layer_containers = {}
    for k in keyshapes:
        m = re.search(r"(.*)\.layers\.(\d+)\.", k)
        if m:
            layer_containers.setdefault(m.group(1), set()).add(int(m.group(2)))
    rep["layer_containers"] = {c: sorted(v)[-1] + 1 for c, v in layer_containers.items()}  # max idx+1

    # 4. Compartment structure — REPORT ONLY (this is a discovery job; the pi0_port gate assumes
    # the pi0_base naming and π0.5-LIBERO may differ — do NOT hard-assert an assumed schema here).
    from infra.pg import pi0_port
    class _Shim:
        def __init__(self, shape): self._shape = tuple(shape); self.ndim = len(shape)
    sd = {k: _Shim(v) for k, v in keyshapes.items()}
    backbone = pi0_port.remap_backbone_lm(sd)              # (still asserts no gemma_expert leak — real)
    matrices = pi0_port.decoder_matrices(backbone)
    rep["compartment_report"] = {
        "backbone_prefix_assumed": pi0_port.BACKBONE_LM_PREFIX,
        "backbone_keys_under_assumed_prefix": len(backbone),
        "decoder_matrices_matched": len(matrices),
        "decoder_matrix_keys_sample": sorted(matrices)[:20],
        "backbone_key_sample": sorted(backbone)[:20],
    }

    (out / "gate0a.json").write_text(json.dumps(rep, indent=2, default=str))     # WRITE FIRST
    print("GATE0A_OK", json.dumps({"lerobot": rep["lerobot"],
                                   "policy_classes": list(rep["policy_classes"].keys()),
                                   "n_keys": rep["n_keys"],
                                   "layer_containers": rep["layer_containers"],
                                   "compartment_report": rep["compartment_report"]}, default=str))


if __name__ == "__main__":
    main()
