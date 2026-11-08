"""CTRLIFACE Gate 0c v3 — DATA + PROCESSOR + ROLLOUT-API plumbing (§9.0 part 3).

Resolves the full recipe in one job (write-as-you-go, fail-loud). v3 fixes: meta.tasks/episodes are
pandas DataFrames (no `or []`); make_pre_post_processors takes the CONFIG (config.type); lerobot.scripts.eval
is gone in 0.5.2 → search lerobot.envs.utils + lerobot.scripts.lerobot_eval for the rollout loop +
observation preprocessing.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.ctrliface_gate0c --type g6e.8xlarge,g6e.4xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/ctrliface_gate0c --budget-min 60
"""
import argparse
import importlib
import importlib.util
import inspect
import json
import pathlib
import subprocess
import sys

POLICY_REPO = "lerobot/pi05_libero_finetuned"
DATASET_REPO = "HuggingFaceVLA/libero"
BACKBONE_PATH = "model.paligemma_with_expert.paligemma.model.language_model"
PROBE_L = 9


def _install():
    subprocess.run(["bash", "-lc",
                    "apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y "
                    "git python3-dev build-essential libgl1 libglib2.0-0 libegl1"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "lerobot[libero,pi] @ git+https://github.com/huggingface/lerobot.git"], check=True)


def _src(obj, limit=6000):
    return inspect.getsource(obj)[:limit]


def _get_submodule(root, dotted):
    m = root
    for p in dotted.split("."):
        m = getattr(m, p)
    return m


def _dump_fns(modname, substrs):
    """{fn_name: {sig, src}} for module-level functions whose name matches any substr."""
    if importlib.util.find_spec(modname) is None:
        return "MISSING"
    m = importlib.import_module(modname)
    out = {}
    for n, obj in inspect.getmembers(m, inspect.isfunction):
        if any(s in n.lower() for s in substrs):
            out[n] = {"sig": str(inspect.signature(obj)), "src": _src(obj)}
    return out or "(no matching fns)"


def _df_head(obj, n=3):
    """Serialize a pandas DataFrame / HF Dataset / Series / dict / list head for inspection."""
    if obj is None:
        return None
    if hasattr(obj, "columns") and hasattr(obj, "head"):          # pandas DataFrame
        return {"type": "DataFrame", "columns": list(obj.columns), "len": int(len(obj)),
                "head": obj.head(n).to_dict(orient="records")}
    if hasattr(obj, "column_names"):                              # HF datasets.Dataset
        return {"type": "HFDataset", "columns": list(obj.column_names), "len": int(len(obj)),
                "head": [obj[i] for i in range(min(n, len(obj)))]}
    if hasattr(obj, "to_dict") and hasattr(obj, "items"):         # pandas Series
        return {"type": "Series", "len": int(len(obj)), "head": dict(list(obj.items())[:n])}
    if isinstance(obj, dict):
        return {"type": "dict", "len": len(obj), "head": dict(list(obj.items())[:n])}
    if isinstance(obj, (list, tuple)):
        return {"type": "list", "len": len(obj), "head": list(obj[:n])}
    return {"type": str(type(obj)), "repr": str(obj)[:400]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default=DATASET_REPO)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rep = {"policy": POLICY_REPO, "dataset": a.dataset}

    def _write():
        (out / "gate0c.json").write_text(json.dumps(rep, indent=2, default=str))

    _install()
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch

    # --- Stage E FIRST: rollout / env / processor source (ground truth) ---
    rep["rollout_search"] = {
        "lerobot.envs.utils": _dump_fns("lerobot.envs.utils", ("rollout", "preprocess_observation", "add_envs_task")),
        "lerobot.scripts.lerobot_eval": _dump_fns("lerobot.scripts.lerobot_eval", ("rollout", "eval_policy", "eval_main")),
        "lerobot.envs.factory": _dump_fns("lerobot.envs.factory", ("make_env",)),
    }
    _write()

    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    dev = "cuda" if torch.cuda.is_available() else "cpu"    # CPU box (Standard quota) gets E/B/processor;
    rep["device"] = dev                                     # GPU stages (forward/grad/action) gated below
    policy = PI05Policy.from_pretrained(POLICY_REPO).to(dev).eval()
    cfg = policy.config
    backbone = _get_submodule(policy, BACKBONE_PATH)
    layers = backbone.layers
    d = int(next(layers[0].parameters()).shape[-1])
    rep["n_layers"], rep["d"] = len(layers), d
    rep["obs_delta_indices"] = getattr(cfg, "observation_delta_indices", None)
    rep["act_delta_indices"] = getattr(cfg, "action_delta_indices", None)
    _write()

    # --- Stage B: dataset metadata (DataFrame-safe) + delta_timestamps + one raw batch ---
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    meta = LeRobotDatasetMetadata(a.dataset)
    rep["fps"] = meta.fps
    rep["dataset_feature_keys"] = list(meta.features.keys())
    rep["meta_tasks"] = _df_head(getattr(meta, "tasks", None))
    rep["meta_episodes"] = _df_head(getattr(meta, "episodes", None))
    fps = meta.fps
    dt = {}
    if rep["act_delta_indices"] is not None:
        dt["action"] = [i / fps for i in rep["act_delta_indices"]]
    if rep["obs_delta_indices"] is not None:
        for k in meta.features:
            if k.startswith("observation"):
                dt[k] = [i / fps for i in rep["obs_delta_indices"]]
    rep["delta_timestamps"] = dt
    _write()

    ds = LeRobotDataset(a.dataset, delta_timestamps=dt or None)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True, num_workers=2)
    raw = next(iter(loader))
    rep["raw_batch_keys"] = {k: (list(v.shape) if torch.is_tensor(v) else str(type(v)))
                             for k, v in raw.items()}
    _write()

    # --- Stage C: processor (config!) → forward → grad-flow on real data ---
    from lerobot.policies.factory import make_pre_post_processors
    pre, post = make_pre_post_processors(cfg, pretrained_path=POLICY_REPO, dataset_stats=meta.stats)
    batch = pre(raw)
    rep["post_processor_batch_keys"] = {k: (list(v.shape) if torch.is_tensor(v) else str(type(v)))
                                        for k, v in batch.items()}
    rep["has_language_tokens_after_pre"] = "observation.language.tokens" in batch
    _write()

    # GPU-only stages (forward/grad + action). On a CPU box (Standard-quota r7i during a GPU-capacity
    # crunch) we still get the recipe above; the GPU confirmation runs when capacity returns.
    rep["proxy_action_method"] = ("predict_action_chunk" if hasattr(policy, "predict_action_chunk")
                                  else "select_action")
    if dev == "cuda":
        from infra.pg.ctrliface_run import capture_grad_hook
        store = {}
        h = layers[PROBE_L].register_forward_hook(capture_grad_hook(store))
        policy.zero_grad(set_to_none=True)
        with torch.enable_grad():
            fwd = policy.forward(batch)
            loss = fwd[0] if isinstance(fwd, tuple) else (fwd["loss"] if isinstance(fwd, dict) else fwd)
            assert loss.requires_grad, "real-batch loss has no grad"
            loss.backward()
        h.remove()
        hs = store["h"]
        rep["real_forward"] = {"loss": float(loss.item()), "h_L_shape": list(hs.shape),
                               "n_prefix": int(hs.shape[1]), "d": int(hs.shape[-1]),
                               "grad_finite": bool(torch.isfinite(hs.grad).all()),
                               "grad_norm": float(hs.grad.norm().item())}
        assert hs.grad is not None and rep["real_forward"]["grad_finite"], "real-data grad bad"
        _write()

        # Stage D: predict_action_chunk determinism under a fixed seed
        one = pre({k: (v[:1] if torch.is_tensor(v) else v[:1]) for k, v in raw.items()})
        fn = getattr(policy, rep["proxy_action_method"])
        rep["proxy_action_sig"] = str(inspect.signature(fn))
        torch.manual_seed(0)
        with torch.no_grad():
            a1 = fn(one)
        torch.manual_seed(0)
        with torch.no_grad():
            a2 = fn(one)
        rep["action_probe"] = {"shape": list(a1.shape),
                               "deterministic_under_seed": bool(torch.allclose(a1, a2)),
                               "max_abs_diff": float((a1 - a2).abs().max().item())}
        _write()

    print("GATE0C_OK", json.dumps({"device": dev, "d": rep["d"],
                                   "lang_tokens_after_pre": rep["has_language_tokens_after_pre"],
                                   "real_forward": rep.get("real_forward", "SKIPPED (cpu)"),
                                   "action_method": rep["proxy_action_method"]}, default=str))


if __name__ == "__main__":
    main()
