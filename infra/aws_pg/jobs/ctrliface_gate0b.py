"""CTRLIFACE Gate 0b — REAL-WEIGHTS preflight (§9.0 part 2): confirm the wiring on the loaded
PI05Policy before the freeze. Staged, WRITE-AS-YOU-GO (a failure at stage N keeps stages 1..N-1),
fail-loud on the science asserts.

Stages:
  A. Load `lerobot.policies.pi05.modeling_pi05.PI05Policy` from lerobot/pi05_libero_finetuned;
     print config (input/output features, chunk/action dims, image keys) + the backbone-layer path
     `model.paligemma_with_expert.paligemma.model.language_model.layers` (assert 18) + hidden d.
  B. Compartment gate (runtime, π0.5-correct): the hooked container is `language_model` (NOT
     gemma_expert), 18 layers.
  C. HOOK-FIRES + GRAD-FLOW on real weights: a synthetic training batch (built from config
     features) → forward → loss → backward; assert ∂loss/∂h_L is finite & non-None; record the
     h_L shape (B, seq, d) → n_prefix (backbone stream length) and d. This is the Gram wiring.
  D. Discover lerobot 0.5.2's in-process rollout/eval API (members + signatures) for the intact
     success (P7) + the positive control, run next.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.ctrliface_gate0b --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/ctrliface_gate0b --budget-min 50
"""
import argparse
import inspect
import json
import pathlib
import subprocess
import sys

POLICY_REPO = "lerobot/pi05_libero_finetuned"
BACKBONE_PATH = "model.paligemma_with_expert.paligemma.model.language_model"
PROBE_L = 9   # a mid-stack backbone layer for the wiring probe


def _install():
    subprocess.run(["bash", "-lc",
                    "apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y "
                    "git python3-dev build-essential libgl1 libglib2.0-0"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "lerobot[libero,pi] @ git+https://github.com/huggingface/lerobot.git"], check=True)


def _get_submodule(root, dotted):
    m = root
    for part in dotted.split("."):
        m = getattr(m, part)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rep = {"repo": POLICY_REPO, "backbone_path": BACKBONE_PATH, "probe_L": PROBE_L}

    def _write():
        (out / "gate0b.json").write_text(json.dumps(rep, indent=2, default=str))

    _install()
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    assert torch.cuda.is_available(), "no CUDA"
    rep["device"] = torch.cuda.get_device_name(0)

    # --- Stage A: load + config + backbone path ---
    policy = PI05Policy.from_pretrained(POLICY_REPO).to("cuda").eval()
    cfg = policy.config
    to_dict = getattr(cfg, "to_dict", None)
    rep["config"] = to_dict() if callable(to_dict) else str(cfg)
    # feature specs (keys + shapes) — the batch schema
    def _featmap(fs):
        out = {}
        for k, v in (fs or {}).items():
            out[k] = {"type": str(getattr(v, "type", "?")), "shape": list(getattr(v, "shape", []))}
        return out
    rep["input_features"] = _featmap(getattr(cfg, "input_features", {}))
    rep["output_features"] = _featmap(getattr(cfg, "output_features", {}))
    for attr in ("chunk_size", "n_action_steps", "n_obs_steps", "max_action_dim", "max_state_dim"):
        rep[f"cfg.{attr}"] = getattr(cfg, attr, None)

    backbone = _get_submodule(policy, BACKBONE_PATH)
    layers = backbone.layers
    rep["n_backbone_layers"] = len(layers)
    assert len(layers) == 18, f"expected 18 backbone layers, got {len(layers)}"
    d = getattr(getattr(backbone, "config", None), "hidden_size", None)
    rep["hidden_size_cfg"] = d
    _write()

    # --- Stage B: runtime compartment gate ---
    assert "language_model" in BACKBONE_PATH and "gemma_expert" not in BACKBONE_PATH, \
        "hook target must be the backbone language_model, not the action expert"
    rep["compartment_gate_runtime"] = {"container": BACKBONE_PATH, "n_layers": len(layers),
                                       "is_backbone_not_expert": True}
    _write()

    # --- Stage C: hook-fires + grad-flow on real weights (synthetic batch from config features) ---
    B = 1
    dev = "cuda"
    batch = {}
    for k, ft in rep["input_features"].items():
        shp = ft["shape"]
        if "VISUAL" in ft["type"] or "image" in k.lower():
            batch[k] = torch.rand(B, *shp, device=dev)                 # images in [0,1]
        else:
            batch[k] = torch.randn(B, *shp, device=dev)                # state etc.
    # action target (chunk_size × action_dim) + pad mask + task string
    act_dim = None
    for k, ft in rep["output_features"].items():
        if "ACTION" in ft["type"] or k == "action":
            act_dim = ft["shape"][-1]
    chunk = rep.get("cfg.chunk_size") or rep.get("cfg.n_action_steps") or 50
    assert act_dim is not None, f"could not find action dim in output_features {rep['output_features']}"
    batch["action"] = torch.randn(B, chunk, act_dim, device=dev)
    batch["action_is_pad"] = torch.zeros(B, chunk, dtype=torch.bool, device=dev)
    batch["task"] = ["pick up the object"] * B
    rep["synthetic_batch_keys"] = {k: (list(v.shape) if torch.is_tensor(v) else v) for k, v in batch.items()}
    _write()

    from infra.pg.ctrliface_run import capture_grad_hook
    store = {}
    h = layers[PROBE_L].register_forward_hook(capture_grad_hook(store))
    policy.zero_grad(set_to_none=True)
    with torch.enable_grad():
        fwd = policy.forward(batch)
        loss = fwd[0] if isinstance(fwd, tuple) else (fwd["loss"] if isinstance(fwd, dict) else fwd)
        assert loss.requires_grad, "loss does not require grad (backbone forward under no_grad?)"
        loss.backward()
    h.remove()
    hs = store["h"]
    rep["hook_fired"] = True
    rep["h_L_shape"] = list(hs.shape)                    # (B, seq, d) → seq = n_prefix
    rep["n_prefix"] = int(hs.shape[1])
    rep["d_runtime"] = int(hs.shape[-1])
    rep["grad_is_none"] = hs.grad is None
    assert hs.grad is not None, "∂loss/∂h_L is None — grad did not reach the backbone layer"
    rep["grad_finite"] = bool(torch.isfinite(hs.grad).all())
    rep["grad_norm"] = float(hs.grad.norm().item())
    rep["loss_value"] = float(loss.item())
    assert rep["grad_finite"] and rep["grad_norm"] > 0, "activation grad not finite/nonzero"
    _write()

    # --- Stage D: discover the in-process rollout/eval API ---
    import importlib
    import importlib.util
    api = {}
    for modname in ("lerobot.scripts.eval", "lerobot.envs.factory", "lerobot.scripts.lerobot_eval"):
        if importlib.util.find_spec(modname) is None:
            api[modname] = "MISSING"
            continue
        m = importlib.import_module(modname)
        fns = {}
        for n, obj in inspect.getmembers(m, inspect.isfunction):
            if any(t in n.lower() for t in ("eval", "rollout", "make_env")):
                fns[n] = str(inspect.signature(obj))
        api[modname] = fns
    rep["rollout_api"] = api
    _write()

    print("GATE0B_OK", json.dumps({"n_prefix": rep["n_prefix"], "d": rep["d_runtime"],
                                   "grad_norm": rep["grad_norm"], "loss": rep["loss_value"],
                                   "rollout_api": {k: (list(v) if isinstance(v, dict) else v)
                                                   for k, v in rep["rollout_api"].items()}}, default=str))


if __name__ == "__main__":
    main()
