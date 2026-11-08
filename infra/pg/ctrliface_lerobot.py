"""CTRLIFACE lerobot bindings — the ONE place the study touches lerobot 0.5.2 / PI05Policy /
LeRobotDataset, so gate0c (validates it), discovery (Gram/proxy), and the rollout jobs share a
single verified code path (no drift). Heavy imports (torch/lerobot) live INSIDE functions so the
module py_compiles on the control plane; the pure logic it feeds is in ctrliface_core /
ctrliface_run (weight-free-tested). Pinned facts (Gate-0a/0b/0c, lerobot 0.5.2):
  * policy class  lerobot.policies.pi05.modeling_pi05.PI05Policy
  * backbone      policy.model.paligemma_with_expert.paligemma.model.language_model (18 Gemma-1 layers)
  * demo dataset  HuggingFaceVLA/libero  (model-card training set)
No try/except hiding — a wrong assumption RAISES (CLAUDE.md).
"""
from __future__ import annotations

POLICY_REPO = "lerobot/pi05_libero_finetuned"
DATASET_REPO = "HuggingFaceVLA/libero"
BACKBONE_PATH = "model.paligemma_with_expert.paligemma.model.language_model"
# LIBERO suite → discovery/confirmation split (frozen spec §Freeze-resolutions).
DISCOVERY_SUITES = ("spatial", "object")
CONFIRM_SUITES = ("goal", "10")          # libero_goal + libero_10 (LIBERO-Long)


def load_policy(repo=POLICY_REPO, device="cuda"):
    """PI05Policy.from_pretrained → device → eval. Returns the policy (nn.Module)."""
    import torch
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    assert torch.cuda.is_available(), "no CUDA"
    return PI05Policy.from_pretrained(repo).to(device).eval()


def build_preprocessor(repo=POLICY_REPO, meta=None):
    """The policy's own pre-processor, which is what turns a raw LeRobot batch into a batch the
    policy can consume.

    THE RAW BATCH IS NOT THE MODEL'S INPUT. `LeRobotDataset` yields the recorded episode fields;
    the policy needs `observation.language.tokens`, which the pre-processor produces from the task
    string. Feeding the raw batch straight to `policy.forward` raises KeyError on that key — which
    is how M-A5 died once the loader was finally returning episodes (M-D70). Gate-0c established
    this recipe and asserted the key appears after `pre(raw)`; this is the same call, shared so a
    third caller does not rediscover it.
    """
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    md = meta if meta is not None else dataset_meta()
    cfg = PI05Config.from_pretrained(repo)
    pre, post = make_pre_post_processors(cfg, pretrained_path=repo, dataset_stats=md.stats)
    return pre, post


def get_backbone(policy):
    """(backbone_module, layers ModuleList, hidden_dim d). Fail-loud on the 18-layer Gemma-1 spine
    and that the container is the language_model backbone, not the action expert."""
    m = policy
    for p in BACKBONE_PATH.split("."):
        m = getattr(m, p)
    layers = m.layers
    assert len(layers) == 18, f"expected 18 backbone layers, got {len(layers)}"
    assert "language_model" in BACKBONE_PATH and "gemma_expert" not in BACKBONE_PATH
    d = int(next(layers[0].parameters()).shape[-1]) if hasattr(layers[0], "parameters") else None
    return m, layers, d


def dataset_meta(repo=DATASET_REPO):
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    return LeRobotDatasetMetadata(repo)


def build_delta_timestamps(policy, meta):
    """Action-chunk / obs-history delta_timestamps from the policy's delta INDICES × 1/fps
    (the lerobot training-loop recipe). Returns a dict {feature_key: [seconds,...]} or {} if the
    policy needs no temporal stacking."""
    cfg = policy.config
    fps = meta.fps
    obs_idx = getattr(cfg, "observation_delta_indices", None)
    act_idx = getattr(cfg, "action_delta_indices", None)
    dt = {}
    if act_idx is not None:
        dt["action"] = [i / fps for i in act_idx]
    if obs_idx is not None:
        for k in meta.features:
            if k.startswith("observation"):
                dt[k] = [i / fps for i in obs_idx]
    return dt


def _episodes_for_suites(meta, suites):
    """Episode indices whose task string matches any suite substring (the discovery/confirm
    firewall). lerobot 0.5.x `meta.episodes` is a pandas DataFrame indexed by episode with a
    per-row task column ('tasks' list or 'task' str). Returns a sorted list of episode ints, or
    None if no per-episode task mapping is present (caller decides)."""
    eps = getattr(meta, "episodes", None)
    if eps is None:
        return None
    keep = []
    # HF datasets.Dataset path (0.5.2 — Gate-0c confirmed)
    if hasattr(eps, "column_names"):
        cols = eps.column_names
        col = "tasks" if "tasks" in cols else ("task" if "task" in cols else None)
        if col is None:
            return None
        has_epi = "episode_index" in cols
        for i in range(len(eps)):
            row = eps[i]
            val = row[col]
            text = (" ".join(val) if isinstance(val, (list, tuple)) else str(val)).lower()
            if any(s in text for s in suites):
                keep.append(int(row["episode_index"]) if has_epi else i)
        return sorted(keep)
    # pandas DataFrame path
    if hasattr(eps, "iterrows"):
        col = "tasks" if "tasks" in eps.columns else ("task" if "task" in eps.columns else None)
        if col is None:
            return None
        for ep_idx, row in eps.iterrows():
            val = row[col]
            text = (" ".join(val) if isinstance(val, (list, tuple)) else str(val)).lower()
            if any(s in text for s in suites):
                keep.append(int(ep_idx))
        return sorted(keep)
    # dict/list fallback
    items = eps.items() if hasattr(eps, "items") else enumerate(eps)
    for ep_idx, info in items:
        tasks = info.get("tasks", []) if isinstance(info, dict) else []
        text = " ".join(tasks).lower() if tasks else ""
        if any(s in text for s in suites):
            keep.append(int(ep_idx))
    return sorted(keep)


def build_loader(repo, delta_timestamps, suites=None, batch_size=2, shuffle=True,
                 num_workers=2, seed=0, max_episodes=None):
    """A DataLoader over LeRobotDataset(repo), optionally filtered to `suites` (discovery/confirm
    firewall) or capped to the first `max_episodes` (so LeRobotDataset downloads only those episode
    files — the full HuggingFaceVLA/libero is 383 files / many GB; the Gram needs only a small demo
    sample). Returns (loader, n_episodes_used, filtered:bool). Fail-loud if a suite filter matches
    zero episodes."""
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    meta = LeRobotDatasetMetadata(repo)
    episodes = None
    filtered = False
    if suites is not None:
        episodes = _episodes_for_suites(meta, suites)
        assert episodes is not None and len(episodes) > 0, \
            f"suite filter {suites} matched no episodes (task naming drift — inspect meta.tasks)"
        filtered = True
    if max_episodes is not None:                          # cap → download only these episode files
        total = int(getattr(meta, "total_episodes", max_episodes))
        pool = episodes if episodes is not None else list(range(total))
        episodes = pool[:max_episodes]
    ds = LeRobotDataset(repo, delta_timestamps=delta_timestamps or None, episodes=episodes)
    g = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                         num_workers=num_workers, generator=g, drop_last=True)
    n_ep = len(episodes) if episodes is not None else getattr(meta, "total_episodes", None)
    return loader, n_ep, filtered


def to_device(batch, device="cuda"):
    import torch
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def loss_fn_factory(policy):
    """A callable batch -> scalar flow-matching loss (differentiable to h_L), for accumulate_gram.
    policy.forward returns (loss, dict) | {'loss':..} | loss."""
    def loss_fn(batch):
        fwd = policy.forward(batch)
        if isinstance(fwd, tuple):
            return fwd[0]
        if isinstance(fwd, dict):
            return fwd["loss"]
        return fwd
    return loss_fn


def action_fn_factory(policy):
    """A callable batch -> predicted action tensor for the deviation proxy. Prefers the full-chunk
    predictor (deterministic given a fixed noise seed) over the queue-managed select_action."""
    import torch
    if hasattr(policy, "predict_action_chunk"):
        def af(batch):
            with torch.no_grad():
                return policy.predict_action_chunk(batch)
        return af

    def af(batch):
        with torch.no_grad():
            if hasattr(policy, "reset"):
                policy.reset()
            return policy.select_action(batch)
    return af
