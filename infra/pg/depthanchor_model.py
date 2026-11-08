"""DEPTHANCHOR model layer — the only place that touches torch, the checkpoint or the GPU.

Everything measured here is named in `studies/depthanchor/SPEC.md`. The three objects:

  `Actor`      one wrapper around `MolmoAct2ForConditionalGeneration.predict_action` that can (a) let
               the model generate its own depth grid, (b) inject an arbitrary grid through the
               released adaptive-depth path, and (c) force the checkpoint's own per-layer depth gate
               to a constant. Every call returns the RAW normalized action chunk (captured before the
               released code clips it), the executed robot-scale chunk, the depth bins the model
               actually used, the generated token ids, and the number of tokens it emitted after the
               grid — so a call whose prefix length changed cannot be scored as if it had not.
  `DepthRefs`  the released VQ-VAE decoder and Depth-Anything-V2, loaded from the vendored copy of
               `generate_depth_annotation.py`, for G1 (is the model's depth depth?) and G2 (is the
               code set an order?).
  `gate_*`     readouts of `action_expert_depth_gate` — a per-layer `Linear(hidden, 1)` whose sigmoid
               multiplies the KEY and VALUE states at every depth-token position before the action
               expert cross-attends to them (`MolmoAct2Model._apply_depth_gate_to_layer_kv_states`).
               `config.action_expert_depth_gate_init_bias = -4.0`, i.e. sigmoid = 0.018 at init, so
               how far training moved this bias is a weights-only fact about how much of the depth
               prefix reaches the expert at all. It is also the matched ablation: forcing the gate to
               0 makes the depth positions carry nothing — they keep their place in the sequence and
               still take softmax mass in the expert's cross-attention, so this is the limit of the
               checkpoint's own gate rather than a deletion — which is a far better ceiling for a depth
               edit than swapping the RGB (spec amendment A1).

INJECTION INVARIANT (spec §2, checked on every call): `depth_cache["image"]` is ALWAYS the same array
as `images[0]`. `_compute_depth_update_mask` then sees cosine similarity 1.0 in every patch, marks
nothing for regeneration, and `_generate_depth_prefix` replays the cached bins through
`_run_depth_decode_step`, so the keys and values the action expert reads are computed from the
injected grid. If any cell were regenerated the returned `depth_bins` would differ from the requested
grid, and `Actor.inject` raises rather than scoring a grid the model did not use.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import time
import types
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from infra.pg import depthanchor_core as core

MODEL_REPO = "allenai/MolmoAct2-Think-LIBERO"
DEPTH_REPO = "allenai/MolmoAct-7B-D-0812"
NORM_TAG = "libero"
VAE_IMAGE_SIZE = 320
DEPTH_INPUT_SIZE = 518
VENDOR_REF = pathlib.Path(__file__).resolve().parent / "vendor" / "molmoact2" / "generate_depth_annotation.py"


# ------------------------------------------------------------------ checkpoint
def _hub_snapshot(repo: str) -> pathlib.Path:
    """Resolve a repo to its local snapshot directory inside the restored HF cache. Offline: the
    weights come from S3 via `run_job.sh`, never from the hub, and a missing repo must raise here
    rather than start a 22 GB download in the middle of a cell."""
    from huggingface_hub import snapshot_download
    return pathlib.Path(snapshot_download(repo, local_files_only=True))


def load_model(dtype: torch.dtype = torch.float32, device: str = "cuda"):
    """The card's own loading recipe: `trust_remote_code=True`, float32 ("performs best at float32
    inference, requiring approximately 26GB with CUDA graphs"), `.to(device).eval()`."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    path = _hub_snapshot(MODEL_REPO)
    processor = AutoProcessor.from_pretrained(str(path), trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(path), trust_remote_code=True, dtype=dtype,
    ).to(device).eval()
    return model, processor, path


# ------------------------------------------------------------------ the actor
@dataclass
class ActResult:
    a_norm: np.ndarray                 # (10, 7) the flow solver's output, BEFORE the released clip
    a_robot: np.ndarray                # (10, 7) what the simulator would execute
    bins: np.ndarray                   # (100,) the grid the model actually used
    token_ids: np.ndarray              # the generated prefix, verbatim
    n_grid_tokens: int                 # how many of them decoded to depth codes
    n_tail_tokens: int                 # how many it emitted after the grid before <depth_end>
    seconds: float
    gate: Optional[np.ndarray] = None  # (num_layers,) the depth gate this call applied
    meta: Dict[str, object] = field(default_factory=dict)

    def as_record(self) -> Dict[str, object]:
        return {"n_grid_tokens": self.n_grid_tokens, "n_tail_tokens": self.n_tail_tokens,
                "seconds": self.seconds, "bins": self.bins.astype(int).tolist(),
                "a_norm": self.a_norm.tolist(), "a_robot": self.a_robot.tolist(),
                "gate": None if self.gate is None else self.gate.tolist(),
                "meta": dict(self.meta)}


EditNotCompleted = core.EditNotCompleted     # defined in the torch-free core; re-exported here


class Actor:
    """One resident model, every call recorded. `frame` is a mapping with keys
    `agentview` (H,W,3 uint8), `wrist` (H,W,3 uint8), `task` (str), `state` (8,) float32."""

    def __init__(self, model, processor, *, enable_cuda_graph: bool = True,
                 num_steps: Optional[int] = None):
        self.model = model
        self.processor = processor
        self.enable_cuda_graph = bool(enable_cuda_graph)
        self.num_steps = num_steps
        self.calls = 0
        # The recording wrappers are installed on the MODEL, once, so the capture dict has to live
        # there too: a second Actor over the same model (the latency stage builds three) would
        # otherwise get a fresh empty dict that the already-installed wrapper never writes to, and
        # every call would raise "the normalized action chunk was not captured".
        captured = getattr(model, "_depthanchor_captured", None)
        if captured is None:
            captured = {}
            model._depthanchor_captured = captured
        self._captured: Dict[str, object] = captured
        self._install_recorders()

    # -------------------------------------------------- recorders and overrides
    def _install_recorders(self) -> None:
        """Two instance-level wrappers, both read-only with respect to the released computation:
        (1) `stats.unnormalize_action` records its input, which is exactly the normalized action
            chunk the flow solver produced — the released path then clips it to [-1,1] before the
            affine map, and for `norm_tag='libero'` the gripper dimension is not normalized at all
            (`normalize_gripper: false`), so the returned robot-scale chunk is a clipped view of the
            model's output and would hide any edit effect that lives outside [-1,1];
        (2) `MolmoAct2Model._depth_gate_from_condition` records the per-layer gate it computed, and
            can force it to a constant. Nothing else is touched.
        """
        stats = self.model._get_robot_stats()
        if not getattr(stats, "_depthanchor_recording", False):
            original = stats.unnormalize_action
            captured = self._captured

            def recording_unnormalize(action, norm_tag, _orig=original, _cap=captured):
                arr = action.detach().float().cpu().numpy() if torch.is_tensor(action) else np.asarray(action)
                _cap["a_norm"] = np.array(arr, dtype=np.float64, copy=True)
                return _orig(action, norm_tag)

            stats.unnormalize_action = recording_unnormalize
            stats._depthanchor_recording = True
        self.stats = stats

        backbone = self.model.model
        if not getattr(backbone, "_depthanchor_gate_hook", False):
            original_gate = backbone._depth_gate_from_condition
            captured = self._captured
            state = {"force": None}

            def recording_gate(*, input_ids, encoder_attention_mask, layer_kv_states,
                               _orig=original_gate, _cap=captured, _state=state):
                gates, mask = _orig(input_ids=input_ids,
                                    encoder_attention_mask=encoder_attention_mask,
                                    layer_kv_states=layer_kv_states)
                if gates is None:
                    _cap["gate"] = None
                    _cap["depth_positions"] = None if mask is None else int(mask.sum().item())
                    return gates, mask
                as_list = list(gates) if isinstance(gates, (list, tuple)) else [gates]
                _cap["gate_trained"] = np.array([float(g.reshape(-1)[0].item()) for g in as_list],
                                                dtype=np.float64)
                if _state["force"] is not None:
                    value = float(_state["force"])
                    gates = [torch.full_like(g, value) for g in as_list]
                    as_list = gates
                _cap["gate"] = np.array([float(g.reshape(-1)[0].item()) for g in as_list],
                                        dtype=np.float64)
                _cap["depth_positions"] = None if mask is None else int(mask.sum().item())
                return (gates if isinstance(gates, list) else gates), mask

            backbone._depth_gate_from_condition = recording_gate
            backbone._depthanchor_gate_hook = True
            backbone._depthanchor_gate_state = state
        self._gate_state = backbone._depthanchor_gate_state

    def force_gate(self, value: Optional[float]) -> None:
        """`None` = the checkpoint's trained gate; a float = that constant in every layer. 0.0 is the
        matched depth ablation (the action expert reads no depth), 1.0 the un-gated prefix."""
        self._gate_state["force"] = None if value is None else float(value)

    # -------------------------------------------------- the two call paths
    def natural(self, frame: Dict[str, object], *, seed: int = 0) -> ActResult:
        """The model generates its own grid: `enable_adaptive_depth=False`, one greedy code at a
        time, exactly the released non-cached path. This is what defines `b_f`."""
        return self._call(frame, bins=None, seed=seed)

    def inject(self, frame: Dict[str, object], bins, *, seed: int = 0) -> ActResult:
        """Replay `bins` through the adaptive path and refuse to return if the model used anything
        else (spec §8: "bank `generated_token_ids` and assert equality with `b'` on every call")."""
        want = np.asarray(bins, dtype=np.int64).reshape(-1)
        if want.shape != (core.N_CELLS,):
            raise ValueError(f"inject needs {core.N_CELLS} bins, got {want.shape}")
        try:
            out = self._call(frame, bins=want, seed=seed)
        except core.EditNotCompleted:
            raise
        except RuntimeError as exc:
            if "did not emit <depth_end>" in str(exc):
                raise core.EditNotCompleted(str(exc)) from exc
            raise
        if not np.array_equal(out.bins, want):
            n_bad = int(np.sum(out.bins != want))
            raise RuntimeError(
                f"the adaptive path regenerated {n_bad} of {core.N_CELLS} cells instead of replaying "
                f"the injected grid; the injection invariant is broken and no Δ from this call is "
                f"interpretable (first mismatch at cell {int(np.flatnonzero(out.bins != want)[0])})")
        return out

    def adaptive(self, frame: Dict[str, object], cache: Optional[Dict[str, object]], *,
                 seed: int = 0) -> Tuple[ActResult, Dict[str, object]]:
        """The released DEPLOYMENT path: `enable_adaptive_depth=True` with the caller-owned cache
        threaded from the previous step, so cells whose RGB patch is unchanged (cosine >= 0.996) reuse
        their cached code and only the changed ones regenerate. `cache=None` on the first step
        regenerates the whole grid. Unlike `inject`, the returned grid is EXPECTED to differ from the
        cached one — that difference is the mechanism this arm exists to exercise — so nothing is
        asserted about it; what is recorded is how many cells changed."""
        res = self._call(frame, bins=None, seed=seed, cache=cache, adaptive=True)
        new_cache = self._captured.get("depth_cache_out")
        if new_cache is None:
            raise RuntimeError("the adaptive path returned no depth cache")
        prev = None if cache is None else np.asarray(cache["depth_bins"], dtype=np.int64).reshape(-1)
        # the checkpoint sometimes closes its depth block before the grid is full, so two consecutive
        # grids need not be the same length; comparing them elementwise would raise instead of measuring
        same_length = prev is not None and prev.shape == res.bins.shape
        res.meta["cells_regenerated"] = int(np.sum(prev != res.bins)) if same_length else None
        res.meta["n_grid_cells"] = int(res.bins.size)
        if prev is not None and not same_length:
            res.meta["grid_length_changed_from"] = int(prev.size)
        res.meta["adaptive_cache_threaded"] = cache is not None
        return res, new_cache

    def _call(self, frame: Dict[str, object], bins: Optional[np.ndarray], seed: int,
              cache: Optional[Dict[str, object]] = None, adaptive: Optional[bool] = None
              ) -> ActResult:
        images, agentview = _frame_images(frame)
        depth_cache = cache
        if bins is not None:
            # THE INVARIANT: the cached image IS the current image, so every patch is unchanged and
            # the released code replays the whole grid instead of regenerating any of it.
            depth_cache = {"image": agentview, "depth_bins": bins.astype(np.int64)}
        use_adaptive = (bins is not None) if adaptive is None else bool(adaptive)
        generator = torch.Generator(device=self.model.device).manual_seed(int(seed))
        # every captured key is cleared, not just the two that are always written: a call that
        # produces no gate would otherwise report the PREVIOUS call's trained gate as its own
        for key in ("a_norm", "gate", "gate_trained", "depth_positions", "depth_cache_out"):
            self._captured.pop(key, None)
        t0 = time.perf_counter()
        out = self.model.predict_action(
            processor=self.processor,
            images=images,
            task=str(frame["task"]),
            state=np.asarray(frame["state"], dtype=np.float32),
            norm_tag=NORM_TAG,
            inference_action_mode="continuous",
            enable_depth_reasoning=True,
            enable_adaptive_depth=use_adaptive,
            depth_cache=depth_cache,
            num_steps=self.num_steps,
            generator=generator,
            normalize_language=True,
            enable_cuda_graph=self.enable_cuda_graph,
        )
        seconds = time.perf_counter() - t0
        self.calls += 1
        a_norm = self._captured.get("a_norm")
        if a_norm is None:
            raise RuntimeError("the normalized action chunk was not captured; the recording wrapper "
                               "on `unnormalize_action` is not installed on the live stats object")
        a_robot = out.actions.detach().float().cpu().numpy()
        a_norm = np.asarray(a_norm, dtype=np.float64)
        a_norm = a_norm.reshape(a_norm.shape[-2], a_norm.shape[-1]) if a_norm.ndim == 3 else a_norm
        a_robot = a_robot.reshape(a_robot.shape[-2], a_robot.shape[-1]) if a_robot.ndim == 3 else a_robot
        token_ids = out.generated_token_ids.detach().cpu().numpy().reshape(-1)
        used = out.depth_bins.detach().cpu().numpy().reshape(-1).astype(np.int64)
        self._captured["depth_cache_out"] = out.depth_cache
        n_grid, n_tail = _split_prefix(token_ids, self.model.config)
        return ActResult(
            a_norm=a_norm, a_robot=a_robot, bins=used, token_ids=token_ids,
            n_grid_tokens=n_grid, n_tail_tokens=n_tail, seconds=seconds,
            gate=self._captured.get("gate"),
            meta={"seed": int(seed), "adaptive": use_adaptive,
                  "gate_forced": self._gate_state["force"],
                  "gate_trained": None if self._captured.get("gate_trained") is None
                                  else np.asarray(self._captured["gate_trained"]).tolist(),
                  "depth_positions": self._captured.get("depth_positions"),
                  "cuda_graph": self.enable_cuda_graph},
        )


def _frame_images(frame: Dict[str, object]) -> Tuple[List[object], np.ndarray]:
    """(images passed to the processor, the exact agentview array used for the depth cache).
    Camera order is the checkpoint's: agentview first, wrist second (`norm_stats` camera_keys)."""
    from PIL import Image
    agentview = np.ascontiguousarray(np.asarray(frame["agentview"], dtype=np.uint8))
    wrist = np.ascontiguousarray(np.asarray(frame["wrist"], dtype=np.uint8))
    for name, arr in (("agentview", agentview), ("wrist", wrist)):
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"{name} image must be HxWx3 uint8, got {arr.shape}")
    return [Image.fromarray(agentview), Image.fromarray(wrist)], agentview


def _split_prefix(token_ids: np.ndarray, config) -> Tuple[int, int]:
    """How many generated tokens were depth codes, and how many came after them before <depth_end>.
    An edited prefix can make the model emit extra tokens before <depth_end> — that changes the
    prefix LENGTH, which changes the action for a reason that is not the depth edit, so the count is
    recorded on every call and any call whose tail differs from the reference's is flagged."""
    start = int(config.depth_token_start_id)
    end = start + int(config.num_depth_tokens)
    is_code = (token_ids >= start) & (token_ids < end)
    n_grid = int(is_code.sum())
    last_code = int(np.flatnonzero(is_code)[-1]) if n_grid else -1
    return n_grid, int(token_ids.size - last_code - 1)


# ------------------------------------------------------------------ the depth references (G1, G2)
def _ensure_cv2() -> str:
    """The vendored annotation script imports `cv2` at module level (its Depth-Anything preprocessing
    is written with OpenCV). A job that installs the simulator gets it as an `hf-libero` dependency; the
    figures job does not install the simulator, so it is ensured here rather than being an invisible
    prerequisite. Idempotent and fail-loud."""
    import importlib.util
    import subprocess
    import sys as _sys
    if importlib.util.find_spec("cv2") is None:
        subprocess.run([_sys.executable, "-m", "pip", "install", "-q", "opencv-python-headless"],
                       check=True)
        importlib.invalidate_caches()
        if importlib.util.find_spec("cv2") is None:
            raise RuntimeError("pip reported success but cv2 is still not importable")
    import cv2
    return cv2.__version__


def _load_vendored_ref():
    """Import the vendored `generate_depth_annotation.py` verbatim, with the two imports it makes for
    reasons this study does not need replaced by stubs: `lerobot.utils.constants` (a dataset root we
    never use) and `av` (video decoding; our frames are parquet images). Stubbing rather than editing
    keeps the vendored file byte-identical to the released one, which is what its sha256 in
    `vendor/molmoact2/README.md` certifies."""
    if "lerobot" not in sys.modules:
        lerobot = types.ModuleType("lerobot")
        utils = types.ModuleType("lerobot.utils")
        constants = types.ModuleType("lerobot.utils.constants")
        constants.HF_LEROBOT_HOME = pathlib.Path(os.environ.get("HF_LEROBOT_HOME", "/work/lerobot"))
        utils.constants = constants
        lerobot.utils = utils
        sys.modules.update({"lerobot": lerobot, "lerobot.utils": utils,
                            "lerobot.utils.constants": constants})
    if "av" not in sys.modules:
        sys.modules["av"] = types.ModuleType("av")
    if not VENDOR_REF.is_file():
        raise FileNotFoundError(f"vendored depth reference missing: {VENDOR_REF}")
    spec = importlib.util.spec_from_file_location("molmoact2_depth_ref", str(VENDOR_REF))
    module = importlib.util.module_from_spec(spec)
    sys.modules["molmoact2_depth_ref"] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class DepthRefs:
    depth_model: object
    vae: object
    module: object
    device: torch.device

    @torch.no_grad()
    def reference_bins(self, rgb: np.ndarray) -> np.ndarray:
        """The bins the released annotation pipeline would write for this frame: Depth-Anything-V2 at
        518, min-max normalized per frame to uint8, nearest-resized to 320, then the VQ-VAE encoder.
        Batch size 1, which the released script calls "exact original preprocessing behavior"."""
        import cv2
        bgr = cv2.cvtColor(np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)), cv2.COLOR_RGB2BGR)
        _, codes = self.module.encode_depth_codes(
            [bgr], self.depth_model, self.vae, depth_input_size=DEPTH_INPUT_SIZE,
            vae_image_size=VAE_IMAGE_SIZE, batch_size=1, device=self.device)
        return np.asarray(codes[0], dtype=np.int64).reshape(-1)

    @torch.no_grad()
    def depth_anything_map(self, rgb: np.ndarray) -> np.ndarray:
        """The raw Depth-Anything-V2 output resized to the source resolution. Larger = nearer
        (disparity convention); the sign is confirmed against simulator ground truth in G1B."""
        import cv2
        bgr = cv2.cvtColor(np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)), cv2.COLOR_RGB2BGR)
        tensor, (h, w) = self.module.depth_image_to_tensor(bgr, DEPTH_INPUT_SIZE, self.device)
        depth = self.depth_model(tensor[None])[0]
        depth = torch.nn.functional.interpolate(depth[None, None], (h, w), mode="bilinear",
                                                align_corners=True)[0, 0]
        return depth.detach().float().cpu().numpy()

    @torch.no_grad()
    def decode_bins(self, bins) -> np.ndarray:
        """A 100-vector of codes -> the VQ-VAE decoder's 320x320 depth image (its own units)."""
        b = np.asarray(bins, dtype=np.int64).reshape(core.GRID, core.GRID)
        codes = torch.from_numpy(b)[None].to(self.device)
        out = self.vae.decode(codes)
        return out.detach().float().cpu().numpy()[0, 0]

    def cell_depths(self, bins) -> np.ndarray:
        """Per-cell mean decoded depth, i.e. the decoded grid at the resolution the codes live on."""
        img = self.decode_bins(bins)
        s = img.shape[0] // core.GRID
        return img.reshape(core.GRID, s, core.GRID, s).mean(axis=(1, 3)).reshape(-1)

    @torch.no_grad()
    def code_ladder(self) -> core.CodeLadder:
        """`d(c)` twice: a uniform grid of the code, and the code alone in a grid of the median code
        (read on its own patch). The gap between the two is the decoder's placement noise, which is
        what G2's neighbour separation has to beat."""
        d_primary = np.empty(core.N_CODES, dtype=np.float64)
        for c in range(core.N_CODES):
            d_primary[c] = float(self.decode_bins(np.full(core.N_CELLS, c)).mean())
        median_code = int(np.argsort(d_primary, kind="stable")[core.N_CODES // 2])
        centre = (core.GRID // 2) * core.GRID + (core.GRID // 2)
        d_local = np.empty(core.N_CODES, dtype=np.float64)
        for c in range(core.N_CODES):
            grid = np.full(core.N_CELLS, median_code, dtype=np.int64)
            grid[centre] = c
            d_local[c] = float(self.cell_depths(grid)[centre])
        return core.CodeLadder(d_primary=d_primary, d_local=d_local)


def load_depth_refs(device: str = "cuda") -> DepthRefs:
    _ensure_cv2()
    module = _load_vendored_ref()
    path = _hub_snapshot(DEPTH_REPO)
    dev = torch.device(device)
    depth_model, vae = module.load_models(
        depth_ckpt=path / module.DEFAULT_DEPTH_CKPT,
        vae_ckpt=path / module.DEFAULT_VAE_CKPT,
        vae_image_size=VAE_IMAGE_SIZE,
        device=dev,
    )
    return DepthRefs(depth_model=depth_model, vae=vae, module=module, device=dev)


# ------------------------------------------------------------------ weights-only gate readout
def gate_weights_readout(model) -> Dict[str, object]:
    """What the trained per-layer depth gate is, before any input: the bias of each gate head, its
    sigmoid at zero-pooled input, and the initialization the config declares. A gate still sitting at
    its `-4.0` init would mean the depth prefix reaches the action expert at ~1.8 % of its magnitude
    whatever the frame, which is measurable from the safetensors alone."""
    backbone = model.model
    gate = backbone.action_expert_depth_gate
    if gate is None:
        return {"present": False}
    heads = list(gate) if isinstance(gate, torch.nn.ModuleList) else [gate]
    bias = np.array([float(h.bias.detach().float().reshape(-1)[0].item()) for h in heads])
    wnorm = np.array([float(h.weight.detach().float().norm().item()) for h in heads])
    init_bias = float(model.config.action_expert_depth_gate_init_bias)
    return {"present": True,
            "per_layer": bool(getattr(model.config, "action_expert_depth_gate_per_layer", False)),
            "n_heads": len(heads), "init_bias": init_bias,
            "bias": bias.tolist(), "sigmoid_at_bias": (1.0 / (1.0 + np.exp(-bias))).tolist(),
            "weight_l2": wnorm.tolist(),
            "sigmoid_at_init_bias": float(1.0 / (1.0 + np.exp(-init_bias)))}


# ------------------------------------------------------------------ provenance
def provenance(model, processor, model_path: pathlib.Path) -> Dict[str, object]:
    cfg = model.config
    return {
        "model_repo": MODEL_REPO,
        "snapshot_dir": str(model_path),
        "snapshot_sha": model_path.name,
        "dtype": str(next(model.parameters()).dtype),
        "device": str(model.device),
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
        "config": {k: getattr(cfg, k, None) for k in (
            "num_depth_codes", "num_depth_tokens", "depth_token_start_id", "depth_start_token_id",
            "depth_end_token_id", "depth_output_token_id", "max_action_horizon", "n_obs_steps",
            "max_action_dim", "flow_matching_num_steps", "action_mode", "enable_depth_reasoning",
            "action_expert_depth_gate", "action_expert_depth_gate_per_layer",
            "action_expert_depth_gate_init_bias", "depth_mode")},
        "norm_stats_libero": _norm_stats_summary(model),
    }


def _norm_stats_summary(model) -> Dict[str, object]:
    stats = model._get_robot_stats()
    meta = stats.get_metadata(NORM_TAG)
    a = meta.get("action_stats", {})
    return {"norm_mode": stats.norm_mode, "action_horizon": meta.get("action_horizon"),
            "n_action_steps": meta.get("n_action_steps"), "setup_type": meta.get("setup_type"),
            "control_mode": meta.get("control_mode"), "normalize_gripper": meta.get("normalize_gripper"),
            "q01": a.get("q01"), "q99": a.get("q99"), "mask": a.get("mask"),
            "camera_keys": meta.get("camera_keys")}
