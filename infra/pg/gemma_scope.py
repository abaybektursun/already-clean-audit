# C5 — Gemma Scope applicability lens (spec §3 "Gemma Scope dependency", §7 W4).
"""Direct loader + JumpReLU forward + residual-stream harvest for Gemma Scope SAEs.

We load Gemma Scope JumpReLU SAEs DIRECTLY from the HF repo
``google/gemma-scope-2b-pt-res`` (entities.GEMMA_SCOPE_REPO), NOT via sae-lens —
sae-lens 6.44.0 pins transformer-lens==2.16.1, which conflicts with transformers
5.x (engineering plan §2). JumpReLU params are self-contained per SAE:

    W_enc  (d_model, d_sae)   encoder weights
    b_enc  (d_sae,)           encoder bias
    W_dec  (d_sae, d_model)   decoder weights
    b_dec  (d_model,)         decoder bias
    threshold (d_sae,)        per-latent JumpReLU threshold θ

stored as ``params.npz`` under ``layer_{N}/width_{W}/average_l0_{L}/``.

JumpReLU forward (Gemma Scope eq. 3-4; params are pre-rescaled at release so NO
input normalization is needed at inference — paper §3.2 "we fold ... so that no
input normalization is required for inference"):

    pre   = x @ W_enc + b_enc
    f     = pre * (pre > threshold)          # JumpReLU gate (shifted Heaviside)
    recon = f @ W_dec + b_dec

Discipline (CLAUDE.md / spec): fail loud. If an SAE's d_model does not match the
model's residual width, RAISE — never silently reshape.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import torch

from infra.pg.entities import GEMMA_SCOPE_REPO

# Gemma Scope trains residual SAEs at the *output* residual stream of a block
# (Fig. 1 "Residual SAE" sits after the block's final add). In transformers the
# Gemma2 decoder layers live at `model.layers[N]`; the block-N output residual is
# that layer module's forward output (hidden_states). We hook the layer module's
# output, which IS the post-block residual stream — the Gemma Scope site.
_PARAMS_FILE = "params.npz"
_PATH_RE = re.compile(r"layer_(\d+)/width_(\d+k)/average_l0_(\d+)/params\.npz$")


@dataclass(frozen=True)
class SAEChoice:
    layer: int
    width: str          # e.g. "16k"
    average_l0: int      # nominal L0 from the path label
    rel_path: str        # repo-relative dir, e.g. "layer_12/width_16k/average_l0_82"

    @property
    def params_path(self) -> str:
        return f"{self.rel_path}/{_PARAMS_FILE}"


def list_residual_saes(repo: str = GEMMA_SCOPE_REPO) -> list[SAEChoice]:
    """List every residual-stream JumpReLU SAE in the repo (introspected, not
    assumed — spec hard-rule 3). Returns SAEChoice rows parsed from real file paths."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo)
    out: list[SAEChoice] = []
    for f in files:
        m = _PATH_RE.search(f)
        if m:
            out.append(SAEChoice(
                layer=int(m.group(1)),
                width=m.group(2),
                average_l0=int(m.group(3)),
                rel_path=f[: f.rindex("/")],
            ))
    return sorted(out, key=lambda c: (c.layer, c.width, c.average_l0))


def pick_mid_residual_sae(
    repo: str = GEMMA_SCOPE_REPO,
    width: str = "16k",
    target_layer: int = 12,
    target_l0: int = 70,
) -> SAEChoice:
    """Pick a real mid-depth residual SAE at the requested width.

    Gemma 2 2B has 26 layers; layer 12 is the mid-network residual SAE the
    Gemma Scope paper reports fidelity for (Fig. 2 top-right, Fig. 13). Among the
    available average_l0 buckets we take the one closest to ``target_l0`` (the
    canonical SAEs target L0≈70-100). Raises if no SAE matches the width."""
    cands = [c for c in list_residual_saes(repo) if c.width == width]
    if not cands:
        widths = sorted({c.width for c in list_residual_saes(repo)})
        raise RuntimeError(
            f"No residual SAE at width={width!r} in {repo}. Available widths: {widths}"
        )
    layers = sorted({c.layer for c in cands})
    layer = min(layers, key=lambda L: abs(L - target_layer))
    at_layer = [c for c in cands if c.layer == layer]
    return min(at_layer, key=lambda c: abs(c.average_l0 - target_l0))


class JumpReLUSAE(torch.nn.Module):
    """JumpReLU SAE with Gemma Scope params loaded directly from the HF repo.

    Buffers (not Parameters — we never train this lens):
        W_enc (d_model, d_sae), b_enc (d_sae,)
        W_dec (d_sae, d_model), b_dec (d_model,), threshold (d_sae,)
    """

    def __init__(self, W_enc, b_enc, W_dec, b_dec, threshold):
        super().__init__()
        self.register_buffer("W_enc", W_enc)
        self.register_buffer("b_enc", b_enc)
        self.register_buffer("W_dec", W_dec)
        self.register_buffer("b_dec", b_dec)
        self.register_buffer("threshold", threshold)
        self.d_model = W_enc.shape[0]
        self.d_sae = W_enc.shape[1]
        # Contract: shapes must be internally consistent or we crash now, not later.
        assert b_enc.shape == (self.d_sae,), (b_enc.shape, self.d_sae)
        assert W_dec.shape == (self.d_sae, self.d_model), (W_dec.shape,)
        assert b_dec.shape == (self.d_model,), (b_dec.shape, self.d_model)
        assert threshold.shape == (self.d_sae,), (threshold.shape, self.d_sae)

    @classmethod
    def from_repo(
        cls,
        choice: SAEChoice,
        repo: str = GEMMA_SCOPE_REPO,
        dtype: torch.dtype = torch.float32,
    ) -> "JumpReLUSAE":
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=repo, filename=choice.params_path)
        npz = np.load(path)
        keys = set(npz.files)
        expected = {"W_enc", "b_enc", "W_dec", "b_dec", "threshold"}
        if not expected.issubset(keys):
            raise RuntimeError(
                f"Gemma Scope params {choice.params_path} missing keys: "
                f"{sorted(expected - keys)}. Present: {sorted(keys)}"
            )
        t = lambda k: torch.from_numpy(npz[k]).to(dtype)
        return cls(t("W_enc"), t("b_enc"), t("W_dec"), t("b_dec"), t("threshold"))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x (..., d_model) -> latents f (..., d_sae). JumpReLU gate (eq. 3)."""
        pre = x @ self.W_enc + self.b_enc
        # Shifted Heaviside: keep pre where it exceeds the per-latent threshold.
        return pre * (pre > self.threshold)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        f = self.encode(x)
        return self.decode(f), f


# ---------------------------------------------------------------------------
# Activation harvest + metrics. All compute is on Modal GPU; these helpers carry
# no Modal dependency so they're unit-importable.
# ---------------------------------------------------------------------------

def resolve_gemma2_layers(language_model) -> torch.nn.ModuleList:
    """Return the Gemma2 decoder `layers` ModuleList from a (possibly nested) LM.

    Works whether passed a Gemma2Model, a Gemma2ForCausalLM, or PG2's
    `model.language_model`. Fails loud if no `.layers` is found."""
    m = language_model
    for attr in ("model", "language_model"):
        if not hasattr(m, "layers") and hasattr(m, attr):
            m = getattr(m, attr)
    if not hasattr(m, "layers"):
        raise RuntimeError(
            f"No `.layers` ModuleList found under {type(language_model).__name__}; "
            f"cannot address the residual stream."
        )
    return m.layers


class ResidualHarvest:
    """Forward hook capturing the post-block residual stream at one decoder layer.

    Gemma Scope's residual SAE sits at the *output* of block N (Fig. 1), i.e. the
    decoder layer's output hidden_states. A Gemma2 layer returns a tuple whose
    [0] is hidden_states; we capture that. Use as a context manager so the hook is
    always removed (no leaked hooks across the two models)."""

    def __init__(self, layers: torch.nn.ModuleList, layer_idx: int):
        self.layer = layers[layer_idx]
        self.captured: torch.Tensor | None = None
        self._handle = None

    def _hook(self, module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        self.captured = hs

    def __enter__(self):
        self._handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()
        self._handle = None
        return False


def fvu(x: torch.Tensor, recon: torch.Tensor) -> float:
    """Fraction of variance unexplained — Gemma Scope's secondary fidelity metric
    (paper §4.1): mean reconstruction SE normalized by the dataset-mean baseline SE.
    FVU = E||x-recon||^2 / E||x - mean(x)||^2 = 1 - explained variance."""
    x = x.float()
    recon = recon.float()
    num = ((x - recon) ** 2).sum(-1).mean()
    denom = ((x - x.mean(0, keepdim=True)) ** 2).sum(-1).mean()
    return (num / denom).item()


def explained_variance(x: torch.Tensor, recon: torch.Tensor) -> float:
    return 1.0 - fvu(x, recon)


def mean_l0(f: torch.Tensor) -> float:
    """Mean number of active latents per token (Gemma Scope sparsity metric)."""
    return (f != 0).float().sum(-1).mean().item()
