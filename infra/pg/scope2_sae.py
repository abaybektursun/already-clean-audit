"""Gemma Scope 2 JumpReLU SAE — minimal self-contained loader (no sae_lens dependency).

Loads one shipped SAE's parameters from the LOCAL HF cache (weights are S3-synced by
run_job.sh; the seed job downloaded exactly the selected folders) and exposes numpy
params for infra.pg.helmholtz2_core.jumprelu_encode/sae_decode. Handles params.npz
(Gemma Scope v1 convention) and .safetensors. Fail-loud on unknown layouts: the error
prints the actual keys/files so the BIND can be corrected from the artifact.
"""
from __future__ import annotations

import pathlib

import numpy as np

REQUIRED = ("W_enc", "W_dec", "b_enc", "b_dec", "threshold")


def _snapshot_dir(repo_id: str) -> pathlib.Path:
    from huggingface_hub import snapshot_download
    return pathlib.Path(snapshot_download(repo_id, local_files_only=True,
                                          allow_patterns=None))


def list_sae_dirs(repo_id: str) -> list:
    """All directories in the local snapshot that contain an SAE params file."""
    root = _snapshot_dir(repo_id)
    out = set()
    for f in root.rglob("*"):
        if f.is_file() and (f.name == "params.npz" or f.suffix == ".safetensors"):
            out.add(str(f.parent.relative_to(root)))
    return sorted(out)


def load_scope2_sae(repo_id: str, sae_dir: str, d_model: int) -> dict:
    """Returns {W_enc (d,F), W_dec (F,d), b_enc (F,), b_dec (d,), threshold (F,)} float32.
    Orientation fixed against d_model; asserts every shape."""
    root = _snapshot_dir(repo_id)
    d = root / sae_dir
    assert d.is_dir(), f"{d} not a directory; have: {list_sae_dirs(repo_id)[:20]}"
    npz = d / "params.npz"
    if npz.exists():
        raw = dict(np.load(npz))
    else:
        st = sorted(d.glob("*.safetensors"))
        assert st, f"no params.npz or .safetensors under {d}: {sorted(p.name for p in d.iterdir())}"
        # safetensors.numpy cannot decode bf16 (Scope-2 stores bf16) -> load via torch, cast fp32
        from safetensors.torch import load_file
        raw = {}
        for f in st:
            raw.update({k: v.float().numpy() for k, v in load_file(str(f)).items()})
    # canonicalize keys: accept exact names or unique substring matches (w_enc, encoder.weight...)
    params = {}
    for want in REQUIRED:
        hits = [k for k in raw if k == want] or \
               [k for k in raw if want.lower().replace("_", "") in k.lower().replace("_", ".").replace(".", "")]
        assert len(hits) == 1, f"key {want!r}: {len(hits)} matches in {sorted(raw)} at {d}"
        params[want] = np.asarray(raw[hits[0]], dtype=np.float32)
    W_enc, W_dec = params["W_enc"], params["W_dec"]
    if W_enc.shape[0] != d_model:            # stored (F, d) -> orient to (d, F)
        assert W_enc.shape[1] == d_model, f"W_enc {W_enc.shape} incompatible with d={d_model}"
        W_enc = W_enc.T
    if W_dec.shape[1] != d_model:
        assert W_dec.shape[0] == d_model, f"W_dec {W_dec.shape} incompatible with d={d_model}"
        W_dec = W_dec.T
    F = W_enc.shape[1]
    assert W_dec.shape == (F, d_model), (W_dec.shape, F, d_model)
    assert params["b_enc"].shape == (F,) and params["b_dec"].shape == (d_model,)
    assert params["threshold"].shape == (F,), params["threshold"].shape
    return {"W_enc": W_enc, "W_dec": W_dec, "b_enc": params["b_enc"],
            "b_dec": params["b_dec"], "threshold": params["threshold"],
            "n_features": F, "d_model": d_model, "dir": sae_dir}
