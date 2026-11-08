"""MERGE-0 gate — pi-trajectory study (start-experiments spec §P0).

A single zero-weights CPU smoke that satisfies the three MERGE-0 gates at once:
  (1) import gate      — the vendored infra.pg.gain_routing loads AND its fp64 decompose
                         runs on the pg-env DLAMI (torch 2.12+cu130 CPU ops, no GPU).
  (2) model_info gate  — HfApi().model_info resolves the three pinned pi-lineage repos
                         (PG1 / L0_g1 / PI0); gated metadata needs the SSM HF token.
  (3) r-family infra   — proves run_job.sh + the pg-env-v1 DLAMI boot clean on an
                         r-family (Standard-quota) box. NO torch.cuda assert (the
                         critical NOT-to-copy block from smoke.py).

Fail-loud, no try/except. Runs with `--no-weights` (nothing here needs the S3 snapshot).
"""
import argparse
import json
import pathlib

from infra.pg.diff_harvest import hf_online_setup
from infra.pg.entities import ENTITIES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # (2) gated metadata needs the token in env BEFORE huggingface_hub touches the hub.
    hf_online_setup()
    from huggingface_hub import HfApi  # noqa: E402  (after token is set)

    api = HfApi()
    resolved = {}
    for sym in ("PG1", "L0_g1", "PI0"):
        ent = ENTITIES[sym]
        info = api.model_info(ent.repo_id, revision=ent.revision or "main")
        resolved[sym] = {"repo_id": ent.repo_id, "revision": ent.revision or "main", "sha": info.sha}
        print(f"[MERGE-0] {sym:6s} {ent.repo_id}@{info.sha}")

    # (1) the vendored decomposer must import AND run fp64 on this box (a rectangular
    #     matrix exercises the out-of-span path; recon identity guards the fp64 mandate).
    import torch
    from infra.pg import gain_routing

    g = torch.Generator().manual_seed(1337)
    wb = torch.randn(64, 48, generator=g, dtype=torch.float64)
    wa = wb + 0.01 * torch.randn(64, 48, generator=g, dtype=torch.float64)
    dec = gain_routing.decompose(wb, wa, top_k=8, check_recon=True)
    assert dec["recon_residual"] is not None and dec["recon_residual"] < 1e-10, \
        f"fp64 recon identity FAIL on the box: {dec['recon_residual']}"

    res = {
        "ok": True,
        "torch": torch.__version__,
        "numpy_cuda_free": True,
        "gain_routing_source_w12_sha": "12399e2",
        "resolved": resolved,
        "decompose_smoke": {
            "shape": dec["shape"], "recon_residual": dec["recon_residual"],
            "E_gain": dec["E_gain"], "f_perp": dec["f_perp"],
        },
    }
    (out / "merge0_smoke.json").write_text(json.dumps(res, indent=2))
    print("MERGE0_OK", json.dumps(res["resolved"]))


if __name__ == "__main__":
    main()
