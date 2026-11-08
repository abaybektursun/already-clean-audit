"""HUMBOLDT A2 — RoPE-coupling decomposition (spec §3/A2). CPU.

The banks live in the residual stream (input space); RoPE acts AFTER projection, per head-dim pair.
Per bank U and rotary band b:  coupling(b, U) = ‖P_b W_type U‖²_F / ‖W_type U‖²_F, summed over heads.
Recorded for q_proj and k_proj, all 18 layers, every bank AND every twin.

The band layout is READ off the config (§3/A2 "read the config, do not assume") by
humboldt_core.rope_bands, which fails loud on any config that is not HF Gemma's rotate_half.

Gradient sub-arm (§3/A2, ≤1h reach test): pre-specified pivot if ERLANGEN's banked gradient
artifacts are unreachable. The reach test runs here and its evidence is recorded either way.

  pg run infra.aws_pg.jobs.humboldt_a2 --type r7i.2xlarge --no-weights \
      --s3-prefix results/humboldt_a2 --budget-min 120
"""
import argparse
import json
import pathlib
import subprocess

from infra.pg import humboldt_core as H
from infra.pg import humboldt_smoke as S
from infra.pg import humboldt_weights as W

ROPE_TYPES = ("q_proj", "k_proj")            # the only types RoPE acts on
ERLANGEN_PREFIXES = ("results/erlangen_basis2/out/", "results/erlangen_coreprov2/out/",
                     "results/erlangen_g/out/", "results/erlangen_d3/out/",
                     "results/erlangen_x2/out/")
BUCKET = "pg-lineage-769063704696"


def _erlangen_reach(out) -> dict:
    """≤1h reach test for ERLANGEN's banked GRADIENT-direction artifacts (§3/A2, §13 row 6).
    Lists the ERLANGEN result prefixes and classifies every array artifact found. Records what IS
    there so the pivot decision is evidence-backed, not asserted."""
    found = []
    for p in ERLANGEN_PREFIXES:
        # `aws s3 ls` exits 1 on an empty/absent prefix. An absent prefix is EVIDENCE for the reach
        # test, not an error — recording it is the point.
        r = subprocess.run(["aws", "s3", "ls", f"s3://{BUCKET}/{p}", "--region", "us-east-1"],
                           check=False, capture_output=True, text=True)
        if r.returncode != 0:
            found.append({"prefix": p, "file": None, "size": 0, "listing_rc": r.returncode})
            continue
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[3].endswith((".npz", ".npy")):
                found.append({"prefix": p, "file": parts[3], "size": int(parts[2])})
    # A gradient-direction bank would be named/derived from gradients. Everything ERLANGEN banked is
    # a ΔW-consensus eigenbasis (erlangen_{core,top}_basis.npz, "bottom-k eigvecs of Σ ΔWᵀΔW").
    grad = [f for f in found if f["file"] and "grad" in f["file"].lower()]
    ev = {"arrays_found": found, "gradient_arrays_found": grad,
          "reachable": bool(grad),
          "note": ("ERLANGEN banked only ΔW-consensus eigenbases (erlangen_core_basis.npz / "
                   "erlangen_top_basis.npz, estimand 'bottom-k eigvecs of Σ ΔWᵀΔW/‖ΔW‖²_F'). No "
                   "gradient-direction bank is on S3. Pre-specified pivot: skip the sub-arm.")}
    (pathlib.Path(out) / "a2_erlangen_reach.json").write_text(json.dumps(ev, indent=2))
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--run-id", default="humboldt_a2")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    layers = ([int(x) for x in a.layers.split(",") if x.strip()] if a.layers.strip()
              else list(range(H.N_LAYERS)))
    if a.smoke:
        layers = layers[:2]

    H.hf_setup()                       # FIRST — XET off + gated-repo auth (§0.2.7)
    import numpy as np
    from huggingface_hub import HfApi
    from transformers import AutoConfig

    gates = H.Gates(out)
    S.run_all_offline(gates)

    api = HfApi()
    base_sha = api.model_info(W.PG1_REPO).sha
    cfg = AutoConfig.from_pretrained(W.PG1_REPO, revision=base_sha)
    text_cfg = cfg.text_config
    bands = H.rope_bands(text_cfg)
    print(f"[A2] rope layout: {json.dumps({k: v for k, v in bands.items() if k != 'inv_freq' and k != 'pairs'})}",
          flush=True)

    bank_dir = W.fetch_banks(H.cache_dir(out, "banks"))
    w_base = W._load_decoder(W.PG1_REPO, base_sha)

    manifest = {"arm": "A2", "run_id": a.run_id, "commit_sha": H.commit_sha(),
                "base_repo": W.PG1_REPO, "base_sha": base_sha, "layers": layers,
                "types": list(ROPE_TYPES), "rope": bands, "seed": a.seed,
                "n_heads_by_type": {"q_proj": bands["num_attention_heads"],
                                    "k_proj": bands["num_key_value_heads"]}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    atlas = H.Atlas(out, arm="A2", run_id=a.run_id)      # bound BEFORE any loop (§0.2.4)
    reach = _erlangen_reach(out)
    # `ok` is the MEASURED reachability, not a literal True. Passing True here made the gate
    # incapable of failing and forced a reader of gates.json to consult `executed_pivot` to learn
    # what actually happened — the same unfalsifiable-gate defect as G1's original form.
    # It stays non-blocking (`record`): an unreachable sub-arm is a legitimate pivot, not an error.
    gates.record("A2_erlangen_gradient_reach", reach["reachable"],
                 executed_pivot=not reach["reachable"], **reach)

    hd = bands["head_dim"]
    for t in ROPE_TYPES:
        n_heads = bands["num_attention_heads"] if t == "q_proj" else bands["num_key_value_heads"]
        for L in layers:
            Wm = w_base[W.matrix_key(t, L)].numpy().astype(np.float64)
            assert Wm.shape[0] == n_heads * hd, \
                f"{t} L{L} rows {Wm.shape[0]} != {n_heads}×{hd} — head layout misread"
            fl = {**H.type_flags(t), "n_heads": n_heads, "head_dim": hd,
                  "n_bands": bands["n_bands"], "rope_convention": bands["convention"]}
            for bid in H.BANK_IDS:
                U = W.load_bank(bank_dir, t, L, bid)
                c = H.rope_coupling(Wm, U, head_dim=hd, n_out_heads=n_heads)
                assert abs(float(c.sum()) - 1.0) < 1e-9, f"coupling sum {c.sum()} at {t} L{L} {bid}"
                atlas.add("rope_band_coupling",
                          vector_ref=atlas.vector(f"A2__coupling__{t}__L{L}__{bid}", c),
                          layer=L, type=t, bank_id=bid, model_id=W.PG1_REPO, model_sha=base_sha,
                          n=int(c.size), flags=fl)
                atlas.add("rope_coupling_participation_ratio", value=H.participation_ratio(c),
                          layer=L, type=t, bank_id=bid, model_id=W.PG1_REPO, model_sha=base_sha,
                          flags={**fl, "flat_reference": bands["n_bands"]})
                atlas.add("rope_coupling_argmax_band", value=float(np.argmax(c)), layer=L, type=t,
                          bank_id=bid, model_id=W.PG1_REPO, model_sha=base_sha,
                          flags={**fl, "inv_freq_at_argmax": bands["inv_freq"][int(np.argmax(c))]})
                atlas.add("rope_coupling_max", value=float(c.max()), layer=L, type=t, bank_id=bid,
                          model_id=W.PG1_REPO, model_sha=base_sha,
                          flags={**fl, "flat_expectation": 1.0 / bands["n_bands"]})
            del Wm
        print(f"[A2] {t}: {len(layers)} layers × {len(H.BANK_IDS)} banks ({atlas.n_rows} rows)",
              flush=True)
        H.s3_sync(out, f"humboldt/{a.run_id}")

    atlas.close()
    rep = atlas.to_parquet()
    (out / "a2_summary.json").write_text(json.dumps(
        {"manifest": manifest, "atlas": rep, "gates": gates.gates,
         "erlangen_reach": reach}, indent=2, default=str))
    H.s3_sync(out, f"humboldt/{a.run_id}")
    print("=" * 78)
    print(f"HUMBOLDT_A2_OK rows={rep['n_rows']} vectors={rep['n_vectors']} "
          f"gradient_subarm={'RECORDED' if reach['reachable'] else 'PIVOT-SKIPPED'}")


if __name__ == "__main__":
    main()
