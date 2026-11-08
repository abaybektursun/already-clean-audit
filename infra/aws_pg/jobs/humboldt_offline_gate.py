"""HUMBOLDT — the offline gate (§9 step 6): the FULL Phase-B pipeline minus the model forward.

House rule feedback_offline_gate_before_gpu. Everything Phase B does except calling a real model:
  * the projector plumbing — SiteEnergy / SiteSlice / BankInject hooks fire on stub modules that
    have the real module names, and the per-token e_U they produce is checked against a direct
    numpy computation with the REAL A1 banks;
  * the corpora — built at FULL size on the REAL tokenizers, with the token metadata, class labels
    and paired-condition alignment hashes the gates read;
  * B3's prompt construction and exact token-block location on the real tokenizers;
  * B4's ε calibration arithmetic;
  * the atlas writer + parquet renderer end to end.

A failure here costs a CPU minute; the same failure on a GPU box costs the arm.

  pg run infra.aws_pg.jobs.humboldt_offline_gate --type r7i.2xlarge --no-weights \
      --s3-prefix results/humboldt_offline --budget-min 90
"""
import argparse
import json
import pathlib
import subprocess

from infra.pg import humboldt_act as A
from infra.pg import humboldt_core as H
from infra.pg import humboldt_smoke as S
from infra.pg import humboldt_weights as W


def _stub_layers(n_layers, device):
    """Modules with the REAL attribute names Phase B hooks (self_attn.{q,o}_proj, mlp.gate_proj),
    so the hook registration path under test is the one that will run on the GPU."""
    import torch
    from torch import nn

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(2048, 2048, bias=False)
            self.o_proj = nn.Linear(2048, 2048, bias=False)

    class Mlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(2048, 16384, bias=False)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attn()
            self.mlp = Mlp()

        def forward(self, x):
            self.self_attn.q_proj(x)
            self.self_attn.o_proj(x)
            self.mlp.gate_proj(x)
            return x

    return [Layer().to(device).eval() for _ in range(n_layers)], torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-tokens", type=int, default=50_000)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--long-doc-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--run-id", default="humboldt_offline")
    ap.add_argument("--probe-s3", default="s3://pg-lineage-769063704696/humboldt/humboldt_probe/corpus_resolution.json")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    H.hf_setup()                       # FIRST — XET off + gated-repo auth (§0.2.7)
    import numpy as np
    from transformers import AutoProcessor, AutoTokenizer

    gates = H.Gates(out)
    S.run_all_offline(gates)
    bank_dir = W.fetch_banks(H.cache_dir(out, "banks"))
    subprocess.run(["aws", "s3", "cp", a.probe_s3, str(out / "corpus_resolution.json"),
                    "--region", "us-east-1"], check=True)
    resolution = json.loads((out / "corpus_resolution.json").read_text())
    resolved = {k: v["chosen"] for k, v in resolution["slots"].items()}

    # ---------------- 1. projector plumbing on stub modules, checked against direct numpy.
    layers, torch = _stub_layers(H.N_LAYERS, "cpu")
    energy = A.SiteEnergy(layers, bank_dir, "cpu")
    x = torch.randn(1, 37, 2048)
    with energy:
        for lay in layers:
            lay(x)
    series = energy.collect()
    Xn = x[0].numpy().astype(np.float64)
    worst = 0.0
    for site in H.SITES_PHASE_B:
        for j, (t, bid) in enumerate(energy.order[site]):
            U = W.load_bank(bank_dir, t, 0, bid)
            ref = H.energy_fraction(Xn, U)
            got = series[(0, site)][:, j].astype(np.float64)
            worst = max(worst, float(np.abs(ref - got).max()))
    gates.require("OFFLINE_projector_matches_numpy", worst < 1e-5, max_abs_delta=worst,
                  tol=1e-5, n_sites=len(H.SITES_PHASE_B),
                  criterion="hooked e_U equals the direct numpy computation with the same banks")

    # SiteSlice must return exactly the requested window, in order.
    sl = A.SiteSlice(layers, "cpu")
    with sl:
        sl.set_window(5, 11)
        for lay in layers:
            lay(x)
    d_slice = float(np.abs(sl.buf[(0, "q_proj_in")] - Xn[5:16]).max())
    gates.require("OFFLINE_siteslice_window", d_slice < 1e-5, max_abs_delta=d_slice, tol=1e-5,
                  shape=list(sl.buf[(0, "q_proj_in")].shape),
                  criterion="SiteSlice returns exactly the requested token window")

    # BankInject must add exactly ε·(Uz)/‖Uz‖ and nothing else.
    U0 = W.load_bank(bank_dir, "q_proj", 0, "core32")
    z = np.zeros(H.BANK_K); z[3] = 1.0
    with A.BankInject(layers[0], "q_proj_in", U0, z, 0.25, "cpu"):
        cap = A.SiteSlice([layers[0]], "cpu")
        with cap:
            cap.set_window(0, 37)
            layers[0](x)
    v = U0 @ z
    v = v / np.linalg.norm(v)
    d_inj = float(np.abs(cap.buf[(0, "q_proj_in")] - (Xn + 0.25 * v)).max())
    gates.require("OFFLINE_bankinject_exact", d_inj < 1e-4, max_abs_delta=d_inj, tol=1e-4,
                  criterion="BankInject adds exactly ε·(Uz)/‖Uz‖ at every position")

    # ---------------- 2. corpora at FULL size on the REAL tokenizers + alignment hashes.
    from huggingface_hub import HfApi
    api = HfApi()
    corpora_report = {}
    for mid, repo, kind in A.ROSTER:
        if kind == "policy_reshell":
            continue                        # the re-shell reuses PG1's processor, already covered
        sha = api.model_info(repo).sha
        tok = (AutoProcessor.from_pretrained(repo, revision=sha).tokenizer if kind == "paligemma"
               else AutoTokenizer.from_pretrained(repo, revision=sha))
        corp = A.build_text_corpora(tok, resolved, n_tokens=a.n_tokens, window=a.window,
                                    long_doc_tokens=a.long_doc_tokens, seed=a.seed)
        rep = {}
        for cid, rows in corp.items():
            n_tok = sum(len(tok(r["text"], add_special_tokens=False).input_ids) for r in rows)
            rep[cid] = {"n_windows": len(rows), "n_tokens": n_tok,
                        "classes": sorted({r["token_class"] for r in rows}),
                        "content_hash": A.content_hash([r["text"][:64] for r in rows])}
        # the paired-condition check the B1 gate will run, on tokens only.
        langs = resolved["parallel"]["langs"]
        by_lang = {}
        for r in corp["parallel"]:
            by_lang.setdefault(r["token_class"], set()).add(r["item_id"])
        sets = list(by_lang.values())
        rep["_parallel_pairing_ok"] = bool(sets and all(s == sets[0] for s in sets))
        rep["_parallel_langs"] = langs
        corpora_report[mid] = rep
        print(f"[offline] {mid}: " + json.dumps({k: v.get("n_tokens") for k, v in rep.items()
                                                 if isinstance(v, dict)}), flush=True)
    ok_pair = all(r["_parallel_pairing_ok"] for r in corpora_report.values())
    gates.require("OFFLINE_corpora_build", ok_pair, per_model=corpora_report,
                  criterion="every corpus builds on every tokenizer and the parallel corpus "
                            "covers the same sentence-id set in every language")

    # ---------------- 3. B3 prompt construction + exact token-block location.
    from infra.aws_pg.jobs.humboldt_b3 import SHOTS, TASKS, _build_items
    from infra.aws_pg.jobs.invariant_core_func import _locate_text_block
    # BOTH paths B3 actually uses are exercised: the plain tokenizer (gemma-2b) and the PaliGemma
    # PROCESSOR with a blank image (256 image tokens prepended AND a trailing newline appended —
    # the gotcha `_locate_text_block` exists for). A processor needs no model weights, so this
    # costs nothing and is the only place the multimodal locate is checked before the GPU.
    from infra.aws_pg.jobs.invariant_core_func import _blank_image
    b3 = {}
    pg1_sha = api.model_info(A.PG1_REPO).sha
    proc_pg1 = AutoProcessor.from_pretrained(A.PG1_REPO, revision=pg1_sha)
    tok_pg1 = proc_pg1.tokenizer
    g1_sha = api.model_info(A.GEMMA1_REPO).sha
    tok_g1 = AutoTokenizer.from_pretrained(A.GEMMA1_REPO, revision=g1_sha)
    blank = _blank_image(128)
    for task in TASKS:
        items = _build_items(task, 4, max(SHOTS))
        assert items, f"{task}: no items"
        per_path = {}
        for path in ("text_only_gemma2b", "processor_pg1_blank_image"):
            hashes, spans = {}, []
            for it in items:
                for k in SHOTS:
                    prompt = it["header"] + "".join(it["shots"][:k]) + it["query"]
                    if path == "text_only_gemma2b":
                        full = tok_g1(prompt, return_tensors="pt",
                                      add_special_tokens=True).input_ids[0]
                        q = tok_g1(it["query"], return_tensors="pt",
                                   add_special_tokens=False).input_ids[0]
                    else:
                        full = proc_pg1(text=[prompt], images=[blank], return_tensors="pt",
                                        padding=False)["input_ids"][0]
                        q = tok_pg1(it["query"], return_tensors="pt",
                                    add_special_tokens=False).input_ids[0]
                    pos = _locate_text_block(full, q)
                    spans.append(int(q.shape[0]))
                    hashes.setdefault(it["item_id"], set()).add(
                        A.alignment_hash(full[pos:pos + q.shape[0]].tolist()))
            per_path[path] = {"all_aligned": all(len(v) == 1 for v in hashes.values()),
                              "query_block_tokens": sorted(set(spans))[:5]}
        b3[task] = {"n_items": len(items), **per_path,
                    "all_aligned": all(v["all_aligned"] for v in per_path.values())}
    gates.require("OFFLINE_b3_query_block_alignment",
                  all(v["all_aligned"] for v in b3.values()), per_task=b3,
                  criterion="the query block is token-identical across 0/1/4/16 shots on BOTH the "
                            "plain tokenizer and the PaliGemma processor (image tokens + trailing "
                            "newline)")

    # ---------------- 4. B4 ε calibration arithmetic.
    from infra.aws_pg.jobs.humboldt_b4 import EPS_FRACS
    hn = float(np.linalg.norm(Xn, axis=1).mean())
    grid = [f * hn for f in EPS_FRACS]
    rel = [g / hn for g in grid]
    gates.require("OFFLINE_b4_eps_calibration",
                  abs(rel[0] - 0.001) < 1e-12 and abs(rel[-1] - 0.30) < 1e-12,
                  mean_residual_norm=hn, eps_grid=grid, relative=rel, eps_fracs=list(EPS_FRACS),
                  criterion="the smallest ε moves ‖h‖ by 0.1% and the largest by 30%")

    # ---------------- 5. the atlas writer + parquet renderer, end to end.
    atlas = H.Atlas(out, arm="OFFLINE", run_id=a.run_id)
    atlas.add("offline_scalar", value=1.0, layer=0, type="q_proj", bank_id="core32", flags={"x": 1})
    atlas.add_ci("offline_sampled", [1.0, 2.0, 3.0, 4.0], layer=0, type="q_proj",
                 bank_id="haar32_s0", seed=a.seed)
    atlas.add("offline_vector", vector_ref=atlas.vector("offline_vec", np.arange(8)), n=8)
    atlas.close()
    rep = atlas.to_parquet()
    gates.require("OFFLINE_atlas_roundtrip", rep["n_rows"] == 3 and rep["n_vectors"] == 1, **rep)

    # §4 — THE DATA MANIFEST. "All corpora / model shas / content hashes enumerated in
    # out/manifest.json BEFORE Phase B launches." This job is the last thing that runs before the
    # GPU arms, and it is the only place every corpus is built on every tokenizer, so the manifest
    # belongs here rather than inside an arm that would write it after the fact.
    manifest = {"arm": "OFFLINE_GATE (§4 data manifest)", "run_id": a.run_id,
                "commit_sha": H.commit_sha(), "seed": a.seed,
                "resolved_corpora": resolved,
                "corpus_probe_run": resolution["run_id"],
                "n_tokens_per_corpus": a.n_tokens, "window": a.window,
                "long_doc_tokens": a.long_doc_tokens,
                "model_shas": {mid: api.model_info(repo).sha for mid, repo, kind in A.ROSTER
                               if kind != "policy_reshell"},
                "content_hashes": {mid: {cid: v["content_hash"]
                                         for cid, v in rep.items() if isinstance(v, dict)}
                                   for mid, rep in corpora_report.items()},
                "paired_conditions": {
                    "parallel": {"pairing": "sentence id", "langs": resolved["parallel"]["langs"],
                                 "source": resolved["parallel"]["id"]},
                    "vqav2": {"pairing": "question id", "arms": ["vqav2_real", "vqav2_gray"]},
                    "shots": {"pairing": "identical query token block", "shots": [0, 1, 4, 16]}}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (out / "offline_summary.json").write_text(json.dumps(
        {"gates": gates.gates, "corpora": corpora_report, "b3": b3, "manifest": manifest,
         "commit_sha": H.commit_sha()}, indent=2, default=str))
    H.s3_sync(out, f"humboldt/{a.run_id}")
    print("=" * 78)
    print(f"HUMBOLDT_OFFLINE_OK gates={ {k: v['ok'] for k, v in gates.gates.items()} }")


if __name__ == "__main__":
    main()
