"""COMWRITE Gate-0 (spec §10.0) — WRITE-FIRST wiring + provenance, verified at the artifacts
before any freeze:

  ledger — the §3 reuse ledger re-grepped MECHANICALLY (import + getattr every reused symbol).
  roster — the frozen 24-FT roster read from the N5c artifact (gauge_common Gate-0 provenance
           note), SHAs re-pinned LIVE via HfApi, the GAUGE sha1 12/12 FT split recorded.
  items  — per-donor eval items streamed + content-hashed; item halves FRESH to COMWRITE
           (sha1 salted with 'comwrite|', §2 — NEVER inherit_core's unsalted split, which
           would reproduce INHERIT's seals on shared items).
  endow_r — the ENDOW-R artifact fetched; the Arm-C probe config/reference band recorded.
  pi0    — the π0 backbone decoder remaps + assert_18x7 (the A5 accessor is wired).
  gpu    — PG1 loads EAGER at the invcore-FUNC-pinned sha; blank-image wikitext ppl REPRODUCES
           the artifact value (|Δ| ≤ 0.5, the inherit_gate0 tol convention); per-donor dynamic
           range: FT beats base on its own task with non-overlapping 95% CIs (else the §10.3
           backup-roster pivot, then UNTESTABLE).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.comwrite_gate0 --type p5.4xlarge,g6e.8xlarge \
      --no-weights --s3-prefix results/comwrite_gate0 --budget-min 180 \
      --args "--stages ledger,roster,items,endow_r,pi0,gpu"
"""
import argparse
import hashlib
import importlib
import json
import pathlib

PPL_TOL = 0.5                       # inherit_gate0 convention (spec §2 ppl instrument)
GAUGE_CS_S3 = "s3://pg-lineage-769063704696/results/gauge_full/out/cs/gauge_cs.json"

# donor slug → (eval task name in projector_ablation.build_tasks, FT repo slug). Frozen §5.1
# primary roster; backups coco35l / refcoco-seg are §10.3 pivots, not defaults.
DONORS = {"vqav2": "vqav2", "textvqa": "textvqa", "cococap": "coco_captions"}

# The §3 instrument-identity ledger, mechanized: (module, symbol) — import + getattr or die.
LEDGER = (
    ("infra.aws_pg.jobs.invariant_core", "_consensus_gram"),
    ("infra.aws_pg.jobs.invariant_core", "_bottom_topk_eigs"),
    ("infra.aws_pg.jobs.invariant_core", "_haar_gram"),
    ("infra.aws_pg.jobs.invariant_core", "_null_band"),
    ("infra.pg.gauge_cs", "consensus_bases"),
    ("infra.pg.gauge_cs", "coeffs"),
    ("infra.pg.gauge_tree", "ccc_with_nulls"),
    ("infra.pg.gauge_tree", "gram_to_distance"),
    ("infra.pg.gauge_common", "sha1_split"),
    ("infra.pg.endow_transplant", "projector_from_basis"),
    ("infra.pg.endow_transplant", "build_cell"),
    ("infra.pg.inherit_pg1", "load_pg1"),
    ("infra.pg.inherit_pg1", "eval_pg1"),
    ("infra.pg.inherit_pg1", "ppl_gate"),
    ("infra.pg.projector_ablation", "build_tasks"),
    ("infra.aws_pg.jobs.xgen_install_fperp", "build_before_after_pg1"),
    ("infra.pg.reuse_estimators", "reuse_stats"),
    ("infra.pg.reuse_estimators", "haar_null"),
    ("infra.pg.pi0_port", "remap_backbone_lm"),
    ("infra.pg.pi0_port", "decoder_matrices"),
    ("infra.pg.pi0_port", "assert_18x7"),
    ("infra.pg.relaw", "project_grad"),
    ("infra.pg.comwrite_core", "u_wr"),                       # NEW, now present
    ("infra.pg.comwrite_core", "project_grad_residual"),
    ("infra.pg.comwrite_surgery", "build_decoder"),
    ("infra.pg.comwrite_adjudicate", "adjudicate"),
)


def comwrite_split(content_hashes, name):
    """Item halves FRESH to COMWRITE: held (sealed) iff sha1('comwrite|'+hash) is odd. Same
    enforce_split fail-loud routing as inherit_core.sha1_split, different (salted) coin."""
    import numpy as np
    from infra.pg.diff_metrics import enforce_split
    held = np.array([int(hashlib.sha1(f"comwrite|{h}".encode()).hexdigest(), 16) % 2 == 1
                     for h in content_hashes], dtype=bool)
    return enforce_split(held, name=name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default="ledger,roster,items,endow_r,pi0,gpu")
    ap.add_argument("--n-items", type=int, default=2048, help="items streamed per donor")
    ap.add_argument("--range-n", type=int, default=200, help="discovery slice for the range check")
    ap.add_argument("--n-ppl-seq", type=int, default=120)
    ap.add_argument("--endow-r-s3", default="", help="s3 uri of the ENDOW-R metrics json (Arm C)")
    ap.add_argument("--gauge-s3", default=GAUGE_CS_S3,
                    help="the GAUGE artifact holding roster+shas+12/12 split")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    stages = set(a.stages.split(","))
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"object": "comwrite_gate0_v1", "ppl_tol": PPL_TOL, "stages": sorted(stages),
           "seed": a.seed}

    def _w():
        (out / "gate0.json").write_text(json.dumps(rep, indent=2, default=str))

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    from infra.aws_pg.jobs import _inherit_shared as SH

    # --- ledger: the §3 re-grep, mechanized. A missing symbol fails HERE, not mid-run. ---
    if "ledger" in stages:
        rep["ledger"] = {}
        for mod, sym in LEDGER:
            m = importlib.import_module(mod)
            assert hasattr(m, sym), f"§3 ledger BROKEN: {mod}.{sym} missing"
            rep["ledger"][f"{mod}.{sym}"] = "ok"
        _w()
        print(f"[gate0] ledger ok ({len(LEDGER)} symbols)", flush=True)

    # --- roster: re-pinned from the GAUGE artifact (§10.0) + LIVE HfApi cross-pin;
    # any drift fails loud (§8 S3/manifest-drift threat). ---
    if "roster" in stages:
        from huggingface_hub import HfApi
        from infra.pg.gauge_common import sha1_split
        from infra.aws_pg.jobs.invariant_core import PG1_REPO
        gauge = json.loads(SH.fetch(a.gauge_s3, out / "gauge_cs.json").read_text())
        repos = sorted(gauge["cloud_repos"])
        assert len(repos) == 24 and len(set(repos)) == 24, f"roster must be 24, got {len(repos)}"
        assert all(r.startswith("google/paligemma-3b-ft-") and r.endswith("-224") for r in repos)
        api = HfApi()
        shas = {r: api.model_info(r).sha for r in repos}
        drift = {r: (shas[r], gauge["cloud_shas"][r]) for r in repos
                 if shas[r] != gauge["cloud_shas"][r]}
        assert not drift, f"FT sha drift vs the GAUGE artifact: {drift}"
        base_sha = api.model_info(PG1_REPO).sha
        assert gauge["base_repo"] == PG1_REPO and base_sha == gauge["base_sha"], \
            f"base drift: live {base_sha[:8]} vs GAUGE {gauge['base_sha'][:8]}"
        disc, seal = sha1_split(repos)
        assert disc == sorted(gauge["discovery"]) and seal == sorted(gauge["confirmation"]), \
            "sha1 12/12 split disagrees with the GAUGE artifact"
        rep["roster"] = {"repos": repos, "shas": shas, "base_repo": PG1_REPO,
                         "base_sha": base_sha, "discovery": disc, "sealed": seal,
                         "gauge_s3": a.gauge_s3}
        for slug in DONORS:
            assert f"google/paligemma-3b-ft-{slug}-224" in repos, f"donor {slug} not in roster"
        _w()
        print(f"[gate0] roster ok (24 FTs pinned vs GAUGE; donors {sorted(DONORS)})", flush=True)

    # --- items: per-donor streamed items + FRESH salted halves. ---
    if "items" in stages:
        from infra.pg.projector_ablation import _content_hash, load_eval_items, build_tasks
        tasks = {t.name: t for t in build_tasks()}
        rep["items"] = {}
        for slug, tname in DONORS.items():
            task = tasks[tname]
            rows = load_eval_items(task, a.n_items)
            hashes = [_content_hash(task, r) for r in rows]
            sp = comwrite_split(hashes, name=f"comwrite_{slug}")
            digest = hashlib.sha256("".join(sorted(hashes)).encode()).hexdigest()
            rep["items"][slug] = {"task": tname, "n": len(rows),
                                  "n_disc": int(sp["n_disc"]),
                                  "n_sealed": int(sp["n_conf"]),
                                  "hashes_sha256": digest}
            (out / f"items_{slug}.json").write_text(json.dumps(
                {"hashes": hashes, "sealed_idx": [int(i) for i in sp["conf_idx"]]}))
        _w()
        print(f"[gate0] items ok {[(s, v['n']) for s, v in rep['items'].items()]}", flush=True)

    # --- endow_r: the Arm-C probe config + reference band, read from the artifact. ---
    if "endow_r" in stages:
        assert a.endow_r_s3, \
            "endow_r stage requested without --endow-r-s3 — a silent skip would stamp " \
            "Gate-0 green with the Arm-C reference wiring unverified (drop the stage " \
            "explicitly via --stages if Arm C is deferred)"
        dst = SH.fetch(a.endow_r_s3, out / "endow_r_metrics.json")
        er = json.loads(dst.read_text())
        assert er, "empty ENDOW-R artifact"
        rep["endow_r"] = {"s3": a.endow_r_s3,
                          "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
                          "top_keys": sorted(er)[:40]}
        _w()
        print(f"[gate0] endow_r artifact ok ({len(er)} top-level keys)", flush=True)

    # --- pi0: the A5 accessor is wired (18×7 asserted). ---
    if "pi0" in stages:
        from infra.aws_pg.jobs.endow_r_r0 import _extract_decoder
        from infra.pg.pi0_port import decoder_matrices, assert_18x7
        dec, sha = _extract_decoder("pi0")
        assert_18x7(decoder_matrices(dec))
        rep["pi0"] = {"sha": sha, "n_keys": len(dec), "assert_18x7": "ok"}
        del dec
        _w()
        print("[gate0] pi0 18x7 ok", flush=True)

    # --- gpu: ppl instrument + per-donor dynamic range. ---
    if "gpu" in stages:
        import numpy as np
        import torch
        from huggingface_hub import HfApi
        from infra.pg import inherit_pg1 as IP
        from infra.pg.projector_ablation import _content_hash, load_eval_items, build_tasks
        from transformers import PaliGemmaForConditionalGeneration

        SH.install_pg()             # pycocoevalcap etc. — the CIDEr path needs it
        func = json.loads(SH.fetch(SH.INVCORE_FUNC_JSON,
                                   "/tmp/comwrite/invcore_func.json").read_text())
        pg1_sha = func["base_sha"]
        model, proc = IP.load_pg1(sha=pg1_sha)
        ppl_ref = float(func["capability_baseline"]["ppl"])
        pp = IP.ppl_gate(model, proc, "cuda", n_seq=a.n_ppl_seq)
        rep["ppl_gate"] = {"ppl": pp["ppl"], "ref": ppl_ref, "delta": pp["ppl"] - ppl_ref}
        assert abs(pp["ppl"] - ppl_ref) <= PPL_TOL, \
            f"ppl gate RED: {pp['ppl']:.3f} vs artifact {ppl_ref:.3f} (sdpa trap / wiring)"
        _w()

        tasks = {t.name: t for t in build_tasks()}
        rep["range"] = {}
        for slug, tname in DONORS.items():
            task = tasks[tname]
            rows = load_eval_items(task, a.range_n * 2)
            hashes = [_content_hash(task, r) for r in rows]
            sp = comwrite_split(hashes, name=f"comwrite_{slug}_range")
            disc_rows = [rows[i] for i in sp["disc_idx"]][: a.range_n]
            base_res = IP.eval_pg1(model, proc, task, disc_rows, "cuda")
            ft_repo = f"google/paligemma-3b-ft-{slug}-224"
            ft_sha = (rep.get("roster", {}).get("shas", {}).get(ft_repo)
                      or HfApi().model_info(ft_repo).sha)     # pinned, and recorded either way
            ft = PaliGemmaForConditionalGeneration.from_pretrained(
                ft_repo, revision=ft_sha, torch_dtype=torch.bfloat16, device_map="cuda",
                attn_implementation="eager").eval()
            ft_res = IP.eval_pg1(ft, proc, task, disc_rows, "cuda")
            del ft
            torch.cuda.empty_cache()
            ok = bool(ft_res["score"] > base_res["score"]
                      and ft_res["ci95"][0] > base_res["ci95"][1])
            rep["range"][slug] = {"base": base_res["score"], "base_ci": base_res["ci95"],
                                  "ft": ft_res["score"], "ft_ci": ft_res["ci95"],
                                  "ft_sha": ft_sha, "range_ok": ok}
            print(f"[gate0] range {slug}: base={base_res['score']:.4f} "
                  f"ft={ft_res['score']:.4f} ok={ok}", flush=True)
            _w()
        assert all(v["range_ok"] for v in rep["range"].values()), \
            "a donor failed the dynamic-range gate — §10.3 backup-roster pivot"

    rep["green"] = True
    _w()
    print("GATE0_OK", json.dumps({k: v for k, v in rep.items()
                                  if k in ("ppl_gate", "range", "roster") and v},
                                 default=str)[:2000])


if __name__ == "__main__":
    main()
