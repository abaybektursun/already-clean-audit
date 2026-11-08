"""republic_p1 — Civic Alignment Profile v0 (project: republic; NOT part of the PaliGemma study).

One self-contained job. Per model:
  1. forward the contrastive civic battery, capture residual-stream activations
     (last non-pad token) at every layer;
  2. per value x layer: S_auroc  = held-out AUROC of a linear probe (5-fold CV over pairs);
  3. A_proj = signed projection of neutral-dilemma activations onto the diff-of-means
     direction, scaled so the affirm pole = +1 and the violate pole = -1;
  4. control_auroc = the value's full-data probe evaluated on non-civic control pairs
     (valence confound check; ~0.5 means the probe is value-specific);
     specificity_gap = S_auroc - control_auroc.

v0.1 de-valencing (added after first light showed civic probes transfer to controls at
~0.9 AUROC, i.e. they ride the generic good/bad-valence axis): per layer, the top
VALENCE_K SVD directions of affirm-violate diff vectors from HALF the control concepts
(VALENCE_FIT_CONCEPTS) are projected out of ALL activations before probing; the OTHER
half of the controls (held out from the fit) supplies control_auroc, so specificity is
measured without circularity. Primary metrics are computed on de-valenced activations;
the same metrics on raw activations are kept with a `_raw` suffix.

Tidy output (one row per number): model, revision, origin, battery_version, value,
layer, metric, estimate, ci_lo, ci_hi -> results.parquet, rewritten after every value
so the 30s S3 streamer keeps the dashboard live. dashboard.html regenerates from
results.parquet only.

Run via the pg launcher from the ntm-frozen-gemma repo root:
  infra/aws_pg/pg run infra.aws_pg.jobs.republic_p1 \
    --args "--models qwen35-9b --battery s3://pg-lineage-769063704696/republic/battery/v0.jsonl" \
    --no-weights --budget-min 90 --s3-prefix republic/runs/<name>
"""
import os
os.environ["HF_HUB_OFFLINE"] = "0"  # republic downloads pinned revisions; AMI baked env sets 1

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import subprocess

import numpy as np
import pandas as pd

BATTERY_S3_DEFAULT = "s3://pg-lineage-769063704696/republic/battery/v0.jsonl"
SEED = 0
VALENCE_K = 4
VALENCE_FIT_CONCEPTS = {"healthy_eating", "punctuality", "politeness", "cleanliness"}

# Verified against the live HF API 2026-07-03 (see republic/research/roster-verification).
# multimodal=True -> ForConditionalGeneration checkpoint: load AutoModelForImageTextToText,
# hidden layer count lives in text_config.
ROSTER = {
    "llama31-8b":  dict(hf_id="meta-llama/Llama-3.1-8B-Instruct", sha="0e9e39f249a16976918f6564b8830bc894c89659", origin="US", multimodal=False),
    "gemma4-e4b":  dict(hf_id="google/gemma-4-E4B-it",            sha="fee6332c1abaafb77f6f9624236c63aa2f1d0187", origin="US", multimodal=True),
    "phi4-mini":   dict(hf_id="microsoft/Phi-4-mini-instruct",    sha="cfbefacb99257ffa30c83adab238a50856ac3083", origin="US", multimodal=False),
    "gptoss-20b":  dict(hf_id="openai/gpt-oss-20b",               sha="6cee5e81ee83917806bbde320786a8fb61efebee", origin="US", multimodal=False, cpu_load=True),  # mxfp4 dequant leaves experts on cpu under device_map; load on cpu then .to(cuda)
    "olmo3-7b":    dict(hf_id="allenai/Olmo-3-7B-Instruct",       sha="6e5971d9eba42665f5bd5a0fcf047f299ce1dccc", origin="US", multimodal=False),
    "qwen35-9b":   dict(hf_id="Qwen/Qwen3.5-9B",                  sha="c202236235762e1c871ad0ccb60c8ee5ba337b9a", origin="CN", multimodal=True),
    "r1d-qwen3-8b": dict(hf_id="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", sha="6e8885a6ff5c1dc5201574c8fd700323f23c25fa", origin="CN", multimodal=False),
    "ministral-8b": dict(hf_id="mistralai/Ministral-8B-Instruct-2410", sha="2f494a194c5b980dfb9772cb92d26cbb671fce5a", origin="EU", multimodal=False),
    # Batch 2 — verified against the live HF API 2026-07-04 (research/roster-verification-2.json)
    "minicpm5-1b":  dict(hf_id="openbmb/MiniCPM5-1B",                sha="4e9de7a0778dc1c362e983e6858f0e77542cbdca", origin="CN", multimodal=False),
    "exaone4-1.2b": dict(hf_id="LGAI-EXAONE/EXAONE-4.0-1.2B",        sha="3abf2810673c7c0778df64a73c2d52eab32d91c4", origin="KR", multimodal=False),
    "smollm3-3b":   dict(hf_id="HuggingFaceTB/SmolLM3-3B",           sha="a07cc9a04f16550a088caea529712d1d335b0ac1", origin="EU", multimodal=False),
    "apertus-4b":   dict(hf_id="swiss-ai/Apertus-v1.1-4B-Instruct",  sha="0d004ae50a03fda815ff7a091a89ab0baec94f3d", origin="CH", multimodal=False),
    "yi1.5-6b":     dict(hf_id="01-ai/Yi-1.5-6B-Chat",               sha="771924d1c83d67527d665913415d7086f11ea9c0", origin="CN", multimodal=False),
    "falcon-h1r-7b": dict(hf_id="tiiuae/Falcon-H1R-7B",              sha="a6f74bf181389908efd6970878d7ee2b42f5d417", origin="UAE", multimodal=False),
    "qwen3-8b":     dict(hf_id="Qwen/Qwen3-8B",                      sha="b968826d9c46dd6066d109eabc6255188de91218", origin="CN", multimodal=False),
    "tlite-8b":     dict(hf_id="t-tech/T-lite-it-2.1",               sha="d125c970c553de58fcee3c937d5e4867d4a448d8", origin="RU", multimodal=False),  # Qwen3-8B fine-tune, not independent pretrain
    "llmjp4-8b":    dict(hf_id="llm-jp/llm-jp-4-8b-instruct",        sha="098f2b2cf33021eba19a6d3582aa3d071ccc0aff", origin="JP", multimodal=False),
    "granite4.1-8b": dict(hf_id="ibm-granite/granite-4.1-8b",        sha="1504002f650e656a0a3789d99574df12e3e94ed0", origin="US", multimodal=False),
    "glm4-9b":      dict(hf_id="zai-org/GLM-4-9B-0414",              sha="645b8482494e31b6b752272bf7f7f273ef0f3caf", origin="CN", multimodal=False),
}


def log(run_dir, msg):
    line = f"- {dt.datetime.now(dt.timezone.utc).strftime('%H:%M:%SZ')} {msg}"
    print(line, flush=True)
    with open(run_dir / "log.md", "a") as f:
        f.write(line + "\n")


# ---- battery -------------------------------------------------------------------------------
def fetch_battery(src, out_dir):
    local = out_dir / "battery.jsonl"
    if src.startswith("s3://"):
        subprocess.run(["aws", "s3", "cp", src, str(local), "--region", "us-east-1"], check=True)
    else:
        local = pathlib.Path(src)
    rows = [json.loads(l) for l in local.read_text().splitlines() if l.strip()]
    sha = hashlib.sha256(local.read_bytes()).hexdigest()[:12]
    return rows, sha


# ---- capture -------------------------------------------------------------------------------
def load_model(spec):
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=spec["sha"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cls_kwargs = dict(revision=spec["sha"], dtype=torch.bfloat16)
    if not spec.get("cpu_load"):
        cls_kwargs["device_map"] = "cuda"
    if spec["multimodal"]:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(spec["hf_id"], **cls_kwargs)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(spec["hf_id"], **cls_kwargs)
    if spec.get("cpu_load"):
        model = model.to("cuda")
    model.eval()
    return tok, model


def capture(tok, model, texts, batch=16):
    """-> float32 [n_texts, n_layers+1, d]: last non-pad token residual stream per layer
    (index 0 = embeddings)."""
    import torch
    feats = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                        output_hidden_states=True)
        last = enc["attention_mask"].sum(1) - 1
        ar = torch.arange(len(last), device=last.device)
        # [B, L+1, d]
        feats.append(torch.stack([h[ar, last] for h in out.hidden_states], 1).float().cpu())
    return torch.cat(feats).numpy()


# ---- probes / metrics ----------------------------------------------------------------------
def probe_value(acts_a, acts_v, acts_d, acts_ctrl, ctrl_labels):
    """All metrics for one value at one layer.
    acts_a/acts_v: [n_pairs, d] affirm/violate (row i of each = same pair).
    acts_d: [n_dilemmas, d]. acts_ctrl: [n_ctrl, d] with ctrl_labels (1=affirm)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n = len(acts_a)
    X = np.concatenate([acts_a, acts_v])
    y = np.array([1] * n + [0] * n)
    pair = np.concatenate([np.arange(n), np.arange(n)])  # keep both polarities of a pair together

    def clf():
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))

    aucs = []
    for tr_p, te_p in KFold(5, shuffle=True, random_state=SEED).split(np.arange(n)):
        tr, te = np.isin(pair, tr_p), np.isin(pair, te_p)
        m = clf().fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    aucs = np.array(aucs)
    s_est, s_half = aucs.mean(), 1.96 * aucs.std(ddof=1) / np.sqrt(len(aucs))

    m_full = clf().fit(X, y)

    # A (v0.2): bounded probe-based alignment, mean over dilemmas of 2*P(affirm)-1 in [-1,+1].
    # (The earlier diff-of-means span projection was ill-conditioned where class means nearly
    # coincide — ministral-8b read A in the +/-300 range.)
    proj = 2.0 * m_full.predict_proba(acts_d)[:, 1] - 1.0
    a_est, a_half = proj.mean(), 1.96 * proj.std(ddof=1) / np.sqrt(len(proj))

    # specificity: full-data probe on control pairs (valence confound reads ~S, specific reads ~0.5)
    c_auc = roc_auc_score(ctrl_labels, m_full.predict_proba(acts_ctrl)[:, 1])

    return {"S_auroc": (s_est, s_est - s_half, s_est + s_half),
            "A_proj": (a_est, a_est - a_half, a_est + a_half),
            "control_auroc": (c_auc, np.nan, np.nan),
            "specificity_gap": (s_est - c_auc, np.nan, np.nan)}


def normalize(acts):
    """Unit-normalize each activation vector (x sqrt(d) to keep logreg-friendly scale).
    v0.2: bounds A_proj — without this, layers where dilemma norms dwarf pair norms produced
    meaningless A in the +/-100 range (ministral-8b late layers)."""
    n = np.linalg.norm(acts, axis=-1, keepdims=True)
    return acts / np.maximum(n, 1e-6) * np.sqrt(acts.shape[-1])


def devalence(acts, fit_a, fit_v, k=VALENCE_K):
    """Project the top-k SVD directions of the fit-control affirm-violate diffs out of
    every activation, per layer. Returns a de-valenced copy."""
    out = acts.copy()
    for layer in range(acts.shape[1]):
        diffs = acts[fit_a, layer] - acts[fit_v, layer]
        _, _, vt = np.linalg.svd(diffs, full_matrices=False)
        v = vt[:k]  # [k, d] orthonormal valence basis
        out[:, layer] -= out[:, layer] @ v.T @ v
    return out


def value_rows(short, spec, battery_version, value, acts, idx, suffix=""):
    """Metric rows for one value across all layers. idx: dict of battery-row indices."""
    rows = []
    for layer in range(acts.shape[1]):
        a = acts[idx["affirm"], layer]
        v = acts[idx["violate"], layer]
        d = acts[idx["dilemma"], layer]
        c = acts[idx["ctrl"], layer]
        for metric, (est, lo, hi) in probe_value(a, v, d, c, idx["ctrl_labels"]).items():
            rows.append(dict(model=short, revision=spec["sha"], origin=spec["origin"],
                             battery_version=battery_version, value=value, layer=layer,
                             metric=metric + suffix, estimate=float(est),
                             ci_lo=float(lo), ci_hi=float(hi)))
    return rows


def analyze_model(battery, acts, short, spec, bver):
    """Yield (value, metric_rows) per civic value. Pure function of battery + raw activations —
    the GPU job streams it; research/recompute.py reruns it locally from saved acts."""
    acts = normalize(acts.astype(np.float32))

    def ctrl_sel(in_fit, pol=None):
        return np.array([i for i, r in enumerate(battery)
                         if r["domain"] == "control" and r["kind"] == "pair"
                         and (r["concept"] in VALENCE_FIT_CONCEPTS) == in_fit
                         and (pol is None or r["polarity"] == pol)])

    fit_a, fit_v = ctrl_sel(True, "affirm"), ctrl_sel(True, "violate")
    ctrl_idx = ctrl_sel(False)  # held-out controls: honest specificity, no circularity
    ctrl_labels = np.array([1 if battery[i]["polarity"] == "affirm" else 0 for i in ctrl_idx])
    acts_dv = devalence(acts, fit_a, fit_v)

    for value in sorted({r["concept"] for r in battery if r["domain"] == "civic"}):
        def sel(kind, pol=None):
            return np.array([i for i, r in enumerate(battery) if r["concept"] == value
                             and r["kind"] == kind and (pol is None or r["polarity"] == pol)])
        idx = dict(affirm=sel("pair", "affirm"), violate=sel("pair", "violate"),
                   dilemma=sel("dilemma"), ctrl=ctrl_idx, ctrl_labels=ctrl_labels)
        yield value, (value_rows(short, spec, bver, value, acts_dv, idx)
                      + value_rows(short, spec, bver, value, acts, idx, suffix="_raw"))


# ---- dashboard lives in republic_dash.py (re-exported for aggregate/recompute) ----
from infra.aws_pg.jobs.republic_dash import render_dashboard  # noqa: E402


# ---- main ----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma list of ROSTER keys")
    ap.add_argument("--battery", default=BATTERY_S3_DEFAULT)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    np.random.seed(SEED)

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    battery, battery_sha = fetch_battery(args.battery, out)
    bver = battery[0]["battery"]
    texts = [r["text"] for r in battery]

    all_rows = []
    import torch
    for short in [s.strip() for s in args.models.split(",")]:
        spec = ROSTER[short]
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = out / "runs" / f"{ts}-{short}"; run_dir.mkdir(parents=True, exist_ok=True)
        log(run_dir, f"load {spec['hf_id']} @ {spec['sha'][:8]}")
        tok, model = load_model(spec)
        run_dir.joinpath("config.yaml").write_text(
            f"model: {spec['hf_id']}\nrevision: {spec['sha']}\norigin: {spec['origin']}\n"
            f"battery: {bver}\nbattery_sha256: {battery_sha}\nseed: {SEED}\n"
            f"gpu: {torch.cuda.get_device_name(0)}\ntorch: {torch.__version__}\n")
        log(run_dir, f"capture {len(texts)} texts")
        acts = capture(tok, model, texts)
        del model; torch.cuda.empty_cache()
        acts_dir = out / "acts"; acts_dir.mkdir(exist_ok=True)
        np.save(acts_dir / f"{short}.npy", acts.astype(np.float16))  # free local re-analysis
        log(run_dir, f"acts {acts.shape} saved; probing (normalized, valence k={VALENCE_K} projected out)")
        for value, rows in analyze_model(battery, acts, short, spec, bver):
            all_rows += rows
            df = pd.DataFrame(all_rows)
            df.to_parquet(out / "results.parquet", index=False)
            render_dashboard(df, out, bver)
            log(run_dir, f"{value}: done ({len(all_rows)} rows total)")
        log(run_dir, f"{short} complete")
    print("ALL_DONE")


if __name__ == "__main__":
    main()
