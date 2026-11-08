"""ATLAS shared job glue — freeze-assert mechanics, parent-artifact pointers, the `atlas|` item
split, the v₁ reconstruction (regrain_f_steer construction, verbatim via bispec_a), and the one
cell-runner every behavioural arm uses.

Freeze mechanics (spec §0 + journal AM-5). `pg` ships ONLY `infra/` to the instance (no `.git`,
no `studies/`), so the tag itself is checked on the CONTROL PLANE (`scripts/atlas_check_freeze.py`,
which has git) before every launch; the DRIVER's half of the contract is enforced here:
`assert_frozen(arm)` recomputes sha256 of the frozen spec copy shipped inside the tarball
(`atlas_spec_frozen.md`) and refuses to run unless it matches the manifest entry for its arm.
A driver whose arm has no manifest entry refuses to launch. No retroactive tags.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

from infra.aws_pg.jobs._regrain_shared import (  # noqa: F401  (re-exported for the atlas drivers)
    GRAIN_DICTS_L0,
    GRAIN_GATE0,
    N_PPL_SEQ,
    ROT_SEED,
    S3_BUCKET,
    dict_bank_name,
    effect_from,
    effect_se,
    fetch,
    install_pg,
    pinned_pg1_sha,
)

# ---- parent artifacts (read-only; every load-bearing number RECOMPUTED from these) ----------
BISPEC_SIGMA = f"{S3_BUCKET}/results/bispec_sigma/out"
BISPEC_GATE0 = f"{S3_BUCKET}/results/bispec_gate0/out/gate0.json"     # AM-1: not "gate0b"
BISPEC_K = f"{S3_BUCKET}/results/bispec_k/out/metrics.json"
BISPEC_K_SEALED = f"{S3_BUCKET}/results/bispec_k_sealed/out/metrics.json"
BISPEC_A = f"{S3_BUCKET}/results/bispec_a/out/metrics.json"
BISPEC_A3 = f"{S3_BUCKET}/results/bispec_a3/out/a3.json"
BISPEC_S = f"{S3_BUCKET}/results/bispec_s/out/metrics.json"
BISPEC_W = f"{S3_BUCKET}/results/bispec_w/out/metrics.json"
BISPEC_V1R = f"{S3_BUCKET}/results/bispec_v1r/out/v1.json"
BISPEC_V1R_ROWS = f"{S3_BUCKET}/results/bispec_v1r/out/named_rows.pt"
BISPEC_LEVELSET = f"{S3_BUCKET}/results/bispec_levelset_law/out/levelset_law.json"
REGRAIN_HARVEST = f"{S3_BUCKET}/results/regrain_pipeA/out/harvest"
REGRAIN_FRONTIER = f"{S3_BUCKET}/results/regrain_pipeA/out/i_frontier/frontier.json"
REGRAIN_CDISC = f"{S3_BUCKET}/results/regrain_c_disc/out"
REGRAIN_C_ENERGY = f"{S3_BUCKET}/results/regrain_c_energy/out/metrics.json"
REGRAIN_GATE0 = f"{S3_BUCKET}/results/regrain_gate0/out/gate0.json"
ATLAS_GATE0 = f"{S3_BUCKET}/results/atlas_gate0/out"
ATLAS_D0 = f"{S3_BUCKET}/results/atlas_d0/out"
ATLAS_CORPUS = f"{S3_BUCKET}/results/atlas_d0/out/atlas_corpus.json"

FUNC_BASE_PPL = 22.4258            # reproduced at Gate-0, never assumed (invcore-FUNC anchor)
PPL_TOL = 0.01
SIGMA1_ANCHOR = 9.433703479082313  # banked bispec_a direction_source.regrain_f_steer_sigma1

_JOBS_DIR = pathlib.Path(__file__).resolve().parent
SPEC_FROZEN = _JOBS_DIR / "atlas_spec_frozen.md"
FREEZE_MANIFEST = _JOBS_DIR / "atlas_freeze.json"


def assert_frozen(arm):
    """The driver half of freeze-or-refuse. Raises unless (a) the manifest exists and names this
    arm, and (b) the frozen spec copy in THIS packaged tree hashes to the manifest's value."""
    assert FREEZE_MANIFEST.exists(), f"no freeze manifest at {FREEZE_MANIFEST} — arm '{arm}' refuses to launch"
    man = json.loads(FREEZE_MANIFEST.read_text())
    assert arm in man, f"arm '{arm}' has no freeze entry — refuses to launch (no retroactive tags)"
    assert SPEC_FROZEN.exists(), "frozen spec copy missing from the packaged tree"
    h = hashlib.sha256(SPEC_FROZEN.read_bytes()).hexdigest()
    assert h == man[arm]["spec_sha256"], (
        f"spec hash mismatch for arm '{arm}': tree={h[:12]} manifest={man[arm]['spec_sha256'][:12]} "
        f"— the packaged spec is not the frozen one; refuse")
    return man[arm]


def atlas_split(content_hashes, name="atlas_vqa"):
    """Fresh `atlas|` salt (spec §4.1). The Gate-0 overlap audit runs beside this; the split
    itself is pure sha1 bucketing, deterministic in the item content hashes."""
    import numpy as np

    from infra.pg.diff_metrics import enforce_split
    held = np.array([int(hashlib.sha1(f"atlas|{h}".encode()).hexdigest(), 16) % 2 == 1
                     for h in content_hashes], dtype=bool)
    return enforce_split(held, name=name)


def load_sigma_bank(site, workdir="/tmp/atlas"):
    import torch
    bank = torch.load(fetch(f"{BISPEC_SIGMA}/sigma_bank.pt", f"{workdir}/sigma_bank.pt"),
                      map_location="cpu", weights_only=False)
    assert site in bank, f"site {site} not in the banked Σ (has {sorted(bank)}); L8 comes from atlas_gate0"
    return bank[site]


def build_v1(harvest_s3, cdisc_s3, site="L0", n_rows=40000, device="cuda", workdir="/tmp/atlas"):
    """R18's direction, rebuilt EXACTLY as regrain_f_steer/bispec_a built it: PC1 of the B*-union
    contribution cloud on 40k banked rows, `use_threshold=True`, fired-row filter. Asserts σ₁
    reproduces the banked 9.4337 (±1.5, the bispec_a band). Returns (v1 fp64 cpu, sigma1, meta)."""
    import torch

    from infra.pg import grain_bsf as GB
    from infra.pg import regrain_core as RC
    L = int(site[1:])
    cd = json.loads(fetch(f"{cdisc_s3}/c_disc.json", f"{workdir}/c_disc.json").read_text())
    bstar = [int(x) for x in cd["bstar"]]
    med_s = int(cd["median_seed"])
    assert int(cd["L"]) == L, f"B* lives at L{cd['L']}, not {site}"
    nm = dict_bank_name("primary", 1, med_s)
    src = f"{GRAIN_DICTS_L0}/{nm}" if med_s == 0 else f"{cdisc_s3}/{nm}"
    ae, _c, _q, _m = GB.load_dict_bank(fetch(src, f"{workdir}/{nm}"))
    ae = ae.to(device).eval()
    meta = json.loads(fetch(f"{harvest_s3}/harvest_meta.json", f"{workdir}/harvest_meta.json").read_text())
    shard0 = meta["shards"][site][0]
    X = torch.load(fetch(f"{harvest_s3}/{shard0}", f"{workdir}/shards/{shard0}"),
                   map_location="cpu", weights_only=False)["X"][:n_rows].double()
    Db = RC.decoder_matrix(ae)[:, bstar].float().to(device)
    with torch.inference_mode():
        xs = X.float().to(device)
        fcode = ae.encode(xs, use_threshold=True)
        contrib = fcode[:, bstar] @ Db.T
        fired = contrib.abs().sum(1) > 0
        Cc = contrib[fired].double()
        mu = Cc.mean(0)
        _U, S, Vh = torch.linalg.svd(Cc - mu, full_matrices=False)
        v1 = Vh[0].cpu().clone()
        sigma1 = float(S[0] / max(1, Cc.shape[0] - 1) ** 0.5)
    del xs, fcode, contrib, Cc
    torch.cuda.empty_cache()
    assert abs(sigma1 - 9.4337) < 1.5, f"sigma1 {sigma1} does not reproduce f_steer's 9.4337 — halt"
    return v1, sigma1, {"bstar_len": len(bstar), "median_seed": med_s,
                        "n_fired_rows": int(fired.sum()), "sigma1": sigma1,
                        "sigma1_anchor": SIGMA1_ANCHOR, "n_rows": n_rows, "X_mean_norm":
                        float(X.norm(dim=1).mean())}


def grammar_cell_ctx(cell, env):
    """Build the intervention context for ANY grammar cell (the ONE constructor both the fill-in
    and sealed drivers use). `env`: layers, L, basis (fp64), sb_slices, hbar, rms, d,
    channel_rms {k,v,q}, head_dim, n_heads, rope_theta, seed."""
    import math

    import torch

    from infra.pg import atlas_core as AC
    from infra.pg import atlas_fit as AF
    from infra.pg import bispec_core as BC
    L, layers = env["L"], env["layers"]
    seed = int(env.get("seed", 0)) + 7919 * abs(hash(AF.cell_key(cell))) % 100000
    if cell["sign"] == "add" and cell["channel"] == "none":
        U = AF.subband_projector(env["basis"], env["sb_slices"], cell["subband"]).float()
        eps_abs = float(cell["amp"]) * env["rms"] * math.sqrt(env["d"])
        hook = AC.make_hook_subband_noise(U, eps_abs, seed, cell["mode"], rt=cell["rt"])
        return lambda: BC.hook_at(layers, L, hook)
    if cell["sign"] == "add":
        ch = cell["channel"]
        crms = env["channel_rms"]["k" if ch.startswith("k") else ch]
        eps_abs = float(cell["amp"]) * crms * math.sqrt(env["head_dim"])
        attr, fn = AC.channel_delta_fn(ch, cell["mode"], eps_abs, seed, env["head_dim"],
                                       env["n_heads"], env["rope_theta"])
        dim = env["head_dim"] * (env["n_heads"] if attr == "q_proj" else 1)
        hook = AC.make_proj_out_hook_add(fn, expect_dim=dim)
        return lambda: AC.proj_hook_at(layers[L], attr, hook)
    # delete
    U = AF.subband_projector(env["basis"], env["sb_slices"], cell["subband"])
    if cell["mode"] == "shared_subspace":
        hook = AC.make_hook_subband_deletion(U, env["hbar"], cell["amp"], seed, per_token=False)
    else:
        pool = AC.subband_pool(U, n_pool=512, seed=seed)
        hook = AC.make_hook_subband_deletion(U, env["hbar"], cell["amp"], seed, per_token=True,
                                             pool_frames=pool)
    return lambda: BC.hook_at(layers, L, hook)


class CellRunner:
    """The one behavioural measurement path every ATLAS arm shares (the bispec_k `measure`
    pattern, factored): VQA per-item soft-acc + blank-image wikitext ppl at n_seq=120 under an
    intervention context, effect on the dead-floor convention, log(1+Δppl) beside raw, saturation
    guard attached (VOID recorded, never imputed). Writes through `writer()` after every cell."""

    def __init__(self, model, proc, task, items, intact, floor, base_ppl, rep, writer,
                 n_ppl_seq=N_PPL_SEQ):
        self.model, self.proc, self.task = model, proc, task
        self.items, self.intact, self.floor, self.base_ppl = items, intact, floor, base_ppl
        self.rep, self._w, self.n_ppl_seq = rep, writer, n_ppl_seq

    def run(self, name, ctx_factory, *, family, arm, draw=0, skip_vqa=False, **extra):
        import math as _m

        from infra.pg import inherit_pg1 as IP
        if skip_vqa:
            r = None
        else:
            r = IP.eval_pg1(self.model, self.proc, self.task, self.items, "cuda", ctx=ctx_factory())
        with ctx_factory():
            p = IP.ppl_gate(self.model, self.proc, "cuda", n_seq=self.n_ppl_seq)["ppl"]
        rec = {"cell": name, "family": family, "arm": arm, "draw": draw,
               "ppl": p, "dppl": p - self.base_ppl,
               "log1p_dppl": _m.log1p(max(p - self.base_ppl, -0.99)), **extra}
        if r is not None:
            span = self.intact["score"] - self.floor
            assert span > 1e-3, "no dynamic range vs the dead floor — halt"
            rec.update({"vqa": r["score"], "dvqa": r["score"] - self.intact["score"],
                        "effect": effect_from(r["per_item"], self.intact["per_item"], self.floor),
                        "effect_se": effect_se(r["per_item"], self.intact["per_item"], self.floor),
                        # single-cell dead flag: the VQA score sits at/below the scrambled floor
                        # (contrast-level VOIDs use bispec_core.saturation_guard on the PAIR)
                        "saturated_dead": bool((r["score"] - self.floor) / span < 0.02)})
        rep_cells = self.rep.setdefault("cells", [])
        rep_cells.append(rec)
        self._w()
        print(f"[atlas] {name} draw{draw}: dppl={rec['dppl']:+.3f}"
              + (f" dVQA={rec['dvqa']:+.4f}" if r is not None else ""), flush=True)
        return rec
