"""PRAGNANZ R-ladder — the program MDE (Q1): four rungs at the GESTALT-fitted site, forced-choice
primary. Spec §1.1/§2 R-ladder; journal FR-2..FR-4 realizations.

Rungs (hooks on layers[site]'s OUTPUT residual, fp32, cast back; teacher-forced scoring only —
no generation anywhere):
  r0  no hook
  r1  per-token argmax assignment from the LIVE residual under the frozen atlas, then redraw
      from that component (arrangement preserved, sample noise added)
  r2  same, with assignments PERMUTED within each forward's flattened token pool (fresh seeded
      permutation per call; component histogram preserved exactly; arrangement destroyed)
  r3  redraw every token from the unit's moment-matched Gaussian (the floor)
P2-LIVE: an r2 hook given the identity permutation must produce BIT-IDENTICAL output to the r1
hook on the same batch (the pragnanz_core noise contract) — checked in-job before any eval.
P1 (frozen): a (unit, K) rung set is VOID if mean(r1) <= mean(r3) + 0.02 (r1 at the floor =
H_instrument for that cell).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.pragnanz_ladder --args "--units pg1,ft-aokvqa-da" \
      --type p5.4xlarge,g6e.2xlarge,g6e.4xlarge,g6e.8xlarge --s3-prefix results/pragnanz_ladder \
      --budget-min 420

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import time

SITE = 9                      # the range-valid site (GESTALT post-critic; L14 has no range)
K_MAIN, K_HALF = 32, 16       # GESTALT winning K + the half-K sensitivity rung
LADDER_SEEDS = (0, 1)
CHOICE_SEED = 20260728        # the frozen PLENUM choice seed
N_ITEMS_EVAL = 800            # confirmation-half head (frozen list, hash-banked)
R1_FLOOR_MARGIN = 0.02
ATLAS_SRC = {
    ("pg1", 32): ("results/gestalt_arm_s8/out/mde_s9/atlases", "real_K32_s0.npz"),
    ("pg1", 16): ("results/gestalt_arm_s8/out/mde_s9/atlases", "real_K16_s0.npz"),
    ("ft-aokvqa-da", 32): ("results/gestalt_arm_d/out/atlas_ftmax/site9", "full_s0.npz"),
}
HARVEST_SRC = {"pg1": "results/gestalt_arm_s8/out/harvest_pg1",
               "ft-aokvqa-da": "results/gestalt_arm_d/out/harvest_ftmax"}


def _pull(path_under_bucket, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "cp", f"s3://pg-lineage-769063704696/{path_under_bucket}",
                    str(dest), "--region", "us-east-1", "--only-show-errors"], check=True)
    return dest


def load_atlas_npz(p, device):
    import numpy as np
    import torch
    z = np.load(p)
    return {k: torch.from_numpy(z[k]).to(device) for k in
            ("log_pi", "mu", "U", "lam", "sigma2")} | {"void": False}


class RungHook:
    """mode: 'r1' | 'r2' | 'r3'. Persistent CPU generators (noise + permutation) — two hooks
    with equal seeds produce identical noise sequences (the P2-live contract)."""

    def __init__(self, layer, mode, atlas=None, mean=None, cov_sqrt=None, seed=0):
        import torch
        self.layer, self.mode, self.atlas = layer, mode, atlas
        self.mean, self.cov_sqrt = mean, cov_sqrt
        self.g_noise = torch.Generator(device="cpu").manual_seed(int(seed))
        self.g_perm = torch.Generator(device="cpu").manual_seed(int(seed) + 777_001)
        self.identity_perm = False           # P2-live switch
        self._handle = None

    def __enter__(self):
        self._handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()
        self._handle = None
        return False

    def _hook(self, module, args, output):
        import torch

        from infra.pg import pragnanz_core as PC
        from infra.pg.gestalt_splat import _log_resp
        hs = output[0] if isinstance(output, tuple) else output
        flat = hs.reshape(-1, hs.shape[-1]).float()
        n, d = flat.shape
        if self.mode == "r3":
            z = torch.randn(n, d, generator=self.g_noise, dtype=torch.float32).to(flat.device)
            new = self.mean.unsqueeze(0) + z @ self.cov_sqrt
        else:
            lg, _ = _log_resp(flat, self.atlas)
            assign = lg.argmax(1)
            if self.mode == "r2" and not self.identity_perm:
                perm = torch.randperm(n, generator=self.g_perm).to(assign.device)
                assign = assign[perm]
            elif self.mode == "r2":
                _ = torch.randperm(n, generator=self.g_perm)      # burn identically (P2 parity)
            K, dd, r = self.atlas["U"].shape
            z = torch.randn(n, r, generator=self.g_noise, dtype=torch.float32).to(flat.device)
            eps = torch.randn(n, d, generator=self.g_noise, dtype=torch.float32).to(flat.device)
            a = assign
            scaled = z * self.atlas["lam"].clamp(min=0).sqrt()[a]
            inplane = torch.einsum("nr,ndr->nd", scaled, self.atlas["U"][a])
            new = (self.atlas["mu"][a] + inplane
                   + self.atlas["sigma2"].clamp(min=0).sqrt()[a].unsqueeze(1) * eps)
        new = new.to(hs.dtype).reshape(hs.shape)
        if isinstance(output, tuple):
            return (new,) + tuple(output[1:])
        return new


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--units", default="pg1,ft-aokvqa-da")
    ap.add_argument("--n-items", type=int, default=N_ITEMS_EVAL)
    ap.add_argument("--smoke", action="store_true", help="24 items, K=32 only, 1 seed")
    a = ap.parse_args()
    t0 = time.time()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch

    from infra.aws_pg.jobs.plenum_fc import PPL_ANCHOR, PPL_REL_TOL
    from infra.pg import gestalt_splat as GS
    from infra.pg import inherit_pg1 as IP
    from infra.pg import lacuna_core as LC
    from infra.pg import lacuna_shell as LS
    from infra.pg import plenum_core as PC_fc
    from infra.pg import plenum_units as PU
    from infra.pg.plenum_core import score_item_fc

    assert torch.cuda.is_available(), "pragnanz_ladder needs CUDA"
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    units = [u.strip() for u in a.units.split(",") if u.strip()]
    n_items = 24 if a.smoke else a.n_items
    ks = (K_MAIN,) if a.smoke else (K_MAIN, K_HALF)
    seeds = (0,) if a.smoke else LADDER_SEEDS
    rep = {"object": "PRAGNANZ R-ladder (MDE)", "site": SITE, "units": units,
           "ks": list(ks), "ladder_seeds": list(seeds), "n_items": n_items,
           "choice_seed": CHOICE_SEED, "chance": PC_fc.CHANCE,
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    cells = []
    rep["cells"] = cells
    banks = {}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
        (out / "ladder_banks.json").write_text(json.dumps(banks, default=str))
    _w()

    # ---- items: CONFIRMATION half of the frozen VQAv2 manifest (out-of-sample vs the
    #      discovery-fit atlases), forced-choice candidates at the frozen seed ----------------
    tasks = LS.mm_tasks()
    vqa = tasks["vqav2"]
    rows, hashes = IP.load_items(vqa, 4000)
    disc, conf = LC.parity_split(hashes)
    c_rows = [r for r, m in zip(rows, disc) if not m]
    c_hashes = [h for h, m in zip(hashes, disc) if not m]
    eval_rows = c_rows[:n_items]
    eval_hashes = c_hashes[:n_items]
    choices = PC_fc.choices_for_task("vqav2", eval_rows, CHOICE_SEED)
    rep["items"] = {"n": len(eval_rows), "half": "confirmation",
                    "sha": hashlib.sha256("".join(eval_hashes).encode()).hexdigest()[:16]}
    banks["item_hashes"] = eval_hashes
    _w()

    # ---- shell + anchor ---------------------------------------------------------------------
    model, proc = LS.load_pg1(device=device)
    pg1_sd = LS.pg1_baseline_sd(model)
    layers = IP.decoder_layers(model)
    ppl = IP.ppl_gate(model, proc, device)
    val = float(ppl["ppl"] if isinstance(ppl, dict) else ppl)
    rel = abs(val - PPL_ANCHOR) / PPL_ANCHOR
    rep["ppl_gate"] = {"ppl": val, "rel": rel, "passes": rel < PPL_REL_TOL}
    assert rel < PPL_REL_TOL, f"ppl anchor FAIL {val}"
    _w()

    # ---- atlases + moments per (unit, K); ft-max K=16 refit in-job from its banked harvest --
    adir = pathlib.Path("/tmp/pz_atlases")
    atl, moments = {}, {}
    for unit in units:
        mz_p = _pull(f"{HARVEST_SRC[unit]}/site{SITE}/moments.npz", adir / unit / "moments.npz")
        mz = np.load(mz_p)
        mean = torch.from_numpy(mz["mean"]).float().to(device)
        Lc = GS.cov_sym_sqrt(torch.from_numpy(mz["cov"])).float().to(device)
        moments[unit] = (mean, Lc)
        for K in ks:
            src = ATLAS_SRC.get((unit, K))
            if src is not None:
                p = _pull(f"{src[0]}/{src[1]}", adir / unit / f"K{K}.npz")
                atl[(unit, K)] = load_atlas_npz(p, device)
            else:
                # in-job refit (ft-max K=16): discovery-fit harvest, frozen fitter, banked
                sdir = adir / unit / "shards"
                sdir.mkdir(parents=True, exist_ok=True)
                subprocess.run(["aws", "s3", "sync",
                                f"s3://pg-lineage-769063704696/{HARVEST_SRC[unit]}/site{SITE}",
                                str(sdir), "--region", "us-east-1", "--only-show-errors",
                                "--exclude", "item_idx*"], check=True)
                shards = sorted(sdir.glob("acts_*.npy"))
                X = torch.cat([torch.from_numpy(np.load(q)).float() for q in shards], 0)
                fit = GS.fit_mppca(GS.fit_subsample(X, seed=0).to(device), K=K, r=2, seed=0)
                assert not fit["void"], f"in-job refit VOID for {unit} K={K}"
                cpu = GS.atlas_to_cpu(fit)
                np.savez(out / f"refit_{unit}_K{K}.npz",
                         **{k: cpu[k].numpy() for k in ("log_pi", "mu", "U", "lam", "sigma2")})
                atl[(unit, K)] = {k: fit[k] for k in ("log_pi", "mu", "U", "lam", "sigma2")}
                atl[(unit, K)]["void"] = False
                del X
                torch.cuda.empty_cache()
        print(f"[PZL] atlases ready for {unit}: K {list(ks)}", flush=True)
    _w()

    # ---- P2-LIVE: identity-permutation r2 ≡ r1 bit-level on one real batch ------------------
    site_layer = layers[SITE]
    img0 = vqa.image_fn(eval_rows[0]).convert("RGB")
    inp = proc(text=[PC_fc.fc_prompt(vqa.prompt_fn(eval_rows[0]))], images=[img0],
               return_tensors="pt").to(device)
    inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
    cap = {}
    A0 = atl[(units[0], K_MAIN)]
    for tag, mode, ident in (("r1", "r1", False), ("r2id", "r2", True)):
        h = RungHook(site_layer, mode, atlas=A0, seed=123)
        h.identity_perm = ident
        cap_h = layers[SITE + 1].register_forward_pre_hook(
            lambda m, args: cap.__setitem__(tag, args[0].detach().clone()))
        with h, torch.inference_mode():
            model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                  pixel_values=inp["pixel_values"])
        cap_h.remove()
    assert torch.equal(cap["r1"], cap["r2id"]), "P2-LIVE FAIL: identity r2 != r1 bit-level"
    rep["p2_live"] = "bit-exact"
    print("[PZL] P2-live PASS (identity r2 == r1)", flush=True)
    _w()

    # ---- the ladder -------------------------------------------------------------------------
    def run_eval(tag, hook_ctx):
        import contextlib
        t1 = time.time()
        per_item = []
        n_unsc = 0
        ctx = hook_ctx if hook_ctx is not None else contextlib.nullcontext()
        with ctx:
            for i, (r, (cands, gold)) in enumerate(zip(eval_rows, choices)):
                img = vqa.image_fn(r).convert("RGB")
                s, nt = score_item_fc(model, proc, device, img,
                                      PC_fc.fc_prompt(vqa.prompt_fn(r)), cands)
                if s is None:
                    n_unsc += 1
                    per_item.append(float("nan"))
                    continue
                mean_lp = np.asarray(s) / np.asarray(nt, dtype=float)
                per_item.append(int(int(mean_lp.argmax()) == gold))
        acc = float(np.nanmean(per_item))
        cells.append({"key": tag, "acc": acc, "n": len(per_item), "n_unscorable": n_unsc,
                      "eval_s": round(time.time() - t1, 1)})
        banks[tag] = {"per_item": per_item}
        _w()
        print(f"[PZL] {tag:>28s} acc={acc:.4f} ({time.time() - t1:.0f}s)", flush=True)
        return acc

    for unit in units:
        info = PU.install_unit(unit, model, proc, pg1_sd, device,
                               {"stitch": 0, "matched_dw": 0, "rand_dec": 0})
        rep.setdefault("unit_info", {})[unit] = str(info)[:200]
        _w()
        mean, Lc = moments[unit]
        run_eval(f"{unit}|r0", None)
        # r3 is K-independent — run once per unit
        for s in seeds:
            run_eval(f"{unit}|r3|s{s}",
                     RungHook(site_layer, "r3", mean=mean, cov_sqrt=Lc, seed=2000 + s))
        for K in ks:
            A = atl[(unit, K)]
            for s in seeds:
                run_eval(f"{unit}|K{K}|r1|s{s}",
                         RungHook(site_layer, "r1", atlas=A, seed=1000 + s))
                run_eval(f"{unit}|K{K}|r2|s{s}",
                         RungHook(site_layer, "r2", atlas=A, seed=1000 + s))
            # P1 floor gate (frozen): r1 VOID if <= r3 + margin
            r1s = [c["acc"] for c in cells if c["key"].startswith(f"{unit}|K{K}|r1")]
            r3s = [c["acc"] for c in cells if c["key"].startswith(f"{unit}|r3|")]
            if r1s and r3s:
                voided = bool(np.mean(r1s) <= np.mean(r3s) + R1_FLOOR_MARGIN)
                cells.append({"key": f"{unit}|K{K}|P1_gate", "r1_mean": float(np.mean(r1s)),
                              "r3_mean": float(np.mean(r3s)), "void": voided})
                _w()
                print(f"[PZL] P1 {unit} K{K}: r1={np.mean(r1s):.4f} r3={np.mean(r3s):.4f} "
                      f"VOID={voided}", flush=True)

    rep["elapsed_s"] = time.time() - t0
    _w()
    print("PRAGNANZ_LADDER_OK", json.dumps({c["key"]: c.get("acc") for c in cells}), flush=True)


if __name__ == "__main__":
    main()
    import os as _os
    _os._exit(0)     # GESTALT D7
