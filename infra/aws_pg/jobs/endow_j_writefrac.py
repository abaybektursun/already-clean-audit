"""ENDOW-J J3 write-fraction battery (spec §6.2) — is each conserved core SPECIALLY avoided by the
robot hop, or is low-write generic? Per-site + frame-clean: for each projection site, the robot-hop
weight change ΔW = π0.5_decoder − PG1_decoder is projected onto (a) the site's conserved core
U_unt-<site> and (b) matched-random spans in the SAME input frame. Generalizes INHERIT's R6
(‖ΔW_q·U_unt‖/‖ΔW_q‖) from q/k to all 6 sites; layer-bootstrap; the frozen J3 specialness rule.

J3 verdict: SPECIAL iff U_unt's write fraction is strictly below matched-random (non-overlapping
bootstrap CIs) at a site — the hop avoids the conserved core beyond chance. Killer (boring branch):
comparable to random everywhere → low-write is generic FT behavior → conservation claims re-read.

CPU (r7i, ≥64GB): loads PG1 + π0.5 decoders once; per-site prefetch + 6 representative layers
(the U_unt(L) at each layer is a consensus bottom-k; gate/up are [16384,2048], so all-18 is slow).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.endow_j_writefrac --type r7i.8xlarge,r7i.4xlarge \
      --no-weights --s3-prefix results/endow_j_writefrac --budget-min 180 --args "--k 128 --n-null 20"
"""
import argparse
import json
import pathlib
import subprocess

from infra.aws_pg.jobs._inherit_shared import INVCORE_SCRATCH, N_FT

SITES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj")


def _prefetch_site(site, workdir):
    subprocess.run(["aws", "s3", "cp", f"{INVCORE_SCRATCH}/", str(workdir), "--recursive",
                    "--exclude", "*", "--include", f"{site}__*", "--region", "us-east-1",
                    "--only-show-errors"], check=True)


def _uunt_layers(site, k, layers, workdir):
    """U_unt-<site>(L) = bottom-k consensus eigvecs of the cached 24-FT ΔWs, at the sampled layers."""
    import numpy as np
    import torch
    from infra.pg.inherit_core import bottom_eigvecs, consensus_gram
    U = {}
    for L in layers:
        deltas = [torch.as_tensor(np.load(f"{workdir}/{site}__L{L}__{i}.npy").astype(np.float64))
                  for i in range(N_FT)]
        U[L], _ = bottom_eigvecs(torch.as_tensor(consensus_gram(deltas)), k)
    return U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--n-null", type=int, default=20)
    ap.add_argument("--layers", default="0,3,6,9,12,15")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in a.layers.split(",")]

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    from infra.pg import endow_writefrac as WF
    from infra.aws_pg.jobs.vlat_retention import _bare_decoder, _extract_decoder

    work = pathlib.Path("/tmp/endow_j")
    work.mkdir(parents=True, exist_ok=True)
    print("[J3] loading PG1 decoder ...", flush=True)
    pg1_dec, pg1_sha = _extract_decoder("PG1", "cpu", None)
    print("[J3] loading pi05 decoder ...", flush=True)
    pi05_dec, pi05_sha = _extract_decoder("pi05", "cpu", None)
    pg1_dec, pi05_dec = _bare_decoder(pg1_dec), _bare_decoder(pi05_dec)
    common = set(pg1_dec) & set(pi05_dec)
    assert any("q_proj" in k for k in common), \
        f"PG1/pi05 decoder keys misaligned — pg1={list(pg1_dec)[:3]} pi05={list(pi05_dec)[:3]}"
    print(f"[J3] decoders loaded ({len(common)} shared keys)", flush=True)
    rep = {"k": a.k, "n_null": a.n_null, "layers": layers, "pg1_sha": str(pg1_sha),
           "pi05_sha": str(pi05_sha), "per_site": {}}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    def robot_dW(site, L):
        import torch
        part = "self_attn" if "proj" in site and site[0] in "qkvo" else "mlp"
        key = f"layers.{L}.{part}.{site}.weight"
        return (pi05_dec[key].to(torch.float64) - pg1_dec[key].to(torch.float64))

    for site in SITES:
        _prefetch_site(site, work)
        U = _uunt_layers(site, a.k, layers, str(work))
        unt_frac = {L: WF.write_fraction(robot_dW(site, L), U[L]) for L in layers}
        rnd_frac = {L: float(np.mean(WF.matched_random_fractions(robot_dW(site, L), a.k, a.n_null, a.seed + L)))
                    for L in layers}
        unt_boot = WF.layer_bootstrap(unt_frac, seed=a.seed)
        rnd_boot = WF.layer_bootstrap(rnd_frac, seed=a.seed + 1)
        verdict = WF.specialness(unt_boot, {"matched_random": rnd_boot})
        rep["per_site"][site] = {"unt": unt_boot, "random": rnd_boot, "verdict": verdict,
                                 "unt_frac_by_L": unt_frac, "rnd_frac_by_L": rnd_frac}
        print(f"[J3] {site}: unt={unt_boot['mean']:.4f}[{unt_boot['ci_lo']:.4f},{unt_boot['ci_hi']:.4f}] "
              f"rnd={rnd_boot['mean']:.4f} special={verdict['special']}", flush=True)
        for f in work.glob(f"{site}__*"):
            f.unlink()
        _w()

    n_special = sum(1 for s in SITES if rep["per_site"][s]["verdict"]["special"])
    rep["j3_summary"] = {"n_sites_special": n_special, "n_sites": len(SITES),
                         "verdict": "conservation_special" if n_special >= len(SITES) / 2
                         else "low_write_generic"}
    rep["green"] = True
    _w()
    print("J3_OK", json.dumps(rep["j3_summary"], default=str))


if __name__ == "__main__":
    main()
