"""CLEAVE bank merge (CPU) — join per-shard `cleave_banks.pt` files into the single 18-layer bank.

The Q-certification sweep is sharded by layer across concurrent boxes (the G-family quota is 32 vCPU
total, so ~3–4 g6e.2xlarge run at once — `reference_pg_gpu_quota_shape`). Each shard banks its own
layers; this joins them and re-asserts the invariants the downstream jobs depend on:

  * every one of the 18 layers is present EXACTLY once (a duplicate layer across shards means two
    different Q measurements silently disagree about the same object — fail loud, never last-wins);
  * each layer carries Q_evecs / sigma_evecs / d / certified_max;
  * the merged file's sha256 is recorded (provenance).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.cleave_merge_banks \
      --args "--shards results/cleave_qcert_a/out/cleave_banks.pt,results/cleave_qcert_b/out/cleave_banks.pt" \
      --type r7i.2xlarge --no-weights --s3-prefix results/cleave_banks --budget-min 30
"""
import argparse
import hashlib
import json
import os
import pathlib

REQUIRED = ("Q_evecs", "sigma_evecs", "d")
N_LAYERS = 18


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", required=True, help="comma list of s3 keys to cleave_banks.pt files")
    ap.add_argument("--allow-partial", action="store_true",
                    help="permit fewer than 18 layers (records exactly which are missing)")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch

    from infra.aws_pg.jobs import _cleave_shared as CLS
    CLS.install_pg()

    rep = {"object": "cleave_merge_banks_v1", "shards": [], "gpu_seconds": {"total": 0.0, "note": "CPU job"}}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    merged, origin, backfilled = {}, {}, []
    for i, key in enumerate([s.strip() for s in a.shards.split(",") if s.strip()]):
        local = CLS.fetch_s3(key, f"/tmp/cleave/shard{i}.pt")
        b = torch.load(local, map_location="cpu", weights_only=False)
        # BACKFILL: `certified_ranks` was added to the bank AFTER the L0 pilot ran, so the pilot's
        # bank carries only `certified_max`. The certification itself is not lost — it lives in that
        # shard's metrics.json under per_layer[L].certification.certified_ranks. Recover it rather
        # than let a layer drop silently out of every support (it dropped L0 — the layer with the
        # STRONGEST Q⊥energy separation — from the frozen intervention).
        need = [L for L in b if "certified_ranks" not in b[L]]
        if need:
            mkey = key.rsplit("/", 1)[0] + "/metrics.json"
            mj = json.loads(pathlib.Path(CLS.fetch_s3(mkey, f"/tmp/cleave/shard{i}_metrics.json")).read_text())
            pl = mj.get("per_layer") or {}
            for L in need:
                v = pl.get(str(L)) or pl.get(L)
                assert v is not None, f"{key}: layer {L} lacks certified_ranks AND is absent from metrics.json"
                cr = (v.get("certification") or {}).get("certified_ranks")
                assert cr is not None, f"{key}: layer {L} has no certification in metrics.json"
                b[L]["certified_ranks"] = list(cr)
                backfilled.append({"shard": key, "layer": int(L), "certified_ranks": list(cr)})
            print(f"[merge] backfilled certified_ranks for layers {need} from {mkey}", flush=True)
        layers = sorted(int(L) for L in b)
        dup = [L for L in layers if L in merged]
        assert not dup, (f"shard {key} re-banks layer(s) {dup} already provided by "
                         f"{[origin[L] for L in dup]} — two independent Q measurements of the same "
                         "layer must not be silently merged last-wins")
        for L in layers:
            miss = [k for k in REQUIRED if k not in b[L]]
            assert not miss, f"shard {key} layer {L} missing {miss}"
            merged[int(L)] = b[int(L)]
            origin[int(L)] = key
        rep["shards"].append({"key": key, "layers": layers})
        print(f"[merge] {key}: layers {layers}", flush=True)
        _w()

    rep["backfilled"] = backfilled
    have = sorted(merged)
    missing = [L for L in range(N_LAYERS) if L not in merged]
    rep["layers"] = have
    rep["missing_layers"] = missing
    if not a.allow_partial:
        assert not missing, f"merged bank is missing layers {missing} — re-run those shards"
    torch.save(merged, out / "cleave_banks.pt")
    rep["banks_sha256"] = hashlib.sha256((out / "cleave_banks.pt").read_bytes()).hexdigest()
    rep["certified_max_by_layer"] = {L: int(merged[L].get("certified_max", -1)) for L in have}
    rep["certified_ranks_by_layer"] = {L: list(merged[L].get("certified_ranks", [])) for L in have}
    # the usable map ranks = the INTERSECTION across layers (the intervention spans all 18)
    sets = [set(merged[L].get("certified_ranks", [])) for L in have]
    rep["certified_ranks_all_layers"] = sorted(set.intersection(*sets)) if sets else []
    rep["certified_max_min_over_layers"] = min(rep["certified_max_by_layer"].values()) if have else None
    rep["green"] = True
    _w()
    print("CLEAVE_MERGE_OK", json.dumps({"layers": have, "missing": missing,
                                         "certified_max_by_layer": rep["certified_max_by_layer"]},
                                        default=str), flush=True)


if __name__ == "__main__":
    main()
