"""Split a harvest's sampled neurons into a discovery set and an audit set (SPEC_D1 §4.4).

usage: python3 split_sample.py <harvest out dir> <n_discovery_neurons> <seed> <out dir>

Reads samples/sample_draw.json (neurons as [layer, index] pairs, poles as ids), draws n_discovery
neurons uniformly without replacement with numpy default_rng(seed), and writes
  <out>/discovery_neurons.json, <out>/audit_neurons.json      ([layer, index] lists)
  <out>/discovery_poles.json, <out>/audit_poles.json          (pole id lists, both poles of each neuron)
  <out>/split.json                                             (seed, counts, source path, sha256 of sample_draw)
Fails if the out dir already holds a split (the split is drawn once).
"""
import hashlib
import json
import pathlib
import sys

import numpy as np


def main(harvest, n_disc, seed, out):
    harvest, out = pathlib.Path(harvest), pathlib.Path(out)
    draw_path = harvest / "samples" / "sample_draw.json"
    draw = json.loads(draw_path.read_text())
    neurons = [tuple(n) for n in draw["neurons"]]
    if (out / "split.json").exists():
        raise FileExistsError(f"{out / 'split.json'} exists; the split is drawn once")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(neurons), size=n_disc, replace=False)
    disc = sorted(neurons[i] for i in idx)
    audit = sorted(n for n in neurons if n not in set(disc))
    assert len(disc) + len(audit) == len(neurons) and not (set(disc) & set(audit))

    def poles(ns):
        return [f"L{l:02d}N{i}{s}" for l, i in ns for s in "+-"]

    out.mkdir(parents=True, exist_ok=True)
    for name, obj in (("discovery_neurons.json", [list(n) for n in disc]), ("audit_neurons.json", [list(n) for n in audit]),
                      ("discovery_poles.json", poles(disc)), ("audit_poles.json", poles(audit)),
                      ("split.json", {"seed": seed, "n_neurons": len(neurons), "n_discovery": len(disc), "n_audit": len(audit),
                                      "source": str(draw_path), "sample_draw_sha256": hashlib.sha256(draw_path.read_bytes()).hexdigest(),
                                      "rule": "numpy default_rng(seed).choice(n, size=n_discovery, replace=False) over the harvest's sample_draw order"})):
        (out / name).write_text(json.dumps(obj, indent=1))
    # every pole id must exist in the harvest's own pole list
    missing = set(poles(disc) + poles(audit)) - set(draw["poles"])
    assert not missing, f"poles not in sample_draw: {sorted(missing)[:5]}"
    print(json.dumps({"discovery": len(disc), "audit": len(audit), "poles": len(draw["poles"]), "out": str(out)}))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
