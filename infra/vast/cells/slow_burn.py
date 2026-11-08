"""Rehearsal-only cell: compute for `seconds` before writing its artifact.

Exists so the §9.1 kill test has a window to kill the container mid-cell and
prove the restart redoes exactly this cell and skips the already-banked ones.
"""
import json
import os
import time

import torch


def run(cell, out_dir):
    seconds = cell.get("seconds", 30)
    t0 = time.time()
    acc = 0.0
    while time.time() - t0 < seconds:
        a = torch.randn(256, 256)
        acc += (a @ a).sum().item()
    with open(os.path.join(out_dir, "slow_burn.json"), "w") as f:
        json.dump({"cell": cell["id"], "seconds": seconds, "acc": acc}, f)
