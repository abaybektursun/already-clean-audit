"""Minimal end-to-end infra smoke: load PaliGemma-2 3B from the cached weights, run one
forward on the GPU, write a JSON. Proves the whole loop — env AMI + weights snapshot cache
+ H100 + S3 sync (--out) + self-terminate + reaper — without needing any dataset.

Mirrors the repo's proven load convention (infra/pg/contrast_a1.py). Fail-loud: no try/except.
"""
import argparse
import json
import pathlib

import numpy as np
import torch
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

from infra.pg.entities import ENTITIES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=ENTITIES["PG"].repo_id)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    assert torch.cuda.is_available(), "no CUDA on this instance"
    device = torch.cuda.get_device_name(0)

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        a.repo, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa",
    ).eval()
    proc = PaliGemmaProcessor.from_pretrained(a.repo)

    img = Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8))
    inp = proc(text="caption en\n", images=img, return_tensors="pt")
    # processor emits float32 pixel_values but the model is bf16 — cast like the repo does.
    moved = {}
    for k, v in inp.items():
        if torch.is_tensor(v):
            v = v.to("cuda")
            if k == "pixel_values":
                v = v.to(torch.bfloat16)
        moved[k] = v
    with torch.inference_mode():
        o = model(**moved)

    res = {
        "ok": True,
        "device": device,
        "torch": torch.__version__,
        "repo": a.repo,
        "logits_shape": list(o.logits.shape),
        "logits_finite": bool(torch.isfinite(o.logits).all()),
    }
    (out / "smoke.json").write_text(json.dumps(res, indent=2))
    print("SMOKE_OK", json.dumps(res))


if __name__ == "__main__":
    main()
