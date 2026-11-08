"""Stand-in probe server (transport smoke): answers every request in $PROBE_DIR/requests with the request itself,
plus the CUDA device name and whether HF_TOKEN is set (never its value), so one round trip checks the station's
environment. The real server follows the same directory protocol (infra/vast/cells/probe_station.py).
"""
import json
import os
import pathlib
import time

import torch

d = pathlib.Path(os.environ["PROBE_DIR"])
dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda"
hf = bool(os.environ.get("HF_TOKEN"))
print(f"[probe_echo] up: {dev}, HF_TOKEN set: {hf}", flush=True)
while True:
    for req in sorted((d / "requests").glob("*.json")):
        res = d / "responses" / req.name
        if res.exists():
            continue
        body = {"rid": req.stem, "server": "probe_echo", "cuda_device": dev, "hf_token_set": hf,
                "answered_utc": time.strftime("%FT%TZ", time.gmtime()), "request": json.loads(req.read_text())}
        tmp = res.with_suffix(".tmp")
        tmp.write_text(json.dumps(body, indent=2) + "\n")
        tmp.rename(res)
        print(f"[probe_echo] answered {req.name}", flush=True)
    time.sleep(0.02)
