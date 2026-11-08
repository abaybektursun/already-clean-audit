"""Weight-free import and architecture smoke for the isolated PARALLAX π runtime."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import atomic_json


LEROBOT_COMMIT = "3f2179f3b69708b6ad009b2e7685dd9d05269ee1"


def run() -> dict[str, object]:
    import torch
    import transformers
    from lerobot.policies.pi0.configuration_pi0 import PI0Config
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy, get_gemma_config as pi0_gemma_config
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy, get_gemma_config as pi05_gemma_config

    pi0 = PI0Config(device="cpu", paligemma_variant="gemma_2b", action_expert_variant="gemma_300m")
    pi05 = PI05Config(device="cpu", paligemma_variant="gemma_2b", action_expert_variant="gemma_300m")
    pi0_backbone = pi0_gemma_config(pi0.paligemma_variant)
    pi05_backbone = pi05_gemma_config(pi05.paligemma_variant)
    assert pi0_backbone.depth == 18 and pi05_backbone.depth == 18
    assert pi0_backbone.width == 2048 and pi05_backbone.width == 2048
    assert PI0Policy.config_class is PI0Config
    assert PI05Policy.config_class is PI05Config
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GREEN",
        "runtime_role": "pi_sidecar",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "lerobot": importlib.metadata.version("lerobot"),
        "lerobot_source_commit": LEROBOT_COMMIT,
        "pi0": {"backbone_layers": pi0_backbone.depth, "backbone_width": pi0_backbone.width},
        "pi05": {"backbone_layers": pi05_backbone.depth, "backbone_width": pi05_backbone.width},
        "core_runtime_imported": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output), "status": report["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
