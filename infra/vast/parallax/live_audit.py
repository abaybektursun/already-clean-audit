"""Read-only Vast market audit with sanitized, reproducible output.

This module never creates, stops, or destroys a resource.  It invokes only
``vastai search ... --raw`` and stores public offer fields.  The credential is
read by the Vast CLI from its normal config and is never put in argv or output.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import atomic_json
from .offers import H100Policy, rank_h100_offers, rank_volume_bundles, rank_volume_offers


H100_QUERY = (
    "gpu_name in [H100_SXM,H100_NVL,H100_PCIE,H100] "
    "num_gpus=1 gpu_ram>=80 reliability>=0.98 disk_space>=500 "
    "duration>=4 rentable=true rented=false"
)
VOLUME_QUERY = "disk_space>=4000 reliability>=0.98"

# Public fields needed to audit cost/suitability.  Host connection details and
# any unrecognized future API fields are intentionally dropped.
OFFER_FIELDS = {
    "id",
    "machine_id",
    "gpu_name",
    "num_gpus",
    "gpu_ram",
    "reliability",
    "verified",
    "rentable",
    "rented",
    "dph",
    "dph_base",
    "dph_total",
    "storage_cost",
    "disk_space",
    "disk_bw",
    "inet_down",
    "inet_up",
    "inet_down_cost",
    "inet_up_cost",
    "duration",
    "geolocation",
    "geolocode",
    "cpu_cores_effective",
    "cpu_ram",
    "cuda_vers",
    "driver_version",
    "static_ip",
}


def _vast_json(arguments: list[str]) -> list[dict[str, Any]]:
    command = ["vastai", *arguments, "--raw"]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, list):
        raise TypeError(f"expected Vast list response, got {type(payload).__name__}")
    return payload


def _sanitize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: row[key] for key in sorted(OFFER_FIELDS) if key in row}
        for row in rows
    ]


def audit(*, scratch_gb: int = 500, volume_gb: int = 4000) -> dict[str, Any]:
    h100_raw = _vast_json(
        [
            "search",
            "offers",
            H100_QUERY,
            "--limit",
            "100",
            "--storage",
            str(scratch_gb),
            "--order",
            "dph_total",
        ]
    )
    volume_raw = _vast_json(
        [
            "search",
            "volumes",
            VOLUME_QUERY,
            "--limit",
            "100",
            "--storage",
            str(volume_gb),
            "--order",
            "storage_cost",
        ]
    )
    # `vastai search` applies `external=false rentable=true verified=true` by
    # default. The raw response currently omits the `verified` field, so the
    # query provenance is the evidence and the pure post-filter must not reject
    # a missing response field.
    policy = H100Policy(
        min_disk_gb=scratch_gb,
        requested_disk_gb=scratch_gb,
        require_verified=False,
    )
    ranked_h100 = rank_h100_offers(h100_raw, policy)
    ranked_volumes = rank_volume_offers(volume_raw, min_capacity_gb=volume_gb)

    bundles: list[dict[str, Any]] = []
    for volume in ranked_volumes[:25]:
        machine_id = int(volume["machine_id"])
        gateway_rows = _vast_json(
            [
                "search",
                "offers",
                f"machine_id={machine_id} num_gpus=1 rentable=true rented=false",
                "--limit",
                "20",
                "--storage",
                "32",
                "--order",
                "dph_total",
            ]
        )
        if not gateway_rows:
            continue
        gateway_rows.sort(key=lambda row: float(row.get("dph_total", float("inf"))))
        bundles.append({"volume": volume, "gateway": gateway_rows[0]})
    ranked_bundles = rank_volume_bundles(bundles, volume_gb=volume_gb)

    # Vast has already quoted dph_total for the requested scratch size. Use the
    # platform quote as the primary rank; the independently recomputed cost is
    # retained as an audit cross-check.
    ranked_h100.sort(
        key=lambda row: (
            float(row.get("dph_total", row["projected_hourly_cost"])),
            -float(row.get("reliability", 0)),
        )
    )
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "vast.ai",
        "mutation": "none_read_only",
        "queries": {
            "h100": H100_QUERY,
            "persistent_volume": VOLUME_QUERY,
        },
        "requested": {
            "h100_scratch_gb": scratch_gb,
            "persistent_volume_gb": volume_gb,
        },
        "selection_policy": {
            "all_regions": True,
            "vast_cli_default_filter": "external=false rentable=true verified=true",
            "h100": "lowest live Vast dph_total among suitable verified offers",
            "volume": "lowest storage_cost among suitable verified local-volume offers",
            "requery_immediately_before_create": True,
            "offer_id_is_not_frozen": True,
        },
        "h100_offer_count": len(ranked_h100),
        "volume_offer_count": len(ranked_volumes),
        "accessible_volume_bundle_count": len(ranked_bundles),
        "selected_h100": _sanitize(ranked_h100[:1]),
        "selected_volume_bundle": (
            {
                "volume": _sanitize([ranked_bundles[0]["volume"]])[0],
                "gateway": _sanitize([ranked_bundles[0]["gateway"]])[0],
                "volume_monthly_cost": ranked_bundles[0]["volume_monthly_cost"],
                "gateway_hourly_cost": ranked_bundles[0]["gateway_hourly_cost"],
            }
            if ranked_bundles
            else None
        ),
        "h100_shortlist": _sanitize(ranked_h100[:10]),
        "volume_shortlist": _sanitize(ranked_volumes[:10]),
        "accessible_volume_bundles": [
            {
                "volume": _sanitize([bundle["volume"]])[0],
                "gateway": _sanitize([bundle["gateway"]])[0],
                "volume_monthly_cost": bundle["volume_monthly_cost"],
                "gateway_hourly_cost": bundle["gateway_hourly_cost"],
            }
            for bundle in ranked_bundles[:10]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch-gb", type=int, default=500)
    parser.add_argument("--volume-gb", type=int, default=4000)
    args = parser.parse_args()
    report = audit(scratch_gb=args.scratch_gb, volume_gb=args.volume_gb)
    if not report["selected_h100"]:
        raise SystemExit("NO_GO: no suitable H100 offers")
    if not report["selected_volume_bundle"]:
        raise SystemExit("NO_GO: no persistent volume has a rentable same-machine gateway")
    atomic_json(args.output, report)
    selected = report["selected_h100"][0]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected_h100_offer": selected["id"],
                "quoted_dph_total": selected.get("dph_total"),
                "mutation": "none_read_only",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
