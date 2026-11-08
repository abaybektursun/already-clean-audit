"""Pure selection policy for Vast offers; contains no credentialed API calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


H100_NAMES = {"H100", "H100 PCIE", "H100 SXM", "H100 NVL"}


@dataclass(frozen=True)
class H100Policy:
    gpu_count: int = 1
    min_gpu_ram_gb: float = 80.0
    min_reliability: float = 0.98
    min_disk_gb: float = 500.0
    requested_disk_gb: float = 500.0
    min_duration_days: float = 4.0
    require_verified: bool = True


def normalized_gpu_name(value: object) -> str:
    return str(value).replace("_", " ").upper()


def projected_hourly_cost(offer: dict[str, Any], requested_disk_gb: float) -> float:
    """Compute + requested local-disk cost using Vast's $/GB/month field."""
    compute = float(offer.get("dph_base", offer.get("dph", 0.0)))
    storage_month = float(offer.get("storage_cost", 0.0)) * requested_disk_gb
    return compute + storage_month / (30.0 * 24.0)


def h100_rejections(offer: dict[str, Any], policy: H100Policy) -> list[str]:
    reasons: list[str] = []
    if normalized_gpu_name(offer.get("gpu_name")) not in H100_NAMES:
        reasons.append("gpu_name")
    if int(offer.get("num_gpus", 0)) != policy.gpu_count:
        reasons.append("gpu_count")
    if float(offer.get("gpu_ram", 0.0)) < policy.min_gpu_ram_gb:
        reasons.append("gpu_ram")
    if float(offer.get("reliability", 0.0)) < policy.min_reliability:
        reasons.append("reliability")
    if float(offer.get("disk_space", 0.0)) < policy.min_disk_gb:
        reasons.append("disk_space")
    if float(offer.get("duration", 0.0)) < policy.min_duration_days:
        reasons.append("duration")
    if policy.require_verified and offer.get("verified") is not True:
        reasons.append("verified")
    if offer.get("rentable") is not True or offer.get("rented") is True:
        reasons.append("availability")
    return reasons


def rank_h100_offers(
    offers: Iterable[dict[str, Any]], policy: H100Policy = H100Policy()
) -> list[dict[str, Any]]:
    suitable: list[dict[str, Any]] = []
    for offer in offers:
        reasons = h100_rejections(offer, policy)
        if reasons:
            continue
        enriched = dict(offer)
        enriched["projected_hourly_cost"] = projected_hourly_cost(
            offer, policy.requested_disk_gb
        )
        suitable.append(enriched)
    return sorted(
        suitable,
        key=lambda item: (
            float(item["projected_hourly_cost"]),
            -float(item.get("reliability", 0.0)),
            -float(item.get("inet_up", 0.0)),
        ),
    )


def rank_volume_offers(
    offers: Iterable[dict[str, Any]], *, min_capacity_gb: float = 4000.0
) -> list[dict[str, Any]]:
    suitable = [
        dict(offer)
        for offer in offers
        if float(offer.get("disk_space", 0.0)) >= min_capacity_gb
        and float(offer.get("reliability", 0.0)) >= 0.98
    ]
    return sorted(
        suitable,
        key=lambda item: (
            float(item.get("storage_cost", float("inf"))),
            -float(item.get("reliability", 0.0)),
            -float(item.get("inet_up", 0.0)),
        ),
    )


def rank_volume_bundles(
    bundles: Iterable[dict[str, Any]], *, volume_gb: float
) -> list[dict[str, Any]]:
    """Rank only volumes that have a currently rentable same-machine gateway."""
    eligible: list[dict[str, Any]] = []
    for bundle in bundles:
        volume = bundle.get("volume")
        gateway = bundle.get("gateway")
        if not isinstance(volume, dict) or not isinstance(gateway, dict):
            continue
        enriched = dict(bundle)
        enriched["volume_monthly_cost"] = float(volume["storage_cost"]) * volume_gb
        enriched["gateway_hourly_cost"] = float(
            gateway.get("dph_total", gateway.get("dph_base", gateway.get("dph", float("inf"))))
        )
        eligible.append(enriched)
    return sorted(
        eligible,
        key=lambda item: (
            float(item["volume_monthly_cost"]),
            float(item["gateway_hourly_cost"]),
            -float(item["volume"].get("reliability", 0.0)),
            -float(item["volume"].get("inet_up", 0.0)),
        ),
    )
