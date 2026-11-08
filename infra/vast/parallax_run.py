"""PARALLAX Vast-only execution entrypoint."""

from __future__ import annotations

from infra.vast.parallax.preflight import main


if __name__ == "__main__":
    raise SystemExit(main())
