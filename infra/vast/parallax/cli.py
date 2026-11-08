"""Local CLI for PARALLAX readiness and execution ledgers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import ArtifactManifest
from .dag import SprintDAG
from .launch_packet import LaunchPacketError, canonical_sha256, validate_launch_packet


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="parallax-control")
    sub = parser.add_subparsers(dest="command", required=True)

    packet = sub.add_parser("validate-packet", help="refuse or authorize a launch packet")
    packet.add_argument("packet", type=Path)
    packet.add_argument("--repo-root", type=Path, default=Path.cwd())
    packet.add_argument("--no-evidence", action="store_true")

    digest = sub.add_parser("packet-sha", help="print canonical packet SHA256")
    digest.add_argument("packet", type=Path)

    manifest = sub.add_parser("manifest", help="write a content-addressed artifact manifest")
    manifest.add_argument("--root", type=Path, required=True)
    manifest.add_argument("--stage", required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("paths", type=Path, nargs="+")

    verify = sub.add_parser("verify-restore", help="verify restored bytes against a manifest")
    verify.add_argument("manifest", type=Path)
    verify.add_argument("restored_root", type=Path)

    dag = sub.add_parser("dag-status", help="initialize or print the truth-sprint ledger")
    dag.add_argument("--matrix", type=Path, required=True)
    dag.add_argument("--state", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "validate-packet":
        try:
            payload = validate_launch_packet(
                args.packet,
                repo_root=args.repo_root,
                verify_evidence=not args.no_evidence,
            )
        except LaunchPacketError as error:
            print(error)
            return 2
        print(json.dumps({"decision": "GO", "packet_sha256": canonical_sha256(payload)}))
        return 0
    if args.command == "packet-sha":
        import yaml

        payload = yaml.safe_load(args.packet.read_text(encoding="utf-8"))
        print(canonical_sha256(payload))
        return 0
    if args.command == "manifest":
        artifact_manifest = ArtifactManifest(args.root, args.stage)
        artifact_manifest.add(*args.paths)
        artifact_manifest.write(args.output)
        print(args.output)
        return 0
    if args.command == "verify-restore":
        errors = ArtifactManifest.verify(args.manifest, args.restored_root)
        if errors:
            print("restore verification failed:\n- " + "\n- ".join(errors))
            return 3
        print("restore verification GREEN")
        return 0
    if args.command == "dag-status":
        state = SprintDAG(args.matrix, args.state).load()
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
