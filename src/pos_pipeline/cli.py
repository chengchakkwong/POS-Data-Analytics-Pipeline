"""Unified command-line entry for POS pipeline jobs."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pos_pipeline",
        description="POS Data Analytics Pipeline jobs",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("daily", help="Extract local cache and sync Firestore")
    subparsers.add_parser("analytics", help="Run ABC/XYZ and target-stock jobs")
    subparsers.add_parser("min-multiple", help="Update guessed min/multiple fields")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "daily":
        from pos_pipeline.jobs.daily import run
    elif args.command == "analytics":
        from pos_pipeline.jobs.analytics import run
    elif args.command == "min-multiple":
        from pos_pipeline.jobs.min_multiple import run
    else:
        parser.error(f"unknown command: {args.command}")

    return run()


if __name__ == "__main__":
    sys.exit(main())