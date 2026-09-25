"""Command-line entry point for Weft.

Subcommands ``run``, ``bench``, and ``validate`` (README.md section 7.3) are not implemented yet.
"""

from __future__ import annotations

import argparse
import sys

from weft import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="weft",
        description="Simulator and benchmark suite for spatial compute-in-memory accelerators.",
    )
    parser.add_argument("--version", action="version", version=f"weft {__version__}")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run", help="Evaluate one model on one hardware config (not implemented).")
    sub.add_parser("bench", help="Run the WeftBench suite (not implemented).")
    sub.add_parser("validate", help="Re-run silicon validation points (not implemented).")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    print(f"weft {args.command}: not implemented yet", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
