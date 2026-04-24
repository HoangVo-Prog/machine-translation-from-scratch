"""Compatibility launcher for the training CLI in src/cli/train.py."""

from __future__ import annotations

import sys
from pathlib import Path

def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.cli.train import main as cli_main
    return cli_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
