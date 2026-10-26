"""Compatibility entry point; prefer python -m edgeformer sensitivity."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgeformer.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["sensitivity", *sys.argv[1:]]))
