"""Compatibility entry point; prefer python -m edgeformer attention."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgeformer.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["attention", *sys.argv[1:]]))
