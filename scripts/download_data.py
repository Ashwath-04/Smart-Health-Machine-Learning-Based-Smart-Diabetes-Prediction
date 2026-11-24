"""Download the Pima Indians Diabetes dataset and verify its pinned SHA-256 checksum."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.data.loader import download_dataset  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download even if the file exists")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    path = download_dataset(load_config(), force=args.force)
    print(f"Dataset ready at {path} (checksum verified)")
