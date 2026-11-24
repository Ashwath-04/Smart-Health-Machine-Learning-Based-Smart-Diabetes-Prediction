"""Final one-time evaluation on the held-out test set, plus error analysis and SHAP.

Usage:
    python scripts/evaluate.py            # requires `python scripts/train.py` first
    python scripts/evaluate.py --no-shap  # skip the (slower) SHAP global analysis
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.evaluation.evaluate import run_evaluation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-shap", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    warnings.filterwarnings("ignore", category=FutureWarning)
    cfg = load_config(args.config)
    if not (cfg.path("paths", "models_dir") / "model.joblib").exists():
        sys.exit("No trained model found. Run `python scripts/train.py` first.")
    res = run_evaluation(cfg, run_shap=not args.no_shap)
    print(res["final"].to_string(index=False))


if __name__ == "__main__":
    main()
