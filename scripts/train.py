"""Train all candidate models and save the deployable artifact.

Usage:
    python scripts/train.py                 # full run (reported results)
    python scripts/train.py --reuse-tuning  # skip Optuna, reuse reports/metrics/tuning_results.json
    python scripts/train.py --quick         # tiny smoke-test settings (NOT for reporting)
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.training.train import quick_overrides, run_training  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--quick", action="store_true", help="Fast smoke-test settings")
    parser.add_argument("--reuse-tuning", action="store_true", help="Reuse saved Optuna results")
    parser.add_argument("--skip-eda", action="store_true", help="Do not regenerate EDA figures")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("optuna").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=FutureWarning)

    cfg = load_config(args.config)
    if args.quick:
        cfg = cfg.copy_with(quick_overrides())
    result = run_training(cfg, reuse_tuning=args.reuse_tuning, skip_eda=args.skip_eda)
    print("\nDeployed model:", result["selection"]["deployed_model"])
    print("Best by CV    :", result["selection"]["best_by_cv"])


if __name__ == "__main__":
    main()
