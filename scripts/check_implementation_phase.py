#!/usr/bin/env python3
"""
Run full implementation-phase checks to verify the project matches and exceeds
evaluation criteria. See docs/EVALUATION_CRITERIA.md.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], desc: str) -> bool:
    print(f"\n--- {desc} ---")
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run implementation-phase checks")
    parser.add_argument("--catalog", default="data/processed_catalog.json", help="Path to processed catalog")
    parser.add_argument("--predictions", default="test_predictions.csv", help="Path to predictions CSV")
    parser.add_argument("--min-recall", type=float, default=0.5, help="Minimum Mean Recall@10 (default: 0.5)")
    parser.add_argument("--skip-eval", action="store_true", help="Skip train-set evaluation (no API keys)")
    parser.add_argument("--skip-csv-gen", action="store_true", help="Skip CSV generation; use existing predictions file")
    args = parser.parse_args()

    all_ok = True

    # 1. Catalog validation
    if not run(
        [sys.executable, "scripts/validate_catalog_quality.py", "--catalog", args.catalog],
        "Catalog validation (≥377, no duplicates, required fields)",
    ):
        all_ok = False

    # 2. Train-set evaluation with optional min-recall
    if not args.skip_eval:
        eval_cmd = [
            sys.executable, "scripts/evaluate.py",
            "--no-rag",
            "--min-recall", str(args.min_recall),
            "--iteration", "implementation_phase_check",
        ]
        if not run(eval_cmd, f"Train-set evaluation (Recall@10 >= {args.min_recall})"):
            all_ok = False
    else:
        print("\n--- Train-set evaluation: SKIPPED (--skip-eval) ---")

    # 3. Generate test predictions CSV (optional)
    if not args.skip_csv_gen:
        if not run(
            [sys.executable, "scripts/generate_test_predictions.py", "--no-rag", "--output", args.predictions],
            "Generate test_predictions.csv",
        ):
            all_ok = False
    else:
        print("\n--- CSV generation: SKIPPED (--skip-csv-gen) ---")

    # 4. Final readiness (catalog + CSV)
    if not run(
        [sys.executable, "scripts/final_readiness_check.py", "--catalog", args.catalog, "--predictions", args.predictions],
        "Final readiness (catalog + predictions CSV)",
    ):
        all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("Implementation phase checks: PASSED")
    else:
        print("Implementation phase checks: ONE OR MORE FAILED")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
