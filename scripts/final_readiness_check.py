#!/usr/bin/env python3
"""Final readiness checker for local submission artifacts."""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Tuple, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.schema import AssessmentCatalog
from src.data_pipeline.validator import DataValidator


def check_catalog(catalog_path: Path) -> Tuple[bool, dict]:
    if not catalog_path.exists():
        return False, {"error": f"Catalog file not found: {catalog_path}"}
    catalog = AssessmentCatalog.load_json(str(catalog_path))
    validator = DataValidator(catalog)
    result = validator.validate_all()
    return result.get("overall_valid", False), {
        "count": len(catalog.assessments),
        "validation": result,
    }


# Accepted CSV header variants (assignment may use Query/Assessment_url or query/assessment_url)
REQUIRED_HEADER_VARIANTS = (
    ["query", "assessment_url"],
    ["Query", "Assessment_url"],
)


def check_predictions_csv(predictions_path: Path) -> Tuple[bool, dict]:
    if not predictions_path.exists():
        return False, {"error": f"Predictions file not found: {predictions_path}"}

    with predictions_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = [h.strip() for h in (reader.fieldnames or [])]
        rows: List[dict] = list(reader)

    if headers not in REQUIRED_HEADER_VARIANTS:
        return False, {
            "error": "CSV headers do not match required format",
            "found_headers": headers,
            "required_headers": "one of: query/assessment_url or Query/Assessment_url",
        }

    # Normalize keys for validation (accept either casing)
    q_key = "Query" if "Query" in headers else "query"
    a_key = "Assessment_url" if "Assessment_url" in headers else "assessment_url"
    missing_rows = [i + 2 for i, row in enumerate(rows) if not (row.get(q_key) or "").strip() or not (row.get(a_key) or "").strip()]
    if missing_rows:
        return False, {
            "error": "Some rows have missing query or assessment_url",
            "invalid_row_numbers": missing_rows[:20],
            "invalid_count": len(missing_rows),
        }

    return True, {
        "rows": len(rows),
        "headers": headers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run final local readiness checks")
    parser.add_argument("--catalog", default="data/processed_catalog.json")
    parser.add_argument("--predictions", default="test_predictions.csv")
    parser.add_argument("--output", default="evaluation_results/final_readiness_report.json")
    args = parser.parse_args()

    catalog_ok, catalog_details = check_catalog(Path(args.catalog))
    csv_ok, csv_details = check_predictions_csv(Path(args.predictions))

    report = {
        "checks": {
            "catalog_validation": {
                "passed": catalog_ok,
                "details": catalog_details,
            },
            "predictions_csv": {
                "passed": csv_ok,
                "details": csv_details,
            },
            "deployment_urls": {
                "passed": False,
                "details": "Manual check required: verify public API/frontend URLs from external network.",
            },
            "approach_pdf": {
                "passed": False,
                "details": "Manual check required: export docs/APPROACH_DOCUMENT.md to 2-page PDF.",
            },
        }
    }

    report["overall_ready"] = all(
        check["passed"] for key, check in report["checks"].items() if key in {"catalog_validation", "predictions_csv"}
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Final readiness report written to: {output}")
    print(f"Catalog validation: {'PASS' if catalog_ok else 'FAIL'}")
    print(f"Predictions CSV: {'PASS' if csv_ok else 'FAIL'}")
    print("Deployment/API PDF checks: MANUAL")

    return 0 if report["overall_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
