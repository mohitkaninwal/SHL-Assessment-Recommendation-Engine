#!/usr/bin/env python3
"""Validate processed catalog quality and optionally check URL accessibility."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.schema import AssessmentCatalog
from src.data_pipeline.validator import DataValidator
from src.data_pipeline.processor import DataProcessor


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate SHL catalog quality")
    parser.add_argument("--catalog", default="data/processed_catalog.json", help="Path to processed catalog JSON")
    parser.add_argument("--output", default="data/catalog_quality_report.json", help="Path to quality report JSON")
    parser.add_argument("--check-urls", action="store_true", help="Run URL accessibility checks")
    parser.add_argument("--url-sample-size", type=int, default=25, help="Number of URLs to test (0 = all)")
    parser.add_argument("--url-timeout", type=int, default=5, help="URL check timeout in seconds")
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    catalog = AssessmentCatalog.load_json(str(catalog_path))
    validator = DataValidator(catalog)
    validation = validator.validate_all()

    report = {
        "catalog_path": str(catalog_path),
        "total_assessments": len(catalog.assessments),
        "validation": validation,
    }

    if args.check_urls:
        processor = DataProcessor(catalog)
        sample_size = None if args.url_sample_size == 0 else args.url_sample_size
        report["url_accessibility"] = processor.validate_urls(sample_size=sample_size, timeout=args.url_timeout)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote quality report to {output}")
    print(f"Overall valid: {validation['overall_valid']}")
    return 0 if validation["overall_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
