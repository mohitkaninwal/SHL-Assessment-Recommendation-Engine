"""
Main script for data collection pipeline
Runs scraping, processing, and validation
"""

import argparse
import logging
import sys
from pathlib import Path
from .scraper import scrape_shl_catalog
from .processor import process_scraped_data
from .schema import AssessmentCatalog

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for data pipeline"""
    parser = argparse.ArgumentParser(description='SHL Assessment Catalog Scraper')
    parser.add_argument(
        '--mode',
        choices=['scrape', 'process', 'full'],
        default='full',
        help='Pipeline mode: scrape only, process only, or full pipeline'
    )
    parser.add_argument(
        '--raw-data',
        type=str,
        default='data/raw_catalog.json',
        help='Path to raw scraped data file'
    )
    parser.add_argument(
        '--processed-data',
        type=str,
        default='data/processed_catalog.json',
        help='Path to processed data file'
    )
    parser.add_argument(
        '--use-selenium',
        action='store_true',
        help='Use Selenium for dynamic content (default: False, uses requests)'
    )
    parser.add_argument(
        '--session-rotation',
        type=int,
        default=5,
        help='Restart browser every N pages to avoid detection (default: 5)'
    )
    parser.add_argument(
        '--validate-urls',
        action='store_true',
        help='Validate URLs are accessible'
    )
    parser.add_argument(
        '--url-sample-size',
        type=int,
        default=10,
        help='Number of URLs to validate (default: 10, use 0 for all)'
    )
    parser.add_argument(
        '--no-backup-raw',
        action='store_true',
        help='Disable timestamped backup of raw data before processing'
    )
    parser.add_argument(
        '--backup-dir',
        type=str,
        default='data/backups',
        help='Directory for raw-data backups (default: data/backups)'
    )
    
    args = parser.parse_args()
    
    # Ensure data directory exists
    Path(args.raw_data).parent.mkdir(parents=True, exist_ok=True)
    Path(args.processed_data).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.mode in ['scrape', 'full']:
            logger.info("=" * 60)
            logger.info("PHASE 1: Web Scraping")
            logger.info("=" * 60)
            
            assessments = scrape_shl_catalog(
                output_path=args.raw_data,
                use_selenium=args.use_selenium,
                session_rotation_interval=args.session_rotation
            )
            
            logger.info(f"✓ Scraped {len(assessments)} Individual Test Solutions")
            
            if len(assessments) < 377:
                logger.warning(f"⚠ Warning: Expected ≥377 assessments, found {len(assessments)}")
            else:
                logger.info(f"✓ Validation passed: Found {len(assessments)} assessments (≥377 required)")
        
        if args.mode in ['process', 'full']:
            logger.info("=" * 60)
            logger.info("PHASE 2: Data Processing & Cleaning")
            logger.info("=" * 60)
            
            url_sample = args.url_sample_size if args.url_sample_size > 0 else None
            
            catalog = process_scraped_data(
                input_path=args.raw_data,
                output_path=args.processed_data,
                validate_urls=args.validate_urls,
                url_sample_size=url_sample,
                backup_raw=not args.no_backup_raw,
                backup_dir=args.backup_dir
            )
            
            logger.info(f"✓ Processed {len(catalog)} assessments")
            
            # Final validation
            logger.info("=" * 60)
            logger.info("PHASE 3: Data Validation")
            logger.info("=" * 60)
            
            # Check count
            if len(catalog) >= 377:
                logger.info(f"✓ Count validation: {len(catalog)} assessments (≥377 required)")
            else:
                logger.error(f"✗ Count validation failed: {len(catalog)} assessments (need ≥377)")
                sys.exit(1)
            
            # Check duplicates
            duplicates = catalog.get_duplicates()
            if duplicates:
                logger.warning(f"⚠ Found {len(duplicates)} duplicate pairs")
            else:
                logger.info("✓ No duplicates found")
            
            # Validate all assessments
            all_valid, errors = catalog.validate_all()
            if all_valid:
                logger.info("✓ All assessments are valid")
            else:
                logger.warning(f"⚠ Found {len(errors)} invalid assessments")
                for idx, error_list in list(errors.items())[:5]:  # Show first 5
                    logger.warning(f"  Assessment {idx}: {error_list}")
            
            # Statistics
            from .processor import DataProcessor
            processor = DataProcessor(catalog)
            stats = processor.get_statistics()
            
            logger.info("=" * 60)
            logger.info("STATISTICS")
            logger.info("=" * 60)
            logger.info(f"Total assessments: {stats['total_assessments']}")
            logger.info(f"Assessments with description: {stats['with_description']}")
            logger.info(f"Assessments with category: {stats['with_category']}")
            logger.info(f"By test type: {stats['by_test_type']}")
            
            logger.info("=" * 60)
            logger.info("✓ Data pipeline completed successfully!")
            logger.info("=" * 60)
            
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
