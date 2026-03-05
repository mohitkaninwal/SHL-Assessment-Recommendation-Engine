"""
Data Processing and Cleaning Module
Handles cleaning, normalization, and validation of scraped data
"""

import logging
import json
import re
from typing import List, Dict, Optional
from pathlib import Path
import requests
from .schema import Assessment, AssessmentCatalog, CatalogMetadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Process and clean scraped assessment data"""
    
    def __init__(self, catalog: AssessmentCatalog):
        """
        Initialize processor with catalog
        
        Args:
            catalog: AssessmentCatalog object
        """
        self.catalog = catalog
        self.cleaning_stats = {
            'duplicates_removed': 0,
            'invalid_removed': 0,
            'urls_fixed': 0,
            'descriptions_cleaned': 0
        }
    
    def clean_data(self) -> AssessmentCatalog:
        """
        Clean and normalize assessment data
        
        Returns:
            Cleaned AssessmentCatalog
        """
        logger.info("Starting data cleaning process")
        
        # Remove duplicates
        original_count = len(self.catalog.assessments)
        self.catalog = self.catalog.remove_duplicates()
        self.cleaning_stats['duplicates_removed'] = original_count - len(self.catalog.assessments)
        
        # Clean individual assessments
        cleaned_assessments = []
        for assessment in self.catalog.assessments:
            cleaned = self._clean_assessment(assessment)
            if cleaned:
                cleaned_assessments.append(cleaned)
            else:
                self.cleaning_stats['invalid_removed'] += 1
        
        # Update catalog
        metadata = CatalogMetadata(
            scraped_at=self.catalog.metadata.scraped_at,
            source_url=self.catalog.metadata.source_url,
            total_count=len(cleaned_assessments)
        )
        
        cleaned_catalog = AssessmentCatalog(cleaned_assessments, metadata)
        
        logger.info(f"Cleaning complete. Removed {self.cleaning_stats['duplicates_removed']} duplicates, "
                   f"{self.cleaning_stats['invalid_removed']} invalid entries")
        
        return cleaned_catalog
    
    def _clean_assessment(self, assessment: Assessment) -> Optional[Assessment]:
        """
        Clean a single assessment
        
        Args:
            assessment: Assessment to clean
            
        Returns:
            Cleaned Assessment or None if invalid
        """
        # Validate required fields
        is_valid, errors = assessment.validate()
        if not is_valid:
            logger.warning(f"Invalid assessment skipped: {errors}")
            return None
        
        # Clean name
        name = assessment.name.strip()
        if not name:
            return None
        
        # Clean URL
        url = assessment.url.strip()
        if not url.startswith('http'):
            url = f"https://www.shl.com{url}"
            self.cleaning_stats['urls_fixed'] += 1
        
        # Clean description
        description = None
        if assessment.description:
            description = assessment.description.strip()
            if description:
                # Remove extra whitespace
                description = ' '.join(description.split())
                self.cleaning_stats['descriptions_cleaned'] += 1
        
        # Normalize yes/no fields
        def _to_yes_no(value: Optional[str]) -> Optional[str]:
            if value is None:
                return None
            value_str = str(value).strip().lower()
            if value_str in {'yes', 'y', 'true', '1'}:
                return 'Yes'
            if value_str in {'no', 'n', 'false', '0'}:
                return 'No'
            return None
        
        remote_support = _to_yes_no(assessment.remote_support)
        adaptive_support = _to_yes_no(assessment.adaptive_support)
        
        duration = None
        if assessment.duration is not None:
            try:
                duration = int(assessment.duration)
            except (TypeError, ValueError):
                match = re.search(r'\d+', str(assessment.duration))
                duration = int(match.group(0)) if match else None
        
        # Normalize test_type
        test_type = None
        if assessment.test_type:
            test_type = assessment.test_type.upper().strip()
            if test_type not in ['K', 'P', 'A', 'C']:  # Common types
                # Try to infer from description
                desc_lower = (description or '').lower()
                if 'knowledge' in desc_lower or 'skill' in desc_lower:
                    test_type = 'K'
                elif 'personality' in desc_lower or 'behavior' in desc_lower:
                    test_type = 'P'
        
        return Assessment(
            name=name,
            url=url,
            test_type=test_type,
            all_test_types=assessment.all_test_types,
            description=description,
            duration=duration,
            remote_support=remote_support,
            adaptive_support=adaptive_support,
            category=assessment.category,
            metadata=assessment.metadata
        )
    
    def validate_urls(self, sample_size: Optional[int] = None, timeout: int = 5) -> Dict:
        """
        Validate that URLs are accessible
        
        Args:
            sample_size: Number of URLs to check (None = all)
            timeout: Request timeout in seconds
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating URLs...")
        
        assessments_to_check = self.catalog.assessments
        if sample_size:
            assessments_to_check = assessments_to_check[:sample_size]
        
        results = {
            'total_checked': len(assessments_to_check),
            'accessible': 0,
            'inaccessible': 0,
            'errors': []
        }
        
        for idx, assessment in enumerate(assessments_to_check):
            try:
                response = requests.head(assessment.url, timeout=timeout, allow_redirects=True)
                if response.status_code < 400:
                    results['accessible'] += 1
                else:
                    results['inaccessible'] += 1
                    results['errors'].append({
                        'index': idx,
                        'url': assessment.url,
                        'status_code': response.status_code
                    })
            except Exception as e:
                results['inaccessible'] += 1
                results['errors'].append({
                    'index': idx,
                    'url': assessment.url,
                    'error': str(e)
                })
            
            # Log progress every 10 URLs
            if (idx + 1) % 10 == 0:
                logger.info(f"Validated {idx + 1}/{len(assessments_to_check)} URLs")
        
        logger.info(f"URL validation complete: {results['accessible']} accessible, "
                   f"{results['inaccessible']} inaccessible")
        
        return results
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the catalog
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_assessments': len(self.catalog.assessments),
            'by_test_type': {},
            'with_description': sum(1 for a in self.catalog.assessments if a.description),
            'with_category': sum(1 for a in self.catalog.assessments if a.category),
            'cleaning_stats': self.cleaning_stats
        }
        
        # Count by test type
        for assessment in self.catalog.assessments:
            test_type = assessment.test_type or 'Unknown'
            stats['by_test_type'][test_type] = stats['by_test_type'].get(test_type, 0) + 1
        
        return stats


def process_scraped_data(
    input_path: str,
    output_path: str,
    validate_urls: bool = False,
    url_sample_size: Optional[int] = 10,
    backup_raw: bool = True,
    backup_dir: str = "data/backups"
) -> AssessmentCatalog:
    """
    Process scraped data: load, clean, and save
    
    Args:
        input_path: Path to raw scraped JSON file
        output_path: Path to save processed data
        validate_urls: Whether to validate URLs
        url_sample_size: Number of URLs to validate (None = all)
        backup_raw: Whether to create timestamped backup of raw input file
        backup_dir: Directory where raw backups are stored
        
    Returns:
        Processed AssessmentCatalog
    """
    logger.info(f"Loading scraped data from {input_path}")
    
    # Backup raw data before any processing to preserve source of truth
    if backup_raw:
        backup_root = Path(backup_dir)
        backup_root.mkdir(parents=True, exist_ok=True)
        timestamp = Path(input_path).stat().st_mtime
        from datetime import datetime
        stamp = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
        backup_path = backup_root / f"{Path(input_path).stem}_{stamp}.json"
        with open(input_path, 'r', encoding='utf-8') as src, open(backup_path, 'w', encoding='utf-8') as dst:
            dst.write(src.read())
        logger.info(f"Created raw data backup at {backup_path}")

    # Load raw data
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Convert to Assessment objects
    assessments = []
    for item in raw_data.get('assessments', []):
        assessment = Assessment.from_dict(item)
        assessments.append(assessment)
    
    # Build metadata
    # Support both the older flat structure:
    #   {"scraped_at": ..., "source_url": ..., "total_count": ..., "assessments": [...]}
    # and a nested structure:
    #   {"metadata": {...}, "assessments": [...]}
    metadata_dict = raw_data.get("metadata")
    if metadata_dict is None:
        # Fallback to flat keys
        metadata_dict = {
            "scraped_at": raw_data.get("scraped_at"),
            "source_url": raw_data.get("source_url"),
            "total_count": raw_data.get("total_count", len(assessments)),
        }
    metadata = CatalogMetadata(**metadata_dict)
    catalog = AssessmentCatalog(assessments, metadata)
    
    logger.info(f"Loaded {len(catalog)} assessments")
    
    # Process and clean
    processor = DataProcessor(catalog)
    cleaned_catalog = processor.clean_data()
    
    # Validate URLs if requested
    if validate_urls:
        validation_results = processor.validate_urls(sample_size=url_sample_size)
        logger.info(f"URL validation: {validation_results['accessible']}/{validation_results['total_checked']} accessible")
    
    # Save processed data
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cleaned_catalog.save_json(output_path)
    logger.info(f"Saved processed data to {output_path}")
    
    # Save statistics
    stats = processor.get_statistics()
    stats_path = output_path.replace('.json', '_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved statistics to {stats_path}")
    
    return cleaned_catalog


if __name__ == "__main__":
    # Example usage
    import sys
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/raw_catalog.json"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "data/processed_catalog.json"
    
    catalog = process_scraped_data(input_file, output_file, validate_urls=True)
    print(f"\nProcessed {len(catalog)} assessments")
