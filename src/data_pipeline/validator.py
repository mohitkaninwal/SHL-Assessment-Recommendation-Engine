"""
Data Validation Module
Validates scraped data meets requirements
"""

import logging
from typing import Dict, List, Tuple
from .schema import AssessmentCatalog

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """Validate assessment catalog data"""
    
    def __init__(self, catalog: AssessmentCatalog):
        """
        Initialize validator
        
        Args:
            catalog: AssessmentCatalog to validate
        """
        self.catalog = catalog
        self.validation_results = {}
    
    def validate_all(self) -> Dict:
        """
        Run all validation checks
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'count_check': self.validate_count(),
            'duplicate_check': self.validate_duplicates(),
            'required_fields_check': self.validate_required_fields(),
            'url_format_check': self.validate_url_format(),
            'individual_tests_only': self.validate_individual_tests_only(),
            'overall_valid': True
        }
        
        # Overall validation
        results['overall_valid'] = all([
            results['count_check']['passed'],
            results['duplicate_check']['passed'],
            results['required_fields_check']['passed'],
            results['url_format_check']['passed'],
            results['individual_tests_only']['passed']
        ])
        
        self.validation_results = results
        return results
    
    def validate_count(self) -> Dict:
        """
        Validate that we have ≥377 Individual Test Solutions
        
        Returns:
            Dictionary with validation result
        """
        count = len(self.catalog.assessments)
        passed = count >= 377
        
        result = {
            'passed': passed,
            'count': count,
            'required': 377,
            'message': f"Found {count} assessments (required: ≥377)"
        }
        
        if not passed:
            result['error'] = f"Insufficient assessments: {count} < 377"
        
        return result
    
    def validate_duplicates(self) -> Dict:
        """
        Check for duplicate entries
        
        Returns:
            Dictionary with validation result
        """
        duplicates = self.catalog.get_duplicates()
        passed = len(duplicates) == 0
        
        result = {
            'passed': passed,
            'duplicate_count': len(duplicates),
            'duplicate_pairs': duplicates[:10],  # First 10 pairs
            'message': f"Found {len(duplicates)} duplicate pairs"
        }
        
        if not passed:
            result['error'] = f"Found {len(duplicates)} duplicate assessment pairs"
        
        return result
    
    def validate_required_fields(self) -> Dict:
        """
        Validate that all assessments have required fields
        
        Returns:
            Dictionary with validation result
        """
        missing_fields = {
            'name': [],
            'url': []
        }
        
        for idx, assessment in enumerate(self.catalog.assessments):
            if not assessment.name or not assessment.name.strip():
                missing_fields['name'].append(idx)
            if not assessment.url or not assessment.url.strip():
                missing_fields['url'].append(idx)
        
        total_issues = len(missing_fields['name']) + len(missing_fields['url'])
        passed = total_issues == 0
        
        result = {
            'passed': passed,
            'missing_name_count': len(missing_fields['name']),
            'missing_url_count': len(missing_fields['url']),
            'missing_fields': missing_fields,
            'message': f"Found {total_issues} assessments with missing required fields"
        }
        
        if not passed:
            result['error'] = f"{len(missing_fields['name'])} missing names, {len(missing_fields['url'])} missing URLs"
        
        return result
    
    def validate_url_format(self) -> Dict:
        """
        Validate URL format
        
        Returns:
            Dictionary with validation result
        """
        invalid_urls = []
        
        for idx, assessment in enumerate(self.catalog.assessments):
            url = assessment.url
            if not url.startswith('http'):
                invalid_urls.append({
                    'index': idx,
                    'url': url,
                    'issue': 'URL does not start with http'
                })
        
        passed = len(invalid_urls) == 0
        
        result = {
            'passed': passed,
            'invalid_count': len(invalid_urls),
            'invalid_urls': invalid_urls[:10],  # First 10
            'message': f"Found {len(invalid_urls)} invalid URL formats"
        }
        
        if not passed:
            result['error'] = f"Found {len(invalid_urls)} URLs with invalid format"
        
        return result
    
    def validate_individual_tests_only(self) -> Dict:
        """
        Validate that catalog contains only Individual Test Solutions
        (no Pre-packaged Job Solutions)
        
        Returns:
            Dictionary with validation result
        """
        excluded_keywords = [
            'pre-packaged',
            'prepackaged',
            'job solution',
            'package',
            'suite',
            'bundle'
        ]
        
        excluded_assessments = []
        
        for idx, assessment in enumerate(self.catalog.assessments):
            name = (assessment.name or '').lower()
            description = (assessment.description or '').lower()
            
            if any(keyword in name or keyword in description for keyword in excluded_keywords):
                excluded_assessments.append({
                    'index': idx,
                    'name': assessment.name,
                    'reason': 'Contains excluded keywords'
                })
        
        passed = len(excluded_assessments) == 0
        
        result = {
            'passed': passed,
            'excluded_count': len(excluded_assessments),
            'excluded_assessments': excluded_assessments[:10],  # First 10
            'message': f"Found {len(excluded_assessments)} excluded assessments"
        }
        
        if not passed:
            result['error'] = f"Found {len(excluded_assessments)} assessments that should be excluded"
        
        return result
    
    def print_report(self):
        """Print validation report"""
        if not self.validation_results:
            self.validate_all()
        
        print("\n" + "=" * 60)
        print("DATA VALIDATION REPORT")
        print("=" * 60)
        
        for check_name, result in self.validation_results.items():
            if check_name == 'overall_valid':
                continue
            
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            print(f"\n{status}: {check_name.replace('_', ' ').title()}")
            print(f"  {result['message']}")
            
            if not result['passed'] and 'error' in result:
                print(f"  Error: {result['error']}")
        
        print("\n" + "=" * 60)
        overall_status = "✓ VALIDATION PASSED" if self.validation_results['overall_valid'] else "✗ VALIDATION FAILED"
        print(overall_status)
        print("=" * 60 + "\n")
        
        return self.validation_results['overall_valid']


def validate_catalog(catalog_path: str) -> bool:
    """
    Validate a catalog file
    
    Args:
        catalog_path: Path to catalog JSON file
        
    Returns:
        True if validation passed, False otherwise
    """
    logger.info(f"Loading catalog from {catalog_path}")
    catalog = AssessmentCatalog.load_json(catalog_path)
    
    logger.info(f"Validating {len(catalog)} assessments")
    validator = DataValidator(catalog)
    
    return validator.print_report()


if __name__ == "__main__":
    import sys
    
    catalog_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed_catalog.json"
    is_valid = validate_catalog(catalog_path)
    sys.exit(0 if is_valid else 1)







