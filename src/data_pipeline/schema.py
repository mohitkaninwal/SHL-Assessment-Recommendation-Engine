"""
Data Schema for SHL Assessment Catalog
Defines the structure and validation for assessment data
"""

from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, asdict, fields
from datetime import datetime
import json


@dataclass
class Assessment:
    """Data class representing a single SHL assessment"""
    
    name: str
    url: str
    test_type: Optional[str] = None  # K=Knowledge & Skills, P=Personality & Behavior, etc.
    all_test_types: Optional[str] = None
    description: Optional[str] = None
    duration: Optional[int] = None
    remote_support: Optional[str] = None
    adaptive_support: Optional[str] = None
    category: Optional[str] = None
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        """Validate and normalize data after initialization"""
        # Ensure URL is absolute
        if self.url and not self.url.startswith('http'):
            self.url = f"https://www.shl.com{self.url}"
        
        # Normalize test_type
        if self.test_type:
            self.test_type = self.test_type.upper().strip()
        
        # Normalize simple yes/no fields
        if self.remote_support:
            self.remote_support = str(self.remote_support).strip()
        if self.adaptive_support:
            self.adaptive_support = str(self.adaptive_support).strip()
        
        # Normalize duration
        if self.duration is not None:
            try:
                self.duration = int(self.duration)
            except (TypeError, ValueError):
                self.duration = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Assessment':
        """Create Assessment from dictionary"""
        allowed = {field.name for field in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate assessment data
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not self.name or not self.name.strip():
            errors.append("Assessment name is required")
        
        if not self.url or not self.url.strip():
            errors.append("Assessment URL is required")
        elif not self.url.startswith('http'):
            errors.append(f"Invalid URL format: {self.url}")
        
        return len(errors) == 0, errors


@dataclass
class CatalogMetadata:
    """Metadata about the catalog"""
    
    scraped_at: str
    source_url: str
    total_count: int
    version: str = "1.0"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class AssessmentCatalog:
    """Container for assessment catalog with validation"""
    
    def __init__(self, assessments: List[Assessment], metadata: Optional[CatalogMetadata] = None):
        """
        Initialize catalog
        
        Args:
            assessments: List of Assessment objects
            metadata: Optional catalog metadata
        """
        self.assessments = assessments
        self.metadata = metadata or CatalogMetadata(
            scraped_at=datetime.now().isoformat(),
            source_url="https://www.shl.com/products/product-catalog/",
            total_count=len(assessments)
        )
    
    def validate_all(self) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Validate all assessments in catalog
        
        Returns:
            Tuple of (all_valid, dict_of_errors_by_index)
        """
        errors = {}
        all_valid = True
        
        for idx, assessment in enumerate(self.assessments):
            is_valid, assessment_errors = assessment.validate()
            if not is_valid:
                all_valid = False
                errors[idx] = assessment_errors
        
        return all_valid, errors
    
    def get_by_test_type(self, test_type: str) -> List[Assessment]:
        """Get all assessments of a specific test type"""
        return [a for a in self.assessments if a.test_type == test_type.upper()]
    
    def get_duplicates(self) -> List[tuple[int, int]]:
        """
        Find duplicate assessments (by URL)
        
        Returns:
            List of tuples (index1, index2) for duplicate pairs
        """
        duplicates = []
        seen_urls = {}
        
        for idx, assessment in enumerate(self.assessments):
            url = assessment.url.lower().strip()
            if url in seen_urls:
                duplicates.append((seen_urls[url], idx))
            else:
                seen_urls[url] = idx
        
        return duplicates
    
    def remove_duplicates(self) -> 'AssessmentCatalog':
        """Remove duplicate assessments, keeping the first occurrence"""
        seen_urls = set()
        unique_assessments = []
        
        for assessment in self.assessments:
            url = assessment.url.lower().strip()
            if url not in seen_urls:
                seen_urls.add(url)
                unique_assessments.append(assessment)
        
        return AssessmentCatalog(unique_assessments, self.metadata)
    
    def to_dict(self) -> Dict:
        """Convert catalog to dictionary"""
        return {
            'metadata': self.metadata.to_dict(),
            'assessments': [a.to_dict() for a in self.assessments]
        }
    
    def save_json(self, filepath: str):
        """Save catalog to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_json(cls, filepath: str) -> 'AssessmentCatalog':
        """Load catalog from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assessments = [Assessment.from_dict(a) for a in data['assessments']]
        metadata = CatalogMetadata(**data['metadata'])
        
        return cls(assessments, metadata)
    
    def save_csv(self, filepath: str):
        """Save catalog to CSV file"""
        import pandas as pd
        
        df_data = []
        for assessment in self.assessments:
            df_data.append({
                'name': assessment.name,
                'url': assessment.url,
                'test_type': assessment.test_type or '',
                'all_test_types': assessment.all_test_types or '',
                'description': assessment.description or '',
                'duration': assessment.duration if assessment.duration is not None else '',
                'remote_support': assessment.remote_support or '',
                'adaptive_support': assessment.adaptive_support or '',
                'category': assessment.category or ''
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(filepath, index=False, encoding='utf-8')
    
    def __len__(self) -> int:
        """Return number of assessments"""
        return len(self.assessments)
    
    def __repr__(self) -> str:
        """String representation"""
        return f"AssessmentCatalog(count={len(self.assessments)}, metadata={self.metadata})"
