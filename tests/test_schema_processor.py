from src.data_pipeline.schema import Assessment, AssessmentCatalog, CatalogMetadata
from src.data_pipeline.processor import DataProcessor


def test_assessment_url_auto_normalization():
    a = Assessment(name="Test", url="/products/example")
    assert a.url.startswith("https://www.shl.com/")


def test_catalog_remove_duplicates_by_url():
    assessments = [
        Assessment(name="A", url="https://www.shl.com/a"),
        Assessment(name="B", url="https://www.shl.com/b"),
        Assessment(name="A2", url="https://www.shl.com/a"),
    ]
    catalog = AssessmentCatalog(assessments, CatalogMetadata("now", "src", 3))
    deduped = catalog.remove_duplicates()
    assert len(deduped.assessments) == 2


def test_processor_cleans_whitespace_and_test_type():
    catalog = AssessmentCatalog(
        [
            Assessment(
                name="  Java Test  ",
                url="https://www.shl.com/java",
                test_type=" k ",
                description="  Some   description   here.  ",
                category="Tech",
            )
        ],
        CatalogMetadata("now", "src", 1),
    )
    processor = DataProcessor(catalog)
    cleaned = processor.clean_data()
    item = cleaned.assessments[0]

    assert item.name == "Java Test"
    assert item.test_type == "K"
    assert item.description == "Some description here."
