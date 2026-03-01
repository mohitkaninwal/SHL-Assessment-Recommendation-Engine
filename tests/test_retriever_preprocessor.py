import pytest

retriever_module = pytest.importorskip("src.recommendation.retriever")
QueryPreprocessor = retriever_module.QueryPreprocessor


def test_clean_query_removes_extra_whitespace_and_symbols():
    query = "  Java@@ developer   with   collaboration!!!  "
    cleaned = QueryPreprocessor.clean_query(query)
    assert cleaned == "Java developer with collaboration!!!"


def test_extract_keywords_removes_basic_stopwords():
    query = "Looking for a python developer with strong teamwork"
    keywords = QueryPreprocessor.extract_keywords(query)
    assert "for" not in keywords
    assert "python" in keywords
    assert "teamwork" in keywords


def test_detect_test_type_preference_k_and_p():
    assert QueryPreprocessor.detect_test_type_preference("Need java coding and software expertise") == "K"
    assert QueryPreprocessor.detect_test_type_preference("Need teamwork and communication skills") == "P"
