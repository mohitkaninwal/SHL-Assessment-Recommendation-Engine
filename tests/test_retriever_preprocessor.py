import pytest
import numpy as np
import json

retriever_module = pytest.importorskip("src.recommendation.retriever")
QueryPreprocessor = retriever_module.QueryPreprocessor
AssessmentRetriever = retriever_module.AssessmentRetriever


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


def test_extract_keywords_from_jd_text_removes_boilerplate():
    jd_like = (
        "About the job Job Description What You Will Be Doing "
        "Develop and experiment with machine learning models like NLP. "
        "Implement emerging LLM technologies. "
        "Essential Proficiency in Python and TensorFlow."
    )
    keywords = QueryPreprocessor.extract_keywords(jd_like)
    assert "job" not in keywords
    assert "description" not in keywords
    assert "doing" not in keywords
    assert "python" in keywords
    assert "nlp" in keywords
    assert "llm" in keywords


def test_detect_test_type_preference_k_and_p():
    assert QueryPreprocessor.detect_test_type_preference("Need java coding and software expertise") == "K"
    assert QueryPreprocessor.detect_test_type_preference("Need teamwork and communication skills") == "P"


def test_domain_scores_detects_mixed_query():
    scores = QueryPreprocessor.domain_scores(
        "Need a Java developer who collaborates with stakeholders"
    )
    assert scores["K"] > 0
    assert scores["P"] > 0


def test_extract_duration_constraints_range_and_hour():
    min_m, max_m = QueryPreprocessor.extract_duration_constraints(
        "Test should be 30-40 mins long"
    )
    assert min_m == 30
    assert max_m == 40

    min_h, max_h = QueryPreprocessor.extract_duration_constraints(
        "Need about an hour for each assessment"
    )
    assert min_h == 45
    assert max_h == 75

    min_upto, max_upto = QueryPreprocessor.extract_duration_constraints(
        "Assessment should not be more than 90 mins"
    )
    assert min_upto == 0
    assert max_upto == 90


class _FakeEmbeddingGenerator:
    def generate_embedding(self, text):
        return np.array([0.1, 0.2, 0.3], dtype=float)


class _FakeVectorDB:
    def __init__(self, results):
        self._results = results

    def search(self, query_embedding, top_k=10, filter_dict=None, include_metadata=True):
        return self._results[:top_k]


def _count_type(results, code):
    count = 0
    for result in results:
        all_types = (result.get("metadata", {}).get("all_test_types") or "").upper().split()
        if code in all_types:
            count += 1
    return count


def test_retrieve_balanced_uses_all_test_types_for_mix():
    mocked_results = [
        {"id": "1", "score": 0.95, "metadata": {"all_test_types": "K P", "test_type": "K"}},
        {"id": "2", "score": 0.92, "metadata": {"all_test_types": "K", "test_type": "K"}},
        {"id": "3", "score": 0.91, "metadata": {"all_test_types": "K", "test_type": "K"}},
        {"id": "4", "score": 0.90, "metadata": {"all_test_types": "P", "test_type": "P"}},
        {"id": "5", "score": 0.89, "metadata": {"all_test_types": "P", "test_type": "P"}},
    ]
    retriever = AssessmentRetriever(
        vector_db=_FakeVectorDB(mocked_results),
        embedding_generator=_FakeEmbeddingGenerator(),
        catalog_path="data/does_not_exist_for_test.json",
    )

    results = retriever.retrieve_balanced(
        query="Need a Java developer who can collaborate with stakeholders",
        top_k=4,
        hard_skill_ratio=0.6,
        min_score=-1.0,
    )

    assert len(results) == 4
    assert _count_type(results, "K") >= 2
    assert _count_type(results, "P") >= 2


def test_hybrid_retrieval_combines_semantic_and_keyword(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    catalog_payload = {
        "assessments": [
            {
                "name": "AI Skills",
                "url": "https://example.com/ai-skills",
                "test_type": "P",
                "all_test_types": "P",
                "description": "Assessment focused on AI, ML, NLP and LLM capabilities",
                "duration": 20,
                "remote_support": "Yes",
                "adaptive_support": "No",
                "category": "AI",
            },
            {
                "name": "General Communication",
                "url": "https://example.com/communication",
                "test_type": "P",
                "all_test_types": "P",
                "description": "Communication and teamwork assessment",
            },
        ]
    }
    catalog_path.write_text(json.dumps(catalog_payload), encoding="utf-8")

    semantic_results = [
        {
            "id": "sem_1",
            "score": 0.99,
            "metadata": {
                "name": "Data Warehousing Concepts",
                "url": "https://example.com/data-warehouse",
                "test_type": "K",
                "all_test_types": "K",
                "description": "Warehouse concepts and ETL",
            },
        }
    ]

    retriever = AssessmentRetriever(
        vector_db=_FakeVectorDB(semantic_results),
        embedding_generator=_FakeEmbeddingGenerator(),
        catalog_path=str(catalog_path),
    )

    results = retriever.retrieve(
        query="Need AI ML NLP and LLM skills",
        top_k=3,
        test_type_filter=None,
        min_score=-1.0,
    )
    names = [r.get("metadata", {}).get("name", "") for r in results]
    assert "AI Skills" in names
