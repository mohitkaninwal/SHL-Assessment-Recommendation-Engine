import pytest

rag_module = pytest.importorskip("src.recommendation.rag_pipeline")
RAGPipeline = rag_module.RAGPipeline


class _DummyLLM:
    def classify_query_intent(self, query):
        return {
            "technical_weight": 0.6,
            "behavioral_weight": 0.4,
            "primary_skills": ["java", "collaboration"],
            "reasoning": "mixed-domain",
        }

    def rerank_assessments(self, query, assessments, top_k=10):
        # Return a skewed K-heavy slice to validate post-rerank balancing.
        return assessments[:top_k]

    def generate_explanation(self, query, recommendations):
        return "ok"


class _DummyRetriever:
    def __init__(self, formatted):
        self._formatted = formatted

    def retrieve_balanced(self, query, top_k=10, hard_skill_ratio=0.6, min_score=-1.0):
        # Raw candidate shape consumed by format_results.
        return [{"metadata": item, "score": 1.0 - (idx * 0.01)} for idx, item in enumerate(self._formatted)]

    def retrieve(self, query, top_k=10, min_score=-1.0):
        return [{"metadata": item, "score": 1.0 - (idx * 0.01)} for idx, item in enumerate(self._formatted)]

    def format_results(self, results):
        out = []
        for result in results:
            md = result["metadata"]
            out.append(
                {
                    "assessment_name": md["name"],
                    "assessment_url": md["url"],
                    "test_type": md.get("test_type", ""),
                    "all_test_types": md.get("all_test_types", ""),
                    "description": md.get("description", ""),
                    "duration": md.get("duration", 0),
                    "remote_support": md.get("remote_support", "Yes"),
                    "adaptive_support": md.get("adaptive_support", "No"),
                    "category": md.get("category", ""),
                    "similarity_score": result.get("score", 0.0),
                }
            )
        return out


def _make_assessment(name, url, all_types, test_type):
    return {
        "name": name,
        "url": url,
        "all_test_types": all_types,
        "test_type": test_type,
        "description": "desc",
        "duration": 10,
        "remote_support": "Yes",
        "adaptive_support": "No",
    }


def _count_code(recs, code):
    total = 0
    for rec in recs:
        raw = (rec.get("all_test_types") or rec.get("test_type") or "").upper().split()
        if code in raw:
            total += 1
    return total


def test_post_rerank_balance_enforced_for_mixed_query():
    # Intentionally order as K-heavy first, P-heavy later.
    formatted = [
        _make_assessment("K1", "u1", "K", "K"),
        _make_assessment("K2", "u2", "K", "K"),
        _make_assessment("K3", "u3", "K", "K"),
        _make_assessment("K4", "u4", "K", "K"),
        _make_assessment("P1", "u5", "P", "P"),
        _make_assessment("P2", "u6", "P", "P"),
        _make_assessment("P3", "u7", "P", "P"),
        _make_assessment("KP1", "u8", "K P", "K"),
    ]

    pipeline = RAGPipeline(
        vector_db=object(),
        embedding_generator=object(),
        llm_client=_DummyLLM(),
        use_llm_reranking=True,
        use_query_expansion=False,
    )
    pipeline.retriever = _DummyRetriever(formatted)

    result = pipeline.recommend(
        query="Need a Java developer with strong collaboration and stakeholder communication",
        top_k=6,
        balance_skills=True,
        include_explanation=False,
    )

    recs = result["recommendations"]
    assert len(recs) == 6
    assert _count_code(recs, "K") >= 3
    assert _count_code(recs, "P") >= 3
