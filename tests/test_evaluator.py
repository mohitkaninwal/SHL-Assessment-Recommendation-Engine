from src.evaluation.evaluator import Evaluator


class FakeEngine:
    def recommend(self, query, top_k=10, balance_skills=True, include_explanation=False):
        if "java" in query.lower():
            recs = [
                {"assessment_url": "u1", "assessment_name": "A", "test_type": "K", "similarity_score": 0.9},
                {"assessment_url": "u2", "assessment_name": "B", "test_type": "P", "similarity_score": 0.8},
            ]
        else:
            recs = [{"assessment_url": "u3", "assessment_name": "C", "test_type": "P", "similarity_score": 0.85}]

        return {
            "query": query,
            "recommendations": recs[:top_k],
            "total_found": len(recs),
            "returned": min(len(recs), top_k),
        }


def test_evaluator_runs_and_returns_metrics(tmp_path):
    evaluator = Evaluator(FakeEngine())

    queries = ["java developer", "teamwork role"]
    ground_truth = [["u1"], ["u3"]]

    output = tmp_path / "results.json"
    results = evaluator.evaluate(queries=queries, ground_truth=ground_truth, output_file=str(output))

    assert "metrics" in results
    assert "mean_recall@10" in results["metrics"]
    assert results["summary"]["total_queries"] == 2
    assert output.exists()


def test_evaluator_applies_url_canonicalization():
    queries = ["java developer"]
    ground_truth = [["https://www.shl.com/solutions/products/product-catalog/view/u1"]]

    # Fake engine returns "u1", so patch it to use product-catalog URL variant.
    class UrlVariantEngine:
        def recommend(self, query, top_k=10, balance_skills=True, include_explanation=False):
            return {
                "query": query,
                "recommendations": [
                    {
                        "assessment_url": "https://www.shl.com/products/product-catalog/view/u1",
                        "assessment_name": "A",
                        "test_type": "K",
                        "similarity_score": 0.9,
                    }
                ],
                "total_found": 1,
                "returned": 1,
            }

    evaluator = Evaluator(UrlVariantEngine())
    results = evaluator.evaluate(queries=queries, ground_truth=ground_truth, save_predictions=False)

    assert results["metrics"]["mean_recall@10"] == 1.0
    assert results["url_normalization"]["enabled"] is True


def test_evaluator_reports_stage_metrics_for_retrieval_and_final():
    class FakeRetriever:
        def retrieve_balanced(self, query, top_k=10, hard_skill_ratio=0.6, min_score=-1.0):
            return [
                {"metadata": {"url": "u1", "name": "A", "test_type": "K", "all_test_types": "K"}},
                {"metadata": {"url": "uX", "name": "X", "test_type": "P", "all_test_types": "P"}},
            ]

        def format_results(self, results):
            formatted = []
            for r in results:
                md = r["metadata"]
                formatted.append(
                    {
                        "assessment_url": md["url"],
                        "assessment_name": md["name"],
                        "test_type": md["test_type"],
                        "all_test_types": md["all_test_types"],
                    }
                )
            return formatted

    class FakePipeline:
        retriever = FakeRetriever()

    class StageEngine:
        pipeline = FakePipeline()

        def recommend(self, query, top_k=10, balance_skills=True, include_explanation=False):
            return {
                "query": query,
                "recommendations": [
                    {"assessment_url": "u1", "assessment_name": "A", "test_type": "K", "similarity_score": 0.9}
                ],
                "query_analysis": {"technical_weight": 0.8, "behavioral_weight": 0.2},
                "total_found": 1,
                "returned": 1,
            }

    evaluator = Evaluator(StageEngine())
    results = evaluator.evaluate(
        queries=["java developer"],
        ground_truth=[["u1"]],
        save_predictions=False,
    )

    assert "stage_metrics" in results
    assert results["stage_metrics"]["retrieval"]["available"] is True
    assert results["stage_metrics"]["final_recommendation"]["available"] is True
    assert "mean_recall@10" in results["stage_metrics"]["retrieval"]["metrics"]
    assert "mean_recall@10" in results["stage_metrics"]["final_recommendation"]["metrics"]
