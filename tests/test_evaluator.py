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
