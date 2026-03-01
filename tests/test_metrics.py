from src.evaluation.metrics import mean_recall_at_k, precision_at_k, calculate_metrics


def test_mean_recall_at_k_basic_case():
    predictions = [["u1", "u2", "u3"], ["u4", "u5", "u6"]]
    ground_truth = [["u1", "u3", "u9"], ["u4", "u8"]]

    # Query1 recall=2/3, Query2 recall=1/2 => mean=7/12
    score = mean_recall_at_k(predictions, ground_truth, k=3)
    assert score == (2 / 3 + 1 / 2) / 2


def test_precision_at_k_basic_case():
    predictions = [["a", "b", "c"], ["d", "e", "f"]]
    ground_truth = [["a", "x"], ["q", "e"]]

    # Query1 precision=1/3, Query2 precision=1/3 => mean=1/3
    score = precision_at_k(predictions, ground_truth, k=3)
    assert score == 1 / 3


def test_calculate_metrics_includes_expected_keys():
    predictions = [["u1", "u2"], ["u3", "u4"]]
    ground_truth = [["u1"], ["u5"]]

    metrics = calculate_metrics(predictions, ground_truth, k_values=[5, 10])
    assert "mean_recall@5" in metrics
    assert "mean_precision@5" in metrics
    assert "f1@5" in metrics
    assert "mean_recall@10" in metrics
    assert "mean_precision@10" in metrics
    assert "f1@10" in metrics
