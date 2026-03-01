from src.evaluation.url_utils import canonicalize_assessment_url, canonicalize_url_lists, unique_url_overlap


def test_canonicalize_assessment_url_normalizes_solutions_path():
    url = "https://www.shl.com/solutions/products/product-catalog/view/python-new/"
    expected = "https://www.shl.com/products/product-catalog/view/python-new"
    assert canonicalize_assessment_url(url) == expected


def test_canonicalize_url_lists_and_overlap():
    predictions = [["https://www.shl.com/products/product-catalog/view/python-new/"]]
    ground_truth = [["https://www.shl.com/solutions/products/product-catalog/view/python-new"]]

    c_pred = canonicalize_url_lists(predictions)
    c_gt = canonicalize_url_lists(ground_truth)
    pred_count, gt_count, overlap = unique_url_overlap(c_pred, c_gt)

    assert pred_count == 1
    assert gt_count == 1
    assert overlap == 1
