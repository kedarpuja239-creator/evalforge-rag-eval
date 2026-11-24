from evalkit.gate import check


def test_passes_when_above_floor_and_close_to_baseline():
    assert check({"mrr": 0.90}, {"mrr": 0.5}, {"mrr": 0.91}) == []


def test_fails_below_floor():
    failures = check({"mrr": 0.4}, {"mrr": 0.5}, {})
    assert len(failures) == 1 and "floor" in failures[0]


def test_fails_when_metric_missing():
    assert "missing" in check({}, {"mrr": 0.5}, {})[0]


def test_fails_on_regression_beyond_max_drop():
    failures = check({"recall@5": 0.80}, {}, {"recall@5": 0.90})
    assert len(failures) == 1 and "regressed" in failures[0]


def test_small_drop_is_tolerated():
    assert check({"recall@5": 0.88}, {}, {"recall@5": 0.90}) == []
