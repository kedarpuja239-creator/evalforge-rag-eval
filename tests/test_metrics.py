from evalkit.metrics import hit_at_k, mean, recall_at_k, reciprocal_rank


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 2) == 0.5
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 3) == 1.0
    assert recall_at_k([], {"a"}, 5) == 0.0
    assert recall_at_k(["a"], set(), 5) == 0.0


def test_hit_at_k():
    assert hit_at_k(["x", "a"], {"a"}, 1) == 0.0
    assert hit_at_k(["x", "a"], {"a"}, 2) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == 1 / 3
    assert reciprocal_rank(["a"], {"a"}) == 1.0
    assert reciprocal_rank(["x"], {"a"}) == 0.0


def test_mrr_is_the_mean_of_reciprocal_ranks():
    # correct page at rank 4 for one question and rank 1 for another
    assert mean([1 / 4, 1.0]) == 0.625


def test_mean_of_nothing_is_zero():
    assert mean([]) == 0.0
