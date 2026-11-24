from evalkit.gate_redteam import pass_counts


def test_uses_stats_when_present():
    assert pass_counts({"results": {"stats": {"successes": 9, "failures": 1}}}) == (9, 10)


def test_falls_back_to_row_success_flags():
    report = {"results": {"results": [{"success": True}, {"success": False}, {"success": True}]}}
    assert pass_counts(report) == (2, 3)


def test_empty_report():
    assert pass_counts({}) == (0, 0)
