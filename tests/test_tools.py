import pytest

from rf_agent.tools import ToolInputError


def test_planted_regression_is_detected(tools):
    rows = {r["temperature_c"]: r for r in tools.compare_builds("25.2.0", "25.3.0")["by_temperature"]}
    assert rows[60]["regression"] and rows[60]["delta_db"] > 4
    assert not rows[25]["regression"] and not rows[-20]["regression"]


def test_partial_fix_is_smaller_but_still_a_regression(tools):
    rows = {r["temperature_c"]: r for r in tools.compare_builds("25.2.0", "25.3.1")["by_temperature"]}
    assert 1.0 < rows[60]["delta_db"] < 3.0


def test_summarize_filters_and_pass_rate(tools):
    out = tools.summarize("25.3.0", 60)["rows"]
    assert len(out) == 1 and out[0]["n"] == 20
    assert 0 <= out[0]["pass_rate_pct"] <= 100


@pytest.mark.parametrize("kwargs", [{"build": "25.3"}, {"temperature_c": 61}])
def test_invalid_inputs_list_valid_options(tools, kwargs):
    with pytest.raises(ToolInputError, match="Valid"):
        tools.summarize(**kwargs)


def test_no_write_tools_exposed(tools):
    names = {f.__name__ for f in tools.all()}
    assert names == {"list_options", "summarize", "compare_builds"}
