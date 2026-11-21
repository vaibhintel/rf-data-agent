from rf_agent.grounding import check_grounding, extract_numbers


def test_build_versions_are_not_numeric_claims():
    assert extract_numbers("Build 25.3.0 is 5.1 dB worse at 60 C") == [5.1, 60.0]


def test_grounded_within_tolerance_and_sign_insensitive():
    src = ['{"delta_db": -5.12, "temperature_c": 60}']
    assert check_grounding("worse by 5.1 dB at 60 C", src).grounded


def test_ungrounded_number_is_flagged():
    r = check_grounding("pass rate is 42%", ['{"pass_rate_pct": 15.0}'])
    assert r.ungrounded == [42.0]


def test_numbers_from_question_are_allowed():
    assert check_grounding("At 60 C nothing changed.", [], allowed=(60,)).grounded
