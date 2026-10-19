import itertools

import numpy as np
import pytest

from src.fuzzy_system import (LATENESS_RANGE, PENALTY_RANGE, PRIORITY_RANGE, PRIORITY_TERMS,
                              RULES, TIGHTNESS_TERMS, FuzzyPenaltySystem, Trapezoid)


@pytest.fixture(scope="module")
def fuzzy():
    return FuzzyPenaltySystem()


def test_output_is_finite_and_in_range_everywhere(fuzzy):
    for priority in np.linspace(*PRIORITY_RANGE, 9):
        for lateness in np.linspace(*LATENESS_RANGE, 37):
            value = fuzzy.penalty(priority, lateness)
            assert np.isfinite(value)
            assert PENALTY_RANGE[0] <= value <= PENALTY_RANGE[1]


def test_inputs_outside_universe_are_clipped(fuzzy):
    assert fuzzy.penalty(99, 10_000) == pytest.approx(fuzzy.penalty(5, LATENESS_RANGE[1]))
    assert fuzzy.penalty(-5, -10_000) == pytest.approx(fuzzy.penalty(1, LATENESS_RANGE[0]))


def test_rule_base_is_complete():
    expected = set(itertools.product(PRIORITY_TERMS, TIGHTNESS_TERMS))
    assert set(RULES) == expected          # every combination has exactly one rule


def test_every_input_activates_at_least_one_rule(fuzzy):
    for priority in np.linspace(*PRIORITY_RANGE, 9):
        for lateness in np.linspace(*LATENESS_RANGE, 37):
            result = fuzzy.infer(priority, lateness)
            assert max(result.rule_strengths.values()) > 0


# Centroid defuzzification of clipped shapes is not perfectly monotone: when two rules with
# the same output fire at different strengths the centroid can wobble slightly. The wobble is
# measured (peak-to-valley) below and bounded; the overall trends must still be monotone.
MAX_RIPPLE = 7.0


def _peak_to_valley(values):
    values = np.asarray(values)
    return float((np.maximum.accumulate(values) - values).max())


def test_penalty_trend_increases_with_lateness(fuzzy):
    for priority in np.linspace(1, 5, 17):
        values = [fuzzy.penalty(priority, x) for x in np.arange(-60, 120.5, 1)]
        assert _peak_to_valley(values) < MAX_RIPPLE
        assert values[-1] > values[0]
        # Clearly late is always penalised more than clearly early.
        assert fuzzy.penalty(priority, 60) > fuzzy.penalty(priority, -45)


def test_penalty_trend_increases_with_priority(fuzzy):
    for lateness in (-20, 0, 15, 40, 90):
        values = [fuzzy.penalty(p, lateness) for p in np.linspace(1, 5, 41)]
        assert _peak_to_valley(values) < MAX_RIPPLE
        assert fuzzy.penalty(5, lateness) >= fuzzy.penalty(1, lateness)


def test_same_delay_costs_more_for_high_priority(fuzzy):
    for lateness in (0, 15, 40, 90):
        assert fuzzy.penalty(5, lateness) > fuzzy.penalty(1, lateness) + 10


def test_documented_example_rules(fuzzy):
    low_comfortable = fuzzy.penalty(1, -60)
    high_tight = fuzzy.penalty(5, 0)
    high_critical = fuzzy.penalty(5, 100)
    assert low_comfortable < 20            # Low
    assert 60 < high_tight < 80            # High
    assert high_critical > 85              # Very high


def test_positive_lateness_means_late_membership(fuzzy):
    late = fuzzy.infer(3, 30)
    early = fuzzy.infer(3, -50)
    assert late.tightness_memberships["Critical"] > 0
    assert early.tightness_memberships["Critical"] == 0
    assert early.tightness_memberships["Comfortable"] == 1


def test_trapezoid_shapes():
    triangle = Trapezoid(0, 5, 5, 10)
    assert triangle.value(0) == 0 and triangle.value(5) == 1 and triangle.value(10) == 0
    assert triangle.value(2.5) == pytest.approx(0.5)
    left_shoulder = Trapezoid(0, 0, 5, 10)
    assert left_shoulder.value(0) == 1
    assert float(left_shoulder(np.array([2.0]))[0]) == 1.0
    right_shoulder = Trapezoid(0, 5, 10, 10)
    assert right_shoulder.value(10) == 1


def test_explanation_mentions_penalty(fuzzy):
    result = fuzzy.infer(4, 10)
    text = " ".join(fuzzy.explain(result))
    assert f"{result.penalty:.1f}" in text
