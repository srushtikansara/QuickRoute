import itertools
import math

import pytest

from src.fuzzy_system import FuzzyPenaltySystem
from src.route_evaluator import (DISTANCE_ONLY, ObjectiveConfig, ParameterError, evaluate_route,
                                 full_route_from_order, make_cost_function)
from tests.helpers import make_problem

FUZZY = FuzzyPenaltySystem()

# depot (0,0); A = node 1 at (10,0); B = node 2 at (10,10). Speed 2 su/min.
SIMPLE = dict(customers=[(10, 0, 5, 5, 4), (10, 10, 1, 3, 20)], speed=2.0)


def simple_problem():
    return make_problem(**SIMPLE)


def test_depot_is_first_and_last_in_closed_route():
    route = full_route_from_order([2, 1])
    assert route[0] == 0 and route[-1] == 0
    result = evaluate_route(simple_problem(), [1, 2], ObjectiveConfig(), FUZZY)
    assert result.full_route == [0, 1, 2, 0]


def test_arrival_times_and_lateness_hand_calculation():
    problem = simple_problem()
    result = evaluate_route(problem, [1, 2], ObjectiveConfig(), FUZZY)
    # depot -> A: 10 su / 2 = 5 min arrival; deadline 4 => 1 min late.
    assert result.arrivals[1] == pytest.approx(5.0)
    assert result.lateness[1] == pytest.approx(1.0)
    assert result.signed_lateness[1] == pytest.approx(1.0)
    # leave A at 5 + 5 (service) = 10, A -> B 10 su / 2 = 5 min => arrival 15; deadline 20 => on time.
    assert result.arrivals[2] == pytest.approx(15.0)
    assert result.lateness[2] == 0.0
    assert result.signed_lateness[2] == pytest.approx(-5.0)
    assert result.n_late == 1
    assert result.pct_late == pytest.approx(50.0)
    assert result.total_lateness == pytest.approx(1.0)
    assert result.priority_weighted_lateness == pytest.approx(5 * 1.0)


def test_distance_time_and_service_totals():
    result = evaluate_route(simple_problem(), [1, 2], ObjectiveConfig(), FUZZY)
    expected_distance = 10 + 10 + math.sqrt(200)
    assert result.total_distance == pytest.approx(expected_distance)
    assert result.total_travel_time == pytest.approx(expected_distance / 2)
    assert result.total_service_time == pytest.approx(8.0)
    assert result.total_time == pytest.approx(expected_distance / 2 + 8.0)


def test_each_delivery_must_be_visited_exactly_once():
    problem = simple_problem()
    with pytest.raises(ValueError):
        evaluate_route(problem, [1], ObjectiveConfig(), FUZZY)
    with pytest.raises(ValueError):
        evaluate_route(problem, [1, 1], ObjectiveConfig(), FUZZY)
    with pytest.raises(ValueError):
        evaluate_route(problem, [0, 1, 2], ObjectiveConfig(), FUZZY)


def test_objective_uses_configured_weights():
    problem = simple_problem()
    config = ObjectiveConfig(0.7, 0.3)
    result = evaluate_route(problem, [1, 2], config, FUZZY)
    norm_distance = result.total_distance / (2 * (10 + math.sqrt(200)))
    norm_penalty = result.total_penalty / (100 * 2)
    assert result.normalized_distance == pytest.approx(norm_distance)
    assert result.normalized_penalty == pytest.approx(norm_penalty)
    assert result.objective == pytest.approx(0.7 * norm_distance + 0.3 * norm_penalty)
    assert result.weighted_distance_term + result.weighted_penalty_term == pytest.approx(result.objective)


def test_distance_only_config_ignores_penalty():
    result = evaluate_route(simple_problem(), [1, 2], DISTANCE_ONLY, FUZZY)
    assert result.objective == pytest.approx(result.normalized_distance)


def test_invalid_weights_raise_clear_errors():
    with pytest.raises(ParameterError, match="must equal 1"):
        ObjectiveConfig(0.6, 0.6).validate()
    with pytest.raises(ParameterError, match="non-negative"):
        ObjectiveConfig(-0.2, 1.2).validate()
    with pytest.raises(ParameterError):
        evaluate_route(simple_problem(), [1, 2], ObjectiveConfig(0.9, 0.9), FUZZY)


def test_invalid_speed_raises():
    with pytest.raises(ParameterError):
        make_problem([(1, 1, 3, 1, 10)], speed=0)


def test_normalisation_scales_do_not_depend_on_the_route():
    problem = simple_problem()
    a = evaluate_route(problem, [1, 2], ObjectiveConfig(), FUZZY)
    b = evaluate_route(problem, [2, 1], ObjectiveConfig(), FUZZY)
    assert a.normalized_distance * problem.distance_reference == pytest.approx(a.total_distance)
    assert b.normalized_distance * problem.distance_reference == pytest.approx(b.total_distance)


def test_shorter_route_is_not_automatically_preferred():
    """Customer A (priority 5, deadline 13) sits in the middle of the shortest tour, so in that
    tour A is served after a 30-minute stop and is very late. Distance-only picks that tour;
    with a penalty weight of 0.8 the hybrid objective picks a longer tour that serves A first."""
    problem = make_problem([
        (0, 12, 5, 5, 13),        # node 1 = A: urgent
        (-10, 10, 1, 30, 1000),   # node 2 = B: relaxed, long service time
        (10, 10, 1, 30, 1000),    # node 3 = C: relaxed, long service time
    ])
    config = ObjectiveConfig(0.2, 0.8)
    orders = list(itertools.permutations([1, 2, 3]))
    best_by_distance = min(orders, key=make_cost_function(problem, DISTANCE_ONLY, FUZZY))
    best_by_hybrid = min(orders, key=make_cost_function(problem, config, FUZZY))
    assert best_by_distance[1] == 1            # A in the middle of the shortest tour
    assert best_by_hybrid[0] == 1              # hybrid serves A first
    short = evaluate_route(problem, list(best_by_distance), config, FUZZY)
    urgent_first = evaluate_route(problem, list(best_by_hybrid), config, FUZZY)
    assert urgent_first.total_distance > short.total_distance     # hybrid route is longer ...
    assert urgent_first.total_lateness < short.total_lateness     # ... but much less late
    assert urgent_first.objective < short.objective


def test_single_delivery_edge_case():
    problem = make_problem([(3, 4, 3, 2, 100)])
    result = evaluate_route(problem, [1], ObjectiveConfig(), FUZZY)
    assert result.total_distance == pytest.approx(10.0)
    assert result.n_late == 0
