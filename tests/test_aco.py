import numpy as np
import pytest

from src.aco import (ACOParams, build_heuristic_matrix, choose_next_node, construct_route, run_aco)
from src.baselines import (ACO_DISTANCE, ACO_HYBRID, NEAREST_NEIGHBOR, brute_force, nearest_neighbor,
                           solve)
from src.data_generator import generate_dataset
from src.data_validation import clean_dataframe
from src.fuzzy_system import FuzzyPenaltySystem
from src.metrics import compare_methods, improvement_percent
from src.route_evaluator import (DISTANCE_ONLY, ObjectiveConfig, ParameterError, RoutingProblem,
                                 evaluate_route, make_cost_function)
from tests.helpers import make_problem

FUZZY = FuzzyPenaltySystem()
FAST = ACOParams(n_ants=10, n_iterations=15, seed=3)


def generated_problem(n, seed=5):
    return RoutingProblem.from_dataframe(clean_dataframe(generate_dataset(n, seed)), 1.0)


@pytest.mark.parametrize("n", [1, 2, 3, 8, 15])
def test_aco_returns_valid_route_visiting_each_delivery_once(n):
    problem = generated_problem(n)
    cost = make_cost_function(problem, ObjectiveConfig(), FUZZY)
    result = run_aco(problem.distance_matrix, cost, FAST)
    assert sorted(result.best_order) == problem.delivery_nodes
    assert 0 not in result.best_order
    full = evaluate_route(problem, result.best_order, ObjectiveConfig(), FUZZY).full_route
    assert full[0] == 0 and full[-1] == 0
    assert np.isfinite(result.best_cost)


def test_aco_is_reproducible_with_the_same_seed():
    problem = generated_problem(10)
    cost = make_cost_function(problem, ObjectiveConfig(), FUZZY)
    a = run_aco(problem.distance_matrix, cost, FAST)
    b = run_aco(problem.distance_matrix, cost, FAST)
    assert a.best_order == b.best_order and a.best_cost == b.best_cost


def test_best_so_far_cost_never_increases():
    problem = generated_problem(10)
    cost = make_cost_function(problem, ObjectiveConfig(), FUZZY)
    history = run_aco(problem.distance_matrix, cost, FAST).history
    assert (history["best_cost"].diff().dropna() <= 1e-12).all()
    assert len(history) == FAST.n_iterations


def test_heuristic_has_no_self_transition_and_no_division_by_zero():
    eta = build_heuristic_matrix(np.array([[0.0, 2.0], [2.0, 0.0]]))
    assert eta[0, 0] == 0 and eta[1, 1] == 0
    assert eta[0, 1] == pytest.approx(0.5, rel=1e-6)
    duplicate_points = build_heuristic_matrix(np.zeros((3, 3)))   # all distances 0
    assert np.all(np.isfinite(duplicate_points))


def test_transition_probability_follows_the_formula():
    """With alpha=1, beta=1 and equal pheromone, the closer node must be chosen more often."""
    distance = np.array([[0, 1, 4], [1, 0, 3], [4, 3, 0]], dtype=float)
    eta = build_heuristic_matrix(distance)
    pheromone = np.ones((3, 3))
    rng = np.random.default_rng(0)
    picks = [choose_next_node(0, [1, 2], pheromone, eta, 1.0, 1.0, rng) for _ in range(4000)]
    share_of_node_1 = picks.count(1) / len(picks)
    assert share_of_node_1 == pytest.approx((1 / 1) / (1 / 1 + 1 / 4), abs=0.03)   # = 0.8


def test_construct_route_visits_every_node_once():
    problem = generated_problem(9)
    eta = build_heuristic_matrix(problem.distance_matrix)
    pheromone = np.ones_like(problem.distance_matrix)
    order = construct_route(0, problem.delivery_nodes, pheromone, eta, 1.0, 2.0, np.random.default_rng(1))
    assert sorted(order) == problem.delivery_nodes


def test_invalid_parameters_raise_clear_errors():
    bad = [dict(n_ants=0), dict(n_iterations=0), dict(alpha=-1), dict(beta=-0.5),
           dict(evaporation_rate=0), dict(evaporation_rate=1.5), dict(initial_pheromone=0),
           dict(deposit_constant=-1), dict(n_ants=2.5)]
    for kwargs in bad:
        with pytest.raises(ParameterError):
            ACOParams(**kwargs).validate()
    problem = generated_problem(4)
    with pytest.raises(ParameterError):
        run_aco(problem.distance_matrix, lambda order: 1.0, ACOParams(n_ants=0))


def test_aco_needs_at_least_one_delivery():
    with pytest.raises(ParameterError):
        run_aco(np.zeros((1, 1)), lambda order: 1.0, FAST)


def test_distance_only_aco_beats_random_routes_on_average():
    problem = generated_problem(14, seed=11)
    cost = make_cost_function(problem, DISTANCE_ONLY, FUZZY)
    rng = np.random.default_rng(0)
    random_costs = [cost(list(rng.permutation(problem.delivery_nodes))) for _ in range(200)]
    result = run_aco(problem.distance_matrix, cost, ACOParams(n_ants=15, n_iterations=40, seed=0))
    assert result.best_cost < np.mean(random_costs)


def test_aco_matches_brute_force_on_a_tiny_instance():
    problem = generated_problem(6, seed=2)
    config = ObjectiveConfig()
    optimum = brute_force(problem, FUZZY, config)
    output = solve(ACO_HYBRID, problem, ACOParams(n_ants=20, n_iterations=60, seed=1), FUZZY, config)
    cost = make_cost_function(problem, config, FUZZY)
    assert cost(output.delivery_order) <= cost(optimum.delivery_order) + 1e-9


def test_brute_force_is_limited_to_small_instances():
    with pytest.raises(ParameterError):
        brute_force(generated_problem(10), FUZZY, ObjectiveConfig())


def test_nearest_neighbor_is_greedy_and_valid():
    problem = make_problem([(10, 0, 3, 1, 99), (3, 0, 3, 1, 99), (6, 0, 3, 1, 99)])
    assert nearest_neighbor(problem).delivery_order == [2, 3, 1]


def test_improvement_formula():
    assert improvement_percent(100, 80) == pytest.approx(20.0)     # lower is better
    assert improvement_percent(100, 120) == pytest.approx(-20.0)
    assert improvement_percent(100, 120, lower_is_better=False) == pytest.approx(20.0)
    assert np.isnan(improvement_percent(0, 5))


def test_compare_methods_runs_and_reports_mean_and_std():
    problem = generated_problem(7)
    outcome = compare_methods(problem, FAST, FUZZY, ObjectiveConfig(), [0, 1, 2])
    summary = outcome["summary"].set_index("method")
    assert summary.loc[NEAREST_NEIGHBOR, "n_runs"] == 1
    assert summary.loc[ACO_DISTANCE, "n_runs"] == 3
    assert summary.loc[ACO_HYBRID, "objective_std"] >= 0
    assert len(outcome["improvements"]) == 3 + 1  # 2 methods vs NN, hybrid vs distance-only, NN vs distance-only
    with pytest.raises(ValueError):
        compare_methods(problem, FAST, FUZZY, ObjectiveConfig(), [])


def test_all_methods_work_with_a_single_delivery():
    problem = make_problem([(5, 5, 3, 2, 60)])
    for method in (NEAREST_NEIGHBOR, ACO_DISTANCE, ACO_HYBRID):
        output = solve(method, problem, FAST, FUZZY, ObjectiveConfig())
        assert output.delivery_order == [1]
