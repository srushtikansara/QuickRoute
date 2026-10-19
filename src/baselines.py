"""Solvers compared in the project.

Every solver returns a SolverOutput so that all methods are evaluated
with the same route assumptions and the same hybrid evaluation objective.
"""

import itertools
import time
from dataclasses import dataclass, replace

import numpy as np

from .aco import ACOParams, run_aco
from .route_evaluator import (DISTANCE_ONLY, ObjectiveConfig, ParameterError,
                              evaluate_route, make_cost_function)

NEAREST_NEIGHBOR = "Nearest Neighbor"
ACO_DISTANCE = "ACO Distance-Only"
ACO_HYBRID = "Hybrid ACO + Fuzzy"
BRUTE_FORCE = "Brute Force (optimal)"
MAX_BRUTE_FORCE_DELIVERIES = 8


@dataclass
class SolverOutput:
    method: str
    delivery_order: list
    runtime_seconds: float
    history: object = None      # DataFrame for ACO methods, otherwise None
    optimiser_cost: float = None


def nearest_neighbor(problem):
    """Greedy: from the current location always drive to the closest unvisited customer."""
    start = time.perf_counter()
    unvisited = problem.delivery_nodes
    order, current = [], problem.depot
    while unvisited:
        next_node = min(unvisited, key=lambda node: (problem.distance_matrix[current, node], node))
        order.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    return SolverOutput(NEAREST_NEIGHBOR, order, time.perf_counter() - start)


def aco_distance_only(problem, params, fuzzy_system, evaluation_config, progress_callback=None):
    """ACO whose cost is distance only (w_d = 1, w_p = 0).

    The best route is later *evaluated* with the hybrid configuration so that it can be
    compared fairly; the hybrid objective is also logged per iteration for plotting.
    """
    cost = make_cost_function(problem, DISTANCE_ONLY, fuzzy_system)
    report = make_cost_function(problem, evaluation_config, fuzzy_system)
    result = run_aco(problem.distance_matrix, cost, params, problem.depot,
                     progress_callback, report_function=report)
    return SolverOutput(ACO_DISTANCE, result.best_order, result.runtime_seconds,
                        result.history, result.best_cost)


def aco_hybrid(problem, params, fuzzy_system, config, progress_callback=None):
    """ACO whose cost is the hybrid objective J = w_d*D_norm + w_p*P_norm."""
    cost = make_cost_function(problem, config, fuzzy_system)
    result = run_aco(problem.distance_matrix, cost, params, problem.depot, progress_callback)
    return SolverOutput(ACO_HYBRID, result.best_order, result.runtime_seconds,
                        result.history, result.best_cost)


def brute_force(problem, fuzzy_system, config, max_deliveries=MAX_BRUTE_FORCE_DELIVERIES):
    """Try every possible order (n! routes) and keep the one with the lowest hybrid objective.

    Only feasible for very small instances, so it is limited to `max_deliveries`.
    """
    if problem.n_deliveries > max_deliveries:
        raise ParameterError(f"Brute force is limited to {max_deliveries} deliveries "
                             f"(this instance has {problem.n_deliveries}).")
    start = time.perf_counter()
    cost = make_cost_function(problem, config, fuzzy_system)
    best_order, best_cost = None, np.inf
    for permutation in itertools.permutations(problem.delivery_nodes):
        value = cost(permutation)
        if value < best_cost:
            best_cost, best_order = value, list(permutation)
    return SolverOutput(BRUTE_FORCE, best_order, time.perf_counter() - start, None, best_cost)


def solve(method, problem, params, fuzzy_system, config, progress_callback=None):
    """Dispatch by method name (used by the dashboard and the experiments)."""
    if method == NEAREST_NEIGHBOR:
        return nearest_neighbor(problem)
    if method == ACO_DISTANCE:
        return aco_distance_only(problem, params, fuzzy_system, config, progress_callback)
    if method == ACO_HYBRID:
        return aco_hybrid(problem, params, fuzzy_system, config, progress_callback)
    if method == BRUTE_FORCE:
        return brute_force(problem, fuzzy_system, config)
    raise ValueError(f"Unknown method: {method}")


def with_seed(params, seed):
    return replace(params, seed=int(seed))
