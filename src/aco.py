"""Ant Colony Optimization (Ant System with an optional elitist boost), from scratch.

The optimiser is generic: it only needs
  * the distance matrix (for the heuristic  eta = 1 / distance), and
  * a cost function  cost(delivery_order) -> float  (lower is better).
Distance-only ACO and hybrid ACO + Fuzzy Logic differ ONLY in the cost function.

Main formulas
  Transition probability (ant at i, feasible set F):
      P(i,j) = tau(i,j)^alpha * eta(i,j)^beta / sum_{k in F} tau(i,k)^alpha * eta(i,k)^beta
  Pheromone update:
      tau(i,j) <- (1 - rho) * tau(i,j) + delta_tau(i,j)
      delta_tau(i,j) = sum over ants that used edge (i,j) of  Q / cost(route)
                       (+ elitist_weight * Q / cost(best-so-far) on the best route's edges)
"""

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .route_evaluator import ParameterError

EPSILON = 1e-9              # prevents division by zero in the heuristic
TAU_MIN, TAU_MAX = 1e-6, 1e6   # keeps pheromone numerically stable


@dataclass
class ACOParams:
    n_ants: int = 20
    n_iterations: int = 100
    alpha: float = 1.0           # pheromone influence
    beta: float = 3.0            # distance-heuristic influence
    evaporation_rate: float = 0.3   # rho
    initial_pheromone: float = 1.0
    deposit_constant: float = 1.0   # Q
    elitist_weight: float = 2.0     # extra reinforcement of the best-so-far route (0 = off)
    seed: int = 0

    def validate(self):
        if int(self.n_ants) != self.n_ants or self.n_ants < 1:
            raise ParameterError("Number of ants must be an integer >= 1.")
        if int(self.n_iterations) != self.n_iterations or self.n_iterations < 1:
            raise ParameterError("Number of iterations must be an integer >= 1.")
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ParameterError("alpha must be a number >= 0.")
        if not np.isfinite(self.beta) or self.beta < 0:
            raise ParameterError("beta must be a number >= 0.")
        if not (0 < self.evaporation_rate <= 1):
            raise ParameterError("Evaporation rate must be in the range (0, 1].")
        if not np.isfinite(self.initial_pheromone) or self.initial_pheromone <= 0:
            raise ParameterError("Initial pheromone must be a positive number.")
        if not np.isfinite(self.deposit_constant) or self.deposit_constant <= 0:
            raise ParameterError("Deposit constant Q must be a positive number.")
        if not np.isfinite(self.elitist_weight) or self.elitist_weight < 0:
            raise ParameterError("Elitist weight must be >= 0.")
        if int(self.seed) != self.seed:
            raise ParameterError("Random seed must be an integer.")
        return self


@dataclass
class ACOResult:
    best_order: list
    best_cost: float
    history: pd.DataFrame
    runtime_seconds: float
    params: ACOParams
    final_pheromone: np.ndarray = field(repr=False, default=None)


def build_heuristic_matrix(distance_matrix):
    """eta(i,j) = 1 / (distance + epsilon); the diagonal is set to 0 (no self-transition)."""
    eta = 1.0 / (distance_matrix + EPSILON)
    np.fill_diagonal(eta, 0.0)
    return eta


def choose_next_node(current, unvisited, pheromone, eta, alpha, beta, rng):
    """Roulette-wheel selection of the next node among the feasible (unvisited) ones."""
    candidates = np.array(unvisited)
    # attractiveness = tau^alpha * eta^beta  (the numerator of P(i,j))
    attractiveness = (pheromone[current, candidates] ** alpha) * (eta[current, candidates] ** beta)
    total = attractiveness.sum()
    if not np.isfinite(total) or total <= 0:
        probabilities = np.full(len(candidates), 1.0 / len(candidates))   # safe fallback
    else:
        probabilities = attractiveness / total
    return int(rng.choice(candidates, p=probabilities))


def construct_route(depot, delivery_nodes, pheromone, eta, alpha, beta, rng):
    """One ant builds a complete delivery order, visiting every customer exactly once."""
    unvisited = list(delivery_nodes)
    order = []
    current = depot
    while unvisited:
        next_node = choose_next_node(current, unvisited, pheromone, eta, alpha, beta, rng)
        order.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    return order


def run_aco(distance_matrix, cost_function, params, depot=0, progress_callback=None,
            report_function=None):
    """Run ACO and return an ACOResult.

    cost_function(order)   -> value to minimise (drives the pheromone update)
    report_function(order) -> optional second measure of the best-so-far route that is
                              logged each iteration (lets different methods be plotted
                              on the same objective).
    progress_callback(iteration, total, best_cost) is called after every iteration.
    """
    params.validate()
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    n_nodes = distance_matrix.shape[0]
    delivery_nodes = [n for n in range(n_nodes) if n != depot]
    if not delivery_nodes:
        raise ParameterError("At least one delivery location is required.")

    start_time = time.perf_counter()
    rng = np.random.default_rng(int(params.seed))

    eta = build_heuristic_matrix(distance_matrix)
    pheromone = np.full((n_nodes, n_nodes), float(params.initial_pheromone))
    np.fill_diagonal(pheromone, 0.0)                     # no self-transitions

    # With one delivery there is only one possible route.
    best_order, best_cost = None, np.inf
    history_rows = []

    for iteration in range(1, params.n_iterations + 1):
        # ---- every ant builds a route and is scored ------------------------------
        iteration_routes, iteration_costs = [], []
        for _ in range(params.n_ants):
            order = construct_route(depot, delivery_nodes, pheromone, eta,
                                    params.alpha, params.beta, rng)
            iteration_routes.append(order)
            iteration_costs.append(cost_function(order))

        best_in_iteration = int(np.argmin(iteration_costs))
        if iteration_costs[best_in_iteration] < best_cost:
            best_cost = float(iteration_costs[best_in_iteration])
            best_order = list(iteration_routes[best_in_iteration])

        # ---- pheromone update: evaporation, then deposition -------------------------
        pheromone *= (1.0 - params.evaporation_rate)
        for order, cost in zip(iteration_routes, iteration_costs):
            _deposit(pheromone, depot, order, params.deposit_constant / max(cost, EPSILON))
        if params.elitist_weight > 0:
            _deposit(pheromone, depot, best_order,
                     params.elitist_weight * params.deposit_constant / max(best_cost, EPSILON))
        np.clip(pheromone, TAU_MIN, TAU_MAX, out=pheromone)
        np.fill_diagonal(pheromone, 0.0)

        history_rows.append({
            "iteration": iteration,
            "best_cost": best_cost,
            "iteration_best_cost": float(iteration_costs[best_in_iteration]),
            "iteration_mean_cost": float(np.mean(iteration_costs)),
            "best_reported": float(report_function(best_order)) if report_function else best_cost,
        })
        if progress_callback is not None:
            progress_callback(iteration, params.n_iterations, best_cost)

    return ACOResult(best_order, best_cost, pd.DataFrame(history_rows),
                     time.perf_counter() - start_time, params, pheromone)


def _deposit(pheromone, depot, order, amount):
    """Add `amount` pheromone on every edge of the closed tour (symmetric problem)."""
    tour = [depot] + list(order) + [depot]
    for i, j in zip(tour[:-1], tour[1:]):
        pheromone[i, j] += amount
        pheromone[j, i] += amount
