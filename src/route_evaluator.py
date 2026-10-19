"""Route timing and the hybrid objective function.

A route is a list of node indices that starts and ends at the depot (node 0):
    [0, 4, 2, 7, 0]
The *delivery order* is the same list without the two depot entries: [4, 2, 7].

TIMING MODEL (single vehicle, closed tour)
    The vehicle leaves the depot at time 0.
    arrival_j   = departure_i + distance(i, j) / speed
    departure_j = arrival_j + service_time_j
    lateness_j  = max(0, arrival_j - deadline_j)            (minutes)
    signed_j    = arrival_j - deadline_j                    (fed to the fuzzy system)

OBJECTIVE
    J = w_d * D / D_ref  +  w_p * P / P_ref
    D      = total route distance (simulated distance units)
    P      = sum of fuzzy penalties of all deliveries (each in 0..100)
    D_ref  = "star tour" length = 2 * sum(distance(depot, customer)).  It is fixed
             for a dataset, so every candidate route is normalised the same way.
    P_ref  = 100 * number_of_deliveries  (the largest possible total penalty).
"""

from dataclasses import dataclass, field

import numpy as np

from .distance import euclidean_distance_matrix
from .fuzzy_system import PENALTY_MAX, FuzzyPenaltySystem


class ParameterError(ValueError):
    """Raised for invalid algorithm or objective parameters."""


@dataclass
class RoutingProblem:
    """All the static data of one routing instance. Node 0 is always the depot."""
    names: list
    ids: list
    coordinates: np.ndarray
    priorities: np.ndarray
    service_times: np.ndarray
    deadlines: np.ndarray
    distance_matrix: np.ndarray
    speed: float = 1.0

    @classmethod
    def from_dataframe(cls, df, speed=1.0):
        """Build a problem from a *cleaned* DataFrame (depot first, see data_validation)."""
        if speed <= 0:
            raise ParameterError("Vehicle speed must be positive.")
        if not bool(df.iloc[0]["is_depot"]) or int(df["is_depot"].sum()) != 1:
            raise ParameterError("The first row must be the single depot row. Use clean_dataframe() first.")
        coords = df[["x_coordinate", "y_coordinate"]].to_numpy(dtype=float)
        return cls(
            names=df["location_name"].tolist(),
            ids=df["delivery_id"].tolist(),
            coordinates=coords,
            priorities=df["priority"].to_numpy(dtype=float),
            service_times=df["service_time_minutes"].to_numpy(dtype=float),
            deadlines=df["delivery_deadline_minutes"].to_numpy(dtype=float),
            distance_matrix=euclidean_distance_matrix(coords),
            speed=float(speed),
        )

    @property
    def n_deliveries(self):
        return len(self.names) - 1

    @property
    def depot(self):
        return 0

    @property
    def delivery_nodes(self):
        return list(range(1, len(self.names)))

    @property
    def distance_reference(self):
        """Fixed normalisation scale for distance (star-tour length)."""
        star_tour = 2.0 * float(self.distance_matrix[0, 1:].sum())
        return star_tour if star_tour > 1e-9 else 1.0

    @property
    def penalty_reference(self):
        """Fixed normalisation scale for the summed penalty."""
        return PENALTY_MAX * self.n_deliveries

    def with_speed(self, speed):
        if speed <= 0:
            raise ParameterError("Vehicle speed must be positive.")
        return RoutingProblem(self.names, self.ids, self.coordinates, self.priorities,
                              self.service_times, self.deadlines, self.distance_matrix, float(speed))


@dataclass
class ObjectiveConfig:
    """Weights of the hybrid objective J = w_distance * D_norm + w_penalty * P_norm."""
    w_distance: float = 0.5
    w_penalty: float = 0.5

    def validate(self):
        for name, value in (("w_distance", self.w_distance), ("w_penalty", self.w_penalty)):
            if not np.isfinite(value) or value < 0:
                raise ParameterError(f"{name} must be a non-negative number (got {value}).")
        if abs(self.w_distance + self.w_penalty - 1.0) > 1e-6:
            raise ParameterError("w_distance + w_penalty must equal 1 "
                                 f"(got {self.w_distance + self.w_penalty:.4f}).")
        return self


DISTANCE_ONLY = ObjectiveConfig(1.0, 0.0)


@dataclass
class RouteResult:
    """Full evaluation of one route (every number the dashboard reports)."""
    full_route: list
    delivery_order: list
    total_distance: float
    total_travel_time: float
    total_service_time: float
    total_time: float
    arrivals: dict = field(default_factory=dict)        # node -> arrival time
    lateness: dict = field(default_factory=dict)        # node -> max(0, arrival - deadline)
    signed_lateness: dict = field(default_factory=dict)  # node -> arrival - deadline
    penalties: dict = field(default_factory=dict)       # node -> fuzzy penalty (0..100)
    total_lateness: float = 0.0
    n_late: int = 0
    pct_late: float = 0.0
    priority_weighted_lateness: float = 0.0
    total_penalty: float = 0.0
    normalized_distance: float = 0.0
    normalized_penalty: float = 0.0
    weighted_distance_term: float = 0.0
    weighted_penalty_term: float = 0.0
    objective: float = 0.0


def full_route_from_order(delivery_order, depot=0):
    """Add the depot at the start and the end: closed tour."""
    return [depot] + list(delivery_order) + [depot]


def check_valid_order(problem, delivery_order):
    """Raise ValueError unless every delivery is visited exactly once."""
    expected = sorted(problem.delivery_nodes)
    if sorted(delivery_order) != expected:
        raise ValueError("A route must visit every delivery exactly once "
                         f"(expected nodes {expected}, got {list(delivery_order)}).")


def evaluate_route(problem, delivery_order, config, fuzzy_system):
    """Simulate the route and compute distance, delays, fuzzy penalties and J."""
    config.validate()
    check_valid_order(problem, delivery_order)

    dist = problem.distance_matrix
    speed = problem.speed
    full_route = full_route_from_order(delivery_order, problem.depot)

    clock = 0.0                      # vehicle starts at time zero
    total_distance = 0.0
    total_service = 0.0
    arrivals, lateness, signed, penalties = {}, {}, {}, {}

    for previous_node, node in zip(full_route[:-1], full_route[1:]):
        leg = dist[previous_node, node]
        total_distance += leg
        clock += leg / speed                     # arrival time at `node`
        if node == problem.depot:
            continue                             # final return to the depot has no deadline
        arrivals[node] = clock
        signed[node] = clock - problem.deadlines[node]
        lateness[node] = max(0.0, signed[node])
        penalties[node] = fuzzy_system.penalty(problem.priorities[node], signed[node])
        clock += problem.service_times[node]     # departure time
        total_service += problem.service_times[node]

    n_late = sum(1 for value in lateness.values() if value > 1e-9)
    total_lateness = float(sum(lateness.values()))
    total_penalty = float(sum(penalties.values()))
    norm_distance = total_distance / problem.distance_reference
    norm_penalty = total_penalty / problem.penalty_reference
    weighted_distance = config.w_distance * norm_distance
    weighted_penalty = config.w_penalty * norm_penalty

    return RouteResult(
        full_route=full_route,
        delivery_order=list(delivery_order),
        total_distance=float(total_distance),
        total_travel_time=float(total_distance / speed),
        total_service_time=float(total_service),
        total_time=float(clock),
        arrivals=arrivals,
        lateness=lateness,
        signed_lateness=signed,
        penalties=penalties,
        total_lateness=total_lateness,
        n_late=n_late,
        pct_late=100.0 * n_late / problem.n_deliveries,
        priority_weighted_lateness=float(sum(problem.priorities[n] * v for n, v in lateness.items())),
        total_penalty=total_penalty,
        normalized_distance=float(norm_distance),
        normalized_penalty=float(norm_penalty),
        weighted_distance_term=float(weighted_distance),
        weighted_penalty_term=float(weighted_penalty),
        objective=float(weighted_distance + weighted_penalty),
    )


def make_cost_function(problem, config, fuzzy_system):
    """Return cost(delivery_order) -> objective value J (lower is better)."""
    config.validate()

    def cost(delivery_order):
        return evaluate_route(problem, delivery_order, config, fuzzy_system).objective

    return cost
