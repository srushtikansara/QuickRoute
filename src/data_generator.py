"""Synthetic delivery dataset generator (reproducible via a random seed)."""

import numpy as np
import pandas as pd

COLUMNS = [
    "delivery_id",
    "location_name",
    "x_coordinate",
    "y_coordinate",
    "priority",
    "service_time_minutes",
    "delivery_deadline_minutes",
    "package_size",
    "is_depot",
]

MAP_SIZE = 100.0          # the map is a MAP_SIZE x MAP_SIZE square of simulated units
DEFAULT_SPEED = 1.0       # simulated distance units per minute (used to size deadlines)
PACKAGE_SIZES = ["Small", "Medium", "Large"]


def generate_dataset(n_deliveries=15, seed=42, speed=DEFAULT_SPEED):
    """Create a synthetic dataset: one depot (row 0) plus ``n_deliveries`` customers.

    How deadlines are made (so the data is "meaningful"):
      deadline = direct travel time from depot + service time + slack
    where slack = random_fraction * estimated_tour_time.
    High-priority customers (4-5) get a smaller random fraction, i.e. tighter
    deadlines, so that priorities and deadlines genuinely interact.
    """
    if not isinstance(n_deliveries, (int, np.integer)) or n_deliveries < 1:
        raise ValueError("n_deliveries must be an integer >= 1")
    if speed <= 0:
        raise ValueError("speed must be positive")

    rng = np.random.default_rng(seed)

    depot_xy = np.array([MAP_SIZE / 2, MAP_SIZE / 2])
    customer_xy = rng.uniform(5, MAP_SIZE - 5, size=(n_deliveries, 2)).round(1)
    priorities = rng.integers(1, 6, size=n_deliveries)
    service_times = rng.integers(3, 11, size=n_deliveries)
    package_sizes = rng.choice(PACKAGE_SIZES, size=n_deliveries)

    # Estimated length of a full tour (simple nearest-neighbour walk) to scale slack.
    estimated_tour_time = _nearest_neighbour_tour_length(depot_xy, customer_xy) / speed
    estimated_tour_time += service_times.sum()

    deadlines = []
    for i in range(n_deliveries):
        direct_time = np.linalg.norm(customer_xy[i] - depot_xy) / speed
        if priorities[i] >= 4:
            slack_fraction = rng.uniform(0.10, 0.60)   # tighter deadline
        else:
            slack_fraction = rng.uniform(0.25, 1.10)   # more relaxed deadline
        deadline = direct_time + service_times[i] + slack_fraction * estimated_tour_time
        deadlines.append(int(round(deadline)))

    rows = [
        {
            "delivery_id": 0,
            "location_name": "Central Depot",
            "x_coordinate": float(depot_xy[0]),
            "y_coordinate": float(depot_xy[1]),
            "priority": 1,
            "service_time_minutes": 0,
            "delivery_deadline_minutes": 0,
            "package_size": "None",
            "is_depot": True,
        }
    ]
    for i in range(n_deliveries):
        rows.append(
            {
                "delivery_id": i + 1,
                "location_name": f"Customer {i + 1:02d}",
                "x_coordinate": float(customer_xy[i, 0]),
                "y_coordinate": float(customer_xy[i, 1]),
                "priority": int(priorities[i]),
                "service_time_minutes": int(service_times[i]),
                "delivery_deadline_minutes": deadlines[i],
                "package_size": str(package_sizes[i]),
                "is_depot": False,
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def _nearest_neighbour_tour_length(depot_xy, customer_xy):
    """Length of a quick nearest-neighbour closed tour (only used to size deadlines)."""
    remaining = list(range(len(customer_xy)))
    current = depot_xy
    total = 0.0
    while remaining:
        distances = [np.linalg.norm(customer_xy[j] - current) for j in remaining]
        k = int(np.argmin(distances))
        total += distances[k]
        current = customer_xy[remaining.pop(k)]
    return total + float(np.linalg.norm(current - depot_xy))
