"""Small helpers shared by the tests."""

import pandas as pd

from src.data_validation import clean_dataframe
from src.route_evaluator import RoutingProblem


def make_problem(customers, speed=1.0, depot_xy=(0.0, 0.0)):
    """Build a RoutingProblem from a list of tuples:
    (x, y, priority, service_time, deadline)."""
    rows = [{"delivery_id": 0, "location_name": "Depot", "x_coordinate": depot_xy[0],
             "y_coordinate": depot_xy[1], "priority": 1, "service_time_minutes": 0,
             "delivery_deadline_minutes": 0, "package_size": "None", "is_depot": True}]
    for i, (x, y, priority, service, deadline) in enumerate(customers, start=1):
        rows.append({"delivery_id": i, "location_name": f"C{i}", "x_coordinate": x,
                     "y_coordinate": y, "priority": priority, "service_time_minutes": service,
                     "delivery_deadline_minutes": deadline, "package_size": "Small",
                     "is_depot": False})
    return RoutingProblem.from_dataframe(clean_dataframe(pd.DataFrame(rows)), speed)
