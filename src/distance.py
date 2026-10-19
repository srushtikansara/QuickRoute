"""Distance utilities.

All coordinates live in a simulated 2-D plane. Distances are therefore
"simulated distance units" (su) unless the caller passes an explicit
conversion ``scale`` (for example 0.1 to mean 1 su = 0.1 km).
"""

import numpy as np

DISTANCE_UNIT_LABEL = "simulated distance units (su)"


def euclidean_distance_matrix(coordinates, scale=1.0):
    """Return the symmetric Euclidean distance matrix for an (n, 2) array.

    d(i, j) = scale * sqrt((x_i - x_j)^2 + (y_i - y_j)^2)

    The diagonal is exactly zero and the matrix is symmetric.
    """
    points = np.asarray(coordinates, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("coordinates must be an array of shape (n, 2)")
    if points.shape[0] == 0:
        raise ValueError("coordinates must contain at least one point")
    if not np.all(np.isfinite(points)):
        raise ValueError("coordinates must be finite numbers")
    if scale <= 0:
        raise ValueError("scale must be positive")

    # Broadcasting: differences[i, j] = points[i] - points[j]
    differences = points[:, None, :] - points[None, :, :]
    matrix = scale * np.sqrt((differences ** 2).sum(axis=2))
    np.fill_diagonal(matrix, 0.0)
    return matrix


def route_length(distance_matrix, full_route):
    """Total length of a route given as a list of node indices (depot included)."""
    total = 0.0
    for current_node, next_node in zip(full_route[:-1], full_route[1:]):
        total += distance_matrix[current_node, next_node]
    return float(total)
