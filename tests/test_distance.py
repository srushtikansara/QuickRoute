import numpy as np
import pytest

from src.data_generator import generate_dataset
from src.distance import euclidean_distance_matrix, route_length


def test_matrix_is_symmetric_with_zero_diagonal():
    df = generate_dataset(12, seed=1)
    matrix = euclidean_distance_matrix(df[["x_coordinate", "y_coordinate"]].to_numpy())
    assert matrix.shape == (13, 13)
    assert np.allclose(matrix, matrix.T)
    assert np.all(np.diag(matrix) == 0.0)
    assert np.all(matrix >= 0)


def test_known_distance_3_4_5():
    matrix = euclidean_distance_matrix([[0, 0], [3, 4]])
    assert matrix[0, 1] == pytest.approx(5.0)


def test_scale_is_applied():
    matrix = euclidean_distance_matrix([[0, 0], [3, 4]], scale=0.1)
    assert matrix[0, 1] == pytest.approx(0.5)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        euclidean_distance_matrix([[0, 0, 0]])
    with pytest.raises(ValueError):
        euclidean_distance_matrix([[0, np.nan]])
    with pytest.raises(ValueError):
        euclidean_distance_matrix([[0, 0]], scale=0)


def test_route_length():
    matrix = euclidean_distance_matrix([[0, 0], [3, 4], [6, 8]])
    assert route_length(matrix, [0, 1, 2, 0]) == pytest.approx(5 + 5 + 10)
