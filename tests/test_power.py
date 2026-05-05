import torch

from power import power


def test_power_matches_torch_matrix_power():
    matrix = torch.tensor(
        [
            [3.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )

    for exponent in (-3, -1, 0, 1, 2, 5):
        expected = torch.linalg.matrix_power(matrix, exponent)
        actual = power(matrix, exponent)

        assert torch.allclose(actual, expected)
