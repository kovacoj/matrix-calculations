import torch

from pade import matrix_exp_pade


def test_pade_matches_torch_matrix_exp_on_small_matrix():
    matrix = torch.tensor(
        [
            [0.2, 0.4, 0.0],
            [0.0, -0.1, 0.3],
            [0.0, 0.0, 0.1],
        ],
        dtype=torch.float64,
    )

    expected = torch.linalg.matrix_exp(matrix)
    actual = matrix_exp_pade(matrix)

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_pade_matches_torch_matrix_exp_after_scaling_and_squaring():
    matrix = torch.tensor(
        [
            [7.0, 2.0, 0.0],
            [0.0, -6.0, 1.0],
            [0.0, 0.0, 4.0],
        ],
        dtype=torch.float64,
    )

    expected = torch.linalg.matrix_exp(matrix)
    actual = matrix_exp_pade(matrix)

    assert torch.allclose(actual, expected, atol=1e-11, rtol=1e-11)
