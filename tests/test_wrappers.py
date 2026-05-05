import torch

from wrappers import jordan_form_func
from wrappers import taylor_extension


@jordan_form_func
def jordan_exp(x):
    return torch.exp(x)


@taylor_extension(n_terms=40)
def taylor_exp(x):
    return torch.exp(x)


def test_jordan_form_wrapper_matches_matrix_exponential():
    jordan_block = torch.tensor(
        [
            [0.5, 1.0, 0.0],
            [0.0, 0.5, 1.0],
            [0.0, 0.0, 0.5],
        ],
        dtype=torch.float64,
    )

    expected = torch.linalg.matrix_exp(jordan_block)
    actual = jordan_exp(jordan_block)

    assert torch.allclose(actual, expected)


def test_taylor_wrapper_matches_matrix_exponential_on_small_nonnormal_matrix():
    matrix = torch.tensor(
        [
            [0.2, 0.4, 0.0],
            [0.0, -0.1, 0.3],
            [0.0, 0.0, 0.1],
        ],
        dtype=torch.float64,
    )

    expected = torch.linalg.matrix_exp(matrix)
    actual = taylor_exp(matrix)

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)
