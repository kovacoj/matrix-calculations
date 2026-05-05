import math

import torch


_THETA_13 = 5.371920351148152
_PADE_13 = (
    64764752532480000.0,
    32382376266240000.0,
    7771770303897600.0,
    1187353796428800.0,
    129060195264000.0,
    10559470521600.0,
    670442572800.0,
    33522128640.0,
    1323241920.0,
    40840800.0,
    960960.0,
    16380.0,
    182.0,
    1.0,
)


def matrix_exp_pade(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2 or matrix.size(0) != matrix.size(1):
        raise ValueError("matrix_exp_pade expects a square matrix")

    identity = torch.eye(matrix.size(0), dtype=matrix.dtype, device=matrix.device)
    norm_1 = matrix.abs().sum(dim=0).max().item()

    if norm_1 == 0.0:
        return identity

    scaling = max(0, math.ceil(math.log2(norm_1 / _THETA_13)))
    scaled = matrix / (2 ** scaling)

    a2 = scaled @ scaled
    a4 = a2 @ a2
    a6 = a4 @ a2

    b = _PADE_13
    u = scaled @ (
        a6 @ (b[13] * a6 + b[11] * a4 + b[9] * a2)
        + b[7] * a6
        + b[5] * a4
        + b[3] * a2
        + b[1] * identity
    )
    v = (
        a6 @ (b[12] * a6 + b[10] * a4 + b[8] * a2)
        + b[6] * a6
        + b[4] * a4
        + b[2] * a2
        + b[0] * identity
    )

    result = torch.linalg.solve(v - u, v + u)

    for _ in range(scaling):
        result = result @ result

    return result
