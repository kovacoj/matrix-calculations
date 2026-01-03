import torch


def power(A: torch.Tensor, n: int) -> torch.Tensor:
    if n == 0:
        return torch.eye(A.shape[0], device=A.device, dtype=A.dtype)

    if n < 0:
        return power(torch.linalg.inv(A), -n)
    if n % 2:
        return A @ power(A, n - 1)

    half = power(A, n // 2)
    return half @ half
