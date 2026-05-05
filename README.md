# Matrix Calculations

Ancillary code for some numerical computations with an aim for numerical stability (not necessarily fast).

Contains differentiable [wrappers](https://github.com/kovacoj1/matrix-calculations/blob/main/src/wrappers.py) for arbitraty matrix function (even a `nn`) using `torch` that enable autograd.

## Demos

- `notebooks/matrix_functions.ipynb`: matrix functions on Jordan blocks and by Taylor expansion.
- `notebooks/linalg.ipynb`: solving a linear system through nonlinear least-squares updates using the bundled `optimizers` submodule.
- `notebooks/exponentiation.ipynb`: integer matrix powers and comparison with `torch.linalg.matrix_power`.
- `notebooks/qr.ipynb` and `notebooks/householder.ipynb`: orthogonalization experiments.
- `notebooks/fft.ipynb`: a compact FFT implementation and autodiff sanity checks.
- `notebooks/jax.ipynb`: exploratory JAX port of the Taylor-extension idea.
