# Matrix Calculations

Small numerical linear algebra experiments in PyTorch, centered on matrix functions and the tradeoff between analytic constructions and practical numerical methods.

The repository is intentionally compact. It is not a production linear algebra package, and it does not try to compete with LAPACK, SciPy, or PyTorch internals. The focus is instead on mathematically transparent implementations that are useful for studying:

- matrix functions induced by scalar functions,
- Jordan-block formulas and their derivative structure,
- Taylor-series approximations of matrix functions,
- orthogonalization and stability,
- optimization-based viewpoints on linear algebra problems.

## Main ideas

The core source code lives in `src/`.

- `src/wrappers.py` implements two ways of lifting scalar functions to matrices:
  - `jordan_form_func`, for matrices already given in Jordan form,
  - `taylor_extension`, which uses PyTorch autograd to build Taylor approximations of matrix functions.
- `src/pade.py` provides a compact scaling-and-squaring Pad\u00e9 baseline for the matrix exponential.
- `src/power.py` contains exponentiation by squaring for integer matrix powers.
- `src/symbolic.py` is a small symbolic differentiation experiment related to the same general theme of function manipulation.

## Notebook guide

The notebooks are the main showcase layer of the repository.

- `notebooks/matrix_functions.ipynb`: matrix functions on Jordan blocks and by Taylor expansion.
- `notebooks/linalg.ipynb`: solving a linear system through nonlinear least-squares updates using the bundled `optimizers` submodule.
- `notebooks/exponentiation.ipynb`: integer matrix powers and comparison with `torch.linalg.matrix_power`.
- `notebooks/qr.ipynb`: stability study comparing classical Gram-Schmidt with Householder QR.
- `notebooks/householder.ipynb`: smaller scratch notebook on Householder reflections.
- `notebooks/fft.ipynb`: a compact FFT implementation and autodiff sanity checks.
- `notebooks/jax.ipynb`: exploratory JAX port of the Taylor-extension idea.

## Limitations

- The Taylor-based matrix-function construction is pedagogically useful, but it is not a replacement for standard robust algorithms.
- The Pad\u00e9 implementation is included as a standard baseline for the matrix exponential, not as a full general-purpose matrix-function framework.
- The Jordan-form-based construction is most informative on controlled examples; it is not intended as a numerically stable black-box routine.
- Several notebooks are exploratory and should be read as computational notes rather than polished software documentation.
- The `optimizers/` directory is a Git submodule with its own lifecycle and tests.

## Reproducible setup

This repository uses `uv` for environment management.

1. Clone the repository and initialize the optimizer submodule.

```bash
git clone https://github.com/kovacoj/matrix-calculations.git
cd matrix-calculations
git submodule update --init --recursive
```

2. Create the environment and install the notebook dependencies.

```bash
uv sync --group notebooks --group dev
```

3. Install the bundled optimizer package into the same environment.

```bash
uv pip install -e ./optimizers
```

4. Start Jupyter from the repository root.

```bash
uv run jupyter lab
```

Because the root project is currently a loose `src/` tree rather than a packaged library, the notebooks import directly from the repository source.

## Quick checks

Run the root test suite with:

```bash
uv run pytest
```

Run the optimizer submodule tests with:

```bash
uv run --directory optimizers pytest
```
