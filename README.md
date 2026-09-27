# ETICS 2026

Numerical experiments for probabilistic inversion, using the **PyCirce** package: a Python implementation of the CIRCE methodology.

## Package: PyCirce

`pycirce` (installed under `src/pycirce`) implements four estimators for the population parameters `(mu, Gamma)`, all assuming a diagonal covariance `Gamma`:

| Class | Method |
| --- | --- |
| `CirceEMdiag` | Expectation-Maximisation (EM) algorithm |
| `CirceECMEdiag` | Expectation Conditional Maximisation Either (ECME) algorithm |
| `CirceREML` | Restricted maximum likelihood, optimised numerically using `L-BFGS-B` |
| `CirceProfile` | Profile likelihood, optimised numerically using `L-BFGS-B` |

Each class shares the same constructor signature:

```python
import pycirce as pyc

model = pyc.CirceEMdiag(
    h=h,                 # Jacobian matrix of the numerical simulator, shape (n_parameters, n_observations)
    z_exp=z_exp,          # experimental observations, shape (n_observations,)
    z_nom=z_nom,           # nominal simulator outputs, shape (n_observations,)
    sig_eps=sig_eps,       # measurement standard deviations, shape (n_observations,)
    initial_mean=None,    # optional, defaults to a vector of ones
    initial_cov=None,     # optional, defaults to the identity matrix
    tolerance=1e-5,       # optional, convergence tolerance
    niter=15000,          # optional, maximum number of iterations
)

mu, gamma, loglik, n_iter = model.estimate(n_starts=10, random_state=0)
```

`estimate()` performs multistart optimisation (from the provided initial point plus randomly perturbed restarts) and returns the trajectory achieving the best final log-likelihood. `CirceEMdiag` and `CirceECMEdiag` return the full iterate trajectories (`mu`, `gamma`, `loglik` as lists, one entry per iteration); `CirceREML` and `CirceProfile` return only the final estimate and a two-point log-likelihood summary (initial vs. final).

## Installation

The project uses [uv](https://docs.astral.sh/uv/) as its build backend.

```bash
uv pip install -e .
```

or with plain `pip`:

```bash
pip install -e .
```

Core dependencies: `numpy`, `pandas`, `scipy` (for `CirceREML`/`CirceProfile`), `coolprop`, `matplotlib`, `seaborn`.

## Examples

The [examples/](examples/) directory contains the Monte-Carlo numerical experiments used in the ETICS 2026 presentation, comparing the four estimators:

- [toycase.py](examples/toycase.py) — Monte-Carlo study on a synthetic linear toy model.
- [blasius.py](examples/blasius.py) — Monte-Carlo study calibrating the Blasius friction correlation, with fluid properties computed via [CoolProp](http://www.coolprop.org/).
- [blasius_bias_gamma.py](examples/blasius_bias_gamma.py) / [blasius_bias_mu_gamma.py](examples/blasius_bias_mu_gamma.py) — bias diagnostics of the variance (`gamma`) and joint mean/variance (`mu`, `gamma`) estimates on the Blasius case.
- [blasius_results.py](examples/blasius_results.py) — post-processing and plotting of the Blasius friction correlation Monte-Carlo study results (batched variance ratios, etc.).

Each `*.py` script under `examples/` that runs a Monte-Carlo study (`toycase.py`, `blasius.py`) parallelises replications with `concurrent.futures.ProcessPoolExecutor` and writes per-estimator `mu`/`gamma` trajectories to CSV files (`results/` or `results_toycase/`), which are then consumed by the corresponding `*_results.py` / `winrate.py` analysis scripts.

## License

GPLv3+. See [pyproject.toml](pyproject.toml) for package metadata.

## Author

Clément Gauchy (clement.gauchy@cea.fr)
