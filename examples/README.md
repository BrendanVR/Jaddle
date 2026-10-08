# Jaddle Examples

Short, self-contained scripts that show how to set up and solve problems with
Jaddle. They are grouped by solver:

- [`lp/`](lp/): linear programs, solved with `jaddle.jaddle_linear`
- [`cp/`](cp/): general convex programs, solved with `jaddle.jaddle_convex`

Every example is written in the "percent" cell format (`# %%` / `# %% [markdown]`).
You can run it as a plain script, or step through it cell by cell in VS Code's
Interactive Window or in Jupyter (via jupytext).

```bash
pip install -e .                  # from the repo root
python examples/lp/intro_example.py
```

## Linear programs (`lp/`)

| Example | What it shows | Extra dependencies |
|---|---|---|
| [`intro_example.py`](lp/intro_example.py) | The smallest end-to-end LP: build an `LP` from NumPy/SciPy arrays, call `jl.solve`, and read off the primal solution and objective. | — |
| [`miplib_standard.py`](lp/miplib_standard.py) | Load a MIPLIB instance with HiGHS, relax integrality, convert it to Jaddle's sparse standard form with `highs_to_standard_form_sparse`, and solve it. | `highspy`, a MIPLIB `.mps` file |
| [`miplib_standard_sciopt.py`](lp/miplib_standard_sciopt.py) | The same workflow using PySCIPOpt: read the relaxed model, run SCIP's presolve, convert with `scip_to_standard_form_sparse`, and add the presolve objective offset back to the reported objective. | `pyscipopt`, a MIPLIB `.mps` file |

### `intro_example.py`

Solves

```
minimise    3 x1 + 2 x2
subject to  x1 + x2 >= 4
            x1, x2 >= 0
```

Jaddle's standard form is `min cᵀx  s.t.  A_eq x = b_eq,  A_ineq x ≤ b_ineq,
l ≤ x ≤ u`, so the `≥` constraint is negated to become `-x1 - x2 ≤ -4`. The
example runs on CPU because the problem is tiny. The expected answer is
`x = (0, 4)` with objective `8`.

### MIPLIB examples

The MPS files are **not** shipped with Jaddle. Download the instances you want
from the [MIPLIB website](https://miplib.zib.de/) and put the `.mps` files in
`data/` at the repo root. Both scripts read `data/<PROBLEM_NAME>.mps`; edit
`PROBLEM_NAME` (or `PATH_TO_MPS`) at the top of the file to change instance.

| Script | Default instance | Presolve | Precision |
|---|---|---|---|
| `miplib_standard.py` | `app1-2` | none | `float64` |
| `miplib_standard_sciopt.py` | `stp3d` | SCIP | `float64` |

Both scripts call `jl.lp_summary_statistics(lp)` before solving, which prints
the problem size and coefficient ranges. This is a good first check when an
instance misbehaves. A GPU is strongly recommended for these.

## Convex programs (`cp/`)

A convex program is defined by a `JaddleCP`: a JAX-traceable objective, a
function returning equality residuals (`= 0`), a function returning inequality
residuals (`≤ 0`), and box bounds. Jaddle differentiates them automatically.

| Example | What it shows | Extra dependencies |
|---|---|---|
| [`isotonic_regression.py`](cp/isotonic_regression.py) | Fit a non-decreasing sequence to noisy cubic data: a least-squares objective with `n-1` ordering inequalities `y[i] - y[i+1] ≤ 0` and box bounds `[-1, 1]`. Plots the fit and prints the maximum constraint violation. | `matplotlib` |
| [`linear_regression.py`](cp/linear_regression.py) | Non-linear regression with Random Fourier Features: mean-squared error over RBF features, subject to a nonlinear norm-ball constraint `‖w‖² ≤ 200`, with unbounded variables. Plots predictions against the data. | `matplotlib`, `scikit-learn` |

Both examples use `jc.solve(cp)` with its default settings, so they are a good
starting template: replace the objective and constraint functions with your own.

## Precision

`jaddle.jaddle_optimisers.configure_jax(profile)` sets JAX's precision
globally and should be called before any arrays are created. The MIPLIB
examples use `"float64"`, which real LPs generally need to reach tight
tolerances. `isotonic_regression.py` uses `"float32"`, which is enough for a
small, well-conditioned problem.

## See also

- [`benchmarks/`](../benchmarks/README.md): batch harnesses that run Jaddle over
  whole directories of MIPLIB instances and over a suite of convex problems.
