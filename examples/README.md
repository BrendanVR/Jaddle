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
| [`battery_sizing.py`](lp/battery_sizing.py) | Batched, differentiable solves: size a battery by gradient descent, where each step solves 32 daily operating LPs at once with `jax.vmap` and differentiates their optimal costs with `jl.make_optimal_value`. Checks the result against one large LP solved by HiGHS and plots the design's convergence and a day's schedule. | `matplotlib` |

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

### `miplib_standard.py`

The MPS files are **not** shipped with Jaddle. Download the instances you want
from the [MIPLIB website](https://miplib.zib.de/) and put the `.mps` files in
`data/` at the repo root. The script reads `data/<PROBLEM_NAME>.mps` (default
`app1-2`, no presolve, `float64`); edit `PROBLEM_NAME` (or `PATH_TO_MPS`) at the
top of the file to change instance.

The script calls `jl.lp_summary_statistics(lp)` before solving, which prints
the problem size and coefficient ranges. This is a good first check when an
instance misbehaves. A GPU is strongly recommended for these.

### `battery_sizing.py`

A site with solar panels buys a battery of some energy capacity (kWh) and
power rating (kW). Operating it for a day at least cost is an LP, and the
size appears only in that LP's upper bounds. The example minimises the
battery's daily cost plus the average optimal operating cost over 32 scenario
days, each with its own prices, demand and solar output.

The gradient of an LP's optimal value with respect to a bound is the reduced
cost there, so `jax.grad` of the batched `jl.make_optimal_value` gives the
design's gradient without differentiating through the solver. Forty Adam steps
bring the daily cost from $6.02 with no battery to $3.51, within 0.1% of the
exact optimum from HiGHS (14.2 kWh against 14.7 kWh; the cost is flat near
the optimum). It takes about 35 seconds on a CPU.

## Convex programs (`cp/`)

A convex program is defined by a `JaddleCP`: a JAX-traceable objective, a
function returning equality residuals (`= 0`), a function returning inequality
residuals (`≤ 0`), and box bounds. Jaddle differentiates them automatically.

| Example | What it shows | Extra dependencies |
|---|---|---|
| [`isotonic_regression.py`](cp/isotonic_regression.py) | Fit a non-decreasing sequence to noisy cubic data: a least-squares objective with `n-1` ordering inequalities `y[i] - y[i+1] ≤ 0` and box bounds `[-1, 1]`. Plots the fit and prints the maximum constraint violation. | `matplotlib` |
| [`linear_regression.py`](cp/linear_regression.py) | Non-linear regression with Random Fourier Features: mean-squared error over RBF features, subject to a nonlinear norm-ball constraint `‖w‖² ≤ 200`, with unbounded variables. Plots predictions against the data. | `matplotlib`, `scikit-learn` |
| [`kernel_svm.py`](cp/kernel_svm.py) | Train an RBF-kernel support vector machine on two interleaved half-moons. The SVM dual is a quadratic program, solved with `jc.quadratic_program`; the intercept is read off the equality constraint's dual. Plots the decision boundary, margins and support vectors. | `matplotlib` |

The first two examples use `jc.solve(cp)` with its default settings, so they
are a good starting template: replace the objective and constraint functions
with your own. `kernel_svm.py` is the template for a problem you can write
down as matrices.

## Precision

`jaddle.jaddle_optimisers.configure_jax(profile)` sets JAX's precision
globally and should be called before any arrays are created. The MIPLIB
examples use `"float64"`, which real LPs generally need to reach tight
tolerances. `isotonic_regression.py` uses `"float32"`, which is enough for a
small, well-conditioned problem.

## See also

- [`benchmarks/`](../benchmarks/README.md): batch harnesses that run Jaddle over
  whole directories of MIPLIB instances and over a suite of convex problems.
