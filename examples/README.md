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
| [`cvxpy_transport.py`](lp/cvxpy_transport.py) | The same kind of LP written in CVXPY: a production and shipping plan with three variables (output per plant with capacity bounds, a plant × market shipment matrix, unmet demand per market), flow-balance, demand and emissions-cap constraints. Solved with `jcv.JaddleSolver` (HiGHS presolve, Jaddle, postsolve); prints the plan and reads each market's marginal cost and the implicit carbon price off the constraints' duals, checked against HiGHS. | `cvxpy` |
| [`miplib.py`](lp/miplib.py) | Load a MIPLIB instance with HiGHS, relax integrality, and solve it with `jl.solve_with_presolve`: HiGHS presolve, a Jaddle solve with feasibility polishing, crossover to a basic solution, and postsolve back to the original model. | a MIPLIB `.mps` file |
| [`battery_sizing.py`](lp/battery_sizing.py) | Batched, differentiable solves: size a battery by gradient descent, where each step solves 32 daily operating LPs at once with `jax.vmap` and differentiates their optimal costs with `jl.make_optimal_value`. The battery's price has economies of scale, so the sizing problem is not one large LP. Checks the result with a second run from a different start and plots the design's convergence and a day's schedule. | `matplotlib` |

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

### `miplib.py`

The MPS files are **not** shipped with Jaddle. Download the instances you want
from the [MIPLIB website](https://miplib.zib.de/) and put the `.mps` files in
`data/` at the repo root. The script reads `data/<PROBLEM_NAME>.mps` (default
`stp3d`, `float64`); edit `PROBLEM_NAME` (or `PATH_TO_MPS`) at the top of the
file to change instance.

It asks for tight feasibility (`1e-8`) with a looser gap (`1e-4`), the case
feasibility polishing is built for, then crosses over to a vertex and prints
the objective and the original problem's certificate residuals. When an
instance misbehaves, `jl.lp_summary_statistics(lp)` prints its size and
coefficient ranges, which is a good first check. A GPU is strongly
recommended for these.

### `battery_sizing.py`

A site with solar panels buys a battery of some energy capacity (kWh) and
power rating (kW). Operating it for a day at least cost is an LP, and the
size appears only in that LP's upper bounds. The example minimises the
battery's daily cost plus the average optimal operating cost over 32 scenario
days, each with its own prices, demand and solar output.

The battery's cost grows as the 0.8 power of its size. With a cost
proportional to size the whole problem would be a single LP (the size as two
extra variables beside every scenario's operating problem), and solving that
directly takes well under a second, far quicker than the approach here.
Economies of scale make the cost concave, so no such LP exists and the
decomposition is needed.

The gradient of an LP's optimal value with respect to a bound is the reduced
cost there, so `jax.grad` of the batched `jl.make_optimal_value` gives the
design's gradient without differentiating through the solver. Forty Adam steps
bring the daily cost from $6.02 with no battery to $3.83. The problem is not
convex, so the example repeats the run from an oversized battery: the two
runs agree within 0.03% in cost (14.3 kWh against 14.6 kWh; the cost is flat
near the optimum). It takes about 50 seconds on a CPU.

## Convex programs (`cp/`)

A convex program is defined by a `JaddleCP`: a JAX-traceable objective, a
function returning equality residuals (`= 0`), a function returning inequality
residuals (`≤ 0`), and box bounds. Jaddle differentiates them automatically.

| Example | What it shows | Extra dependencies |
|---|---|---|
| [`isotonic_regression.py`](cp/isotonic_regression.py) | Fit a non-decreasing sequence to noisy cubic data: a least-squares objective with `n-1` ordering inequalities `y[i] - y[i+1] ≤ 0` and box bounds `[-1, 1]`. Plots the fit and prints the maximum constraint violation. | `matplotlib` |
| [`linear_regression.py`](cp/linear_regression.py) | Non-linear regression with Random Fourier Features: mean-squared error over RBF features, subject to a nonlinear norm-ball constraint `‖w‖² ≤ 200`, with unbounded variables. Plots predictions against the data. | `matplotlib`, `scikit-learn` |
| [`lasso_regression.py`](cp/lasso_regression.py) | Sparse regression with an L1 penalty, split into positive and negative parts so the problem is smooth with bound constraints. Checks the fit against scikit-learn, solves the whole regularization path with one `vmap`-ed `jc.make_solver`, then tunes `alpha` by gradient descent on the validation loss, differentiating the solution with `jc.make_solution`. Finishes with one penalty per feature (50 hyperparameters), which brings the test loss to the noise floor. | `matplotlib`, `scikit-learn` |
| [`fair_logistic_regression.py`](cp/fair_logistic_regression.py) | Logistic regression on historically biased lending data, with a demographic-parity constraint (Zafar et al., 2017) on the covariance between group and score. Shows that leaving the group attribute out does not remove the bias, traces the fairness/accuracy trade-off in one `vmap`-ed solve over the constraint bound, and plots the price of fairness from the constraint's duals. | `matplotlib` |
| [`cvxpy_lasso.py`](cp/cvxpy_lasso.py) | The same sparse-regression data modelled in CVXPY: the lasso solved through `jcv.JaddleSolver` (CVXPY's QP form, solved by `jc.quadratic_program`), and L1-penalised logistic regression through `jcv.solve`, which compiles the expression tree into a smooth `JaddleCP`. Both are checked against Clarabel, and the compiled problem's size is compared with CVXPY's exponential-cone form. | `cvxpy` |
| [`cvxpy_channel_capacity.py`](cp/cvxpy_channel_capacity.py) | CVXPY's channel capacity example (Boyd & Vandenberghe, ex. 4.57): the mutual information `c @ x + Σ entr(P x) / log 2` maximised over input distributions on the simplex, solved with `jcv.solve`, which keeps `entr` smooth instead of using exponential cones. Recovers `1 − H₂(0.25)` for a binary symmetric channel, then checks a random 8 × 6 channel against Blahut–Arimoto and Clarabel, and reads the optimality conditions (equal divergences on the used inputs, the dual of `sum(x) == 1`) off the solution. | `cvxpy` |
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
