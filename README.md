<img width="1536" height="1024" alt="jaddle_logo_full" src="https://github.com/user-attachments/assets/d4e61518-3557-4273-b4d0-0d89510e0f3f" />

# Jaddle: The JAX Saddle Solver

*First-order primal–dual solvers for large-scale linear and convex programs, written entirely in JAX.*

Jaddle solves constrained optimisation problems by casting them as saddle-point
problems on the Lagrangian and running primal–dual iterations to convergence.
It provides two solvers:

- **`jaddle.jaddle_linear`**: a restarted, adaptively step-sized PDHG-family
  solver for sparse linear programs, in the tradition of PDLP and cuPDLP.
- **`jaddle.jaddle_convex`**: a general solver for smooth convex programs. The
  user supplies the objective and constraints as ordinary JAX functions, and
  Jaddle derives everything else by automatic differentiation.

Both solvers run unchanged on CPU, GPU and TPU. Jaddle is not intended to
replace a commercial or production-grade solver. It aims to be a compact,
readable and hackable platform for first-order optimisation that still holds
up on demanding benchmarks.

## Contents

- [Why JAX](#why-jax)
- [Installation](#installation)
- [Quick start](#quick-start)
- [The linear programming solver](#the-linear-programming-solver)
- [The convex programming solver](#the-convex-programming-solver)
- [Numerical precision](#numerical-precision)
- [Benchmarks](#benchmarks)
- [Repository layout](#repository-layout)

## Why JAX

Jaddle has no hand-written kernels. Every solver in the library is expressed
in high-level JAX, and JAX's compiler and transformations supply the
performance and portability. This brings several concrete benefits.

**Compiled performance from Python.** Each solver's whole iteration loop is
traced once and compiled by XLA into a single program. Primal and dual updates,
step size line searches, projections, convergence checks and restart decisions
all run on the accelerator, many epochs at a time, without returning control to
Python. Python steps in only about once a second, to enforce time limits and
print progress, and not at all when there is no logging or budget to enforce.
The LP solver is therefore limited by the speed of sparse matrix–vector
products, which is the limit any first-order LP method faces, rather than by
interpreter overhead or host–device synchronisation.

**Hardware portability.** The same source runs on a laptop CPU, a consumer GPU
or a TPU pod. There are no device-specific code paths to maintain, and moving
between hardware targets requires no code changes. Switching from CPU to GPU is
a matter of installing the appropriate `jax` wheel.

**Automatic differentiation.** The convex solver needs only the objective and
the constraint functions. Gradients of the Lagrangian with respect to the
primal and dual variables come from `jax.grad`, so users never derive,
implement or debug a gradient by hand. Any function JAX can trace can be used
as an objective or constraint, from a least-squares loss to a neural network.
LP solves are differentiable in turn: the optimal value and solution can be
differentiated with respect to the LP's data, and solves batch with `vmap`
(see [Batched and differentiable solves](#batched-and-differentiable-solves)).

**Composable optimisation primitives.** The convex solver builds on
[Optax](https://github.com/google-deepmind/optax). It uses Optax's projection
operators, and in `alternating` mode it accepts any Optax gradient
transformation as the update rule. A new primal–dual variant is often only a
few lines of code: for example, giving the primal player plain gradient descent
and the dual player Nesterov momentum.

**Precision as a configuration choice.** A single call switches the whole
library between float64, float32 and float16 arithmetic. Nothing is
duplicated per precision, so the trade-off between accuracy and throughput can
be explored without code changes.

**Functional state.** Solver state is an immutable pytree (`SaddleState`, plus
the step size state). Warm starting, checkpointing and resuming a solve
therefore amount to passing a value back in. The solver keeps no hidden mutable
state.

**A small, readable codebase.** Because the compiler handles fusion, memory
layout and device placement, the full library is roughly 5,700 lines of
Python. Algorithms remain close to their mathematical description, which makes
Jaddle well suited to research and experimentation.

## Installation

Jaddle requires Python 3.11 or later and is available on
[PyPI](https://pypi.org/project/jaddle/):

```bash
pip install jaddle
```

This installs the core dependencies: `jax`, `optax`, `numpy`, `scipy` and
`highspy`, and runs on CPU. Optional features are available as extras, which
can be combined, for example `pip install "jaddle[cuda12,examples]"`:

| Extra | Installs | Used for |
|---|---|---|
| `cuda12`, `cuda13` | CUDA-enabled JAX | GPU execution. Choose the extra that matches your CUDA driver; see the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for TPU and other platforms. |
| `scip` | `pyscipopt` | Reading and presolving MPS files with SCIP (`jaddle.sciopt_helpers`) |
| `glop` | `ortools` | Presolving with OR-Tools glop (`jaddle.glop_helpers`). Also requires building the helper with `tools/glop_presolve/build.sh`. |
| `examples` | `matplotlib`, `scikit-learn` | The convex examples |
| `test` | `pytest` | Running the test suite |
| `dev` | `examples`, `test`, `build`, `twine` | Development and packaging |

Jaddle is developed on Linux, including WSL 2. Continuous integration tests
it on Python 3.11, 3.12 and 3.13.

To work on Jaddle itself, or to run the examples, benchmarks and tests, install
from a clone of the repository instead:

```bash
git clone https://github.com/BrendanVR/Jaddle.git
cd Jaddle
pip install -e ".[dev]"
```

### Running the tests

From the repository root:

```bash
pytest
```

The suite runs on CPU in double precision and takes well under a minute. It
checks both solvers against exact or reference solutions (SciPy's HiGHS
interface for LPs, closed-form solutions for the convex problems), the
defined-variable presolve, and the self-contained examples. GitHub Actions runs
it on every push and pull request.

## Quick start

### A linear program

```python
import numpy as np
import scipy.sparse as sp
import jaddle.jaddle_linear as jl

# minimise 3 x1 + 2 x2  subject to  x1 + x2 >= 4,  x >= 0
lp = jl.LP(
    c=np.array([3.0, 2.0]),
    A_eq=sp.csc_matrix((0, 2)), b_eq=np.zeros(0),
    A_ineq=sp.csc_matrix([[-1.0, -1.0]]), b_ineq=np.array([-4.0]),
    lower_bounds=np.zeros(2), upper_bounds=np.full(2, np.inf),
)

result = jl.solve(lp, verbose=True)
x = result["solution"].primal
print(result["stop_reason"], lp.objective(x))
```

### A convex program

```python
import jax.numpy as jnp
import jaddle.jaddle_convex as jc

y = jnp.linspace(-1.0, 1.0, 100) ** 3   # data to fit

cp = jc.JaddleCP(
    num_variables=100,
    objective=lambda p: jnp.sum((p - y) ** 2),
    constraints_eq=lambda p: jnp.zeros(0),
    constraints_ineq=lambda p: p[:-1] - p[1:],  # p must be non-decreasing
    lower_bounds=-jnp.ones(100),
    upper_bounds=jnp.ones(100),
)

result = jc.solve(cp)
p = result["solution"].primal
```

The [`examples/`](https://github.com/BrendanVR/Jaddle/blob/main/examples/README.md) directory contains complete, annotated
versions of both, together with examples that load and solve MIPLIB instances.

## The linear programming solver

### Problem form

`jaddle_linear.solve` accepts linear programs in the standard form

```
minimise    cᵀx
subject to  A_eq   x  = b_eq
            A_ineq x <= b_ineq
            l <= x <= u
```

where bounds may be infinite. Problems can be built directly from NumPy and
SciPy sparse arrays with `jl.LP`. They can also be converted from MPS files
through HiGHS (`highs_helpers.highs_to_standard_form_sparse`), SCIP
(`sciopt_helpers.scip_to_standard_form_sparse`) or OR-Tools glop
(`glop_helpers.glop_presolve`). Internally, constraint matrices are stored as
JAX `BCOO` sparse arrays.

The solver seeks a saddle point of the Lagrangian

```
L(x, y) = cᵀx + y_eqᵀ(A_eq x − b_eq) + y_ineqᵀ(A_ineq x − b_ineq),   y_ineq >= 0,
```

minimising over `x` within its bounds and maximising over the duals `y`.

### Algorithm

Each iteration takes a projected primal descent step and a projected dual
ascent step. The key components follow the design of PDLP and cuPDLP, with
some refinements of our own.

**Update schemes** (`update_mode`)

| Mode | Description |
|---|---|
| `"alternating"` (default) | Gauss–Seidel primal–dual iteration: a primal step, then a dual step evaluated at the new primal iterate. |
| `"pdhg"` | Chambolle–Pock PDHG: the dual step is taken at the extrapolated point `2x⁽ᵏ⁺¹⁾ − x⁽ᵏ⁾`. |
| `"halpern"` | Restarted Halpern-anchored PDHG. Each step blends the PDHG update back towards an anchor point, which accelerates convergence of the last iterate. Best used with restarts. |

**Adaptive step size.** All modes use a cuPDLP-style adaptive step. At every
iteration the step `η` is checked against a local estimate of the interaction
term `(Δy)ᵀ A (Δx)`. A step that overshoots is rejected and shrunk. An accepted
step is advanced conservatively. The step size learned during a solve is
carried across restarts.

**Primal weight.** The primal and dual step sizes are `η / k` and `η · k`. The
primal weight `k` is initialised from the ratio `‖c‖ / ‖b‖`. At each restart it
is rebalanced from the relative movement of the primal and dual iterates, using
a log-space blend controlled by `k_theta` and clamped to
`[1/k_scale, k_scale]`.

**Adaptive restarts.** A restart resets the averaging (and, for `halpern`, the
anchor) and keeps the current iterate as a warm start. This stops the
iteration from settling into slow rotational orbits. Restarts fire on any of
three conditions, all judged by a normalised KKT merit:

- *sufficient progress*: the merit falls below `restart_decay` times its value
  at the last restart;
- *stalling*: the merit falls below `necessary_decay` times its value at the
  start of the cycle and then starts to rise;
- *no progress*: the cycle reaches its length cap.

The merit can be checked within an epoch (`restart_check_every`), so a restart
does not have to wait for the epoch boundary.

**Diagonal preconditioning.** Before solving, the problem is equilibrated by
Ruiz (L∞) scaling followed by Pock–Chambolle (L1) scaling. By default the
scaling is applied to the augmented matrix `[[A, b], [cᵀ, 0]]`, so cost and
right-hand-side magnitudes also inform the scaling. The objective and
right-hand side are then normalised. All reported quantities are mapped back
to the original problem's units.

**Termination.** The solver stops when the standard LP optimality certificate
holds to a relative tolerance, following the PDLP and HiGHS conventions:

| Test | Normalised by | Tolerance argument |
|---|---|---|
| Primal feasibility | `1 + ‖b‖` | `primal_feasibility_tolerance` |
| Dual feasibility | `1 + ‖c‖` | `dual_feasibility_tolerance` |
| Duality gap | `1 + \|primal obj\| + \|dual obj\|` | `dual_gap_tolerance` |

All three tolerances default to `1e-3`. The norms are L2 by default
(`termination_norm`). Reduced costs are split between the dual residual and
the dual objective as in PDLP, so far-away finite bounds do not swamp the gap.

### Principal options

| Argument | Default | Purpose |
|---|---|---|
| `update_mode` | `"alternating"` | Iteration scheme (see above). |
| `iterations_per_epoch` | `256` | Iterations per compiled epoch. Metrics, logging and termination checks run between epochs. |
| `max_epochs`, `max_seconds` | `None` | Iteration and wall-clock budgets. `max_seconds` includes setup and compilation. |
| `restarts`, `epochs_per_restart` | `True`, `None` | Adaptive restarts and the length cap of a restart cycle (`None` = no cap). |
| `restart_decay`, `necessary_decay` | `0.2`, `0.8` | Thresholds for the sufficient-progress and stalling restarts. |
| `k_scale`, `k_theta`, `k_init` | `1e8`, `0.5`, `None` | Clamp band, smoothing and initial value for the primal weight. |
| `ruiz_iterations`, `pc_iterations` | `10`, `1` | Number of preconditioning sweeps. |
| `initial_solution`, `initial_opt_state` | `None` | Warm start from a previous solve's `solution` and `opt_state`. |
| `vertex_bias` | `0.0` | Small cost perturbation that steers the solver to a vertex of the optimal face rather than its interior. Useful before crossover. |
| `verbose`, `log_every` | `False`, `1` | Per-epoch progress logging. |

### Results

`solve` returns a dictionary with the following keys:

| Key | Contents |
|---|---|
| `solution` | A `SaddleState(primal, dual_ineq, dual_eq)` in the original problem's units. |
| `converged` | Whether a termination criterion was met. |
| `stop_reason` | `"certificate"`, `"primal_stall"` (only when the opt-in `primal_stop` heuristic is enabled), `"max_epochs"`, `"time_limit"` or `"interrupted"`. |
| `opt_state` | Final step size and primal weight, for warm starting. |
| `solve_seconds` | Time spent in the iteration loop, including the first-epoch compile. |
| `corrected_seconds` | Iteration-loop time with the one-off compile amortised out. |
| `epochs` | Number of epochs run. |

### Batched and differentiable solves

`jl.make_solver(lp, **options)` turns `solve()` into a pure JAX function of
the LP's numbers. It takes the same options and reproduces `solve()` exactly,
but it has no logging, time limit or `vertex_bias`. The numbers travel as a
`jl.LPValues` pytree: the cost, the stored nonzeros of `A_eq` and `A_ineq` on
`lp`'s sparsity pattern, the right-hand sides and the bounds
(`jaddle_lp.values()` extracts them, `jaddle_lp.with_values(v)` puts them
back). The function can be jitted, batched with `jax.vmap` and differentiated:

```python
values = jl.to_jaddle_sparse(lp).values()
solve_fn = jl.make_solver(lp, max_epochs=1000)
result = jax.jit(solve_fn)(values)       # SolveCoreResult: solution, converged, ...

costs = values.c * (1 + 0.1 * jax.random.normal(key, (32,) + values.c.shape))
batch = jl.solve_batch(lp, values._replace(c=costs))   # 32 solves at once
```

In a batch the cost, right-hand sides, bounds and matrix values can all vary;
any field without a batch axis is shared. Each member is scaled as `solve()`
would scale it, and finished members wait for the slowest. Members are not
bit-identical to separate solves, because XLA rounds batched reductions
slightly differently, but each certifies to the requested tolerance.

Three functions differentiate through a solve:

| Function | Differentiates | How |
|---|---|---|
| `jl.make_optimal_value` | the optimal value `z*` w.r.t. every number | envelope theorem: `∂z*/∂c = x*`, `∂z*/∂b = −y*`, `∂z*/∂A_ij = y*_i x*_j`, and the reduced costs for the bounds |
| `jl.make_solution` | `x*` w.r.t. `b`, `A` and the bounds | implicit differentiation of the active constraints at a nondegenerate vertex (raises an error elsewhere) |
| `jl.make_perturbed_solution` | a smoothed `x*` w.r.t. `c` | perturbed optimizer: averages solves at `c + σZ`, batched |

An LP's solution is piecewise constant in its cost, so `dx*/dc` is zero
almost everywhere. That's why the smoothed, perturbed version exists for
learning costs. On degenerate LPs a first-order method can return a point
inside an optimal face, where `make_solution`'s derivative is undefined.
All three are checked against finite differences of HiGHS solutions in the
tests.

### Supporting tools

- **Presolve.** `jaddle.presolve.eliminate_defined_variables` substitutes out
  variables defined by dense equality rows (`z = Σ aⱼxⱼ`). Rows of this kind
  hide objective structure from the scaling and can stall first-order methods.
  External presolve is available through HiGHS, SCIP and glop.
- **Certification.** `jl.evaluate_lp_certificate` reports the relative primal,
  dual and gap residuals of any primal–dual pair independently of a solve.
- **Feasibility polishing.** `jl.solve_with_polishing` is `jl.solve` with
  PDLP-style feasibility polishing: once the gap has closed, short zero-objective
  (`jl.primal_polish`) and zero-RHS (`jl.dual_polish`) sub-solves push the
  residuals down, and the polished pair is returned only if it certifies. It
  pays off when feasibility is tighter than the gap, e.g. `tol=1e-8,
  dual_gap_tolerance=1e-4`: on stp3d (float64) that certifies in 24 s, where a
  plain solve is still at PFR 5e-6 after 150 s.
- **Infeasibility detection.** `jl.detect_infeasibility` classifies an LP as
  `"primal_infeasible"`, `"dual_infeasible"` (unbounded if feasible),
  `"optimal"` or `"undetermined"` the PDLP way: diverging iterates give a
  candidate Farkas or unbounded ray, refined by a small least-squares fix and
  checked with `jl.evaluate_infeasibility_certificate`.
- **Diagnostics.** `jl.lp_summary_statistics` prints the problem dimensions and
  coefficient ranges, which is the first thing to check when an instance
  misbehaves.

## The convex programming solver

### Problem form

`jaddle_convex.solve` accepts a `JaddleCP`:

```python
jc.JaddleCP(
    num_variables,      # n
    objective,          # f(x) -> scalar, convex and differentiable
    constraints_eq,     # h(x) -> vector, constrained to h(x) = 0
    constraints_ineq,   # g(x) -> vector, constrained to g(x) <= 0
    lower_bounds,       # l, may contain -inf
    upper_bounds,       # u, may contain +inf
)
```

All functions must be traceable by JAX. The solver forms the Lagrangian
`f(x) + λᵀg(x) + μᵀh(x)` with `λ >= 0`, and obtains every gradient it needs by
automatic differentiation. No derivatives are required from the user.

### Algorithm

**Update schemes** (`update_mode`)

| Mode | Gradient evaluations per iteration | Description |
|---|---|---|
| `"extragradient"` (default) | 2 | Korpelevich extragradient: a look-ahead step followed by a corrector step. Contractive for monotone problems. |
| `"forward_reflected"` | 1 | Malitsky–Tam forward-reflected-backward: one gradient per iteration plus a reflection term built from the previous gradient. Requires the adaptive step. |
| `"alternating"` | 1 | Primal step, then a dual step at the new primal point. The update rule is a user-supplied Optax optimiser. |

**Adaptive step size.** With `adaptive_eta="auto"` (the default), the
extragradient and forward-reflected schemes use a Malitsky–Tam line search
based on a local Lipschitz estimate. The two gradient evaluations that
extragradient already performs give an estimate of the local Lipschitz
constant at no extra cost. The step is accepted while it remains within the
admissible bound, and is rejected and shrunk otherwise. The learned step is
carried across restarts.

**Primal weight.** As in the LP solver, a primal weight `k` balances the primal
and dual step sizes. It is initialised from the gradient of the objective and
the constraint values at the origin, rebalanced at each restart, and clamped
to `[1/k_scale, k_scale]` (default `k_scale = 10`).

**Custom optimisers.** In `alternating` mode, `optimiser` accepts any Optax
`GradientTransformation`. It is applied separately to the primal and dual
players through `jaddle_optimisers.create_saddle_optimiser`. Ready-made
choices include `jo.gd`, `jo.optimistic_gd`, `jo.optimistic_adadelta` and
`jo.gd_dual_momentum`.

**Restarts and averaging.** Adaptive restarts (`restarts=True`) and iterate
averaging (`average=True`, with an optional `weight_function`) are available
but off by default.

**Termination.** The solver stops when all three of the following fall below
their tolerances:

| Test | Tolerance argument | Default |
|---|---|---|
| Stationarity (projected primal gradient norm of the Lagrangian) | `primal_grad_norm_tolerance` | `1e-2` |
| Primal feasibility | `primal_feasibility_tolerance` | `1e-3` |
| Complementary slackness | `complementarity_slack_tolerance` | `1e-3` |

### Principal options

| Argument | Default | Purpose |
|---|---|---|
| `update_mode` | `"extragradient"` | Iteration scheme (see above). |
| `adaptive_eta` | `"auto"` | Adaptive line-searched step. Pass a float to set the seed, or `None` to use the optimiser's fixed learning rate. |
| `optimiser` | `None` | Optax optimiser for `alternating` mode. When `None`, gradient descent with learning rate 0.5 is used. |
| `iterations_per_epoch` | `1000` | Iterations per compiled epoch. |
| `max_epochs`, `max_seconds` | `None` | Iteration and wall-clock budgets. |
| `k_scale`, `k_theta`, `k_init` | `10.0`, `0.5`, `None` | Primal weight controls. `k_scale=None` disables the primal weight. |
| `restarts`, `epochs_per_restart` | `False`, `None` | Adaptive restarts (`epochs_per_restart=None` = no cycle cap). |
| `initial_solution`, `initial_opt_state` | `None` | Warm start. |

`solve` returns a dictionary containing `solution` (a `SaddleState`),
`converged`, `stop_reason`, `opt_state` and `solve_seconds`.

## Numerical precision

Call `jaddle.jaddle_optimisers.configure_jax(profile)` before creating any
arrays:

| Profile | Arithmetic | Recommended for |
|---|---|---|
| `"float64"` (default) | Double precision, full-precision matrix products | Real-world LPs and any problem needing tight tolerances. Wide-dynamic-range instances can stall on rounding error in single precision. |
| `"float32"` | Single precision, TF32 matrix products | Throughput on well-conditioned problems. |
| `"float16"` | Half precision, accumulating in float32 | Memory- and bandwidth-limited experiments. |

The profile can also be set through the `JADDLE_JAX_PROFILE` environment
variable. `configure_jax` also enables a persistent XLA compilation cache in
`~/.cache/jaddle_jax`, so repeated solves of problems with the same shape skip
recompilation.

## Benchmarks

Jaddle's LP solver is evaluated on the LP relaxations of the 383 MIPLIB 2017
instances used in the PDLP papers (100k–10M nonzeros). With default solver
settings, HiGHS presolve, a relative tolerance of `1e-4` and a 3600-second
limit per instance, Jaddle certifies optimality on **376 of the 383**
instances. Two instances are rendered empty by HiGHS presolve. We achieved an SGM10 of 7.2 seconds for the full solve, with a corrected score of 2.96 seconds when removing Jaddles problem scaling and compile time overheads. We do not converge on the fhnw-binschedule0, fhnw-binschedule1, hgms30, hgms62 and map16715-04 instances. However, by changing the cost_col_floor input option, which alters problem scaling, we can rescue convergence on fhnw-binschedule0.

The benchmark harnesses, instance diagnostics and instructions for reproducing
these results are described in [`benchmarks/README.md`](https://github.com/BrendanVR/Jaddle/blob/main/benchmarks/README.md).

## Repository layout

| Path | Contents |
|---|---|
| [`jaddle/jaddle_linear.py`](https://github.com/BrendanVR/Jaddle/blob/main/jaddle/jaddle_linear.py) | LP solver, scaling, certification and polishing |
| [`jaddle/jaddle_convex.py`](https://github.com/BrendanVR/Jaddle/blob/main/jaddle/jaddle_convex.py) | Convex solver |
| [`jaddle/jaddle_basic_types.py`](https://github.com/BrendanVR/Jaddle/blob/main/jaddle/jaddle_basic_types.py) | `SaddleState`, `JaddleCP`, `LP` and `JaddleLP` |
| [`jaddle/jaddle_optimisers.py`](https://github.com/BrendanVR/Jaddle/blob/main/jaddle/jaddle_optimisers.py) | Precision profiles and Optax saddle-point optimisers |
| [`jaddle/presolve.py`](https://github.com/BrendanVR/Jaddle/blob/main/jaddle/presolve.py) | Defined-variable elimination and postsolve |
| [`jaddle/highs_helpers.py`](https://github.com/BrendanVR/Jaddle/blob/main/jaddle/highs_helpers.py), [`sciopt_helpers.py`](https://github.com/BrendanVR/Jaddle/blob/main/jaddle/sciopt_helpers.py), [`glop_helpers.py`](https://github.com/BrendanVR/Jaddle/blob/main/jaddle/glop_helpers.py) | Converters and presolve interfaces for HiGHS, SCIP and OR-Tools glop |
| [`examples/`](https://github.com/BrendanVR/Jaddle/blob/main/examples/README.md) | Worked LP and convex examples |
| [`benchmarks/`](https://github.com/BrendanVR/Jaddle/blob/main/benchmarks/README.md) | MIPLIB and convex benchmark harnesses |
| [`tools/glop_presolve/`](https://github.com/BrendanVR/Jaddle/tree/main/tools/glop_presolve) | C++ helper exposing glop's presolver |
| [`tests/`](https://github.com/BrendanVR/Jaddle/tree/main/tests) | Test suite (`pytest`) |

## License

Jaddle is released under the MIT License. See [LICENSE.txt](https://github.com/BrendanVR/Jaddle/blob/main/LICENSE.txt).
