# Changelog

## Unreleased

### Added

- `jl.solve_with_refinement`: `jl.solve` with iterative refinement
  (Gleixner, Steffy & Wolter, INFORMS JOC 2016). A first solve stops at a
  loose `inner_tol` (1e-4). Each round then solves a correction LP at the
  current point and adds its scaled-down solution to the point. The
  correction LP is the same problem shifted to the point, with the primal
  residual as its right-hand side and the reduced cost as its objective,
  both scaled up to O(1) by powers of 2. Each round restarts the solver on a
  fresh, well-scaled problem, so the feasibility and gap tails of a long
  solve become the first epochs of a short one. The correction LP is in
  slack form, so its matrix is built once and every round reuses the same
  compiled programs. The best point is returned. Refinement stops when the
  point certifies, after `max_rounds` (8) rounds, or when a round fails to
  halve the worst residual-to-tolerance ratio. The result has a
  `"refinement"` dict with the rounds, epochs per round, scales and residual
  history. On stp3d (float64, presolved, `tol=1e-8`, gap `1e-4`) it
  certified in one round, about 94 s, with the gap at 9.7e-9. A plain solve
  is still uncertified after 150 s. Polishing is faster there (24 s), but
  only refinement also brings the gap down to the tolerance. On small
  random LPs in float64 it is no better than a plain solve. The
  experimental `fix_threshold` fixes near-bound columns in the correction
  LP and is off by default: it broke stp3d's dual. Use it on presolved
  models with `jl.solve_with_presolve(model,
  reduced_solver=jl.solve_with_refinement)`. LPs only: conic problems raise
  `NotImplementedError`.

- `benchmarks/benchmark.py --refine` solves with `jl.solve_with_refinement`
  (it cannot be combined with `--polish`). The new CSV column
  `jaddle_refine_rounds` records the number of rounds.

- Exponential cones: `jl.Cones(exp=k)` adds `k` three-variable exponential
  cones to the tail of `x`, projected with Friberg's root search (ported from
  SCS, with each point projected at unit norm: SCS's absolute tolerances took
  a heuristic point 40% off at a norm of 1e-3). The dual test on cone blocks
  is now `−Π_K(−r)`, correct for cones that are not self-dual.
  `jcv.JaddleConicSolver` accepts CVXPY's `ExpCone`, so `log`, `log_det`,
  `log_sum_exp`, `logistic`, `entr`, `rel_entr` and friends go through it.
  See `examples/conic/cvxpy_kelly.py`. Inside the solver each exponential
  cone's root search starts from the root found at the previous iteration,
  and Newton accepts its step once it falls below 1e-9 (quadratic
  convergence leaves an error near 1e-18) rather than SCS's 1e-15: the
  projection is 1.3-2.2x faster and an exponential-cone epoch about 20%
  faster on CPU, 10-20% on GPU. Off CPU the projection's first four Newton
  steps are unrolled so they fuse into one kernel (a rolled loop pays a
  kernel launch per step): a 100-asset, 1000-scenario Kelly epoch went from
  69 to 50 ms on GPU, while CPU, where unrolling was slower, is unchanged.

- Second-order and PSD cones in the LP solver. `jl.LP(..., cones=jl.Cones(
  soc=..., psd=...))` constrains the tail of `x` to a product of cones (PSD
  blocks in `svec` form); each step projects onto them, scaling and the
  diagonal metric keep one scale per cone block, and the certificate measures
  dual infeasibility as the reduced cost's distance to the cone. `jl.solve`,
  `make_solver`, `solve_batch` and `make_optimal_value` take conic problems;
  presolve, crossover, polishing, infeasibility detection and `make_solution`
  raise `NotImplementedError` for them. Pure LPs are unaffected (bit-identical
  results). New module `jaddle.cones`.

- `jcv.JaddleConicSolver`: a CVXPY conic solver for problems with zero,
  nonnegative, second-order and PSD cones. See `examples/conic/`.
  A cone constraint on a variable itself (`X >> 0` for a variable `X`) puts
  the cone on that variable directly, instead of on slack copies tied to it
  by equality rows. On the 200-node Max-Cut SDP that took about 6,100
  iterations and 9.2 s down to about 2,000 and 4.0 s
  (`cone_substitution=False` restores the slack form).

- `jcv.solve` compiles CVXPY's `entr` atom. Below `jcv.ENTR_FLOOR` (1e-8)
  it is continued by its tangent line, so iterates outside `entr`'s domain
  get a finite value and gradient. The result is exact when the solution
  keeps the argument above the floor. See
  `examples/cp/cvxpy_channel_capacity.py`.

- `examples/cp/cvxpy_robust_kalman.py`: CVXPY's robust Kalman filtering
  example, the least-squares smoother on the QP route and the Huber-of-norm
  smoother on the compiled route.

### Changed

- The adaptive step size's retry is no longer a loop inside each PDHG
  iteration. A trial step that overshoots is rejected and the iteration does
  not move; the next one retries from the same point with the shrunk step,
  and the iteration count advances only on accepted steps. The accepted
  steps are the same as before, but a rejection (0-2% of steps on the LPs
  checked) now uses up one of the epoch's iterations, so results can differ
  slightly. The data-dependent loop cost about 30 us per iteration on GPU:
  map18 went from 307 to 248 us per iteration and a small LP from 137 to
  104 us. CPU is unchanged or slightly faster.

- `jl.solve` no longer recompiles for every call. Its device programs are
  cached per settings and problem shape, and the problem's numbers go in as
  arguments rather than being baked into the compiled code. A second solve of
  the same shapes skips tracing and compiling: a small LP re-solve takes
  0.05 s instead of 0.8 s, a 500-asset CVXPY portfolio sweep 0.08–0.11 s per
  point instead of 1.1 s, and uccase12 0.6 s instead of 3.9 s. Per-epoch speed
  and results are unchanged (bit-identical on CPU). `solve_with_polishing`'s
  sub-solves and repeated benchmark runs benefit the same way.

- LP `make_solver`, `solve_batch`, `make_optimal_value`, `make_solution` and
  `make_perturbed_solution` now take a builder, as the convex ones do:
  `build_lp(params)` returns an `LP` or `JaddleLP`, and the returned function
  takes `params` (any pytree). Gradients are carried back through
  `build_lp`, so costs, right-hand sides, bounds and matrix entries can all
  depend on `params`. The sparsity pattern is read from whatever does not
  depend on `params`. The old `LPValues` interface is
  `make_solver(jaddle_lp.with_values)`, called with an `LPValues`. Breaking:
  the functions no longer take a prototype LP, `solve_fn()` no longer
  defaults to the LP's own values, the `.pattern` attribute is gone,
  `solve_batch(build_lp, params, in_axes=0)` replaces
  `solve_batch(lp, values)`, and the perturbed solution is
  `x_fn(params, key)`.
- `to_jaddle_sparse` (and `JaddleLP`) accept traced numbers. A dense matrix
  that is itself traced stores every entry, and a `BCOO` keeps its stored
  entries.

## 0.2.2

### Added

- `jaddle.jaddle_cvxpy`, a CVXPY front end (`pip install "jaddle[cvxpy]"`).
  `JaddleSolver` is a CVXPY QP solver, used as
  `prob.solve(solver=JaddleSolver())`. It solves LPs with
  `solve_with_presolve` and QPs with `quadratic_program`. `jcv.solve(prob)`
  compiles any other DCP problem into a `JaddleCP` for the convex solver.
  Smooth atoms (logistic, log-sum-exp, Huber, ...) stay smooth there, and
  piecewise-linear ones become epigraph rows, avoiding CVXPY's
  exponential-cone reformulation. Results are written back as CVXPY's
  (status, values, duals). See `examples/lp/cvxpy_transport.py` and
  `examples/cp/cvxpy_lasso.py`.

- LP and convex `solve`/`make_solver`: `diag_adapt` (with `diag_theta` and
  `diag_clamp`), a per-coordinate diagonal step metric on top of the scaling,
  adapted at each restart. The convex solver supports it only for adaptive
  extragradient with restarts, and leaves it off by default.

- LP `solve`/`make_solver`: `gap_test` selects the relative gap tested
  against `dual_gap_tolerance`. `"pdlp"` is PDLP's signed
  |primal_obj − dual_obj|; `"abs"` sums the absolute complementarity terms so
  they cannot cancel.

- Convex `solve`/`make_solver`: `gradient_scaling` (with `gradient_scaling_probes`)
  rescales the objective and each constraint row by its gradient norm at the
  start point, as IPOPT does. Off by default; the CVXPY compiler turns it on.

### Fixed

- Convex `solve`: a solve whose residuals had become NaN (a diverged run)
  passed the convergence test and was reported as certified. The test now
  requires every residual to be within tolerance, so NaN fails it.

### Changed

- LP `solve`/`make_solver`: `diag_adapt=True` is the default. On the MIPLIB
  2017 LP set it certifies 380/383 presolved instances (376 before) and cuts
  the shifted geometric-mean solve time from 8.62 s to 7.04 s.
- LP certificate: the default gap test is now `gap_test="pdlp"` (the signed
  gap, as in PDLP and cuPDLP). Pass `gap_test="abs"` for the previous,
  stricter no-cancellation test.

## 0.2.1

### Changed

- Convex `solve`: a solve that meets its convergence criteria now reports
  `stop_reason="certificate"`, as the LP solver does, instead of
  `"converged"`.
- Convex `solve` no longer prints its end-of-solve summary unless
  `verbose=True`. In both solvers the max-epochs and time-limit messages are
  printed only with `verbose=True`; the Ctrl-C notice is still always printed.

### Added

- `CITATION.cff`, for GitHub's "Cite this repository".
- Docstrings for `jaddle_linear.to_jaddle_sparse`,
  `jaddle_linear.lp_summary_statistics` and the optimisers in
  `jaddle_optimisers`.

## 0.2.0

### Added

**LP solver (`jaddle.jaddle_linear`)**

- `load`: read an MPS or LP file into a `JaddleLP`, optionally presolved by
  HiGHS.
- `solve_with_presolve`: HiGHS presolve, a Jaddle solve of the reduced LP and
  HiGHS postsolve of the primal and dual solution. The result is certified on
  the LP as given (`converged`), with the reduced solve's verdict in
  `reduced_converged`. Options `polish=True` and `run_crossover=True`.
- `crossover`: take a first-order solution to an optimal basic solution with
  HiGHS.
- `detect_infeasibility` and `evaluate_infeasibility_certificate`: PDLP-style
  primal and dual infeasibility detection.
- Batched and differentiable solves: `make_solver`, `solve_batch`,
  `make_optimal_value`, `make_solution`, `make_perturbed_solution`, the
  `LPValues` pytree and `SolveCoreResult`.
- `solve_with_polishing` now takes the three certificate tolerances, a time
  limit and an objective row for the primal polish
  (`polish_objective_slack`), with a configurable trigger (`polish_trigger`,
  `first_polish_epoch`, `polish_ready_factor`, `polish_fraction`).
- `evaluate_lp_certificate` takes `norm` and `dual_residual`.

**Convex solver (`jaddle.jaddle_convex`)**

- `quadratic_program`: QPs from `Q`, `c` and constraint matrices, with
  symmetric KKT equilibration and an LP-style certificate in the problem's
  own units.
- Batched and differentiable solves: `make_solver`, `make_optimal_value`,
  `make_solution` and `SolveCoreResult`.
- `report_best` (default on): test and return whichever of the average and the
  last iterate has the lower KKT merit.
- `JaddleCP` takes an optional `residuals` function to supply a problem's own
  stopping test.

**HiGHS helpers (`jaddle.highs_helpers`)**

- `read_model`, `jaddle_lp_to_highs`, `highs_duals_to_jaddle`,
  `jaddle_duals_to_highs`, `standard_form_row_map` and
  `standard_form_basis_to_highs`.

**Examples**

- `lp/battery_sizing.py`, `cp/lasso_regression.py`,
  `cp/fair_logistic_regression.py` and `cp/kernel_svm.py`.
- `lp/miplib.py` replaces `lp/miplib_standard.py`.

### Changed

- LP and convex `solve`: `epochs_per_restart` now defaults to `None` (no cap
  on the restart cycle length) instead of `10`.
- Convex `solve`: `update_mode` defaults to `"alternating"` instead of
  `"extragradient"`, and `average` defaults to `True`.
- The LP iteration loop runs on the device in chunks of about a second, which
  makes solves 2–3× faster. Runs are no longer bit-identical to 0.1.0's after
  a restart.

### Removed

- LP `solve`: `eq_projection_threshold`.
- `solve_with_polishing`: `max_rounds`.

### Fixed

- LP `solve`: the relative dual feasibility residual left out the cost
  normalisation factor. Under the default augmented scaling the factor is
  close to 1, but with `scaled_augmented=False` the solver could report false
  certificates.
- LP `solve`: a warm start's duals were mapped into the scaled problem without
  the cost normalisation factor, so warm starts did not start from the point
  supplied.

## 0.1.0

First release.
