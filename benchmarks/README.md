# Jaddle Benchmarks

Scripts for measuring Jaddle's solvers. They come in three kinds:

| Kind | Scripts | Purpose |
|---|---|---|
| **LP harnesses** | [`benchmark.py`](benchmark.py), [`benchmark_sciopt.py`](benchmark_sciopt.py), [`benchmark_glop.py`](benchmark_glop.py) | Solve the LP relaxations of a directory of MIPLIB instances with `jaddle_linear.solve`, then write a CSV and a README-ready markdown table. They differ only in the presolver used, and in whether a reference optimum is computed. |
| **Convex A/B suites** | [`benchmark_convex.py`](benchmark_convex.py), [`benchmark_convex_heavy.py`](benchmark_convex_heavy.py) | Compare `jaddle_convex.solve` configurations (update mode, adaptive step, restarts) on synthetic convex problems. |
| **Instance diagnostics** | [`scan_bigm.py`](scan_bigm.py) | Flag MPS instances whose big-M structure is known to stall saddle-point solvers. The LP harnesses use it to skip those instances. |

Run every script from the repo root as `python benchmarks/<script>.py`. The
scripts import from each other (`benchmark_sciopt` imports `run_jaddle` from
`benchmark`, and all the LP harnesses import the detectors from `scan_bigm`).
Python resolves these imports because it puts the script's own directory on
`sys.path`.

## Getting the data

The MPS files are **not** shipped with Jaddle (`data/` is gitignored). Download
instances from the [MIPLIB website](https://miplib.zib.de/) and put the
`.mps` files in `data/` at the repo root. The LP harnesses and `scan_bigm.py`
run on every `*.mps` file in that directory. Use `--data-dir` to point them
somewhere else.

The PDLP papers' reference set is the 383-instance list from FirstOrderLp.jl
(`mip_relaxations_instance_list`, 100k–10M nonzeros).

## LP harnesses

All three harnesses run Jaddle with the same settings (`jaddle_solve_kwargs` in
`benchmark.py`): `iterations_per_epoch=640`, `epochs_per_restart=10`, and the
primal, dual and gap tolerances all set to `--tol`.

`benchmark.py` hands each instance file to `jl.solve_with_presolve`: HiGHS
presolve, the defined-variable elimination, Jaddle on the reduced LP, then
postsolve of both the primal and the dual back to the original problem. The
reported objective is the original one. `benchmark_sciopt.py` and
`benchmark_glop.py` presolve with SCIP or glop, which offer no postsolve here,
and solve the reduced LP with `run_jaddle`. They add the presolve's constant
objective offset back so the reported objective refers to the full problem.

| Script | Presolve | Reference optimum | Default CSV |
|---|---|---|---|
| `benchmark.py` | HiGHS | Yes. A HiGHS solve (`simplex` by default) is used as a ground-truth objective oracle, and the relative gap to it is reported. | `benchmark_results.csv` |
| `benchmark_sciopt.py` | SCIP (PySCIPOpt) | No | `benchmark_sciopt_results.csv` |
| `benchmark_glop.py` | OR-Tools glop (PDLP's presolver and settings) | No | `benchmark_glop_results.csv` |

CSVs are written to the repo root. `benchmark.py` rewrites its CSV after every
instance, so a crash partway through a sweep keeps the results so far. It also
clears JAX's compile caches between instances to keep host memory flat over
long runs.

### Usage

```bash
# Every instance in data/ (100 MB cap by default)
python benchmarks/benchmark.py

# A named subset
python benchmarks/benchmark.py --only stp3d boeing

# A tighter tolerance and a per-instance time budget
python benchmarks/benchmark.py --tol 1e-4 --max-seconds 600

# Jaddle only: skip the HiGHS reference solve (much faster on large LPs)
python benchmarks/benchmark.py --highs-solver none

# The same sweep with a different presolver
python benchmarks/benchmark_sciopt.py --max-mb 50
python benchmarks/benchmark_glop.py --skip-bigm --skip-bigm-column
```

`benchmark_glop.py` needs a small C++ helper, because the OR-Tools Python
bindings don't expose glop's presolver. Build it once:

```bash
tools/glop_presolve/build.sh   # downloads the matching OR-Tools C++ release into third_party/
```

### Options shared by all three harnesses

| Option | Default | Meaning |
|---|---|---|
| `--data-dir DIR` | `data/` | Directory to glob for `*.mps` files. |
| `--only NAME ...` | all | Restrict the run to these instance names (without `.mps`). |
| `--max-mb X` / `--min-mb X` | none / `0` | Skip files above or below a size in MB. By default there is no upper limit. |
| `--tol X` | `1e-3` | Relative tolerance for primal feasibility, dual feasibility and duality gap. |
| `--max-epochs N` | none | Cap on Jaddle epochs. |
| `--max-seconds S` | none | Per-instance Jaddle wall-clock budget, including setup and XLA compile. It is checked at epoch boundaries. |
| `--update-mode` | `alternating` | Passed to `jl.solve`: `alternating`, `pdhg` or `halpern`. |
| `--cost-col-floor X` | `0` | Passed to `jl.solve` as `cost_col_floor`. A costed column whose largest scaled matrix entry is below `X` is rescaled so that entry becomes 1. This fixes epigraph objectives such as fhnw-binschedule0. `0` disables it, as in `jl.solve`'s default; try `1e-2` to enable it. |
| `--jax-profile` | `float64` | Passed to `configure_jax`: `float64`, `float32` or `float16`. |
| `--skip-bigm` | off | Skip instances with cost big-M (see [`scan_bigm.py`](#scan_bigmpy)). |
| `--skip-bigm-matrix` | off | Skip instances with matrix (row) big-M. |
| `--skip-bigm-column` | off | Skip instances with column big-M. |
| `--eliminate-defined-vars` | on | After presolve, substitute out variables defined by a dense equality row `z = Σ aⱼxⱼ` (see `jaddle.presolve.eliminate_defined_variables`). Use `--no-eliminate-defined-vars` to turn it off. |
| `--jaddle-verbose` | on | Run Jaddle with `verbose=True`, printing its per-epoch log and restart messages. Use `--no-jaddle-verbose` to silence it; with no `--max-epochs` or `--max-seconds` budget either, each solve then runs as a single device call. |
| `--csv PATH` | see table above | Where to write the results. |

### Options for a single harness

| Script | Option | Meaning |
|---|---|---|
| `benchmark.py` | `--highs-solver {simplex,ipm,pdlp,none}` | Solver for the reference optimum. Use `none` to skip it, in which case `rel_obj_gap` is NaN. The reference solve often takes longer than Jaddle's, so skip it for A/B runs if you already know the optima. |
| `benchmark.py` | `--highs-kkt-tolerance X` | KKT tolerance for the reference solve (defaults to `--tol`). |
| `benchmark.py` | `--highs-verbose` | Show HiGHS's own log. |
| `benchmark_sciopt.py` | `--scip-verbose` | Show SCIP's presolve log. |
| `benchmark_glop.py` | `--glop-verbose` | Show glop's presolve log. |

### Reading the output

Three timings are reported for Jaddle:

- **`jaddle_solve_seconds`**: time spent in the iteration loop, including the
  first-epoch XLA compile but excluding scaling and setup. This is the headline
  figure.
- **`jaddle_corrected_seconds`**: the same, with the one-off compile amortised
  out: `n · (solve − first_epoch) / (n − 1)`. It estimates the runtime if
  every epoch had run at the warm rate.
- **`jaddle_wall_seconds`**: the whole call. In `benchmark.py` that includes
  presolve (`jaddle_presolve_seconds`), postsolve and any finishing solve.

`jaddle_converged` can be true for more than one reason, so check
`jaddle_stop_reason` alongside it:

| `jaddle_stop_reason` | Meaning |
|---|---|
| `certificate` | Full LP optimality certificate met (primal feasibility, dual feasibility and gap). |
| `primal_stall` | Primal-stop heuristic fired: the point is feasible but not certified optimal. |
| `max_epochs` / `time_limit` | Budget exhausted. |
| `presolve_infeasible` / `presolve_solved` | HiGHS presolve settled the problem itself (`benchmark.py` only). |

`jaddle_converged` means certified on the **presolved** LP, as in earlier
sweeps. `benchmark.py` also judges the postsolved point on the LP as given:
`jaddle_original_certified`, with that certificate's residuals in
`jaddle_orig_pfr`, `jaddle_orig_dfr` and `jaddle_orig_gap`. When the point
misses the original certificate, up to 50 epochs of `jl.solve` on the original
LP, warm-started from it, try to finish (`jaddle_finish_epochs`). The two
verdicts can disagree when presolve rescales the problem, because the
tolerances are relative. leo1's presolve folds a row with 1e7 coefficients into
the cost (‖c‖ 1 → 3e9): the objective is right, but the original's dual test
fails. A certificate on a drastically rescaled reduced LP can also be false.
On proteindesign122trx11p8 (‖c‖ 1 → 6e6 after elimination), one run
"certified" the reduced LP after 13 epochs with an objective 7% above the
optimum, and `jaddle_original_certified` was false.

`benchmark.py` also reports `rel_obj_gap = |jaddle − opt| / (1 + |opt|)`
(PDLP's normalisation) against the HiGHS optimum, and both the original and the
presolved sizes (`n_vars`/`n_cons`, `n_vars_presolved`/`n_cons_presolved`,
counted as HiGHS columns and rows).

## Convex A/B suites

Positional arguments select which configs or modes to run. The only flag is
`--jaddle-verbose` (off by default), which shows Jaddle's own per-epoch log
above each result line.

### `benchmark_convex.py`

Runs every (problem, config) pair on a small suite of synthetic problems.
Each run has a budget of 300 epochs × 200 iterations and uses tolerance `1e-6`.

| Problem | Structure |
|---|---|
| `isotonic` | 2000-variable least squares with ordering inequalities and box bounds |
| `ridge_ball` | RFF regression with an active `‖w‖² ≤ r` constraint |
| `simplex_qp` | Badly scaled QP on the probability simplex (equality constraint plus bounds) |
| `small_lp` | Random feasible bounded LP, posed as a convex program |
| `logistic_l1` | Sparse logistic regression with an L1-ball constraint written as many inequalities |

| Config | Settings |
|---|---|
| `alt_gd` | `alternating` with plain gradient descent (`jo.gd(0.05)`) |
| `eg_ad` | `extragradient` with adaptive step |
| `eg_ad_rs` | `eg_ad` with restarts every 5 epochs |
| `frb_ad` | `forward_reflected` with adaptive step |
| `frb_ad_rs` | `frb_ad` with restarts every 5 epochs |

```bash
python benchmarks/benchmark_convex.py              # all configs
python benchmarks/benchmark_convex.py eg_ad frb_ad # only these configs
python benchmarks/benchmark_convex.py eg_ad --jaddle-verbose  # with Jaddle's log
```

Each line of output gives convergence, epochs, solve time (excluding compile)
and final objective.

### `benchmark_convex_heavy.py`

A single gradient-heavy problem: dense L1-constrained logistic regression,
20000 × 1000. Each gradient costs two dense matvecs, so per-iteration cost is
dominated by gradient evaluations. The small suite above is dominated by
overhead instead. Use this script to weigh cost per epoch against epochs to
converge across update modes.

```bash
python benchmarks/benchmark_convex_heavy.py                    # extragradient and forward_reflected
python benchmarks/benchmark_convex_heavy.py forward_reflected  # one mode
```

Each line of output gives convergence, epochs, total time, time per epoch and
objective.

## `scan_bigm.py`

Scans every `.mps` file in a directory and flags three independent kinds of
big-M / penalty structure. Each one stalls Jaddle in a different way, and Ruiz
scaling cannot remove any of them:

| Flag | Detector | Rule (defaults) | Typical instances | Failure mode |
|---|---|---|---|---|
| **Cost** | `is_bigm_cost` | `max\|c\| / median\|c\| > ratio`, and either `max\|c\| > abs` or `median\|c\| < med-floor` | `glass4`, `binkar10_1`, `mas74`, `mas76` | Stalls at a feasible point with a frozen duality gap, well short of the optimum |
| **Matrix (row)** | `is_bigm_matrix` | median per-row spread `max\|A_row\|/min\|A_row\| > 1e3`, and at least 50% of rows wide (spread > `1e2`) | `sp150x300d`, `neos-3754480-nidda` | Plateau in primal feasibility |
| **Column** | `is_bigm_column` | at least one dense big-M row: `max\|A_row\| > 1e3` and touching at least 30% of the columns | `germanrr`, `leo1`, `trento1` | Primal never approaches feasibility |

```bash
python benchmarks/scan_bigm.py
python benchmarks/scan_bigm.py --data-dir data --ratio 1e3 --abs 1e4
python benchmarks/scan_bigm.py --row-spread 1e3 --row-frac 0.5 --wide-row 1e2
python benchmarks/scan_bigm.py --col-row-abs 1e3 --col-row-density 0.3
```

Each MPS file is read once and all three checks run on it. The output is one
table per flag, sorted by severity, followed by a list of the flagged names.
The LP harnesses' `--skip-bigm*` options call the same detector functions with
the same default thresholds, so the scanner's output is exactly what those
options will skip.

## Tips

- **A/B comparisons:** pass `--highs-solver none` to `benchmark.py`, or use
  the SCIP or glop harness, so the reference solve doesn't dominate wall time.
- **GPU nondeterminism:** identical GPU runs can differ by several epochs,
  because scatter-add order varies. Repeat both arms before trusting a small
  difference in epoch counts.
- **Large sweeps:** use `--max-seconds` to bound how long any one instance
  can take, and `--max-mb`/`--min-mb` to choose a size band.
