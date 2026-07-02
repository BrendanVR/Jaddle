# %% [markdown]
# # Jaddle LP Benchmark Harness
#
# Runs Jaddle's saddle-point LP solver against HiGHS-PDLP (a same-class
# first-order method) on a set of MIPLIB LP relaxations, and emits a
# README-ready markdown table plus a CSV.
#
# The MPS files are **not** shipped with Jaddle (they are large and
# gitignored). Download the instances you want from the MIPLIB website
# (https://miplib.zib.de/) and drop the `.mps` files into the `data/`
# directory at the repo root. This harness globs whatever is present there.
#
# Usage:
#     python examples/lp/benchmark.py
#     python examples/lp/benchmark.py --max-mb 50          # skip huge instances
#     python examples/lp/benchmark.py --only stp3d boeing  # subset by name
#     python examples/lp/benchmark.py --tol 1e-4 --max-epochs 500

# %%
import os

# Suppress INFO and WARNING logs from XLA/JAX before importing JAX.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import csv
import glob
import time

import highspy as hspy

import jaddle.jaddle_optimisers as jo
import jaddle.jaddle_linear as jl
import jaddle.highs_helpers as hh

import jax.numpy as jnp
import jax.experimental.sparse as jsp

from scan_bigm import is_bigm_cost, is_bigm_matrix, is_bigm_column

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")


def parse_args():
    p = argparse.ArgumentParser(description="Jaddle vs HiGHS-PDLP LP benchmark.")
    p.add_argument(
        "--data-dir", default=DATA_DIR, help="Directory of .mps files to glob."
    )
    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Problem names (without .mps) to restrict to. Default: all in data-dir.",
    )
    p.add_argument(
        "--max-mb",
        type=float,
        default=100.0,
        help="Skip .mps files larger than this many MB (default 100; huge "
        "instances can OOM or run for very long). Set 0 to disable.",
    )
    p.add_argument(
        "--min-mb",
        type=float,
        default=0.0,
        help="Skip .mps files smaller than this many MB (default 0; raise to "
        "target larger LPs only).",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="Relative optimality tolerance for both solvers (default 1e-3).",
    )
    p.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Cap Jaddle epochs (None = run to convergence).",
    )
    p.add_argument(
        "--csv",
        default=os.path.join(REPO_ROOT, "benchmark_results.csv"),
        help="Path to write CSV results.",
    )
    p.add_argument(
        "--highs-solver",
        default="simplex",
        choices=("simplex", "ipm", "pdlp", "none"),
        help="HiGHS solver used for the reference optimum (default simplex). "
        "Pass 'none' to skip the reference solve entirely and report NaN for the "
        "optimum (and rel_obj_gap) -- useful for very large LPs where the exact "
        "solve is prohibitively slow; jaddle still runs and its objective is "
        "reported.",
    )
    p.add_argument(
        "--skip-bigm",
        action="store_true",
        help="Skip instances with a big-M / penalty COST structure (wide "
        "dynamic range in the objective coefficients, e.g. glass4, binkar10_1). "
        "These stall the saddle-point solver at a feasible-but-suboptimal point "
        "and waste the epoch budget. See benchmarks/scan_bigm.py.",
    )
    p.add_argument(
        "--skip-bigm-matrix",
        action="store_true",
        help="Skip instances with a big-M / penalty MATRIX structure (rows where "
        "one coefficient dwarfs its rowmates, e.g. sp150x300d, neos-3754480-"
        "nidda). Like cost big-M but constraint-side: Ruiz scaling cannot flatten "
        "the within-row skew, so the saddle solver stalls on a primal-feasibility "
        "plateau. See benchmarks/scan_bigm.py.",
    )
    p.add_argument(
        "--skip-bigm-column",
        action="store_true",
        help="Skip instances with a big-M / penalty COLUMN structure (a dense "
        "row with a huge coefficient shared across most columns, e.g. germanrr, "
        "leo1, trento1). The column-side mirror of matrix big-M: Ruiz cannot "
        "flatten within-column anisotropy threaded through one shared row, so the "
        "saddle solver never approaches primal feasibility (germanrr: PFR frozen "
        "~1e7, objective still drifting at 80 epochs). See benchmarks/scan_bigm.py.",
    )
    return p.parse_args()


def load_relaxed_lp(path, highs_solver="simplex"):
    """Load an MPS file via HiGHS, relax integrality, solve for a trusted
    reference objective with the requested HiGHS solver, and convert to Jaddle's
    sparse standard form.

    HiGHS is used here only as an objective *oracle* (ground-truth optimum), not
    as a speed competitor -- a same-class PDLP-vs-PDHG timing race is misleading
    because the two stop on different relative-gap criteria. An exact solver
    (``simplex`` / ``ipm``) is an unambiguous answer key for Jaddle's objective
    gap; ``pdlp`` is available for a same-class comparison point.

    ``highs_solver`` selects the HiGHS solver: ``simplex``/``ipm``/``pdlp``, or
    ``none`` to skip the reference solve entirely (the exact solve can be
    prohibitively slow on very large LPs); then ``opt_obj`` is returned as NaN
    and ``highs_status`` as "skipped". The presolve + conversion still run, since
    jaddle solves the presolved LP and needs its offset.

    Returns (jaddle_lp, opt_obj, highs_status, highs_seconds, offset), where
    `highs_seconds` is the wall time of the HiGHS reference solve (NaN when
    skipped) and `offset` is the presolved model's constant objective offset (add
    it to jaddle's c^T x to compare against the full-problem `opt_obj`).
    """
    highs = hspy.Highs()
    highs.readModel(path)

    # Relax integrality so we solve the LP relaxation.
    for col in range(highs.numVariables):
        highs.changeColIntegrality(col, hspy.HighsVarType.kContinuous)

    highs.setOptionValue("output_flag", "false")

    if highs_solver == "none":
        opt_obj = float("nan")
        highs_status = "skipped"
        highs_seconds = float("nan")
    else:
        # Solve for the ground-truth objective with the requested HiGHS solver.
        # Default tolerances. Time only the run() call (excl. read/relax), the
        # like-for-like counterpart to jaddle's solve-only timer.
        highs.setOptionValue("solver", highs_solver)
        t0 = time.perf_counter()
        highs.run()
        highs_seconds = time.perf_counter() - t0

        info = highs.getInfo()
        opt_obj = info.objective_function_value
        highs_status = highs.modelStatusToString(highs.getModelStatus())

    highs.presolve()
    highs_lp = highs.getPresolvedLp()
    jaddle_lp = hh.highs_to_standard_form_sparse(highs_lp)

    if jaddle_lp.A_ineq.shape == (0, 0) and jaddle_lp.A_eq.shape == (0, 0):
        raise ValueError(
            f"Presolved LP {path} has no constraints (A_ineq and A_eq are empty)."
        )

    # Presolve folds eliminated variables' cost contributions into a constant
    # objective offset that lives OUTSIDE the reduced `c` vector. The converter
    # (highs_to_standard_form_sparse) reads only col_cost_, so jaddle's objective
    # is c^T x on the reduced problem; HiGHS reports c^T x + offset_. We carry the
    # offset out so the reported objective is comparable to the full-problem
    # optimum `opt_obj`. The offset is a pure constant — it shifts the objective
    # but not the argmin, so it stays a reporting concern and never enters solve().
    offset = float(highs_lp.offset_)
    return jaddle_lp, opt_obj, highs_status, highs_seconds, offset


def run_jaddle(jaddle_lp, tol, max_epochs):
    """Solve with Jaddle's saddle-point solver. Returns a dict of metrics.

    Three times are reported:
      * ``jaddle_solve_seconds`` -- the solver's internal iterate-loop time
        (incl. first-epoch XLA compile, excl. scaling/sparse setup). This is
        the like-for-like figure against HiGHS-PDLP's own ``run()`` timer,
        which likewise excludes problem setup. Use this in the headline table.
      * ``jaddle_corrected_seconds`` -- the same solve-loop time with the one-off
        first-epoch XLA compile amortised out:
        ``n_epochs * (solve - first_epoch) / (n_epochs - 1)``. Estimates the
        steady-state runtime had every epoch run at the warm per-epoch rate.
      * ``jaddle_wall_seconds`` -- the full ``jl.solve()`` call including
        scaling and setup, for transparency.
    """

    t0 = time.perf_counter()
    result = jl.solve(
        jaddle_lp,
        verbose=True,
        primal_feasibility_tolerance=tol,
        dual_feasibility_tolerance=tol,
        dual_gap_tolerance=tol,
        update_mode="pdhg",
        halpern_reanchor_per_epoch=True,
        iterations_per_epoch=256,
        k_scale=1e3,
        k_theta=1e-3,
        adaptive_eta=0.0,
        max_epochs=max_epochs,
        restarts=100,
        restart_decay=0.9,
        epochs_per_restart=10,
        restart_multiplier=1.2,
        vertex_bias=1e-3,
    )
    wall_seconds = time.perf_counter() - t0

    solution = result["solution"]
    converged = result["converged"]
    stop_reason = result["stop_reason"]
    solve_seconds = result["solve_seconds"]
    corrected_seconds = result["corrected_seconds"]

    obj = float(jaddle_lp.objective(solution.primal))
    eq_res = float(jaddle_lp.eq_slack(solution.primal))
    ineq_res = float(jaddle_lp.ineq_slack(solution.primal))
    return {
        "jaddle_obj": obj,
        "jaddle_converged": bool(converged),
        # "certificate" = full LP optimality cert met; "primal_stall" = the
        # primal_stop heuristic fired (feasible but not certified optimal, so the
        # objective may be suboptimal even though converged=True); "max_epochs" =
        # budget exhausted. Disambiguates the two ways converged can be True.
        "jaddle_stop_reason": stop_reason,
        "jaddle_solve_seconds": solve_seconds,
        # Steady-state solve time with the first-epoch XLA compile amortised out:
        # n_epochs * (solve - first_epoch) / (n_epochs - 1). Falls back to
        # solve_seconds when there are fewer than two epochs.
        "jaddle_corrected_seconds": corrected_seconds,
        "jaddle_wall_seconds": wall_seconds,
        "jaddle_eq_res": eq_res,
        "jaddle_ineq_res": ineq_res,
    }


def rel_obj_gap(jaddle_obj, opt_obj):
    """Relative gap to the exact optimum |jaddle - opt| / (1 + |opt|),
    PDLP-convention normalisation."""
    return abs(jaddle_obj - opt_obj) / (1.0 + abs(opt_obj))


def discover_instances(args):
    paths = sorted(glob.glob(os.path.join(args.data_dir, "*.mps")))
    instances = []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        if args.only and name not in args.only:
            continue
        size_mb = os.path.getsize(path) / 1e6
        if args.max_mb and size_mb > args.max_mb:
            print(f"  skip {name} ({size_mb:.0f} MB > --max-mb {args.max_mb:.0f})")
            continue
        if args.min_mb and size_mb < args.min_mb:
            print(f"  skip {name} ({size_mb:.1f} MB < --min-mb {args.min_mb:.1f})")
            continue
        if args.skip_bigm and is_bigm_cost(path):
            print(f"  skip {name} (big-M cost structure; --skip-bigm)")
            continue
        if args.skip_bigm_matrix and is_bigm_matrix(path):
            print(f"  skip {name} (big-M matrix structure; --skip-bigm-matrix)")
            continue
        if args.skip_bigm_column and is_bigm_column(path):
            print(f"  skip {name} (big-M column structure; --skip-bigm-column)")
            continue
        instances.append((name, path, size_mb))
    return instances


def main():
    args = parse_args()
    jo.configure_jax("float64")

    instances = discover_instances(args)
    if not instances:
        print(
            f"No .mps files to run in {args.data_dir}. Download MIPLIB instances "
            "(https://miplib.zib.de/) into data/ and retry."
        )
        return

    print(f"Running {len(instances)} instance(s) at tol={args.tol:g}\n")
    rows = []
    for name, path, size_mb in instances:
        print(f"=== {name} ({size_mb:.1f} MB) ===")
        row = {"problem": name, "size_mb": round(size_mb, 1)}
        try:
            jaddle_lp, opt_obj, highs_status, highs_seconds, offset = load_relaxed_lp(
                path, highs_solver=args.highs_solver
            )

            row.update(
                {
                    "n_vars": int(jaddle_lp.num_variables()),
                    "n_cons": int(jaddle_lp.num_constraints()),
                    "opt_obj": opt_obj,
                    "highs_status": highs_status,
                    "highs_solve_seconds": highs_seconds,
                }
            )
            jres = run_jaddle(jaddle_lp, args.tol, args.max_epochs)
            # jaddle solves the presolved reduced problem (objective = c^T x);
            # add the presolve offset to compare against the full-problem opt_obj.
            jres["jaddle_obj"] += offset
            row.update(jres)
            row["offset"] = offset
            row["rel_obj_gap"] = rel_obj_gap(jres["jaddle_obj"], opt_obj)
            row["error"] = ""
            highs_time_str = (
                "skipped"
                if args.highs_solver == "none"
                else f"solve={highs_seconds:.2f}s"
            )
            print(
                f"  optimum (HiGHS {args.highs_solver}): {opt_obj:.6g} "
                f"({highs_time_str})  |  "
                f"Jaddle: obj={jres['jaddle_obj']:.6g} "
                f"(solve={jres['jaddle_solve_seconds']:.2f}s, "
                f"corrected={jres['jaddle_corrected_seconds']:.2f}s, "
                f"wall={jres['jaddle_wall_seconds']:.2f}s, "
                f"converged={jres['jaddle_converged']}, "
                f"rel_gap={row['rel_obj_gap']:.2e})"
            )
        except Exception as exc:  # keep the run going if one instance blows up
            row["error"] = repr(exc)
            print(f"  ERROR: {exc!r}")
        rows.append(row)
        print()

    write_csv(args.csv, rows)
    print_markdown(rows, highs_solver=args.highs_solver)
    print(f"\nCSV written to {args.csv}")


CSV_FIELDS = [
    "problem",
    "size_mb",
    "n_vars",
    "n_cons",
    "opt_obj",
    "offset",
    "highs_status",
    "highs_solve_seconds",
    "jaddle_obj",
    "jaddle_converged",
    "jaddle_stop_reason",
    "jaddle_solve_seconds",
    "jaddle_corrected_seconds",
    "jaddle_wall_seconds",
    "jaddle_eq_res",
    "jaddle_ineq_res",
    "rel_obj_gap",
    "error",
]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def _fmt(x, spec="{:.4g}"):
    if x == "" or x is None:
        return "—"
    try:
        return spec.format(x)
    except (ValueError, TypeError):
        return str(x)


def print_markdown(rows, highs_solver="simplex"):
    print("\n## Benchmark Results\n")
    if highs_solver == "none":
        oracle = (
            "Optimum is not computed (--highs-solver none), so rel_obj_gap is "
            "unavailable."
        )
    else:
        oracle = (
            f"Optimum is HiGHS's {highs_solver} solver, used as a ground-truth "
            "objective oracle."
        )
    print(
        f"_{oracle} Jaddle solve time is solve-only (iterate loop incl. "
        "first-epoch XLA compile, excl. setup/scaling); corrected time amortises "
        "the one-off first-epoch compile out "
        "(`n·(solve−first)/(n−1)`); see `jaddle_wall_seconds` in the CSV for full "
        "call time._\n"
    )
    print(
        "| Problem | Vars | Cons | Optimum | HiGHS solve (s) | "
        "Jaddle obj | Jaddle solve (s) | Jaddle corrected (s) | "
        "Converged | Stop | Rel. gap to opt |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|---:|")
    for r in rows:
        if r.get("error"):
            print(f"| {r['problem']} | — | — | — | — | — | — | — | ⚠️ error | — | — |")
            continue
        conv = "✅" if r.get("jaddle_converged") else "❌"
        print(
            f"| {r['problem']} | {_fmt(r.get('n_vars'), '{:d}')} | "
            f"{_fmt(r.get('n_cons'), '{:d}')} | {_fmt(r.get('opt_obj'))} | "
            f"{_fmt(r.get('highs_solve_seconds'), '{:.2f}')} | "
            f"{_fmt(r.get('jaddle_obj'))} | "
            f"{_fmt(r.get('jaddle_solve_seconds'), '{:.2f}')} | "
            f"{_fmt(r.get('jaddle_corrected_seconds'), '{:.2f}')} | {conv} | "
            f"{r.get('jaddle_stop_reason') or '—'} | "
            f"{_fmt(r.get('rel_obj_gap'), '{:.2e}')} |"
        )


if __name__ == "__main__":
    main()

# %%
