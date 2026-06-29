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
    return p.parse_args()


def load_relaxed_lp(path):
    """Load an MPS file via HiGHS, relax integrality, solve to optimality with
    HiGHS's exact solver to obtain a trusted reference objective, and convert to
    Jaddle's sparse standard form.

    HiGHS is used here only as an objective *oracle* (ground-truth optimum), not
    as a speed competitor -- a same-class PDLP-vs-PDHG timing race is misleading
    because the two stop on different relative-gap criteria. The exact optimum is
    an unambiguous answer key for Jaddle's objective gap.

    Returns (jaddle_lp, opt_obj, highs_status, offset), where `offset` is the
    presolved model's constant objective offset (add it to jaddle's c^T x to
    compare against the full-problem `opt_obj`).
    """
    highs = hspy.Highs()
    highs.readModel(path)

    # Relax integrality so we solve the LP relaxation.
    for col in range(highs.numVariables):
        highs.changeColIntegrality(col, hspy.HighsVarType.kContinuous)

    # Solve to optimality with HiGHS's default exact solver (simplex/IPM) to get
    # the ground-truth objective. Default tolerances; no PDLP.
    highs.setOptionValue("output_flag", "false")
    highs.run()

    info = highs.getInfo()
    opt_obj = info.objective_function_value
    highs_status = highs.modelStatusToString(highs.getModelStatus())

    highs.presolve()
    highs_lp = highs.getPresolvedLp()
    jaddle_lp = hh.highs_to_standard_form_sparse(highs_lp)
    # Presolve folds eliminated variables' cost contributions into a constant
    # objective offset that lives OUTSIDE the reduced `c` vector. The converter
    # (highs_to_standard_form_sparse) reads only col_cost_, so jaddle's objective
    # is c^T x on the reduced problem; HiGHS reports c^T x + offset_. We carry the
    # offset out so the reported objective is comparable to the full-problem
    # optimum `opt_obj`. The offset is a pure constant — it shifts the objective
    # but not the argmin, so it stays a reporting concern and never enters solve().
    offset = float(highs_lp.offset_)
    return jaddle_lp, opt_obj, highs_status, offset


def run_jaddle(jaddle_lp, tol, max_epochs):
    """Solve with Jaddle's saddle-point solver. Returns a dict of metrics.

    Two times are reported:
      * ``jaddle_solve_seconds`` -- the solver's internal iterate-loop time
        (incl. first-epoch XLA compile, excl. scaling/sparse setup). This is
        the like-for-like figure against HiGHS-PDLP's own ``run()`` timer,
        which likewise excludes problem setup. Use this in the headline table.
      * ``jaddle_wall_seconds`` -- the full ``jl.solve()`` call including
        scaling and setup, for transparency.
    """
    t0 = time.perf_counter()
    solution, converged, solve_seconds = jl.solve(
        jaddle_lp,
        verbose=True,
        log_every=10,
        primal_feasibility_tolerance=tol,
        dual_feasibility_tolerance=tol,
        dual_gap_tolerance=tol,
        update_mode="pdhg",
        k_scale=1e2,
        adaptive_eta=1,
        iterations_per_epoch=1000,
        restarts=10,
        max_epochs=max_epochs,
        return_timing=True,
    )
    wall_seconds = time.perf_counter() - t0

    obj = float(jaddle_lp.objective(solution.primal))
    eq_res = float(jaddle_lp.eq_slack(solution.primal))
    ineq_res = float(jaddle_lp.ineq_slack(solution.primal))
    return {
        "jaddle_obj": obj,
        "jaddle_converged": bool(converged),
        "jaddle_solve_seconds": solve_seconds,
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
            jaddle_lp, opt_obj, highs_status, offset = load_relaxed_lp(path)

            if jaddle_lp.A_eq.shape[0] == 0:
                import jax.experimental.sparse as jsp
                import jax.numpy as jnp

                jaddle_lp.A_eq = jsp.BCOO(
                    (jnp.zeros((1,)), jnp.array([[0, 0]])),
                    shape=(1, jaddle_lp.num_variables()),
                )
                jaddle_lp.b_eq = jnp.zeros(1)

            if jaddle_lp.A_ineq.shape[0] == 0:
                import jax.experimental.sparse as jsp
                import jax.numpy as jnp

                jaddle_lp.A_ineq = jsp.BCOO(
                    (jnp.zeros((1,)), jnp.array([[0, 0]])),
                    shape=(1, jaddle_lp.num_variables()),
                )
                jaddle_lp.b_ineq = jnp.zeros(1)

            row.update(
                {
                    "n_vars": int(jaddle_lp.num_variables()),
                    "n_cons": int(jaddle_lp.num_constraints()),
                    "opt_obj": opt_obj,
                    "highs_status": highs_status,
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
            print(
                f"  optimum (HiGHS exact): {opt_obj:.6g}  |  "
                f"Jaddle: obj={jres['jaddle_obj']:.6g} "
                f"(solve={jres['jaddle_solve_seconds']:.2f}s, "
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
    print_markdown(rows)
    print(f"\nCSV written to {args.csv}")


CSV_FIELDS = [
    "problem",
    "size_mb",
    "n_vars",
    "n_cons",
    "opt_obj",
    "offset",
    "highs_status",
    "jaddle_obj",
    "jaddle_converged",
    "jaddle_solve_seconds",
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


def print_markdown(rows):
    print("\n## Benchmark Results\n")
    print(
        "_Optimum is HiGHS's exact solver (simplex/IPM) solved to optimality, "
        "used as a ground-truth objective oracle. Jaddle time is solve-only "
        "(iterate loop incl. first-epoch XLA compile, excl. setup/scaling); see "
        "`jaddle_wall_seconds` in the CSV for full call time._\n"
    )
    print(
        "| Problem | Vars | Cons | Optimum | "
        "Jaddle obj | Jaddle solve (s) | Converged | Rel. gap to opt |"
    )
    print("|---|---:|---:|---:|---:|---:|:---:|---:|")
    for r in rows:
        if r.get("error"):
            print(f"| {r['problem']} | — | — | — | — | — | ⚠️ error | — |")
            continue
        conv = "✅" if r.get("jaddle_converged") else "❌"
        print(
            f"| {r['problem']} | {_fmt(r.get('n_vars'), '{:d}')} | "
            f"{_fmt(r.get('n_cons'), '{:d}')} | {_fmt(r.get('opt_obj'))} | "
            f"{_fmt(r.get('jaddle_obj'))} | "
            f"{_fmt(r.get('jaddle_solve_seconds'), '{:.2f}')} | {conv} | "
            f"{_fmt(r.get('rel_obj_gap'), '{:.2e}')} |"
        )


if __name__ == "__main__":
    main()

# %%
