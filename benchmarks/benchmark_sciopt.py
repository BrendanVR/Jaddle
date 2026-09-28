# %% [markdown]
# # Jaddle LP Benchmark Harness (SCIP presolve)
#
# Same as benchmark.py, but the LP relaxation is read and presolved with
# PySCIPOpt instead of HiGHS, and there is no reference solve: each instance is
# presolved by SCIP and handed straight to Jaddle. The Jaddle solve config is
# shared with benchmark.py (run_jaddle is imported from there).
#
# Usage:
#     python benchmarks/bechmark_sciopt.py
#     python benchmarks/bechmark_sciopt.py --max-mb 50          # skip huge instances
#     python benchmarks/bechmark_sciopt.py --only stp3d boeing  # subset by name
#     python benchmarks/bechmark_sciopt.py --tol 1e-4 --max-epochs 500

# %%
import os

# Suppress INFO and WARNING logs from XLA/JAX before importing JAX.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import csv
import time

import jaddle.jaddle_optimisers as jo
import jaddle.sciopt_helpers as sh

from benchmark import DATA_DIR, REPO_ROOT, _fmt, discover_instances, run_jaddle


def parse_args():
    p = argparse.ArgumentParser(description="Jaddle LP benchmark with SCIP presolve.")
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
        help="Relative optimality tolerance for Jaddle (default 1e-3).",
    )
    p.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Cap Jaddle epochs (None = run to convergence).",
    )
    p.add_argument(
        "--csv",
        default=os.path.join(REPO_ROOT, "benchmark_sciopt_results.csv"),
        help="Path to write CSV results.",
    )
    p.add_argument(
        "--skip-bigm",
        action="store_true",
        help="Skip instances with a big-M / penalty COST structure. See "
        "benchmarks/scan_bigm.py.",
    )
    p.add_argument(
        "--skip-bigm-matrix",
        action="store_true",
        help="Skip instances with a big-M / penalty MATRIX structure. See "
        "benchmarks/scan_bigm.py.",
    )
    p.add_argument(
        "--skip-bigm-column",
        action="store_true",
        help="Skip instances with a big-M / penalty COLUMN structure. See "
        "benchmarks/scan_bigm.py.",
    )
    p.add_argument(
        "--scip-verbose",
        action="store_true",
        help="Enable SCIP's own presolve logging.",
    )
    p.add_argument(
        "--jax-profile",
        default="float64",
        choices=["float64", "float32", "float16"],
        help="JAX precision profile passed to jaddle_optimisers.configure_jax "
        "(default: float64).",
    )
    return p.parse_args()


def load_presolved_lp(path, scip_verbose=False):
    """Read an MPS file with PySCIPOpt, relax integrality, presolve with SCIP and
    convert to Jaddle's sparse standard form.

    Returns (jaddle_lp, presolve_seconds, offset), where `presolve_seconds` is
    the wall time of read + presolve + conversion and `offset` is the presolved
    model's constant objective offset (add it to jaddle's c^T x to get the
    full-problem objective).
    """
    t0 = time.perf_counter()
    model = sh.read_relaxed_model(path, presolve=True, quiet=not scip_verbose)
    lp, offset = sh.scip_to_standard_form_sparse(model)
    presolve_seconds = time.perf_counter() - t0

    if lp.A_ineq.shape[0] == 0 and lp.A_eq.shape[0] == 0:
        raise ValueError(
            f"Presolved LP {path} has no constraints (A_ineq and A_eq are empty)."
        )
    return lp, presolve_seconds, float(offset)


def main():
    args = parse_args()
    jo.configure_jax(args.jax_profile)

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
            jaddle_lp, presolve_seconds, offset = load_presolved_lp(
                path, scip_verbose=args.scip_verbose
            )
            row.update(
                {
                    "n_vars": int(jaddle_lp.num_variables()),
                    "n_cons": int(jaddle_lp.num_constraints()),
                    "presolve_seconds": presolve_seconds,
                }
            )
            jres = run_jaddle(jaddle_lp, args.tol, args.max_epochs)
            # jaddle solves the presolved reduced problem (objective = c^T x);
            # add the presolve offset to report the full-problem objective.
            jres["jaddle_obj"] += offset
            row.update(jres)
            row["offset"] = offset
            row["error"] = ""
            print(
                f"  SCIP presolve={presolve_seconds:.2f}s  |  "
                f"Jaddle: obj={jres['jaddle_obj']:.6g} "
                f"(solve={jres['jaddle_solve_seconds']:.2f}s, "
                f"corrected={jres['jaddle_corrected_seconds']:.2f}s, "
                f"wall={jres['jaddle_wall_seconds']:.2f}s, "
                f"converged={jres['jaddle_converged']})"
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
    "offset",
    "presolve_seconds",
    "jaddle_obj",
    "jaddle_converged",
    "jaddle_stop_reason",
    "jaddle_solve_seconds",
    "jaddle_corrected_seconds",
    "jaddle_wall_seconds",
    "jaddle_eq_res",
    "jaddle_ineq_res",
    "error",
]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def print_markdown(rows):
    print("\n## Benchmark Results (SCIP presolve)\n")
    print(
        "_LPs presolved with SCIP; no reference optimum is computed. Jaddle solve "
        "time is solve-only (iterate loop incl. first-epoch XLA compile, excl. "
        "setup/scaling); corrected time amortises the one-off first-epoch compile "
        "out (`n·(solve−first)/(n−1)`); see `jaddle_wall_seconds` in the CSV for "
        "full call time._\n"
    )
    print(
        "| Problem | Vars | Cons | SCIP presolve (s) | Jaddle obj | "
        "Jaddle solve (s) | Jaddle corrected (s) | Converged | Stop |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|:---:|:---:|")
    for r in rows:
        if r.get("error"):
            print(f"| {r['problem']} | — | — | — | — | — | — | ⚠️ error | — |")
            continue
        conv = "✅" if r.get("jaddle_converged") else "❌"
        print(
            f"| {r['problem']} | {_fmt(r.get('n_vars'), '{:d}')} | "
            f"{_fmt(r.get('n_cons'), '{:d}')} | "
            f"{_fmt(r.get('presolve_seconds'), '{:.2f}')} | "
            f"{_fmt(r.get('jaddle_obj'))} | "
            f"{_fmt(r.get('jaddle_solve_seconds'), '{:.2f}')} | "
            f"{_fmt(r.get('jaddle_corrected_seconds'), '{:.2f}')} | {conv} | "
            f"{r.get('jaddle_stop_reason') or '—'} |"
        )


if __name__ == "__main__":
    main()

# %%
