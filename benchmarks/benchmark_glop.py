# %% [markdown]
# # Jaddle LP Benchmark Harness (glop presolve)
#
# Same as benchmark_sciopt.py, but the LP relaxation is presolved with OR-Tools'
# glop presolver (the one PDLP uses, with PDLP's settings) instead of SCIP. No
# reference solve. The OR-Tools Python bindings don't expose glop's presolve, so
# it runs in a C++ helper; build it once with:
#
#     tools/glop_presolve/build.sh
#
# Usage:
#     python benchmarks/benchmark_glop.py
#     python benchmarks/benchmark_glop.py --max-mb 50          # skip huge instances
#     python benchmarks/benchmark_glop.py --only stp3d boeing  # subset by name
#     python benchmarks/benchmark_glop.py --tol 1e-4 --max-epochs 500

# %%
import os

# Suppress INFO and WARNING logs from XLA/JAX before importing JAX.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time

import jaddle.glop_helpers as gh
import jaddle.presolve as presolve

from benchmark_sciopt import make_parser, run_benchmark


def parse_args():
    p = make_parser(
        "Jaddle LP benchmark with glop presolve.", "benchmark_glop_results.csv"
    )
    p.add_argument(
        "--glop-verbose",
        action="store_true",
        help="Print glop's own presolve log.",
    )
    return p.parse_args()


def load_presolved_lp(path, glop_verbose=False, eliminate_defined_vars=False):
    """Read an MPS file, relax integrality, presolve with glop and convert to
    Jaddle's sparse standard form.

    Returns (jaddle_lp, presolve_seconds, offset), where `presolve_seconds` is
    the wall time of read + presolve + conversion and `offset` is the presolved
    model's constant objective offset (add it to jaddle's c^T x to get the
    full-problem objective).
    """
    t0 = time.perf_counter()
    lp, offset, _ = gh.glop_presolve(path, verbose=glop_verbose)
    if eliminate_defined_vars:
        lp, extra_offset, _ = presolve.eliminate_defined_variables(lp, verbose=True)
        offset += extra_offset
    presolve_seconds = time.perf_counter() - t0

    if lp.A_ineq.shape[0] == 0 and lp.A_eq.shape[0] == 0:
        raise ValueError(
            f"Presolved LP {path} has no constraints (A_ineq and A_eq are empty)."
        )
    return lp, presolve_seconds, float(offset)


def main():
    args = parse_args()
    run_benchmark(
        args,
        lambda path: load_presolved_lp(
            path,
            glop_verbose=args.glop_verbose,
            eliminate_defined_vars=args.eliminate_defined_vars,
        ),
        presolver="glop",
    )


if __name__ == "__main__":
    main()

# %%
