# %% [markdown]
# # Jaddle vs MPAX LP Benchmark
#
# Runs Jaddle and MPAX (https://github.com/MIT-Lu-Lab/MPAX, a JAX PDHG solver)
# on the same presolved LP relaxations, under the same tolerance and time
# budget, and checks both answers with one shared certificate.
#
# Fairness choices:
# * **Same input.** Each instance is presolved once (HiGHS by default, or not at
#   all with `--presolve none`) and the identical LP is handed to both solvers.
# * **Same stopping rule.** Both stop on relative primal residual, dual residual
#   and duality gap below `--tol`. MPAX is run with `eps_abs = eps_rel = tol`,
#   which gives the same `residual / (1 + norm)` form Jaddle uses, and in the
#   L2 norm by default to match Jaddle (`--mpax-norm inf` restores MPAX's own
#   default). The one remaining difference: MPAX normalises the gap by
#   `1 + max(|p|, |d|)`, Jaddle by `1 + |p| + |d|`.
# * **Shared verification.** Every returned primal-dual pair is re-checked with
#   `jaddle_linear.evaluate_lp_certificate` in the original units, so neither
#   solver is judged only by its own stopping test. "Verified" means all three
#   relative residuals are within `--tol` (plus 1% to absorb rounding).
# * **Same timing.** Each solve runs in its own worker process with the whole
#   GPU. The clock starts once the worker has loaded the LP and covers scaling,
#   XLA compilation and iteration for both solvers. MPAX has no time limit of
#   its own, so the worker is killed when `--max-seconds` runs out; Jaddle uses
#   its built-in `max_seconds`, with the same kill as a backstop.
#
# Requires `pip install mpax`. MPS files go in `data/`, as for the other
# harnesses.
#
# Usage:
#     python benchmarks/benchmark_mpax.py --tol 1e-4 --max-seconds 600
#     python benchmarks/benchmark_mpax.py --only stp3d boeing
#     python benchmarks/benchmark_mpax.py --presolve none --no-eliminate-defined-vars
#     python benchmarks/benchmark_mpax.py --solvers mpax --mpax-algorithm rapdhg

# %%
import os
import sys

# The parent process only presolves and reports, so keep it off the GPU and
# leave the whole device to the worker processes. Remember the user's own
# setting so the workers get it back.
_IS_WORKER = len(sys.argv) == 3 and sys.argv[1] == "--worker"
_USER_JAX_PLATFORMS = os.environ.get("JAX_PLATFORMS")
if not _IS_WORKER:
    os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import csv
import importlib.util
import json
import math
import subprocess
import tempfile
import threading
import time

import highspy as hspy
import numpy as np
import scipy.sparse as sp

import jaddle.highs_helpers as hh
import jaddle.presolve as presolve
from jaddle.jaddle_basic_types import LP

from benchmark import _fmt, discover_instances, load_relaxed_lp
from benchmark_sciopt import make_parser

SOLVERS = ("jaddle", "mpax")
# A Jaddle solve may overrun max_seconds by up to one epoch before stopping
# itself; only kill it once this much extra time has passed.
JADDLE_KILL_GRACE_SECONDS = 120.0
# Verification slack on top of --tol, to absorb rounding differences between
# a solver's internal residuals and the shared re-check.
VERIFY_SLACK = 1.01
# Shift for the shifted geometric mean of solve times (PDLP convention).
SGM_SHIFT_SECONDS = 10.0


def parse_args():
    p = make_parser("Jaddle vs MPAX LP benchmark.", "benchmark_mpax_results.csv")
    p.add_argument(
        "--solvers",
        nargs="+",
        choices=SOLVERS,
        default=list(SOLVERS),
        help="Solvers to run (default: both).",
    )
    p.add_argument(
        "--presolve",
        choices=("highs", "none"),
        default="highs",
        help="Presolve applied before both solvers (default highs). 'none' "
        "solves the raw relaxation, as in the PDLP and MPAX papers; combine with "
        "--no-eliminate-defined-vars for a fully unpresolved run.",
    )
    p.add_argument(
        "--mpax-algorithm",
        choices=("r2hpdhg", "rapdhg"),
        default="r2hpdhg",
        help="MPAX solver: r2HPDHG (reflected restarted Halpern PDHG, MPAX's "
        "LP solver; default) or raPDHG (restarted average PDHG).",
    )
    p.add_argument(
        "--mpax-norm",
        choices=("2", "inf"),
        default="2",
        help="Norm for MPAX's termination residuals (default 2, matching "
        "Jaddle's default; MPAX's own default is inf).",
    )
    p.add_argument(
        "--mpax-iteration-limit",
        type=int,
        default=None,
        help="Optional MPAX iteration cap (default: none).",
    )
    p.add_argument(
        "--startup-timeout",
        type=float,
        default=600.0,
        help="Seconds a worker may take to start and load the LP before it is "
        "treated as failed (default 600).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Stream each solver's own progress log.",
    )
    return p.parse_args()


# %% [markdown]
# ## Loading and passing the LP to the workers


# %%
def load_lp(path, presolve_mode, eliminate_defined_vars):
    """Return (lp, offset, presolve_seconds) with lp a scipy-backed LP."""
    t0 = time.perf_counter()
    if presolve_mode == "highs":
        lp, _, _, _, offset = load_relaxed_lp(
            path, highs_solver="none", eliminate_defined_vars=eliminate_defined_vars
        )
    else:
        highs = hspy.Highs()
        highs.setOptionValue("output_flag", False)
        highs.readModel(path)
        highs_lp = highs.getLp()
        lp = hh.highs_to_standard_form_sparse(highs_lp)
        offset = float(highs_lp.offset_)
        if eliminate_defined_vars:
            lp, extra_offset, _ = presolve.eliminate_defined_variables(lp, verbose=True)
            offset += extra_offset
    return lp, float(offset), time.perf_counter() - t0


def save_lp(lp, directory):
    np.savez(
        os.path.join(directory, "vectors.npz"),
        c=np.asarray(lp.c, dtype=np.float64),
        b_eq=np.asarray(lp.b_eq, dtype=np.float64),
        b_ineq=np.asarray(lp.b_ineq, dtype=np.float64),
        lower_bounds=np.asarray(lp.lower_bounds, dtype=np.float64),
        upper_bounds=np.asarray(lp.upper_bounds, dtype=np.float64),
    )
    sp.save_npz(os.path.join(directory, "A_eq.npz"), sp.csr_matrix(lp.A_eq))
    sp.save_npz(os.path.join(directory, "A_ineq.npz"), sp.csr_matrix(lp.A_ineq))


def read_lp(directory):
    v = np.load(os.path.join(directory, "vectors.npz"))
    return LP(
        c=v["c"],
        A_eq=sp.load_npz(os.path.join(directory, "A_eq.npz")).tocsc(),
        b_eq=v["b_eq"],
        A_ineq=sp.load_npz(os.path.join(directory, "A_ineq.npz")).tocsc(),
        b_ineq=v["b_ineq"],
        lower_bounds=v["lower_bounds"],
        upper_bounds=v["upper_bounds"],
    )


# %% [markdown]
# ## Worker: one solve in its own process


# %%
def worker_main(config_path):
    with open(config_path) as f:
        cfg = json.load(f)
    out = {"solver": cfg["solver"]}
    try:
        import jaddle.jaddle_optimisers as jo

        jo.configure_jax(cfg["jax_profile"])
        import jax
        import jaddle.jaddle_linear as jl

        lp = read_lp(cfg["lp_dir"])
        print("SOLVE_START", flush=True)
        t0 = time.perf_counter()
        if cfg["solver"] == "jaddle":
            from benchmark import jaddle_solve_kwargs

            result = jl.solve(
                lp,
                **jaddle_solve_kwargs(
                    cfg["tol"],
                    cfg["max_epochs"],
                    cfg["update_mode"],
                    cfg["max_seconds"],
                    verbose=cfg["verbose"],
                ),
            )
            sol = jax.block_until_ready(result["solution"])
            x, y_eq, y_ineq = sol.primal, sol.dual_eq, sol.dual_ineq
            out["status"] = result["stop_reason"]
            out["claims_optimal"] = result["stop_reason"] == "certificate"
            out["epochs"] = int(result["epochs"])
        else:
            from mpax import create_lp, r2HPDHG, raPDHG
            from mpax.utils import TerminationStatus

            # MPAX's form is Ax = b, Gx >= h, l <= x <= u; Jaddle's is
            # A_eq x = b_eq, A_ineq x <= b_ineq, so G = -A_ineq, h = -b_ineq.
            problem = create_lp(
                lp.c,
                lp.A_eq,
                lp.b_eq,
                -lp.A_ineq,
                -lp.b_ineq,
                lp.lower_bounds,
                lp.upper_bounds,
            )
            options = dict(
                eps_abs=cfg["tol"],
                eps_rel=cfg["tol"],
                optimality_norm=float(cfg["mpax_norm"]),
                verbose=cfg["verbose"],
            )
            if cfg["mpax_iteration_limit"]:
                options["iteration_limit"] = cfg["mpax_iteration_limit"]
            solver = r2HPDHG if cfg["mpax_algorithm"] == "r2hpdhg" else raPDHG
            result = solver(**options).optimize(problem)
            x = jax.block_until_ready(result.primal_solution)
            y = np.asarray(result.dual_solution)
            n_eq = lp.A_eq.shape[0]
            # MPAX's equality duals have the opposite sign to Jaddle's; its
            # inequality duals match once G = -A_ineq is accounted for.
            y_eq, y_ineq = -y[:n_eq], y[n_eq:]
            status = TerminationStatus(int(result.termination_status)).name.lower()
            out["status"] = status
            out["claims_optimal"] = status == "optimal"
            out["iterations"] = int(result.iteration_count)
        out["wall_seconds"] = time.perf_counter() - t0

        cert = jl.evaluate_lp_certificate(
            jl.to_jaddle_sparse(lp),
            jax.numpy.asarray(x),
            jax.numpy.asarray(y_eq),
            jax.numpy.asarray(y_ineq),
        )
        out["objective"] = float(cert["objective"])
        out["rel_primal"] = float(cert["relative_primal_feasibility_residual"])
        out["rel_dual"] = float(cert["relative_dual_feasibility_residual"])
        out["rel_gap"] = float(cert["relative_gap"])
        out["error"] = ""
    except Exception as exc:
        out["error"] = repr(exc)
    with open(cfg["out_path"], "w") as f:
        json.dump(out, f)


def run_worker(solver, lp_dir, args):
    """Run one solve in a worker process, enforcing the time budget. Returns
    the worker's result dict."""
    out_path = os.path.join(lp_dir, f"{solver}_result.json")
    config_path = os.path.join(lp_dir, f"{solver}_config.json")
    with open(config_path, "w") as f:
        json.dump(
            {
                "solver": solver,
                "lp_dir": lp_dir,
                "out_path": out_path,
                "tol": args.tol,
                "max_epochs": args.max_epochs,
                "max_seconds": args.max_seconds,
                "update_mode": args.update_mode,
                "jax_profile": args.jax_profile,
                "mpax_algorithm": args.mpax_algorithm,
                "mpax_norm": args.mpax_norm,
                "mpax_iteration_limit": args.mpax_iteration_limit,
                "verbose": args.verbose,
            },
            f,
        )

    env = dict(os.environ)
    if _USER_JAX_PLATFORMS is None:
        env.pop("JAX_PLATFORMS", None)
    else:
        env["JAX_PLATFORMS"] = _USER_JAX_PLATFORMS

    proc = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--worker", config_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    started = threading.Event()
    tail = []

    def pump():
        for line in proc.stdout:
            if line.startswith("SOLVE_START"):
                started.set()
                continue
            tail.append(line)
            del tail[:-40]
            if args.verbose:
                print("    " + line, end="", flush=True)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()

    killed = False
    if not started.wait(args.startup_timeout) and proc.poll() is None:
        proc.kill()
        killed = True
    elif args.max_seconds is not None:
        limit = args.max_seconds
        if solver == "jaddle":
            limit += JADDLE_KILL_GRACE_SECONDS
        try:
            proc.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            proc.kill()
            killed = True
    proc.wait()
    reader.join(timeout=10)

    if killed:
        return {
            "solver": solver,
            "status": "time_limit" if started.is_set() else "startup_timeout",
            "claims_optimal": False,
            "wall_seconds": args.max_seconds if started.is_set() else math.nan,
            "error": "",
        }
    try:
        with open(out_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {
            "solver": solver,
            "status": "crashed",
            "claims_optimal": False,
            "error": f"exit code {proc.returncode}: " + "".join(tail[-5:]).strip(),
        }


# %% [markdown]
# ## Reporting


# %%
CSV_FIELDS = [
    "problem",
    "size_mb",
    "n_vars",
    "n_cons",
    "presolve",
    "presolve_seconds",
    "offset",
    "solver",
    "status",
    "claims_optimal",
    "verified",
    "wall_seconds",
    "epochs",
    "iterations",
    "objective",
    "rel_primal",
    "rel_dual",
    "rel_gap",
    "error",
]


def is_verified(res, tol):
    keys = ("rel_primal", "rel_dual", "rel_gap")
    if res.get("error") or any(k not in res for k in keys):
        return False
    return all(res[k] <= VERIFY_SLACK * tol for k in keys)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def shifted_geomean(times, shift=SGM_SHIFT_SECONDS):
    if not times:
        return math.nan
    return math.exp(sum(math.log(t + shift) for t in times) / len(times)) - shift


def print_report(rows, args):
    solvers = [s for s in SOLVERS if s in args.solvers]
    names = {"jaddle": "Jaddle", "mpax": f"MPAX ({args.mpax_algorithm})"}
    by_problem = {}
    for r in rows:
        by_problem.setdefault(r["problem"], {})[r.get("solver")] = r

    print(f"\n## Jaddle vs MPAX (tol {args.tol:g}, presolve: {args.presolve})\n")
    header = "| Problem | Vars | Cons |"
    rule = "|---|---:|---:|"
    for s in solvers:
        header += f" {names[s]} | {names[s]} time (s) |"
        rule += ":---:|---:|"
    if len(solvers) == 2:
        header += " Obj. rel. diff |"
        rule += "---:|"
    print(header)
    print(rule)
    for problem, res in by_problem.items():
        first = next(iter(res.values()))
        line = (
            f"| {problem} | {_fmt(first.get('n_vars'), '{:d}')} | "
            f"{_fmt(first.get('n_cons'), '{:d}')} |"
        )
        for s in solvers:
            r = res.get(s, {})
            if r.get("error"):
                cell = "⚠️ error"
            elif r.get("verified"):
                cell = "✅"
            else:
                cell = f"❌ {r.get('status', '—')}"
            line += f" {cell} | {_fmt(r.get('wall_seconds'), '{:.2f}')} |"
        if len(solvers) == 2:
            a, b = (res.get(s, {}).get("objective") for s in solvers)
            diff = (
                abs(a - b) / (1 + abs(a)) if a is not None and b is not None else None
            )
            line += f" {_fmt(diff, '{:.1e}')} |"
        print(line)

    print()
    for s in solvers:
        results = [res[s] for res in by_problem.values() if s in res]
        ran = [r for r in results if not r.get("error")]
        verified = sum(bool(r.get("verified")) for r in ran)
        cap = args.max_seconds
        times = [
            r["wall_seconds"] if r.get("verified") or cap is None else cap
            for r in ran
            if r.get("wall_seconds") is not None and not math.isnan(r["wall_seconds"])
        ]
        print(
            f"- **{names[s]}**: verified {verified}/{len(results)}; shifted "
            f"geometric mean time {shifted_geomean(times):.2f} s (shift "
            f"{SGM_SHIFT_SECONDS:g} s"
            + (", unverified solves counted at the time limit)" if cap else ")")
        )


# %%
def main():
    args = parse_args()
    if "mpax" in args.solvers and importlib.util.find_spec("mpax") is None:
        sys.exit("MPAX is not installed: pip install mpax")

    instances = discover_instances(args)
    if not instances:
        print(
            f"No .mps files to run in {args.data_dir}. Download MIPLIB instances "
            "(https://miplib.zib.de/) into data/ and retry."
        )
        return

    print(
        f"Running {len(instances)} instance(s) with {', '.join(args.solvers)} at "
        f"tol={args.tol:g}, max_seconds={args.max_seconds}, presolve={args.presolve}\n"
    )
    rows = []
    for name, path, size_mb in instances:
        print(f"=== {name} ({size_mb:.1f} MB) ===")
        base = {"problem": name, "size_mb": round(size_mb, 1), "presolve": args.presolve}
        try:
            lp, offset, presolve_seconds = load_lp(
                path, args.presolve, args.eliminate_defined_vars
            )
        except Exception as exc:
            print(f"  ERROR loading: {exc!r}")
            rows.append({**base, "error": repr(exc)})
            write_csv(args.csv, rows)
            continue
        base.update(
            n_vars=int(lp.num_variables()),
            n_cons=int(lp.num_constraints()),
            presolve_seconds=presolve_seconds,
            offset=offset,
        )
        with tempfile.TemporaryDirectory(prefix="jaddle_mpax_") as lp_dir:
            save_lp(lp, lp_dir)
            del lp
            for solver in args.solvers:
                res = run_worker(solver, lp_dir, args)
                if "objective" in res:
                    res["objective"] += offset
                res["verified"] = is_verified(res, args.tol)
                rows.append({**base, **res})
                if res.get("error"):
                    print(f"  {solver}: ERROR {res['error']}")
                else:
                    print(
                        f"  {solver}: {res['status']} in "
                        f"{_fmt(res.get('wall_seconds'), '{:.2f}')}s, "
                        f"obj={_fmt(res.get('objective'), '{:.6g}')}, "
                        f"verified={res['verified']} (primal "
                        f"{_fmt(res.get('rel_primal'), '{:.1e}')}, dual "
                        f"{_fmt(res.get('rel_dual'), '{:.1e}')}, gap "
                        f"{_fmt(res.get('rel_gap'), '{:.1e}')})"
                    )
                write_csv(args.csv, rows)
        print()

    write_csv(args.csv, rows)
    print_report(rows, args)
    print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    if _IS_WORKER:
        worker_main(sys.argv[2])
    else:
        main()

# %%
