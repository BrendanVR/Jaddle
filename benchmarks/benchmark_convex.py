"""A/B benchmark for jaddle_convex.solve on a small suite of convex problems.

Runs every (problem, config) pair and prints convergence, epochs, solve time
(excluding compile) and final objective.

Usage:
    python benchmarks/benchmark_convex.py                 # all configs
    python benchmarks/benchmark_convex.py eg_ad pdhg_ad   # only these configs
    python benchmarks/benchmark_convex.py eg_ad --jaddle-verbose  # show solver logs
    python benchmarks/benchmark_convex.py --misscaled     # mis-scaled copies too
"""

import argparse
import io
import contextlib
import numpy as np
import jax.numpy as jnp
import jaddle.jaddle_convex as jc
import jaddle.jaddle_optimisers as jo

from benchmark import add_jaddle_verbose_flag

jo.configure_jax("float64")
rng = np.random.default_rng(0)


def isotonic(n=2000):
    y = np.linspace(-1, 1, n) + 0.15 * rng.standard_normal(n)
    return jc.JaddleCP(n, lambda p: jnp.sum((p - y) ** 2), lambda p: jnp.zeros(0),
                       lambda p: p[:-1] - p[1:], -jnp.ones(n), jnp.ones(n))


def ridge_ball():
    # RFF regression with ||w||^2 <= r (active), like examples/cp/linear_regression.py
    n, d = 200, 50
    X = np.linspace(0, 2 * np.pi, n)
    y = 3 * np.sin(X) + 2 * np.cos(5 * X) + rng.normal(0, 0.5, n)
    W = rng.normal(0, np.sqrt(20.0), d)
    b = rng.uniform(0, 2 * np.pi, d)
    F = np.sqrt(2.0 / d) * np.cos(np.outer(X, W) + b)
    return jc.JaddleCP(d, lambda w: jnp.mean((F @ w - y) ** 2), lambda w: jnp.zeros(0),
                       lambda w: jnp.array([w @ w - 10.0]),
                       -jnp.inf * jnp.ones(d), jnp.inf * jnp.ones(d))


def simplex_qp(n=500):
    # min ||x-a||^2 + q.x  s.t. sum x = 1, x>=0 (eq + bounds; badly scaled cost)
    a = rng.standard_normal(n)
    q = 10.0 * rng.standard_normal(n)
    return jc.JaddleCP(n, lambda x: jnp.sum((x - a) ** 2) + q @ x,
                       lambda x: jnp.array([jnp.sum(x) - 1.0]), lambda x: jnp.zeros(0),
                       jnp.zeros(n), jnp.inf * jnp.ones(n))


def small_lp(m=60, n=120):
    # random feasible bounded LP: min c.x s.t. Ax <= b, 0<=x<=10  (k matters here)
    A = rng.standard_normal((m, n))
    x0 = rng.uniform(0, 1, n)
    b = A @ x0 + rng.uniform(0.1, 1, m)
    c = rng.standard_normal(n) * 5
    return jc.JaddleCP(n, lambda x: c @ x, lambda x: jnp.zeros(0), lambda x: A @ x - b,
                       jnp.zeros(n), 10 * jnp.ones(n))


def logistic_l1(n=400, d=40):
    # sparse logistic regression with ||w||_1 <= t as  -s<=w<=s, sum s <= t  (many ineqs)
    Xm = rng.standard_normal((n, d))
    wt = np.zeros(d)
    wt[:5] = 3
    yl = (Xm @ wt + rng.standard_normal(n) > 0) * 2.0 - 1

    def obj(z):
        w = z[:d]
        return jnp.mean(jnp.logaddexp(0.0, -yl * (Xm @ w)))

    def ineq(z):
        w, s = z[:d], z[d:]
        return jnp.concatenate([w - s, -w - s, jnp.array([jnp.sum(s) - 5.0])])

    return jc.JaddleCP(2 * d, obj, lambda z: jnp.zeros(0), ineq,
                       -jnp.inf * jnp.ones(2 * d), jnp.inf * jnp.ones(2 * d))


PROBLEMS = {"isotonic": isotonic(), "ridge_ball": ridge_ball(), "simplex_qp": simplex_qp(),
            "small_lp": small_lp(), "logistic_l1": logistic_l1()}

TOL = dict(primal_grad_norm_tolerance=1e-6, primal_feasibility_tolerance=1e-6,
           complementarity_slack_tolerance=1e-6)

# --misscaled: each problem again with its objective x MISSCALE_OBJECTIVE and
# each constraint row x 10^U(-3, 3), the same problem in mismatched units. The
# convergence tolerances are absolute, so they scale up with the objective
# (TOL_MISSCALED), and the printed objective is divided back.
MISSCALE_OBJECTIVE = 1e3
TOL_MISSCALED = {k: v * MISSCALE_OBJECTIVE for k, v in TOL.items()}


def misscaled(cp, seed=0):
    x0 = jnp.zeros(cp.num_variables)
    r = np.random.default_rng(seed)
    e_eq = 10.0 ** r.uniform(-3, 3, cp.constraints_eq(x0).shape[0])
    e_ineq = 10.0 ** r.uniform(-3, 3, cp.constraints_ineq(x0).shape[0])
    return jc.JaddleCP(cp.num_variables,
                       lambda x: MISSCALE_OBJECTIVE * cp.objective(x),
                       lambda x: e_eq * cp.constraints_eq(x),
                       lambda x: e_ineq * cp.constraints_ineq(x),
                       cp.lower_bounds, cp.upper_bounds)

# jc.solve's defaults until 2026-10-20, which the configs below were tuned
# against; each states its departures from them.
OLD_DEFAULTS = dict(k_scale=10.0, restarts=False, diag_adapt=False,
                    gradient_scaling=False)

CONFIGS = {
    "default": dict(),
    "old_default": OLD_DEFAULTS,
    "alt_gd": dict(OLD_DEFAULTS, update_mode="alternating", optimiser=jo.gd(0.05),
                   adaptive_eta=False),
    "alt_ad": dict(OLD_DEFAULTS, update_mode="alternating", step_size=0.5),
    "alt_ad_rs": dict(OLD_DEFAULTS, update_mode="alternating", step_size=0.5,
                      restarts=True, epochs_per_restart=5),
    "alt_ad_rs_dg": dict(OLD_DEFAULTS, update_mode="alternating", step_size=0.5,
                         restarts=True, epochs_per_restart=5, diag_adapt=True),
    "eg_ad": dict(OLD_DEFAULTS, update_mode="extragradient", step_size=0.5),
    "eg_ad_rs": dict(OLD_DEFAULTS, update_mode="extragradient", step_size=0.5,
                     restarts=True, epochs_per_restart=5),
    "eg_ad_rs_dg": dict(OLD_DEFAULTS, update_mode="extragradient", step_size=0.5,
                        restarts=True, epochs_per_restart=5, diag_adapt=True),
    "eg_ad_rs_gs": dict(OLD_DEFAULTS, update_mode="extragradient", step_size=0.5,
                        restarts=True, epochs_per_restart=5, gradient_scaling=True),
    "pdhg_ad": dict(OLD_DEFAULTS, update_mode="pdhg", step_size=0.5),
    "pdhg_ad_rs": dict(OLD_DEFAULTS, update_mode="pdhg", step_size=0.5,
                       restarts=True, epochs_per_restart=5),
    "pdhg_ad_rs_dg": dict(OLD_DEFAULTS, update_mode="pdhg", step_size=0.5,
                          restarts=True, epochs_per_restart=5, diag_adapt=True),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("configs", nargs="*",
                        help=f"Configs to run (default: all of {', '.join(CONFIGS)}).")
    add_jaddle_verbose_flag(parser, default=False)
    parser.add_argument("--misscaled", action="store_true",
                        help="Also run each problem in mismatched units (objective "
                        f"x{MISSCALE_OBJECTIVE:g}, rows x 10^U(-3, 3)).")
    args = parser.parse_args()
    unknown = sorted(set(args.configs) - set(CONFIGS))
    if unknown:
        parser.error(f"unknown config(s): {', '.join(unknown)}")
    runs = [(name, cp, TOL, 1.0) for name, cp in PROBLEMS.items()]
    if args.misscaled:
        runs += [(name + "_ms", misscaled(cp), TOL_MISSCALED, MISSCALE_OBJECTIVE)
                 for name, cp in PROBLEMS.items()]
    for pname, cp, tol, obj_scale in runs:
        for cname, cfg in CONFIGS.items():
            if args.configs and cname not in args.configs:
                continue
            # Show Jaddle's own output only when verbose; otherwise keep the
            # table clean.
            quiet = contextlib.nullcontext() if args.jaddle_verbose else (
                contextlib.redirect_stdout(io.StringIO()))
            with quiet:
                out = jc.solve(cp, max_epochs=300, iterations_per_epoch=200,
                               verbose=args.jaddle_verbose, **tol, **cfg)
            ep = out["epochs"]
            obj = float(cp.objective(out["solution"].primal)) / obj_scale
            print(f"{pname:15s} {cname:11s} conv={str(out['converged']):5s} epochs={ep:4d} "
                  f"t={out['solve_seconds']:6.2f}s obj={obj:.8e}", flush=True)
