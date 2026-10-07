"""A/B benchmark for jaddle_convex.solve on a small suite of convex problems.

Runs every (problem, config) pair and prints convergence, epochs, solve time
(excluding compile) and final objective.

Usage:
    python benchmarks/benchmark_convex.py                 # all configs
    python benchmarks/benchmark_convex.py eg_ad frb_ad    # only these configs
"""

import io
import sys
import contextlib
import numpy as np
import jax.numpy as jnp
import jaddle.jaddle_convex as jc
import jaddle.jaddle_optimisers as jo

jo.configure_jax("x64")
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

CONFIGS = {
    "alt_gd": dict(update_mode="alternating", optimiser=jo.gd(0.05)),
    "eg_ad": dict(update_mode="extragradient", adaptive_eta=0.5),
    "eg_ad_rs": dict(update_mode="extragradient", adaptive_eta=0.5, restarts=True,
                     epochs_per_restart=5),
    "frb_ad": dict(update_mode="forward_reflected", adaptive_eta=0.5),
    "frb_ad_rs": dict(update_mode="forward_reflected", adaptive_eta=0.5,
                      restarts=True, epochs_per_restart=5),
}


if __name__ == "__main__":
    only = sys.argv[1:] or None
    for pname, cp in PROBLEMS.items():
        for cname, cfg in CONFIGS.items():
            if only and cname not in only:
                continue
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                out = jc.solve(cp, max_epochs=300, iterations_per_epoch=200, **TOL, **cfg)
            ep = int(buf.getvalue().split("Epochs to solution: ")[1].split()[0])
            obj = float(cp.objective(out["solution"].primal))
            print(f"{pname:12s} {cname:10s} conv={str(out['converged']):5s} epochs={ep:4d} "
                  f"t={out['solve_seconds']:6.2f}s obj={obj:.8e}", flush=True)
