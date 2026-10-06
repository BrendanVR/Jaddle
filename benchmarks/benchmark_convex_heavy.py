"""Gradient-heavy benchmark for jaddle_convex.solve update modes.

Dense l1-constrained logistic regression (20000 x 1000): each gradient is two
dense matvecs, so per-iteration cost is dominated by gradient evaluations (unlike
the small suite in benchmark_convex.py, which is overhead-bound). Use it to compare
per-epoch cost against epochs-to-converge across update modes.

Usage:
    python benchmarks/benchmark_convex_heavy.py                      # both modes
    python benchmarks/benchmark_convex_heavy.py forward_reflected    # one mode
"""

import io
import sys
import contextlib
import numpy as np
import jax
import jax.numpy as jnp
import jaddle.jaddle_convex as jc
import jaddle.jaddle_optimisers as jo

jo.configure_jax("x64")
print(jax.devices())
rng = np.random.default_rng(0)

n, d = 20000, 1000
Xm = jnp.asarray(rng.standard_normal((n, d)))
wt = np.zeros(d)
wt[:20] = 1
yl = jnp.asarray((np.asarray(Xm) @ wt + rng.standard_normal(n) > 0) * 2.0 - 1)


def obj(z):
    return jnp.mean(jnp.logaddexp(0.0, -yl * (Xm @ z[:d])))


def ineq(z):
    # ||w||_1 <= 3 as -s <= w <= s, sum s <= 3
    return jnp.concatenate(
        [z[:d] - z[d:], -z[:d] - z[d:], jnp.array([jnp.sum(z[d:]) - 3.0])]
    )


cp = jc.JaddleCP(2 * d, obj, lambda z: jnp.zeros(0), ineq,
                 -jnp.inf * jnp.ones(2 * d), jnp.inf * jnp.ones(2 * d))

if __name__ == "__main__":
    modes = sys.argv[1:] or ["extragradient", "forward_reflected"]
    for mode in modes:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = jc.solve(cp, max_epochs=300, iterations_per_epoch=200, update_mode=mode,
                           adaptive_eta=0.5, primal_grad_norm_tolerance=1e-6,
                           primal_feasibility_tolerance=1e-6,
                           complementarity_slack_tolerance=1e-6)
        ep = int(buf.getvalue().split("Epochs to solution: ")[1].split()[0])
        print(f"{mode:18s} conv={out['converged']} epochs={ep} t={out['solve_seconds']:.2f}s "
              f"per-epoch={out['solve_seconds'] / ep:.3f}s "
              f"obj={float(obj(out['solution'].primal)):.8e}", flush=True)
