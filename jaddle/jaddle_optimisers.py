import os

import optax
import jax
import jax.numpy as jnp
from typing import Any, Optional
from jaddle.jaddle_basic_types import (
    ExtragradientState,
    ScheduleLike,
    LP,
    SaddleState,
)
import os


def configure_jax(jax_profile: Optional[str] = None):
    """
    Configure JAX environment variables for different profiling modes.

    Args:
        jax_profile: Optional string to specify the profiling mode. Can be "balanced", "max_speed", or None.
                     If None, it will read from the JADDLE_JAX_PROFILE environment variable, defaulting to "max_speed".

    This function sets environment variables to optimize JAX's performance based on the chosen profile.
    It also prints out the active configuration for verification.
    """
    if jax_profile is not None:
        os.environ["JADDLE_JAX_PROFILE"] = jax_profile

    JAX_PROFILE = os.environ.get("JADDLE_JAX_PROFILE", "safe").lower()

    def _append_xla_flag(flag: str):
        current = os.environ.get("XLA_FLAGS", "")
        if flag not in current:
            os.environ["XLA_FLAGS"] = f"{current} {flag}".strip()

    # Suppress INFO and WARNING logs from XLA/JAX
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    if JAX_PROFILE in ["balanced", "max_speed"]:
        os.environ.setdefault("JAX_ENABLE_X64", "0")
        os.environ.setdefault(
            "JAX_COMPILATION_CACHE_DIR",
            os.path.expanduser("~/.cache/jaddle_jax"),
        )

    if JAX_PROFILE == "max_speed":
        os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "tensorfloat32")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
        _append_xla_flag("--xla_gpu_autotune_level=4")
    elif JAX_PROFILE in ["safe", "balanced"]:
        os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "high")

    # The "x64" profile enables double precision for the convergence-critical
    # path — the analogue of PDLP's float64 default. On wide-dynamic-range
    # MIPLIB rows, float32 residuals plateau on rounding and "stalled" becomes
    # indistinguishable from "out of precision". Must run before JAX initialises.
    # Use full-precision matmuls (no TF32) so the extra mantissa actually counts.
    if JAX_PROFILE == "x64":
        os.environ.setdefault("JAX_ENABLE_X64", "1")
        os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
        os.environ.setdefault(
            "JAX_COMPILATION_CACHE_DIR",
            os.path.expanduser("~/.cache/jaddle_jax"),
        )
        # The env var only takes effect if read before JAX initialises; since
        # this module already imported jax, set the live config too so x64 works
        # even when configure_jax("x64") is called after import.
        jax.config.update("jax_enable_x64", True)

    print(
        "[JAX Profile] "
        f"mode={JAX_PROFILE}, "
        f"x64={os.environ.get('JAX_ENABLE_X64', 'default')}, "
        f"matmul_precision={os.environ.get('JAX_DEFAULT_MATMUL_PRECISION', 'default')}, "
        f"cache={os.environ.get('JAX_COMPILATION_CACHE_DIR', 'disabled')}, "
        f"xla_flags='{os.environ.get('XLA_FLAGS', '')}'"
    )


def _saddle_param_labels(params):
    if hasattr(params, "_fields"):
        return type(params)(
            primal="primal_opt",
            dual_ineq="dual_opt",
            dual_eq="dual_opt",
        )
    return {
        "primal": "primal_opt",
        "dual_ineq": "dual_opt",
        "dual_eq": "dual_opt",
    }


def estimate_operator_norm(lp: LP, num_iters: int = 20, seed: int = 0):
    """
    Estimate the largest singular value of the stacked constraint matrix
    [A_eq; A_ineq] via power iteration on A^T A.

    This gives an estimate of the operator norm ||A||_2, which is the
    theoretically justified step-size bound for primal-dual saddle-point
    methods: convergence requires lr < 1 / ||A||_2.

    Args:
        lp: Linear program with BCOO sparse constraint matrices.
        num_iters: Number of power iterations (20 is usually sufficient).
        seed: Random seed for initial vector.

    Returns:
        Estimated largest singular value (float).
    """
    n = lp.num_variables()
    key = jax.random.PRNGKey(seed)
    v = jax.random.normal(key, (n,), dtype=lp.c.dtype)
    v = v / jnp.linalg.norm(v)

    def power_step(v, _):
        # Compute A @ v by stacking A_eq and A_ineq
        Av_eq = lp.A_eq @ v
        Av_ineq = lp.A_ineq @ v
        # Compute A^T @ (A @ v)
        ATAv = lp.A_eq_T @ Av_eq + lp.A_ineq_T @ Av_ineq
        norm = jnp.linalg.norm(ATAv)
        # Avoid division by zero
        v_new = ATAv / jnp.maximum(norm, 1e-30)
        return v_new, norm

    v, norms = jax.lax.scan(power_step, v, None, length=num_iters)

    # Final Rayleigh quotient: sigma_max = sqrt(v^T A^T A v)
    Av_eq = lp.A_eq @ v
    Av_ineq = lp.A_ineq @ v
    ATAv = lp.A_eq_T @ Av_eq + lp.A_ineq_T @ Av_ineq
    sigma_max = jnp.sqrt(jnp.dot(v, ATAv))

    return sigma_max


def estimate_condition_number(
    lp: LP, num_iters: int = 20, seed: int = 0, regularisation: float = 1e-6
):
    """
    Estimate the condition number kappa = sigma_max / sigma_min of the stacked
    constraint matrix [A_eq; A_ineq].

    sigma_max is estimated via power iteration on A^T A.
    sigma_min is estimated via inverse power iteration, i.e. power iteration
    on (A^T A + mu I)^{-1}, solved approximately with conjugate-gradient steps.
    A small Tikhonov regularisation mu is added for numerical stability.

    Args:
        lp: Linear program with BCOO sparse constraint matrices.
        num_iters: Number of iterations for both power and inverse power (default 20).
        seed: Random seed for initial vectors.
        regularisation: Tikhonov regularisation for inverse iteration (default 1e-6).

    Returns:
        (sigma_max, sigma_min, kappa) as floats.
    """
    sigma_max = estimate_operator_norm(lp, num_iters=num_iters, seed=seed)

    # Inverse power iteration for sigma_min:
    # We want the smallest eigenvalue of A^T A. We iterate
    # v <- (A^T A + mu I)^{-1} v / ||(A^T A + mu I)^{-1} v||
    # solving the linear system with CG.
    n = lp.num_variables()
    key = jax.random.PRNGKey(seed + 1)
    v = jax.random.normal(key, (n,), dtype=lp.c.dtype)
    v = v / jnp.linalg.norm(v)

    def matvec_regularised(x):
        """Compute (A^T A + mu I) x."""
        Ax_eq = lp.A_eq @ x
        Ax_ineq = lp.A_ineq @ x
        ATAx = lp.A_eq_T @ Ax_eq + lp.A_ineq_T @ Ax_ineq
        return ATAx + regularisation * x

    def cg_solve(b, x0, num_cg_iters=50):
        """Solve (A^T A + mu I) x = b via conjugate gradient."""

        def cg_step(carry, _):
            x, r, p = carry
            Ap = matvec_regularised(p)
            rTr = jnp.dot(r, r)
            alpha = rTr / jnp.maximum(jnp.dot(p, Ap), 1e-30)
            x = x + alpha * p
            r_new = r - alpha * Ap
            beta = jnp.dot(r_new, r_new) / jnp.maximum(rTr, 1e-30)
            p = r_new + beta * p
            return (x, r_new, p), None

        r0 = b - matvec_regularised(x0)
        p0 = r0
        (x, _, _), _ = jax.lax.scan(cg_step, (x0, r0, p0), None, length=num_cg_iters)
        return x

    def inverse_power_step(v, _):
        # Solve (A^T A + mu I) w = v
        w = cg_solve(v, jnp.zeros_like(v))
        norm = jnp.linalg.norm(w)
        v_new = w / jnp.maximum(norm, 1e-30)
        return v_new, norm

    v, _ = jax.lax.scan(inverse_power_step, v, None, length=num_iters)

    # Rayleigh quotient for sigma_min: sqrt(v^T A^T A v)
    Av_eq = lp.A_eq @ v
    Av_ineq = lp.A_ineq @ v
    ATAv = lp.A_eq_T @ Av_eq + lp.A_ineq_T @ Av_ineq
    sigma_min = jnp.sqrt(jnp.maximum(jnp.dot(v, ATAv), 0.0))

    # Clamp sigma_min to avoid degenerate condition numbers
    sigma_min = jnp.maximum(sigma_min, 1e-12)
    kappa = sigma_max / sigma_min

    return float(sigma_max), float(sigma_min), float(kappa)


def create_saddle_optimiser(
    primal_optimizer: optax.GradientTransformation,
    dual_optimizer: optax.GradientTransformation | None = None,
):
    if dual_optimizer is None:
        dual_optimizer = primal_optimizer
    optimiser = optax.partition(
        {
            "primal_opt": primal_optimizer,
            "dual_opt": dual_optimizer,
        },
        param_labels=_saddle_param_labels,
    )

    return optimiser


def _resolve_schedule(schedule: ScheduleLike, step: jnp.ndarray) -> jnp.ndarray:
    if callable(schedule):
        return jnp.asarray(schedule(step))
    return jnp.asarray(schedule)


def extragradient(
    learning_rate: ScheduleLike,
) -> optax.GradientTransformation:
    """Extragradient (Korpelevich 1976) as an optax GradientTransformation.

    Each step performs two gradient evaluations at (params, grad):
      1. Look-ahead:   x_half = params - lr * grad
      2. Corrector:    update  = -lr * grad_half   (caller must supply grad_half)

    Because optax's update_fn receives a single gradient, this implementation
    uses a two-call protocol:

      - Call 1 (look-ahead):  pass the gradient at the current point.
                              The transform returns a zero update (no move yet)
                              and stores the look-ahead point internally.
      - Call 2 (corrector):   pass the gradient at the look-ahead point.
                              The transform returns the corrector step and
                              resets the phase back to look-ahead.

    Usage::

        opt = extragradient(learning_rate=0.01)
        state = opt.init(params)

        # Inside your loop:
        grad_current = compute_grad(params)
        _, state = opt.update(grad_current, state, params)   # look-ahead, no move
        x_half = optax.apply_updates(params, jnp.zeros_like(params))  # params unchanged

        # Actually x_half is stored inside state; retrieve it:
        x_half = state.x_half

        grad_half = compute_grad(x_half)
        updates, state = opt.update(grad_half, state, params)  # corrector step
        params = optax.apply_updates(params, updates)

    For a saddle-point problem, wrap two ``extragradient`` instances (one for
    primal, one for dual) inside :func:`create_saddle_optimiser`.

    Args:
        learning_rate: Step size (scalar or callable ``step -> float``).

    Returns:
        An ``optax.GradientTransformation`` with the two-call extragradient
        protocol described above.
    """

    def init_fn(params):
        return ExtragradientState(
            x_half=jax.tree.map(jnp.zeros_like, params),
            phase=jnp.array(0, dtype=jnp.int32),
            step=jnp.array(0, dtype=jnp.int32),
        )

    def update_fn(updates, state, params):
        lr = _resolve_schedule(learning_rate, state.step)

        # Look-ahead phase: x_half = params - lr * grad
        x_half = jax.tree.map(lambda p, g: p - lr * g, params, updates)

        # Corrector phase: corrector = -lr * grad_half  (applied to *original* params)
        corrector = jax.tree.map(lambda g: -lr * g, updates)

        # Phase 0 → look-ahead (return zero update, store x_half, do NOT advance step)
        # Phase 1 → corrector  (return corrector update, clear x_half, advance step)
        is_corrector = state.phase == 1

        out_updates = jax.tree.map(
            lambda c: jnp.where(is_corrector, c, jnp.zeros_like(c)),
            corrector,
        )
        new_x_half = jax.tree.map(
            lambda xh, p: jnp.where(is_corrector, p, xh),
            x_half,
            params,
        )
        new_phase = jnp.where(is_corrector, jnp.array(0, dtype=jnp.int32), jnp.array(1, dtype=jnp.int32))
        new_step = jnp.where(is_corrector, state.step + 1, state.step)

        new_state = ExtragradientState(
            x_half=new_x_half,
            phase=new_phase,
            step=new_step,
        )
        return out_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


