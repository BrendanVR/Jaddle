import optax
import jax
import jax.numpy as jnp
import jaddle.jaddle_basic_types as jt


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


def _saddle_param_labels_granular(params):
    if hasattr(params, "_fields"):
        return type(params)(
            primal="primal_opt",
            dual_ineq="dual_ineq_opt",
            dual_eq="dual_eq_opt",
        )
    return {
        "primal": "primal_opt",
        "dual_ineq": "dual_ineq_opt",
        "dual_eq": "dual_eq_opt",
    }


def scale_by_inverse_metric(metric, epsilon: float = 1e-8):
    if metric is None:
        metric_array = None
        metric_tree = None
    else:
        try:
            metric_array = jnp.asarray(metric)
            metric_tree = None
        except TypeError:
            metric_array = None
            metric_tree = jax.tree_util.tree_map(lambda m: jnp.asarray(m), metric)

    def _is_array_leaf(value):
        return (
            hasattr(value, "dtype")
            and hasattr(value, "shape")
            and not isinstance(value, optax.MaskedNode)
        )

    def _scale_with_array(grad_leaf, metric_leaf):
        if not _is_array_leaf(grad_leaf):
            return grad_leaf
        return grad_leaf / (jnp.abs(metric_leaf) + epsilon)

    def _scale_with_tree(grad_leaf, metric_leaf):
        if not _is_array_leaf(grad_leaf):
            return grad_leaf
        if isinstance(metric_leaf, optax.MaskedNode):
            return grad_leaf
        return grad_leaf / (jnp.abs(metric_leaf) + epsilon)

    def init_fn(_):
        return ()

    def update_fn(updates, state, params=None):
        del params
        if metric_array is None and metric_tree is None:
            return updates, state

        if metric_array is not None:
            scaled_updates = jax.tree_util.tree_map(
                lambda g: _scale_with_array(g, metric_array), updates
            )
            return scaled_updates, state

        scaled_updates = jax.tree_util.tree_map(_scale_with_tree, updates, metric_tree)

        return scaled_updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def estimate_operator_norm(lp: jt.LP, num_iters: int = 20, seed: int = 0):
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
    lp: jt.LP, num_iters: int = 20, seed: int = 0, regularisation: float = 1e-6
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


def default_learning_rate_schedule(
    lp: jt.LP,
    scale: float = 1.0,
    transition_steps: int = int(1e3),
    decay_rate: float = None,
    decay_constant: float = 1.0,
    end_value: float = 1e-5,
    num_power_iters: int = 20,
    kind: str = "exponential",
):
    """
    Compute a problem-dependent learning rate schedule based on the estimated
    operator norm and condition number of the constraint matrix.

    The initial learning rate is set to scale / sigma_max (operator norm bound).
    The decay rate is set to 1 - decay_constant / kappa (condition number),
    so well-conditioned problems decay faster and ill-conditioned problems
    decay more slowly to allow progress along poorly-conditioned directions.

    Args:
        lp: Linear program (after any scaling has been applied).
        scale: Multiplier on 1/sigma_max for the initial LR (default 1.0).
        transition_steps: Steps between each decay application (default 10000).
        decay_rate: Override for decay rate. If None (default), computed from
                    the condition number as 1 - decay_constant / kappa.
        decay_constant: Constant c in decay_rate = 1 - c/kappa (default 0.5).
        end_value_fraction: Floor LR as a fraction of init_value (default 1e-5).
        num_power_iters: Power iterations for norm estimates (default 20).
        kind: Type of decay schedule (default "exponential").

    Returns:
        (schedule, sigma_max, sigma_min, kappa): An optax schedule and the
        estimated spectral quantities.
    """
    sigma_max, sigma_min, kappa = estimate_condition_number(
        lp, num_iters=num_power_iters
    )

    init_value = scale / jnp.maximum(sigma_max, 1e-12)

    if decay_rate is None:
        # Condition-number-dependent decay: well-conditioned -> faster decay
        decay_rate = float(jnp.clip(1.0 - decay_constant / kappa, 0.5, 0.999999))

    if kind == "exponential":
        schedule = optax.exponential_decay(
            init_value=float(init_value),
            transition_steps=transition_steps,
            decay_rate=decay_rate,
            end_value=float(end_value),
        )
    elif kind == "polynomial":
        schedule = optax.polynomial_schedule(
            init_value=float(init_value),
            end_value=float(end_value),
            power=1.0,
            transition_steps=transition_steps,
        )
    elif kind == "constant":
        schedule = optax.constant_schedule(float(init_value))
    else:
        raise ValueError(f"Unsupported schedule kind: {kind}")

    return schedule, sigma_max, sigma_min, kappa, init_value, decay_rate


def create_saddle_optimiser(
    primal_optimizer: optax.GradientTransformation,
    dual_optimizer: optax.GradientTransformation,
):
    optimiser = optax.partition(
        {
            "primal_opt": primal_optimizer,
            "dual_opt": dual_optimizer,
        },
        param_labels=_saddle_param_labels,
    )

    return optimiser


def optimistic_adam_metric_saddle(
    lr_primal=1e-3,
    lr_dual_ineq=1e-3,
    alpha: float = 5e-2,
    nesterov=True,
):
    primal_optimiser = optax.inject_hyperparams(optax.optimistic_adam_v2)(
        learning_rate=lr_primal,
        alpha=alpha,
        nesterov=nesterov,
    )
    dual_optimiser = optax.inject_hyperparams(optax.optimistic_adam_v2)(
        learning_rate=lr_dual_ineq,
        alpha=alpha,
        nesterov=nesterov,
    )

    return create_saddle_optimiser(primal_optimiser, dual_optimiser)
