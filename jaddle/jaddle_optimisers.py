import optax
import jax
import jax.numpy as jnp
from typing import Any, Optional, Sequence
from optax.projections import projection_box, projection_non_negative
from jaddle.jaddle_basic_types import (
    HedgePoolState,
    HedgeSaddleState,
    ScheduleLike,
    LP,
    SaddleState,
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


def optimistic_adam_saddle(
    lr_primal=1e-3,
    lr_dual=1e-3,
    alpha: float = 5e-2,
    nesterov=True,
):
    primal_optimiser = optax.inject_hyperparams(optax.optimistic_adam_v2)(
        learning_rate=lr_primal,
        alpha=alpha,
        nesterov=nesterov,
    )
    dual_optimiser = optax.inject_hyperparams(optax.optimistic_adam_v2)(
        learning_rate=lr_dual,
        alpha=alpha,
        nesterov=nesterov,
    )

    return create_saddle_optimiser(primal_optimiser, dual_optimiser)


def _resolve_schedule(schedule: ScheduleLike, step: jnp.ndarray) -> jnp.ndarray:
    if callable(schedule):
        return jnp.asarray(schedule(step), dtype=jnp.float32)
    return jnp.asarray(schedule, dtype=jnp.float32)


def _normalise_log_weights(log_weights: jnp.ndarray) -> jnp.ndarray:
    return log_weights - jax.nn.logsumexp(log_weights)


def hedge_ensemble_saddle(
    primal_experts: Sequence[optax.GradientTransformation],
    dual_experts: Sequence[optax.GradientTransformation],
    primal_eta: ScheduleLike = 5e-2,
    dual_eta: ScheduleLike = 5e-2,
    loss_clip: float = 1e2,
    lp: Optional[LP] = None,
):
    """
    Hedge-style ensemble over primal and dual optimiser pools.

    expert_loss_mode:
        - "dot_product": legacy first-order proxy loss.
        - "projected_kkt_merit": LP-aware loss based on one-step projected
          KKT merit change for each expert.
    lp:
        Required when expert_loss_mode="projected_kkt_merit".
    """
    if len(primal_experts) == 0:
        raise ValueError("primal_experts must contain at least one optimiser")
    if len(dual_experts) == 0:
        raise ValueError("dual_experts must contain at least one optimiser")

    primal_count = len(primal_experts)
    dual_count = len(dual_experts)
    use_primal_hedge_loss = primal_count > 1
    use_dual_hedge_loss = dual_count > 1
    has_ineq = lp is not None and lp.num_ineq_constraints() > 0
    has_eq = lp is not None and lp.num_eq_constraints() > 0

    if lp is None and (use_primal_hedge_loss or use_dual_hedge_loss):
        raise ValueError("lp must be provided for projected-KKT hedge losses")

    init_primal_log_weights = _normalise_log_weights(
        jnp.zeros((primal_count,), dtype=jnp.float32)
    )
    init_dual_log_weights = _normalise_log_weights(
        jnp.zeros((dual_count,), dtype=jnp.float32)
    )

    def init_fn(params: SaddleState) -> HedgeSaddleState:
        primal_states = tuple(opt.init(params.primal) for opt in primal_experts)
        dual_params = (params.dual_ineq, params.dual_eq)
        dual_states = tuple(opt.init(dual_params) for opt in dual_experts)

        return HedgeSaddleState(
            primal=HedgePoolState(primal_states, init_primal_log_weights),
            dual=HedgePoolState(dual_states, init_dual_log_weights),
            step=jnp.array(0, dtype=jnp.int32),
        )

    @jax.jit
    def update_fn(
        updates: SaddleState,
        state: HedgeSaddleState,
        params: SaddleState,
    ):
        grad_primal = updates.primal
        grad_dual_ineq = updates.dual_ineq
        grad_dual_eq = updates.dual_eq

        if use_primal_hedge_loss:
            primal_stationarity_grad = (
                lp.c + lp.A_ineq_T @ params.dual_ineq + lp.A_eq_T @ params.dual_eq
            )

        if use_dual_hedge_loss:
            objective_at_params_primal = lp.objective(params.primal)
            complementarity_scale = 1.0 + jnp.abs(objective_at_params_primal)

            if has_ineq:
                ineq_residual_at_params_primal = lp.A_ineq @ params.primal - lp.b_ineq
                max_ineq_violation_at_params_primal = jnp.max(
                    jnp.maximum(ineq_residual_at_params_primal, 0.0)
                )
            else:
                ineq_residual_at_params_primal = None
                max_ineq_violation_at_params_primal = jnp.array(
                    0.0, dtype=params.primal.dtype
                )

            if has_eq:
                eq_residual_at_params_primal = lp.A_eq @ params.primal - lp.b_eq
                max_eq_violation_at_params_primal = jnp.max(
                    jnp.abs(eq_residual_at_params_primal)
                )
            else:
                max_eq_violation_at_params_primal = jnp.array(
                    0.0, dtype=params.primal.dtype
                )

            constraint_bound_at_params_primal = jnp.maximum(
                max_ineq_violation_at_params_primal,
                max_eq_violation_at_params_primal,
            )

        def _primal_candidate_merit(candidate_primal: jnp.ndarray) -> jnp.ndarray:
            projected_primal = projection_box(
                candidate_primal - primal_stationarity_grad,
                lp.lower_bounds,
                lp.upper_bounds,
            )
            primal_grad_norm = jnp.max(jnp.abs(candidate_primal - projected_primal))

            objective_value = lp.objective(candidate_primal)

            if has_ineq:
                ineq_residual = lp.A_ineq @ candidate_primal - lp.b_ineq
                max_ineq_violation = jnp.max(jnp.maximum(ineq_residual, 0.0))
                complementarity_slack = jnp.max(
                    jnp.abs(params.dual_ineq * ineq_residual)
                ) / (1.0 + jnp.abs(objective_value))
            else:
                max_ineq_violation = jnp.array(0.0, dtype=params.primal.dtype)
                complementarity_slack = jnp.array(0.0, dtype=params.primal.dtype)

            if has_eq:
                eq_residual = lp.A_eq @ candidate_primal - lp.b_eq
                max_eq_violation = jnp.max(jnp.abs(eq_residual))
            else:
                max_eq_violation = jnp.array(0.0, dtype=params.primal.dtype)

            constraint_bound = jnp.maximum(max_ineq_violation, max_eq_violation)
            return jnp.maximum(
                primal_grad_norm,
                jnp.maximum(complementarity_slack, constraint_bound),
            )

        def _dual_candidate_merit(
            candidate_dual_ineq: jnp.ndarray,
            candidate_dual_eq: jnp.ndarray,
        ) -> jnp.ndarray:
            candidate_stationarity_grad = (
                lp.c + lp.A_ineq_T @ candidate_dual_ineq + lp.A_eq_T @ candidate_dual_eq
            )
            projected_primal = projection_box(
                params.primal - candidate_stationarity_grad,
                lp.lower_bounds,
                lp.upper_bounds,
            )
            primal_grad_norm = jnp.max(jnp.abs(params.primal - projected_primal))

            if has_ineq:
                complementarity_slack = (
                    jnp.max(
                        jnp.abs(candidate_dual_ineq * ineq_residual_at_params_primal)
                    )
                    / complementarity_scale
                )
            else:
                complementarity_slack = jnp.array(0.0, dtype=params.primal.dtype)

            return jnp.maximum(
                primal_grad_norm,
                jnp.maximum(
                    complementarity_slack,
                    constraint_bound_at_params_primal,
                ),
            )

        def _replace_tuple_entry(items, index, value):
            return items[:index] + (value,) + items[index + 1 :]

        def _make_primal_branch(expert_index, expert):
            def _branch(expert_states):
                expert_update, expert_state = expert.update(
                    grad_primal,
                    expert_states[expert_index],
                    params.primal,
                )
                next_states = _replace_tuple_entry(
                    expert_states,
                    expert_index,
                    expert_state,
                )

                if use_primal_hedge_loss:
                    candidate_primal = projection_box(
                        params.primal + expert_update,
                        lp.lower_bounds,
                        lp.upper_bounds,
                    )
                    loss = _primal_candidate_merit(candidate_primal)
                else:
                    loss = jnp.array(0.0, dtype=params.primal.dtype)

                return next_states, (expert_update, loss)

            return _branch

        primal_branches = tuple(
            _make_primal_branch(expert_index, expert)
            for expert_index, expert in enumerate(primal_experts)
        )

        def primal_scan_step(expert_states, expert_index):
            return jax.lax.switch(expert_index, primal_branches, expert_states)

        next_primal_states, (primal_updates_stacked, primal_losses) = jax.lax.scan(
            primal_scan_step,
            state.primal.expert_states,
            jnp.arange(primal_count, dtype=jnp.int32),
        )

        if use_primal_hedge_loss:
            primal_weights = jax.nn.softmax(state.primal.log_weights)
            mixed_primal_update = jnp.tensordot(
                primal_weights,
                primal_updates_stacked,
                axes=(0, 0),
            )
        else:
            mixed_primal_update = primal_updates_stacked[0]

        grad_dual = (grad_dual_ineq, grad_dual_eq)
        params_dual = (params.dual_ineq, params.dual_eq)

        def _make_dual_branch(expert_index, expert):
            def _branch(expert_states):
                expert_update, expert_state = expert.update(
                    grad_dual,
                    expert_states[expert_index],
                    params_dual,
                )
                update_dual_ineq, update_dual_eq = expert_update
                next_states = _replace_tuple_entry(
                    expert_states,
                    expert_index,
                    expert_state,
                )

                if use_dual_hedge_loss:
                    candidate_dual_ineq = projection_non_negative(
                        params.dual_ineq + update_dual_ineq
                    )
                    candidate_dual_eq = params.dual_eq + update_dual_eq
                    loss = _dual_candidate_merit(candidate_dual_ineq, candidate_dual_eq)
                else:
                    loss = jnp.array(0.0, dtype=params.primal.dtype)

                return next_states, ((update_dual_ineq, update_dual_eq), loss)

            return _branch

        dual_branches = tuple(
            _make_dual_branch(expert_index, expert)
            for expert_index, expert in enumerate(dual_experts)
        )

        def dual_scan_step(expert_states, expert_index):
            return jax.lax.switch(expert_index, dual_branches, expert_states)

        next_dual_states, (
            (dual_updates_ineq_stacked, dual_updates_eq_stacked),
            dual_losses,
        ) = jax.lax.scan(
            dual_scan_step,
            state.dual.expert_states,
            jnp.arange(dual_count, dtype=jnp.int32),
        )

        if use_dual_hedge_loss:
            dual_weights = jax.nn.softmax(state.dual.log_weights)
            mixed_dual_ineq_update = jnp.tensordot(
                dual_weights,
                dual_updates_ineq_stacked,
                axes=(0, 0),
            )
            mixed_dual_eq_update = jnp.tensordot(
                dual_weights,
                dual_updates_eq_stacked,
                axes=(0, 0),
            )
        else:
            mixed_dual_ineq_update = dual_updates_ineq_stacked[0]
            mixed_dual_eq_update = dual_updates_eq_stacked[0]

        if use_primal_hedge_loss:
            primal_eta_value = _resolve_schedule(primal_eta, state.step)
            primal_losses = jnp.clip(primal_losses, -loss_clip, loss_clip)
            next_primal_log_weights = _normalise_log_weights(
                state.primal.log_weights - primal_eta_value * primal_losses
            )
        else:
            next_primal_log_weights = state.primal.log_weights

        if use_dual_hedge_loss:
            dual_eta_value = _resolve_schedule(dual_eta, state.step)
            dual_losses = jnp.clip(dual_losses, -loss_clip, loss_clip)
            next_dual_log_weights = _normalise_log_weights(
                state.dual.log_weights - dual_eta_value * dual_losses
            )
        else:
            next_dual_log_weights = state.dual.log_weights

        next_state = HedgeSaddleState(
            primal=HedgePoolState(next_primal_states, next_primal_log_weights),
            dual=HedgePoolState(next_dual_states, next_dual_log_weights),
            step=state.step + jnp.array(1, dtype=jnp.int32),
        )

        mixed_update = SaddleState(
            primal=mixed_primal_update,
            dual_ineq=mixed_dual_ineq_update,
            dual_eq=mixed_dual_eq_update,
        )

        return mixed_update, next_state

    return optax.GradientTransformation(init_fn, update_fn)


def hedge_weights_from_state(opt_state: Any):
    """
    Extract (primal_weights, dual_weights) from a hedge ensemble optimiser state.

    Returns:
        Tuple of jnp arrays (primal_weights, dual_weights) if available, else None.
    """
    try:
        primal_log_weights = opt_state.primal.log_weights
        dual_log_weights = opt_state.dual.log_weights
    except AttributeError:
        return None

    return jax.nn.softmax(primal_log_weights), jax.nn.softmax(dual_log_weights)
