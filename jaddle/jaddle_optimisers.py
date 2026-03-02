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


def _mean_abs(x: jnp.ndarray) -> jnp.ndarray:
    if x.size == 0:
        return jnp.array(0.0, dtype=jnp.float32)
    return jnp.mean(jnp.abs(x))


def hedge_ensemble_saddle(
    lp: LP,
    primal_experts: Sequence[optax.GradientTransformation],
    dual_experts: Sequence[optax.GradientTransformation],
    primal_eta: ScheduleLike = 1.0,
    dual_eta: ScheduleLike = 1.0,
    loss_clip: float = 1e2,
    loss_scale_ema_decay: float = 0.95,
    loss_scale_floor: float = 1e-4,
    exploration_rate: float = 0.02,
    center_losses: bool = True,
):
    if len(primal_experts) == 0:
        raise ValueError("primal_experts must contain at least one optimiser")
    if len(dual_experts) == 0:
        raise ValueError("dual_experts must contain at least one optimiser")

    primal_count = len(primal_experts)
    dual_count = len(dual_experts)

    if not (0.0 <= exploration_rate < 1.0):
        raise ValueError("exploration_rate must satisfy 0 <= exploration_rate < 1")

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
            primal=HedgePoolState(
                expert_states=primal_states,
                log_weights=init_primal_log_weights,
                loss_scale=jnp.array(1.0, dtype=jnp.float32),
                last_raw_losses=jnp.zeros((primal_count,), dtype=jnp.float32),
                last_normalized_losses=jnp.zeros((primal_count,), dtype=jnp.float32),
                last_clipped_losses=jnp.zeros((primal_count,), dtype=jnp.float32),
                last_centered_losses=jnp.zeros((primal_count,), dtype=jnp.float32),
            ),
            dual=HedgePoolState(
                expert_states=dual_states,
                log_weights=init_dual_log_weights,
                loss_scale=jnp.array(1.0, dtype=jnp.float32),
                last_raw_losses=jnp.zeros((dual_count,), dtype=jnp.float32),
                last_normalized_losses=jnp.zeros((dual_count,), dtype=jnp.float32),
                last_clipped_losses=jnp.zeros((dual_count,), dtype=jnp.float32),
                last_centered_losses=jnp.zeros((dual_count,), dtype=jnp.float32),
            ),
            step=jnp.array(0, dtype=jnp.int32),
            last_primal_eta=jnp.array(0.0, dtype=jnp.float32),
            last_dual_eta=jnp.array(0.0, dtype=jnp.float32),
        )

    def update_fn(
        updates: SaddleState,
        state: HedgeSaddleState,
        params: SaddleState,
    ):
        grad_primal = updates.primal
        grad_dual_ineq = updates.dual_ineq
        grad_dual_eq = updates.dual_eq

        primal_base_weights = jax.nn.softmax(state.primal.log_weights)
        dual_base_weights = jax.nn.softmax(state.dual.log_weights)

        primal_weights = (1.0 - exploration_rate) * primal_base_weights + (
            exploration_rate / primal_count
        )
        dual_weights = (1.0 - exploration_rate) * dual_base_weights + (
            exploration_rate / dual_count
        )

        # ---------- Primal experts ----------
        primal_step_updates = []
        primal_next_states = []
        primal_losses = []

        for i, expert in enumerate(primal_experts):
            expert_update, expert_state = expert.update(
                grad_primal,
                state.primal.expert_states[i],
                params.primal,
            )

            primal_candidate = projection_box(
                optax.apply_updates(params.primal, expert_update),
                lp.lower_bounds,
                lp.upper_bounds,
            )
            primal_effective_update = primal_candidate - params.primal
            primal_loss = grad_primal @ primal_effective_update

            primal_step_updates.append(primal_effective_update)
            primal_next_states.append(expert_state)
            primal_losses.append(primal_loss)

        primal_updates_stacked = jnp.stack(primal_step_updates, axis=0)
        mixed_primal_update = jnp.tensordot(
            primal_weights, primal_updates_stacked, axes=(0, 0)
        )

        # ---------- Dual experts ----------
        dual_step_updates_ineq = []
        dual_step_updates_eq = []
        dual_next_states = []
        dual_losses = []

        grad_dual = (grad_dual_ineq, grad_dual_eq)
        params_dual = (params.dual_ineq, params.dual_eq)

        for i, expert in enumerate(dual_experts):
            expert_update, expert_state = expert.update(
                grad_dual,
                state.dual.expert_states[i],
                params_dual,
            )
            upd_ineq, upd_eq = expert_update
            dual_candidate_ineq = projection_non_negative(params.dual_ineq + upd_ineq)
            dual_candidate_eq = params.dual_eq + upd_eq  # No projection for eq dual

            dual_effective_update_ineq = dual_candidate_ineq - params.dual_ineq
            dual_effective_update_eq = dual_candidate_eq - params.dual_eq

            dual_loss = (
                grad_dual_ineq @ dual_effective_update_ineq
                + grad_dual_eq @ dual_effective_update_eq
            )

            dual_step_updates_ineq.append(dual_effective_update_ineq)
            dual_step_updates_eq.append(dual_effective_update_eq)
            dual_next_states.append(expert_state)
            dual_losses.append(-dual_loss)

        dual_updates_ineq_stacked = jnp.stack(dual_step_updates_ineq, axis=0)
        dual_updates_eq_stacked = jnp.stack(dual_step_updates_eq, axis=0)

        mixed_dual_ineq_update = jnp.tensordot(
            dual_weights, dual_updates_ineq_stacked, axes=(0, 0)
        )
        mixed_dual_eq_update = jnp.tensordot(
            dual_weights, dual_updates_eq_stacked, axes=(0, 0)
        )

        # ---------- Update log-weights ----------
        primal_eta_value = _resolve_schedule(primal_eta, state.step)
        dual_eta_value = _resolve_schedule(dual_eta, state.step)

        primal_losses = jnp.asarray(primal_losses, dtype=jnp.float32)
        dual_losses = jnp.asarray(dual_losses, dtype=jnp.float32)

        primal_loss_scale = loss_scale_ema_decay * state.primal.loss_scale + (
            1.0 - loss_scale_ema_decay
        ) * _mean_abs(primal_losses)
        primal_loss_scale = jnp.maximum(primal_loss_scale, loss_scale_floor)

        dual_loss_scale = loss_scale_ema_decay * state.dual.loss_scale + (
            1.0 - loss_scale_ema_decay
        ) * _mean_abs(dual_losses)
        dual_loss_scale = jnp.maximum(dual_loss_scale, loss_scale_floor)

        normalized_primal_losses = primal_losses / primal_loss_scale
        normalized_dual_losses = dual_losses / dual_loss_scale

        clipped_primal_losses = jnp.clip(
            normalized_primal_losses,
            -loss_clip,
            loss_clip,
        )
        clipped_dual_losses = jnp.clip(
            normalized_dual_losses,
            -loss_clip,
            loss_clip,
        )

        if center_losses:
            centered_primal_losses = clipped_primal_losses - jnp.sum(
                primal_weights * clipped_primal_losses
            )
            centered_dual_losses = clipped_dual_losses - jnp.sum(
                dual_weights * clipped_dual_losses
            )
        else:
            centered_primal_losses = clipped_primal_losses
            centered_dual_losses = clipped_dual_losses

        next_primal_log_weights = _normalise_log_weights(
            state.primal.log_weights - primal_eta_value * centered_primal_losses
        )
        next_dual_log_weights = _normalise_log_weights(
            state.dual.log_weights - dual_eta_value * centered_dual_losses
        )

        next_state = HedgeSaddleState(
            primal=HedgePoolState(
                expert_states=tuple(primal_next_states),
                log_weights=next_primal_log_weights,
                loss_scale=primal_loss_scale,
                last_raw_losses=primal_losses,
                last_normalized_losses=normalized_primal_losses,
                last_clipped_losses=clipped_primal_losses,
                last_centered_losses=centered_primal_losses,
            ),
            dual=HedgePoolState(
                expert_states=tuple(dual_next_states),
                log_weights=next_dual_log_weights,
                loss_scale=dual_loss_scale,
                last_raw_losses=dual_losses,
                last_normalized_losses=normalized_dual_losses,
                last_clipped_losses=clipped_dual_losses,
                last_centered_losses=centered_dual_losses,
            ),
            step=state.step + jnp.array(1, dtype=jnp.int32),
            last_primal_eta=primal_eta_value,
            last_dual_eta=dual_eta_value,
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


def hedge_diagnostics_from_state(opt_state: Any):
    """Extract hedge diagnostics (weights, losses, scales, eta) if available."""
    try:
        primal_weights = jax.nn.softmax(opt_state.primal.log_weights)
        dual_weights = jax.nn.softmax(opt_state.dual.log_weights)

        return {
            "primal_weights": primal_weights,
            "dual_weights": dual_weights,
            "primal_raw_losses": opt_state.primal.last_raw_losses,
            "dual_raw_losses": opt_state.dual.last_raw_losses,
            "primal_normalized_losses": opt_state.primal.last_normalized_losses,
            "dual_normalized_losses": opt_state.dual.last_normalized_losses,
            "primal_clipped_losses": opt_state.primal.last_clipped_losses,
            "dual_clipped_losses": opt_state.dual.last_clipped_losses,
            "primal_centered_losses": opt_state.primal.last_centered_losses,
            "dual_centered_losses": opt_state.dual.last_centered_losses,
            "primal_loss_scale": opt_state.primal.loss_scale,
            "dual_loss_scale": opt_state.dual.loss_scale,
            "primal_eta": opt_state.last_primal_eta,
            "dual_eta": opt_state.last_dual_eta,
            "step": opt_state.step,
        }
    except AttributeError:
        return None
