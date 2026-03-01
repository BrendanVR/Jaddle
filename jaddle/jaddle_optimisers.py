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
    lp: LP,
    primal_experts: Sequence[optax.GradientTransformation],
    dual_experts: Sequence[optax.GradientTransformation],
    primal_eta: ScheduleLike = 1.0,
    dual_eta: ScheduleLike = 1.0,
    loss_clip: float = 1e2,
):
    if len(primal_experts) == 0:
        raise ValueError("primal_experts must contain at least one optimiser")
    if len(dual_experts) == 0:
        raise ValueError("dual_experts must contain at least one optimiser")

    primal_count = len(primal_experts)
    dual_count = len(dual_experts)

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

    def update_fn(
        updates: SaddleState,
        state: HedgeSaddleState,
        params: SaddleState,
    ):
        grad_primal = updates.primal
        grad_dual_ineq = updates.dual_ineq
        grad_dual_eq = updates.dual_eq

        primal_weights = jax.nn.softmax(state.primal.log_weights)
        dual_weights = jax.nn.softmax(state.dual.log_weights)

        primal_step_updates = []
        primal_next_states = []
        primal_expert_solutions = []
        primal_next_states = []

        for expert_index, expert in enumerate(primal_experts):

            expert_update, expert_state = expert.update(
                grad_primal,
                state.primal.expert_states[expert_index],
                params.primal,
            )

            primal_expert_solution = projection_box(
                optax.apply_updates(params.primal, expert_update),
                lp.lower_bounds,
                lp.upper_bounds,
            )

            primal_expert_solutions.append(primal_expert_solution)
            primal_step_updates.append(expert_update)
            primal_next_states.append(expert_state)

        primal_updates_stacked = jnp.stack(primal_step_updates, axis=0)
        mixed_primal_update = jnp.tensordot(
            primal_weights,
            primal_updates_stacked,
            axes=(0, 0),
        )

        mixed_primal_solution = projection_box(
            optax.apply_updates(params.primal, mixed_primal_update),
            lp.lower_bounds,
            lp.upper_bounds,
        )

        dual_step_updates_ineq = []
        dual_step_updates_eq = []
        dual_expert_solutions_ineq = []
        dual_expert_solutions_eq = []
        dual_next_states = []

        grad_dual = (grad_dual_ineq, grad_dual_eq)
        params_dual = (params.dual_ineq, params.dual_eq)

        for expert_index, expert in enumerate(dual_experts):
            expert_update, expert_state = expert.update(
                grad_dual,
                state.dual.expert_states[expert_index],
                params_dual,
            )
            update_dual_ineq, update_dual_eq = expert_update

            dual_expert_solution_ineq = projection_non_negative(
                optax.apply_updates(params.dual_ineq, update_dual_ineq),
            )

            dual_expert_solution_eq = optax.apply_updates(
                params.dual_eq, update_dual_eq
            )

            dual_expert_solutions_ineq.append(dual_expert_solution_ineq)
            dual_expert_solutions_eq.append(dual_expert_solution_eq)
            dual_step_updates_ineq.append(update_dual_ineq)
            dual_step_updates_eq.append(update_dual_eq)
            dual_next_states.append(expert_state)

        dual_updates_ineq_stacked = jnp.stack(dual_step_updates_ineq, axis=0)
        dual_updates_eq_stacked = jnp.stack(dual_step_updates_eq, axis=0)

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

        mixed_dual_ineq_solution = projection_non_negative(
            optax.apply_updates(params.dual_ineq, mixed_dual_ineq_update),
        )

        mixed_dual_eq_solution = optax.apply_updates(
            params.dual_eq, mixed_dual_eq_update
        )

        next_primal_states = tuple(primal_next_states)
        next_dual_states = tuple(dual_next_states)

        primal_eta_value = _resolve_schedule(primal_eta, state.step)
        dual_eta_value = _resolve_schedule(dual_eta, state.step)

        primal_losses = []

        for expert_solution in primal_expert_solutions:
            primal_loss = (
                lp.c @ expert_solution
                + mixed_dual_ineq_solution @ (lp.A_ineq @ expert_solution - lp.b_ineq)
                + mixed_dual_eq_solution @ (lp.A_eq @ expert_solution - lp.b_eq)
            ) - (
                lp.c @ mixed_primal_solution
                + mixed_dual_ineq_solution
                @ (lp.A_ineq @ mixed_primal_solution - lp.b_ineq)
                + mixed_dual_eq_solution @ (lp.A_eq @ mixed_primal_solution - lp.b_eq)
            )
            primal_losses.append(-primal_loss)

        dual_losses = []

        for expert_solution_ineq, expert_solution_eq in zip(
            dual_expert_solutions_ineq, dual_expert_solutions_eq
        ):
            dual_loss_ineq = expert_solution_ineq @ (
                lp.A_ineq @ mixed_primal_solution - lp.b_ineq
            )
            dual_loss_eq = expert_solution_eq @ (
                lp.A_eq @ mixed_primal_solution - lp.b_eq
            )
            dual_loss = (
                dual_loss_ineq
                + dual_loss_eq
                - (
                    mixed_dual_ineq_solution
                    @ (lp.A_ineq @ mixed_primal_solution - lp.b_ineq)
                    + mixed_dual_eq_solution
                    @ (lp.A_eq @ mixed_primal_solution - lp.b_eq)
                )
            )
            dual_losses.append(dual_loss)

        primal_losses = jnp.array(primal_losses)
        dual_losses = jnp.array(dual_losses)

        primal_losses = jnp.clip(primal_losses, -loss_clip, loss_clip)
        dual_losses = jnp.clip(dual_losses, -loss_clip, loss_clip)

        next_primal_log_weights = _normalise_log_weights(
            state.primal.log_weights - primal_eta_value * primal_losses
        )
        next_dual_log_weights = _normalise_log_weights(
            state.dual.log_weights - dual_eta_value * dual_losses
        )

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
