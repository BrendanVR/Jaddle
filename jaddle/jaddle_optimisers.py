import os

import optax
import jax
import jax.numpy as jnp
from typing import Any, Optional, Sequence
from optax.projections import projection_box, projection_non_negative
from jaddle.jaddle_basic_types import (
    ExtragradientState,
    HedgePoolState,
    HedgeSaddleState,
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


def _normalise_log_weights(log_weights: jnp.ndarray) -> jnp.ndarray:
    return log_weights - jax.nn.logsumexp(log_weights)


def _normalise_log_weights_with_mask(
    log_weights: jnp.ndarray, active_mask: jnp.ndarray
) -> jnp.ndarray:
    mask = jnp.asarray(active_mask)
    safe_mask = jnp.where(jnp.sum(mask) > 0, mask, jnp.ones_like(mask))
    masked_logits = jnp.where(
        safe_mask > 0, log_weights, jnp.full_like(log_weights, -1e30)
    )
    return masked_logits - jax.nn.logsumexp(masked_logits)


def _masked_softmax(log_weights: jnp.ndarray, active_mask: jnp.ndarray) -> jnp.ndarray:
    logits = _normalise_log_weights_with_mask(log_weights, active_mask)
    probs = jax.nn.softmax(logits)
    mask = jnp.asarray(active_mask)
    denom = jnp.maximum(jnp.sum(probs * mask), 1e-30)
    return (probs * mask) / denom


def _mean_abs(x: jnp.ndarray) -> jnp.ndarray:
    if x.size == 0:
        return jnp.array(0.0)
    return jnp.mean(jnp.abs(x))


def _smooth_clamp(target, prev, theta, lo, hi):
    """Log-space EMA of `target` toward `prev`, then clamp to [lo, hi].

    Shared by the per-expert k/eta controls; mirrors the smoothing used in
    solve()'s per_iterate_k path so both layers behave identically.
    """
    log_val = theta * jnp.log(target) + (1.0 - theta) * jnp.log(prev)
    return jnp.clip(jnp.exp(log_val), lo, hi)


def hedge_ensemble_saddle(
    lp: LP,
    experts: Sequence[
        tuple[optax.GradientTransformation, optax.GradientTransformation]
    ],
    eta: ScheduleLike = 1.0,
    loss_clip: float = 1e2,
    loss_scale_ema_decay: float = 0.95,
    loss_scale_floor: float = 1e-4,
    exploration_rate: float = 0.02,
    center_losses: bool = True,
    per_expert_k: bool = False,
    per_expert_k_theta: float = 0.1,
    per_expert_k_lo: float = 0.1,
    per_expert_k_hi: float = 10.0,
):
    """Hedge ensemble over JOINT primal/dual players for a saddle problem.

    Each entry of ``experts`` is a ``(primal_opt, dual_opt)`` pair: one player
    that proposes both a primal step and a dual step. A single weight vector
    governs the pool, so a player's primal and dual moves are mixed with the
    SAME weight and the player is scored on the change in the squared KKT
    residual its step would produce (see the per-player loss below): a step
    that moves toward the saddle scores < 0, a do-nothing step scores 0.

    Optional ``per_expert_k`` lets each player self-balance its primal/dual
    step ratio ``k_i = sqrt(||dx_i|| / ||dy_i||)`` from its own update norms
    before mixing (matvec-free). The companion magnitude controller
    (``per_expert_eta``) was removed: it computed the PDLP step bound from the
    optimiser's already-LR-scaled updates, which is dimensionally wrong (it
    reads a small step as "too small" and amplifies ``eta`` without bound),
    so magnitude is left to each expert's own learning rate.
    """
    if len(experts) == 0:
        raise ValueError("experts must contain at least one (primal, dual) pair")

    primal_experts = tuple(p for (p, _) in experts)
    dual_experts = tuple(d for (_, d) in experts)
    count = len(experts)

    if not (0.0 <= exploration_rate < 1.0):
        raise ValueError("exploration_rate must satisfy 0 <= exploration_rate < 1")

    init_log_weights = _normalise_log_weights(jnp.zeros((count,)))

    def init_fn(params: SaddleState) -> HedgeSaddleState:
        primal_states = tuple(opt.init(params.primal) for opt in primal_experts)
        dual_params = (params.dual_ineq, params.dual_eq)
        dual_states = tuple(opt.init(dual_params) for opt in dual_experts)

        return HedgeSaddleState(
            pool=HedgePoolState(
                primal_states=primal_states,
                dual_states=dual_states,
                log_weights=init_log_weights,
                active_mask=jnp.ones((count,)),
                loss_scale=jnp.array(1.0),
                last_raw_losses=jnp.zeros((count,)),
                last_normalized_losses=jnp.zeros((count,)),
                last_clipped_losses=jnp.zeros((count,)),
                last_centered_losses=jnp.zeros((count,)),
                expert_k=jnp.ones((count,)),
            ),
            step=jnp.array(0, dtype=jnp.int32),
            last_eta=jnp.array(0.0),
        )

    def update_fn(
        updates: SaddleState,
        state: HedgeSaddleState,
        params: SaddleState,
    ):
        grad_primal = updates.primal
        grad_dual_ineq = updates.dual_ineq
        grad_dual_eq = updates.dual_eq

        active_mask = jnp.asarray(state.pool.active_mask)
        base_weights = _masked_softmax(state.pool.log_weights, active_mask)
        weights = (1.0 - exploration_rate) * base_weights + (exploration_rate / count)
        weights = weights * active_mask
        weights = weights / jnp.maximum(jnp.sum(weights), 1e-30)

        grad_dual = (grad_dual_ineq, grad_dual_eq)
        params_dual = (params.dual_ineq, params.dual_eq)

        # ---------- Raw player updates (pre-control, pre-projection) ----------
        # Each player proposes a primal and a dual step. Raw optax updates are
        # collected first so per-player k_i/eta_i (which need the player's own
        # primal/dual update norms together) can scale them before projection
        # (scale-then-project), matching solve()'s per_iterate_k convention.
        primal_raw_updates = []
        primal_next_states = []
        dual_raw_updates = []
        dual_next_states = []
        for i in range(count):
            p_update, p_state = primal_experts[i].update(
                grad_primal, state.pool.primal_states[i], params.primal
            )
            d_update, d_state = dual_experts[i].update(
                grad_dual, state.pool.dual_states[i], params_dual
            )
            primal_raw_updates.append(p_update)
            primal_next_states.append(p_state)
            dual_raw_updates.append(d_update)
            dual_next_states.append(d_state)

        # ---------- Per-player ratio control k_i ----------
        # k_i = sqrt(||dx_i|| / ||dy_i||) balances a player's primal vs dual
        # step magnitude (matvec-free, from the raw update norms); the step is
        # then split (1/k_i, k_i) so the dual/primal ratio is k_i**2. Magnitude
        # is left to each expert's own learning rate.
        if per_expert_k:
            prev_k = state.pool.expert_k
            next_expert_k = []
            for i in range(count):
                up = primal_raw_updates[i]
                ud_ineq, ud_eq = dual_raw_updates[i]
                ud = jnp.concatenate([ud_eq, ud_ineq])
                norm_p = jnp.sqrt(up @ up) + 1e-30
                norm_d = jnp.sqrt(ud @ ud) + 1e-30
                k_i = _smooth_clamp(
                    jnp.sqrt(norm_p / norm_d),
                    prev_k[i],
                    per_expert_k_theta,
                    per_expert_k_lo,
                    per_expert_k_hi,
                )
                next_expert_k.append(k_i)
            next_expert_k = jnp.stack(next_expert_k)
            primal_scale = 1.0 / next_expert_k
            dual_scale = next_expert_k
        else:
            next_expert_k = state.pool.expert_k
            primal_scale = jnp.ones((count,))
            dual_scale = jnp.ones((count,))

        # ---------- Per-player joint loss: scale, project, score ----------
        # The score is the change in the (non-negative) squared KKT residual
        # the player's step would produce:
        #
        #   merit(z) = ½‖grad_primal‖²            (primal stationarity, c + Aᵀy)
        #            + ½‖grad_dual‖²              (dual feasibility, b − Ax)
        #   loss_i   = merit(z + step_i) − merit(z)
        #
        # A do-nothing player gets loss 0 (neutral); a player that genuinely
        # moves toward the saddle gets loss < 0 and is rewarded. This replaces
        # the old grad·step linearization, under which the inert player won —
        # grad·step is signed and a zero step scored ~0, beating real players
        # whose noisy one-step progress is frequently positive.
        #
        # At z + step_i the residuals shift exactly by the constraint matvecs:
        #   grad_primal_new = grad_primal + Aᵀ·dy_i   (grad_primal = c + Aᵀy)
        #   grad_dual_new   = grad_dual   − A·dx_i     (grad_dual   = b − Ax)
        # so this costs two matvecs per player per iteration (Aᵀ·dy_i, A·dx_i).
        merit_now = 0.5 * (grad_primal @ grad_primal) + 0.5 * (
            grad_dual_ineq @ grad_dual_ineq + grad_dual_eq @ grad_dual_eq
        )

        primal_step_updates = []
        dual_step_updates_ineq = []
        dual_step_updates_eq = []
        joint_losses = []
        for i in range(count):
            # Primal half.
            scaled_primal = primal_raw_updates[i] * primal_scale[i]
            primal_candidate = projection_box(
                optax.apply_updates(params.primal, scaled_primal),
                lp.lower_bounds,
                lp.upper_bounds,
            )
            primal_effective = primal_candidate - params.primal

            # Dual half.
            upd_ineq, upd_eq = dual_raw_updates[i]
            upd_ineq = upd_ineq * dual_scale[i]
            upd_eq = upd_eq * dual_scale[i]
            dual_candidate_ineq = projection_non_negative(params.dual_ineq + upd_ineq)
            dual_candidate_eq = params.dual_eq + upd_eq  # No projection for eq dual
            dual_effective_ineq = dual_candidate_ineq - params.dual_ineq
            dual_effective_eq = dual_candidate_eq - params.dual_eq

            # Merit at z + step_i. dy stacks as [eq, ineq] to match lp.A's row
            # order (A = [A_eq; A_ineq]); A·dx splits back the same way.
            dy_stacked = jnp.concatenate([dual_effective_eq, dual_effective_ineq])
            ATdy = lp.A_T @ dy_stacked
            Adx = lp.A @ primal_effective
            grad_primal_new = grad_primal + ATdy
            grad_dual_eq_new = grad_dual_eq - Adx[: lp.n_eq]
            grad_dual_ineq_new = grad_dual_ineq - Adx[lp.n_eq :]
            merit_new = 0.5 * (grad_primal_new @ grad_primal_new) + 0.5 * (
                grad_dual_ineq_new @ grad_dual_ineq_new
                + grad_dual_eq_new @ grad_dual_eq_new
            )
            joint_losses.append(merit_new - merit_now)

            primal_step_updates.append(primal_effective)
            dual_step_updates_ineq.append(dual_effective_ineq)
            dual_step_updates_eq.append(dual_effective_eq)

        # Mix with the SINGLE shared weight vector.
        mixed_primal_update = jnp.tensordot(
            weights, jnp.stack(primal_step_updates, axis=0), axes=(0, 0)
        )
        mixed_dual_ineq_update = jnp.tensordot(
            weights, jnp.stack(dual_step_updates_ineq, axis=0), axes=(0, 0)
        )
        mixed_dual_eq_update = jnp.tensordot(
            weights, jnp.stack(dual_step_updates_eq, axis=0), axes=(0, 0)
        )

        # ---------- Update log-weights (one Hedge update for the pool) ----------
        eta_value = _resolve_schedule(eta, state.step)

        joint_losses = jnp.asarray(joint_losses)
        joint_losses = jnp.where(active_mask > 0, joint_losses, 0.0)

        loss_scale = loss_scale_ema_decay * state.pool.loss_scale + (
            1.0 - loss_scale_ema_decay
        ) * _mean_abs(joint_losses)
        loss_scale = jnp.maximum(loss_scale, loss_scale_floor)

        normalized_losses = joint_losses / loss_scale
        clipped_losses = jnp.clip(normalized_losses, -loss_clip, loss_clip)

        if center_losses:
            centered_losses = clipped_losses - jnp.sum(weights * clipped_losses)
        else:
            centered_losses = clipped_losses

        next_log_weights = _normalise_log_weights_with_mask(
            state.pool.log_weights - eta_value * centered_losses,
            active_mask,
        )

        next_state = HedgeSaddleState(
            pool=HedgePoolState(
                primal_states=tuple(primal_next_states),
                dual_states=tuple(dual_next_states),
                log_weights=next_log_weights,
                active_mask=active_mask,
                loss_scale=loss_scale,
                last_raw_losses=joint_losses,
                last_normalized_losses=normalized_losses,
                last_clipped_losses=clipped_losses,
                last_centered_losses=centered_losses,
                expert_k=next_expert_k,
            ),
            step=state.step + jnp.array(1, dtype=jnp.int32),
            last_eta=eta_value,
        )

        mixed_update = SaddleState(
            primal=mixed_primal_update,
            dual_ineq=mixed_dual_ineq_update,
            dual_eq=mixed_dual_eq_update,
        )

        return mixed_update, next_state

    return optax.GradientTransformation(init_fn, update_fn)


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


def hedge_weights_from_state(opt_state: Any):
    """
    Extract the joint-player weight vector from a hedge ensemble optimiser
    state.

    Returns:
        A jnp array of player weights if available, else None.
    """
    try:
        log_weights = opt_state.pool.log_weights
        active_mask = opt_state.pool.active_mask
    except AttributeError:
        return None

    return _masked_softmax(log_weights, active_mask)


def hedge_diagnostics_from_state(opt_state: Any):
    """Extract hedge diagnostics (weights, losses, scale, eta) if available."""
    try:
        weights = _masked_softmax(
            opt_state.pool.log_weights,
            opt_state.pool.active_mask,
        )

        return {
            "weights": weights,
            "active_mask": opt_state.pool.active_mask,
            "raw_losses": opt_state.pool.last_raw_losses,
            "normalized_losses": opt_state.pool.last_normalized_losses,
            "clipped_losses": opt_state.pool.last_clipped_losses,
            "centered_losses": opt_state.pool.last_centered_losses,
            "loss_scale": opt_state.pool.loss_scale,
            "eta": opt_state.last_eta,
            "expert_k": opt_state.pool.expert_k,
            "step": opt_state.step,
        }
    except AttributeError:
        return None
