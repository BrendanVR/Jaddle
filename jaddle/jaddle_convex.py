# %%
import jax
import jax.numpy as jnp
from optax.projections import projection_non_negative, projection_box
import optax
import functools
from typing import NamedTuple
import time
from jaddle.jaddle_basic_types import CP, SaddleState
import jaddle.jaddle_optimisers as jo
import numpy as np

_CONVEX_RUN_EPOCH_CACHE = {}


def __sps(
    max_iter,
    start_iter,
    cp: CP,
    optimiser,
    initial_solution,
    initial_avg_state=None,
    initial_opt_state=None,
    weight_function=lambda _: 1.0,
    total_weight=0.0,
    average=True,
    update_mode="synchronous",
    k_scaling=False,
    k_init=1.0,
):
    # The stepping scheme is selected by `update_mode`. This derived boolean
    # keeps the per-scheme branching below readable while the string stays the
    # single source of truth.
    extragradient = update_mode == "extragradient"
    # k-scaling is an orthogonal option (any update_mode): a primal weight k
    # rescales the primal/dual gradients by (1/k, k) before opt_update, so the
    # dual/primal step ratio is k**2. When on, k is packed into opt_state and
    # rebalanced at each restart in `solve` (PDLP-style); constant within an
    # epoch.

    def projection_primal(primal_state):
        return projection_box(primal_state, cp.lower_bounds, cp.upper_bounds)

    def langrangian(state):
        return (
            cp.objective(state.primal)
            + state.dual_ineq @ cp.constraints_ineq(state.primal)
            + state.dual_eq @ cp.constraints_eq(state.primal)
        )

    def grad(state):
        gradient = jax.grad(langrangian)(state)
        return SaddleState(
            primal=gradient.primal,
            dual_ineq=-gradient.dual_ineq,
            dual_eq=-gradient.dual_eq,
        )

    def opt_update(gradient, opt_state, state):
        return optimiser.update(gradient, opt_state, state)

    def zero_dual(gradient):
        return SaddleState(
            primal=gradient.primal,
            dual_ineq=jnp.zeros_like(gradient.dual_ineq),
            dual_eq=jnp.zeros_like(gradient.dual_eq),
        )

    def keep_only_primal(updates):
        return SaddleState(
            primal=updates.primal,
            dual_ineq=jnp.zeros_like(updates.dual_ineq),
            dual_eq=jnp.zeros_like(updates.dual_eq),
        )

    def keep_only_dual(updates):
        return SaddleState(
            primal=jnp.zeros_like(updates.primal),
            dual_ineq=updates.dual_ineq,
            dual_eq=updates.dual_eq,
        )

    def scale_by_k(gradient, k):
        # Primal weight split: (1/k, k) on (primal, dual) gradients.
        return SaddleState(
            primal=gradient.primal / k,
            dual_ineq=gradient.dual_ineq * k,
            dual_eq=gradient.dual_eq * k,
        )

    # When k-scaling is on, k is carried as the last element of opt_state. These
    # helpers keep the step bodies agnostic to whether k is packed or not.
    def unpack_k(opt_state):
        if k_scaling:
            return opt_state
        return opt_state, None

    def pack_k(opt_state, k):
        if k_scaling:
            return (opt_state, k)
        return opt_state

    cache_key = (
        id(cp),
        id(optimiser),
        id(weight_function),
        bool(average),
        update_mode,
        bool(k_scaling),
    )
    run_epoch = _CONVEX_RUN_EPOCH_CACHE.get(cache_key)

    if run_epoch is None:

        @functools.partial(
            jax.jit,
            static_argnames=("max_iter",),
        )
        def run_epoch(
            max_iter,
            start_iter,
            state,
            average_state,
            opt_state,
            total_weight=0.0,
        ):
            def step(carry, _):
                (
                    i,
                    state,
                    average_state,
                    opt_state,
                    total_weight,
                ) = carry

                opt_state, k = unpack_k(opt_state)

                if update_mode == "alternating":
                    gradient_start = grad(state)
                    if k_scaling:
                        gradient_start = scale_by_k(gradient_start, k)

                    primal_updates, _ = opt_update(
                        zero_dual(gradient_start),
                        opt_state,
                        state,
                    )
                    primal_updates = keep_only_primal(primal_updates)
                    state = optax.apply_updates(state, primal_updates)
                    state = SaddleState(
                        primal=projection_primal(state.primal),
                        dual_ineq=state.dual_ineq,
                        dual_eq=state.dual_eq,
                    )

                    gradient_after_primal = grad(state)
                    if k_scaling:
                        gradient_after_primal = scale_by_k(gradient_after_primal, k)
                    combined_gradient = SaddleState(
                        primal=gradient_start.primal,
                        dual_ineq=gradient_after_primal.dual_ineq,
                        dual_eq=gradient_after_primal.dual_eq,
                    )
                    combined_updates, opt_state = opt_update(
                        combined_gradient,
                        opt_state,
                        state,
                    )
                    dual_updates = keep_only_dual(combined_updates)
                    state = optax.apply_updates(state, dual_updates)
                    state = SaddleState(
                        primal=state.primal,
                        dual_ineq=projection_non_negative(state.dual_ineq),
                        dual_eq=state.dual_eq,
                    )
                elif extragradient:
                    # --- Look-ahead gradient ---
                    g = grad(state)

                    scaled_g = scale_by_k(g, k) if k_scaling else g
                    la_updates, _ = opt_update(scaled_g, opt_state, state)
                    state_half = optax.apply_updates(state, la_updates)
                    state_half = SaddleState(
                        primal=projection_primal(state_half.primal),
                        dual_ineq=projection_non_negative(state_half.dual_ineq),
                        dual_eq=state_half.dual_eq,
                    )

                    # Corrector: gradient at look-ahead point, applied from original state
                    g_half = grad(state_half)
                    scaled_g_half = scale_by_k(g_half, k) if k_scaling else g_half
                    corr_updates, opt_state = opt_update(
                        scaled_g_half, opt_state, state
                    )
                    state = optax.apply_updates(state, corr_updates)
                    state = SaddleState(
                        primal=projection_primal(state.primal),
                        dual_ineq=projection_non_negative(state.dual_ineq),
                        dual_eq=state.dual_eq,
                    )
                else:
                    gradient = grad(state)
                    if k_scaling:
                        gradient = scale_by_k(gradient, k)
                    updates, opt_state = opt_update(gradient, opt_state, state)
                    state = optax.apply_updates(state, updates)
                    state = SaddleState(
                        primal=projection_primal(state.primal),
                        dual_ineq=projection_non_negative(state.dual_ineq),
                        dual_eq=state.dual_eq,
                    )

                opt_state = pack_k(opt_state, k)

                w = weight_function(i)
                total_weight = jax.lax.cond(
                    average,
                    lambda: total_weight + w,
                    lambda: total_weight,
                )

                average_state = jax.lax.cond(
                    average,
                    lambda: optax.incremental_update(
                        state, average_state, w / total_weight
                    ),
                    lambda: average_state,
                )

                return (
                    i + 1,
                    state,
                    average_state,
                    opt_state,
                    total_weight,
                ), None

            (i, state, average_state, opt_state, total_weight), _ = jax.lax.scan(
                step,
                (start_iter, state, average_state, opt_state, total_weight),
                None,
                length=max_iter,
            )

            return i, state, average_state, opt_state, total_weight

        _CONVEX_RUN_EPOCH_CACHE[cache_key] = run_epoch

    state = initial_solution

    if initial_avg_state is not None:
        average_state = initial_avg_state
    else:
        average_state = initial_solution

    if initial_opt_state is not None:
        opt_state = initial_opt_state
    elif k_scaling:
        dtype = initial_solution.primal.dtype
        opt_state = (
            optimiser.init(initial_solution),
            jnp.asarray(k_init, dtype),
        )
    else:
        opt_state = optimiser.init(initial_solution)

    return run_epoch(
        max_iter,
        start_iter,
        state,
        average_state,
        opt_state,
        total_weight,
    )


def solve(
    cp: CP,
    optimiser=None,
    max_epochs=None,
    initial_solution=None,
    initial_opt_state=None,
    iterations_per_epoch=int(1e3),
    primal_grad_norm_tolerance=1e-2,
    primal_feasibility_tolerance=1e-3,
    complementarity_slack_tolerance=1e-3,
    weight_function=lambda _: 1.0,
    verbose=False,
    log_every=10,
    average=False,
    update_mode="alternating",
    output_opt_state=False,
    k_scaling=True,
    k_theta=0.5,
    k_lo=0.1,
    k_hi=10.0,
    k_init=None,
    restarts=0,
    epochs_per_restart=10,
    restart_multiplier=1.0,
    restart_decay=0.2,
    iterations_per_epoch_decay=1.0,
    iterations_per_epoch_min=100,
    precompile=True,
):
    """
    Solve a convex saddle-point problem via saddle-point optimisation.

    Args:
        update_mode: Selects the stepping scheme (single source of truth):
            ``"synchronous"`` (default), ``"alternating"``, or
            ``"extragradient"`` (Korpelevich two-call).
        k_scaling: Enable primal-weight (k) scaling (default ``False``). When
            ``True`` — orthogonal to ``update_mode``, so it composes with all
            three schemes — a primal weight ``k`` rescales the primal/dual
            gradients by ``(1/k, k)`` before each ``opt_update``, making the
            dual/primal step ratio ``k**2``. ``k`` is initialised from ``k_init``
            and rebalanced at each restart (PDLP-style) from primal-vs-dual
            iterate movement; it is constant within an epoch (not adapted per
            iteration). Tuned by ``k_theta``/``k_lo``/``k_hi`` and ``k_init``.
        k_theta: Smoothing coefficient for the log-space primal-weight update at
            each restart (default 0.5 = geometric mean of the movement-based
            target and the current weight, matching PDLP). Smaller = slower
            adaptation. Only used when ``k_scaling=True``.
        k_lo, k_hi: Clamp band for ``k`` (default [0.1, 10]). Also clamps the
            ``||c||/||b||`` init. Only used when ``k_scaling=True``.
        k_init: Initial primal weight ``k``. ``None`` (default) initialises it to
            the PDLP heuristic ``||c|| / ||b||`` (objective vs RHS norms), where
            ``c = grad(objective)(0)`` and ``b = -[c_eq(0); c_ineq(0)]``. Pass a
            float to override (``1.0`` = symmetric steps). Only used when
            ``k_scaling=True``.
        restarts: Maximum number of warm restarts (default 0 = disabled). Each
            restart resets the optimiser momentum and averaging while keeping the
            current iterate as a warm start. A restart fires when the normalised
            KKT merit drops below ``restart_decay`` × the merit at the last
            restart (sufficient-progress restart) or the cycle-length cap is
            exhausted (no-progress restart).
        epochs_per_restart: Length cap (epochs) of the first restart cycle
            (default 10). Subsequent caps grow by ``restart_multiplier``.
        restart_multiplier: Geometric growth factor for cycle-length caps
            (default 1.0 = fixed length, 2.0 = doubling).
        restart_decay: Sufficient-progress threshold (default 0.2). A restart
            fires early when the merit drops below this fraction of the merit at
            the last restart.
        iterations_per_epoch_decay: Multiplicative decay applied to
            ``iterations_per_epoch`` after each restart (default 1.0 = no
            decay). Values < 1 shrink the epoch length at each restart to spend
            more time checking convergence.
        iterations_per_epoch_min: Floor for the decayed epoch length (default
            100). Only used when ``iterations_per_epoch_decay < 1``.
    """

    if optimiser is None:
        optimiser = jo.optimisitic_gd(1 / 2)

    if log_every < 1:
        raise ValueError("log_every must be >= 1")

    if verbose:
        print("----------------------------------------------")

    valid_update_modes = [
        "synchronous",
        "alternating",
        "extragradient",
    ]
    if update_mode not in valid_update_modes:
        raise ValueError(f"update_mode must be one of {valid_update_modes}")

    if verbose:
        print("====Starting Solve====")
        print("----------------------------------------------")

    def projection_primal(primal_state):
        return projection_box(primal_state, cp.lower_bounds, cp.upper_bounds)

    def langrangian(state):
        return (
            cp.objective(state.primal)
            + state.dual_ineq @ cp.constraints_ineq(state.primal)
            + state.dual_eq @ cp.constraints_eq(state.primal)
        )

    def grad(state):
        gradient = jax.grad(langrangian)(state)
        return SaddleState(
            primal=gradient.primal,
            dual_ineq=gradient.dual_ineq,
            dual_eq=gradient.dual_eq,
        )

    has_dual_bound = cp.dual_bound is not None

    @jax.jit
    def compute_epoch_metrics(average_state):
        objective_value = cp.objective(average_state.primal)

        gradient = grad(average_state)

        grad_primal = gradient.primal
        grad_dual_ineq = gradient.dual_ineq
        grad_dual_eq = gradient.dual_eq

        # Unscale constraint violations to original space

        projected_primal = projection_primal(average_state.primal - grad_primal)
        projected_gradient_residual = average_state.primal - projected_primal
        primal_grad_norm = jnp.max(jnp.abs(projected_gradient_residual))

        ineq_violations = jnp.maximum(grad_dual_ineq, 0.0)
        max_ineq_violation = (
            jnp.max(ineq_violations) if ineq_violations.size > 0 else jnp.zeros(())
        )

        eq_violations = jnp.abs(grad_dual_eq)
        max_eq_violation = (
            jnp.max(eq_violations) if eq_violations.size > 0 else jnp.zeros(())
        )

        complementarity_slack = (
            jnp.max(jnp.abs(average_state.dual_ineq * grad_dual_ineq))
            if average_state.dual_ineq.size > 0
            else jnp.zeros(())
        ) / (1.0 + jnp.abs(objective_value))

        constraint_bound = jnp.maximum(max_ineq_violation, max_eq_violation)

        if has_dual_bound:
            dual_bound = cp.dual_bound(
                average_state.dual_ineq,
                average_state.dual_eq,
            )
            duality_gap = objective_value - dual_bound
        else:
            duality_gap = jnp.nan
        dual_gap_is_finite = has_dual_bound & jnp.isfinite(duality_gap)

        return (
            objective_value,
            primal_grad_norm,
            complementarity_slack,
            constraint_bound,
            duality_gap,
            dual_gap_is_finite,
        )

    def check_convergence(
        primal_grad_norm,
        complementarity_slack,
        constraint_bound,
    ):
        return (
            (primal_grad_norm > primal_grad_norm_tolerance)
            | (complementarity_slack > complementarity_slack_tolerance)
            | (constraint_bound > primal_feasibility_tolerance)
        )

    def check_max_epochs(count):
        return count >= max_epochs

    def cond_fun(loop_vars):
        (
            i,
            state,
            average_state,
            opt_state,
            previous_objective,
            primal_grad_norm,
            complementarity_slack,
            constraint_bound,
            count,
            objective_value,
            duality_gap,
            dual_gap_is_finite,
            total_weight,
        ) = loop_vars

        return check_convergence(
            primal_grad_norm, complementarity_slack, constraint_bound
        )

    def body_fun(loop_vars):
        (
            i,
            state,
            average_state,
            opt_state,
            previous_objective,
            primal_grad_norm,
            complementarity_slack,
            constraint_bound,
            count,
            objective_value,
            duality_gap,
            dual_gap_is_finite,
            total_weight,
        ) = loop_vars

        (
            i,
            state,
            average_state,
            opt_state,
            total_weight,
        ) = __sps(
            iterations_per_epoch,
            i,
            cp,
            optimiser,
            state,
            average_state,
            opt_state,
            weight_function,
            total_weight,
            average,
            update_mode,
            k_scaling,
            k_init,
        )

        (
            objective_value,
            primal_grad_norm,
            complementarity_slack,
            constraint_bound,
            duality_gap,
            dual_gap_is_finite,
        ) = compute_epoch_metrics(average_state if average else state)

        count += 1

        return (
            i,
            state,
            average_state,
            opt_state,
            previous_objective,
            primal_grad_norm,
            complementarity_slack,
            constraint_bound,
            count,
            objective_value,
            duality_gap,
            dual_gap_is_finite,
            total_weight,
        )

    i = 1

    if initial_solution is None:
        initial_solution = cp.initial_solution()

    # PDLP-style primal-weight initialisation. When k-scaling is on and k_init is
    # left as None we derive it from the objective/RHS norms ||c|| / ||b||, which
    # puts the primal/dual step ratio in the right order of magnitude before
    # iteration 1 instead of starting symmetric. For a linear objective
    # c = grad(obj)(0), and for constraints of the form Ax - b, evaluating at
    # x = 0 gives -b, so b = -[c_eq(0); c_ineq(0)]. This is generic over
    # CP/LP/JaddleLP since it only uses the objective/constraint callables.
    if k_scaling and k_init is None:
        zero = jnp.zeros_like(initial_solution.primal)
        c = jax.grad(cp.objective)(zero)
        b = jnp.concatenate([cp.constraints_eq(zero), cp.constraints_ineq(zero)])
        norm_c = jnp.linalg.norm(c) + 1e-30
        norm_b = jnp.linalg.norm(b) + 1e-30
        k_init = float(jnp.clip(norm_c / norm_b, k_lo, k_hi))
    elif k_init is None:
        k_init = 1.0

    is_converged = True
    state = initial_solution
    average_state = initial_solution
    if initial_opt_state is not None:
        opt_state = initial_opt_state
    elif k_scaling:
        dtype = initial_solution.primal.dtype
        opt_state = (
            optimiser.init(initial_solution),
            jnp.asarray(k_init, dtype),
        )
    else:
        opt_state = optimiser.init(initial_solution)
    primal_grad_norm = jnp.inf
    complementarity_slack = jnp.inf
    constraint_bound = jnp.inf
    objective_value = jnp.inf
    duality_gap = jnp.inf
    dual_gap_is_finite = False
    count = 0
    total_weight = 0.0
    current_iterations_per_epoch = iterations_per_epoch

    restarts_done = 0
    restart_i_offset = 0
    epochs_since_restart = 0
    current_cycle_cap = float(epochs_per_restart)
    merit_at_last_restart = jnp.inf
    # Iterate at the last restart (or the start), used to rebalance the primal
    # weight k from the primal-vs-dual movement over the restart cycle.
    state_at_last_restart = initial_solution

    def kkt_merit(primal_grad_norm, complementarity_slack, constraint_bound):
        # Normalised KKT merit for the restart trigger. Each term is divided by
        # (1 + its initial value) so all three are O(1) at the start. Since we
        # don't track initial values, just use 1.0 as the normaliser (the merit
        # is only compared with itself at successive restarts, so the scale
        # cancels). The maximum of the three terms drives the trigger.
        return jnp.maximum(
            jnp.maximum(primal_grad_norm, complementarity_slack),
            constraint_bound,
        )

    if precompile:
        # Warm both jitted functions the hot loop uses (the scan body via
        # __sps, and the end-of-epoch metrics) with the exact argument
        # signatures the timed loop will feed them, so epoch 1 doesn't pay
        # their first-call compile inside the timed measurement.
        #
        # NOTE: convex's run_epoch takes max_iter as a *static* argname (the
        # scan length is baked into the compile), so it must be warmed with the
        # real `current_iterations_per_epoch` value — warming with max_iter=1
        # would compile a throwaway length-1 executable and leave the real one
        # to compile cold in epoch 1. (Linear can use max_iter=1 because there
        # it's a traced while_loop bound, not a static scan length.)
        _precompile_result = __sps(
            current_iterations_per_epoch,
            i - restart_i_offset,
            cp,
            optimiser,
            state,
            average_state,
            opt_state,
            weight_function,
            total_weight,
            average,
            update_mode,
            k_scaling,
            k_init,
        )
        _precompile_metrics = compute_epoch_metrics(average_state if average else state)
        jax.block_until_ready((_precompile_result, _precompile_metrics))

    start_time = time.time()

    try:
        while check_convergence(
            primal_grad_norm,
            complementarity_slack,
            constraint_bound,
        ):
            if max_epochs:
                if check_max_epochs(count):
                    is_converged = False
                    print(f"Reached maximum epochs: {max_epochs}. Stopping.")
                    print("----------------------------------------------")
                    break

            start_epoch_time = time.time()
            (
                shifted_i,
                state,
                average_state,
                opt_state,
                total_weight,
            ) = __sps(
                current_iterations_per_epoch,
                i - restart_i_offset,
                cp,
                optimiser,
                state,
                average_state,
                opt_state,
                weight_function,
                total_weight,
                average,
                update_mode,
                k_scaling,
                k_init,
            )
            i = shifted_i + restart_i_offset

            (
                objective_value,
                primal_grad_norm,
                complementarity_slack,
                constraint_bound,
                duality_gap,
                dual_gap_is_finite,
            ) = compute_epoch_metrics(average_state if average else state)

            finish_epoch_time = time.time()
            count += 1

            if verbose and (count == 1 or count % log_every == 0):
                dual_gap_status = (
                    "finite"
                    if bool(dual_gap_is_finite)
                    else ("unavailable" if not has_dual_bound else "dual-infeasible")
                )
                print(
                    f"|Epoch {count}|"
                    f"|Obj{objective_value:.2e}|"
                    f"|PGN {primal_grad_norm:.2e}|"
                    f"|CS {complementarity_slack:.2e}|"
                    f"|PFR {constraint_bound:.2e}|"
                    f"|RDG {duality_gap/(1 + jnp.abs(objective_value)):.2e} ({dual_gap_status})|"
                    f"|Time {finish_epoch_time - start_epoch_time:.2f}s|"
                )
                print("----------------------------------------------")

                if k_scaling:
                    print(f"  primal weight k={float(opt_state[1]):.3f}")
                    print("----------------------------------------------")

            # --- Adaptive restart decision ---
            if restarts and restarts_done < restarts:
                epochs_since_restart += 1

                merit = kkt_merit(
                    primal_grad_norm, complementarity_slack, constraint_bound
                )

                # Two-point restart: pick the better of average and iterate.
                restart_point = average_state if average else state
                restart_merit = merit
                restart_used_avg = bool(average)
                if average:
                    (
                        _obj,
                        st_pgn,
                        st_cs,
                        st_cb,
                        _dg,
                        _dgf,
                    ) = compute_epoch_metrics(state)
                    state_merit = kkt_merit(st_pgn, st_cs, st_cb)
                    if bool(state_merit < merit):
                        restart_point = state
                        restart_merit = state_merit
                        restart_used_avg = False

                if not jnp.isfinite(merit_at_last_restart):
                    merit_at_last_restart = restart_merit

                sufficient_progress = bool(
                    restart_merit <= restart_decay * merit_at_last_restart
                )
                cycle_exhausted = epochs_since_restart >= current_cycle_cap

                if sufficient_progress or cycle_exhausted:
                    state = restart_point
                    if k_scaling:
                        # PDLP-style primal-weight rebalance: drive k from the
                        # primal-vs-dual *movement* over the just-finished cycle
                        # (distance between iterates), not per-step gradient
                        # norms. log-space geometric-mean blend with the current
                        # weight (k_theta), then clamp.
                        dp = state.primal - state_at_last_restart.primal
                        dd = jnp.concatenate(
                            [
                                state.dual_eq - state_at_last_restart.dual_eq,
                                state.dual_ineq - state_at_last_restart.dual_ineq,
                            ]
                        )
                        move_p = jnp.linalg.norm(dp) + 1e-30
                        move_d = jnp.linalg.norm(dd) + 1e-30
                        k_target = move_p / move_d
                        k_prev = opt_state[1]
                        log_k = k_theta * jnp.log(k_target) + (1.0 - k_theta) * jnp.log(
                            k_prev
                        )
                        k_new = jnp.clip(jnp.exp(log_k), k_lo, k_hi)
                        opt_state = (optimiser.init(state), k_new)
                    else:
                        opt_state = optimiser.init(state)
                    average_state = state
                    state_at_last_restart = state
                    total_weight = 0.0
                    restart_i_offset = i - 1
                    merit_at_last_restart = restart_merit
                    epochs_since_restart = 0
                    current_cycle_cap *= restart_multiplier
                    current_iterations_per_epoch = max(
                        iterations_per_epoch_min,
                        int(current_iterations_per_epoch * iterations_per_epoch_decay),
                    )
                    restarts_done += 1
                    if verbose:
                        reason = (
                            "sufficient-progress"
                            if sufficient_progress
                            else "cycle-cap"
                        )
                        which = "avg" if restart_used_avg else "iterate"
                        k_msg = f", k={float(opt_state[1]):.3e}" if k_scaling else ""
                        print(
                            f"Restart {restarts_done}/{restarts} at epoch {count} "
                            f"({reason}, merit={float(restart_merit):.2e} "
                            f"[{which}], next cap={current_cycle_cap:.0f} epochs, "
                            f"iters/epoch={current_iterations_per_epoch}{k_msg})"
                        )
                        print("----------------------------------------------")

        if average:
            output = average_state
        else:
            output = state
    except KeyboardInterrupt:
        is_converged = False
        if average:
            output = average_state
        else:
            output = state
        print("KeyboardInterrupt received. Returning current solution.")
        print("----------------------------------------------")

    output = jax.block_until_ready(output)
    end_time = time.time()
    print(f"Time to solution: {end_time - start_time:.2f} seconds")
    print("----------------------------------------------")
    print(f"Epochs to solution: {count}")
    print("----------------------------------------------")
    print(f"Objective: {cp.objective(output.primal):.5e}")
    print("----------------------------------------------")

    if output_opt_state:
        return output, is_converged, opt_state
    else:
        return output, is_converged


# %%
