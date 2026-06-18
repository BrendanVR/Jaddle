# %%
import jax
import jax.numpy as jnp
from optax.projections import projection_non_negative, projection_box
import optax
import functools
from typing import NamedTuple
import time
from jaddle.jaddle_basic_types import CP, SaddleState, HedgeSaddleState
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
    per_iterate_k=False,
    per_iterate_k_theta=0.1,
    per_iterate_k_lo=0.1,
    per_iterate_k_hi=10.0,
    extragradient=False,
    k_init=1.0,
):

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

    cache_key = (
        id(cp),
        id(optimiser),
        id(weight_function),
        bool(average),
        update_mode,
        bool(per_iterate_k),
        float(per_iterate_k_theta),
        float(per_iterate_k_lo),
        float(per_iterate_k_hi),
        bool(extragradient),
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

                if update_mode == "alternating":
                    gradient_start = grad(state)

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
                elif per_iterate_k:
                    opt_state, k = opt_state

                    g = grad(state)
                    updates, opt_state = opt_update(g, opt_state, state)

                    up = updates.primal
                    ud = jnp.concatenate([updates.dual_eq, updates.dual_ineq])
                    norm_p = jnp.sqrt(up @ up) + 1e-30
                    norm_d = jnp.sqrt(ud @ ud) + 1e-30
                    k_target = jnp.sqrt(norm_p / norm_d)
                    log_k = per_iterate_k_theta * jnp.log(k_target) + (
                        1.0 - per_iterate_k_theta
                    ) * jnp.log(k)
                    k = jnp.clip(jnp.exp(log_k), per_iterate_k_lo, per_iterate_k_hi)

                    updates = SaddleState(
                        primal=updates.primal / k,
                        dual_ineq=updates.dual_ineq * k,
                        dual_eq=updates.dual_eq * k,
                    )
                    state = optax.apply_updates(state, updates)
                    state = SaddleState(
                        primal=projection_primal(state.primal),
                        dual_ineq=projection_non_negative(state.dual_ineq),
                        dual_eq=state.dual_eq,
                    )
                    opt_state = (opt_state, k)
                elif extragradient:
                    opt_state, k = opt_state

                    # --- Look-ahead gradient ---
                    g = grad(state)

                    up = g.primal
                    ud = jnp.concatenate([g.dual_eq, g.dual_ineq])
                    norm_p = jnp.sqrt(up @ up) + 1e-30
                    norm_d = jnp.sqrt(ud @ ud) + 1e-30
                    k_target = jnp.sqrt(norm_p / norm_d)
                    log_k = per_iterate_k_theta * jnp.log(k_target) + (
                        1.0 - per_iterate_k_theta
                    ) * jnp.log(k)
                    k = jnp.clip(jnp.exp(log_k), per_iterate_k_lo, per_iterate_k_hi)

                    scaled_g = SaddleState(
                        primal=g.primal / k,
                        dual_ineq=g.dual_ineq * k,
                        dual_eq=g.dual_eq * k,
                    )
                    la_updates, _ = opt_update(scaled_g, opt_state, state)
                    state_half = optax.apply_updates(state, la_updates)
                    state_half = SaddleState(
                        primal=projection_primal(state_half.primal),
                        dual_ineq=projection_non_negative(state_half.dual_ineq),
                        dual_eq=state_half.dual_eq,
                    )

                    # Corrector: gradient at look-ahead point, applied from original state
                    g_half = grad(state_half)
                    scaled_g_half = SaddleState(
                        primal=g_half.primal / k,
                        dual_ineq=g_half.dual_ineq * k,
                        dual_eq=g_half.dual_eq * k,
                    )
                    corr_updates, opt_state = opt_update(scaled_g_half, opt_state, state)
                    state = optax.apply_updates(state, corr_updates)
                    state = SaddleState(
                        primal=projection_primal(state.primal),
                        dual_ineq=projection_non_negative(state.dual_ineq),
                        dual_eq=state.dual_eq,
                    )
                    opt_state = (opt_state, k)
                else:
                    gradient = grad(state)
                    updates, opt_state = opt_update(gradient, opt_state, state)
                    state = optax.apply_updates(state, updates)
                    state = SaddleState(
                        primal=projection_primal(state.primal),
                        dual_ineq=projection_non_negative(state.dual_ineq),
                        dual_eq=state.dual_eq,
                    )

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
    elif per_iterate_k or extragradient:
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
    optimiser,
    max_epochs=None,
    initial_solution=None,
    initial_opt_state=None,
    iterations_per_epoch=int(1e4),
    progress_tolerance=1e-2,
    constraint_tolerance=1e-3,
    complementarity_tolerance=1e-3,
    weight_function=lambda _: 1.0,
    verbose=False,
    log_every=10,
    average=True,
    update_mode="synchronous",
    expert_diagnostics=False,
    output_opt_state=False,
    per_iterate_k=False,
    per_iterate_k_theta=0.1,
    per_iterate_k_lo=0.1,
    per_iterate_k_hi=10.0,
    extragradient=False,
    k_init=1.0,
):
    """
    Solve a convex saddle-point problem via saddle-point optimisation.

    Args:
        per_iterate_k: Adapt the primal/dual step ratio ``k`` every iteration
            inside the scan, matvec-free. The ratio target is
            ``sqrt(||u_primal|| / ||u_dual||)``; log-space smoothed by
            ``per_iterate_k_theta`` and clamped to ``[per_iterate_k_lo,
            per_iterate_k_hi]``. Mutually exclusive with ``extragradient``.
        per_iterate_k_theta: Smoothing coefficient for the log-space k update
            (default 0.1). Smaller = slower adaptation.
        per_iterate_k_lo, per_iterate_k_hi: Clamp band for k (default [0.1, 10]).
        extragradient: Korpelevich extragradient (two-call) with per-iterate k
            adaptation from look-ahead gradient norms. Mutually exclusive with
            ``per_iterate_k``.
        k_init: Initial value of ``k`` (default 1.0 = symmetric steps).
    """

    if log_every < 1:
        raise ValueError("log_every must be >= 1")

    if verbose:
        print("----------------------------------------------")

    if update_mode not in ["synchronous", "alternating"]:
        raise ValueError("update_mode must be one of ['synchronous', 'alternating']")
    if per_iterate_k and update_mode != "synchronous":
        raise ValueError(
            "per_iterate_k is only implemented for update_mode='synchronous'."
        )
    if extragradient and update_mode != "synchronous":
        raise ValueError(
            "extragradient is only implemented for update_mode='synchronous'."
        )
    if per_iterate_k and extragradient:
        raise ValueError(
            "per_iterate_k and extragradient both control per-iterate stepping; "
            "use extragradient=True to get the look-ahead/corrector step with "
            "built-in k adaptation."
        )
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
        max_ineq_violation = jnp.max(ineq_violations) if ineq_violations.size > 0 else jnp.zeros(())

        eq_violations = jnp.abs(grad_dual_eq)
        max_eq_violation = jnp.max(eq_violations) if eq_violations.size > 0 else jnp.zeros(())

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
            (primal_grad_norm > progress_tolerance)
            | (complementarity_slack > complementarity_tolerance)
            | (constraint_bound > constraint_tolerance)
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
            per_iterate_k,
            per_iterate_k_theta,
            per_iterate_k_lo,
            per_iterate_k_hi,
            extragradient,
            k_init,
        )

        (
            objective_value,
            primal_grad_norm,
            complementarity_slack,
            constraint_bound,
            duality_gap,
            dual_gap_is_finite,
        ) = jax.lax.cond(
            average,
            lambda: compute_epoch_metrics(average_state),
            lambda: compute_epoch_metrics(state),
        )

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

    is_converged = True
    state = initial_solution
    average_state = initial_solution
    if initial_opt_state is not None:
        opt_state = initial_opt_state
    elif per_iterate_k or extragradient:
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

    start_time = time.time()
    missing_expert_state_warning_printed = False

    def print_expert_weights(epoch_count, state_for_weights):
        nonlocal missing_expert_state_warning_printed
        if not expert_diagnostics:
            return

        extracted = jo.hedge_diagnostics_from_state(state_for_weights)
        if extracted is None:
            extracted = jo.hedge_weights_from_state(state_for_weights)
        if extracted is None:
            if (not missing_expert_state_warning_printed) and verbose:
                print(
                    "Expert diagnostics requested, but optimiser state has no hedge weights."
                )
                print("----------------------------------------------")
                missing_expert_state_warning_printed = True
            return

        if isinstance(extracted, dict):
            weights = extracted["weights"]
            losses = extracted["clipped_losses"]
            centered_losses = extracted["centered_losses"]
            hedge_eta = extracted["eta"]
        else:
            # hedge_weights_from_state fallback: bare weight vector.
            weights = extracted
            losses = centered_losses = hedge_eta = None

        if verbose:
            print(
                f"Player Weights (epoch {epoch_count}): {np.asarray(weights)}"
            )
            if losses is not None:
                print(
                    f"Player Losses (epoch {epoch_count}): {np.asarray(losses)}"
                )
                print(
                    f"Centered Losses (epoch {epoch_count}): "
                    f"{np.asarray(centered_losses)}"
                )
            if hedge_eta is not None:
                print(
                    f"Hedge Eta (epoch {epoch_count}): {float(hedge_eta):.3e}"
                )
            print("----------------------------------------------")

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
                per_iterate_k,
                per_iterate_k_theta,
                per_iterate_k_lo,
                per_iterate_k_hi,
                extragradient,
                k_init,
            )

            (
                objective_value,
                primal_grad_norm,
                complementarity_slack,
                constraint_bound,
                duality_gap,
                dual_gap_is_finite,
            ) = jax.lax.cond(
                average,
                lambda: compute_epoch_metrics(average_state),
                lambda: compute_epoch_metrics(state),
            )

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
                    f"|CB {constraint_bound:.2e}|"
                    f"|DG {duality_gap:.2e} ({dual_gap_status})|"
                    f"|Time {finish_epoch_time - start_epoch_time:.2f}s|"
                )
                print("----------------------------------------------")

                print_expert_weights(count, opt_state)

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
