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
            else:
                gradient = grad(state)
                updates, opt_state = opt_update(gradient, opt_state, state)
                state = optax.apply_updates(state, updates)
                state = SaddleState(
                    primal=projection_primal(state.primal),
                    dual_ineq=projection_non_negative(state.dual_ineq),
                    dual_eq=state.dual_eq,
                )

            total_weight = jax.lax.cond(
                average,
                lambda: total_weight + weight_function(i),
                lambda: total_weight,
            )

            average_state = jax.lax.cond(
                average,
                lambda: optax.incremental_update(
                    state, average_state, weight_function(i) / total_weight
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

    state = initial_solution

    if initial_avg_state is not None:
        average_state = initial_avg_state
    else:
        average_state = initial_solution

    if initial_opt_state is not None:
        opt_state = initial_opt_state
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
    average=True,
    update_mode="synchronous",
    expert_diagnostics=False,
    output_opt_state=False,
):
    """
    Solve a linear program via saddle-point optimisation.

    Args:
        restarts: Number of warm restarts. 0 = no restarts (default).
            Each restart resets the optimizer state (momentum) and averaging
            while keeping the current iterate as a warm start. The LR schedule
            also restarts from its initial value.
        epochs_per_restart: Number of epochs in the first restart cycle
            (default 10). Subsequent cycles grow by restart_multiplier.
        restart_multiplier: Geometric growth factor for cycle lengths
            (default 1.0 = fixed length, 2.0 = doubling).
        primal_dual_lr_ratio: Ratio of primal to dual learning rates (default 1.0).
    """

    print("----------------------------------------------")

    if update_mode not in ["synchronous", "alternating"]:
        raise ValueError("update_mode must be one of ['synchronous', 'alternating']")
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
        max_ineq_violation = jnp.max(ineq_violations)

        eq_violations = jnp.abs(grad_dual_eq)
        max_eq_violation = jnp.max(eq_violations)

        complementarity_slack = jnp.max(
            jnp.abs(average_state.dual_ineq * grad_dual_ineq)
        ) / (1.0 + jnp.abs(objective_value))

        constraint_bound = jnp.maximum(max_ineq_violation, max_eq_violation)

        return (
            objective_value,
            primal_grad_norm,
            complementarity_slack,
            constraint_bound,
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
            total_weight,
        ) = loop_vars

        return check_convergence(
            primal_grad_norm, complementarity_slack, constraint_bound, count
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
        )

        (
            objective_value,
            primal_grad_norm,
            complementarity_slack,
            constraint_bound,
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
            total_weight,
        )

    i = 1

    if initial_solution is None:
        initial_solution = cp.initial_solution()

    is_converged = True
    state = initial_solution
    average_state = initial_solution
    opt_state = (
        initial_opt_state
        if initial_opt_state is not None
        else optimiser.init(initial_solution)
    )
    primal_grad_norm = jnp.inf
    complementarity_slack = jnp.inf
    constraint_bound = jnp.inf
    objective_value = jnp.inf
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

        if isinstance(extracted, tuple):
            primal_weights, dual_weights = extracted
            primal_losses = dual_losses = None
            primal_eta = dual_eta = None
        else:
            primal_weights = extracted["primal_weights"]
            dual_weights = extracted["dual_weights"]
            primal_losses = extracted["primal_clipped_losses"]
            dual_losses = extracted["dual_clipped_losses"]
            primal_centered_losses = extracted["primal_centered_losses"]
            dual_centered_losses = extracted["dual_centered_losses"]
            primal_eta = extracted["primal_eta"]
            dual_eta = extracted["dual_eta"]

        if verbose:
            print(
                f"Expert Weights (epoch {epoch_count}): "
                f"primal={np.asarray(primal_weights)}, dual={np.asarray(dual_weights)}"
            )
            if primal_losses is not None and dual_losses is not None:
                print(
                    f"Expert Losses (epoch {epoch_count}): "
                    f"primal={np.asarray(primal_losses)}, dual={np.asarray(dual_losses)}"
                )
                print(
                    f"Centered Losses (epoch {epoch_count}): "
                    f"primal={np.asarray(primal_centered_losses)}, "
                    f"dual={np.asarray(dual_centered_losses)}"
                )
            if primal_eta is not None and dual_eta is not None:
                print(
                    f"Hedge Etas (epoch {epoch_count}): "
                    f"primal={float(primal_eta):.3e}, dual={float(dual_eta):.3e}"
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
            )

            (
                objective_value,
                primal_grad_norm,
                complementarity_slack,
                constraint_bound,
            ) = jax.lax.cond(
                average,
                lambda: compute_epoch_metrics(average_state),
                lambda: compute_epoch_metrics(state),
            )

            finish_epoch_time = time.time()
            count += 1

            print(
                f"|Epoch {count}|"
                f"|Obj{objective_value:.2e}|"
                f"|PGN {primal_grad_norm:.2e}|"
                f"|CS {complementarity_slack:.2e}|"
                f"|CB {constraint_bound:.2e}|"
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
