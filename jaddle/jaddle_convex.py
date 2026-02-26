# %%
import jax
import jax.numpy as jnp
from optax.projections import projection_non_negative, projection_box
import optax
import functools
from typing import NamedTuple
import time
from jaddle.jaddle_basic_types import CP, SaddleState


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
    primal_damping=0.0,
    dual_damping_ineq=0.0,
    dual_damping_eq=0.0,
    average=True,
):

    @jax.jit
    def projection_primal(primal_state):
        return projection_box(primal_state, cp.lower_bounds, cp.upper_bounds)

    @jax.jit
    def langrangian(state):
        return (
            cp.objective(state.primal)
            + state.dual_ineq @ cp.constraints_ineq(state.primal)
            + state.dual_eq @ cp.constraints_eq(state.primal)
        )

    @jax.jit
    def grad(state):
        gradient = jax.grad(langrangian)(state)
        return SaddleState(
            primal=gradient.primal + primal_damping * state.primal,
            dual_ineq=-gradient.dual_ineq + dual_damping_ineq * state.dual_ineq,
            dual_eq=-gradient.dual_eq + dual_damping_eq * state.dual_eq,
        )

    @jax.jit
    def opt_update(gradient, opt_state, state):
        return optimiser.update(gradient, opt_state, state)

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
        @jax.jit
        def step(carry, _):
            (
                i,
                state,
                average_state,
                opt_state,
                total_weight,
            ) = carry

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
    optimiser=None,
    max_epochs=100,
    initial_solution=None,
    iterations_per_epoch=int(1e4),
    dual_damping_ineq=0.0,
    dual_damping_eq=0.0,
    primal_damping=0.0,
    progress_tolerance=1e-2,
    constraint_tolerance=1e-4,
    complementarity_tolerance=1e-4,
    weight_function=lambda _: 1.0,
    verbose=False,
    average=True,
):

    @jax.jit
    def projection_primal(primal_state):
        return projection_box(primal_state, cp.lower_bounds, cp.upper_bounds)

    @jax.jit
    def langrangian(state):
        return (
            cp.objective(state.primal)
            + state.dual_ineq @ cp.constraints_ineq(state.primal)
            + state.dual_eq @ cp.constraints_eq(state.primal)
        )

    @jax.jit
    def grad(state):
        gradient = jax.grad(langrangian)(state)
        return SaddleState(
            primal=gradient.primal + primal_damping * state.primal,
            dual_ineq=gradient.dual_ineq + dual_damping_ineq * state.dual_ineq,
            dual_eq=gradient.dual_eq + dual_damping_eq * state.dual_eq,
        )

    @jax.jit
    def compute_epoch_metrics(average_state):
        objective_value = cp.objective(average_state.primal)

        gradient = grad(average_state)
        grad_primal = gradient.primal
        grad_dual_ineq = gradient.dual_ineq
        grad_dual_eq = gradient.dual_eq

        projected_primal = projection_primal(average_state.primal - grad_primal)
        projected_gradient_residual = average_state.primal - projected_primal
        primal_grad_norm = jnp.max(jnp.abs(projected_gradient_residual))

        ineq_violations = jnp.maximum(grad_dual_ineq, 0.0)
        max_ineq_violation = jnp.max(ineq_violations)

        eq_violations = jnp.abs(grad_dual_eq)
        max_eq_violation = jnp.max(eq_violations)

        complementarity_slack = jnp.max(
            jnp.abs(average_state.dual_ineq * grad_dual_ineq)
        )

        constraint_bound = jnp.maximum(max_ineq_violation, max_eq_violation)

        return (
            objective_value,
            primal_grad_norm,
            complementarity_slack,
            constraint_bound,
        )

    def check_convergence(
        primal_grad_norm, complementarity_slack, constraint_bound, count
    ):
        return (
            (primal_grad_norm > progress_tolerance)
            | (complementarity_slack > complementarity_tolerance)
            | (constraint_bound > constraint_tolerance)
        ) & (count < max_epochs)

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
        ) & (count < max_epochs)

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
            primal_damping,
            dual_damping_ineq,
            dual_damping_eq,
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

    if initial_solution is None:
        initial_solution = cp.initial_solution()

    i = 1
    state = initial_solution
    average_state = initial_solution
    opt_state = optimiser.init(initial_solution)
    previous_objective = jnp.inf
    primal_grad_norm = jnp.inf
    complementarity_slack = jnp.inf
    constraint_bound = jnp.inf
    objective_value = jnp.inf
    count = 0
    total_weight = 0.0

    # Initialize loop variables
    loop_vars = (
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

    start_time = time.time()
    # Run the while loop
    if verbose == False:

        loop_vars = jax.lax.while_loop(cond_fun, body_fun, loop_vars)

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

        end_time = time.time()
        print(f"Time to solution: {end_time - start_time:.2f} seconds")

        if average:
            return average_state
        else:
            return state

    else:
        while (
            (primal_grad_norm > progress_tolerance)
            | (complementarity_slack > complementarity_tolerance)
            | (constraint_bound > constraint_tolerance)
        ) & (count < max_epochs):
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
                primal_damping,
                dual_damping_ineq,
                dual_damping_eq,
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

            print(
                f"Objective {objective_value:.2e}: Primal Grad Norm={primal_grad_norm:.2e}, Compl. Slack={complementarity_slack:.2e}, Constraint Bound={constraint_bound:.2e}"
            )

        end_time = time.time()
        print(f"Time to solution: {end_time - start_time:.2f} seconds")
        if average:
            return average_state
        else:
            return state


# %%
