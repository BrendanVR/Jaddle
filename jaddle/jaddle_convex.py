# %%
import jax
import jax.numpy as jnp
from optax.projections import projection_non_negative, projection_box
import optax
import functools
from typing import NamedTuple
import time


class SaddleState(NamedTuple):
    primal: jnp.ndarray
    dual_ineq: jnp.ndarray
    dual_eq: jnp.ndarray


# %%
# Basic Types
class CP:
    def __init__(
        self,
        num_variables,
        objective,
        constraints_eq,
        constraints_ineq,
        lower_bounds,
        upper_bounds,
    ):
        self.num_variables = num_variables
        self.objective = objective
        self.constraints_eq = constraints_eq
        self.constraints_ineq = constraints_ineq
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def initial_primal_solution(self):
        return jnp.zeros(self.num_variables)

    def num_eq_constraints(self):
        return len(self.constraints_eq(self.initial_primal_solution()))

    def num_ineq_constraints(self):
        return len(self.constraints_ineq(self.initial_primal_solution()))

    def num_constraints(self):
        return self.num_eq_constraints() + self.num_ineq_constraints()

    def ineq_slack(self, x):
        return jnp.max(jnp.maximum(self.constraints_ineq(x), 0.0))

    def eq_slack(self, x):
        return jnp.max(jnp.abs(self.constraints_eq(x)))

    def complementarity_slack(self, x, dual_ineq):
        return dual_ineq * (self.constraints_ineq(x))

    def initial_solution(self):
        return SaddleState(
            primal=jnp.zeros(self.num_variables),
            dual_ineq=jnp.zeros(self.num_ineq_constraints()),
            dual_eq=jnp.zeros(self.num_eq_constraints()),
        )


# %%
# Solvers for constrained linear optimisation via saddle point formulation
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

            total_weight += weight_function(i)

            average_state = optax.incremental_update(
                state, average_state, weight_function(i) / total_weight
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
    optimiser_builder=None,
    lr_controller=None,
    iterations_per_epoch_controller=None,
    lr_state=None,
    reset_opt_state_on_lr_change=False,
    return_diagnostics=False,
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

    if optimiser is not None and optimiser_builder is not None:
        raise ValueError("Cannot specify both optimiser and optimiser_builder")

    if optimiser is None and optimiser_builder is None:
        raise ValueError("Must specify either optimiser or optimiser_builder")

    if initial_solution is not None:
        initial_solution = initial_solution

    if initial_solution is None:
        initial_solution = cp.initial_solution()

    if lr_controller is not None and optimiser_builder is None:
        raise ValueError("optimiser_builder must be provided when lr_controller is set")

    if lr_controller is not None:
        if lr_state is None:
            lr_state = {}
        optimiser = optimiser_builder(lr_state)

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

        return (
            (primal_grad_norm > progress_tolerance)
            | (complementarity_slack > complementarity_tolerance)
            | (constraint_bound > constraint_tolerance)
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
        ) = compute_epoch_metrics(average_state)
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
    dual_ineq_norm = jnp.inf
    dual_eq_norm = jnp.inf
    diagnostics = {
        "objective_value": objective_value,
        "primal_grad_norm": primal_grad_norm,
        "complementarity_slack": complementarity_slack,
        "constraint_bound": constraint_bound,
        "dual_ineq_norm": dual_ineq_norm,
        "dual_eq_norm": dual_eq_norm,
        "count": count,
        "iterations_per_epoch": iterations_per_epoch,
    }
    current_iterations_per_epoch = iterations_per_epoch

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
    if (
        lr_controller is None
        and iterations_per_epoch_controller is None
        and verbose == False
    ):

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

        dual_ineq_norm = jnp.linalg.norm(average_state.dual_ineq)
        dual_eq_norm = jnp.linalg.norm(average_state.dual_eq)

        end_time = time.time()
        print(f"Time to solution: {end_time - start_time:.2f} seconds")
        if return_diagnostics:
            diagnostics = {
                "objective_value": objective_value,
                "primal_grad_norm": primal_grad_norm,
                "complementarity_slack": complementarity_slack,
                "constraint_bound": constraint_bound,
                "dual_ineq_norm": dual_ineq_norm,
                "dual_eq_norm": dual_eq_norm,
                "count": count,
                "iterations_per_epoch": current_iterations_per_epoch,
            }
            return average_state, diagnostics
        return average_state

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
                current_iterations_per_epoch,
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
            ) = compute_epoch_metrics(average_state)
            count += 1

            dual_ineq_norm = jnp.linalg.norm(average_state.dual_ineq)
            dual_eq_norm = jnp.linalg.norm(average_state.dual_eq)

            diagnostics = {
                "objective_value": objective_value,
                "primal_grad_norm": primal_grad_norm,
                "complementarity_slack": complementarity_slack,
                "constraint_bound": constraint_bound,
                "dual_ineq_norm": dual_ineq_norm,
                "dual_eq_norm": dual_eq_norm,
                "count": count,
                "iterations_per_epoch": current_iterations_per_epoch,
            }

            if lr_controller is not None:
                controller_output = lr_controller(diagnostics, count, lr_state)
                controller_reset_opt_state = reset_opt_state_on_lr_change
                if controller_output is not None:
                    if (
                        isinstance(controller_output, tuple)
                        and len(controller_output) == 2
                    ):
                        lr_state, controller_reset_opt_state = controller_output
                    elif isinstance(controller_output, dict):
                        lr_state = controller_output.get("lr_state", lr_state)
                        controller_reset_opt_state = controller_output.get(
                            "reset_opt_state", controller_reset_opt_state
                        )
                    else:
                        lr_state = controller_output

                optimiser = optimiser_builder(lr_state)
                if controller_reset_opt_state:
                    opt_state = optimiser.init(state)

            if iterations_per_epoch_controller is not None:
                next_iterations_per_epoch = iterations_per_epoch_controller(
                    diagnostics,
                    count,
                    current_iterations_per_epoch,
                )
                if next_iterations_per_epoch is not None:
                    current_iterations_per_epoch = max(
                        int(next_iterations_per_epoch), 1
                    )

            if verbose:
                print(
                    f"Objective {objective_value:.2e}: Primal Grad Norm={primal_grad_norm:.2e}, Compl. Slack={complementarity_slack:.2e}, Constraint Bound={constraint_bound:.2e}"
                )

        end_time = time.time()
        print(f"Time to solution: {end_time - start_time:.2f} seconds")
        if return_diagnostics:
            if lr_controller is not None:
                return average_state, diagnostics, lr_state
            return average_state, diagnostics
        return average_state


# %%
