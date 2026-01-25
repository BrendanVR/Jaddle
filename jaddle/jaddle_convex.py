# %%
import jax
import jax.numpy as jnp
from optax.projections import projection_non_negative, projection_box
import optax
import functools
from typing import NamedTuple


class PrimalState(NamedTuple):
    primal: jnp.ndarray


class DualState(NamedTuple):
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

    def primal_initial_solution(self):
        return PrimalState(
            primal=jnp.zeros(self.num_variables),
        )

    def dual_initial_solution(self):
        return DualState(
            dual_ineq=jnp.zeros(self.num_ineq_constraints()),
            dual_eq=jnp.zeros(self.num_eq_constraints()),
        )


# %%
# Solvers for constrained convex optimisation via saddle point formulation
def __sps(
    max_iter,
    start_iter,
    cp: CP,
    optimiser_primal,
    optimiser_dual,
    primal_initial_solution,
    dual_initial_solution,
    primal_initial_avg_state=None,
    dual_initial_avg_state=None,
    primal_initial_opt_state=None,
    dual_initial_opt_state=None,
    exponential_weighting=0.01,
):

    primal_projection = lambda x: projection_box(x, cp.lower_bounds, cp.upper_bounds)

    @jax.jit
    def lagrangian(primal_state, dual_state):
        return (
            cp.objective(primal_state.primal)
            + dual_state.dual_eq @ (cp.constraints_eq(primal_state.primal))
            + dual_state.dual_ineq @ (cp.constraints_ineq(primal_state.primal))
        )

    @jax.jit
    def grad_primal(primal_state, dual_state):
        return jax.grad(lagrangian, argnums=0)(primal_state, dual_state)

    @jax.jit
    def grad_dual(primal_state, dual_state):
        grad = jax.grad(lagrangian, argnums=1)(primal_state, dual_state)
        return DualState(
            dual_ineq=-grad.dual_ineq,
            dual_eq=-grad.dual_eq,
        )

    @jax.jit
    def primal_opt_update(gradient, opt_state, state):
        return optimiser_primal.update(gradient, opt_state, state)

    @jax.jit
    def dual_opt_update(gradient, opt_state, state):
        return optimiser_dual.update(gradient, opt_state, state)

    @functools.partial(jax.jit, static_argnames=("max_iter", "exponential_weighting"))
    def run_epoch(
        max_iter,
        exponential_weighting,
        start_iter,
        primal_state,
        dual_state,
        primal_average_state,
        dual_average_state,
        primal_opt_state,
        dual_opt_state,
    ):
        def step(carry, _):
            (
                i,
                primal_state,
                dual_state,
                primal_average_state,
                dual_average_state,
                primal_opt_state,
                dual_opt_state,
            ) = carry

            gradient_primal = grad_primal(primal_state, dual_state)
            updates, primal_opt_state = primal_opt_update(
                gradient_primal, primal_opt_state, primal_state
            )
            primal_state = optax.apply_updates(primal_state, updates)
            primal_state = PrimalState(primal=primal_projection(primal_state.primal))

            primal_average_state = optax.incremental_update(
                primal_state, primal_average_state, exponential_weighting
            )

            gradient_dual = grad_dual(primal_state, dual_state)
            updates, dual_opt_state = dual_opt_update(
                gradient_dual, dual_opt_state, dual_state
            )
            dual_state = optax.apply_updates(dual_state, updates)
            dual_state = DualState(
                dual_ineq=projection_non_negative(dual_state.dual_ineq),
                dual_eq=dual_state.dual_eq,
            )

            dual_average_state = optax.incremental_update(
                dual_state, dual_average_state, exponential_weighting
            )

            return (
                i + 1,
                primal_state,
                dual_state,
                primal_average_state,
                dual_average_state,
                primal_opt_state,
                dual_opt_state,
            ), None

        (
            i,
            primal_state,
            dual_state,
            primal_average_state,
            dual_average_state,
            primal_opt_state,
            dual_opt_state,
        ), _ = jax.lax.scan(
            step,
            (
                start_iter,
                primal_state,
                dual_state,
                primal_average_state,
                dual_average_state,
                primal_opt_state,
                dual_opt_state,
            ),
            None,
            length=max_iter,
        )

        return (
            i,
            primal_state,
            dual_state,
            primal_average_state,
            dual_average_state,
            primal_opt_state,
            dual_opt_state,
        )

    primal_state = primal_initial_solution
    dual_state = dual_initial_solution

    if primal_initial_avg_state is not None:
        primal_average_state = primal_initial_avg_state
    else:
        primal_average_state = primal_initial_solution

    if dual_initial_avg_state is not None:
        dual_average_state = dual_initial_avg_state
    else:
        dual_average_state = dual_initial_solution

    if primal_initial_opt_state is not None:
        primal_opt_state = primal_initial_opt_state
    else:
        primal_opt_state = optimiser_primal.init(primal_initial_solution)

    if dual_initial_opt_state is not None:
        dual_opt_state = dual_initial_opt_state
    else:
        dual_opt_state = optimiser_dual.init(dual_initial_solution)

    return run_epoch(
        max_iter,
        exponential_weighting,
        start_iter,
        primal_state,
        dual_state,
        primal_average_state,
        dual_average_state,
        primal_opt_state,
        dual_opt_state,
    )


def solve(
    cp: CP,
    primal_initial_solution=None,
    dual_initial_solution=None,
    primal_optimiser=None,
    dual_optimiser=None,
    iterations_per_epoch=int(1e3),
    constraint_tolerance=1e-5,
    progress_tolerance=1e-5,
    complementarity_tolerance=1e-5,
    exponential_weighting=0.01,
    max_epochs=1000,
):

    if primal_initial_solution is None:
        primal_initial_solution = cp.primal_initial_solution()

    if dual_initial_solution is None:
        dual_initial_solution = cp.dual_initial_solution()

    if primal_optimiser is None:
        lr = optax.cosine_decay_schedule(
            init_value=1e0,
            decay_steps=int(1e4),
            exponent=1.5,
            alpha=1e-4,
        )
        primal_optimiser = optax.optimistic_adam_v2(
            learning_rate=lr,
            alpha=0.1,
            nesterov=True,
        )

    if dual_optimiser is None:
        lr = optax.cosine_decay_schedule(
            init_value=1e0,
            decay_steps=int(1e4),
            exponent=1.5,
            alpha=1e-4,
        )
        dual_optimiser = optax.optimistic_adam_v2(
            learning_rate=lr,
            alpha=0.1,
            nesterov=True,
        )

    i = 1
    primal_state = primal_initial_solution
    dual_state = dual_initial_solution
    primal_average_state = primal_initial_solution
    dual_average_state = dual_initial_solution
    primal_opt_state = primal_optimiser.init(primal_initial_solution)
    dual_opt_state = dual_optimiser.init(dual_initial_solution)
    progress = jnp.inf
    max_complementarity_slack = jnp.inf
    constraints_satisfied = False
    count = 0

    def cond_fun(loop_vars):
        (
            i,
            primal_state,
            dual_state,
            primal_average_state,
            dual_average_state,
            primal_opt_state,
            dual_opt_state,
            progress,
            max_complementarity_slack,
            constraints_satisfied,
            count,
        ) = loop_vars

        return (
            (progress > progress_tolerance)
            | (max_complementarity_slack > complementarity_tolerance)
            | (~constraints_satisfied)
        ) & (count < max_epochs)

    def body_fun(loop_vars):
        (
            i,
            primal_state,
            dual_state,
            primal_average_state,
            dual_average_state,
            primal_opt_state,
            dual_opt_state,
            progress,
            max_complementarity_slack,
            constraints_satisfied,
            count,
        ) = loop_vars

        (
            i,
            primal_state,
            dual_state,
            primal_average_state,
            dual_average_state,
            primal_opt_state,
            dual_opt_state,
        ) = __sps(
            iterations_per_epoch,
            i,
            cp,
            primal_optimiser,
            dual_optimiser,
            primal_state,
            dual_state,
            primal_average_state,
            dual_average_state,
            primal_opt_state,
            dual_opt_state,
            exponential_weighting,
        )

        objective_value = cp.objective(primal_average_state.primal)

        progress = jnp.abs(
            cp.objective(primal_average_state.primal) - objective_value
        ) / (1.0 + jnp.abs(objective_value))

        ineq_violations = jnp.maximum(cp.ineq_slack(primal_average_state.primal), 0.0)
        max_ineq_violation = jnp.max(ineq_violations)

        eq_violations = jnp.abs(cp.eq_slack(primal_average_state.primal))
        max_eq_violation = jnp.max(eq_violations)

        complentariy_slack = cp.complementarity_slack(
            primal_average_state.primal, dual_average_state.dual_ineq
        )
        max_complementarity_slack = jnp.abs(jnp.sum(complentariy_slack))

        constraints_satisfied = (max_ineq_violation < constraint_tolerance) & (
            max_eq_violation < constraint_tolerance
        )
        count += 1

        return (
            i,
            primal_state,
            dual_state,
            primal_average_state,
            dual_average_state,
            primal_opt_state,
            dual_opt_state,
            progress,
            max_complementarity_slack,
            constraints_satisfied,
            count,
        )

    # Initialize loop variables
    loop_vars = (
        i,
        primal_state,
        dual_state,
        primal_average_state,
        dual_average_state,
        primal_opt_state,
        dual_opt_state,
        progress,
        max_complementarity_slack,
        constraints_satisfied,
        count,
    )

    # Run the while loop
    loop_vars = jax.lax.while_loop(cond_fun, body_fun, loop_vars)

    (
        i,
        primal_state,
        dual_state,
        primal_average_state,
        dual_average_state,
        primal_opt_state,
        dual_opt_state,
        progress,
        max_complementarity_slack,
        constraints_satisfied,
        count,
    ) = loop_vars

    return primal_average_state, dual_average_state
