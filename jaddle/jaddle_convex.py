# %%
import jax
import jax.numpy as jnp
from optax.projections import projection_non_negative, projection_box
import optax
import jaddle.jaddle_optimisers as jo
import time


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
        return (dual_ineq * (self.constraints_ineq(x))).sum()

    def initial_solution(self):
        return {
            "primal": jnp.zeros(self.num_variables),
            "dual_ineq": jnp.zeros(self.num_ineq_constraints()),
            "dual_eq": jnp.zeros(self.num_eq_constraints()),
        }


# %%
# Solvers for constrained convex optimisation via saddle point formulation
def __sps(
    max_iter,
    start_iter,
    cp: CP,
    optimiser,
    initial_solution,
    initial_avg_state=None,
    initial_opt_state=None,
    exponential_weighting=0.01,
):

    primal_projection = lambda x: projection_box(x, cp.lower_bounds, cp.upper_bounds)

    def lagrangian(state):
        return (
            cp.objective(state["primal"])
            + state["dual_eq"] @ (cp.constraints_eq(state["primal"]))
            + state["dual_ineq"] @ (cp.constraints_ineq(state["primal"]))
        )

    def grad(state):
        return jax.grad(lagrangian)(state)

    @jax.jit
    def opt_update(gradient, opt_state, state):
        return optimiser.update(gradient, opt_state, state)

    @jax.jit
    def body_fun(_, loop_vars):
        (
            i,
            state,
            average_state,
            opt_state,
        ) = loop_vars

        gradient = grad(state)
        gradient["dual_eq"] = -gradient["dual_eq"]  # ascent for dual eq
        gradient["dual_ineq"] = -gradient["dual_ineq"]  # ascent for dual ineq
        updates, opt_state = opt_update(gradient, opt_state, state)
        state = optax.apply_updates(state, updates)
        state["primal"] = primal_projection(state["primal"])
        state["dual_ineq"] = projection_non_negative(state["dual_ineq"])

        average_state = optax.incremental_update(
            state, average_state, exponential_weighting
        )

        return (
            i + 1,
            state,
            average_state,
            opt_state,
        )

    @jax.jit
    def the_final_show():
        state = initial_solution
        if initial_avg_state is not None:
            average_state = initial_avg_state
        else:
            average_state = initial_solution
        if initial_opt_state is not None:
            opt_state = initial_opt_state
        else:
            opt_state = optimiser.init(initial_solution)

        loop_vars = (
            start_iter,
            state,
            average_state,
            opt_state,
        )

        loop_vars = jax.lax.fori_loop(0, max_iter, body_fun, loop_vars)
        i, state, average_state, opt_state = loop_vars

        return i, state, average_state, opt_state

    return the_final_show()


def solve(
    cp: CP,
    iterations_per_epoch=int(1e4),
    initial_solution=None,
    optimiser=None,
    constraint_tolerance=1e-4,
    progress_tolerance=1e-4,
    complementarity_tolerance=1e-4,
    exponential_weighting=0.01,
    max_epochs=1000,
):
    if initial_solution is None:
        initial_solution = cp.initial_solution()

    if optimiser is None:
        optimiser = jo.adamdelta_saddle()

    i = 1
    state = initial_solution
    average_state = initial_solution
    opt_state = None
    progress = jnp.inf
    max_complementarity_slack = jnp.inf
    constraints_satisfied = False
    count = 0

    start_time_total = time.time()

    while (
        progress > progress_tolerance
        or max_complementarity_slack > complementarity_tolerance
    ) or not constraints_satisfied:
        start_time = time.time()
        i, state, new_average_state, opt_state = __sps(
            iterations_per_epoch,
            i,
            cp,
            optimiser,
            state,
            average_state,
            opt_state,
            exponential_weighting,
        )
        end_time = time.time()

        objective_value = cp.objective(new_average_state["primal"])

        progress = (cp.objective(average_state["primal"]) - objective_value) / (
            1.0 + jnp.abs(objective_value)
        )

        ineq_violations = jnp.maximum(cp.ineq_slack(new_average_state["primal"]), 0.0)
        max_ineq_violation = jnp.max(ineq_violations)

        eq_violations = jnp.abs(cp.eq_slack(new_average_state["primal"]))
        max_eq_violation = jnp.max(eq_violations)

        complentariy_slack = cp.complementarity_slack(
            new_average_state["primal"], new_average_state["dual_ineq"]
        ) / (1.0 + jnp.abs(objective_value))
        max_complementarity_slack = jnp.abs(jnp.sum(complentariy_slack))

        print("--------------------------------")
        print(f"Epoch time: {end_time - start_time:.2f} seconds")
        print(
            f"|obj: {objective_value:.6f} |prog: {progress:.6f}|ineq_viol: {max_ineq_violation:.6f}|eq_viol: {max_eq_violation:.6f}|comp_slack: {max_complementarity_slack:.6f}|"
        )
        print("--------------------------------")

        constraints_satisfied = (
            max_ineq_violation < constraint_tolerance
            and max_eq_violation < constraint_tolerance
        )
        average_state = new_average_state
        count += 1

        if count >= max_epochs:
            print("Maximum epochs reached, terminating solver.")
            break

    end_time_total = time.time()
    print(f"Total solve time: {end_time_total - start_time_total:.2f} seconds")
    print(f"Total epochs: {count}")

    return average_state


# %%
