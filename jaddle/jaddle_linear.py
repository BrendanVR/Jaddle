# %%
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsp
from optax.projections import projection_non_negative, projection_box
import optax
import numpy as np
import scipy.sparse as sp
import functools
from typing import NamedTuple
import time
from jax.scipy.sparse.linalg import gmres


class SaddleState(NamedTuple):
    primal: jnp.ndarray
    dual_ineq: jnp.ndarray
    dual_eq: jnp.ndarray


# %%
# Basic Types
class LP:
    def __init__(
        self,
        c,
        A_eq,
        b_eq,
        A_ineq,
        b_ineq,
        lower_bounds,
        upper_bounds,
    ):
        self.c = c
        self.A_eq = A_eq
        self.A_eq_T = self.A_eq.T
        self.b_eq = b_eq
        self.A_ineq = A_ineq
        self.A_ineq_T = self.A_ineq.T
        self.b_ineq = b_ineq
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def objective(self, x):
        return self.c @ x

    def num_variables(self):
        return len(self.c)

    def num_eq_constraints(self):
        return self.A_eq.shape[0]

    def num_ineq_constraints(self):
        return self.A_ineq.shape[0]

    def num_constraints(self):
        return self.A_eq.shape[0] + self.A_ineq.shape[0]

    def ineq_slack(self, x):
        return jnp.max(jnp.maximum(self.A_ineq @ x - self.b_ineq, 0.0))

    def eq_slack(self, x):
        return jnp.max(jnp.abs(self.A_eq @ x - self.b_eq))

    def diff_eq_slack(self, x):
        return self.A_eq @ x - self.b_eq

    def complementarity_slack(self, x, dual_ineq):
        return (dual_ineq * (self.A_ineq @ x - self.b_ineq)).sum()

    def initial_solution(self):

        return SaddleState(
            primal=jnp.zeros(self.num_variables()),
            dual_ineq=jnp.zeros(self.num_ineq_constraints()),
            dual_eq=jnp.zeros(self.num_eq_constraints()),
        )


# %%
# Solvers for constrained linear optimisation via saddle point formulation
def __sps(
    max_iter,
    start_iter,
    lp: LP,
    optimiser,
    initial_solution,
    initial_avg_state=None,
    initial_opt_state=None,
    exponential_weighting=0.01,
):

    @jax.jit
    def projection_primal(primal_state):
        return projection_box(primal_state, lp.lower_bounds, lp.upper_bounds)

    @jax.jit
    def grad(state):
        grad_primal = lp.c + lp.A_ineq_T @ state.dual_ineq + lp.A_eq_T @ state.dual_eq
        grad_dual_ineq = lp.b_ineq - lp.A_ineq @ state.primal
        grad_dual_eq = lp.b_eq - lp.A_eq @ state.primal
        return SaddleState(
            primal=grad_primal,
            dual_ineq=grad_dual_ineq,
            dual_eq=grad_dual_eq,
        )

    @jax.jit
    def opt_update(gradient, opt_state, state):
        return optimiser.update(gradient, opt_state, state)

    @functools.partial(
        jax.jit,
        static_argnames=(
            "max_iter",
            "exponential_weighting",
        ),
    )
    def run_epoch(
        max_iter,
        exponential_weighting,
        start_iter,
        state,
        average_state,
        opt_state,
    ):
        @jax.jit
        def step(carry, _):
            i, state, average_state, opt_state = carry

            gradient = grad(state)
            updates, opt_state = opt_update(gradient, opt_state, state)
            state = optax.apply_updates(state, updates)
            state = SaddleState(
                primal=projection_primal(state.primal),
                dual_ineq=projection_non_negative(state.dual_ineq),
                dual_eq=state.dual_eq,
            )

            average_state = optax.incremental_update(
                state, average_state, exponential_weighting
            )

            return (i + 1, state, average_state, opt_state), None

        (i, state, average_state, opt_state), _ = jax.lax.scan(
            step,
            (start_iter, state, average_state, opt_state),
            None,
            length=max_iter,
        )

        return i, state, average_state, opt_state

    state = initial_solution

    if initial_avg_state is not None:
        average_state = initial_avg_state
    else:
        # primal_average_state = jax.tree_util.tree_map(
        #     lambda x: x.copy(), primal_initial_solution
        # )
        average_state = initial_solution

    if initial_opt_state is not None:
        opt_state = initial_opt_state
    else:
        opt_state = optimiser.init(initial_solution)

    return run_epoch(
        max_iter,
        exponential_weighting,
        start_iter,
        state,
        average_state,
        opt_state,
    )


def solve(
    lp: LP,
    optimiser,
    initial_solution=None,
    iterations_per_epoch=int(1e3),
    constraint_tolerance=1e-4,
    progress_tolerance=1e-4,
    complementarity_tolerance=1e-4,
    exponential_weighting=0.01,
    max_epochs=1000,
    verbose=False,
    project_to_feasible=False,
    use_double_precision=False,
    initialize_to_feasible=False,
):

    if use_double_precision:
        jax.config.update("jax_enable_x64", True)
        lp = __to_jaddle_sparse64(lp)
    else:
        jax.config.update("jax_enable_x64", False)
        lp = __to_jaddle_sparse(lp)

    if initial_solution is not None:
        initial_solution = initial_solution

    if initial_solution is None:
        initial_solution = lp.initial_solution()

    if initialize_to_feasible:
        residual = lp.diff_eq_slack(initial_solution.primal)

        def matvec(v):
            return lp.A_eq @ (lp.A_eq.T @ v)

        nu, _ = gmres(matvec, residual)
        primal_projected = initial_solution.primal - lp.A_eq.T @ nu

        # Update state with projected primal
        initial_solution = SaddleState(
            primal=primal_projected,
            dual_ineq=initial_solution.dual_ineq,
            dual_eq=initial_solution.dual_eq,
        )

    lp_summary_statistics(lp)

    i = 1
    state = initial_solution
    average_state = initial_solution
    opt_state = optimiser.init(initial_solution)
    previous_objective = jnp.inf
    primal_grad_norm = jnp.inf
    complementarity_slack = jnp.inf
    constraint_bound = jnp.inf
    count = 0

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
        ) = loop_vars

        (
            i,
            state,
            average_state,
            opt_state,
        ) = __sps(
            iterations_per_epoch,
            i,
            lp,
            optimiser,
            state,
            average_state,
            opt_state,
            exponential_weighting,
        )

        grad_primal = (
            lp.c
            + lp.A_ineq_T @ average_state.dual_ineq
            + lp.A_eq_T @ average_state.dual_eq
        )
        grad_dual_ineq = lp.A_ineq @ average_state.primal - lp.b_ineq
        grad_dual_eq = lp.A_eq @ average_state.primal - lp.b_eq

        finite_lower = jnp.isfinite(lp.lower_bounds)
        finite_upper = jnp.isfinite(lp.upper_bounds)

        lower_active = finite_lower & (
            average_state.primal <= (lp.lower_bounds + constraint_tolerance)
        )
        upper_active = finite_upper & (
            average_state.primal >= (lp.upper_bounds - constraint_tolerance)
        )

        dual_lower = jnp.where(lower_active, jnp.maximum(grad_primal, 0.0), 0.0)
        dual_upper = jnp.where(upper_active, jnp.maximum(-grad_primal, 0.0), 0.0)

        stationarity_residual = grad_primal - dual_lower + dual_upper
        primal_grad_norm = jnp.max(jnp.abs(stationarity_residual))

        ineq_violations = jnp.maximum(grad_dual_ineq, 0.0)
        max_ineq_violation = jnp.max(ineq_violations)

        eq_violations = jnp.abs(grad_dual_eq)
        max_eq_violation = jnp.max(eq_violations)

        lower_violation = jnp.where(
            finite_lower,
            jnp.maximum(lp.lower_bounds - average_state.primal, 0.0),
            0.0,
        )
        upper_violation = jnp.where(
            finite_upper,
            jnp.maximum(average_state.primal - lp.upper_bounds, 0.0),
            0.0,
        )
        max_bound_violation = jnp.maximum(
            jnp.max(lower_violation), jnp.max(upper_violation)
        )

        lower_slack = jnp.where(
            finite_lower, average_state.primal - lp.lower_bounds, 0.0
        )
        upper_slack = jnp.where(
            finite_upper, lp.upper_bounds - average_state.primal, 0.0
        )
        bound_complementarity_slack = jnp.maximum(
            jnp.max(jnp.abs(dual_lower * lower_slack)),
            jnp.max(jnp.abs(dual_upper * upper_slack)),
        )

        inequality_complementarity_slack = jnp.max(
            jnp.abs(average_state.dual_ineq * grad_dual_ineq)
        )
        complementarity_slack = jnp.maximum(
            inequality_complementarity_slack, bound_complementarity_slack
        )

        constraint_bound = jnp.maximum(
            jnp.maximum(max_ineq_violation, max_eq_violation), max_bound_violation
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
        )

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
        ) = loop_vars

        if project_to_feasible:
            residual = lp.diff_eq_slack(average_state.primal)

            def matvec(v):
                return lp.A_eq @ (lp.A_eq.T @ v)

            nu, _ = gmres(matvec, residual)
            primal_projected = average_state.primal - lp.A_eq.T @ nu

            # Update state with projected primal
            average_state = SaddleState(
                primal=primal_projected,
                dual_ineq=average_state.dual_ineq,
                dual_eq=average_state.dual_eq,
            )

        end_time = time.time()
        print(f"Time to solution: {end_time - start_time:.2f} seconds")
        return average_state

    else:
        while cond_fun(loop_vars):
            loop_vars = body_fun(loop_vars)
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
            ) = loop_vars

            print(
                f"Epoch {count}: Primal Grad Norm={primal_grad_norm:.2e}, Compl. Slack={complementarity_slack:.2e}, Constraint Bound={constraint_bound:.2e}"
            )

        if project_to_feasible:
            residual = lp.diff_eq_slack(average_state.primal)

            def matvec(v):
                return lp.A_eq @ (lp.A_eq.T @ v)

            nu, _ = gmres(matvec, residual)
            primal_projected = average_state.primal - lp.A_eq.T @ nu

            # Update state with projected primal
            average_state = SaddleState(
                primal=primal_projected,
                dual_ineq=average_state.dual_ineq,
                dual_eq=average_state.dual_eq,
            )

        end_time = time.time()
        print(f"Time to solution: {end_time - start_time:.2f} seconds")
        return average_state


# %%
def __to_jaddle_sparse(lp: LP):
    A_eq_sp = lp.A_eq.astype(np.float32)
    A_ineq_sp = lp.A_ineq.astype(np.float32)

    A_eq = jsp.BCOO.from_scipy_sparse(A_eq_sp)
    A_ineq = jsp.BCOO.from_scipy_sparse(A_ineq_sp)

    lp_jax = LP(
        jnp.array(lp.c, dtype=jnp.float32),
        A_eq,
        jnp.array(lp.b_eq, dtype=jnp.float32),
        A_ineq,
        jnp.array(lp.b_ineq, dtype=jnp.float32),
        jnp.array(lp.lower_bounds, dtype=jnp.float32),
        jnp.array(lp.upper_bounds, dtype=jnp.float32),
    )
    return lp_jax


def __to_jaddle_sparse64(lp: LP):
    A_eq_sp = lp.A_eq.astype(np.float64)
    A_ineq_sp = lp.A_ineq.astype(np.float64)

    A_eq = jsp.BCOO.from_scipy_sparse(A_eq_sp)
    A_ineq = jsp.BCOO.from_scipy_sparse(A_ineq_sp)

    lp_jax = LP(
        jnp.array(lp.c, dtype=jnp.float64),
        A_eq,
        jnp.array(lp.b_eq, dtype=jnp.float64),
        A_ineq,
        jnp.array(lp.b_ineq, dtype=jnp.float64),
        jnp.array(lp.lower_bounds, dtype=jnp.float64),
        jnp.array(lp.upper_bounds, dtype=jnp.float64),
    )
    return lp_jax


def lp_summary_statistics(lp: LP):
    num_vars = lp.num_variables()
    num_eq = lp.num_eq_constraints()
    num_ineq = lp.num_ineq_constraints()

    if lp.A_eq.data.size > 0:
        min_A_eq = np.minimum(np.min(lp.A_eq.data), 0.0)
        max_A_eq = np.maximum(np.max(lp.A_eq.data), 0.0)
        min_b_eq = np.min(lp.b_eq)
        max_b_eq = np.max(lp.b_eq)
    else:
        min_A_eq = None
        max_A_eq = None
        min_b_eq = None
        max_b_eq = None

    if lp.A_ineq.data.size > 0:
        min_A_ineq = np.minimum(np.min(lp.A_ineq.data), 0.0)
        max_A_ineq = np.maximum(np.max(lp.A_ineq.data), 0.0)
        min_b_ineq = np.min(lp.b_ineq)
        max_b_ineq = np.max(lp.b_ineq)
    else:
        min_A_ineq = None
        max_A_ineq = None
        min_b_ineq = None
        max_b_ineq = None

    min_c = np.min(lp.c)
    max_c = np.max(lp.c)

    print("--------------------------------")
    print("LP Summary Statistics:")
    print(f"Number of variables: {num_vars}")
    print(f"Number of equality constraints: {num_eq}")
    print(f"Number of inequality constraints: {num_ineq}")
    print(f"[Min, Max] of c: [{min_c}, {max_c}]")
    print(f"[Min, Max] of A_eq: [{min_A_eq}, {max_A_eq}]")
    print(f"[Min, Max] of b_eq: [{min_b_eq}, {max_b_eq}]")
    print(f"[Min, Max] of A_ineq: [{min_A_ineq}, {max_A_ineq}]")
    print(f"[Min, Max] of b_ineq: [{min_b_ineq}, {max_b_ineq}]")
    print("--------------------------------")


# %%
