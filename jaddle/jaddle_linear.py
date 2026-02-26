# %%
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsp
from optax.projections import projection_non_negative, projection_box
import optax
import numpy as np
import functools
from typing import NamedTuple
import time
from scipy import sparse as sp


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
    weight_function=lambda _: 1.0,
    total_weight=0.0,
    primal_damping=0.0,
    dual_damping_ineq=0.0,
    dual_damping_eq=0.0,
    average=True,
):

    @jax.jit
    def projection_primal(primal_state):
        return projection_box(primal_state, lp.lower_bounds, lp.upper_bounds)

    @jax.jit
    def grad(state):
        grad_primal = (
            lp.c
            + lp.A_ineq_T @ state.dual_ineq
            + lp.A_eq_T @ state.dual_eq
            + primal_damping * state.primal
        )
        grad_dual_ineq = (
            lp.b_ineq - lp.A_ineq @ state.primal + dual_damping_ineq * state.dual_ineq
        )
        grad_dual_eq = (
            lp.b_eq - lp.A_eq @ state.primal + dual_damping_eq * state.dual_eq
        )
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
    lp: LP,
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
    scale=None,
):
    lp_summary_statistics(lp)

    if initial_solution is None:
        initial_solution = lp.initial_solution()

    if scale == "ruiz":
        lp, row_scale, col_scale = ruiz_scaling(lp)

    elif scale == "pc":
        lp, row_scale, col_scale = pc_scaling(lp)

    @jax.jit
    def compute_epoch_metrics(average_state):
        objective_value = lp.objective(average_state.primal)

        grad_primal = (
            lp.c
            + lp.A_ineq_T @ average_state.dual_ineq
            + lp.A_eq_T @ average_state.dual_eq
        )
        grad_dual_ineq = lp.A_ineq @ average_state.primal - lp.b_ineq
        grad_dual_eq = lp.A_eq @ average_state.primal - lp.b_eq

        projected_primal = projection_box(
            average_state.primal - grad_primal,
            lp.lower_bounds,
            lp.upper_bounds,
        )
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
            lp,
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
            if scale in ["ruiz", "pc"]:
                return SaddleState(
                    primal=average_state.primal * col_scale,
                    dual_ineq=average_state.dual_ineq
                    * row_scale[len(average_state.dual_eq) :],
                    dual_eq=average_state.dual_eq
                    * row_scale[: len(average_state.dual_eq)],
                )
            else:
                return average_state
        else:
            if scale in ["ruiz", "pc"]:
                return SaddleState(
                    primal=state.primal * col_scale,
                    dual_ineq=state.dual_ineq * row_scale[len(state.dual_eq) :],
                    dual_eq=state.dual_eq * row_scale[: len(state.dual_eq)],
                )
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
                lp,
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
            if scale in ["ruiz", "pc"]:
                return SaddleState(
                    primal=average_state.primal * col_scale,
                    dual_ineq=average_state.dual_ineq
                    * row_scale[len(average_state.dual_eq) :],
                    dual_eq=average_state.dual_eq
                    * row_scale[: len(average_state.dual_eq)],
                )
            else:
                return average_state
        else:
            if scale in ["ruiz", "pc"]:
                return SaddleState(
                    primal=state.primal * col_scale,
                    dual_ineq=state.dual_ineq * row_scale[len(state.dual_eq) :],
                    dual_eq=state.dual_eq * row_scale[: len(state.dual_eq)],
                )
            else:
                return state


# %%
def to_jaddle_sparse(lp: LP):
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


def project_onto_equality_constraints(
    lp: LP, x: jnp.ndarray, num_iterations: int = 100, step_size: float = 0.1
) -> jnp.ndarray:
    """
    Project a point onto the equality constraints of an LP using iterative projection.

    Args:
        lp: Linear program with equality constraints
        x: Point to project
        num_iterations: Number of projection iterations
        step_size: Step size for each projection step

    Returns:
        Projected point satisfying A_eq @ x ≈ b_eq
    """

    def project_step(x_curr):
        residual = lp.A_eq @ x_curr - lp.b_eq
        gradient = lp.A_eq_T @ residual
        return x_curr - step_size * gradient

    x_proj = x
    for _ in range(num_iterations):
        x_proj = project_step(x_proj)

    return x_proj


def __convert_to_scipy(jsp_mat: jsp.BCOO) -> sp.csc_matrix:
    data = np.array(jsp_mat.data)
    indices = np.array(jsp_mat.indices)
    row, col = indices[:, 0], indices[:, 1]

    return sp.csc_matrix((data, (row, col)), shape=jsp_mat.shape)


def ruiz_scaling(lp: LP, max_iter=10, threshold=1e-8, clip_bounds=(1e-6, 1e6)):
    """
    Applies Ruiz scaling to an LP in standard form with sparse matrices:
        min c^T x
        s.t. A_eq x = b_eq
             A_ineq x <= b_ineq
             lower_bounds <= x <= upper_bounds
    Returns scaled LP, row_scaling, col_scaling
    """

    # Stack all constraint matrices for scaling
    A_eq = __convert_to_scipy(lp.A_eq)
    A_ineq = __convert_to_scipy(lp.A_ineq)
    A = sp.vstack([A_eq, A_ineq]).tocsc()
    m, n = A.shape
    row_scale = np.ones(m)
    col_scale = np.ones(n)
    c_scaled = lp.c.copy()
    b_scaled = np.concatenate([lp.b_eq, lp.b_ineq])
    lower_bounds = lp.lower_bounds.copy()
    upper_bounds = lp.upper_bounds.copy()

    A_scaled = A.copy()

    for _ in range(max_iter):
        # Row norms (infinity norm)
        row_norms = np.abs(A_scaled).max(axis=1).todense().A.flatten()
        row_norms = np.maximum(row_norms, np.abs(b_scaled * row_scale))
        row_norms = np.where(row_norms <= threshold, 1.0, row_norms)
        row_s = 1.0 / np.sqrt(row_norms)
        row_s = np.clip(row_s, clip_bounds[0], clip_bounds[1])
        # Scale rows
        A_scaled = sp.diags(row_s) @ A_scaled
        row_scale *= row_s

        # Column norms (infinity norm)
        col_norms = np.abs(A_scaled).max(axis=0).todense().A.flatten()
        col_norms = np.where(col_norms <= 1e-8, 1.0, col_norms)
        col_s = 1.0 / np.sqrt(col_norms)
        col_s = np.clip(col_s, 1e-6, 1e6)
        # Scale columns
        A_scaled = A_scaled @ sp.diags(col_s)
        col_scale *= col_s

    # Split back A_eq and A_ineq, b_eq and b_ineq
    A_eq_scaled = A_scaled[: lp.A_eq.shape[0], :]
    A_ineq_scaled = A_scaled[lp.A_eq.shape[0] :, :]
    c_scaled = c_scaled * col_scale
    b_scaled = b_scaled * row_scale

    b_eq_scaled = b_scaled[: lp.A_eq.shape[0]]
    b_ineq_scaled = b_scaled[lp.A_eq.shape[0] :]
    lower_bounds_scaled = lower_bounds / col_scale
    upper_bounds_scaled = upper_bounds / col_scale

    lp_scaled = LP(
        c_scaled,
        A_eq_scaled,
        b_eq_scaled,
        A_ineq_scaled,
        b_ineq_scaled,
        lower_bounds_scaled,
        upper_bounds_scaled,
    )

    lp_scaled = to_jaddle_sparse(lp_scaled)

    return lp_scaled, row_scale, col_scale


def pc_scaling(lp: LP, max_iter=1, threshold=1e-8, clip_bounds=(1e-6, 1e6)):
    """
    Applies PC scaling to an LP in standard form with sparse matrices:
        min c^T x
        s.t. A_eq x = b_eq
                A_ineq x <= b_ineq
                lower_bounds <= x <= upper_bounds
    Returns scaled LP, row_scaling, col_scaling
    """

    # Stack all constraint matrices for scaling
    A_eq = __convert_to_scipy(lp.A_eq)
    A_ineq = __convert_to_scipy(lp.A_ineq)
    A = sp.vstack([A_eq, A_ineq]).tocsc()
    m, n = A.shape
    row_scale = np.ones(m)
    col_scale = np.ones(n)
    c_scaled = lp.c.copy()
    b_scaled = np.concatenate([lp.b_eq, lp.b_ineq])
    lower_bounds = lp.lower_bounds.copy()
    upper_bounds = lp.upper_bounds.copy()

    A_scaled = A.copy()

    for _ in range(max_iter):
        # Row norms (1-norm)
        row_norms = np.abs(A_scaled).sum(axis=1).A.flatten()
        row_norms = np.maximum(row_norms, np.abs(b_scaled * row_scale))
        row_norms = np.where(row_norms <= threshold, 1.0, row_norms)
        row_s = 1.0 / np.sqrt(row_norms)
        row_s = np.clip(row_s, clip_bounds[0], clip_bounds[1])
        # Scale rows
        A_scaled = sp.diags(row_s) @ A_scaled
        row_scale *= row_s

        # Column norms (1-norm)
        col_norms = np.abs(A_scaled).sum(axis=0).A.flatten()
        col_norms = np.where(col_norms <= threshold, 1.0, col_norms)
        col_s = 1.0 / np.sqrt(col_norms)
        col_s = np.clip(col_s, clip_bounds[0], clip_bounds[1])
        # Scale columns
        A_scaled = A_scaled @ sp.diags(col_s)
        col_scale *= col_s

    # Split back A_eq and A_ineq, b_eq and b_ineq
    A_eq_scaled = A_scaled[: lp.A_eq.shape[0], :]
    A_ineq_scaled = A_scaled[lp.A_eq.shape[0] :, :]
    c_scaled = c_scaled * col_scale
    b_scaled = b_scaled * row_scale

    b_eq_scaled = b_scaled[: lp.A_eq.shape[0]]
    b_ineq_scaled = b_scaled[lp.A_eq.shape[0] :]
    lower_bounds_scaled = lower_bounds / col_scale
    upper_bounds_scaled = upper_bounds / col_scale

    lp_scaled = LP(
        c_scaled,
        A_eq_scaled,
        b_eq_scaled,
        A_ineq_scaled,
        b_ineq_scaled,
        lower_bounds_scaled,
        upper_bounds_scaled,
    )

    lp_scaled = to_jaddle_sparse(lp_scaled)

    return lp_scaled, row_scale, col_scale
