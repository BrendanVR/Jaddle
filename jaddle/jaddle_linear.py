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


class PrimalState(NamedTuple):
    primal: jnp.ndarray


class DualState(NamedTuple):
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
        self.b_eq = b_eq
        self.A_ineq = A_ineq
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

    def complementarity_slack(self, x, dual_ineq):
        return (dual_ineq * (self.A_ineq @ x - self.b_ineq)).sum()

    def primal_initial_solution(self):
        return PrimalState(primal=jnp.zeros(self.num_variables()))

    def dual_initial_solution(self):
        return DualState(
            dual_ineq=jnp.zeros(self.num_ineq_constraints()),
            dual_eq=jnp.zeros(self.num_eq_constraints()),
        )


# %%
# Solvers for constrained linear optimisation via saddle point formulation
def __sps(
    max_iter,
    start_iter,
    lp: LP,
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

    # Cache transposes to avoid recomputing each gradient step
    A_eq_T = lp.A_eq.T
    A_ineq_T = lp.A_ineq.T

    @jax.jit
    def primal_projection(x):
        return projection_box(x, lp.lower_bounds, lp.upper_bounds)

    @jax.jit
    def grad_primal(primal_state, dual_state):
        grad_primal = (
            lp.c + A_ineq_T @ dual_state.dual_ineq + A_eq_T @ dual_state.dual_eq
        )
        return PrimalState(primal=grad_primal)

    @jax.jit
    def grad_dual(primal_state, dual_state):
        grad_dual_ineq = lp.b_ineq - lp.A_ineq @ primal_state.primal
        grad_dual_eq = lp.b_eq - lp.A_eq @ primal_state.primal
        return DualState(
            dual_ineq=grad_dual_ineq,
            dual_eq=grad_dual_eq,
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
    lp: LP,
    primal_initial_solution=None,
    dual_initial_solution=None,
    primal_optimiser=None,
    dual_optimiser=None,
    iterations_per_epoch=int(1e3),
    constraint_tolerance=1e-5,
    progress_tolerance=1e-5,
    complementarity_tolerance=1e-5,
    exponential_weighting=0.01,
    scale_A=False,
    scale_b=False,
    scale_c=False,
    max_epochs=1000,
):

    if primal_initial_solution is None:
        primal_initial_solution = lp.primal_initial_solution()

    if dual_initial_solution is None:
        dual_initial_solution = lp.dual_initial_solution()

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

    if scale_A and not sp.issparse(lp.A_eq):
        lp, rs_ruiz, cs_ruiz = __ruiz_scaling(lp)
        lp, rs_pc, cs_pc = __pc_scaling(lp)
        lp_summary_statistics(lp)
        lp = __to_jaddle(lp)

    elif scale_A and sp.issparse(lp.A_eq):
        lp, rs_ruiz, cs_ruiz = __ruiz_scaling_sparse(lp)
        lp, rs_pc, cs_pc = __pc_scaling_sparse(lp)
        lp_summary_statistics(lp)
        lp = __to_jaddle_sparse(lp)

    elif not sp.issparse(lp.A_eq):
        lp_summary_statistics(lp)
        lp = __to_jaddle(lp)

    else:
        lp_summary_statistics(lp)
        lp = __to_jaddle_sparse(lp)

    if scale_c:
        c_scale = jnp.max(jnp.abs(lp.c))
        lp.c = lp.c / c_scale

    if scale_b:
        b_scale_eq = jnp.max(jnp.abs(lp.b_eq))

        def _scale_eq(_):
            return (lp.b_eq / b_scale_eq, lp.A_eq / b_scale_eq)

        def _no_scale(_):
            return (lp.b_eq, lp.A_eq)

        lp.b_eq, lp.A_eq = jax.lax.cond(
            b_scale_eq > 0, _scale_eq, _no_scale, operand=None
        )

        b_scale_ineq = jnp.max(jnp.abs(lp.b_ineq))

        def _scale_ineq(_):
            return (lp.b_ineq / b_scale_ineq, lp.A_ineq / b_scale_ineq)

        def _no_scale_ineq(_):
            return (lp.b_ineq, lp.A_ineq)

        lp.b_ineq, lp.A_ineq = jax.lax.cond(
            b_scale_ineq > 0, _scale_ineq, _no_scale_ineq, operand=None
        )

    i = 1
    primal_state = state = primal_initial_solution
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
            lp,
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

        objective_value = lp.objective(primal_average_state.primal)

        if scale_c:
            objective_value = c_scale * objective_value
            progress = jnp.abs(
                c_scale * lp.objective(primal_average_state.primal) - objective_value
            ) / (1.0 + jnp.abs(objective_value))
        else:
            progress = jnp.abs(
                lp.objective(primal_average_state.primal) - objective_value
            ) / (1.0 + jnp.abs(objective_value))

        ineq_violations = jnp.maximum(
            lp.A_ineq @ primal_average_state.primal - lp.b_ineq, 0.0
        )
        max_ineq_violation = jnp.max(ineq_violations)

        eq_violations = jnp.abs(lp.A_eq @ primal_average_state.primal - lp.b_eq)
        max_eq_violation = jnp.max(eq_violations)

        complentariy_slack = (
            dual_average_state.dual_ineq
            * (lp.A_ineq @ primal_average_state.primal - lp.b_ineq)
            / (1.0 + jnp.abs(objective_value))
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

    if scale_A:
        primal_average_state = PrimalState(
            primal=primal_average_state.primal * cs_pc * cs_ruiz,
        )
        dual_average_state = DualState(
            dual_eq=dual_average_state.dual_eq
            * rs_pc[: lp.A_eq.shape[0]]
            * rs_ruiz[: lp.A_eq.shape[0]],
            dual_ineq=dual_average_state.dual_ineq
            * rs_pc[lp.A_eq.shape[0] :]
            * rs_ruiz[lp.A_eq.shape[0] :],
        )

    return primal_average_state, dual_average_state


# %%
def __ruiz_scaling(lp: LP, max_iter=10, threshold=1e-8, clip_bounds=(1e-6, 1e6)):
    """
    Applies Ruiz scaling to an LP in standard form:
        min c^T x
        s.t. A_eq x = b_eq
             A_ineq x <= b_ineq
             lower_bounds <= x <= upper_bounds
    Returns scaled (c, A_eq, b_eq, A_ineq, b_ineq, lower_bounds, upper_bounds), row_scaling, col_scaling
    """

    # Stack all constraint matrices for scaling
    A = np.vstack([lp.A_eq, lp.A_ineq])
    m, n = A.shape
    row_scale = np.ones(m)
    col_scale = np.ones(n)
    c_scaled = lp.c.copy()
    b_scaled = np.concatenate([lp.b_eq, lp.b_ineq])
    lower_bounds = lp.lower_bounds.copy()
    upper_bounds = lp.upper_bounds.copy()

    for _ in range(max_iter):
        row_norms = np.linalg.norm(A, axis=1, ord=np.inf)
        row_norms = np.where(row_norms <= threshold, 1.0, row_norms)

        col_norms = np.linalg.norm(A, axis=0, ord=np.inf)
        col_norms = np.where(col_norms <= threshold, 1.0, col_norms)

        row_s = 1.0 / np.sqrt(row_norms)
        row_s = np.clip(row_s, clip_bounds[0], clip_bounds[1])
        A = (A.T * row_s).T
        row_scale *= row_s

        col_s = 1.0 / np.sqrt(col_norms)
        col_s = np.clip(col_s, clip_bounds[0], clip_bounds[1])
        A = A * col_s
        col_scale *= col_s

    # Split back A_eq and A_ineq, b_eq and b_ineq
    A_eq_scaled = A[: lp.A_eq.shape[0], :]
    A_ineq_scaled = A[lp.A_eq.shape[0] :, :]
    c_scaled = c_scaled * col_scale
    b_scaled = b_scaled * row_scale

    b_eq_scaled = b_scaled[: lp.A_eq.shape[0]]
    b_ineq_scaled = b_scaled[lp.A_eq.shape[0] :]
    lower_bounds_scaled = lower_bounds / col_scale
    upper_bounds_scaled = upper_bounds / col_scale

    lp = LP(
        c_scaled,
        A_eq_scaled,
        b_eq_scaled,
        A_ineq_scaled,
        b_ineq_scaled,
        lower_bounds_scaled,
        upper_bounds_scaled,
    )

    return lp, row_scale, col_scale


def __ruiz_scaling_sparse(lp: LP, max_iter=10, threshold=1e-8, clip_bounds=(1e-6, 1e6)):
    """
    Applies Ruiz scaling to an LP in standard form with sparse matrices:
        min c^T x
        s.t. A_eq x = b_eq
             A_ineq x <= b_ineq
             lower_bounds <= x <= upper_bounds
    Returns scaled LP, row_scaling, col_scaling
    """

    # Stack all constraint matrices for scaling
    A_eq = sp.csc_matrix(lp.A_eq)
    A_ineq = sp.csc_matrix(lp.A_ineq)
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

    return lp_scaled, row_scale, col_scale


def __pc_scaling(lp: LP, max_iter=1, threshold=1e-8, clip_bounds=(1e-6, 1e6)):
    """
    Applies Ruiz scaling to an LP in standard form:
        min c^T x
        s.t. A_eq x = b_eq
             A_ineq x <= b_ineq
             lower_bounds <= x <= upper_bounds
    Returns scaled (c, A_eq, b_eq, A_ineq, b_ineq, lower_bounds, upper_bounds), row_scaling, col_scaling
    """

    # Stack all constraint matrices for scaling
    A = np.vstack([lp.A_eq, lp.A_ineq])
    m, n = A.shape
    row_scale = np.ones(m)
    col_scale = np.ones(n)
    c_scaled = lp.c.copy()
    b_scaled = np.concatenate([lp.b_eq, lp.b_ineq])
    lower_bounds = lp.lower_bounds.copy()
    upper_bounds = lp.upper_bounds.copy()

    for _ in range(max_iter):
        row_norms = np.linalg.norm(A, axis=1, ord=1)
        row_norms = np.where(row_norms <= threshold, 1.0, row_norms)

        col_norms = np.linalg.norm(A, axis=0, ord=1)
        col_norms = np.where(col_norms <= threshold, 1.0, col_norms)

        row_s = 1.0 / np.sqrt(row_norms)
        row_s = np.clip(row_s, clip_bounds[0], clip_bounds[1])
        A = (A.T * row_s).T
        row_scale *= row_s

        col_s = 1.0 / np.sqrt(col_norms)
        col_s = np.clip(col_s, clip_bounds[0], clip_bounds[1])
        A = A * col_s
        col_scale *= col_s

    # Split back A_eq and A_ineq, b_eq and b_ineq
    A_eq_scaled = A[: lp.A_eq.shape[0], :]
    A_ineq_scaled = A[lp.A_eq.shape[0] :, :]
    c_scaled = c_scaled * col_scale
    b_scaled = b_scaled * row_scale

    b_eq_scaled = b_scaled[: lp.A_eq.shape[0]]
    b_ineq_scaled = b_scaled[lp.A_eq.shape[0] :]
    lower_bounds_scaled = lower_bounds / col_scale
    upper_bounds_scaled = upper_bounds / col_scale

    lp = LP(
        c_scaled,
        A_eq_scaled,
        b_eq_scaled,
        A_ineq_scaled,
        b_ineq_scaled,
        lower_bounds_scaled,
        upper_bounds_scaled,
    )

    return lp, row_scale, col_scale


def __pc_scaling_sparse(lp: LP, max_iter=1, threshold=1e-8, clip_bounds=(1e-6, 1e6)):
    """
    Applies PC scaling to an LP in standard form with sparse matrices:
        min c^T x
        s.t. A_eq x = b_eq
                A_ineq x <= b_ineq
                lower_bounds <= x <= upper_bounds
    Returns scaled LP, row_scaling, col_scaling
    """

    # Stack all constraint matrices for scaling
    A_eq = sp.csc_matrix(lp.A_eq)
    A_ineq = sp.csc_matrix(lp.A_ineq)
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

    return lp_scaled, row_scale, col_scale


def __to_jaddle(lp: LP):
    lp_jax = LP(
        jnp.array(lp.c),
        jnp.array(lp.A_eq),
        jnp.array(lp.b_eq),
        jnp.array(lp.A_ineq),
        jnp.array(lp.b_ineq),
        jnp.array(lp.lower_bounds),
        jnp.array(lp.upper_bounds),
    )
    return lp_jax


def __to_jaddle_sparse(lp: LP):
    lp_jax = LP(
        jnp.array(lp.c),
        jsp.BCOO.from_scipy_sparse(lp.A_eq),
        jnp.array(lp.b_eq),
        jsp.BCOO.from_scipy_sparse(lp.A_ineq),
        jnp.array(lp.b_ineq),
        jnp.array(lp.lower_bounds),
        jnp.array(lp.upper_bounds),
    )
    return lp_jax


def lp_summary_statistics(lp: LP):
    num_vars = lp.num_variables()
    num_eq = lp.num_eq_constraints()
    num_ineq = lp.num_ineq_constraints()
    min_A_eq = np.min(lp.A_eq) if num_eq > 0 else None
    max_A_eq = np.max(lp.A_eq) if num_eq > 0 else None
    min_b_eq = np.min(lp.b_eq) if num_eq > 0 else None
    max_b_eq = np.max(lp.b_eq) if num_eq > 0 else None
    min_A_ineq = np.min(lp.A_ineq) if num_ineq > 0 else None
    max_A_ineq = np.max(lp.A_ineq) if num_ineq > 0 else None
    min_b_ineq = np.min(lp.b_ineq) if num_ineq > 0 else None
    max_b_ineq = np.max(lp.b_ineq) if num_ineq > 0 else None
    min_c = np.min(lp.c)
    max_c = np.max(lp.c)
    bound_range = lp.upper_bounds - lp.lower_bounds
    max_bound_range = np.max(bound_range)
    min_bound_range = np.min(bound_range)

    print("--------------------------------")
    print("LP Summary Statistics:")
    print(f"Number of variables: {num_vars}")
    print(f"Number of equality constraints: {num_eq}")
    print(f"Number of inequality constraints: {num_ineq}")
    print(f"[Min, Max] of c: [{min_c:.2f}, {max_c:.2f}]")
    print(f"[Min, Max] of A_eq: [{min_A_eq:.2f}, {max_A_eq:.2f}]")
    print(f"[Min, Max] of b_eq: [{min_b_eq:.2f}, {max_b_eq:.2f}]")
    print(f"[Min, Max] of A_ineq: [{min_A_ineq:.2f}, {max_A_ineq:.2f}]")
    print(f"[Min, Max] of b_ineq: [{min_b_ineq:.2f}, {max_b_ineq:.2f}]")
    print(f"[Min, Max] of bound range: [{min_bound_range:.2f}, {max_bound_range:.2f}]")
    print("--------------------------------")


# %%
