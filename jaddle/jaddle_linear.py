# %%
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsp
import jaddle.jaddle_optimisers as jo
from optax.projections import projection_non_negative, projection_box
import optax
import numpy as np
import scipy.sparse as sp
import time


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

    def initial_solution(self):
        return {
            "primal": jnp.zeros(self.num_variables()),
            "dual_ineq": jnp.zeros(self.num_ineq_constraints()),
            "dual_eq": jnp.zeros(self.num_eq_constraints()),
        }


class JaddleLP:
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

    def initial_solution(self):
        return {
            "primal": jnp.zeros(self.num_variables()),
            "dual_ineq": jnp.zeros(self.num_ineq_constraints()),
            "dual_eq": jnp.zeros(self.num_eq_constraints()),
        }


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

    def initial_solution(self):
        return {
            "primal": jnp.zeros(self.num_variables()),
            "dual_ineq": jnp.zeros(self.num_ineq_constraints()),
            "dual_eq": jnp.zeros(self.num_eq_constraints()),
        }


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

    primal_projection = lambda x: projection_box(x, lp.lower_bounds, lp.upper_bounds)

    @jax.jit
    def grad(state):
        grad_primal = (
            lp.c + lp.A_ineq.T @ state["dual_ineq"] + lp.A_eq.T @ state["dual_eq"]
        )
        grad_dual_ineq = lp.b_ineq - lp.A_ineq @ state["primal"]
        grad_dual_eq = lp.b_eq - lp.A_eq @ state["primal"]
        return {
            "primal": grad_primal,
            "dual_ineq": grad_dual_ineq,
            "dual_eq": grad_dual_eq,
        }

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
    iterations_per_epoch,
    lp: LP,
    initial_solution,
    optimiser=None,
    constraint_tolerance=1e-5,
    progress_tolerance=1e-4,
    complementarity_tolerance=1e-4,
    exponential_weighting=0.01,
    scale_rc=False,
    scale_objective=False,
):

    if optimiser is None:
        optimiser = jo.adamdelta_saddle()

    if scale_rc and not sp.issparse(lp.A_eq):
        lp, rs_ruiz, cs_ruiz = __ruiz_scaling(lp)
        lp, rs_pc, cs_pc = __pc_scaling(lp)
        lp_summary_statistics(lp)
        lp = __to_jaddle(lp)

    elif scale_rc and sp.issparse(lp.A_eq):
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

    if scale_objective:
        print("--------------------------------")
        obj_scale = jnp.max(jnp.abs(lp.c))
        lp.c = lp.c / obj_scale
        print(f"Objective scaled by factor: {obj_scale:.6f}")
        print(f"New [Min, Max] of c: [{jnp.min(lp.c):.6f}, {jnp.max(lp.c):.6f}]")
        print("--------------------------------")

    i = 1
    state = initial_solution
    average_state = initial_solution
    opt_state = None
    progress = jnp.inf
    max_complementarity_slack = jnp.inf
    constraints_satisfied = False

    start_time_total = time.time()

    while (
        progress > progress_tolerance
        or max_complementarity_slack > complementarity_tolerance
    ) or not constraints_satisfied:
        start_time = time.time()
        i, state, new_average_state, opt_state = __sps(
            iterations_per_epoch,
            i,
            lp,
            optimiser,
            state,
            average_state,
            opt_state,
            exponential_weighting,
        )
        end_time = time.time()

        objective_value = lp.c @ new_average_state["primal"]

        progress = jnp.abs(lp.c @ average_state["primal"] - objective_value) / (
            1.0 + jnp.abs(objective_value)
        )

        ineq_violations = jnp.maximum(
            lp.A_ineq @ new_average_state["primal"] - lp.b_ineq, 0.0
        ) / (1.0 + jnp.abs(lp.b_ineq))
        max_ineq_violation = jnp.max(ineq_violations)

        eq_violations = jnp.abs(lp.A_eq @ new_average_state["primal"] - lp.b_eq) / (
            1.0 + jnp.abs(lp.b_eq)
        )
        max_eq_violation = jnp.max(eq_violations)

        complentariy_slack = (
            new_average_state["dual_ineq"]
            * (lp.A_ineq @ new_average_state["primal"] - lp.b_ineq)
            / (1.0 + jnp.abs(objective_value))
        )
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

    end_time_total = time.time()
    print("Total solve time:", end_time_total - start_time_total)

    if scale_rc:
        average_state["primal"] = average_state["primal"] * cs_pc * cs_ruiz
        average_state["dual_eq"] = (
            average_state["dual_eq"]
            * rs_pc[: lp.A_eq.shape[0]]
            * rs_ruiz[: lp.A_eq.shape[0]]
        )
        average_state["dual_ineq"] = (
            average_state["dual_ineq"]
            * rs_pc[lp.A_eq.shape[0] :]
            * rs_ruiz[lp.A_eq.shape[0] :]
        )

    return average_state


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
