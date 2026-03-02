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
from jaddle.jaddle_basic_types import LP, SaddleState, HedgeSaddleState
import jaddle.jaddle_optimisers as jo

np.set_printoptions(precision=2, suppress=True)


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

    def projection_primal(primal_state):
        return projection_box(primal_state, lp.lower_bounds, lp.upper_bounds)

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
            state = jax.tree_util.tree_map(lambda s, u: s + u, state, updates)
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
    optimiser,
    max_epochs=None,
    initial_solution=None,
    initial_opt_state=None,
    iterations_per_epoch=int(1e4),
    dual_damping_ineq=0.0,
    dual_damping_eq=0.0,
    primal_damping=0.0,
    progress_tolerance=1e-2,
    constraint_tolerance=1e-3,
    complementarity_tolerance=1e-3,
    weight_function=lambda _: 1.0,
    verbose=False,
    average=True,
    scale=None,
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
    print("====Starting Solve====")
    print("----------------------------------------------")

    if initial_solution is None:
        initial_solution = lp.initial_solution()

    if scale == "ruiz":
        lp, row_scale, col_scale = ruiz_scaling(lp)
        print("Applied Ruiz scaling to the LP.")
        print("----------------------------------------------")

    elif scale == "pc":
        lp, row_scale, col_scale = pc_scaling(lp)
        print("Applied PC scaling to the LP.")
        print("----------------------------------------------")

    elif scale == "ruiz+pc":
        lp, row_scale_ruiz, col_scale_ruiz = ruiz_scaling(lp)
        lp, row_scale_pc, col_scale_pc = pc_scaling(lp)

        row_scale, col_scale = (
            row_scale_ruiz * row_scale_pc,
            col_scale_ruiz * col_scale_pc,
        )

        print("Applied combined Ruiz + PC scaling to the LP.")
        print("----------------------------------------------")

    else:
        row_scale = np.ones(lp.A_eq.shape[0] + lp.A_ineq.shape[0])
        col_scale = np.ones(lp.c.shape[0])

    row_scale_ineq = row_scale[len(lp.b_eq) :]
    row_scale_eq = row_scale[: len(lp.b_eq)]

    # Convert to jax arrays for use inside jitted functions
    jnp_row_scale_ineq = jnp.array(row_scale_ineq)
    jnp_row_scale_eq = jnp.array(row_scale_eq)

    initial_solution = SaddleState(
        primal=initial_solution.primal / col_scale,
        dual_ineq=initial_solution.dual_ineq / jnp_row_scale_ineq,
        dual_eq=initial_solution.dual_eq / jnp_row_scale_eq,
    )

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

        # Unscale constraint violations to original space
        grad_dual_ineq_unscaled = grad_dual_ineq / jnp_row_scale_ineq
        grad_dual_eq_unscaled = grad_dual_eq / jnp_row_scale_eq

        projected_primal = projection_box(
            average_state.primal - grad_primal,
            lp.lower_bounds,
            lp.upper_bounds,
        )
        projected_gradient_residual = average_state.primal - projected_primal
        primal_grad_norm = jnp.max(jnp.abs(projected_gradient_residual))

        ineq_violations = jnp.maximum(grad_dual_ineq_unscaled, 0.0)
        max_ineq_violation = jnp.max(ineq_violations)

        eq_violations = jnp.abs(grad_dual_eq_unscaled)
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
            average,
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
    is_converged = True

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
                average,
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
    print(f"Objective: {lp.objective(output.primal):.5e}")
    print("----------------------------------------------")

    if scale in ["ruiz", "pc", "ruiz+pc"]:
        output = SaddleState(
            primal=output.primal * col_scale,
            dual_ineq=output.dual_ineq * jnp_row_scale_ineq,
            dual_eq=output.dual_eq * jnp_row_scale_eq,
        )
    if output_opt_state:
        return output, is_converged, opt_state
    else:
        return output, is_converged


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
        num_nnz_A_eq = lp.A_eq.data.size
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
        num_nnz_A_ineq = lp.A_ineq.data.size
    else:
        min_A_ineq = None
        max_A_ineq = None
        min_b_ineq = None
        max_b_ineq = None

    min_c = np.min(lp.c)
    max_c = np.max(lp.c)

    print("--------------------------------")
    print("LP Summary Statistics")
    print("--------------------------------")
    print(f"Number of variables: {num_vars}")
    print(f"Number of equality constraints: {num_eq}")
    print(f"Number of inequality constraints: {num_ineq}")
    print(f"Number of nonzeros in A_eq: {num_nnz_A_eq}")
    print(f"Number of nonzeros in A_ineq: {num_nnz_A_ineq}")
    print(f"[Min, Max] of c: [{min_c}, {max_c}]")
    print(f"[Min, Max] of A_eq: [{min_A_eq}, {max_A_eq}]")
    print(f"[Min, Max] of b_eq: [{min_b_eq}, {max_b_eq}]")
    print(f"[Min, Max] of A_ineq: [{min_A_ineq}, {max_A_ineq}]")
    print(f"[Min, Max] of b_ineq: [{min_b_ineq}, {max_b_ineq}]")
    print("-----------------------------------------------")


# %%


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


# %%
