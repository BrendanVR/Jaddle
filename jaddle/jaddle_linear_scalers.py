import jaddle.jaddle_linear as jl
import jax.experimental.sparse as jsp
import numpy as np
import scipy.sparse as sp
from typing import NamedTuple


class Scaled_LP(NamedTuple):
    lp: jl.LP
    row_scale: np.ndarray
    col_scale: np.ndarray


class Scaled_Solution(NamedTuple):
    state: jl.SaddleState
    row_scale: np.ndarray
    col_scale: np.ndarray


def unscale_solution(scaled_sol: Scaled_Solution) -> jl.SaddleState:
    primal_unscaled = scaled_sol.state.primal * scaled_sol.col_scale
    dual_eq_unscaled = (
        scaled_sol.state.dual_eq * scaled_sol.row_scale[: len(scaled_sol.state.dual_eq)]
    )
    dual_ineq_unscaled = (
        scaled_sol.state.dual_ineq
        * scaled_sol.row_scale[len(scaled_sol.state.dual_eq) :]
    )
    return jl.SaddleState(primal_unscaled, dual_ineq_unscaled, dual_eq_unscaled)


def __convert_to_scipy(jsp_mat: jsp.BCOO) -> sp.csc_matrix:
    data = np.array(jsp_mat.data)
    indices = np.array(jsp_mat.indices)
    row, col = indices[:, 0], indices[:, 1]

    return sp.csc_matrix((data, (row, col)), shape=jsp_mat.shape)


def ruiz_scaling(lp: jl.LP, max_iter=10, threshold=1e-8, clip_bounds=(1e-6, 1e6)):
    """
    Applies Ruiz scaling to an jl.LP in standard form with sparse matrices:
        min c^T x
        s.t. A_eq x = b_eq
             A_ineq x <= b_ineq
             lower_bounds <= x <= upper_bounds
    Returns scaled jl.LP, row_scaling, col_scaling
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

    lp_scaled = jl.LP(
        c_scaled,
        A_eq_scaled,
        b_eq_scaled,
        A_ineq_scaled,
        b_ineq_scaled,
        lower_bounds_scaled,
        upper_bounds_scaled,
    )

    lp_scaled = jl.to_jaddle_sparse(lp_scaled)

    return Scaled_LP(lp_scaled, row_scale, col_scale)


def pc_scaling(lp: jl.LP, max_iter=1, threshold=1e-8, clip_bounds=(1e-6, 1e6)):
    """
    Applies PC scaling to an jl.LP in standard form with sparse matrices:
        min c^T x
        s.t. A_eq x = b_eq
                A_ineq x <= b_ineq
                lower_bounds <= x <= upper_bounds
    Returns scaled jl.LP, row_scaling, col_scaling
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

    lp_scaled = jl.LP(
        c_scaled,
        A_eq_scaled,
        b_eq_scaled,
        A_ineq_scaled,
        b_ineq_scaled,
        lower_bounds_scaled,
        upper_bounds_scaled,
    )

    lp_scaled = jl.to_jaddle_sparse(lp_scaled)

    return Scaled_LP(lp_scaled, row_scale, col_scale)
