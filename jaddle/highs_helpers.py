import numpy as np
import scipy.sparse as sp
import highspy as hspy
import jaddle.jaddle_linear as jl
import jax.numpy as jnp
import jax.experimental.sparse as jsp


def highs_to_standard_form_sparse(lp):
    """
    Converts a HighsLp object to standard form matrices:
        min c^T x
        s.t. A_eq x = b_eq
             A_ineq x <= b_ineq
             x >= lower_bounds
             x <= upper_bounds
    Returns: c, A_eq, b_eq, A_ineq, b_ineq, lower_bounds, upper_bounds
    """
    c = np.array(lp.col_cost_, dtype=np.float32)
    lower_bounds = np.array(lp.col_lower_, dtype=np.float32)
    upper_bounds = np.array(lp.col_upper_, dtype=np.float32)

    # Build A matrix from sparse representation
    num_row = lp.a_matrix_.num_row_
    num_col = lp.a_matrix_.num_col_
    A = sp.csc_matrix(
        (lp.a_matrix_.value_, lp.a_matrix_.index_, lp.a_matrix_.start_),
        shape=(num_row, num_col),
        dtype=np.float32,
    )

    row_lower = np.array(lp.row_lower_, dtype=np.float32)
    row_upper = np.array(lp.row_upper_, dtype=np.float32)

    # Equality constraints: row_lower == row_upper
    eq_mask = np.equal(row_lower, row_upper)
    A_eq = A[eq_mask, :].tocsc()
    b_eq = row_lower[eq_mask]

    # Inequality constraints
    ineq_mask = ~eq_mask
    A_ineq_rows = A[ineq_mask, :]
    row_lower_ineq = row_lower[ineq_mask]
    row_upper_ineq = row_upper[ineq_mask]

    finite_upper = np.isfinite(row_upper_ineq)
    finite_lower = np.isfinite(row_lower_ineq)

    # Build inequality matrices more efficiently
    matrices = []
    vectors = []

    if finite_upper.any():
        matrices.append(A_ineq_rows[finite_upper].tocsc())
        vectors.append(row_upper_ineq[finite_upper])

    if finite_lower.any():
        matrices.append((-A_ineq_rows[finite_lower]).tocsc())
        vectors.append(-row_lower_ineq[finite_lower])

    # Add variable bounds
    # finite_upper_bounds = np.isfinite(upper_bounds)
    # finite_lower_bounds = np.isfinite(lower_bounds)

    # if finite_upper_bounds.any():
    #     I_upper = sp.csc_matrix(
    #         (
    #             np.ones(np.sum(finite_upper_bounds)),
    #             (
    #                 np.arange(np.sum(finite_upper_bounds)),
    #                 np.where(finite_upper_bounds)[0],
    #             ),
    #         ),
    #         shape=(np.sum(finite_upper_bounds), num_col),
    #         dtype=np.float32,
    #     )
    #     matrices.append(I_upper)
    #     vectors.append(upper_bounds[finite_upper_bounds])

    # if finite_lower_bounds.any():
    #     I_lower = sp.csc_matrix(
    #         (
    #             np.ones(np.sum(finite_lower_bounds)),
    #             (
    #                 np.arange(np.sum(finite_lower_bounds)),
    #                 np.where(finite_lower_bounds)[0],
    #             ),
    #         ),
    #         shape=(np.sum(finite_lower_bounds), num_col),
    #         dtype=np.float32,
    #     )
    #     matrices.append(-I_lower)
    #     vectors.append(-lower_bounds[finite_lower_bounds])

    A_ineq = sp.vstack(matrices, format="csc")
    b_ineq = np.concatenate(vectors)

    return jl.LP(c, A_eq, b_eq, A_ineq, b_ineq, lower_bounds, upper_bounds)
