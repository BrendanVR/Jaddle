import numpy as np
import scipy.sparse as sp
import highspy as hspy
import jaddle.jaddle_linear as jl
import jax.numpy as jnp
import jax.experimental.sparse as jsp


def highs_to_standard_form_sparse(lp: hspy.HighsLp) -> jl.LP:
    """
    Converts a HighsLp object to standard form matrices:
        min c^T x
        s.t. A_eq x = b_eq
             A_ineq x <= b_ineq
             x >= lower_bounds
             x <= upper_bounds
    Returns: jl.LP
    """

    c = np.array(lp.col_cost_, dtype=np.float64)
    lower_bounds = np.array(lp.col_lower_, dtype=np.float64)
    upper_bounds = np.array(lp.col_upper_, dtype=np.float64)

    # Build A matrix from sparse representation
    num_row = lp.a_matrix_.num_row_
    num_col = lp.a_matrix_.num_col_
    A = sp.csc_matrix(
        (lp.a_matrix_.value_, lp.a_matrix_.index_, lp.a_matrix_.start_),
        shape=(num_row, num_col),
        dtype=np.float64,
    )

    row_lower = np.array(lp.row_lower_, dtype=np.float64)
    row_upper = np.array(lp.row_upper_, dtype=np.float64)

    # Equality constraints: use a numeric tolerance and require finite bounds
    eps = 1e-8
    finite_lower_all = np.isfinite(row_lower)
    finite_upper_all = np.isfinite(row_upper)
    eq_mask = (
        finite_lower_all & finite_upper_all & (np.abs(row_lower - row_upper) <= eps)
    )

    A_eq = A[eq_mask, :].tocsc().astype(np.float64)
    b_eq = row_lower[eq_mask].astype(np.float64)

    # Inequality constraints (rows not treated as equalities)
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
        matrices.append(A_ineq_rows[finite_upper].tocsc().astype(np.float64))
        vectors.append(row_upper_ineq[finite_upper].astype(np.float64))

    if finite_lower.any():
        matrices.append((-A_ineq_rows[finite_lower]).tocsc().astype(np.float64))
        vectors.append((-row_lower_ineq[finite_lower]).astype(np.float64))

    if len(matrices) > 0:
        A_ineq = sp.vstack(matrices, format="csc")
        b_ineq = np.concatenate(vectors)
    else:
        # No inequality rows: return empty (0 x num_col) matrix and empty RHS
        A_ineq = sp.csc_matrix((0, num_col), dtype=np.float64)
        b_ineq = np.empty(0, dtype=np.float64)

    return jl.LP(c, A_eq, b_eq, A_ineq, b_ineq, lower_bounds, upper_bounds)
