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

    A_ineq = sp.vstack(matrices, format="csc")
    b_ineq = np.concatenate(vectors)

    return jl.LP(c, A_eq, b_eq, A_ineq, b_ineq, lower_bounds, upper_bounds)
