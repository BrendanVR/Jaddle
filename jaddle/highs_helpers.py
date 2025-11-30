import numpy as np
import scipy.sparse as sp
import highspy as hspy
import jaddle.jaddle_linear as jl
import jax.numpy as jnp


def highs_from_standard_form(lp: jl.LP):
    highs_lp = hspy.HighsLp()
    highs_lp.num_col_ = lp.A_eq.shape[1]
    highs_lp.num_row_ = lp.A_eq.shape[0] + lp.A_ineq.shape[0]
    highs_lp.col_cost_ = np.array(lp.c)
    highs_lp.row_lower_ = np.array(lp.b_eq.tolist() + [-np.inf] * lp.A_ineq.shape[0])
    highs_lp.row_upper_ = np.array(lp.b_eq.tolist() + lp.b_ineq.tolist())
    highs_lp.col_lower_ = np.array(lp.lower_bounds)
    highs_lp.col_upper_ = np.array(lp.upper_bounds)
    A = np.vstack([lp.A_eq, lp.A_ineq])
    A = sp.csc_matrix(A)
    highs_lp.a_matrix_.num_row_ = A.shape[0]
    highs_lp.a_matrix_.num_col_ = A.shape[1]
    highs_lp.a_matrix_.start_ = A.indptr
    highs_lp.a_matrix_.index_ = A.indices
    highs_lp.a_matrix_.value_ = A.data

    return highs_lp


def highs_from_standard_form_sparse(lp: jl.LP):
    highs_lp = hspy.HighsLp()
    highs_lp.num_col_ = lp.A_eq.shape[1]
    highs_lp.num_row_ = lp.A_eq.shape[0] + lp.A_ineq.shape[0]
    highs_lp.col_cost_ = np.array(lp.c)
    highs_lp.row_lower_ = np.array(lp.b_eq.tolist() + [-np.inf] * lp.A_ineq.shape[0])
    highs_lp.row_upper_ = np.array(lp.b_eq.tolist() + lp.b_ineq.tolist())
    highs_lp.col_lower_ = np.array(lp.lower_bounds)
    highs_lp.col_upper_ = np.array(lp.upper_bounds)
    A = sp.vstack([lp.A_eq, lp.A_ineq]).tocsc()
    highs_lp.a_matrix_.num_row_ = A.shape[0]
    highs_lp.a_matrix_.num_col_ = A.shape[1]
    highs_lp.a_matrix_.start_ = A.indptr
    highs_lp.a_matrix_.index_ = A.indices
    highs_lp.a_matrix_.value_ = A.data

    return highs_lp


def highs_to_standard_form(lp):
    """
    Converts a HighsLp object to standard form matrices:
        min c^T x
        s.t. A_eq x = b_eq
             A_ineq x <= b_ineq
             x >= lower_bounds
             x <= upper_bounds
    Returns: c, A_eq, b_eq, A_ineq, b_ineq, lower_bounds, upper_bounds
    """
    c = np.array(lp.col_cost_)
    lower_bounds = np.array(lp.col_lower_)
    upper_bounds = np.array(lp.col_upper_)
    row_lower = np.array(lp.row_lower_)
    row_upper = np.array(lp.row_upper_)

    # Build A matrix from sparse representation
    num_row = lp.a_matrix_.num_row_
    num_col = lp.a_matrix_.num_col_
    A = sp.csc_matrix(
        (lp.a_matrix_.value_, lp.a_matrix_.index_, lp.a_matrix_.start_),
        shape=(num_row, num_col),
    ).toarray()

    # Equality constraints: row_lower == row_upper
    eq_mask = np.isclose(row_lower, row_upper)
    A_eq = A[eq_mask, :]
    b_eq = row_lower[eq_mask]

    # Inequality constraints: row_lower != row_upper
    ineq_mask = ~eq_mask

    # Only include constraints if the bound is finite
    finite_upper = np.isfinite(row_upper[ineq_mask])
    finite_lower = np.isfinite(row_lower[ineq_mask])

    # Upper bound constraints (A_row x <= row_upper)
    A_ineq_le = A[ineq_mask, :][finite_upper]
    b_ineq_le = row_upper[ineq_mask][finite_upper]

    # Lower bound constraints (A_row x >= row_lower) -> -A_row x <= -row_lower
    A_ineq_ge = -A[ineq_mask, :][finite_lower]
    b_ineq_ge = -row_lower[ineq_mask][finite_lower]

    A_ineq = np.vstack([A_ineq_le, A_ineq_ge])
    b_ineq = np.concatenate([b_ineq_le, b_ineq_ge])

    lp = jl.LP(c, A_eq, b_eq, A_ineq, b_ineq, lower_bounds, upper_bounds)
    return lp


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
    c = np.array(lp.col_cost_)
    lower_bounds = np.array(lp.col_lower_)
    upper_bounds = np.array(lp.col_upper_)
    row_lower = np.array(lp.row_lower_)
    row_upper = np.array(lp.row_upper_)

    # Build A matrix from sparse representation
    num_row = lp.a_matrix_.num_row_
    num_col = lp.a_matrix_.num_col_
    A = sp.csc_matrix(
        (lp.a_matrix_.value_, lp.a_matrix_.index_, lp.a_matrix_.start_),
        shape=(num_row, num_col),
    )

    # Equality constraints: row_lower == row_upper
    eq_mask = np.isclose(row_lower, row_upper)
    A_eq = A[eq_mask, :]
    b_eq = row_lower[eq_mask]

    # Inequality constraints: row_lower != row_upper
    ineq_mask = ~eq_mask

    # Only include constraints if the bound is finite
    finite_upper = np.isfinite(row_upper[ineq_mask])
    finite_lower = np.isfinite(row_lower[ineq_mask])

    # Upper bound constraints (A_row x <= row_upper)
    A_ineq_le = A[ineq_mask, :][finite_upper]
    b_ineq_le = row_upper[ineq_mask][finite_upper]

    # Lower bound constraints (A_row x >= row_lower) -> -A_row x <= -row_lower
    A_ineq_ge = -A[ineq_mask, :][finite_lower]
    b_ineq_ge = -row_lower[ineq_mask][finite_lower]

    A_ineq = sp.vstack([A_ineq_le, A_ineq_ge]).tocsc()
    b_ineq = np.concatenate([b_ineq_le, b_ineq_ge])

    lp = jl.LP(c, A_eq, b_eq, A_ineq, b_ineq, lower_bounds, upper_bounds)
    return lp


def highs_linear_solver(
    lp,
    feasibility_tolerance=1e-7,
    method="simplex",
    warm_start=None,
    dual_warmstart=False,
):
    highs = hspy.Highs()
    highs.passModel(lp)
    highs.setOptionValue("solver", method)
    highs.setOptionValue("presolve", "off")
    highs.setOptionValue("primal_feasibility_tolerance", feasibility_tolerance)
    highs.setOptionValue("dual_feasibility_tolerance", feasibility_tolerance)
    highs.setOptionValue("optimality_tolerance", 1e-4)

    if warm_start is not None:
        solution = hspy.HighsSolution()
        solution.col_value = np.array(warm_start["primal"])
        if dual_warmstart:
            solution.row_dual = np.array(
                warm_start["dual_eq"].tolist() + warm_start["dual_ineq"].tolist()
            )
        highs.setSolution(solution)

    highs.run()
    highs_solution = highs.getSolution()
    highs_primal = jnp.array(highs_solution.col_value)

    return highs_primal


def create_highs_solution(primal):
    solution = hspy.HighsSolution()
    solution.col_value = np.array(primal)
    return solution
