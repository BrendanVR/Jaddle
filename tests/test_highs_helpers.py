import numpy as np
import scipy.sparse as sp

from jaddle.highs_helpers import highs_to_standard_form_sparse


class MockAMatrix:
    def __init__(self, num_row, num_col, data, indices, indptr):
        self.num_row_ = num_row
        self.num_col_ = num_col
        self.value_ = np.array(data, dtype=np.float32)
        self.index_ = np.array(indices, dtype=np.int32)
        self.start_ = np.array(indptr, dtype=np.int32)


class MockHighsLp:
    def __init__(
        self,
        col_cost,
        col_lower,
        col_upper,
        row_lower,
        row_upper,
        a_data,
        a_indices,
        a_indptr,
        num_row,
        num_col,
    ):
        self.col_cost_ = np.array(col_cost, dtype=np.float32)
        self.col_lower_ = np.array(col_lower, dtype=np.float32)
        self.col_upper_ = np.array(col_upper, dtype=np.float32)
        self.row_lower_ = np.array(row_lower, dtype=np.float32)
        self.row_upper_ = np.array(row_upper, dtype=np.float32)
        self.a_matrix_ = MockAMatrix(num_row, num_col, a_data, a_indices, a_indptr)


def test_equality_within_eps():
    # One row where lower and upper differ by less than eps -> equality
    num_row = 1
    num_col = 1
    a_data = [1.0]
    a_indices = [0]
    a_indptr = [0, 1]

    row_lower = [1.0]
    row_upper = [1.0 + 1e-9]

    lp = MockHighsLp(
        col_cost=[0.0],
        col_lower=[0.0],
        col_upper=[np.inf],
        row_lower=row_lower,
        row_upper=row_upper,
        a_data=a_data,
        a_indices=a_indices,
        a_indptr=a_indptr,
        num_row=num_row,
        num_col=num_col,
    )

    jl_lp = highs_to_standard_form_sparse(lp)

    assert jl_lp.A_eq.shape[0] == 1
    assert jl_lp.b_eq.size == 1
    assert jl_lp.A_ineq.shape[0] == 0
    assert jl_lp.b_ineq.size == 0


def test_no_inequalities():
    # All rows are equalities; ensure A_ineq is empty shaped (0, ncols)
    num_row = 2
    num_col = 2
    # Identity matrix
    a_data = [1.0, 1.0, 1.0, 1.0]
    a_indices = [0, 1, 0, 1]
    a_indptr = [0, 2, 4]

    row_lower = [0.0, 5.0]
    row_upper = [0.0, 5.0]

    lp = MockHighsLp(
        col_cost=[0.0, 0.0],
        col_lower=[-np.inf, -np.inf],
        col_upper=[np.inf, np.inf],
        row_lower=row_lower,
        row_upper=row_upper,
        a_data=a_data,
        a_indices=a_indices,
        a_indptr=a_indptr,
        num_row=num_row,
        num_col=num_col,
    )

    jl_lp = highs_to_standard_form_sparse(lp)

    assert jl_lp.A_eq.shape[0] == 2
    assert jl_lp.A_ineq.shape == (0, num_col)
    assert jl_lp.b_ineq.size == 0


def test_large_difference_becomes_inequalities():
    # Row lower and upper differ significantly -> converted to inequalities
    num_row = 1
    num_col = 1
    a_data = [2.0]
    a_indices = [0]
    a_indptr = [0, 1]

    row_lower = [1.0]
    row_upper = [3.0]

    lp = MockHighsLp(
        col_cost=[0.0],
        col_lower=[-np.inf],
        col_upper=[np.inf],
        row_lower=row_lower,
        row_upper=row_upper,
        a_data=a_data,
        a_indices=a_indices,
        a_indptr=a_indptr,
        num_row=num_row,
        num_col=num_col,
    )

    jl_lp = highs_to_standard_form_sparse(lp)

    # Should produce two inequality rows: A x <= upper and -A x <= -lower
    assert jl_lp.A_eq.shape[0] == 0
    assert jl_lp.A_ineq.shape[0] == 2
    assert jl_lp.b_ineq.size == 2
