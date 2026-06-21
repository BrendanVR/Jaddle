import jax.numpy as jnp
from typing import NamedTuple
import optax
from typing import Any, Callable, NamedTuple, Sequence, Union
import jax
import numpy as np

ScheduleLike = Union[float, Callable[[jnp.ndarray], jnp.ndarray]]


# %%
# Basic Types
class SaddleState(NamedTuple):
    primal: jnp.ndarray
    dual_ineq: jnp.ndarray
    dual_eq: jnp.ndarray


class CP:
    def __init__(
        self,
        num_variables,
        objective,
        constraints_eq,
        constraints_ineq,
        lower_bounds,
        upper_bounds,
        dual_bound=None,
    ):
        self.num_variables = num_variables
        self.objective = objective
        self.constraints_eq = constraints_eq
        self.constraints_ineq = constraints_ineq
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.dual_bound = dual_bound

    def initial_primal_solution(self):
        return jnp.zeros(self.num_variables)

    def num_eq_constraints(self):
        return len(self.constraints_eq(self.initial_primal_solution()))

    def num_ineq_constraints(self):
        return len(self.constraints_ineq(self.initial_primal_solution()))

    def num_constraints(self):
        return self.num_eq_constraints() + self.num_ineq_constraints()

    def ineq_slack(self, x):
        return jnp.max(jnp.maximum(self.constraints_ineq(x), 0.0))

    def eq_slack(self, x):
        return jnp.max(jnp.abs(self.constraints_eq(x)))

    def complementarity_slack(self, x, dual_ineq):
        return dual_ineq * (self.constraints_ineq(x))

    def initial_solution(self):
        return SaddleState(
            primal=jnp.zeros(self.num_variables),
            dual_ineq=jnp.zeros(self.num_ineq_constraints()),
            dual_eq=jnp.zeros(self.num_eq_constraints()),
        )


def _least_norm_dual(A_eq, A_ineq, c, n_eq, n_ineq):
    """Least-norm dual: min ‖y‖ s.t. Aᵀy ≈ -c (LP stationarity condition).

    Returns (dual_eq, dual_ineq). ineq duals are clipped to ≥ 0.
    Works with both dense JAX arrays and BCOO sparse matrices via scipy lsqr.
    Falls back to zeros on failure.
    """
    import scipy.sparse
    import scipy.sparse.linalg

    def _to_scipy_csr(mat, n_rows, n_cols):
        if hasattr(mat, "indices"):  # BCOO
            mat = mat.sum_duplicates()
            rows = np.asarray(mat.indices[:, 0])
            cols = np.asarray(mat.indices[:, 1])
            data = np.asarray(mat.data)
            return scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))
        return scipy.sparse.csr_matrix(np.asarray(mat))

    try:
        n_vars = int(c.shape[0])
        A_eq_sp = _to_scipy_csr(A_eq, n_eq, n_vars)
        A_ineq_sp = _to_scipy_csr(A_ineq, n_ineq, n_vars)
        A = scipy.sparse.vstack([A_eq_sp, A_ineq_sp], format="csr")
        rhs = -np.asarray(c)
        y, *_ = scipy.sparse.linalg.lsqr(A.T, rhs)
        y = jnp.array(y)
        return y[:n_eq], jnp.maximum(y[n_eq:], 0.0)
    except Exception:
        return jnp.zeros(n_eq), jnp.zeros(n_ineq)


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

    def diff_eq_slack(self, x):
        return self.A_eq @ x - self.b_eq

    def complementarity_slack(self, x, dual_ineq):
        return (dual_ineq * (self.A_ineq @ x - self.b_ineq)).sum()

    def initial_solution(self):
        primal = optax.projections.projection_box(
            jnp.zeros(self.num_variables()),
            self.lower_bounds,
            self.upper_bounds,
        )
        dual_eq, dual_ineq = _least_norm_dual(
            jnp.array(self.A_eq),
            jnp.array(self.A_ineq),
            jnp.array(self.c),
            self.num_eq_constraints(),
            self.num_ineq_constraints(),
        )
        return SaddleState(primal=primal, dual_ineq=dual_ineq, dual_eq=dual_eq)


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
        # Fused [A_eq; A_ineq] for 2-matvec gradient computation.
        #
        # Both A and Aᵀ are kept as BCOO. BCSR was benchmarked here (Aᵀ
        # materialised as its own row-major CSR, not a lazy `.T`) and was MUCH
        # slower than BCOO on this workload — do not switch back to BCSR without
        # re-benchmarking; BCOO's matvec wins for these matrices/precision/device.
        #
        # Aᵀ is stored as an explicit transposed BCOO (column-swapped indices)
        # rather than the lazy `self.A.T`, so `Aᵀ @ y` runs its own matvec
        # instead of the transposed-operand code path.
        import jax.experimental.sparse as _jsp

        self.n_eq = A_eq.shape[0]
        A_bcoo = _jsp.bcoo_concatenate([A_eq, A_ineq], dimension=0)
        A_T_bcoo = _jsp.BCOO(
            (A_bcoo.data, A_bcoo.indices[:, ::-1]),
            shape=A_bcoo.shape[::-1],
        )

        # Column-swapping A's indices destroys row-major sort order, leaving A_T
        # with indices_sorted=False/unique_indices=False. JAX's BCOO matvec has a
        # faster kernel for sorted+unique indices (and the unsorted path can't
        # skip duplicate accumulation), so sort A_T once at build time. The LP
        # has no duplicate (row, col) entries, so unique_indices is safe to set.
        A_bcoo = A_bcoo.sort_indices()
        A_bcoo.unique_indices = True

        A_T_bcoo = A_T_bcoo.sort_indices()
        A_T_bcoo.unique_indices = True

        self.A = A_bcoo
        self.A_T = A_T_bcoo
        self.b = jnp.concatenate([b_eq, b_ineq])

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
        # print("---> Computing least-norm dual initial solution...")
        # print("----------------------------------------------")
        # primal = optax.projections.projection_box(
        #     jnp.zeros(self.num_variables()),
        #     self.lower_bounds,
        #     self.upper_bounds,
        # )
        primal = jnp.zeros(self.num_variables())
        # n_eq = self.num_eq_constraints()
        # n_ineq = self.num_ineq_constraints()
        # dual_eq, dual_ineq = _least_norm_dual(
        #     self.A_eq, self.A_ineq, self.c, n_eq, n_ineq
        # )
        dual_eq = jnp.zeros(self.num_eq_constraints())
        dual_ineq = jnp.zeros(self.num_ineq_constraints())
        return SaddleState(primal=primal, dual_ineq=dual_ineq, dual_eq=dual_eq)
