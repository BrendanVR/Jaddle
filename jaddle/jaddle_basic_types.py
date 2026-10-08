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


class JaddleCP:
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
        return jnp.max(jnp.maximum(self.A_ineq @ x - self.b_ineq, 0.0), initial=0.0)

    def eq_slack(self, x):
        return jnp.max(jnp.abs(self.A_eq @ x - self.b_eq), initial=0.0)

    def diff_eq_slack(self, x):
        return self.A_eq @ x - self.b_eq

    def complementarity_slack(self, x, dual_ineq):
        return (dual_ineq * (self.A_ineq @ x - self.b_ineq)).sum()


def _row_major_order(rows, cols, shape):
    """Permutation putting the unique COO entries ``(rows, cols)`` in row-major
    order, via scipy's O(nnz) CSR conversion (a counting sort) on the host."""
    import scipy.sparse as sp

    perm = sp.csr_matrix(
        (np.arange(rows.size, dtype=np.int64), (rows, cols)), shape=shape
    )
    perm.sort_indices()
    return perm.data


def scipy_to_bcoo(A, dtype):
    """Sorted, deduplicated BCOO of the scipy matrix ``A`` with ``dtype`` data.

    Accepts any scipy sparse format. Sorting, deduplication and the dtype cast
    all happen on the host, so building the device array compiles nothing; the
    cast goes straight from the scipy width to ``dtype`` (float16 included).
    """
    import scipy.sparse as sp
    import jax.experimental.sparse as _jsp

    np_float = np.float64 if jnp.dtype(dtype).itemsize == 8 else np.float32
    # sorted_indices() copies, so sum_duplicates() never touches the caller's A.
    A = sp.csr_matrix(A, dtype=np_float).sorted_indices()
    A.sum_duplicates()
    rows = np.repeat(np.arange(A.shape[0], dtype=np.int32), np.diff(A.indptr))
    indices = np.column_stack([rows, A.indices.astype(np.int32)])
    # jax.device_put, not jnp.asarray: the latter compiles a per-shape
    # jit(stage) for every numpy upload.
    return _jsp.BCOO(
        (jax.device_put(A.data.astype(jnp.dtype(dtype))), jax.device_put(indices)),
        shape=A.shape,
        indices_sorted=True,
        unique_indices=True,
    )


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
        #
        # JAX's BCOO matvec has a faster kernel for sorted+unique indices (and
        # the unsorted path can't skip duplicate accumulation), so both are
        # stored row-major sorted. The LP has no duplicate (row, col) entries,
        # so unique_indices is safe to set. The sort is done on the HOST:
        # BCOO.sort_indices() is a device lax.sort compiled per shape, which was
        # ~2-3 s of every solve's setup (stp3d) for a permutation scipy's O(nnz)
        # counting sort finds in milliseconds.
        import jax.experimental.sparse as _jsp

        self.n_eq = A_eq.shape[0]
        m, n = A_eq.shape[0] + A_ineq.shape[0], A_eq.shape[1]
        eq_idx, ineq_idx = np.asarray(A_eq.indices), np.asarray(A_ineq.indices)
        idx = np.concatenate([eq_idx, ineq_idx + np.array([self.n_eq, 0], eq_idx.dtype)])
        data = np.concatenate([np.asarray(A_eq.data), np.asarray(A_ineq.data)])

        # Stacking two row-sorted blocks (ineq rows offset below eq) is sorted.
        if not (A_eq.indices_sorted and A_ineq.indices_sorted):
            order = _row_major_order(idx[:, 0], idx[:, 1], (m, n))
            idx, data = idx[order], data[order]
        order_T = _row_major_order(idx[:, 1], idx[:, 0], (n, m))

        def bcoo(d, i, shape):
            return _jsp.BCOO(
                (jax.device_put(d), jax.device_put(i)),
                shape=shape,
                indices_sorted=True,
                unique_indices=True,
            )

        self.A = bcoo(data, idx, (m, n))
        self.A_T = bcoo(
            data[order_T], np.ascontiguousarray(idx[order_T][:, ::-1]), (n, m)
        )
        self.b = jax.device_put(np.concatenate([np.asarray(b_eq), np.asarray(b_ineq)]))

    @classmethod
    def from_scipy(cls, c, A_eq, b_eq, A_ineq, b_ineq, lower_bounds, upper_bounds):
        """Build a ``JaddleLP`` directly from scipy sparse blocks + numpy vectors.

        Mirrors ``jaddle_linear.to_jaddle_sparse``: resolves the active precision
        profile (float64 under x64, else float32; float16 lives only in JAX since
        scipy cannot hold it), converts each constraint block to a sorted,
        deduplicated BCOO, and casts the data array to the profile dtype. This is
        the JAX-native entry point — callers (e.g. ``highs_to_standard_form_sparse``)
        no longer route through a scipy ``LP``.
        """
        from jaddle.jaddle_optimisers import jaddle_dtype

        float_dtype = jaddle_dtype()
        if float_dtype == jnp.float64 and not jax.config.jax_enable_x64:
            float_dtype = jnp.float32

        return cls(
            jnp.asarray(c, dtype=float_dtype),
            scipy_to_bcoo(A_eq, float_dtype),
            jnp.asarray(b_eq, dtype=float_dtype),
            scipy_to_bcoo(A_ineq, float_dtype),
            jnp.asarray(b_ineq, dtype=float_dtype),
            jnp.asarray(lower_bounds, dtype=float_dtype),
            jnp.asarray(upper_bounds, dtype=float_dtype),
        )

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
        return jnp.max(jnp.maximum(self.A_ineq @ x - self.b_ineq, 0.0), initial=0.0)

    def eq_slack(self, x):
        return jnp.max(jnp.abs(self.A_eq @ x - self.b_eq), initial=0.0)

    def diff_eq_slack(self, x):
        return self.A_eq @ x - self.b_eq

    def complementarity_slack(self, x, dual_ineq):
        return (dual_ineq * (self.A_ineq @ x - self.b_ineq)).sum()

    def initial_solution(self):
        """Default start point: the box-projected zero primal with zero duals
        (the PDLP default). Built on the host and uploaded, so it compiles
        nothing."""
        lb, ub = np.asarray(self.lower_bounds), np.asarray(self.upper_bounds)
        dtype = jnp.result_type(float, lb.dtype)
        return SaddleState(
            primal=jax.device_put(np.clip(np.zeros(lb.shape, dtype), lb, ub)),
            dual_ineq=jax.device_put(np.zeros(self.num_ineq_constraints(), dtype)),
            dual_eq=jax.device_put(np.zeros(self.num_eq_constraints(), dtype)),
        )

    def to_scipy(self) -> "LP":
        """Materialise a scipy-backed ``LP`` (CSC matrices, numpy vectors).

        ``JaddleLP`` is the device-side representation the saddle solver iterates
        on; the scaling, primal/dual polish, crossover and eq-projection LU stages
        run host-side on scipy sparse (sparse direct solve, ``lsmr``, boolean
        row-slicing — none of which JAX provides). Those stages call this to get
        the scipy view they need; the solver never does.
        """
        import scipy.sparse as sp

        def _to_csc(bcoo):
            data = np.asarray(bcoo.data)
            idx = np.asarray(bcoo.indices)
            return sp.csc_matrix(
                (data, (idx[:, 0], idx[:, 1])),
                shape=bcoo.shape,
                dtype=np.float64,
            )

        return LP(
            np.asarray(self.c, dtype=np.float64),
            _to_csc(self.A_eq),
            np.asarray(self.b_eq, dtype=np.float64),
            _to_csc(self.A_ineq),
            np.asarray(self.b_ineq, dtype=np.float64),
            np.asarray(self.lower_bounds, dtype=np.float64),
            np.asarray(self.upper_bounds, dtype=np.float64),
        )


# %%
