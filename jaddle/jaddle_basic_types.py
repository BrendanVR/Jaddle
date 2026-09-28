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


def _diagonal_dual(A_eq, A_ineq, c, n_eq, n_ineq):
    """Diagonal (Jacobi) dual stationarity seed for the LP condition Aᵀy ≈ -c.

    A cheap, solve-free warm start: instead of a full (bound-constrained) least
    squares, project -c onto each constraint row independently, ignoring the
    off-diagonal coupling between rows. For row j of A this gives the 1-D
    least-squares estimate

        y_j = (Aⱼ · (-c)) / ‖Aⱼ‖²   = -(A @ c)_j / rownorm²_j

    so the whole vector is just one matvec ``A @ c`` plus the squared row norms —
    no iterative solver. Inequality duals are clipped to ≥ 0 for dual
    feasibility. Crude vs the exact least-norm dual, but it places the duals'
    signs and magnitudes in the right ballpark, which is what a warm start needs;
    the solver refines from there.

    Returns (dual_eq, dual_ineq). Works with dense JAX arrays and BCOO sparse
    matrices (anything supporting ``@`` and elementwise square).
    """
    c = jnp.asarray(c)

    def _row_sq_norms(A, n_rows):
        # Squared row norms ‖Aⱼ‖². For BCOO, square the data and scatter-add by
        # row index (axis-reductions on sparse `A*A` stay sparse and don't
        # densify cleanly); for dense, just sum of squares along the columns.
        if hasattr(A, "indices"):  # BCOO
            rows = A.indices[:, 0]
            return jnp.zeros(n_rows, dtype=A.data.dtype).at[rows].add(A.data**2)
        return jnp.sum(jnp.asarray(A) ** 2, axis=1)

    def _row_dual(A, n_rows):
        # (A @ c) is the per-row inner product Aⱼ·c; rownorm²_j = sum_k A_jk².
        Ac = A @ c
        row_sq = _row_sq_norms(A, n_rows)
        return -Ac / (row_sq + 1e-12)

    dual_eq = _row_dual(A_eq, n_eq) if n_eq > 0 else jnp.zeros(0, dtype=c.dtype)
    dual_ineq = (
        jnp.maximum(_row_dual(A_ineq, n_ineq), 0.0)
        if n_ineq > 0
        else jnp.zeros(0, dtype=c.dtype)
    )
    return dual_eq, dual_ineq


def _diagonal_primal(A_eq, b_eq, A_ineq, b_ineq, lower, upper, n_vars):
    """Diagonal (Jacobi) primal feasibility seed for the LP condition Ax ≈ b.

    The transpose-symmetric counterpart of :func:`_diagonal_dual`. Instead of
    solving the dual stationarity ``Aᵀy ≈ -c`` row-by-row, it solves primal
    feasibility ``Ax ≈ b`` column-by-column. Taking a single Jacobi step from
    ``x = 0`` (where the residual ``Ax - b`` is just ``-b``), the 1-D
    least-squares correction along column ``k`` is

        x_k = (Aₖ · b) / ‖Aₖ‖²   = (Aᵀb)_k / colnorm²_k

    so the whole vector is one transposed matvec ``Aᵀ @ b`` plus the squared
    column norms — no iterative solver. The equality and inequality blocks act
    on the same ``x``, so their numerators and denominators are accumulated
    together. The result is clipped to ``[lower, upper]`` (the primal analogue of
    the ``≥ 0`` clip on inequality duals) so the seed lands inside the feasible
    box. Crude vs an exact projection, but it places ``x`` in the right ballpark;
    the solver refines from there.

    Works with dense JAX arrays and BCOO sparse matrices (anything supporting
    ``@`` and elementwise square).
    """

    def _col_sq_norms(A):
        # Squared column norms ‖Aₖ‖². For BCOO, square the data and scatter-add
        # by column index (axis 1 of indices); for dense, sum of squares down
        # the rows.
        if hasattr(A, "indices"):  # BCOO
            cols = A.indices[:, 1]
            return jnp.zeros(n_vars, dtype=A.data.dtype).at[cols].add(A.data**2)
        return jnp.sum(jnp.asarray(A) ** 2, axis=0)

    numer = jnp.zeros(n_vars)
    denom = jnp.zeros(n_vars)
    if A_eq.shape[0] > 0:
        numer = numer + A_eq.T @ jnp.asarray(b_eq)
        denom = denom + _col_sq_norms(A_eq)
    if A_ineq.shape[0] > 0:
        numer = numer + A_ineq.T @ jnp.asarray(b_ineq)
        denom = denom + _col_sq_norms(A_ineq)

    x = numer / (denom + 1e-12)
    return optax.projections.projection_box(x, lower, upper)


def _cgls_refine(A, A_T, b, x0, iterations):
    """Refine ``x0`` toward ``min ‖Ax − b‖²`` with a few CGLS iterations.

    CGLS is conjugate gradient applied implicitly to the normal equations
    ``AᵀAx = Aᵀb`` without ever forming ``AᵀA``: each iteration costs exactly
    one ``A @ p`` and one ``Aᵀ @ r`` matvec, and ‖Ax − b‖ decreases
    monotonically over the growing Krylov subspace — unlike re-iterating the
    Jacobi step of :func:`_diagonal_primal`, which need not converge at all.
    Inequality rows are treated as targets ``Ax ≈ b`` (the same convention as
    the one-shot diagonal seed); the caller box-projects the result. Runs
    eagerly, once, at seed time, so a plain Python loop suffices.
    """
    x = x0
    r = b - A @ x
    s = A_T @ r
    p = s
    gamma = jnp.vdot(s, s)
    for _ in range(iterations):
        if not np.isfinite(float(gamma)) or float(gamma) <= 0.0:
            break
        q = A @ p
        q_sq = jnp.vdot(q, q)
        if not np.isfinite(float(q_sq)) or float(q_sq) <= 0.0:
            break
        alpha = gamma / q_sq
        x = x + alpha * p
        r = r - alpha * q
        s = A_T @ r
        gamma_new = jnp.vdot(s, s)
        p = s + (gamma_new / gamma) * p
        gamma = gamma_new
    return x


def _seed_kkt_score(A, A_T, b, c, n_eq, lower, upper, state):
    """Relative KKT score of a candidate start point (scaled space, ∞-norms).

    ``max(primal, dual)`` residual, each normalised PDLP-style (primal by
    ``1+‖b‖∞``, dual by ``1+‖c‖∞``) so the two sides are comparable. The dual
    violation uses the same bound-class rules as the solver's DFR metric:
    projected-gradient magnitude for boxed variables, classical reduced-cost
    sign rules for one-sided/free ones. Used by the ``initial_solution``
    keep-better gate to compare seed candidates; costs 2 matvecs.
    """
    Ax = A @ state.primal
    eq_res = jnp.max(jnp.abs(Ax[:n_eq] - b[:n_eq]), initial=0.0)
    ineq_res = jnp.max(jnp.maximum(Ax[n_eq:] - b[n_eq:], 0.0), initial=0.0)
    primal_rel = jnp.maximum(eq_res, ineq_res) / (
        1.0 + jnp.max(jnp.abs(b), initial=0.0)
    )

    dual = jnp.concatenate([state.dual_eq, state.dual_ineq])
    reduced_cost = c + A_T @ dual
    finite_lower = jnp.isfinite(lower)
    finite_upper = jnp.isfinite(upper)
    has_both = finite_lower & finite_upper
    has_only_lower = finite_lower & (~finite_upper)
    has_only_upper = (~finite_lower) & finite_upper
    proj = optax.projections.projection_box(state.primal - reduced_cost, lower, upper)
    dual_violation = jnp.where(
        has_both,
        jnp.abs(state.primal - proj),
        jnp.where(
            has_only_lower,
            jnp.maximum(-reduced_cost, 0.0),
            jnp.where(
                has_only_upper,
                jnp.maximum(reduced_cost, 0.0),
                jnp.abs(reduced_cost),  # free variable
            ),
        ),
    )
    dual_rel = jnp.max(dual_violation, initial=0.0) / (
        1.0 + jnp.max(jnp.abs(c), initial=0.0)
    )
    return jnp.maximum(primal_rel, dual_rel)


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
        import jax.experimental.sparse as _jsp
        from jaddle.jaddle_optimisers import jaddle_dtype

        float_dtype = jaddle_dtype()
        if float_dtype == jnp.float64 and not jax.config.jax_enable_x64:
            float_dtype = jnp.float32
        np_float = np.float64 if float_dtype == jnp.float64 else np.float32

        def _to_bcoo(A):
            A = A.astype(np_float).sorted_indices()
            A.sum_duplicates()
            bcoo = _jsp.BCOO.from_scipy_sparse(A.tocoo()).sort_indices()
            return _jsp.BCOO(
                (bcoo.data.astype(float_dtype), bcoo.indices), shape=bcoo.shape
            )

        return cls(
            jnp.asarray(c, dtype=float_dtype),
            _to_bcoo(A_eq),
            jnp.asarray(b_eq, dtype=float_dtype),
            _to_bcoo(A_ineq),
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
        return jnp.max(jnp.maximum(self.A_ineq @ x - self.b_ineq, 0.0))

    def eq_slack(self, x):
        return jnp.max(jnp.abs(self.A_eq @ x - self.b_eq))

    def diff_eq_slack(self, x):
        return self.A_eq @ x - self.b_eq

    def complementarity_slack(self, x, dual_ineq):
        return (dual_ineq * (self.A_ineq @ x - self.b_ineq)).sum()

    def initial_solution(
        self,
        seed_iterations=0,
        seed_crash=False,
        seed_couple=True,
        seed_gate=True,
        verbose=False,
    ):
        """Build the default start point (in this LP's — i.e. scaled — space).

        Assembles the diagonal (Jacobi) primal/dual seeds, then optionally:

        - ``seed_iterations``: refine the primal with that many CGLS
          iterations on ``min ‖Ax − b‖²`` (2 matvecs each) before
          box-projecting — a strictly stronger version of the one-shot
          diagonal step. 0 (default) keeps the one-shot seed.
        - ``seed_crash``: simplex-style crash — variables whose reduced cost
          ``r = c + Aᵀy`` at the dual seed has a clear sign are pushed to the
          finite bound that sign selects (``r > 0`` → lower, ``r < 0`` →
          upper), making the primal complementarity-consistent with the dual
          seed and more vertex-like.
        - ``seed_couple``: complementarity coupling — inequality duals are
          zeroed on rows that are strictly slack at the (final) primal seed,
          so the start point doesn't fabricate a complementarity violation
          the solver must first undo.
        - ``seed_gate``: keep-better gate — the assembled seed's relative KKT
          score (:func:`_seed_kkt_score`) is compared against the
          box-projected zero start (the PDLP default) and the better
          candidate is returned, so seeding is never worse than starting
          cold.
        """
        n_eq = self.num_eq_constraints()
        n_ineq = self.num_ineq_constraints()
        primal = _diagonal_primal(
            self.A_eq,
            self.b_eq,
            self.A_ineq,
            self.b_ineq,
            self.lower_bounds,
            self.upper_bounds,
            self.num_variables(),
        )
        dual_eq, dual_ineq = _diagonal_dual(
            self.A_eq, self.A_ineq, self.c, n_eq, n_ineq
        )

        if seed_iterations > 0 and self.num_constraints() > 0:
            primal = _cgls_refine(self.A, self.A_T, self.b, primal, seed_iterations)
            primal = optax.projections.projection_box(
                primal, self.lower_bounds, self.upper_bounds
            )

        if seed_crash:
            dual = jnp.concatenate([dual_eq, dual_ineq])
            reduced_cost = self.c + self.A_T @ dual
            crash_lower = (reduced_cost > 0) & jnp.isfinite(self.lower_bounds)
            crash_upper = (reduced_cost < 0) & jnp.isfinite(self.upper_bounds)
            primal = jnp.where(
                crash_lower,
                self.lower_bounds,
                jnp.where(crash_upper, self.upper_bounds, primal),
            )

        if seed_couple and n_ineq > 0:
            # Zero the dual on rows strictly slack at the primal seed
            # (relative tolerance, so near-active rows keep their dual): a
            # positive multiplier on a slack row is a complementarity
            # violation the seed would otherwise hand the solver.
            slack = self.b_ineq - self.A_ineq @ primal
            strictly_slack = slack > 1e-6 * (1.0 + jnp.abs(self.b_ineq))
            dual_ineq = jnp.where(strictly_slack, 0.0, dual_ineq)

        seeded = SaddleState(primal=primal, dual_ineq=dual_ineq, dual_eq=dual_eq)
        if not seed_gate:
            return seeded

        zero_start = SaddleState(
            primal=optax.projections.projection_box(
                jnp.zeros_like(primal), self.lower_bounds, self.upper_bounds
            ),
            dual_ineq=jnp.zeros_like(dual_ineq),
            dual_eq=jnp.zeros_like(dual_eq),
        )
        seeded_score = _seed_kkt_score(
            self.A, self.A_T, self.b, self.c, n_eq,
            self.lower_bounds, self.upper_bounds, seeded,
        )
        zero_score = _seed_kkt_score(
            self.A, self.A_T, self.b, self.c, n_eq,
            self.lower_bounds, self.upper_bounds, zero_start,
        )
        use_seed = bool(seeded_score <= zero_score)
        if verbose:
            winner = "diagonal seed" if use_seed else "zero start"
            print(
                f"Seed gate: diagonal seed KKT {float(seeded_score):.3e} "
                f"vs zero start {float(zero_score):.3e} -> {winner}"
            )
        return seeded if use_seed else zero_start

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
