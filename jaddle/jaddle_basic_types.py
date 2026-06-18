import jax.numpy as jnp
from typing import NamedTuple
import optax
from typing import Any, Callable, NamedTuple, Sequence, Union
import jax
import numpy as np

ScheduleLike = Union[float, Callable[[jnp.ndarray], jnp.ndarray]]


class HedgePoolState(NamedTuple):
    # A pool of joint primal/dual players. `primal_states` and `dual_states`
    # are tuples of equal length; index i is one player's optax state for its
    # primal optimiser and its dual optimiser respectively. A single
    # `log_weights` / `active_mask` governs the whole pool — primal and dual
    # updates of a player are mixed with the SAME weight, which is what makes
    # the players jointly selected rather than two independent populations.
    primal_states: tuple
    dual_states: tuple
    log_weights: jnp.ndarray
    active_mask: jnp.ndarray
    loss_scale: jnp.ndarray
    last_raw_losses: jnp.ndarray
    last_normalized_losses: jnp.ndarray
    last_clipped_losses: jnp.ndarray
    last_centered_losses: jnp.ndarray
    # Per-player step-ratio (k) control, a (count,) array. Populated only when
    # the ensemble is built with per_expert_k; otherwise held at 1.0 and unused.
    expert_k: jnp.ndarray = None

    def prune(self, threshold: float, min_keep: int = 1):
        """Mask experts with low weight while keeping state size fixed."""
        weights = np.asarray(jax.nn.softmax(self.log_weights))
        active_mask = np.asarray(self.active_mask) > 0
        effective_weights = np.where(active_mask, weights, 0.0)
        keep_mask = np.logical_and(active_mask, effective_weights > threshold)

        if keep_mask.sum() < min_keep:
            active_idx = np.where(active_mask)[0]
            if active_idx.size == 0:
                active_idx = np.arange(weights.shape[0])

            ranked_active = active_idx[np.argsort(-effective_weights[active_idx])]
            topk_idx = ranked_active[: min(min_keep, ranked_active.size)]
            keep_mask = np.zeros_like(effective_weights, dtype=bool)
            keep_mask[topk_idx] = True

        idx = np.where(keep_mask)[0]
        keep_mask_jnp = jnp.array(keep_mask, dtype=jnp.float64)
        masked_logits = jnp.where(
            keep_mask_jnp > 0,
            self.log_weights,
            jnp.full_like(self.log_weights, -1e30),
        )
        pruned_log_weights = masked_logits - jax.nn.logsumexp(masked_logits)

        return (
            HedgePoolState(
                primal_states=self.primal_states,
                dual_states=self.dual_states,
                log_weights=pruned_log_weights,
                active_mask=keep_mask_jnp,
                loss_scale=self.loss_scale,
                last_raw_losses=self.last_raw_losses,
                last_normalized_losses=self.last_normalized_losses,
                last_clipped_losses=self.last_clipped_losses,
                last_centered_losses=self.last_centered_losses,
                expert_k=self.expert_k,
            ),
            jnp.array(idx),
        )


class HedgeSaddleState(NamedTuple):
    # Single pool of joint primal/dual players. `eta` is the Hedge temperature
    # for the (one) weight vector.
    pool: HedgePoolState
    step: jnp.ndarray
    last_eta: jnp.ndarray

    def prune(self, threshold: float, min_keep: int = 1, min_step: int = 0):
        if int(self.step) < min_step:
            idx = jnp.arange(len(self.pool.primal_states))
            return self, idx

        pruned_pool, idx = self.pool.prune(threshold, min_keep=min_keep)
        return (
            HedgeSaddleState(
                pruned_pool,
                self.step,
                self.last_eta,
            ),
            idx,
        )


class ExtragradientState(NamedTuple):
    # Stored look-ahead point x_half = params - lr * grad (phase 0 output).
    x_half: Any
    # 0 = look-ahead phase (next call stores x_half, returns zero update),
    # 1 = corrector phase  (next call returns corrector, advances step).
    phase: jnp.ndarray
    step: jnp.ndarray


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
        # Fused [A_eq; A_ineq] for 2-matvec gradient computation
        import jax.experimental.sparse as _jsp

        self.n_eq = A_eq.shape[0]
        self.A = _jsp.bcoo_concatenate([A_eq, A_ineq], dimension=0)
        self.A_T = self.A.T
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
