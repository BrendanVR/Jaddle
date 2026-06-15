import jax.numpy as jnp
from typing import NamedTuple
import optax
from typing import Any, Callable, NamedTuple, Sequence, Union
import jax
import numpy as np

ScheduleLike = Union[float, Callable[[jnp.ndarray], jnp.ndarray]]


class HedgePoolState(NamedTuple):
    expert_states: tuple
    log_weights: jnp.ndarray
    active_mask: jnp.ndarray
    loss_scale: jnp.ndarray
    last_raw_losses: jnp.ndarray
    last_normalized_losses: jnp.ndarray
    last_clipped_losses: jnp.ndarray
    last_centered_losses: jnp.ndarray

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
                expert_states=self.expert_states,
                log_weights=pruned_log_weights,
                active_mask=keep_mask_jnp,
                loss_scale=self.loss_scale,
                last_raw_losses=self.last_raw_losses,
                last_normalized_losses=self.last_normalized_losses,
                last_clipped_losses=self.last_clipped_losses,
                last_centered_losses=self.last_centered_losses,
            ),
            jnp.array(idx),
        )


class HedgeSaddleState(NamedTuple):
    primal: HedgePoolState
    dual: HedgePoolState
    step: jnp.ndarray
    last_primal_eta: jnp.ndarray
    last_dual_eta: jnp.ndarray

    def prune(self, threshold: float, min_keep: int = 1, min_step: int = 0):
        if int(self.step) < min_step:
            primal_idx = jnp.arange(len(self.primal.expert_states))
            dual_idx = jnp.arange(len(self.dual.expert_states))
            return self, primal_idx, dual_idx

        pruned_primal, primal_idx = self.primal.prune(threshold, min_keep=min_keep)
        pruned_dual, dual_idx = self.dual.prune(threshold, min_keep=min_keep)
        return (
            HedgeSaddleState(
                pruned_primal,
                pruned_dual,
                self.step,
                self.last_primal_eta,
                self.last_dual_eta,
            ),
            primal_idx,
            dual_idx,
        )


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
        return SaddleState(
            primal=optax.projections.projection_box(
                jnp.zeros(self.num_variables()),
                self.lower_bounds,
                self.upper_bounds,
            ),
            dual_ineq=jnp.zeros(self.num_ineq_constraints()),
            dual_eq=jnp.zeros(self.num_eq_constraints()),
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
        return SaddleState(
            primal=optax.projections.projection_box(
                jnp.zeros(self.num_variables()),
                self.lower_bounds,
                self.upper_bounds,
            ),
            dual_ineq=jnp.zeros(self.num_ineq_constraints()),
            dual_eq=jnp.zeros(self.num_eq_constraints()),
        )
