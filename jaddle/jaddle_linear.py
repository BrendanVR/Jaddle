# %%
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsp
from optax.projections import projection_non_negative, projection_box
import optax
import numpy as np
import functools
from typing import NamedTuple
import time
from scipy import sparse as sp
from jaddle.jaddle_basic_types import LP, JaddleLP, SaddleState
from scipy.sparse.linalg import gmres
from jax.scipy.sparse.linalg import gmres
import jaddle.jaddle_optimisers as jo

np.set_printoptions(precision=2, suppress=True)


_LINEAR_RUN_EPOCH_CACHE = {}


# %%
def estimate_augmented_spectral_norm(
    lp: JaddleLP,
    num_iters: int = 30,
    seed: int = 0,
):
    """Approximate the spectral norm (largest singular value) of the augmented
    matrix

        M = [[ A,    -b ],
             [ cᵀ,    0 ]]

    where ``A = [A_eq; A_ineq]`` (m×n), ``b`` is the length-m fused RHS appended
    as the extra column, and ``c`` the length-n cost appended (transposed) as
    the extra row. M has shape (m+1)×(n+1). (The signs of the ±b/±c blocks do
    not affect singular values, so this equals σ_max of [[A, b], [cᵀ, 0]].)

    The estimate is computed by power iteration on the normal operator MᵀM via
    matvecs only — never forming M, MᵀM, or any dense matrix. M acts on a
    vector ``[u; s]`` (u length n, scalar s) as

        M [u; s] = [A u - b s ;  cᵀ u],

    and its transpose on ``[p; q]`` (p length m, scalar q) as

        Mᵀ [p; q] = [Aᵀ p + c q ;  -bᵀ p].

    Each iteration applies Mᵀ(M·v) and renormalises; the returned value is
    sqrt(λ_max(MᵀM)) = σ_max(M). ``num_iters`` power steps are run from a fixed
    random start (``seed``); 30 is comfortably enough for a tight estimate on
    LP-scale matrices.
    """
    A = lp.A  # (m, n) fused [A_eq; A_ineq], BCOO
    A_T = lp.A_T  # (n, m) explicit transpose
    c = lp.c  # (n,)
    b = lp.b  # (m,)

    def M_matvec(u, s):
        # M @ [u; s] = [A u - b s; cᵀ u]
        top = A @ u - b * s
        bottom = c @ u
        return top, bottom

    def MT_matvec(p, q):
        # Mᵀ @ [p; q] = [Aᵀ p + c q; -bᵀ p]
        top = A_T @ p + c * q
        bottom = -(b @ p)
        return top, bottom

    def normal_op(u, s):
        # MᵀM @ [u; s]
        p, q = M_matvec(u, s)
        return MT_matvec(p, q)

    def vnorm(u, s):
        return jnp.sqrt(u @ u + s * s)

    key = jax.random.PRNGKey(seed)
    u = jax.random.normal(key, c.shape, dtype=c.dtype)
    s = jnp.array(1.0, dtype=c.dtype)
    n = vnorm(u, s)
    u, s = u / n, s / n

    def body(_, carry):
        u, s = carry
        u, s = normal_op(u, s)
        n = vnorm(u, s)
        return u / n, s / n

    u, s = jax.lax.fori_loop(0, num_iters, body, (u, s))

    # Rayleigh quotient on MᵀM gives λ_max; σ_max = sqrt(λ_max).
    Mu, Ms = M_matvec(u, s)
    lambda_max = Mu @ Mu + Ms * Ms
    return jnp.sqrt(lambda_max)


def estimate_augmented_inf_norm(lp: JaddleLP):
    """Compute the exact ∞-norm (max absolute row sum) of the augmented matrix

        M = [[ A,    -b ],
             [ cᵀ,    0 ]]

    where ``A = [A_eq; A_ineq]`` (m×n), ``b`` is the length-m fused RHS and ``c``
    the length-n cost. This is PDLP's step-size seed convention (Algorithm 1,
    line 2: ``η ← 1/‖·‖∞``), but on the augmented system rather than on ``A``
    alone — the ±b/±c signs do not affect absolute row sums so this equals
    ‖[[A, b], [cᵀ, 0]]‖∞.

    The ∞-norm is the largest ℓ₁ row norm. Row ``i`` of the top block is
    ``[A_i, -b_i]`` with absolute sum ``Σ_j |A_ij| + |b_i|``; the single bottom
    row is ``[cᵀ, 0]`` with absolute sum ``Σ_j |c_j|``. Per-row absolute sums of
    ``A`` are obtained with one matvec ``|A| · 1`` (no dense matrix formed, no
    power iteration), so this is a single O(nnz) pass and depends only on the
    matrix data, not a random seed.
    """
    A = lp.A  # (m, n) fused [A_eq; A_ineq], BCOO
    b = lp.b  # (m,)
    c = lp.c  # (n,)

    # Absolute row sums of A via one matvec against the all-ones vector.
    abs_A = jsp.BCOO((jnp.abs(A.data), A.indices), shape=A.shape)
    ones_n = jnp.ones(A.shape[1], dtype=c.dtype)
    top_row_sums = abs_A @ ones_n + jnp.abs(b)  # rows [A_i, -b_i]
    bottom_row_sum = jnp.sum(jnp.abs(c))  # row [cᵀ, 0]

    return jnp.maximum(jnp.max(top_row_sums, initial=0.0), bottom_row_sum)


# %%
# Solvers for constrained linear optimisation via saddle point formulation
def __sps(
    max_iter,
    start_iter,
    lp: JaddleLP,
    optimiser,
    initial_solution,
    initial_avg_state=None,
    initial_opt_state=None,
    weight_function=lambda _: 1.0,
    total_weight=0.0,
    primal_damping=0.0,
    dual_damping_ineq=0.0,
    dual_damping_eq=0.0,
    average=True,
    update_mode="synchronous",
    k_scaling=False,
    k_init=1.0,
    adaptive_eta=None,
    gate=None,
):
    # Per-iteration convergence gate. `gate` is None (disabled) or the tuple
    # (row_scale_all, b_norm, gate_tol, col_scale, c_norm, dfr_tol): a cheap
    # trigger checked every iteration inside the scan. When the CURRENT iterate is
    # BOTH primal-feasible (relative PFR < gate_tol) AND dual-feasible (relative
    # DFR < dfr_tol) the carried converged_flag latches True and the remaining
    # scan steps become no-ops (state frozen). It is only a trigger to return
    # early to the host — the host still runs the full 3-part certificate on the
    # average iterate — so it can never cause a false stop, only an earlier host
    # check. Wired for synchronous (whose `grad` computes A @ x and the reduced
    # cost) and pdhg (which carries both); halpern and extragradient thread the
    # flag through inertly (never latch — see the step bodies for why).
    gate_enabled = gate is not None
    if gate_enabled:
        (
            gate_row_scale_all,
            gate_b_norm,
            gate_tol,
            gate_col_scale,
            gate_c_norm,
            gate_dfr_tol,
            gate_c_max,
            gate_c_norm_robust,
            gate_gap_tol,
            gate_report_c,
        ) = gate

    # The stepping scheme is selected by `update_mode`. This derived boolean
    # keeps the dense per-scheme branching below readable while the string stays
    # the single source of truth.
    extragradient = update_mode == "extragradient"
    # cuPDLP-style per-iteration adaptive step size. When `adaptive_eta` is not
    # None the stepping scheme replaces the optimiser's fixed learning rate with
    # a single scalar base step eta, line-searched every iteration: a trial step
    # is taken, the largest admissible step eta_bar = move / (2|interaction|) is
    # formed from the trial movement, and the step is rejected + shrunk if it
    # overshot eta_bar. The primal/dual steps are tau=eta/k, sigma=eta*k (k is the
    # primal weight). eta is packed alongside k in the k-slot of opt_state.
    # Requires k_scaling (it needs k). Supported for pdhg (extrapolation) and
    # extragradient (corrector) — the two schemes that are contractive on the
    # bilinear saddle; each supplies its own per-iteration `trial`, while the
    # retry loop and eta advancement are shared. Plain Arrow-Hurwicz
    # (synchronous) and Gauss-Seidel (alternating) are not contractive here and a
    # line search cannot fix that, so they are excluded.
    adaptive_modes = ("pdhg", "extragradient")
    adaptive_step = adaptive_eta is not None and update_mode in adaptive_modes
    # Halpern-anchored PDHG (restarted Halpern). The base operator T(z) is the
    # adaptive PDHG step; each iterate is then anchored back toward z_0 (the
    # iterate at the start of the current restart cycle):
    #     z_{k+1} = lambda_k z_0 + (1 - lambda_k) T(z_k),   lambda_k = 1/(k+2),
    # with the local index k = i - start_iter reset each restart (so lambda_k
    # restarts from 1/2). The anchor combination is a convex combination of two
    # feasible iterates, so feasibility is preserved without re-projection.
    # Halpern always rides the adaptive PDHG step, so it implies adaptive_step
    # and requires adaptive_eta + k_scaling. The anchor z_0 is carried in the
    # k-slot alongside (k, eta).
    halpern = update_mode == "halpern"
    if halpern:
        if adaptive_eta is None:
            raise ValueError("update_mode='halpern' requires adaptive_eta")
        adaptive_step = True
    if adaptive_step and not k_scaling:
        raise ValueError("adaptive_eta requires k_scaling (primal weight k)")
    # k-scaling is an orthogonal option (any update_mode): a primal weight k
    # rescales the primal/dual gradients by (1/k, k) before opt_update, so the
    # dual/primal step ratio is k**2. When on, k is packed into opt_state and
    # rebalanced at each restart in `solve` (PDLP-style); constant within an
    # epoch.

    def projection_primal(primal_state):
        return projection_box(primal_state, lp.lower_bounds, lp.upper_bounds)

    def grad(state, return_Ax=False):
        # Fused matvecs: 2 sparse ops (A @ x, Aᵀ @ y) instead of 4. The
        # controllers below act on post-optimiser update norms, not on A·dx, so
        # there is nothing to gain from returning Ax — keep the single matvec
        # pair and return only the gradient. `return_Ax` exposes the already-
        # computed A @ x for the per-iteration convergence gate (no extra matvec).
        dual = jnp.concatenate([state.dual_eq, state.dual_ineq])
        Ax = lp.A @ state.primal  # shape: (n_eq + n_ineq,)
        ATd = lp.A_T @ dual  # shape: (n_vars,)
        grad_primal = lp.c + ATd + primal_damping * state.primal
        residual = lp.b - Ax
        grad_dual_eq = residual[: lp.n_eq] + dual_damping_eq * state.dual_eq
        grad_dual_ineq = residual[lp.n_eq :] + dual_damping_ineq * state.dual_ineq
        g = SaddleState(
            primal=grad_primal,
            dual_ineq=grad_dual_ineq,
            dual_eq=grad_dual_eq,
        )
        if return_Ax:
            return g, Ax
        return g

    def grad_primal_only(state):
        # Primal partial only: c + Aᵀd (+ damping). One sparse matvec (Aᵀ @ d);
        # the A @ x matvec that the full `grad` does for the dual residual is
        # skipped entirely. Used by the alternating/pdhg primal half.
        dual = jnp.concatenate([state.dual_eq, state.dual_ineq])
        ATd = lp.A_T @ dual
        return lp.c + ATd + primal_damping * state.primal

    def grad_dual_only(state):
        # Dual partials only: b - Ax (+ damping). One sparse matvec (A @ x); the
        # Aᵀ @ d matvec is skipped. Used by the alternating/pdhg dual half.
        Ax = lp.A @ state.primal
        residual = lp.b - Ax
        grad_dual_eq = residual[: lp.n_eq] + dual_damping_eq * state.dual_eq
        grad_dual_ineq = residual[lp.n_eq :] + dual_damping_ineq * state.dual_ineq
        return grad_dual_ineq, grad_dual_eq

    def grad_dual_only_from_Ax(Ax, state):
        # Same as grad_dual_only but reuses a pre-computed A @ state.primal,
        # saving the matvec when Ax is already available at the call site.
        residual = lp.b - Ax
        grad_dual_eq = residual[: lp.n_eq] + dual_damping_eq * state.dual_eq
        grad_dual_ineq = residual[lp.n_eq :] + dual_damping_ineq * state.dual_ineq
        return grad_dual_ineq, grad_dual_eq

    def gate_tripped(Ax, reduced_cost, state):
        # Cheap per-iteration convergence trigger evaluating the SAME 3-part LP
        # certificate the host `converged()` uses (relative PFR, DFR, and the
        # sign-guarded duality-gap RDG), but on the CURRENT iterate, reusing
        # Ax = A @ x and reduced_cost = c + Aᵀy the step already computes (no
        # extra matvec). All three are required: a primal+dual-feasible iterate
        # whose duality gap is still open must NOT freeze, or restart-driven modes
        # get trapped (ns1830653). It is a trigger only — the host re-runs the
        # full certificate on the AVERAGE iterate, so the gate can never cause a
        # false stop, only an earlier host check. This mirrors compute_epoch_
        # metrics; keep the two in sync. Returns a scalar bool; only meaningful
        # when gate_enabled.
        primal = state.primal
        dual = jnp.concatenate([state.dual_eq, state.dual_ineq])

        # ---- PFR: constraint violation unscaled by row_scale to true units,
        # max over eq (|·|) and ineq (positive part) rows, ÷ (1 + ‖b‖). ----
        Ax_minus_b = Ax - lp.b
        grad_dual_eq = Ax_minus_b[: lp.n_eq]
        grad_dual_ineq = Ax_minus_b[lp.n_eq :]
        violations_unscaled = Ax_minus_b / gate_row_scale_all
        eq_viol = jnp.abs(violations_unscaled[: lp.n_eq])
        ineq_viol = jnp.maximum(violations_unscaled[lp.n_eq :], 0.0)
        pfr = jnp.maximum(
            jnp.max(eq_viol, initial=0.0), jnp.max(ineq_viol, initial=0.0)
        ) / (1.0 + gate_b_norm)

        # ---- DFR: projected-gradient / reduced-cost dual-feasibility residual in
        # true units. reduced_cost = c_scaled + A_scaledᵀy carries col_scale, so
        # r_true = r/col_scale; x_true = x·col_scale, bounds_true = b·col_scale. --
        reduced_cost_true = reduced_cost / gate_col_scale
        primal_true = primal * gate_col_scale
        lb_true = lp.lower_bounds * gate_col_scale
        ub_true = lp.upper_bounds * gate_col_scale
        finite_lower = jnp.isfinite(lb_true)
        finite_upper = jnp.isfinite(ub_true)
        has_both = finite_lower & finite_upper
        has_only_lower = finite_lower & (~finite_upper)
        has_only_upper = (~finite_lower) & finite_upper
        proj = projection_box(primal_true - reduced_cost_true, lb_true, ub_true)
        dual_viol = jnp.where(
            has_both,
            jnp.abs(primal_true - proj),
            jnp.where(
                has_only_lower,
                jnp.maximum(-reduced_cost_true, 0.0),
                jnp.where(
                    has_only_upper,
                    jnp.maximum(reduced_cost_true, 0.0),
                    jnp.abs(reduced_cost_true),  # free variable
                ),
            ),
        )
        dfr = jnp.max(dual_viol, initial=0.0) / (1.0 + gate_c_norm)

        # ---- Duality gap: the three-way complementarity decomposition, in scaled
        # space then ·c_max, exactly as compute_epoch_metrics builds it. The
        # box_infimum is sign-guarded to −∞ on wrong-sign reduced costs (band
        # tol·(1+‖c‖_robust)) so the gap is +∞ (undefined) at dual-infeasible
        # points rather than fabricated finite. reduced_cost / bounds here are the
        # SCALED versions (the decomposition multiplies the scaled primal). ----
        objective_value = (
            lp.objective(primal) if gate_report_c is None else gate_report_c @ primal
        ) * gate_c_max
        lb_s = lp.lower_bounds
        ub_s = lp.upper_bounds
        lower_term = reduced_cost * lb_s
        upper_term = reduced_cost * ub_s
        _dg_band = gate_dfr_tol * (1.0 + gate_c_norm_robust)
        neg_inf = jnp.asarray(-jnp.inf, reduced_cost.dtype)
        lower_only = jnp.where(reduced_cost_true >= -_dg_band, lower_term, neg_inf)
        upper_only = jnp.where(reduced_cost_true <= _dg_band, upper_term, neg_inf)
        free_term = jnp.where(jnp.abs(reduced_cost_true) <= _dg_band, 0.0, neg_inf)
        box_infimum = jnp.where(
            has_both,
            jnp.minimum(lower_term, upper_term),
            jnp.where(
                has_only_lower,
                lower_only,
                jnp.where(has_only_upper, upper_only, free_term),
            ),
        )
        gap_bound = (reduced_cost @ primal - jnp.sum(box_infimum)) * gate_c_max
        gap_ineq = -(state.dual_ineq @ grad_dual_ineq) * gate_c_max
        gap_eq = -(state.dual_eq @ grad_dual_eq) * gate_c_max
        duality_gap = gap_bound + gap_ineq + gap_eq
        gap_finite = jnp.isfinite(duality_gap)
        dual_bound = objective_value - duality_gap
        rdg = jnp.abs(duality_gap) / (
            1.0 + jnp.abs(objective_value) + jnp.abs(dual_bound)
        )
        # No-cancellation guard (matches converged()): the summed magnitudes of
        # the three gap components must also be within tolerance, so a gap that is
        # small only through sign cancellation doesn't falsely trip.
        rdg_abs = (jnp.abs(gap_bound) + jnp.abs(gap_ineq) + jnp.abs(gap_eq)) / (
            1.0 + jnp.abs(objective_value) + jnp.abs(dual_bound)
        )

        return (
            (pfr < gate_tol)
            & (dfr < gate_dfr_tol)
            & gap_finite
            & (rdg < gate_gap_tol)
            & (rdg_abs < gate_gap_tol)
        )

    def freeze_leaf(converged_flag, old, new):
        # No-op tail: once converged_flag latches, keep the pre-step value so the
        # remaining iterations don't move the iterate/average/opt-state. Work is
        # computed unconditionally (jnp.where, not lax.cond) to stay type-stable
        # under scan; the matvec on the frozen tail is not saved — correctness /
        # convergence control is the goal, not tail-matvec savings.
        return jax.tree.map(lambda o, n: jnp.where(converged_flag, o, n), old, new)

    def opt_update(gradient, opt_state, state):
        return optimiser.update(gradient, opt_state, state)

    def keep_only_primal(updates):
        return SaddleState(
            primal=updates.primal,
            dual_ineq=jnp.zeros_like(updates.dual_ineq),
            dual_eq=jnp.zeros_like(updates.dual_eq),
        )

    def keep_only_dual(updates):
        return SaddleState(
            primal=jnp.zeros_like(updates.primal),
            dual_ineq=updates.dual_ineq,
            dual_eq=updates.dual_eq,
        )

    def scale_by_k(gradient, k):
        # Primal weight split: (1/k, k) on (primal, dual) gradients.
        return SaddleState(
            primal=gradient.primal / k,
            dual_ineq=gradient.dual_ineq * k,
            dual_eq=gradient.dual_eq * k,
        )

    # When k-scaling is on, k is carried as the last element of opt_state. These
    # helpers keep the step bodies agnostic to whether k is packed or not.
    # In adaptive_step mode the packed k-slot is a (k, eta) pair so the
    # per-iteration step size eta rides alongside the primal weight k; otherwise
    # it is just k (or absent when k_scaling is off). These helpers keep the step
    # bodies agnostic to the packing.
    # Adaptive PDHG carries A @ x_old in the k-slot so the step can reuse last
    # iteration's A @ x_new instead of recomputing it — a 3->2 matvec cut per
    # iteration. Plain adaptive pdhg carries (k, eta, Ax). Halpern's anchor
    # blend changes the primal after the matvec, so the carried Ax_new alone
    # wouldn't match next iter's x_old — but A is linear, so the blended
    # iterate's matvec is one axpy away:
    #     A @ (lam z_0 + (1-lam) cand) = lam (A @ z_0) + (1-lam) Ax_new.
    # The anchor z_0 is constant within an epoch (restarts / re-anchors only
    # fire at epoch boundaries in `solve`), so A @ z_0 is loop-invariant.
    # Halpern therefore carries (k, eta, anchor, Ax_anchor, Ax_state) with
    # Ax_anchor = A @ anchor.primal and Ax_state = A @ state.primal — the same
    # 3->2 matvec cut. Extragradient is a different operator with no A @ x_old
    # to reuse and keeps its (k, eta) slot.
    carry_Ax = adaptive_step and update_mode == "pdhg" and not halpern

    def unpack_k(opt_state):
        if k_scaling:
            if halpern:
                inner, (k, eta, anchor, Ax_anchor, Ax_state) = opt_state
                return inner, k, eta, anchor, Ax_anchor, Ax_state
            if carry_Ax:
                inner, (k, eta, Ax) = opt_state
                return inner, k, eta, Ax
            if adaptive_step:
                inner, (k, eta) = opt_state
                return inner, k, eta
            return opt_state
        return opt_state, None

    def pack_k(opt_state, k, eta=None, anchor=None, Ax=None, Ax_anchor=None):
        if k_scaling:
            if halpern:
                return (opt_state, (k, eta, anchor, Ax_anchor, Ax))
            if carry_Ax:
                return (opt_state, (k, eta, Ax))
            if adaptive_step:
                return (opt_state, (k, eta))
            return (opt_state, k)
        return opt_state

    cache_key = (
        id(lp),
        id(optimiser),
        id(weight_function),
        float(primal_damping),
        float(dual_damping_ineq),
        float(dual_damping_eq),
        average,
        update_mode,
        bool(k_scaling),
        adaptive_step,
        halpern,
        gate_enabled,
    )
    run_epoch = _LINEAR_RUN_EPOCH_CACHE.get(cache_key)

    if run_epoch is None:

        # `average_state` is deliberately not donated: when averaging is off the
        # caller passes the same buffer for both `state` and `average_state`, and
        # XLA rejects donating one buffer twice.
        @functools.partial(
            jax.jit,
            static_argnames=("max_iter",),
            donate_argnames=("state", "opt_state", "total_weight"),
        )
        def run_epoch(
            start_iter,
            state,
            average_state,
            opt_state,
            total_weight=0.0,
            *,
            max_iter,
        ):
            apply_updates = optax.apply_updates

            # ---- Shared machinery for the cuPDLP-style adaptive line search ----
            # Each adaptive update_mode supplies a `trial(eta, state, k)` that
            # takes one raw step at base step `eta` and returns
            # (candidate_state, eta_bar), where eta_bar is the largest admissible
            # base step implied by the trial movement. The retry/reject loop and
            # the eta advancement are identical across modes and live here.
            def _descent_bound(state, cand, k, interaction):
                # eta_bar = move / (2 |interaction|), move = k‖dx‖² + (1/k)‖dy‖².
                # interaction is the mode-specific coupling term (already abs'd).
                dx = cand.primal - state.primal
                dy_eq = cand.dual_eq - state.dual_eq
                dy_ineq = cand.dual_ineq - state.dual_ineq
                move = k * jnp.vdot(dx, dx) + (1.0 / k) * (
                    jnp.vdot(dy_eq, dy_eq) + jnp.vdot(dy_ineq, dy_ineq)
                )
                # No movement => any step is fine (avoid 0/0); flag with +inf so
                # the retry loop accepts and the eta-growth branch is suppressed.
                return jnp.where(interaction > 0.0, move / (2.0 * interaction), jnp.inf)

            def make_adaptive_step(trial):
                # `trial(eta, state, k, Ax_old)` returns (cand, eta_bar, Ax_new):
                # Ax_old = A @ state.primal (carried, not recomputed); Ax_new =
                # A @ cand.primal (carried forward to the next iteration — for
                # halpern, blended with Ax_anchor first). When the path doesn't
                # carry Ax (extragradient), Ax_old/Ax_new are None.
                def step(carry, _):
                    i, state, average_state, opt_state, total_weight = carry
                    if halpern:
                        opt_state, k, eta, anchor, Ax_anchor, Ax_old = unpack_k(
                            opt_state
                        )
                    elif carry_Ax:
                        opt_state, k, eta, Ax_old = unpack_k(opt_state)
                    else:
                        # extragradient: adaptive but doesn't carry Ax.
                        opt_state, k, eta = unpack_k(opt_state)
                        Ax_old = None
                    ip1 = jnp.asarray(i + 1, eta.dtype)

                    # Retry: while the trial step exceeds its admissible bound,
                    # shrink eta to just under eta_bar and re-trial. The
                    # (1-(i+1)^-0.3) factor < 1 guarantees strict decrease, so the
                    # loop terminates. Carry (eta, cand, eta_bar, Ax_new). Ax_old
                    # and the reduced cost rc = c + Aᵀy are eta-invariant (they
                    # depend on `state`, fixed across retries), so rc is captured
                    # from the first trial (not recomputed) and Ax_old is closed
                    # over. `trial` returns rc so the gate reuses the Aᵀy the trial
                    # already forms — no extra matvec.
                    def cond(c):
                        eta_c, _, eta_bar_c, _, _ = c
                        return eta_c > eta_bar_c

                    def body(c):
                        eta_c, _, eta_bar_c, _, rc_c = c
                        eta_s = jnp.minimum((1.0 - ip1 ** (-0.3)) * eta_bar_c, eta_c)
                        cand_s, eta_bar_s, Ax_new_s, _ = trial(eta_s, state, k, Ax_old)
                        return (eta_s, cand_s, eta_bar_s, Ax_new_s, rc_c)

                    cand0, eta_bar0, Ax_new0, rc0 = trial(eta, state, k, Ax_old)
                    eta0, cand, eta_bar, Ax_new, _ = jax.lax.while_loop(
                        cond, body, (eta, cand0, eta_bar0, Ax_new0, rc0)
                    )

                    # Gate on the CURRENT iterate. pdhg carries Ax_old =
                    # A @ state.primal and exposes rc0 = c + Aᵀy at `state`, so
                    # both the PFR and DFR conditions are free. Halpern is excluded
                    # even though it carries Ax: its convergence is driven by the
                    # anchor schedule (z_{k+1}=lam·z_0+(1-lam)·T(z_k)) and per-epoch
                    # reanchoring, so freezing the tail mid-cycle discards anchor
                    # progress and COSTS epochs (a KKT-residual trip doesn't imply
                    # the anchor average has settled). Extragradient carries no Ax
                    # (Ax_old is None) so it never gates either. Trigger-only, same
                    # as the synchronous path — the host runs the full certificate.
                    if gate_enabled and Ax_old is not None and not halpern:
                        gate_bool = gate_tripped(Ax_old, rc0, state)
                    else:
                        gate_bool = False

                    if halpern:
                        # Halpern anchor: blend T(z_k)=cand back toward z_0.
                        # lambda_k = 1/(k_local+1) where k_local is the
                        # restart-shifted iteration index `i` (== i_global -
                        # restart_i_offset, reset to ~1 each cycle by `solve`), so
                        # lambda decays as the cycle progresses and re-warms toward
                        # 1/2 at each restart. The anchor z_0 itself is reset to
                        # the cycle-start iterate in `solve`. Convex combination of
                        # two feasible iterates stays feasible — no re-projection.
                        k_local = jnp.asarray(i, eta.dtype)
                        lam = 1.0 / (k_local + 1.0)
                        new_state = jax.tree.map(
                            lambda z0, tz: lam * z0 + (1.0 - lam) * tz,
                            anchor,
                            cand,
                        )
                        # Advance the matvec carry through the blend by linearity
                        # of A: A @ new_primal = lam Ax_anchor + (1-lam) Ax_new.
                        # One axpy replaces next iteration's A @ state.primal.
                        Ax_state_next = lam * Ax_anchor + (1.0 - lam) * Ax_new
                    else:
                        new_state = cand

                    # Advance eta for the next iterate (growth allowed once the
                    # step is accepted). When eta_bar is +inf the step did not move
                    # (e.g. pinned on the box): hold eta rather than letting the
                    # growth branch run away to NaN.
                    eta_next = jnp.minimum(
                        (1.0 - ip1 ** (-0.3)) * eta_bar,
                        (1.0 + ip1 ** (-0.6)) * eta0,
                    )
                    eta_next = jnp.where(jnp.isfinite(eta_bar), eta_next, eta0)
                    eta_next = jnp.where(jnp.isfinite(eta_next), eta_next, eta0)
                    eta_next = jnp.maximum(eta_next, 1e-12)
                    if halpern:
                        opt_state = pack_k(
                            opt_state,
                            k,
                            eta_next,
                            anchor=anchor,
                            Ax=Ax_state_next,
                            Ax_anchor=Ax_anchor,
                        )
                    else:
                        # Carry A @ new_state.primal (== Ax_new, since new_state ==
                        # cand on the pdhg path) for the next iteration's Ax_old.
                        opt_state = pack_k(opt_state, k, eta_next, Ax=Ax_new)

                    if average:
                        w = weight_function(i)
                        total_weight = total_weight + w
                        average_state = optax.incremental_update(
                            new_state, average_state, w / total_weight
                        )

                    # gate_bool (computed above from Ax_old) is the per-step trip;
                    # the wrapper OR-latches it into converged_flag. Paths without
                    # a carried Ax (extragradient) emit False.
                    return (
                        i + 1,
                        new_state,
                        average_state,
                        opt_state,
                        total_weight,
                    ), gate_bool

                return step

            if (adaptive_step and update_mode == "pdhg") or halpern:
                # PDHG raw step: primal first, then dual reads the EXTRAPOLATED
                # primal x_bar = 2 x_new - x_old. interaction = dyᵀ A dx (one A·dx).
                # Halpern uses this same PDHG operator as its base T(z); the anchor
                # combination is applied in make_adaptive_step.
                def _trial(eta, state, k, Ax_old):
                    tau = eta / k
                    sigma = eta * k
                    gp = grad_primal_only(state)
                    x_new = projection_primal(state.primal - tau * gp)
                    # Ax_old = A @ state.primal is carried from the previous
                    # iteration (pdhg: last iter's A @ x_new; halpern: the
                    # anchor-blended lam·Ax_anchor + (1-lam)·Ax_new), saving one
                    # matvec. Only Ax_new = A @ x_new is computed here.
                    # A_dx = Ax_new - Ax_old; x_bar = 2*x_new - x_old so
                    # A @ x_bar = Ax_old + 2 * A_dx.
                    if Ax_old is None:
                        Ax_old = lp.A @ state.primal
                    Ax_new = lp.A @ x_new
                    A_dx = Ax_new - Ax_old
                    Ax_bar = Ax_old + 2.0 * A_dx
                    # Dual variables are unchanged from state at this point, so
                    # state can be passed directly — grad_dual_only_from_Ax only
                    # reads state.dual_eq / state.dual_ineq.
                    gd_ineq, gd_eq = grad_dual_only_from_Ax(Ax_bar, state)
                    dual_ineq = projection_non_negative(
                        state.dual_ineq - sigma * gd_ineq
                    )
                    dual_eq = state.dual_eq - sigma * gd_eq
                    cand = SaddleState(
                        primal=x_new, dual_ineq=dual_ineq, dual_eq=dual_eq
                    )
                    dy = jnp.concatenate(
                        [dual_eq - state.dual_eq, dual_ineq - state.dual_ineq]
                    )
                    interaction = jnp.abs(jnp.vdot(dy, A_dx))
                    # gp = c + Aᵀd (+ damping) is the reduced cost at `state`;
                    # returned so the gate's DFR test reuses it (no extra matvec).
                    return cand, _descent_bound(state, cand, k, interaction), Ax_new, gp

                step = make_adaptive_step(_trial)

            elif adaptive_step and update_mode == "extragradient":
                # Extragradient (Korpelevich) raw step with a Malitsky-Tam local
                # Lipschitz line search. The look-ahead and corrector already
                # evaluate the gradient twice, so the local Lipschitz estimate
                #     L_hat = ‖g_half - g‖_w / ‖z_half - z‖_w
                # (w = the k-weighted norm: k on the primal block, 1/k on the dual)
                # comes with NO extra matvec, unlike the pdhg family's A·dx. The
                # extragradient step is admissible while eta · L_hat <= 1/sqrt(2)
                # (Malitsky-Tam), so the largest admissible base step is
                #     eta_bar = (1/sqrt(2)) / L_hat.
                # The shared retry loop shrinks eta toward eta_bar; the corrector
                # is taken at the accepted eta, evaluated at the original state
                # (Korpelevich convention).
                _MT = 1.0 / jnp.sqrt(2.0)

                def _trial(eta, state, k, Ax_old):
                    # Extragradient doesn't carry Ax (its line search is matvec-free
                    # and it isn't the default path); Ax_old is None, Ax_new is None.
                    tau = eta / k
                    sigma = eta * k
                    g = grad(state)
                    # Look-ahead z_half = proj(z - step ∘ g): descend primal,
                    # subtract the optax-convention dual gradient (matches the
                    # non-adaptive extragradient / pdhg sign).
                    xh = projection_primal(state.primal - tau * g.primal)
                    yh_ineq = projection_non_negative(
                        state.dual_ineq - sigma * g.dual_ineq
                    )
                    yh_eq = state.dual_eq - sigma * g.dual_eq
                    state_half = SaddleState(
                        primal=xh, dual_ineq=yh_ineq, dual_eq=yh_eq
                    )
                    g_half = grad(state_half)
                    # Corrector from the ORIGINAL state using the look-ahead grad.
                    x_new = projection_primal(state.primal - tau * g_half.primal)
                    dual_ineq = projection_non_negative(
                        state.dual_ineq - sigma * g_half.dual_ineq
                    )
                    dual_eq = state.dual_eq - sigma * g_half.dual_eq
                    cand = SaddleState(
                        primal=x_new, dual_ineq=dual_ineq, dual_eq=dual_eq
                    )

                    # Local Lipschitz estimate in the k-weighted norm, measured on
                    # the look-ahead displacement (the same z used for g, g_half).
                    def _wnorm2(p, de, di):
                        return k * jnp.vdot(p, p) + (1.0 / k) * (
                            jnp.vdot(de, de) + jnp.vdot(di, di)
                        )

                    dg2 = _wnorm2(
                        g_half.primal - g.primal,
                        g_half.dual_eq - g.dual_eq,
                        g_half.dual_ineq - g.dual_ineq,
                    )
                    dz2 = _wnorm2(
                        xh - state.primal,
                        yh_eq - state.dual_eq,
                        yh_ineq - state.dual_ineq,
                    )
                    # eta_bar = (1/sqrt2) ‖dz‖ / ‖dg‖. No gradient change (dg2 -> 0)
                    # means the step is locally unconstrained: flag with +inf so
                    # the retry accepts and eta-growth is suppressed.
                    eta_bar = jnp.where(dg2 > 0.0, _MT * jnp.sqrt(dz2 / dg2), jnp.inf)
                    # 4th value = reduced cost at `state` (g.primal), for tuple-
                    # arity parity with the pdhg trial. Unused here: extragradient
                    # carries no Ax so its step never gates.
                    return cand, eta_bar, None, g.primal

                step = make_adaptive_step(_trial)

            elif update_mode == "alternating":

                def step(carry, _):
                    i, state, average_state, opt_state, total_weight = carry
                    opt_state, k = unpack_k(opt_state)

                    # 1) Primal-only update. Only the primal gradient is needed
                    #    here, so compute just c + Aᵀd (one matvec) instead of the
                    #    full grad (which would also do A @ x for an unused dual).
                    gp = grad_primal_only(state)
                    if k_scaling:
                        gp = gp / k
                    primal_gradient = SaddleState(
                        primal=gp,
                        dual_ineq=jnp.zeros_like(state.dual_ineq),
                        dual_eq=jnp.zeros_like(state.dual_eq),
                    )
                    primal_updates, _ = opt_update(primal_gradient, opt_state, state)
                    state = apply_updates(state, keep_only_primal(primal_updates))
                    state = SaddleState(
                        primal=projection_primal(state.primal),
                        dual_ineq=state.dual_ineq,
                        dual_eq=state.dual_eq,
                    )

                    # 2) Dual-only update (post-primal dual gradients). Only the
                    #    dual gradient is needed, so compute just b - Ax (one
                    #    matvec) instead of the full grad.
                    gd_ineq, gd_eq = grad_dual_only(state)
                    if k_scaling:
                        gd_ineq = gd_ineq * k
                        gd_eq = gd_eq * k
                    combined_gradient = SaddleState(
                        primal=gp,
                        dual_ineq=gd_ineq,
                        dual_eq=gd_eq,
                    )
                    dual_updates, opt_state = opt_update(
                        combined_gradient, opt_state, state
                    )
                    state = apply_updates(state, keep_only_dual(dual_updates))
                    state = SaddleState(
                        primal=state.primal,
                        dual_ineq=projection_non_negative(state.dual_ineq),
                        dual_eq=state.dual_eq,
                    )
                    opt_state = pack_k(opt_state, k)

                    # `average` is a Python-level static, so when False the
                    # incremental_update is dropped from the hot loop entirely.
                    if average:
                        w = weight_function(i)
                        total_weight = total_weight + w
                        average_state = optax.incremental_update(
                            state, average_state, w / total_weight
                        )

                    return (i + 1, state, average_state, opt_state, total_weight), False

            elif update_mode == "pdhg":
                # Chambolle-Pock PDHG: identical to `alternating` (Gauss-Seidel
                # primal-then-dual), except the dual gradient is evaluated at the
                # EXTRAPOLATED primal x_bar = 2 x^{k+1} - x^k instead of at
                # x^{k+1}. That over-relaxation is the only thing separating plain
                # Arrow-Hurwicz from true PDHG, and it is what lifts the
                # step-size restriction / buys the O(1/k) convergence. It costs
                # one axpy on the primal (no extra matvec): the dual gradient
                # b - A x_bar is linear in the primal, so we feed grad() a state
                # whose primal is x_bar.
                def step(carry, _):
                    i, state, average_state, opt_state, total_weight = carry
                    opt_state, k = unpack_k(opt_state)

                    x_old = state.primal

                    # 1) Primal-only update (same as alternating): c + Aᵀd, one
                    #    matvec.
                    gp = grad_primal_only(state)
                    if k_scaling:
                        gp = gp / k
                    primal_gradient = SaddleState(
                        primal=gp,
                        dual_ineq=jnp.zeros_like(state.dual_ineq),
                        dual_eq=jnp.zeros_like(state.dual_eq),
                    )
                    primal_updates, _ = opt_update(primal_gradient, opt_state, state)
                    state = apply_updates(state, keep_only_primal(primal_updates))
                    state = SaddleState(
                        primal=projection_primal(state.primal),
                        dual_ineq=state.dual_ineq,
                        dual_eq=state.dual_eq,
                    )

                    # 2) Dual-only update, but the dual gradient reads the
                    #    extrapolated primal x_bar = 2 x^{k+1} - x^k. Just b - A
                    #    x_bar (one matvec) — the primal half of the full grad is
                    #    unused here.
                    x_bar = 2.0 * state.primal - x_old
                    extrapolated = SaddleState(
                        primal=x_bar,
                        dual_ineq=state.dual_ineq,
                        dual_eq=state.dual_eq,
                    )
                    gd_ineq, gd_eq = grad_dual_only(extrapolated)
                    if k_scaling:
                        gd_ineq = gd_ineq * k
                        gd_eq = gd_eq * k
                    combined_gradient = SaddleState(
                        primal=gp,
                        dual_ineq=gd_ineq,
                        dual_eq=gd_eq,
                    )
                    dual_updates, opt_state = opt_update(
                        combined_gradient, opt_state, state
                    )
                    state = apply_updates(state, keep_only_dual(dual_updates))
                    state = SaddleState(
                        primal=state.primal,
                        dual_ineq=projection_non_negative(state.dual_ineq),
                        dual_eq=state.dual_eq,
                    )
                    opt_state = pack_k(opt_state, k)

                    if average:
                        w = weight_function(i)
                        total_weight = total_weight + w
                        average_state = optax.incremental_update(
                            state, average_state, w / total_weight
                        )

                    return (i + 1, state, average_state, opt_state, total_weight), False

            elif extragradient:
                # Extragradient (Korpelevich) using jo.extragradient's two-call
                # protocol, but routing gradients through the user-supplied
                # optimiser for adaptive scaling (adam etc.). This gives the
                # stabilising effect of the base optimiser plus the corrector
                # step's second gradient evaluation.
                #
                # Each iteration:
                #   Look-ahead: pass g at state through optimiser → la_updates,
                #       la_opt_state (non-committed); state_half = proj(state +
                #       la_updates).
                #   Corrector:  pass g_half at state_half through the ORIGINAL
                #       opt_state (not la_opt_state) → corr_updates, opt_state
                #       (committed); state = proj(state + corr_updates).
                #
                # When k-scaling is on, a primal weight k rescales each gradient
                # by (1/k, k) for (primal, dual) before opt_update, so the
                # dual/primal step ratio is k**2. k is constant within the epoch
                # — initialised from k_init and rebalanced at each restart in
                # `solve` (PDLP-style), not adapted per iteration.
                def step(carry, _):
                    i, state, average_state, opt_state, total_weight = carry
                    opt_state, k = unpack_k(opt_state)

                    # --- Look-ahead gradient ---
                    g = grad(state)

                    # Look-ahead: run the user's optimiser on g at state to
                    # get the look-ahead point. la_opt_state is NOT committed —
                    # we discard it and reuse the original opt_state for the
                    # corrector so that momentum/statistics only advance once.
                    scaled_g = scale_by_k(g, k) if k_scaling else g
                    la_updates, _ = opt_update(scaled_g, opt_state, state)
                    state_half = apply_updates(state, la_updates)
                    state_half = SaddleState(
                        primal=projection_primal(state_half.primal),
                        dual_ineq=projection_non_negative(state_half.dual_ineq),
                        dual_eq=state_half.dual_eq,
                    )

                    # Corrector: run the user's optimiser on g_half at
                    # state_half, but applied from original state (Korpelevich
                    # convention). opt_state IS committed here.
                    g_half = grad(state_half)
                    scaled_g_half = scale_by_k(g_half, k) if k_scaling else g_half
                    corr_updates, opt_state = opt_update(
                        scaled_g_half, opt_state, state
                    )
                    state = apply_updates(state, corr_updates)
                    state = SaddleState(
                        primal=projection_primal(state.primal),
                        dual_ineq=projection_non_negative(state.dual_ineq),
                        dual_eq=state.dual_eq,
                    )
                    opt_state = pack_k(opt_state, k)

                    if average:
                        w = weight_function(i)
                        total_weight = total_weight + w
                        average_state = optax.incremental_update(
                            state, average_state, w / total_weight
                        )

                    return (i + 1, state, average_state, opt_state, total_weight), False

            else:

                def step(carry, _):
                    i, state, average_state, opt_state, total_weight = carry
                    opt_state, k = unpack_k(opt_state)

                    # `grad` already forms Ax = A @ state.primal; reuse it for the
                    # per-iteration primal-feasibility gate (no extra matvec). The
                    # gate reads the CURRENT (pre-update) iterate — the host
                    # re-check on the average is authoritative, so an early/late
                    # trip is harmless.
                    if gate_enabled:
                        g, Ax = grad(state, return_Ax=True)
                        # g.primal = c + Aᵀy + damping is the reduced cost.
                        gate_bool = gate_tripped(Ax, g.primal, state)
                    else:
                        g = grad(state)
                        gate_bool = False
                    if k_scaling:
                        g = scale_by_k(g, k)
                    updates, opt_state = opt_update(g, opt_state, state)
                    state = apply_updates(state, updates)
                    state = SaddleState(
                        primal=projection_primal(state.primal),
                        dual_ineq=projection_non_negative(state.dual_ineq),
                        dual_eq=state.dual_eq,
                    )
                    opt_state = pack_k(opt_state, k)

                    # `average` is a Python-level static, so when False the
                    # incremental_update is dropped from the hot loop entirely.
                    if average:
                        w = weight_function(i)
                        total_weight = total_weight + w
                        average_state = optax.incremental_update(
                            state, average_state, w / total_weight
                        )

                    return (
                        i + 1,
                        state,
                        average_state,
                        opt_state,
                        total_weight,
                    ), gate_bool

            # Fixed iteration count per epoch: lax.scan (static `max_iter`) lets
            # XLA pipeline the loop body better than a while_loop whose only exit
            # condition is `i < end_iter`. `max_iter` is a static_argname, so a
            # changing iterations_per_epoch (restart decay) triggers a recompile —
            # the same tradeoff the convex solver already takes.
            #
            # scan requires the carry's output dtypes to match its input dtypes
            # exactly. optax's inject_hyperparams carries an `is_initial_step`
            # flag that is int at init but bool after the first update; some
            # optimisers (e.g. optimistic_gradient_descent) thus drift the carry
            # dtype on iteration 1, which scan rejects (while_loop tolerated it).
            # Cast each step's output carry back to the input carry's dtypes so
            # the loop is type-stable regardless of the user optimiser.
            # Adaptive PDHG carries A @ x in the k-slot to save a matvec per
            # iteration. The slot enters as (k, eta) from `solve` (which never sees
            # Ax); seed Ax = A @ x_old here (one matvec/epoch — negligible) and
            # strip it from the returned opt_state so the caller's contract is
            # unchanged. Halpern's caller-facing slot is (k, eta, anchor); seed
            # Ax_anchor = A @ anchor.primal and Ax_state = A @ state.primal the
            # same way (two matvecs/epoch). Re-seeding every epoch also resets any
            # rounding drift the blended carry accumulated within the prior epoch.
            if carry_Ax:
                inner, (k0, eta0) = opt_state
                opt_state = (inner, (k0, eta0, lp.A @ state.primal))
            if halpern:
                inner, (k0, eta0, anchor0) = opt_state
                opt_state = (
                    inner,
                    (
                        k0,
                        eta0,
                        anchor0,
                        lp.A @ anchor0.primal,
                        lp.A @ state.primal,
                    ),
                )

            step_carry = (start_iter, state, average_state, opt_state, total_weight)
            _carry_dtypes = jax.tree.map(lambda x: jnp.asarray(x).dtype, step_carry)

            # The scan carry adds a latching converged_flag as its 6th element.
            # The inner `step` variants stay 5-tuple functions; `step_typed`
            # unpacks the flag, runs the step, OR-latches the step's gate output,
            # and — once latched — freezes state/average/opt/total_weight to the
            # pre-step values so the remaining iterations are no-ops (the loop
            # index still advances so `i` reflects the true iteration count). `i`
            # is intentionally NOT frozen so a converged epoch still reports where
            # it stopped. converged_flag rides as a scalar bool; seed it False.
            converged_flag0 = jnp.asarray(False)
            init_carry = (*step_carry, converged_flag0)

            def step_typed(carry, _):
                *inner_carry, converged_flag = carry
                new_inner, gate_bool = step(tuple(inner_carry), _)
                new_inner = jax.tree.map(
                    lambda v, dt: v.astype(dt), new_inner, _carry_dtypes
                )
                # Freeze everything except the loop index once converged. The
                # index (element 0) always advances; elements 1.. freeze.
                i_new = new_inner[0]
                frozen_tail = freeze_leaf(
                    converged_flag, tuple(inner_carry[1:]), tuple(new_inner[1:])
                )
                new_flag = jnp.logical_or(
                    converged_flag, jnp.asarray(gate_bool, converged_flag.dtype)
                )
                return (i_new, *frozen_tail, new_flag), None

            scan_out, _ = jax.lax.scan(
                step_typed,
                init_carry,
                None,
                length=max_iter,
            )
            i, state, average_state, opt_state, total_weight, converged_flag = scan_out

            if carry_Ax:
                inner, (k0, eta0, _Ax) = opt_state
                opt_state = (inner, (k0, eta0))
            if halpern:
                inner, (k0, eta0, anchor0, _AxA, _AxS) = opt_state
                opt_state = (inner, (k0, eta0, anchor0))

            return i, state, average_state, opt_state, total_weight, converged_flag

        _LINEAR_RUN_EPOCH_CACHE[cache_key] = run_epoch

    state = initial_solution

    if initial_avg_state is not None:
        average_state = initial_avg_state
    else:
        average_state = initial_solution

    # `state` is donated to run_epoch; if `average_state` aliases the same buffer
    # (the common `average=False` case) XLA rejects the call (`f(donate(a), a)`).
    # Give `average_state` its own buffer. One copy per epoch, off the hot path.
    if average_state is state:
        average_state = jax.tree.map(lambda x: x + 0, average_state)

    if initial_opt_state is not None:
        opt_state = initial_opt_state
    elif k_scaling:
        # Pack the primal weight k alongside the optax state. In adaptive_step
        # mode the slot is a (k, eta) pair so the per-iteration step rides along.
        dtype = initial_solution.primal.dtype
        if halpern:
            # Halpern carries the anchor z_0 (the cycle-start iterate) in the
            # k-slot. On a bare __sps call the anchor seeds from the incoming
            # state; across a restart cycle `solve` threads it via opt_state.
            k_slot = (
                jnp.asarray(k_init, dtype),
                jnp.asarray(adaptive_eta, dtype),
                jax.tree.map(lambda x: x + 0, initial_solution),
            )
        elif adaptive_step:
            k_slot = (jnp.asarray(k_init, dtype), jnp.asarray(adaptive_eta, dtype))
        else:
            k_slot = jnp.asarray(k_init, dtype)
        opt_state = (optimiser.init(initial_solution), k_slot)
    else:
        opt_state = optimiser.init(initial_solution)

    # run_epoch DONATES `state` and `opt_state`. The caller's restart path may
    # hand us a `state`/`opt_state` pair that shares buffers: `state =
    # restart_point` aliases the live `average_state`, and `optimiser.init(state)`
    # / inject_hyperparams reuse buffers tied to that same `state` (e.g. the
    # scalar learning-rate float64[]). Donating two args that alias one buffer
    # double-frees it, so the next epoch reads a deleted buffer. Break any such
    # aliasing with an independent copy of each donated tree — one cheap pass per
    # epoch, off the per-iteration hot path. (XLA elides the copy when there is
    # nothing to alias.)
    state = jax.tree.map(lambda x: x + 0, state)
    # jnp.copy (not `x + 0`) for opt_state: `+ 0` promotes boolean leaves such
    # as optax adadelta's `is_initial_step` from bool to int32, breaking the
    # scan/while carry dtype match.
    opt_state = jax.tree.map(jnp.copy, opt_state)

    return run_epoch(
        start_iter,
        state,
        average_state,
        opt_state,
        total_weight,
        max_iter=max_iter,
    )


def set_saddle_lrs(opt_state, primal_lr, dual_lr):
    """Overwrite the injected ``learning_rate`` hyperparameters in a
    ``create_saddle_optimiser`` (``optax.partition`` over ``"primal_opt"`` /
    ``"dual_opt"``) state without changing its tree structure, so the jitted
    epoch loop is not retraced.

    Requires the two sub-optimisers to be built with
    ``optax.inject_hyperparams(...)(learning_rate=...)`` so the learning rate is
    a live array leaf rather than a baked-in schedule closure.
    """
    inner = dict(opt_state.inner_states)

    def _set(sub, lr):
        hp = dict(sub.inner_state.hyperparams)
        hp["learning_rate"] = jnp.asarray(lr)
        return sub._replace(inner_state=sub.inner_state._replace(hyperparams=hp))

    inner["primal_opt"] = _set(inner["primal_opt"], primal_lr)
    inner["dual_opt"] = _set(inner["dual_opt"], dual_lr)
    return opt_state._replace(inner_states=inner)


def solve(
    lp: "JaddleLP | LP",
    optimiser=None,
    max_epochs=None,
    initial_solution=None,
    initial_opt_state=None,
    iterations_per_epoch=int(1e3),
    dual_damping_ineq=0.0,
    dual_damping_eq=0.0,
    primal_damping=0.0,
    primal_feasibility_tolerance=1e-3,
    dual_feasibility_tolerance=1e-3,
    dual_gap_tolerance=1e-3,
    weight_function=lambda _: 1.0,
    verbose=False,
    log_every=1,
    average=True,
    report_best=True,
    update_mode="pdhg",
    k_scale=None,
    k_theta=0.5,
    k_init=None,
    k_update_per_epoch=True,
    adaptive_eta=None,
    scale="ruiz+pc",
    scaled_objective=False,
    restarts=0,
    epochs_per_restart=10,
    restart_multiplier=1.0,
    restart_decay=0.2,
    necessary_decay=0.8,
    primal_stop=False,
    primal_stop_window=5,
    primal_stop_obj_tol=1e-4,
    halpern_reanchor_per_epoch=False,
    iterations_per_epoch_decay=1.0,
    iterations_per_epoch_min=100,
    eq_projection_threshold=None,
    vertex_bias=0.0,
    vertex_bias_seed=0,
    reference_objective=None,
    per_iter_gate=True,
    gate_tol=None,
    feasibility_polish=False,
    polish_gap_slack=10.0,
    polish_residual_slack=10.0,
    polish_stall_window=10,
    polish_stall_ratio=0.5,
    polish_max_epochs=20,
    polish_max_attempts=2,
    polish_cooldown=10,
):
    """
    Solve a linear program via saddle-point optimisation.

    Termination uses the standard LP optimality certificate, all tested
    RELATIVELY (PDLP/HiGHS convention): primal feasibility
    (``primal_feasibility_tolerance``, normalised by 1+‖b‖), dual feasibility
    (``dual_feasibility_tolerance``, normalised by 1+‖c‖), and a finite duality
    gap within ``dual_gap_tolerance`` (normalised by 1+|primal_obj|+|dual_obj|,
    the PDLP/cuPDLP convention, so RDG is directly comparable to PDLP).

    Adaptive restarts (PDLP-style) accelerate ill-conditioned problems. A
    restart resets the optimiser momentum/averaging while keeping the current
    iterate as a warm start, which prevents the saddle iteration from settling
    into slow rotational orbits. A restart fires when either the normalised KKT
    merit decays past ``restart_decay`` of its value at the last restart
    (sufficient-progress restart) or the current cycle reaches its length cap
    (no-progress restart). Set ``restarts=0`` to disable.

    Args:
        restarts: Maximum number of warm restarts. 0 = no restarts (default).
            Each restart resets the optimiser state (momentum) and averaging
            while keeping the current iterate as a warm start. The LR schedule /
            ``weight_function`` iteration counter also restarts from its initial
            value.
        epochs_per_restart: Length cap of the first restart cycle, expressed in
            epochs AT THE DEFAULT ``iterations_per_epoch`` (default 10) but
            internally converted to and tracked in ITERATIONS
            (``epochs_per_restart * iterations_per_epoch``), so the cycle-cap
            restart fires at the same point in the optimisation trajectory
            regardless of ``iterations_per_epoch``. This matters because a
            restart is destructive (it wipes PDHG momentum/averaging via
            ``optimiser.init``): tying the cap to a raw epoch count made the
            restart cadence an accident of how the iteration budget was chopped
            into epochs — e.g. on momentum1, ``iterations_per_epoch=1000``
            triggered a cap-exhaustion restart at 10,000 iterations while still
            dual-infeasible, wiping out a trajectory that would otherwise have
            converged smoothly, while ``iterations_per_epoch=10000`` reached
            full convergence in under 100,000 iterations before the same cap
            ever fired. Subsequent cycle caps grow by ``restart_multiplier``.
        restart_multiplier: Geometric growth factor for cycle-length caps
            (default 1.0 = fixed length, 2.0 = doubling).
        restart_decay: Sufficient-progress threshold (default 0.2; cuPDLP
            β_sufficient). A restart fires when the KKT merit drops below
            ``restart_decay`` times its value at the last restart.
        necessary_decay: Necessary-decay threshold for the cuPDLP "stalling"
            restart condition (default 0.8; cuPDLP β_necessary). A restart also
            fires when the merit has decayed below ``necessary_decay`` times its
            cycle-start value AND has risen versus the previous epoch (rotational
            turnaround). Must be > ``restart_decay`` to be meaningful.
        update_mode: Selects the per-iterate stepping scheme. One of:
            * ``"synchronous"`` (default): simultaneous primal/dual gradient
              descent-ascent through the user optimiser.
            * ``"alternating"``: primal step, then a dual step using the
              post-primal dual gradient (Gauss–Seidel ordering).
            * ``"extragradient"``: Korpelevich look-ahead/corrector step. Two
              gradient evals per iter.
            * ``"pdhg"``: Chambolle–Pock PDHG (primal step then dual step on the
              extrapolated primal x_bar = 2x^{k+1} − x^k).
            * ``"halpern"``: restarted Halpern-anchored PDHG. Each iterate is the
              adaptive PDHG step T(z) blended back toward an anchor z_0:
              ``z_{k+1} = lambda_k z_0 + (1−lambda_k) T(z_k)``, ``lambda_k =
              1/(k+1)`` (cycle-local k). The anchor z_0 and lambda counter reset
              to the current iterate at each restart, giving last-iterate
              acceleration. Implies the ``adaptive_eta`` line search on the inner
              PDHG operator, so it requires both ``adaptive_eta`` and ``k_scale``;
              best paired with ``restarts > 0``.
        k_scale: Primal-weight (k) scaling control. ``None`` disables it;
            otherwise a float sets a symmetric clamp band ``[1/k_scale,
            k_scale]`` for ``k`` (default ``10`` → ``[0.1, 10]``). When enabled —
            orthogonal to ``update_mode``, so it composes with all three schemes
            — a primal weight ``k`` rescales the primal/dual gradients by
            ``(1/k, k)`` before each ``opt_update``, making the dual/primal step
            ratio ``k**2``. ``k`` is initialised from ``k_init`` and rebalanced
            at each restart (PDLP-style) from primal-vs-dual iterate movement; it
            is constant within an epoch (not adapted per iteration). Tuned by
            ``k_theta``/``k_scale`` and ``k_init``.
        k_init: Initial primal weight ``k``. ``None`` (default) initialises it to
            the PDLP heuristic ``||c|| / ||b||`` (objective vs RHS norms, in the
            scaled space the solver iterates in). Pass a float to override
            (``1.0`` = symmetric steps, the PC/Ruiz-scaled baseline). Only used
            when ``k_scale`` is set.
        k_update_per_epoch: When ``True`` (default), the primal weight ``k`` is
            rebalanced at every epoch boundary (not only at restarts) using the
            primal-vs-dual iterate movement over the just-finished epoch — same
            log-space geometric-mean blend (``k_theta``) and ``[1/k_scale,
            k_scale]`` clamp as the restart rebalance. Unlike a restart it does
            NOT reset momentum/averaging or (for halpern) re-anchor ``z_0`` / reset
            ``eta``; only the ``k`` component of the optimiser state changes, so
            the gradient reweighting tracks the local primal/dual progress within a
            restart cycle. Active only for ``update_mode`` in
            ``{"pdhg", "extragradient", "halpern"}`` with ``k_scale`` set. Set
            ``False`` to keep the prior behaviour (k frozen between restarts).
        adaptive_eta: Enables a cuPDLP-style per-iteration adaptive step size with
            a line search. ``None`` (default) keeps the optimiser's fixed learning
            rate. A float seeds a single scalar base step ``eta`` that drives the
            primal step ``tau = eta / k`` and dual step ``sigma = eta * k``. Each
            iteration takes a trial step, forms the largest admissible step, and
            rejects + shrinks ``eta`` if the trial overshot; ``eta`` is then
            advanced with a two-sided guard. The optimiser's adam/adadelta scaling
            is bypassed in the hot loop. Supported for ``update_mode='pdhg'`` (the
            admissible step comes from the interaction term
            ``(y^{k+1}-y^k)ᵀ A (x^{k+1}-x^k)``, one extra matvec) and
            ``update_mode='extragradient'`` (a Malitsky-Tam local-Lipschitz test
            from the two gradients it already evaluates, no extra matvec). Other
            modes raise. Requires ``k_scale`` set. ``eta`` resets to this seed at
            each restart. A reasonable seed is ``1.0`` on Ruiz+PC-scaled problems.
        k_theta: Smoothing coefficient for the log-space primal-weight update at
            each restart (default 0.5 = geometric mean of the movement-based
            target and the current weight, matching PDLP). Smaller = slower
            adaptation. Only used when ``k_scale`` is set.
        primal_stop: Opt-in, dual-free termination (default ``False``). When
            ``True``, termination ignores the dual certificate entirely and stops
            on **primal feasibility** (``constraint_bound`` within
            ``primal_feasibility_tolerance``) **and** an **objective stall**. This
            is a heuristic, not an optimality certificate — it trades the dual's
            mathematical optimality guarantee for robust termination on problems
            where the dual is junk. The duality gap and dual residuals are still
            computed and reported as diagnostics, just not used to gate.
        primal_stop_window: Number of recent epochs over which the objective
            stall is measured (default 5). Only used when ``primal_stop=True``.
        primal_stop_obj_tol: Relative-change threshold for the objective stall:
            stop when ``|obj_now - obj_{window ago}| / (1 + |obj_now|)`` falls
            below this (default 1e-4). Only used when ``primal_stop=True``.
        halpern_reanchor_per_epoch: Opt-in (default ``False``, ``"halpern"`` only).
            When ``True``, re-anchor ``z_0`` to the current iterate and reset the
            ``lambda_k = 1/(k+1)`` counter at **every** epoch boundary, not only at
            restarts — i.e. each epoch starts a fresh Halpern cycle. This re-warms
            ``lambda`` toward 1/2 each epoch (strong pull to the cycle-start point)
            instead of letting it decay across the whole restart cycle. Note an
            epoch is a fixed iteration chunk, not a progress-driven boundary, so
            this re-anchors on a schedule rather than on convergence; it can help
            on rotation-limited problems but discards the long-horizon anchor that
            gives Halpern its last-iterate acceleration. Both ``eta`` and ``k`` are
            left to their usual per-epoch handling. No effect unless
            ``update_mode="halpern"``. On an epoch where a real restart fires, the
            restart's own re-anchor takes precedence (no double re-anchor).
        eq_projection_threshold: When set, after each epoch the unscaled equality
            residual is checked; if it exceeds this value the primal (and average)
            are projected onto the equality manifold ``A_eq x = b_eq`` via the
            precomputed factorisation of ``A_eq A_eq^T``. Default ``None``
            disables projection. Only useful when equality feasibility is the
            bottleneck; has no effect when there are no equality constraints.
        reference_objective: True optimal objective value ``z*`` from a reference
            solver, in the ORIGINAL problem's units. Purely diagnostic: when
            supplied and ``verbose=True``, each epoch also logs the true relative
            objective error ``|cᵀx − z*| / (1 + |z*|)`` (``OBJERR``) alongside the
            reported relative duality gap (``RDG``). The gap is gated by the dual
            (it is a complementarity sum that carries the dual's lag), so the
            primal typically reaches optimality well before the gap closes;
            comparing ``OBJERR`` against ``RDG`` quantifies how much of the gap is
            dual lag versus genuine primal suboptimality. Does not affect
            termination — convergence still uses the full LP certificate.
        per_iter_gate: When ``True`` (default), a cheap primal-feasibility TRIGGER
            is evaluated every iteration inside the scan, reusing the ``A @ x``
            the step already computes (no extra matvec). Once the current
            iterate's relative primal residual drops below ``gate_tol`` the flag
            latches and the remaining iterations of the epoch become no-ops
            (state/average frozen), so the epoch returns a converged iterate to
            the host instead of drifting past it. It is only a trigger: the host
            still runs the full 3-part certificate on the average iterate, so the
            gate can only cause an earlier host check, never a false stop. Active
            for ``synchronous`` and ``pdhg`` (both have ``A @ x`` and the reduced
            cost already in hand). ``halpern`` is excluded — its anchor schedule
            makes an early freeze counterproductive — and ``extragradient`` has no
            carried ``A @ x``, so neither is gated. The trip requires BOTH primal
            and dual feasibility (relative PFR and DFR), matching the certificate,
            so it can't fire on a primal-feasible-but-dual-lagging iterate.
        gate_tol: Threshold for the per-iteration gate's relative primal residual.
            Defaults to ``primal_feasibility_tolerance`` (the same threshold the
            certificate's PFR test uses).
        feasibility_polish: PDLP-style feasibility polishing (the "feasibility
            polishing" phase of the PDLP deployment paper). Saddle solves are
            frequently FEASIBILITY-tail-limited: the duality gap and one residual
            converge quickly while the other residual crawls down a sublinear
            tail. When one side of the certificate is the lone blocker, the main
            loop pauses and solves that side's far easier feasibility problem
            with a nested ``solve()``, warm-started from the current point:
            * primal polish — the original constraints with ``c = 0`` (the
              optimal dual is 0, so PDHG contracts fast), started at ``(x, 0)``;
            * dual polish — the homogenised problem with ``b = 0`` and every
              finite bound moved to 0 (bound classes, and hence the dual sign
              conditions, are preserved; the optimal primal is 0), started at
              ``(0, y)``.
            Polishing targets the FINISHING regime only — the lagging residual
            must already be within ``polish_residual_slack`` of tolerance.
            Deep plateaus (mzzv11's PFR~1e-1 wall) are out of scope: from
            there the warm start drags the trap into the sub-problem, and a
            cold-started feasible point is objective-agnostic so its
            recombined duality gap explodes (both measured; see the trigger
            and ``_attempt_polish`` comments).
            The polished side is recombined with the untouched other side and the
            FULL certificate is re-evaluated on the combination: certified →
            terminate; KKT merit improved → warm-start the main loop from it
            (restart-style reset, not counted against the ``restarts`` budget);
            otherwise the candidate is discarded. A sub-solve that exhausts its
            budget before certifying doubles the side's budget for the next
            attempt.
            A polish attempt can therefore never make the returned point worse.
            Each attempt is a nested solve on the modified LP and pays its own
            scaling + XLA compile (the epoch-fn cache is keyed on the LP object).
        polish_gap_slack: Gap condition (necessary): polishing fires only once
            ``RDG <= polish_gap_slack * dual_gap_tolerance`` (the gap is
            essentially there; feasibility is the blocker). Default 10.
        polish_residual_slack: Finishing-regime condition (necessary): the
            lagging residual must satisfy ``residual <= polish_residual_slack *
            its_tolerance``, so attempts are not burned warm-starting from deep
            plateaus the sub-solve inherits. Default 10.
        polish_stall_window / polish_stall_ratio: Stall condition (necessary):
            the lagging residual has improved by less than a factor
            ``1/polish_stall_ratio`` over the last ``polish_stall_window``
            epochs. A residual still making progress converges cheaper by
            letting the main loop run (an eager gap-only trigger COST epochs on
            neos-1593097). Defaults 10 / 0.5 (less than 2x in 10 epochs).
        polish_max_epochs: Epoch budget of a side's first polish attempt; doubles
            after any attempt whose sub-solve ran out of budget. Default 20.
        polish_max_attempts: Maximum attempts per side. Default 2.
        polish_cooldown: Minimum epochs between attempts — and, since the counter
            starts at epoch 0, the earliest epoch a first attempt can fire.
            Default 10.

    Returns:
        dict: The solution together with diagnostics. Keys:
            * ``"solution"``: the ``SaddleState`` (primal/dual iterate), unscaled
              back to the original problem's units.
            * ``"converged"``: ``bool``, whether the solve terminated by meeting
              the LP optimality certificate or a ``primal_stop`` heuristic stop
              (see ``"stop_reason"`` to disambiguate).
            * ``"opt_state"``: the final optimiser state, for warm-starting a
              subsequent solve via ``initial_opt_state``.
            * ``"stop_reason"``: ``str`` recording *why* the solve terminated:
              ``"certificate"`` (full LP optimality certificate met),
              ``"primal_stall"`` (the ``primal_stop`` heuristic fired — feasible
              but not certified optimal, so the objective may be suboptimal even
              though ``"converged"`` is ``True``), ``"max_epochs"`` (epoch budget
              exhausted), or ``"interrupted"`` (KeyboardInterrupt).
            * ``"solve_seconds"``: ``float`` wall time of the epoch loop (incl. the
              first-epoch XLA compile but not the scaling / sparse-setup phase).
            * ``"corrected_seconds"``: ``float`` steady-state runtime with the
              one-off first-epoch XLA compile amortised out:
              ``n * (solve_seconds - first_epoch_seconds) / (n - 1)`` where ``n``
              is the epoch count. Falls back to ``solve_seconds`` when it can't be
              formed (fewer than two epochs).
            * ``"epochs"``: ``int``, number of epochs run.
            * ``"polish"``: feasibility-polishing diagnostics —
              ``{"primal_attempts", "dual_attempts", "adopted"}`` (all 0 when
              ``feasibility_polish`` is off).
    """

    if lp.A_ineq.shape[0] == 0:
        lp.A_ineq = jsp.BCOO.fromdense(
            jnp.zeros((1, lp.A_eq.shape[1]), dtype=lp.A_eq.dtype)
        )
        lp.b_ineq = jnp.zeros((1,), dtype=lp.b_eq.dtype)

    if lp.A_eq.shape[0] == 0:
        lp.A_eq = jsp.BCOO.fromdense(
            jnp.zeros((1, lp.A_ineq.shape[1]), dtype=lp.A_ineq.dtype)
        )
        lp.b_eq = jnp.zeros((1,), dtype=lp.b_ineq.dtype)

    if optimiser is None:
        optimiser = jo.gd(0.5)

    if log_every < 1:
        raise ValueError("log_every must be >= 1")

    if verbose:
        print("----------------------------------------------")

    valid_update_modes = [
        "synchronous",
        "alternating",
        "extragradient",
        "pdhg",
        "halpern",
    ]
    if update_mode not in valid_update_modes:
        raise ValueError(f"update_mode must be one of {valid_update_modes}")

    # ``k_scale`` is the public knob for primal-weight scaling: ``None`` disables
    # it, otherwise it sets a symmetric clamp band ``[1/k_scale, k_scale]``.
    k_scaling = k_scale is not None
    if k_scaling:
        k_lo, k_hi = 1.0 / k_scale, k_scale
    else:
        k_lo, k_hi = None, None

    # cuPDLP per-iteration adaptive step size (needs the primal weight k). When
    # on, the optimiser's fixed LR is bypassed in the hot loop and eta drives the
    # step directly via the per-iteration line search. Supported for the saddle
    # stepping schemes synchronous/alternating/pdhg/extragradient.
    # Halpern-anchored PDHG (restarted Halpern). Rides the adaptive PDHG step, so
    # it implies adaptive_step and requires adaptive_eta + k_scale. The anchor z_0
    # is reset to the cycle-start iterate at each restart.
    halpern = update_mode == "halpern"
    _adaptive_modes = ("pdhg", "extragradient")
    adaptive_step = (
        adaptive_eta is not None and update_mode in _adaptive_modes
    ) or halpern
    if halpern and adaptive_eta is None:
        raise ValueError("update_mode='halpern' requires adaptive_eta")
    if adaptive_eta is not None and not halpern:
        if update_mode not in _adaptive_modes:
            raise ValueError(
                f"adaptive_eta is only supported with update_mode in {_adaptive_modes} "
                "or 'halpern'"
            )
    if adaptive_step and not k_scaling:
        raise ValueError("adaptive_eta / halpern requires k_scale (primal weight k)")

    if verbose:
        print("====Starting Solve====")
        print("----------------------------------------------")

    # solve() takes a JaddleLP (the JAX-native, device-side representation). The
    # scaling stage runs host-side on scipy (build [[A,b],[c,0]], diag-apply), so
    # we materialise a scipy view once here; scaling returns a scipy LP that
    # to_jaddle_sparse() converts back to the JaddleLP the solver iterates on. A
    # raw scipy LP (e.g. hand-built in examples) is already in scipy form.
    if isinstance(lp, JaddleLP):
        lp = lp.to_scipy()

    # Feasibility polishing builds its sub-problems (c=0 / homogenised b=0) from
    # the ORIGINAL (unscaled) problem, so keep a reference before scaling rebinds
    # `lp`. The scaling functions construct a new LP (they never mutate their
    # input), so sharing the constraint-matrix references is safe.
    _polish_base_lp = lp if feasibility_polish else None

    if scale == "ruiz":
        lp, row_scale, col_scale = ruiz_scaling(lp)

        original_lp = lp
        lp = to_jaddle_sparse(lp)

        if verbose:
            print("Applied Ruiz scaling to the LP.")
            print("----------------------------------------------")

    elif scale == "pc":
        lp, row_scale, col_scale = pc_scaling(lp)

        original_lp = lp
        lp = to_jaddle_sparse(lp)

        if verbose:
            print("Applied PC scaling to the LP.")
            print("----------------------------------------------")

    elif scale == "ruiz+pc":
        # Augmented Ruiz: equilibrate [[A,b],[c,0]] so cost and RHS information also
        # drive the equilibration. Conditions the constraint (esp. equality) block
        # better on cost/RHS-dominated problems (momentum1: A-only Ruiz froze the
        # primal at a far-from-optimal point; augmented converges in ~15 epochs).
        # A-only Ruiz (ruiz_scaling(augmented=False)) is available as a knob but is
        # not the default — it broke both momentum1 and boeing once the relative
        # convergence test + true-units norm fixes were in place. PC then applies
        # its single Pock-Chambolle finishing pass.
        lp, row_scale_ruiz, col_scale_ruiz = ruiz_scaling(lp, augmented=True)
        lp, row_scale_pc, col_scale_pc = pc_scaling(lp)

        row_scale, col_scale = (
            row_scale_ruiz * row_scale_pc,
            col_scale_ruiz * col_scale_pc,
        )

        original_lp = lp
        lp = to_jaddle_sparse(lp)

        if verbose:
            print("Applied combined Ruiz + PC scaling to the LP.")
            print("----------------------------------------------")

    else:
        row_scale = np.ones(lp.A_eq.shape[0] + lp.A_ineq.shape[0])
        col_scale = np.ones(lp.c.shape[0])

        original_lp = lp
        lp = to_jaddle_sparse(lp)

    # Polish sub-solves re-derive the step-size seed on their own scaled
    # operator, so keep the caller's sentinel (0.0 = derive) before it is
    # resolved for the main problem below.
    _orig_adaptive_eta = adaptive_eta
    if adaptive_eta == 0.0:
        adaptive_eta = 1 / estimate_augmented_spectral_norm(lp)
        print(f"Adaptive step size seed set to 1/||A||_2 = {adaptive_eta:.3e}")
        print("----------------------------------------------")

    # A user-supplied initial_solution is given in the LP's original (unscaled)
    # space, so it must be mapped into the scaled space the solver iterates in.
    # The default from lp.initial_solution() is already built from the scaled lp
    # and must NOT be rescaled again.
    user_supplied_initial = initial_solution is not None
    if initial_solution is None:
        initial_solution = lp.initial_solution()

    # lp.initial_solution() allocates with the JAX default float width (f32, or
    # f64 under x64), which mismatches the profile dtype the LP data carries
    # (e.g. float16). Cast the state to match so the solve runs end-to-end in
    # the active precision instead of silently upcasting.
    _state_dtype = lp.c.dtype
    initial_solution = SaddleState(
        primal=initial_solution.primal.astype(_state_dtype),
        dual_ineq=initial_solution.dual_ineq.astype(_state_dtype),
        dual_eq=initial_solution.dual_eq.astype(_state_dtype),
    )

    row_scale_ineq = row_scale[len(lp.b_eq) :]
    row_scale_eq = row_scale[: len(lp.b_eq)]

    # Convert to jax arrays for use inside jitted functions. Match the state
    # dtype so dividing by the scales doesn't upcast the state back out of the
    # profile precision (numpy float64 scale * jax float16 -> float64).
    jnp_row_scale_ineq = jnp.array(row_scale_ineq, dtype=_state_dtype)
    jnp_row_scale_eq = jnp.array(row_scale_eq, dtype=_state_dtype)
    col_scale = jnp.asarray(col_scale, dtype=_state_dtype)

    # Precompute equality-constraint projection: x ← x - A_eq^T (A_eq A_eq^T)^{-1} (A_eq x - b_eq).
    # The factorisation is done once in scipy (scaled space); the apply is a cheap
    # pair of matvecs. Only built when eq_projection_threshold is set and there are
    # equality constraints.
    _eq_project = None
    if eq_projection_threshold is not None and lp.A_eq.shape[0] > 0:
        import scipy.sparse.linalg as spla

        _A_eq_sp = __convert_to_scipy(lp.A_eq)
        AeqAeqT = _A_eq_sp @ _A_eq_sp.T
        _eq_factor = spla.factorized(AeqAeqT.tocsc())
        _b_eq_np = np.array(lp.b_eq)

        def _eq_project(primal):
            # Run entirely in numpy/scipy to avoid materialising large JAX sparse
            # intermediates on the GPU. Pull the primal to CPU, project, push back.
            x = np.asarray(primal)
            residual = _A_eq_sp @ x - _b_eq_np
            correction = _eq_factor(residual)
            return jnp.array(x - _A_eq_sp.T @ correction)

    if scaled_objective:
        c_max = jnp.max(jnp.abs(lp.c))
        lp.c = lp.c / c_max

    else:
        c_max = 1.0

    # --- Vertex-biasing cost perturbation (Mangasarian tie-break) -------------
    # First-order saddle methods converge to the analytic centre of the optimal
    # FACE — the maximally-interior optimum — which is the worst possible warm
    # start for a vertex crossover (degenerate LPs then have far more "interior"
    # variables than rows; see [[crossover-polish]]). Adding a small perturbation
    # `c ← c + vertex_bias·r` to the cost used by the DYNAMICS breaks ties on the
    # optimal face so the solver settles on a unique VERTEX; for vertex_bias below
    # the LP's optimal-partition threshold that vertex is an exact optimal vertex
    # of the original problem. The convergence METRICS keep the TRUE cost
    # (`c_true` below), so we stop when the iterate is near-optimal for the real
    # LP while being pulled toward a vertex — and polish/crossover run against the
    # true cost too. Default 0.0 = off (unchanged behaviour).
    c_true = lp.c
    if vertex_bias:
        rng = np.random.default_rng(vertex_bias_seed)
        # Per-variable perturbation, scaled by |c| magnitude so the relative tilt
        # is uniform; deterministic given the seed. Sign random so it tilts each
        # variable toward whichever bound the face allows.
        r = jnp.asarray(
            rng.standard_normal(lp.c.shape[0]).astype(np.float64), dtype=lp.c.dtype
        )
        c_scale_mag = float(jnp.max(jnp.abs(lp.c))) + 1e-30
        lp.c = lp.c + (vertex_bias * c_scale_mag) * r

    if user_supplied_initial:
        # Map the user's original-space solution into scaled space (the inverse
        # of the output unscaling: primal *= col_scale, dual *= row_scale).
        initial_solution = SaddleState(
            primal=initial_solution.primal / col_scale,
            dual_ineq=initial_solution.dual_ineq / jnp_row_scale_ineq,
            dual_eq=initial_solution.dual_eq / jnp_row_scale_eq,
        )

    dual_feasibility_threshold = (
        float(dual_feasibility_tolerance)
        if dual_feasibility_tolerance is not None
        else 0.0
    )

    # When vertex_bias is off, the working cost IS the true cost — use lp.objective
    # exactly as before so the zero-bias compiled graph is byte-identical to the
    # pre-vertex_bias version (no extra captured constant). Only when biased do we
    # substitute c_true so the reported objective reflects the real problem.
    _report_c = c_true if vertex_bias else None

    @jax.jit
    def compute_epoch_metrics(average_state):
        # Report the TRUE objective the user cares about; the dynamics /
        # dual-feasibility / gap below run on the (possibly vertex-biased) working
        # cost lp.c, since that is the problem actually being solved.
        if _report_c is None:
            objective_value = lp.objective(average_state.primal) * c_max
        else:
            objective_value = (_report_c @ average_state.primal) * c_max

        dual_avg = jnp.concatenate([average_state.dual_eq, average_state.dual_ineq])
        Ax_avg = lp.A @ average_state.primal
        grad_primal = lp.c + lp.A_T @ dual_avg
        Ax_minus_b = Ax_avg - lp.b
        grad_dual_eq = Ax_minus_b[: lp.n_eq]
        grad_dual_ineq = Ax_minus_b[lp.n_eq :]

        # Unscale constraint violations to original space
        grad_dual_ineq_unscaled = grad_dual_ineq / jnp_row_scale_ineq
        grad_dual_eq_unscaled = grad_dual_eq / jnp_row_scale_eq

        # Reduced cost / primal / bounds, all UNSCALED to the original problem
        # space. grad_primal = c_scaled + A_scaledᵀy carries the col_scale factor
        # (c_scaled = c_true·col_scale), so r_true = grad_primal / col_scale;
        # x_true = x_scaled·col_scale and lb/ub_true = lb/ub_scaled·col_scale
        # (since lb_scaled = lb_true / col_scale). Reporting dual feasibility in
        # true units makes it consistent with the constraint (PFR) residuals,
        # which are already unscaled by row_scale; without this the reduced cost
        # is inflated by up to max(col_scale) on badly-scaled columns (≈5.8x /
        # 289x spread observed on boeing), making DFR/PGN artificially harsh.
        reduced_cost_true = grad_primal / col_scale
        primal_true = average_state.primal * col_scale
        lower_bounds_true = lp.lower_bounds * col_scale
        upper_bounds_true = lp.upper_bounds * col_scale

        projected_primal = projection_box(
            primal_true - reduced_cost_true,
            lower_bounds_true,
            upper_bounds_true,
        )
        projected_gradient_residual = primal_true - projected_primal
        primal_grad_norm = jnp.max(jnp.abs(projected_gradient_residual))

        # initial=0.0 gives the reduction an identity so problems with no
        # inequality (or no equality) constraints — i.e. a zero-size
        # violations array — report 0 violation instead of raising.
        ineq_violations = jnp.maximum(grad_dual_ineq_unscaled, 0.0)
        max_ineq_violation = jnp.max(ineq_violations, initial=0.0)

        eq_violations = jnp.abs(grad_dual_eq_unscaled)
        max_eq_violation = jnp.max(eq_violations, initial=0.0)

        complementarity_slack = jnp.max(
            jnp.abs(average_state.dual_ineq * grad_dual_ineq_unscaled),
            initial=0.0,
        ) / (1.0 + jnp.abs(objective_value))

        constraint_bound = jnp.maximum(max_ineq_violation, max_eq_violation)

        # Scaled-space reduced cost / bounds — used ONLY by the duality-gap
        # decomposition below, which is built in scaled space and rescaled by
        # c_max (changing these to true units would mismatch the still-scaled
        # `average_state.primal` it multiplies; the gap is intentionally left
        # untouched here). Dual FEASIBILITY uses the true-space versions instead.
        reduced_cost = grad_primal
        lower_bounds = lp.lower_bounds
        upper_bounds = lp.upper_bounds
        finite_lower = jnp.isfinite(lower_bounds)
        finite_upper = jnp.isfinite(upper_bounds)

        has_both_bounds = finite_lower & finite_upper
        has_only_lower = finite_lower & (~finite_upper)
        has_only_upper = (~finite_lower) & finite_upper
        has_no_bounds = (~finite_lower) & (~finite_upper)

        lower_term = reduced_cost * lower_bounds
        upper_term = reduced_cost * upper_bounds

        # box_infimum = inf over the box of rᵢ·xᵢ — the dual contribution of each
        # variable's bound. This is only FINITE when the reduced cost has a
        # dual-feasible sign for the variable's bound class; otherwise the infimum
        # is genuinely −∞ (the dual bound is unbounded below there) and the gap is
        # undefined. The old code took the finite bound·r product unconditionally,
        # which FABRICATED a finite dual bound at dual-infeasible points and
        # produced a misleading (often negative) duality gap. Guarding with −∞ on
        # the wrong sign makes duality_gap = +∞ → dual_gap_is_finite = False, so
        # the convergence test (which already requires a finite gap) correctly
        # refuses to certify until the dual is feasible. The wrong-sign reduced
        # cost is still surfaced in DFR. (Sign logic is scale-invariant.)
        #   has_only_lower ([l, +∞)): dual-feasible iff r ≥ 0; else −∞.
        #   has_only_upper ((−∞, u]): dual-feasible iff r ≤ 0; else −∞.
        #   has_no_bounds  (free):    dual-feasible iff r == 0; else −∞.
        # The sign guard must be CONSISTENT with how dual feasibility is TESTED in
        # `converged`: a RELATIVE residual, wrong-sign reduced cost ÷ (1+‖c‖)
        # against `dual_feasibility_tolerance`. So a wrong-sign reduced cost is
        # tolerated up to `tol·(1+‖c‖)` in absolute true units. Without matching
        # this, a point that is dual-feasible-to-tolerance (DFR_rel passes) is
        # spuriously reported as gap-infinite and never certifies a PDLP-optimal
        # point (boeing: DFR_rel 5.4e-4 < 1e-3 but the gap was +∞). Beyond that
        # band the term is −∞ (genuine dual infeasibility → gap undefined).
        # The band uses `c_norm_robust` (median-based, NOT the ∞-norm) so big-M /
        # penalty cost coefficients can't inflate it and fabricate a finite gap at
        # a dual-infeasible point — see the c_norm_robust definition in `solve` for
        # the binkar10_1 failure this guards against. On well-scaled problems
        # c_norm_robust == c_norm, so boeing is unchanged. Both are defined below
        # in `solve` and resolved at call time via closure.
        _dg_tol = jnp.asarray(
            dual_feasibility_threshold * (1.0 + c_norm_robust),
            reduced_cost_true.dtype,
        )
        neg_inf = jnp.asarray(-jnp.inf, reduced_cost.dtype)
        lower_only_term = jnp.where(reduced_cost_true >= -_dg_tol, lower_term, neg_inf)
        upper_only_term = jnp.where(reduced_cost_true <= _dg_tol, upper_term, neg_inf)
        free_term = jnp.where(jnp.abs(reduced_cost_true) <= _dg_tol, 0.0, neg_inf)

        box_infimum = jnp.where(
            has_both_bounds,
            jnp.minimum(lower_term, upper_term),
            jnp.where(
                has_only_lower,
                lower_only_term,
                jnp.where(has_only_upper, upper_only_term, free_term),
            ),
        )

        # Dual feasibility in TRUE units (reduced_cost_true / bounds_true), so it
        # is consistent with the row-scale-unscaled PFR. For box-constrained
        # variables the violation is the projected-gradient magnitude
        # |x - proj(x - r, lb, ub)|; for one-sided or free variables the classical
        # reduced-cost sign rules apply. (Bound-class masks are scale-invariant —
        # col_scale > 0 preserves finiteness — so the masks above are reused.)
        proj_box = projection_box(
            primal_true - reduced_cost_true, lower_bounds_true, upper_bounds_true
        )
        dual_feasibility_violation = jnp.where(
            has_both_bounds,
            jnp.abs(primal_true - proj_box),
            jnp.where(
                has_only_lower,
                jnp.maximum(-reduced_cost_true, 0.0),
                jnp.where(
                    has_only_upper,
                    jnp.maximum(reduced_cost_true, 0.0),
                    jnp.abs(reduced_cost_true),  # has_no_bounds (free variable)
                ),
            ),
        )
        dual_feasibility_residual = jnp.max(dual_feasibility_violation)

        # Duality gap, computed directly as its three-way decomposition. At a
        # dual-feasible point these terms sum to the gap in true (unscaled-
        # objective) units:
        #   gap = [rᵀx − Σ box_infimum]        bound / reduced-cost complementarity
        #       + yᵢₙₑ_qᵀ(bᵢₙₑ_q − Aᵢₙₑ_q x)     inequality complementarity (slack·dual)
        #       + y_eqᵀ(b_eq − A_eq x)         equality primal-residual coupling
        # Each is computed in scaled space then rescaled by `c_max`, so summing
        # them gives a unit-consistent gap (this is why the gap is built from the
        # decomposition rather than `objective − dual_bound`, which mixed scaled
        # and true units). Watching which term dominates localises why a large gap
        # persists even when per-constraint complementarity looks tiny.
        gap_bound_comp = (
            reduced_cost @ average_state.primal - jnp.sum(box_infimum)
        ) * c_max
        gap_ineq_comp = -(average_state.dual_ineq @ grad_dual_ineq) * c_max
        gap_eq_comp = -(average_state.dual_eq @ grad_dual_eq) * c_max

        duality_gap = gap_bound_comp + gap_ineq_comp + gap_eq_comp

        return (
            objective_value,
            primal_grad_norm,
            complementarity_slack,
            constraint_bound,
            dual_feasibility_residual,
            duality_gap,
            jnp.isfinite(duality_gap),
            gap_bound_comp,
            gap_ineq_comp,
            gap_eq_comp,
        )

    def relative_gap(duality_gap, objective_value):
        # PDLP/cuPDLP relative duality gap: |primal_obj − dual_obj| normalised by
        # (1 + |primal_obj| + |dual_obj|). The numerator IS `duality_gap` (built
        # as the unit-consistent complementarity decomposition in
        # compute_epoch_metrics — equal to primal_obj − dual_obj at a dual-feasible
        # point), and the dual objective is recovered exactly and unit-consistently
        # as `dual_obj = primal_obj − duality_gap` rather than recomputing bᵀy +
        # Σbox_infimum (which would risk the scaled/true unit mismatch the
        # decomposition exists to avoid). Adding |dual_obj| to the denominator
        # matches PDLP's convention so RDG magnitudes are directly comparable to
        # PDLP/HiGHS-PDLP certificates; it only enlarges the denominator, so it
        # slightly loosens the effective `dual_gap_tolerance` versus the old
        # (1+|obj|)-only normalisation.
        # At dual-infeasible points the gap is +inf (box_infimum sign-guarded to
        # −inf), making dual_bound = −inf and the raw ratio inf/inf = nan. Report
        # +inf there instead — the gap is undefined/unbounded, not nan — so the
        # log and any |gap|-based comparison read sensibly. Convergence still gates
        # on the separate `dual_gap_is_finite` flag regardless.
        dual_bound = objective_value - duality_gap
        return jnp.where(
            jnp.isfinite(duality_gap),
            jnp.abs(duality_gap)
            / (1.0 + jnp.abs(objective_value) + jnp.abs(dual_bound)),
            jnp.inf,
        )

    def converged(
        constraint_bound,
        dual_feasibility_residual,
        duality_gap,
        dual_gap_is_finite,
        objective_value,
        gap_bound_comp,
        gap_ineq_comp,
        gap_eq_comp,
    ):
        # Standard LP optimality certificate — all three conditions must hold:
        #   * primal feasibility: constraint_bound within tolerance
        #   * dual feasibility: reduced-cost residual within tolerance
        #   * a finite, small duality gap
        # All three are tested RELATIVELY: the gap by the PDLP normalisation
        # (1+|primal_obj|+|dual_obj|) via `relative_gap`, and primal/dual
        # feasibility by (1+‖b‖)/(1+‖c‖) — PDLP/HiGHS's relative-residual stopping
        # convention, and the same normalisation the restart `kkt_merit` uses. The
        # absolute residuals are misleadingly large on problems with big ‖b‖/‖c‖
        # (boeing: ‖b‖≈2952, ‖c‖≈43), so an absolute test never certified points
        # that were PDLP-optimal (e.g. PFR_abs 2.5e-3 but PFR_rel 8.6e-7; DFR_abs
        # 2.4e-2 but DFR_rel 5.4e-4 — both well inside 1e-3 relative). `b_norm`/
        # `c_norm` are defined below in `solve` and resolved at call time via the
        # closure.
        relative_duality_gap = relative_gap(duality_gap, objective_value)
        relative_primal_residual = constraint_bound / (1.0 + b_norm)
        relative_dual_residual = dual_feasibility_residual / (1.0 + c_norm)

        # No-cancellation guard. `duality_gap` is the SIGNED sum of three
        # complementarity terms (bound, inequality, equality coupling). At a true
        # KKT point each term is ~0, so the sum is ~0 honestly. But on degenerate
        # problems with large dual multipliers the equality-coupling term
        # `y_eqᵀ(b−A_eq x)` can be large and negative — (large y)·(small residual)
        # — and CANCEL a large positive bound-complementarity term, making the
        # signed sum tiny while the point is nowhere near complementary. qap10
        # (QAP assignment relaxation) certified ~1.4–2.4% suboptimal this way:
        # gap_bound≈+5.2, gap_eq≈−5.2, sum≈0.04 → RDG 6.5e-5, yet the recovered
        # dual_obj sat BELOW the true optimum (weak-duality violation). Requiring
        # the sum of the |component|s (not their signed sum) to be small closes
        # this: cancellation can shrink the signed sum but not the absolute sum.
        # Normalised by the same PDLP denominator so the band matches the RDG test.
        dual_bound = objective_value - duality_gap
        gap_denom = 1.0 + jnp.abs(objective_value) + jnp.abs(dual_bound)
        relative_gap_abs = (
            jnp.abs(gap_bound_comp) + jnp.abs(gap_ineq_comp) + jnp.abs(gap_eq_comp)
        ) / gap_denom
        return (
            (relative_primal_residual <= primal_feasibility_tolerance)
            & (relative_dual_residual <= dual_feasibility_threshold)
            & dual_gap_is_finite
            & (relative_duality_gap <= dual_gap_tolerance)
            & (relative_gap_abs <= dual_gap_tolerance)
        )

    def converged_primal(constraint_bound, obj_window):
        # Dual-free, opt-in stopping rule for "I just want the primal solved".
        # Optimality is certified heuristically (no dual): primal feasibility
        # (`constraint_bound`, already the correct per-block ineq/eq measure) AND
        # the objective having stalled — relative change across the last
        # `primal_stop_window` epochs below `primal_stop_obj_tol`. `obj_window`
        # holds the most recent objective values (oldest first). The duality gap
        # and dual residuals are still computed but do NOT gate here.
        #
        # Primal feasibility is tested RELATIVELY — constraint_bound / (1+‖b‖) —
        # to match `converged()` and the printed PFR. The raw absolute
        # `constraint_bound` is misleadingly large on big-‖b‖ problems (e.g. big-M
        # relaxations): an absolute `constraint_bound <= tol` test never fires even
        # when the relative PFR is deep inside tolerance, so primal_stop ran to
        # max_epochs while reporting a feasible-looking PFR.
        relative_primal_residual = constraint_bound / (1.0 + b_norm)
        feasible = relative_primal_residual <= primal_feasibility_tolerance
        oldest = obj_window[0]
        newest = obj_window[-1]
        rel_change = jnp.abs(newest - oldest) / (1.0 + jnp.abs(newest))
        window_full = jnp.all(jnp.isfinite(obj_window))
        stalled = window_full & (rel_change < primal_stop_obj_tol)
        return feasible & stalled

    def check_max_epochs(count):
        return count >= max_epochs

    # PDLP-style primal-weight initialisation. When k-scaling is on and k_init is
    # left as None we derive it from the objective/RHS norms ||c|| / ||b|| (in
    # the scaled space the solver iterates in), which puts the primal/dual step
    # ratio in the right order of magnitude before iteration 1 instead of
    # starting symmetric.
    if k_scaling and k_init is None:
        norm_c = float(jnp.linalg.norm(lp.c)) + 1e-30
        norm_b = float(jnp.linalg.norm(lp.b)) + 1e-30
        k_init = float(np.clip(norm_c / norm_b, k_lo, k_hi))
    elif k_init is None:
        k_init = 1.0

    i = 1
    state = initial_solution
    average_state = initial_solution
    if initial_opt_state is not None:
        opt_state = initial_opt_state
    elif k_scaling:
        # Pack the primal weight k with the optax state. In adaptive_step mode the
        # k-slot is a (k, eta) pair carrying the per-iteration step size too.
        _pik_dtype = initial_solution.primal.dtype
        if halpern:
            # Halpern carries the anchor z_0 (cycle-start iterate) in the k-slot,
            # seeded from the initial solution and reset at each restart below.
            _k_slot = (
                jnp.asarray(k_init, _pik_dtype),
                jnp.asarray(adaptive_eta, _pik_dtype),
                jax.tree.map(lambda x: x + 0, initial_solution),
            )
        elif adaptive_step:
            _k_slot = (
                jnp.asarray(k_init, _pik_dtype),
                jnp.asarray(adaptive_eta, _pik_dtype),
            )
        else:
            _k_slot = jnp.asarray(k_init, _pik_dtype)
        opt_state = (optimiser.init(initial_solution), _k_slot)
    else:
        opt_state = optimiser.init(initial_solution)
    primal_grad_norm = jnp.inf
    complementarity_slack = jnp.inf
    constraint_bound = jnp.inf
    dual_feasibility_residual = jnp.inf
    objective_value = jnp.inf
    duality_gap = jnp.inf
    dual_gap_is_finite = False
    # Gap components (bound / inequality / equality complementarity). Seeded to inf
    # so the no-cancellation guard in converged() can't fire before the first epoch
    # has computed real metrics (inf gives relative_gap_abs = nan ≤ tol → False).
    gap_bound_comp = jnp.inf
    gap_ineq_comp = jnp.inf
    gap_eq_comp = jnp.inf
    count = 0
    # Wall time of the very first epoch (which pays the one-off XLA compile).
    # Captured so callers can form a "corrected" steady-state runtime that
    # amortises out the compile: n_epochs * (solve - first) / (n_epochs - 1).
    first_epoch_seconds = None
    total_weight = 0.0
    reported_used_avg = average
    # Whether the per-iteration primal-feasibility gate latched during the most
    # recent epoch. Diagnostic (surfaced in verbose logging); does not affect
    # stopping — the certificate below decides that.
    epoch_gate_tripped = False
    is_converged = True
    # Why the loop terminated. Defaults to the budget-exhausted case; is_done()
    # overwrites it with the test that actually fired ("certificate" or
    # "primal_stall"), and the max_epochs break path leaves it as "max_epochs".
    stop_reason = "max_epochs"
    current_iterations_per_epoch = iterations_per_epoch

    # Iterate at the last restart (or the start), used to rebalance the primal
    # weight k from the primal-vs-dual movement over the restart cycle. Must be an
    # independent copy: `state` aliases initial_solution's buffer and is donated
    # to (freed by) __sps each epoch, so sharing it would read as deleted at the
    # first restart's k-rebalance.
    state_at_last_restart = jax.tree.map(lambda x: x + 0, initial_solution)

    # Per-epoch primal-weight rebalance (k_update_per_epoch): reference iterate at
    # the end of the previous epoch, used to drive k from the primal-vs-dual
    # movement over the just-finished epoch. Independent copy for the same
    # buffer-donation reason as state_at_last_restart. Only the three contractive
    # schemes that can carry a meaningful k qualify.
    k_per_epoch = (
        k_scaling
        and k_update_per_epoch
        and update_mode in ("pdhg", "extragradient", "halpern")
    )
    state_at_last_epoch = jax.tree.map(lambda x: x + 0, initial_solution)

    # Rolling window of recent objective values for the opt-in primal_stop rule
    # (oldest first). Seeded with inf so the window is not "full" until enough
    # real epochs have elapsed.
    obj_window = jnp.full((max(int(primal_stop_window), 1),), jnp.inf)

    # Adaptive restart bookkeeping. `restart_i_offset` is subtracted from the
    # global iteration counter `i` before it is handed to the optimiser /
    # weight_function, so a restart re-zeros the LR schedule without disturbing
    # the running epoch/iteration accounting.
    restarts_done = 0
    restart_i_offset = 0
    epochs_since_restart = 0
    # The cycle-exhaustion cap is tracked in ITERATIONS, not epochs, so that
    # `cycle_exhausted` fires at the same point in the optimisation trajectory
    # regardless of `iterations_per_epoch`. A restart is a destructive reset
    # (optimiser.init(state) wipes PDHG momentum/averaging), so its cadence must
    # not depend on how the same iteration budget happens to be chopped into
    # epochs. Seed the cap in iterations from the epoch-count knob so the
    # default (epochs_per_restart=10) means the same thing it always has at the
    # default iterations_per_epoch; iterations_since_restart accumulates the
    # ACTUAL per-epoch iteration count, so it stays correct even under
    # iterations_per_epoch_decay.
    iterations_since_restart = 0
    current_cycle_cap_iters = float(epochs_per_restart) * float(iterations_per_epoch)
    merit_at_last_restart = jnp.inf
    # Previous epoch's RELATIVE (tolerance-comparable) primal/dual residuals.
    # Used only to gate the cycle-cap restart while the merit is non-finite
    # (dual-infeasible): see the `still_improving` guard below. inf until the
    # first epoch sets it, so the guard can't fire before there is a real
    # comparison point.
    prev_relative_pfr = jnp.inf
    prev_relative_dfr = jnp.inf
    # Previous epoch's restart merit, for cuPDLP condition (ii) (stalling: merit
    # rising again after necessary decay). inf until the first epoch sets it.
    prev_epoch_merit = jnp.inf

    # Feasibility-polish bookkeeping. Attempt counters and epoch budgets are per
    # side; the relative-residual windows drive the stall trigger; seeding
    # `last_polish_epoch` at 0 makes `polish_cooldown` double as the earliest
    # epoch a first attempt can fire.
    polish_attempts = {"primal": 0, "dual": 0}
    polish_budget = {
        "primal": int(polish_max_epochs),
        "dual": int(polish_max_epochs),
    }
    last_polish_epoch = 0
    polish_adopted = 0
    pfr_window = []
    dfr_window = []

    # Normalisation constants for the restart merit (PDLP-style). Each KKT
    # residual is divided by 1 + its natural scale so the three terms are
    # comparable across very differently-scaled problems and the merit reflects
    # *relative* progress, not absolute units. These MUST be in TRUE (original)
    # units, because the residuals they normalise are reported in true units
    # (constraint_bound is row-scale-unscaled; the reduced cost is /col_scale):
    # b_true = b_scaled / row_scale, c_true = c_scaled / col_scale. Using the
    # scaled-space norms here was a unit mismatch — on boeing it made the relative
    # convergence test and the gap sign-guard band wrong (true ‖c‖≈43 vs scaled
    # ‖c‖≈0.8), so the gap read +∞ at a point whose true relative gap was ~2e-3.
    _row_scale_all = jnp.concatenate([jnp_row_scale_eq, jnp_row_scale_ineq])
    b_norm = float(jnp.max(jnp.abs(lp.b / _row_scale_all))) if lp.b.size else 0.0
    c_norm = float(jnp.max(jnp.abs(lp.c / col_scale))) if lp.c.size else 0.0

    # Per-iteration gate PFR tolerance. Defaults to the primal-feasibility
    # tolerance — the same threshold the certificate's PFR test uses. The full
    # `_gate` config tuple is assembled below, once `c_norm_robust` (needed for
    # the gate's gap sign-guard, matching the host certificate) is available.
    _gate_tol = gate_tol if gate_tol is not None else primal_feasibility_tolerance

    # Robust cost norm for the gap's dual-feasibility SIGN-GUARD band only. The
    # band `_dg_tol = tol·(1+norm)` decides whether a wrong-sign reduced cost is
    # tolerated (term finite) or genuinely dual-infeasible (term −∞ → gap +∞).
    # Keying it to the ∞-norm `c_norm` breaks on big-M / penalty problems where a
    # handful of huge cost coefficients dominate: binkar10_1 has 90 entries of 1e6
    # against a median |c|≈25, so c_norm=1e6 inflated the band to ~1000 and a point
    # with reduced costs as negative as −61 was treated as dual-feasible, making
    # `duality_gap` finite and tiny (RDG 5e-4) at a point a TRUE 14% above optimum
    # (gap ≈ 1031). The certificate then falsely fired. A median-based norm tracks
    # the problem's NATURAL cost scale (unpolluted by big-M), so the band stays
    # tight enough to reject big-M wrong-sign mass while still matching c_norm on
    # well-scaled problems (boeing: median≈∞-norm≈43, band unchanged → its genuine
    # near-feasible dual, needing ~0.024 slack, still certifies). The *reported*
    # DFR residual test below keeps the PDLP ∞-norm convention; only the gap
    # finiteness guard uses this robust norm. `min(c_norm, …)` ensures we never
    # LOOSEN relative to the ∞-norm — only tighten when big-M pollution is present.
    if lp.c.size:
        _c_true = jnp.abs(lp.c / col_scale)
        _c_nonzero = _c_true[_c_true > 0]
        if _c_nonzero.size:
            _c_median = float(jnp.median(_c_nonzero))
            # 1e3·median is a generous multiple of the natural scale — wide enough
            # to never clip a genuine reduced cost, narrow enough that big-M
            # coefficients (≫1e3× the median) cannot inflate the band.
            c_norm_robust = min(c_norm, max(_c_median * 1e3, 1.0))
        else:
            c_norm_robust = c_norm
    else:
        c_norm_robust = 0.0

    # Per-iteration convergence gate config passed to __sps. The gate is a cheap
    # TRIGGER checked every iteration inside the scan; the host still runs the
    # full 3-part certificate (converged()) on the average iterate, so the gate
    # can only cause an earlier host check, never a false stop. It evaluates the
    # SAME three conditions the host certificate does (relative PFR, DFR, and the
    # sign-guarded duality-gap RDG) on the CURRENT iterate, reusing the A@x and
    # reduced cost the step already computes — no extra matvec. Requiring the gap
    # too (not just PFR+DFR) is essential: a primal+dual-feasible-but-gap-open
    # iterate must NOT freeze, or restart-driven modes get trapped (ns1830653:
    # PFR+DFR-only froze the tail every epoch and never closed the gap). The
    # tuple carries every scaled-space constant the three tests need. Wired for
    # synchronous and pdhg (both expose A@x and the reduced cost); halpern and
    # extragradient thread the flag inertly (see the step bodies).
    _gate = (
        (
            _row_scale_all,
            b_norm,
            _gate_tol,
            col_scale,
            c_norm,
            dual_feasibility_threshold,
            c_max,
            c_norm_robust,
            float(dual_gap_tolerance),
            _report_c,
        )
        if per_iter_gate
        else None
    )

    def kkt_merit(
        constraint_bound,
        dual_feasibility_residual,
        duality_gap,
        dual_gap_is_finite,
        objective_value,
    ):
        # Single scalar quality measure driving the restart trigger, normalised
        # like PDLP's relative KKT residual: primal feasibility by (1 + ‖b‖),
        # dual feasibility by (1 + ‖c‖), and the duality gap by (1 + |obj|). The
        # gap is only counted when finite (dual feasible); otherwise it is inf so
        # the feasibility terms dominate. Normalising makes the restart trigger
        # see *rotational stalling* (relative gap not shrinking) rather than raw
        # residual magnitude, which is the regime where epoch-level restarts pay
        # off most.
        gap_term = jnp.where(
            dual_gap_is_finite,
            relative_gap(duality_gap, objective_value),
            jnp.inf,
        )
        primal_term = constraint_bound / (1.0 + b_norm)
        dual_term = dual_feasibility_residual / (1.0 + c_norm)
        return jnp.maximum(jnp.maximum(primal_term, dual_term), gap_term)

    def _read_k_eta(opt_state):
        # Surface the live primal weight k and adaptive step eta from the
        # opt_state k-slot for the epoch trace. Layout: (inner, k) for plain
        # k-scaling, (inner, (k, eta[, anchor])) once adaptive_step/halpern adds
        # the step (and anchor) to the slot. Returns (k, eta) with eta None when
        # there is no adaptive step to report.
        if not k_scaling:
            return None, None
        slot = opt_state[1]
        if isinstance(slot, tuple):
            return float(slot[0]), float(slot[1])
        return float(slot), None

    def print_epoch_metrics(epoch_time=None, k_val=None, eta_val=None):
        time_str = f"|Time {epoch_time:.2f}s|" if epoch_time is not None else ""
        ke_str = ""
        if k_val is not None:
            ke_str += f"|k {k_val:.2e}|"
        if eta_val is not None:
            ke_str += f"|eta {eta_val:.2e}|"
        # PFR/DFR are printed RELATIVE — normalised by (1+‖b‖)/(1+‖c‖), the same
        # way `converged()` tests them — so the logged values match the stopping
        # criterion (an absolute DFR of 2e-2 can be a relative 5e-4 that passes).
        # RDG is already relative (÷(1+|obj|)).
        relative_pfr = constraint_bound / (1.0 + b_norm)
        relative_dfr = dual_feasibility_residual / (1.0 + c_norm)
        # OBJERR: true relative objective error vs a supplied reference optimum.
        # The duality gap (RDG) is a complementarity sum gated by the dual, so it
        # lags the primal; OBJERR isolates genuine primal suboptimality, and the
        # OBJERR≪RDG gap is the dual-lag artifact. Diagnostic only.
        objerr_str = ""
        if reference_objective is not None:
            obj_err = abs(objective_value - reference_objective) / (
                1.0 + abs(reference_objective)
            )
            objerr_str = f"|OBJERR {obj_err:.2e}|"
        gate_str = "|gate|" if epoch_gate_tripped else ""
        print(
            f"|Epoch {count}|"
            f"|Obj{objective_value:.2e}|"
            f"|PFR {relative_pfr:.2e}|"
            f"|DFR {relative_dfr:.2e}|"
            f"|RDG {relative_gap(duality_gap, objective_value):.2e}|"
            f"{objerr_str}"
            f"{gate_str}"
            f"{ke_str}"
            f"{time_str}"
        )
        print("----------------------------------------------")

    def is_done():
        nonlocal stop_reason
        certificate_met = bool(
            converged(
                constraint_bound,
                dual_feasibility_residual,
                duality_gap,
                dual_gap_is_finite,
                objective_value,
                gap_bound_comp,
                gap_ineq_comp,
                gap_eq_comp,
            )
        )
        # Check the certificate first: when both the full certificate and the
        # primal-stall heuristic would fire on the same epoch, attribute the stop
        # to the certificate — it is the stronger (truly optimal) reason.
        if certificate_met:
            stop_reason = "certificate"
            return True
        if primal_stop and bool(converged_primal(constraint_bound, obj_window)):
            stop_reason = "primal_stall"
            return True
        return False

    # PDLP-style primal-weight rebalance from the primal-vs-dual *movement*
    # between two iterates (distance, not per-step gradient norms). Shared by the
    # restart rebalance and the per-epoch rebalance: log-space geometric-mean
    # blend of the movement-ratio target with the current weight (k_theta), then
    # clamp to [k_lo, k_hi]. Squared norms avoid two sqrts; the ratio is preserved.
    def _rebalance_k(new_state, ref_state, k_prev):
        dp = new_state.primal - ref_state.primal
        dd = jnp.concatenate(
            [
                new_state.dual_eq - ref_state.dual_eq,
                new_state.dual_ineq - ref_state.dual_ineq,
            ]
        )
        move_p2 = jnp.vdot(dp, dp) + 1e-60
        move_d2 = jnp.vdot(dd, dd) + 1e-60
        k_target = jnp.sqrt(move_p2 / move_d2)
        log_k = k_theta * jnp.log(k_target) + (1.0 - k_theta) * jnp.log(k_prev)
        return jnp.clip(jnp.exp(log_k), k_lo, k_hi)

    def _attempt_polish(side):
        # One feasibility-polish attempt (see the docstring): a nested solve() on
        # the side's feasibility problem, built from the ORIGINAL (unscaled)
        # data and warm-started from the currently-reported point. Returns the
        # recombined candidate — polished side + untouched other side — mapped
        # into the MAIN solve's scaled space; the caller certifies/adopts it.
        seed = average_state if (average and reported_used_avg) else state
        base = _polish_base_lp
        if side == "primal":
            # Pure feasibility: min 0 subject to the original constraints. The
            # optimal dual is 0, so the dual is warm-started there.
            sub_lp = LP(
                np.zeros_like(base.c),
                base.A_eq,
                base.b_eq,
                base.A_ineq,
                base.b_ineq,
                base.lower_bounds,
                base.upper_bounds,
            )
            init = SaddleState(
                primal=seed.primal * col_scale,
                dual_ineq=jnp.zeros_like(seed.dual_ineq),
                dual_eq=jnp.zeros_like(seed.dual_eq),
            )
        else:
            # Homogenised dual-feasibility problem: b = 0 and every finite bound
            # moved to 0. Bound CLASSES are preserved, so the dual sign
            # conditions match the original problem; the optimal primal is 0
            # and the primal is warm-started there.
            sub_lp = LP(
                base.c,
                base.A_eq,
                np.zeros_like(base.b_eq),
                base.A_ineq,
                np.zeros_like(base.b_ineq),
                np.where(np.isfinite(base.lower_bounds), 0.0, -np.inf),
                np.where(np.isfinite(base.upper_bounds), 0.0, np.inf),
            )
            init = SaddleState(
                primal=jnp.zeros_like(seed.primal),
                dual_ineq=seed.dual_ineq * jnp_row_scale_ineq * c_max,
                dual_eq=seed.dual_eq * jnp_row_scale_eq * c_max,
            )
        # Warm-started from the trigger point — valid because the trigger
        # additionally requires the lagging residual to already be within
        # polish_residual_slack of tolerance (the FINISHING regime). Do NOT
        # loosen that gate and lean on this warm start for deep plateaus: from
        # mzzv11's PFR~1e-1 plateau the warm-started feasibility problem did
        # not converge in 60 epochs (the trap rides in with the iterate) while
        # a cold start certified in 2 — but the cold point is objective-
        # agnostic, so its recombined gap explodes and the candidate is
        # useless. Escaping deep plateaus needs a different tool.
        sub = solve(
            sub_lp,
            optimiser=optimiser,
            max_epochs=polish_budget[side],
            initial_solution=init,
            iterations_per_epoch=iterations_per_epoch,
            primal_feasibility_tolerance=primal_feasibility_tolerance,
            dual_feasibility_tolerance=dual_feasibility_tolerance,
            dual_gap_tolerance=dual_gap_tolerance,
            weight_function=weight_function,
            verbose=verbose,
            log_every=log_every,
            average=average,
            report_best=report_best,
            update_mode=update_mode,
            k_scale=k_scale,
            k_theta=k_theta,
            # k_init is left to the ||c||/||b|| derivation ON PURPOSE: it
            # degenerates in exactly the right direction on the polish problems
            # (c=0 → k_lo: no objective force on the primal; b=0 → k_hi,
            # symmetrically). Seeding the main solve's learned k instead left
            # mzzv11's 2-epoch feasibility problem unconverged at 20 epochs.
            adaptive_eta=_orig_adaptive_eta,
            scale=scale,
            # c=0 on the primal side would make the objective scale max|c| = 0;
            # both sides run with the unscaled cost.
            scaled_objective=False,
            restarts=restarts,
            epochs_per_restart=epochs_per_restart,
            restart_multiplier=restart_multiplier,
            restart_decay=restart_decay,
            necessary_decay=necessary_decay,
            per_iter_gate=per_iter_gate,
            feasibility_polish=False,
        )
        if sub["stop_reason"] == "interrupted":
            # The nested solve caught the Ctrl-C; re-raise so the main solve's
            # interrupt path (return the current best point) still runs.
            raise KeyboardInterrupt
        sol = sub["solution"]  # original units
        if side == "primal":
            cand = SaddleState(
                primal=sol.primal / col_scale,
                dual_ineq=seed.dual_ineq + 0,
                dual_eq=seed.dual_eq + 0,
            )
        else:
            cand = SaddleState(
                primal=seed.primal + 0,
                dual_ineq=sol.dual_ineq / (jnp_row_scale_ineq * c_max),
                dual_eq=sol.dual_eq / (jnp_row_scale_eq * c_max),
            )
        return cand, bool(sub["converged"])

    start_time = time.time()

    try:
        while not is_done():
            if max_epochs:
                if check_max_epochs(count):
                    is_converged = False
                    print(f"Reached maximum epochs: {max_epochs}. Stopping.")
                    print("----------------------------------------------")
                    break

            start_epoch_time = time.time()
            (
                shifted_i,
                state,
                average_state,
                opt_state,
                total_weight,
                epoch_converged_flag,
            ) = __sps(
                current_iterations_per_epoch,
                i - restart_i_offset,
                lp,
                optimiser,
                state,
                average_state,
                opt_state,
                weight_function,
                total_weight,
                primal_damping,
                dual_damping_ineq,
                dual_damping_eq,
                average,
                update_mode,
                k_scaling=k_scaling,
                k_init=k_init,
                adaptive_eta=adaptive_eta,
                gate=_gate,
            )
            # __sps increments the (restart-shifted) counter; restore global i.
            # `shifted_i` comes back as a JAX array (it is the scan-carried loop
            # index). Coerce to a Python int so the `start_iter` argument fed to
            # __sps next epoch (i - restart_i_offset) stays a Python int. Otherwise
            # it flips int -> ArrayImpl after epoch 1, and since start_iter is a
            # traced (non-static) argument that type change retraces run_epoch —
            # a second ~0.6s XLA compile billed to epoch 2.
            i = int(shifted_i) + restart_i_offset
            # Whether the per-iteration primal-feasibility gate latched this
            # epoch (state frozen for the tail). Informational only — the host
            # still runs the full certificate below and that decides stopping.
            epoch_gate_tripped = bool(epoch_converged_flag)
            # Same flip for total_weight: it returns as a float64 JAX array but
            # enters epoch 1 as a weak-typed Python float. Coerce back to a Python
            # float so its type is stable across epochs — otherwise the weak->strong
            # change retraces run_epoch a second time, billed to epoch 2.
            total_weight = float(total_weight)

            # JAX dispatch is async: __sps returns futures that may still be in
            # flight. Force them here so the epoch timer captures the real
            # compute cost rather than billing the tail to the final
            # block_until_ready (and thus to total runtime, not any epoch).
            jax.block_until_ready(state)

            metrics = compute_epoch_metrics(average_state if average else state)
            # `reported_used_avg` tracks which point the bound metrics describe;
            # the actual state objects are re-resolved from this flag at use
            # sites (output, restart) so they stay current after eq-projection.
            reported_used_avg = average

            # report_best: when averaging is on, the average and the last iterate
            # are different points and either can be the better solution
            # (averaging stabilises rotational problems but lags the last iterate
            # when it is already contracting). Compute the iterate's metrics too
            # and report/converge on whichever has the lower KKT merit. This costs
            # a second matvec pair per epoch, so it is opt-in.
            if report_best and average:
                state_metrics = compute_epoch_metrics(state)
                avg_merit = kkt_merit(
                    metrics[3], metrics[4], metrics[5], metrics[6], metrics[0]
                )
                st_merit = kkt_merit(
                    state_metrics[3],
                    state_metrics[4],
                    state_metrics[5],
                    state_metrics[6],
                    state_metrics[0],
                )
                if bool(st_merit < avg_merit):
                    metrics = state_metrics
                    reported_used_avg = False

            (
                objective_value,
                primal_grad_norm,
                complementarity_slack,
                constraint_bound,
                dual_feasibility_residual,
                duality_gap,
                dual_gap_is_finite,
                gap_bound_comp,
                gap_ineq_comp,
                gap_eq_comp,
            ) = metrics

            count += 1

            # Equality-constraint projection: if the unscaled equality residual
            # exceeds the threshold, project the primal (and average) onto the
            # equality manifold. Done after metrics so the logged residual reflects
            # the pre-projection state; the projected iterate is the warm-start for
            # the next epoch.
            if _eq_project is not None:
                eq_residual = float(
                    np.max(
                        np.abs(_A_eq_sp @ np.asarray(state.primal) - _b_eq_np)
                        / np.asarray(jnp_row_scale_eq)
                    )
                )
                if eq_residual > eq_projection_threshold:
                    projected_primal = _eq_project(state.primal)
                    state = SaddleState(
                        primal=projected_primal,
                        dual_eq=state.dual_eq,
                        dual_ineq=state.dual_ineq,
                    )
                    if average:
                        projected_avg_primal = _eq_project(average_state.primal)
                        average_state = SaddleState(
                            primal=projected_avg_primal,
                            dual_eq=average_state.dual_eq,
                            dual_ineq=average_state.dual_ineq,
                        )
                    if verbose:
                        print(
                            f"  → Equality projection (eq_residual={eq_residual:.2e})"
                        )

            # Roll the latest objective into the primal_stop window (oldest first).
            obj_window = jnp.concatenate(
                [obj_window[1:], jnp.reshape(objective_value, (1,))]
            )

            finish_epoch_time = time.time()

            if count == 1:
                first_epoch_seconds = finish_epoch_time - start_epoch_time

            if verbose and (count == 1 or count % log_every == 0):
                _k_trace, _eta_trace = _read_k_eta(opt_state)
                print_epoch_metrics(
                    finish_epoch_time - start_epoch_time,
                    k_val=_k_trace,
                    eta_val=_eta_trace,
                )

            # --- Adaptive restart decision ---
            restarted_this_epoch = False
            if restarts and restarts_done < restarts:
                epochs_since_restart += 1
                iterations_since_restart += current_iterations_per_epoch

                # `merit` is the metric of the *reported* point: with
                # report_best it is already the better of {average, iterate};
                # otherwise it is the average (averaging on) or the last iterate.
                # This drives the restart *trigger*.
                merit = kkt_merit(
                    constraint_bound,
                    dual_feasibility_residual,
                    duality_gap,
                    dual_gap_is_finite,
                    objective_value,
                )

                # --- Two-point restart candidate (PDLP-style) ---
                # When averaging is on, the average and the current iterate are
                # genuinely different points and either can be the better warm
                # start, so restart to whichever has the lower merit instead of
                # always discarding a frequently-better average.
                cycle_exhausted = iterations_since_restart >= current_cycle_cap_iters
                # Resolve the restart point from the *current* state/average
                # variables via the report_best decision flag, not an object
                # captured before metrics. The equality-projection block above
                # may have rebound state and average_state to fresh (projected)
                # iterates; a stale object would warm-start off the equality
                # manifold.
                restart_used_avg = reported_used_avg if average else False
                restart_point = average_state if restart_used_avg else state
                restart_merit = merit
                near_threshold = bool(merit <= restart_decay * merit_at_last_restart)
                if average and report_best:
                    # report_best already evaluated both points this epoch and
                    # `reported_used_avg`/`merit` describe the better of the two —
                    # reuse them directly, no extra matvec.
                    pass
                elif average and (cycle_exhausted or near_threshold):
                    # report_best is off, so the iterate's metrics were not
                    # computed yet. The second `compute_epoch_metrics(state)` is a
                    # full matvec pair; gate it to epochs where a restart can
                    # actually fire (cycle exhausted, or the average is already
                    # near the sufficient-progress threshold so the better
                    # state-point could tip it over). On other epochs neither
                    # point triggers, so the extra metrics would be wasted.
                    (
                        st_obj,
                        _st_pgn,
                        _st_cs,
                        st_cb,
                        st_dfr,
                        st_dg,
                        st_dgf,
                        *_rest,
                    ) = compute_epoch_metrics(state)
                    state_merit = kkt_merit(st_cb, st_dfr, st_dg, st_dgf, st_obj)
                    if bool(state_merit < merit):
                        restart_point = state
                        restart_merit = state_merit
                        restart_used_avg = False

                # A non-finite merit carries no progress signal — it just means
                # the iterate is dual-infeasible so the duality gap (hence the KKT
                # merit) is +∞ (see the box-infimum sign guard). Without this gate
                # the progress tests below degenerate to `inf <= restart_decay*inf`
                # → True, firing a restart EVERY epoch and exhausting the restart
                # budget in the first few epochs (e.g. neos-3754480-nidda: 10/10
                # restarts in 10 epochs, all on merit=inf), leaving none for the
                # feasibility tail where they actually help. Only the length-based
                # `cycle_exhausted` path may fire on an inf merit.
                merit_is_finite = bool(jnp.isfinite(restart_merit))

                # `cycle_exhausted` is the one trigger allowed to fire on a
                # non-finite merit (see above) — it's the safety valve that keeps
                # a permanently dual-infeasible run restarting at all. But while
                # still dual-infeasible, BOTH raw feasibility residuals monotone-
                # decreasing epoch-over-epoch means the run is mid-flight on a
                # perfectly good trajectory, not stuck — restarting there only
                # destroys momentum for no gain (momentum1: an
                # iterations_per_epoch=1000 run hit merit=inf cycle-exhaustion at
                # 10,000 iterations while PFR/DFR were still improving every
                # epoch; the optimiser.init() reset wiped the trajectory and it
                # never recovered, while iterations_per_epoch=10000 reached full
                # convergence before the same cap fired). Only suppress the
                # exhaustion path this way — sufficient_progress/stalling_restart
                # already require merit_is_finite and are unaffected.
                relative_pfr = float(constraint_bound) / (1.0 + b_norm)
                relative_dfr = float(dual_feasibility_residual) / (1.0 + c_norm)
                still_improving = (not merit_is_finite) and (
                    relative_pfr < prev_relative_pfr
                    and relative_dfr < prev_relative_dfr
                )
                if cycle_exhausted and not merit_is_finite and still_improving:
                    cycle_exhausted = False
                prev_relative_pfr = relative_pfr
                prev_relative_dfr = relative_dfr

                # Seed the baseline on the first finite merit so the
                # sufficient-progress test has something real to compare against
                # (avoids a spurious restart against the initial inf baseline). Only
                # seed from a FINITE merit, so a dual-infeasible early run leaves the
                # baseline inf until a real value appears rather than locking it to
                # inf.
                if not jnp.isfinite(merit_at_last_restart) and merit_is_finite:
                    merit_at_last_restart = restart_merit

                sufficient_progress = merit_is_finite and bool(
                    restart_merit <= restart_decay * merit_at_last_restart
                )

                # cuPDLP.jl condition (ii) — "necessary decay + stalling": restart
                # once the merit has decayed to <= necessary_decay (0.8) of its
                # cycle-start value AND has started rising again vs the previous
                # epoch (the rotational-stall turnaround). Catches the oscillating
                # feasibility tail that the absolute sufficient-decay test (0.2x)
                # misses. `restart_decay` keeps driving the (i) sufficient trigger;
                # `necessary_decay` is the looser (ii) threshold.
                stalling_restart = (
                    merit_is_finite
                    and bool(jnp.isfinite(prev_epoch_merit))
                    and bool(restart_merit <= necessary_decay * merit_at_last_restart)
                    and bool(restart_merit > prev_epoch_merit)
                )

                if sufficient_progress or stalling_restart or cycle_exhausted:
                    restarted_this_epoch = True
                    # Warm-start restart from the better of {average, iterate};
                    # reset momentum, averaging, weight accumulation and the LR /
                    # weight_function schedule (via the iteration offset).
                    state = restart_point
                    if k_scaling:
                        # PDLP-style primal-weight rebalance: drive k from the
                        # primal-vs-dual *movement* over the just-finished cycle
                        # (distance between iterates), not per-step gradient
                        # norms. log-space geometric-mean blend with the current
                        # weight (k_theta), then clamp. Reset momentum.
                        # The k-slot is (k, eta[, anchor]) in adaptive_step mode,
                        # plain k otherwise. Read k accordingly.
                        k_prev = opt_state[1][0] if adaptive_step else opt_state[1]
                        k_new = _rebalance_k(state, state_at_last_restart, k_prev)
                        if halpern:
                            # Restarted Halpern: reset eta AND re-anchor z_0 to the
                            # cycle-start iterate `state`. The lambda counter resets
                            # via restart_i_offset below. Independent anchor copy —
                            # `state` is donated to __sps next epoch.
                            _eta_dtype = state.primal.dtype
                            k_slot = (
                                k_new,
                                jnp.asarray(adaptive_eta, _eta_dtype),
                                jax.tree.map(lambda x: x + 0, state),
                            )
                        elif adaptive_step:
                            # Carry the learned step size across the restart.
                            # The adaptive rule grows eta ~100-250x above its
                            # 1/||A|| seed over a cycle; re-seeding here forced
                            # the rule to re-climb from scratch after every
                            # restart (a stretch of tiny, conservative steps).
                            # PDLP convention resets averaging/momentum at a
                            # restart but keeps the step size, so carry the live
                            # eta from the current opt_state instead of reseeding.
                            k_slot = (k_new, opt_state[1][1])
                        else:
                            k_slot = k_new
                        opt_state = (optimiser.init(state), k_slot)
                    else:
                        opt_state = optimiser.init(state)
                    average_state = state
                    # __sps donates `state` each epoch (freeing the buffer in
                    # place), so state_at_last_restart must be an independent
                    # copy — otherwise it aliases the donated buffer and reads as
                    # deleted at the next restart's k-rebalance (line ~1534).
                    state_at_last_restart = jax.tree.map(lambda x: x + 0, state)
                    # Reset the averaging accumulator alongside `average_state`.
                    # The running mean is avg += (w/total_weight)·(new − avg); if
                    # total_weight keeps the prior cycles' accumulated weight while
                    # average_state is re-seeded to the cycle-start point, the
                    # post-restart ratio w/total_weight is far too small and the new
                    # cycle's average stays frozen near the restart point. This made
                    # 5×1000 (restarts fire on epoch boundaries) diverge from 1×5000
                    # (fewer restarts) — same total iters, different result. Resetting
                    # restores epoch-granularity invariance of the averaging.
                    total_weight = 0.0
                    restart_i_offset = i - 1
                    merit_at_last_restart = restart_merit
                    # New cycle: no previous-epoch merit yet for condition (ii).
                    prev_epoch_merit = jnp.inf
                    epochs_since_restart = 0
                    iterations_since_restart = 0
                    current_cycle_cap_iters *= restart_multiplier
                    current_iterations_per_epoch = max(
                        iterations_per_epoch_min,
                        int(current_iterations_per_epoch * iterations_per_epoch_decay),
                    )
                    restarts_done += 1
                    if verbose:
                        if sufficient_progress:
                            reason = "sufficient-progress"
                        elif stalling_restart:
                            reason = "stalling"
                        else:
                            reason = "cycle-cap"
                        which = "avg" if restart_used_avg else "iterate"
                        if k_scaling:
                            _k_show = opt_state[1][0] if adaptive_step else opt_state[1]
                            k_msg = f", k={float(_k_show):.3e}"
                        else:
                            k_msg = ""
                        print(
                            f"Restart {restarts_done}/{restarts} at epoch {count} "
                            f"({reason}, merit={float(restart_merit):.2e} "
                            f"[{which}], next cap={current_cycle_cap_iters:.0f} iters, "
                            f"iters/epoch={current_iterations_per_epoch}{k_msg})"
                        )
                        print("----------------------------------------------")
                else:
                    # No restart this epoch: record the merit so the next epoch's
                    # condition (ii) can detect a turnaround (merit rising again).
                    prev_epoch_merit = restart_merit

            # Per-epoch primal-weight rebalance. A restart already rebalanced k
            # (and reset momentum/anchor/eta), so only adjust k on epochs where no
            # restart fired. Unlike the restart path this leaves the optimiser
            # state, averaging, halpern anchor and adaptive eta untouched — it only
            # rewrites the k component of the k-slot, tracking primal/dual progress
            # within the current restart cycle.
            if k_per_epoch and not restarted_this_epoch:
                k_prev = opt_state[1][0] if adaptive_step else opt_state[1]
                k_new = _rebalance_k(state, state_at_last_epoch, k_prev)
                if adaptive_step or halpern:
                    # k-slot is (k, eta) for adaptive pdhg/extragradient and
                    # (k, eta, anchor) for halpern; rewrite only the k leaf.
                    opt_state = (opt_state[0], (k_new, *opt_state[1][1:]))
                else:
                    opt_state = (opt_state[0], k_new)
            if k_per_epoch:
                # Reference for next epoch's movement. Independent copy: `state` is
                # donated to (freed by) __sps next epoch.
                state_at_last_epoch = jax.tree.map(lambda x: x + 0, state)

            # Optional per-epoch Halpern re-anchor: start a fresh Halpern cycle at
            # every epoch boundary instead of only at restarts. Mirrors the
            # restart re-anchor (anchor z_0 := current iterate, lambda counter
            # reset via restart_i_offset so lambda re-warms toward 1/2), but
            # leaves eta and k to their usual per-epoch handling. Skipped when a
            # real restart already fired this epoch — the restart's re-anchor
            # (above) takes precedence, so we don't re-anchor twice.
            if halpern and halpern_reanchor_per_epoch and not restarted_this_epoch:
                # k-slot is (k, eta, anchor); rewrite only the anchor leaf with an
                # independent copy (state is donated to __sps next epoch).
                opt_state = (
                    opt_state[0],
                    (
                        opt_state[1][0],
                        opt_state[1][1],
                        jax.tree.map(lambda x: x + 0, state),
                    ),
                )
                # Reset the cycle-local index so lambda_k = 1/(k_local+1) re-warms
                # toward 1/2 next epoch; without this lambda would stay ~0 and the
                # fresh anchor would carry no weight (a silent no-op).
                restart_i_offset = i - 1

            # --- Feasibility polishing (PDLP-style) ---
            # When one side of the certificate is the lone blocker, solve that
            # side's far easier feasibility problem and recombine (see the
            # docstring). Checked after the restart decision so a restarting
            # epoch is left alone.
            if feasibility_polish and not restarted_this_epoch:
                rel_pfr = float(constraint_bound) / (1.0 + b_norm)
                rel_dfr = float(dual_feasibility_residual) / (1.0 + c_norm)
                rdg = float(relative_gap(duality_gap, objective_value))
                pfr_window.append(rel_pfr)
                dfr_window.append(rel_dfr)
                if len(pfr_window) > int(polish_stall_window):
                    pfr_window.pop(0)
                    dfr_window.pop(0)
                gap_ok = bool(dual_gap_is_finite) and rdg <= polish_gap_slack * float(
                    dual_gap_tolerance
                )
                window_full = len(pfr_window) == int(polish_stall_window)
                pfr_stalled = window_full and rel_pfr > polish_stall_ratio * (
                    pfr_window[0]
                )
                dfr_stalled = window_full and rel_dfr > polish_stall_ratio * (
                    dfr_window[0]
                )

                # All conditions are NECESSARY. Near-tolerance (finishing
                # regime): the sub-solve is warm-started from the current
                # point, and from a deep plateau the warm start drags the trap
                # into the sub-problem (see _attempt_polish). Stalled: a
                # residual still improving will get there cheaper by letting
                # the main loop run (measured: an eager gap-only trigger COST
                # epochs on neos-1593097). gap_ok: a stalled residual with a
                # wide-open gap is not a finishing case.
                side = None
                if (
                    primal_feasibility_tolerance
                    < rel_pfr
                    <= polish_residual_slack * primal_feasibility_tolerance
                    and rel_dfr <= dual_feasibility_threshold
                    and gap_ok
                    and pfr_stalled
                ):
                    side = "primal"
                elif (
                    dual_feasibility_threshold
                    < rel_dfr
                    <= polish_residual_slack * dual_feasibility_threshold
                    and rel_pfr <= primal_feasibility_tolerance
                    and gap_ok
                    and dfr_stalled
                ):
                    side = "dual"

                if (
                    side is not None
                    and polish_attempts[side] < polish_max_attempts
                    and count - last_polish_epoch >= polish_cooldown
                ):
                    if verbose:
                        print(
                            f"→ Feasibility polish ({side}), budget "
                            f"{polish_budget[side]} epochs (PFR {rel_pfr:.2e}, "
                            f"DFR {rel_dfr:.2e}, RDG {rdg:.2e})"
                        )
                        print("----------------------------------------------")
                    polish_attempts[side] += 1
                    last_polish_epoch = count
                    cand, sub_converged = _attempt_polish(side)
                    if not sub_converged:
                        # The sub-solve ran out of budget before certifying its
                        # feasibility problem; give the next attempt more room
                        # regardless of whether the partial result is adopted.
                        polish_budget[side] *= 2

                    # Adoption compares TOLERANCE-NORMALISED certificate
                    # distance (>1 = blocking), NOT the raw KKT merit: the raw
                    # max() is often pinned by a residual that is already
                    # within its (perhaps loose) tolerance, so a candidate that
                    # fixes the actual blocker would compare equal and be
                    # spuriously discarded.
                    def _cert_distance(obj, cb, dfr, dg, dgf, gb, gi, ge):
                        rp = (cb / (1.0 + b_norm)) / primal_feasibility_tolerance
                        rd = (dfr / (1.0 + c_norm)) / max(
                            dual_feasibility_threshold, 1e-30
                        )
                        if bool(dgf):
                            rg = float(relative_gap(dg, obj)) / dual_gap_tolerance
                            gap_denom = 1.0 + abs(float(obj)) + abs(float(obj - dg))
                            rga = (
                                (abs(float(gb)) + abs(float(gi)) + abs(float(ge)))
                                / gap_denom
                                / dual_gap_tolerance
                            )
                        else:
                            rg = rga = np.inf
                        return max(float(rp), float(rd), rg, rga)

                    cand_metrics = compute_epoch_metrics(cand)
                    cand_merit = _cert_distance(
                        cand_metrics[0],
                        cand_metrics[3],
                        cand_metrics[4],
                        cand_metrics[5],
                        cand_metrics[6],
                        cand_metrics[7],
                        cand_metrics[8],
                        cand_metrics[9],
                    )
                    cur_merit = _cert_distance(
                        objective_value,
                        constraint_bound,
                        dual_feasibility_residual,
                        duality_gap,
                        dual_gap_is_finite,
                        gap_bound_comp,
                        gap_ineq_comp,
                        gap_eq_comp,
                    )
                    cand_certified = bool(
                        converged(
                            cand_metrics[3],
                            cand_metrics[4],
                            cand_metrics[5],
                            cand_metrics[6],
                            cand_metrics[0],
                            cand_metrics[7],
                            cand_metrics[8],
                            cand_metrics[9],
                        )
                    )
                    if cand_certified or cand_merit < cur_merit:
                        # Adopt. Certified → the loop exits at the next is_done()
                        # (the certificate reads the metric variables rebound
                        # here). Merely merit-improving → warm-start the main
                        # loop from the combination with a restart-style reset:
                        # momentum/averaging cleared, k and eta carried, LR
                        # schedule re-zeroed. Not counted against `restarts`.
                        polish_adopted += 1
                        (
                            objective_value,
                            primal_grad_norm,
                            complementarity_slack,
                            constraint_bound,
                            dual_feasibility_residual,
                            duality_gap,
                            dual_gap_is_finite,
                            gap_bound_comp,
                            gap_ineq_comp,
                            gap_eq_comp,
                        ) = cand_metrics
                        state = cand
                        average_state = cand
                        reported_used_avg = average
                        if k_scaling:
                            k_slot = opt_state[1]
                            if halpern:
                                # Re-anchor Halpern at the adopted point
                                # (independent copy; `state` is donated next
                                # epoch).
                                k_slot = (
                                    k_slot[0],
                                    k_slot[1],
                                    jax.tree.map(lambda x: x + 0, cand),
                                )
                            opt_state = (optimiser.init(cand), k_slot)
                        else:
                            opt_state = optimiser.init(cand)
                        total_weight = 0.0
                        restart_i_offset = i - 1
                        state_at_last_restart = jax.tree.map(lambda x: x + 0, cand)
                        state_at_last_epoch = jax.tree.map(lambda x: x + 0, cand)
                        # The restart controller's baseline is in raw KKT-merit
                        # units, not the tolerance-normalised cert distance used
                        # for the adoption decision above.
                        merit_at_last_restart = kkt_merit(
                            cand_metrics[3],
                            cand_metrics[4],
                            cand_metrics[5],
                            cand_metrics[6],
                            cand_metrics[0],
                        )
                        prev_epoch_merit = jnp.inf
                        epochs_since_restart = 0
                        iterations_since_restart = 0
                        pfr_window.clear()
                        dfr_window.clear()
                        if verbose:
                            outcome = (
                                "certified" if cand_certified else "merit improved"
                            )
                            print(
                                f"→ Polish adopted ({outcome}: "
                                f"{float(cand_merit):.2e} vs {float(cur_merit):.2e})"
                            )
                            print("----------------------------------------------")
                    else:
                        # Discarded. Budget already doubled above if the
                        # sub-solve ran out of room; a converged-but-still-worse
                        # candidate means the recombination is the problem, and
                        # more sub-epochs would not change it.
                        if verbose:
                            print(
                                f"→ Polish discarded (merit {float(cand_merit):.2e}"
                                f" vs {float(cur_merit):.2e}; sub-solve "
                                f"{'converged' if sub_converged else 'unconverged'})"
                            )
                            print("----------------------------------------------")

        # The while-loop exits the iteration *after* the converging epoch, so its
        # metrics were computed but only printed if it landed on a log_every
        # boundary. Print the final converged epoch's criteria here (skip when we
        # broke out via max_epochs, which prints its own message and leaves
        # is_converged False).
        if verbose and is_converged and count > 0:
            print("Convergence criteria met.")
            if report_best and average:
                print(
                    f"Reported point: {'average' if reported_used_avg else 'iterate'}"
                )
            print("----------------------------------------------")
            print_epoch_metrics()

        if report_best and average:
            output = average_state if reported_used_avg else state
        elif average:
            output = average_state
        else:
            output = state
    except KeyboardInterrupt:
        is_converged = False
        stop_reason = "interrupted"
        if report_best and average:
            output = average_state if reported_used_avg else state
        elif average:
            output = average_state
        else:
            output = state
        print("KeyboardInterrupt received. Returning current solution.")
        print("----------------------------------------------")

    output = jax.block_until_ready(output)

    end_time = time.time()

    lp.c = lp.c * c_max
    print(f"Time to solution: {end_time - start_time:.2f} seconds")
    print("----------------------------------------------")
    print(f"Epochs to solution: {count}")
    print("----------------------------------------------")
    # Report against the TRUE cost (lp.c may carry the vertex-bias perturbation).
    print(f"Objective: {float((c_true * c_max) @ output.primal):.5e}")
    print("----------------------------------------------")

    if scale in ["ruiz", "pc", "ruiz+pc"]:
        output = SaddleState(
            primal=output.primal * col_scale,
            dual_ineq=output.dual_ineq * jnp_row_scale_ineq,
            dual_eq=output.dual_eq * jnp_row_scale_eq,
        )

    if scaled_objective:
        output = SaddleState(
            primal=output.primal,
            dual_ineq=output.dual_ineq * c_max,
            dual_eq=output.dual_eq * c_max,
        )

    # The internal solve time (epoch loop, incl. first-epoch XLA compile but
    # NOT the scaling / spectral-norm / sparse-setup phase before start_time).
    # This is the fair like-for-like figure against a baseline solver's own
    # run() timer, which also excludes problem setup.
    solve_seconds = end_time - start_time

    # "Corrected" (steady-state) runtime: amortise the one-off first-epoch XLA
    # compile out of the total. Extrapolates what the solve would have cost had
    # every epoch run at the warm per-epoch rate:
    #     n * (solve - first_epoch) / (n - 1)
    # Falls back to the raw solve time when it can't be formed (fewer than two
    # epochs, or the first-epoch time was never captured).
    if first_epoch_seconds is not None and count > 1:
        corrected_seconds = count * (solve_seconds - first_epoch_seconds) / (count - 1)
    else:
        corrected_seconds = solve_seconds

    return {
        "solution": output,
        "converged": is_converged,
        "opt_state": opt_state,
        "stop_reason": stop_reason,
        "solve_seconds": solve_seconds,
        "corrected_seconds": corrected_seconds,
        "epochs": count,
        "polish": {
            "primal_attempts": polish_attempts["primal"],
            "dual_attempts": polish_attempts["dual"],
            "adopted": polish_adopted,
        },
    }


# %%
def to_jaddle_sparse(lp: LP):
    # Resolve to the active precision profile's float width: float64 (x64,
    # PDLP-style double precision), float32, or float16. jaddle_dtype() is the
    # single source of truth; float64 requires x64 to be enabled (otherwise JAX
    # silently truncates and spams warnings), so guard against that mismatch.
    float_dtype = jo.jaddle_dtype()
    if float_dtype == jnp.float64 and not jax.config.jax_enable_x64:
        float_dtype = jnp.float32
    # scipy.sparse cannot hold float16, so build the matrices in the nearest
    # scipy-supported width and only cast the on-device BCOO data to the profile
    # dtype afterwards. Half precision lives in JAX, not in the scipy CSR.
    np_float = np.float64 if float_dtype == jnp.float64 else np.float32

    A_eq_sp = lp.A_eq.astype(np_float)
    A_eq_sp = A_eq_sp.sorted_indices()
    A_eq_sp.sum_duplicates()
    A_eq_sp = A_eq_sp.tocoo()

    A_ineq_sp = lp.A_ineq.astype(np_float)
    A_ineq_sp = A_ineq_sp.sorted_indices()
    A_ineq_sp.sum_duplicates()
    A_ineq_sp = A_ineq_sp.tocoo()

    A_eq = jsp.BCOO.from_scipy_sparse(A_eq_sp).sort_indices()
    A_ineq = jsp.BCOO.from_scipy_sparse(A_ineq_sp).sort_indices()
    # Cast the sparse data array (indices stay integer) to the profile dtype.
    A_eq = jsp.BCOO((A_eq.data.astype(float_dtype), A_eq.indices), shape=A_eq.shape)
    A_ineq = jsp.BCOO(
        (A_ineq.data.astype(float_dtype), A_ineq.indices), shape=A_ineq.shape
    )

    lp_jax = JaddleLP(
        jnp.array(lp.c, dtype=float_dtype),
        A_eq,
        jnp.array(lp.b_eq, dtype=float_dtype),
        A_ineq,
        jnp.array(lp.b_ineq, dtype=float_dtype),
        jnp.array(lp.lower_bounds, dtype=float_dtype),
        jnp.array(lp.upper_bounds, dtype=float_dtype),
    )
    return lp_jax


def lp_summary_statistics(lp: LP):
    num_vars = lp.num_variables()
    num_eq = lp.num_eq_constraints()
    num_ineq = lp.num_ineq_constraints()

    if lp.A_eq.data.size > 0:
        min_A_eq = np.minimum(np.min(lp.A_eq.data), 0.0)
        max_A_eq = np.maximum(np.max(lp.A_eq.data), 0.0)
        min_b_eq = np.min(lp.b_eq)
        max_b_eq = np.max(lp.b_eq)
        num_nnz_A_eq = lp.A_eq.data.size
    else:
        min_A_eq = None
        max_A_eq = None
        min_b_eq = None
        max_b_eq = None
        num_nnz_A_eq = None

    if lp.A_ineq.data.size > 0:
        min_A_ineq = np.minimum(np.min(lp.A_ineq.data), 0.0)
        max_A_ineq = np.maximum(np.max(lp.A_ineq.data), 0.0)
        min_b_ineq = np.min(lp.b_ineq)
        max_b_ineq = np.max(lp.b_ineq)
        num_nnz_A_ineq = lp.A_ineq.data.size
    else:
        min_A_ineq = None
        max_A_ineq = None
        min_b_ineq = None
        max_b_ineq = None
        num_nnz_A_ineq = None

    min_c = np.min(lp.c)
    max_c = np.max(lp.c)

    print("--------------------------------")
    print("LP Summary Statistics")
    print("--------------------------------")
    print(f"Number of variables: {num_vars}")
    print(f"Number of equality constraints: {num_eq}")
    print(f"Number of inequality constraints: {num_ineq}")
    print(f"Number of nonzeros in A_eq: {num_nnz_A_eq}")
    print(f"Number of nonzeros in A_ineq: {num_nnz_A_ineq}")
    print(f"[Min, Max] of c: [{min_c}, {max_c}]")
    print(f"[Min, Max] of A_eq: [{min_A_eq}, {max_A_eq}]")
    print(f"[Min, Max] of b_eq: [{min_b_eq}, {max_b_eq}]")
    print(f"[Min, Max] of A_ineq: [{min_A_ineq}, {max_A_ineq}]")
    print(f"[Min, Max] of b_ineq: [{min_b_ineq}, {max_b_ineq}]")
    print("----------------------------------------------")


# %%


def __convert_to_scipy(jsp_mat: jsp.BCOO) -> sp.csc_matrix:
    data = np.array(jsp_mat.data)
    indices = np.array(jsp_mat.indices)
    row, col = indices[:, 0], indices[:, 1]

    return sp.csc_matrix((data, (row, col)), shape=jsp_mat.shape)


def __build_scaling_coo(lp: LP, augmented: bool):
    """Build the (absolute) COO operand the equilibration loops iterate over.

    Returns ``(absdata, row_idx, col_idx, n_rows, n_cols, A, b, m, n)`` where the
    first five describe ``|M|`` (the augmented ``[[A,b],[c^T/c_norm,0]]`` when
    ``augmented`` else ``A`` alone) as flat COO arrays suitable for JAX
    ``segment_*`` reductions, and ``A``/``b``/``m``/``n`` are the unscaled
    constraint operator and RHS used to apply the final row/col scales.

    The matrix is assembled once in scipy (cheap: ~14 ms even on stp3d); only the
    iterative equilibration -- the part that dominates -- runs in JAX.
    """
    A = sp.vstack([lp.A_eq, lp.A_ineq]).tocsr()
    m, n = A.shape
    b = np.concatenate([lp.b_eq, lp.b_ineq])
    c = lp.c

    if augmented:
        c_norm = np.max(np.abs(c)) or 1.0
        b_col = sp.csc_matrix(b.reshape(-1, 1))
        c_row = sp.csc_matrix((c / c_norm).reshape(1, -1))
        zero = sp.csc_matrix((1, 1))
        M = sp.bmat([[A, b_col], [c_row, zero]]).tocoo()
    else:
        M = A.tocoo()

    absdata = jnp.asarray(np.abs(M.data))
    row_idx = jnp.asarray(M.row.astype(np.int32))
    col_idx = jnp.asarray(M.col.astype(np.int32))
    return absdata, row_idx, col_idx, M.shape[0], M.shape[1], A, b, m, n


@functools.partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
def __equilibrate_jax(
    absdata,
    row_idx,
    col_idx,
    n_rows,
    n_cols,
    max_iter,
    use_max,
    clip_bounds,
    threshold,
):
    """Run ``max_iter`` Sinkhorn-Ruiz equilibration sweeps in JAX.

    ``use_max=True`` gives the L-infinity (Ruiz) norm via ``segment_max``;
    ``use_max=False`` gives the L1 (Pock-Chambolle) norm via ``segment_sum``.
    Row/col norms of ``D_r M D_c`` factor as ``row_scale * reduce(|M| * col_scale)``
    so the scaled matrix is never rematerialised -- each sweep is two gathers and
    two segmented reductions over the nnz, mirroring the original numpy loop. The
    empty-row/col guard is implicit: ``segment_*`` yields 0 for absent segments,
    which the ``<= threshold -> 1.0`` clamp maps to a unit (no-op) scale.
    """
    lo, hi = clip_bounds

    def reduce_segments(vals, seg, num):
        if use_max:
            return jax.ops.segment_max(vals, seg, num_segments=num)
        return jax.ops.segment_sum(vals, seg, num_segments=num)

    def body(_, carry):
        row_scale, col_scale = carry
        row_norms = reduce_segments(absdata * col_scale[col_idx], row_idx, n_rows)
        row_norms = row_norms * row_scale
        row_norms = jnp.where(row_norms <= threshold, 1.0, row_norms)
        row_scale = row_scale * jnp.clip(1.0 / jnp.sqrt(row_norms), lo, hi)

        col_norms = reduce_segments(absdata * row_scale[row_idx], col_idx, n_cols)
        col_norms = col_norms * col_scale
        col_norms = jnp.where(col_norms <= threshold, 1.0, col_norms)
        col_scale = col_scale * jnp.clip(1.0 / jnp.sqrt(col_norms), lo, hi)
        return row_scale, col_scale

    row_scale = jnp.ones(n_rows, dtype=absdata.dtype)
    col_scale = jnp.ones(n_cols, dtype=absdata.dtype)
    return jax.lax.fori_loop(0, max_iter, body, (row_scale, col_scale))


def __apply_scaling(lp: LP, A, b, dr, dc):
    """Apply row scale ``dr`` (length m) and col scale ``dc`` (length n) to the LP."""
    A_final = sp.diags(dr) @ A @ sp.diags(dc)
    b_scaled = dr * b

    n_eq = lp.A_eq.shape[0]
    lp_scaled = LP(
        lp.c * dc,
        A_final[:n_eq, :],
        b_scaled[:n_eq],
        A_final[n_eq:, :],
        b_scaled[n_eq:],
        lp.lower_bounds / dc,
        lp.upper_bounds / dc,
    )
    return lp_scaled, dr, dc


def ruiz_scaling(
    lp: LP, max_iter=30, threshold=1e-8, clip_bounds=(1e-6, 1e6), augmented=True
):
    """
    Applies Ruiz scaling to an LP in standard form with sparse matrices:
        min c^T x
        s.t. A_eq x = b_eq
             A_ineq x <= b_ineq
             lower_bounds <= x <= upper_bounds
    Returns scaled LP, row_scaling (length m), col_scaling (length n).

    ``augmented`` selects what is equilibrated:
      * ``False`` (default, PDLP-style): equilibrate ``A`` alone — the operator
        that defines the saddle dynamics — and let ``b``/``c`` ride the resulting
        row/col scales. The augmented variant lets the appended ``b`` column and
        ``c`` row absorb scaling, which under-equilibrates the constraint ROWS on
        badly-scaled problems (observed on `boeing`: cols hit [1,1] but rows
        retained a 130x spread, freezing complementarity / dual feasibility).
      * ``True``: equilibrate the augmented ``[[A, b], [c^T, 0]]`` so cost and RHS
        information also drive the equilibration (the previous default).
    """

    # Build |M| once in scipy, then equilibrate in JAX. The L-infinity row/col
    # norms factor as row_scale * max_over_row(|M| * col_scale), so the scaled
    # matrix is never rematerialised. The scale vectors are sized to M (augmented:
    # m+1/n+1; A-only: m/n); the dr/dc slices below take the first m/n entries —
    # the LP's true dimensions — dropping the augmented row/col when present.
    absdata, row_idx, col_idx, n_rows, n_cols, A, b, m, n = __build_scaling_coo(
        lp, augmented
    )

    row_scale, col_scale = __equilibrate_jax(
        absdata,
        row_idx,
        col_idx,
        n_rows,
        n_cols,
        max_iter,
        True,  # L-infinity (Ruiz) via segment_max
        clip_bounds,
        threshold,
    )

    dr = np.asarray(row_scale)[:m]
    dc = np.asarray(col_scale)[:n]
    return __apply_scaling(lp, A, b, dr, dc)


def pc_scaling(lp: LP, max_iter=10, threshold=1e-8, clip_bounds=(1e-6, 1e6)):
    """
    Applies PC scaling to an LP in standard form with sparse matrices:
        min c^T x
        s.t. A_eq x = b_eq
                A_ineq x <= b_ineq
                lower_bounds <= x <= upper_bounds
    Returns scaled LP, row_scaling (length m), col_scaling (length n).

    Scaling is derived from the augmented matrix [[A, b], [c^T, 0]] so that
    both cost and constraint information drive the equilibration.
    """

    # Build |M| once in scipy, then equilibrate in JAX. The L1 row/col norms of
    # D_r M D_c factor as row_scale * (|M| @ col_scale) and col_scale *
    # (|M|^T @ row_scale) — i.e. segment_sum over the nnz — so the scaled matrix
    # is never rematerialised. PC always uses the augmented [[A,b],[c,0]].
    absdata, row_idx, col_idx, n_rows, n_cols, A, b, m, n = __build_scaling_coo(
        lp, augmented=True
    )

    row_scale, col_scale = __equilibrate_jax(
        absdata,
        row_idx,
        col_idx,
        n_rows,
        n_cols,
        max_iter,
        True,
        clip_bounds,
        threshold,
    )

    dr = np.asarray(row_scale)[:m]
    dc = np.asarray(col_scale)[:n]
    return __apply_scaling(lp, A, b, dr, dc)


def project_onto_eq(lp: JaddleLP, primal: jnp.ndarray, tol: 1e-6) -> jnp.ndarray:
    """
    Projects a primal solution onto the equality constraints using JAX GMRES.

    Solves: min ||x - primal||_2 s.t. A_eq @ x = b_eq

    Args:
        lp: Linear program with equality constraints
        primal: Candidate primal solution to project
        tol: Tolerance for GMRES convergence

    Returns:
        Projected primal solution satisfying A_eq @ x = b_eq
    """

    # Solve normal equations: A_eq^T @ A_eq @ delta = A_eq^T @ (b_eq - A_eq @ primal)

    A_eq_T = lp.A_eq.transpose()

    residual = lp.b_eq - lp.A_eq @ primal

    def matvec(v):
        return A_eq_T @ (lp.A_eq @ v)

    delta, info = gmres(matvec, A_eq_T @ residual, tol=tol)

    if info != 0:
        print(f"GMRES did not converge (info={info})")

    return primal + delta


def primal_polish(
    lp: LP,
    warm: SaddleState,
    active_tol: float = 1e-6,
    bound_tol: float = 1e-12,
    atol: float = 1e-12,
    damp: float = 1e-6,
    max_passes: int = 20,
):
    """
    Polish a warm primal by a **bound active-set least squares** on the active
    constraints -- a fast, robust polish that respects the box ``[lb, ub]``
    EXACTLY (not by post-hoc clipping) and cannot blow up.

    Two nested active sets: an OUTER loop over the tight inequality rows (added
    as equalities, growing as the result violates dropped rows) wraps an INNER
    bound active-set solve (``bounded_lstsq``) that fixes variables at their
    bounds and frees them by reduced-gradient sign, converging to the exact
    bounded optimum of the active system. Each inner step is a single warm-started
    damped ``lsmr`` Krylov solve over the free columns only -- far faster than
    bound-constrained ``lsq_linear`` (trust-region) on large over-determined
    active sets (e.g. momentum1 ~11k x 5k), for the same exact-bounded answer.

    The active rows ``A_active``/``b_active`` are the equalities plus the tight
    inequalities (each tight ``<=`` row treated as an equality to hit). The inner
    solve minimizes ``||A_active x - b_active||^2 + damp^2 ||x - x0||^2`` over the
    free variables, warm-started from and anchored near the warm point.

    Trade-offs: no exact dual multipliers are produced (warm duals passed
    through). When bounds and constraints genuinely conflict, no point hits all
    active rows inside the box -- the bound active-set still returns the
    box-feasible least-squares point and the caller's keep-better gate discards it
    if worse. It cannot diverge unboundedly (every iterate is box-feasible by
    construction; ``damp`` anchors the step).

    ``active_tol`` defaults to ``None`` => derived from the warm point's
    feasibility (``max(1e-6, worst primal residual)``), so the active set tracks
    how converged the point is. It also sets the bound-activity tolerance. ``atol``
    is the ``lsmr`` tolerance; ``damp`` is the Tikhonov damping toward ``x0``;
    ``max_passes`` caps each active-set loop.

    Returns a new ``SaddleState`` (polished primal, warm duals). Does not mutate
    ``warm``.
    """
    from scipy.sparse.linalg import lsmr

    # primal_polish runs host-side on scipy (boolean row-slicing, lsmr). Accept a
    # JaddleLP (the JAX-native solver representation) by materialising its scipy
    # view; a scipy LP is passed through unchanged.
    if isinstance(lp, JaddleLP):
        lp = lp.to_scipy()

    x = np.asarray(warm.primal, dtype=np.float64)
    lb = np.asarray(lp.lower_bounds, dtype=np.float64)
    ub = np.asarray(lp.upper_bounds, dtype=np.float64)
    A_eq = lp.A_eq.tocsc().astype(np.float64)
    A_ineq = lp.A_ineq.tocsc().astype(np.float64)
    b_eq = np.asarray(lp.b_eq, dtype=np.float64)
    b_ineq = np.asarray(lp.b_ineq, dtype=np.float64)

    if active_tol is None:
        # Guard empty constraint blocks: eq_slack/ineq_slack do jnp.max over an
        # empty array (-> error) when that constraint class is absent.
        eq_s = float(np.abs(A_eq @ x - b_eq).max()) if A_eq.shape[0] > 0 else 0.0
        ineq_s = (
            float(np.maximum(A_ineq @ x - b_ineq, 0.0).max())
            if A_ineq.shape[0] > 0
            else 0.0
        )
        active_tol = max(1e-6, 10 * max(eq_s, ineq_s))

    # bound_tol: how close to a bound counts as "at" it. Tie to active_tol.
    if bound_tol is None:
        bound_tol = active_tol

    def bounded_lstsq(A_act, b_act, x0):
        """Exact bounded least-squares of ||A_act x - b_act||^2 s.t. lb <= x <= ub
        by a BOUND active-set loop over lsmr. x0 must be box-feasible.

        Variables pinned at a bound are FIXED (dropped from the unknowns, folded
        into the RHS); the inner lsmr solve runs over the FREE columns only. After
        each solve we (a) FIX any free var the step pushed past a bound, at that
        bound, and (b) FREE any fixed var whose reduced gradient
        g = A_act^T (A_act x - b_act) points back into the box. The loop ends when
        the active set stops changing -- the KKT point of the bounded problem.
        """
        x_cur = np.clip(x0, lb, ub)
        at_lb = x_cur <= lb + bound_tol
        at_ub = x_cur >= ub - bound_tol
        fixed = at_lb | at_ub
        # Pin fixed vars exactly onto the bound they sit on.
        x_cur = np.where(at_ub, ub, np.where(at_lb, lb, x_cur))

        for _ in range(max_passes):
            free = ~fixed
            if not free.any():
                break
            A_free = A_act[:, free]
            rhs = b_act - A_act[:, fixed] @ x_cur[fixed]  # fold fixed cols in
            sol = lsmr(A_free, rhs, atol=atol, btol=atol, damp=damp, x0=x_cur[free])[0]

            x_trial = x_cur.copy()
            x_trial[free] = sol

            # (a) FIX free vars the step pushed outside the box, at that bound.
            newly_fixed = free & (
                (x_trial < lb - bound_tol) | (x_trial > ub + bound_tol)
            )
            x_cur = np.clip(x_trial, lb, ub)
            if newly_fixed.any():
                fixed = fixed | newly_fixed
                continue

            # (b) FREE fixed vars whose reduced gradient points into the box.
            # At a lower bound, descent needs g < 0 (increase x_i); at an upper
            # bound, g > 0 (decrease x_i).
            g = A_act.T @ (A_act @ x_cur - b_act)
            release = fixed & (
                ((x_cur <= lb + bound_tol) & (g < -atol))
                | ((x_cur >= ub - bound_tol) & (g > atol))
            )
            if not release.any():
                break  # KKT satisfied for the bounded problem
            fixed = fixed & ~release
        return x_cur

    # Outer loop: the constraint active set.
    if A_ineq.shape[0] > 0:
        ineq_active = np.abs(A_ineq @ x - b_ineq) <= active_tol
    else:
        ineq_active = np.zeros(0, dtype=bool)

    x_new = np.clip(x, lb, ub)
    for _pass in range(max_passes):
        if A_ineq.shape[0] > 0:
            A_act = sp.vstack([A_eq, A_ineq[ineq_active]], format="csc")
            b_act = np.concatenate([b_eq, b_ineq[ineq_active]])
        else:
            A_act, b_act = A_eq, b_eq

        x_new = bounded_lstsq(A_act, b_act, x_new)

        if A_ineq.shape[0] == 0:
            break
        violated = (A_ineq @ x_new - b_ineq) > active_tol
        newly = violated & ~ineq_active
        if not newly.any():
            break
        ineq_active = ineq_active | newly  # add violated rows, re-solve

    return SaddleState(
        primal=jnp.asarray(x_new),
        dual_ineq=warm.dual_ineq,
        dual_eq=warm.dual_eq,
    )


# %%
