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
):
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

    def grad(state):
        # Fused matvecs: 2 sparse ops (A @ x, Aᵀ @ y) instead of 4. The
        # controllers below act on post-optimiser update norms, not on A·dx, so
        # there is nothing to gain from returning Ax — keep the single matvec
        # pair and return only the gradient.
        dual = jnp.concatenate([state.dual_eq, state.dual_ineq])
        Ax = lp.A @ state.primal  # shape: (n_eq + n_ineq,)
        ATd = lp.A_T @ dual  # shape: (n_vars,)
        grad_primal = lp.c + ATd + primal_damping * state.primal
        residual = lp.b - Ax
        grad_dual_eq = residual[: lp.n_eq] + dual_damping_eq * state.dual_eq
        grad_dual_ineq = residual[lp.n_eq :] + dual_damping_ineq * state.dual_ineq
        return SaddleState(
            primal=grad_primal,
            dual_ineq=grad_dual_ineq,
            dual_eq=grad_dual_eq,
        )

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
    def unpack_k(opt_state):
        if k_scaling:
            if halpern:
                inner, (k, eta, anchor) = opt_state
                return inner, k, eta, anchor
            if adaptive_step:
                inner, (k, eta) = opt_state
                return inner, k, eta
            return opt_state
        return opt_state, None

    def pack_k(opt_state, k, eta=None, anchor=None):
        if k_scaling:
            if halpern:
                return (opt_state, (k, eta, anchor))
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
    )
    run_epoch = _LINEAR_RUN_EPOCH_CACHE.get(cache_key)

    if run_epoch is None:

        # `average_state` is deliberately not donated: when averaging is off the
        # caller passes the same buffer for both `state` and `average_state`, and
        # XLA rejects donating one buffer twice.
        @functools.partial(
            jax.jit,
            donate_argnames=("state", "opt_state", "total_weight"),
        )
        def run_epoch(
            max_iter,
            start_iter,
            state,
            average_state,
            opt_state,
            total_weight=0.0,
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
                return jnp.where(
                    interaction > 0.0, move / (2.0 * interaction), jnp.inf
                )

            def make_adaptive_step(trial):
                def step(carry, _):
                    i, state, average_state, opt_state, total_weight = carry
                    if halpern:
                        opt_state, k, eta, anchor = unpack_k(opt_state)
                    else:
                        opt_state, k, eta = unpack_k(opt_state)
                    ip1 = jnp.asarray(i + 1, eta.dtype)

                    # Retry: while the trial step exceeds its admissible bound,
                    # shrink eta to just under eta_bar and re-trial. The
                    # (1-(i+1)^-0.3) factor < 1 guarantees strict decrease, so the
                    # loop terminates. Carry (eta, cand, eta_bar).
                    def cond(c):
                        eta_c, _, eta_bar_c = c
                        return eta_c > eta_bar_c

                    def body(c):
                        eta_c, _, eta_bar_c = c
                        eta_s = jnp.minimum((1.0 - ip1 ** (-0.3)) * eta_bar_c, eta_c)
                        cand_s, eta_bar_s = trial(eta_s, state, k)
                        return (eta_s, cand_s, eta_bar_s)

                    cand0, eta_bar0 = trial(eta, state, k)
                    eta0, cand, eta_bar = jax.lax.while_loop(
                        cond, body, (eta, cand0, eta_bar0)
                    )

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
                        opt_state = pack_k(opt_state, k, eta_next, anchor)
                    else:
                        opt_state = pack_k(opt_state, k, eta_next)

                    if average:
                        w = weight_function(i)
                        total_weight = total_weight + w
                        average_state = optax.incremental_update(
                            new_state, average_state, w / total_weight
                        )

                    return (i + 1, new_state, average_state, opt_state, total_weight), None

                return step

            if (adaptive_step and update_mode == "pdhg") or halpern:
                # PDHG raw step: primal first, then dual reads the EXTRAPOLATED
                # primal x_bar = 2 x_new - x_old. interaction = dyᵀ A dx (one A·dx).
                # Halpern uses this same PDHG operator as its base T(z); the anchor
                # combination is applied in make_adaptive_step.
                def _trial(eta, state, k):
                    tau = eta / k
                    sigma = eta * k
                    gp = grad_primal_only(state)
                    x_new = projection_primal(state.primal - tau * gp)
                    x_bar = 2.0 * x_new - state.primal
                    extrapolated = SaddleState(
                        primal=x_bar,
                        dual_ineq=state.dual_ineq,
                        dual_eq=state.dual_eq,
                    )
                    gd_ineq, gd_eq = grad_dual_only(extrapolated)
                    dual_ineq = projection_non_negative(state.dual_ineq - sigma * gd_ineq)
                    dual_eq = state.dual_eq - sigma * gd_eq
                    cand = SaddleState(
                        primal=x_new, dual_ineq=dual_ineq, dual_eq=dual_eq
                    )
                    A_dx = lp.A @ (x_new - state.primal)
                    dy = jnp.concatenate(
                        [dual_eq - state.dual_eq, dual_ineq - state.dual_ineq]
                    )
                    interaction = jnp.abs(jnp.vdot(dy, A_dx))
                    return cand, _descent_bound(state, cand, k, interaction)

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

                def _trial(eta, state, k):
                    tau = eta / k
                    sigma = eta * k
                    g = grad(state)
                    # Look-ahead z_half = proj(z - step ∘ g): descend primal,
                    # subtract the optax-convention dual gradient (matches the
                    # non-adaptive extragradient / pdhg sign).
                    xh = projection_primal(state.primal - tau * g.primal)
                    yh_ineq = projection_non_negative(state.dual_ineq - sigma * g.dual_ineq)
                    yh_eq = state.dual_eq - sigma * g.dual_eq
                    state_half = SaddleState(primal=xh, dual_ineq=yh_ineq, dual_eq=yh_eq)
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
                    eta_bar = jnp.where(
                        dg2 > 0.0, _MT * jnp.sqrt(dz2 / dg2), jnp.inf
                    )
                    return cand, eta_bar

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

                    return (i + 1, state, average_state, opt_state, total_weight), None

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

                    return (i + 1, state, average_state, opt_state, total_weight), None

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

                    return (i + 1, state, average_state, opt_state, total_weight), None

            else:

                def step(carry, _):
                    i, state, average_state, opt_state, total_weight = carry
                    opt_state, k = unpack_k(opt_state)

                    g = grad(state)
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

                    return (i + 1, state, average_state, opt_state, total_weight), None

            end_iter = start_iter + max_iter

            def cond(carry):
                i, *_ = carry
                return i < end_iter

            i, state, average_state, opt_state, total_weight = jax.lax.while_loop(
                cond,
                lambda carry: step(carry, None)[0],
                (start_iter, state, average_state, opt_state, total_weight),
            )

            return i, state, average_state, opt_state, total_weight

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

    return run_epoch(
        max_iter,
        start_iter,
        state,
        average_state,
        opt_state,
        total_weight,
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
    lp: LP,
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
    dual_gap_tolerance=1e-4,
    primal_grad_norm_tolerance=None,
    complementarity_slack_tolerance=None,
    weight_function=lambda _: 1.0,
    verbose=False,
    log_every=1,
    average=True,
    report_best=True,
    update_mode="pdhg",
    k_scale=None,
    k_theta=0.01,
    k_init=None,
    adaptive_eta=None,
    scale="ruiz+pc",
    output_opt_state=False,
    scaled_objective=False,
    restarts=0,
    epochs_per_restart=10,
    restart_multiplier=1.0,
    restart_decay=0.2,
    primal_stop=False,
    primal_stop_window=5,
    primal_stop_obj_tol=1e-4,
    iterations_per_epoch_decay=1.0,
    iterations_per_epoch_min=100,
    eq_projection_threshold=None,
    precompile=True,
    vertex_bias=0.0,
    vertex_bias_seed=0,
):
    """
    Solve a linear program via saddle-point optimisation.

    Termination uses the standard LP optimality certificate: primal feasibility
    (``primal_feasibility_tolerance``), dual feasibility
    (``dual_feasibility_tolerance``), and a finite duality gap within
    ``dual_gap_tolerance``). An alternative KKT certificate — primal feasibility
    + complementary slackness + small primal gradient norm (‖c + Aᵀy‖) — fires
    when both ``primal_grad_norm_tolerance`` and
    ``complementarity_slack_tolerance`` are set; convergence is declared when
    either certificate is satisfied.

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
        epochs_per_restart: Length cap (in epochs) of the first restart cycle
            (default 10). Subsequent cycle caps grow by ``restart_multiplier``.
        restart_multiplier: Geometric growth factor for cycle-length caps
            (default 1.0 = fixed length, 2.0 = doubling).
        restart_decay: Sufficient-progress threshold (default 0.2). A restart
            fires early if the KKT merit drops below ``restart_decay`` times its
            value at the last restart.
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
        eq_projection_threshold: When set, after each epoch the unscaled equality
            residual is checked; if it exceeds this value the primal (and average)
            are projected onto the equality manifold ``A_eq x = b_eq`` via the
            precomputed factorisation of ``A_eq A_eq^T``. Default ``None``
            disables projection. Only useful when equality feasibility is the
            bottleneck; has no effect when there are no equality constraints.
    """

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
        lp, row_scale_ruiz, col_scale_ruiz = ruiz_scaling(lp)
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

        projected_primal = projection_box(
            average_state.primal - grad_primal,
            lp.lower_bounds,
            lp.upper_bounds,
        )
        projected_gradient_residual = average_state.primal - projected_primal
        primal_grad_norm = jnp.max(jnp.abs(projected_gradient_residual))

        ineq_violations = jnp.maximum(grad_dual_ineq_unscaled, 0.0)
        max_ineq_violation = jnp.max(ineq_violations)

        eq_violations = jnp.abs(grad_dual_eq_unscaled)
        max_eq_violation = jnp.max(eq_violations)

        complementarity_slack = jnp.max(
            jnp.abs(average_state.dual_ineq * grad_dual_ineq_unscaled)
        ) / (1.0 + jnp.abs(objective_value))

        constraint_bound = jnp.maximum(max_ineq_violation, max_eq_violation)

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

        box_infimum = jnp.where(
            has_both_bounds,
            jnp.minimum(lower_term, upper_term),
            jnp.where(
                has_only_lower,
                lower_term,
                jnp.where(has_only_upper, upper_term, 0.0),
            ),
        )

        # For box-constrained variables the violation is the projected-gradient
        # magnitude: |x - proj(x - r, lb, ub)|.  For one-sided or free
        # variables the classical reduced-cost sign rules apply.
        proj_box = projection_box(
            average_state.primal - reduced_cost, lower_bounds, upper_bounds
        )
        dual_feasibility_violation = jnp.where(
            has_both_bounds,
            jnp.abs(average_state.primal - proj_box),
            jnp.where(
                has_only_lower,
                jnp.maximum(-reduced_cost, 0.0),
                jnp.where(
                    has_only_upper,
                    jnp.maximum(reduced_cost, 0.0),
                    jnp.abs(reduced_cost),  # has_no_bounds (free variable)
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

    def converged(
        constraint_bound,
        dual_feasibility_residual,
        duality_gap,
        dual_gap_is_finite,
        primal_grad_norm,
        complementarity_slack,
        objective_value,
    ):
        # Standard LP optimality certificate — all three conditions must hold:
        #   * primal feasibility: constraint_bound within tolerance
        #   * dual feasibility: reduced-cost residual within tolerance
        #   * a finite, small duality gap
        # The gap decomposition is rescaled by `c_max` into true objective units
        # (see the decomposition comment above), so it is unit-consistent with
        # `objective_value`. We test a *relative* gap, normalised by
        # (1 + |objective_value|), matching HiGHS/PDLP's relative-gap stopping
        # convention so benchmark comparisons are apples-to-apples. The same
        # (1 + |obj|) normalisation is used for `complementarity_slack` above.
        relative_duality_gap = jnp.abs(duality_gap) / (1.0 + jnp.abs(objective_value))
        standard = (
            (constraint_bound <= primal_feasibility_tolerance)
            & (dual_feasibility_residual <= dual_feasibility_threshold)
            & dual_gap_is_finite
            & (relative_duality_gap <= dual_gap_tolerance)
        )
        # Alternative KKT certificate: primal feasibility + complementary
        # slackness + small primal gradient norm (‖c + Aᵀy‖). Active only when
        # both tolerances are set; otherwise this branch never fires.
        if (
            primal_grad_norm_tolerance is not None
            and complementarity_slack_tolerance is not None
        ):
            kkt_alt = (
                (constraint_bound <= primal_feasibility_tolerance)
                & (primal_grad_norm <= primal_grad_norm_tolerance)
                & (complementarity_slack <= complementarity_slack_tolerance)
            )
            return standard | kkt_alt
        return standard

    def converged_primal(constraint_bound, obj_window):
        # Dual-free, opt-in stopping rule for "I just want the primal solved".
        # Optimality is certified heuristically (no dual): primal feasibility
        # (`constraint_bound`, already the correct per-block ineq/eq measure) AND
        # the objective having stalled — relative change across the last
        # `primal_stop_window` epochs below `primal_stop_obj_tol`. `obj_window`
        # holds the most recent objective values (oldest first). The duality gap
        # and dual residuals are still computed but do NOT gate here.
        feasible = constraint_bound <= primal_feasibility_tolerance
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
    count = 0
    total_weight = 0.0
    reported_used_avg = average
    is_converged = True
    current_iterations_per_epoch = iterations_per_epoch

    # Iterate at the last restart (or the start), used to rebalance the primal
    # weight k from the primal-vs-dual movement over the restart cycle. Must be an
    # independent copy: `state` aliases initial_solution's buffer and is donated
    # to (freed by) __sps each epoch, so sharing it would read as deleted at the
    # first restart's k-rebalance.
    state_at_last_restart = jax.tree.map(lambda x: x + 0, initial_solution)

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
    current_cycle_cap = float(epochs_per_restart)
    merit_at_last_restart = jnp.inf

    # Normalisation constants for the restart merit (PDLP-style). Each KKT
    # residual is divided by 1 + its natural scale so the three terms are
    # comparable across very differently-scaled problems and the merit reflects
    # *relative* progress, not absolute units. Computed in the scaled space the
    # solver iterates in; `c_max` puts the objective back in true units.
    b_norm = float(jnp.max(jnp.abs(lp.b))) if lp.b.size else 0.0
    c_norm = float(jnp.max(jnp.abs(lp.c))) if lp.c.size else 0.0

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
            jnp.abs(duality_gap) / (1.0 + jnp.abs(objective_value)),
            jnp.inf,
        )
        primal_term = constraint_bound / (1.0 + b_norm)
        dual_term = dual_feasibility_residual / (1.0 + c_norm)
        return jnp.maximum(jnp.maximum(primal_term, dual_term), gap_term)

    if precompile:
        if verbose:
            # The precompile below runs a full epoch + metrics through XLA before
            # the loop. On large LPs (e.g. nug, m≈20k) the first XLA compile can
            # take tens of seconds with NO output — which looks like a hang. Print
            # a bracket so the wait is visible. (Pass precompile=False to fold the
            # compile into epoch 1 instead.)
            print("Precompiling epoch (first XLA compile; may take a while)...")
            print("----------------------------------------------")
            _precompile_t0 = time.time()
        # run_epoch donates its state/opt_state buffers, so feed the warm-up
        # *copies* — the real state/opt_state the timed loop uses must survive.
        _warm_state = jax.tree.map(lambda x: x + 0, state)
        _warm_opt_state = jax.tree.map(lambda x: x + 0, opt_state)
        _precompile_result = __sps(
            1,
            0,
            lp,
            optimiser,
            _warm_state,
            average_state,
            _warm_opt_state,
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
        )
        # Warm the end-of-epoch metrics fn too, with the exact state the hot
        # loop will feed it (average vs iterate is a Python-static choice), so
        # epoch 1 doesn't pay its first-call compile inside the timed loop.
        _precompile_metrics = compute_epoch_metrics(average_state if average else state)
        jax.block_until_ready((_precompile_result, _precompile_metrics))
        if verbose:
            print(f"Precompile done in {time.time() - _precompile_t0:.1f}s")
            print("----------------------------------------------")

    def print_epoch_metrics(epoch_time=None):
        dual_gap_status = "finite" if bool(dual_gap_is_finite) else "dual-infeasible"
        time_str = f"|Time {epoch_time:.2f}s|" if epoch_time is not None else ""
        print(
            f"|Epoch {count}|"
            f"|Obj{objective_value:.2e}|"
            f"|PGN {primal_grad_norm:.2e}|"
            f"|CS {complementarity_slack:.2e}|"
            f"|PFR {constraint_bound:.2e}|"
            f"|DFR {dual_feasibility_residual:.2e}|"
            f"|RDG {duality_gap / (1.0 + jnp.abs(objective_value)):.2e} ({dual_gap_status})|"
            f"{time_str}"
        )
        print("----------------------------------------------")

    def is_done():
        if primal_stop:
            return bool(converged_primal(constraint_bound, obj_window))
        return bool(
            converged(
                constraint_bound,
                dual_feasibility_residual,
                duality_gap,
                dual_gap_is_finite,
                primal_grad_norm,
                complementarity_slack,
                objective_value,
            )
        )

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
            )
            # __sps increments the (restart-shifted) counter; restore global i.
            i = shifted_i + restart_i_offset

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

            if verbose and (count == 1 or count % log_every == 0):
                print_epoch_metrics(finish_epoch_time - start_epoch_time)

            # --- Adaptive restart decision ---
            if restarts and restarts_done < restarts:
                epochs_since_restart += 1

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
                cycle_exhausted = epochs_since_restart >= current_cycle_cap
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

                # Seed the baseline on the first finite merit so the
                # sufficient-progress test has something real to compare against
                # (avoids a spurious restart against the initial inf baseline).
                if not jnp.isfinite(merit_at_last_restart):
                    merit_at_last_restart = restart_merit

                sufficient_progress = bool(
                    restart_merit <= restart_decay * merit_at_last_restart
                )

                if sufficient_progress or cycle_exhausted:
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
                        dp = state.primal - state_at_last_restart.primal
                        dd = jnp.concatenate(
                            [
                                state.dual_eq - state_at_last_restart.dual_eq,
                                state.dual_ineq - state_at_last_restart.dual_ineq,
                            ]
                        )
                        move_p = jnp.linalg.norm(dp) + 1e-30
                        move_d = jnp.linalg.norm(dd) + 1e-30
                        k_target = move_p / move_d
                        # The k-slot is (k, eta[, anchor]) in adaptive_step mode,
                        # plain k otherwise. Read k accordingly.
                        k_prev = opt_state[1][0] if adaptive_step else opt_state[1]
                        log_k = k_theta * jnp.log(k_target) + (1.0 - k_theta) * jnp.log(
                            k_prev
                        )
                        k_new = jnp.clip(jnp.exp(log_k), k_lo, k_hi)
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
                            # Reset eta to its seed alongside the momentum/LR
                            # schedule reset, so the adaptive rule re-warms from a
                            # known step in the new cycle.
                            _eta_dtype = state.primal.dtype
                            k_slot = (k_new, jnp.asarray(adaptive_eta, _eta_dtype))
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
                    # total_weight = 0.0
                    restart_i_offset = i - 1
                    merit_at_last_restart = restart_merit
                    epochs_since_restart = 0
                    current_cycle_cap *= restart_multiplier
                    current_iterations_per_epoch = max(
                        iterations_per_epoch_min,
                        int(current_iterations_per_epoch * iterations_per_epoch_decay),
                    )
                    restarts_done += 1
                    if verbose:
                        reason = (
                            "sufficient-progress"
                            if sufficient_progress
                            else "cycle-cap"
                        )
                        which = "avg" if restart_used_avg else "iterate"
                        if k_scaling:
                            _k_show = opt_state[1][0] if adaptive_step else opt_state[1]
                            k_msg = f", k={float(_k_show):.3e}"
                        else:
                            k_msg = ""
                        print(
                            f"Restart {restarts_done}/{restarts} at epoch {count} "
                            f"({reason}, merit={float(restart_merit):.2e} "
                            f"[{which}], next cap={current_cycle_cap:.0f} epochs, "
                            f"iters/epoch={current_iterations_per_epoch}{k_msg})"
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

    if output_opt_state:
        return output, is_converged, opt_state
    else:
        return output, is_converged


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


def ruiz_scaling(lp: LP, max_iter=30, threshold=1e-8, clip_bounds=(1e-6, 1e6)):
    """
    Applies Ruiz scaling to an LP in standard form with sparse matrices:
        min c^T x
        s.t. A_eq x = b_eq
             A_ineq x <= b_ineq
             lower_bounds <= x <= upper_bounds
    Returns scaled LP, row_scaling (length m), col_scaling (length n).

    Scaling is derived from the augmented matrix [[A, b], [c^T, 0]] so that
    both cost and constraint information drive the equilibration.
    """

    A_eq = lp.A_eq
    A_ineq = lp.A_ineq
    A = sp.vstack([A_eq, A_ineq]).tocsr()
    m, n = A.shape
    b = np.concatenate([lp.b_eq, lp.b_ineq])
    c = lp.c.copy()
    lower_bounds = lp.lower_bounds.copy()
    upper_bounds = lp.upper_bounds.copy()

    c_norm = np.max(np.abs(c)) or 1.0

    # Build [[A, b], [c^T/c_norm, 0]] as a sparse (m+1) x (n+1) matrix
    b_col = sp.csc_matrix(b.reshape(-1, 1))
    c_row = sp.csc_matrix((c / c_norm).reshape(1, -1))
    zero = sp.csc_matrix((1, 1))
    M = sp.bmat([[A, b_col], [c_row, zero]]).tocsc()

    row_scale = np.ones(m + 1)
    col_scale = np.ones(n + 1)

    # Precompute |M| once and reuse it for every equilibration step. The norms
    # of D_r @ M @ D_c are obtained by scaling the row/col norms of |M| with the
    # current scale vectors, so we never rematerialise the scaled matrix.
    absM = M.copy()
    absM.data = np.abs(absM.data)
    absM_csr = absM.tocsr()
    absM_csc = absM.tocsc()
    rows = absM_csr.indices  # column index of each nnz, grouped by row
    cols = absM_csc.indices  # row index of each nnz, grouped by col

    for _ in range(max_iter):
        # Row max of D_r @ M @ D_c == row_scale * max over each row of (|M| * col_scale)
        scaled = absM_csr.data * col_scale[rows]
        row_norms = np.maximum.reduceat(scaled, absM_csr.indptr[:-1]) * row_scale
        # reduceat misbehaves on empty rows; guard with indptr deltas
        empty = np.diff(absM_csr.indptr) == 0
        row_norms[empty] = 0.0
        row_norms = np.where(row_norms <= threshold, 1.0, row_norms)
        row_s = np.clip(1.0 / np.sqrt(row_norms), clip_bounds[0], clip_bounds[1])
        row_scale *= row_s

        scaled = absM_csc.data * row_scale[cols]
        col_norms = np.maximum.reduceat(scaled, absM_csc.indptr[:-1]) * col_scale
        empty = np.diff(absM_csc.indptr) == 0
        col_norms[empty] = 0.0
        col_norms = np.where(col_norms <= threshold, 1.0, col_norms)
        col_s = np.clip(1.0 / np.sqrt(col_norms), clip_bounds[0], clip_bounds[1])
        col_scale *= col_s

    dr = row_scale[:m]
    dc = col_scale[:n]

    A_final = sp.diags(dr) @ A @ sp.diags(dc)
    b_scaled = dr * b
    c_scaled = c * dc

    A_eq_scaled = A_final[: lp.A_eq.shape[0], :]
    A_ineq_scaled = A_final[lp.A_eq.shape[0] :, :]
    b_eq_scaled = b_scaled[: lp.A_eq.shape[0]]
    b_ineq_scaled = b_scaled[lp.A_eq.shape[0] :]
    lower_bounds_scaled = lower_bounds / dc
    upper_bounds_scaled = upper_bounds / dc

    lp_scaled = LP(
        c_scaled,
        A_eq_scaled,
        b_eq_scaled,
        A_ineq_scaled,
        b_ineq_scaled,
        lower_bounds_scaled,
        upper_bounds_scaled,
    )

    return lp_scaled, dr, dc


def pc_scaling(lp: LP, max_iter=1, threshold=1e-8, clip_bounds=(1e-6, 1e6)):
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

    A_eq = lp.A_eq
    A_ineq = lp.A_ineq
    A = sp.vstack([A_eq, A_ineq]).tocsr()
    m, n = A.shape
    b = np.concatenate([lp.b_eq, lp.b_ineq])
    c = lp.c.copy()
    lower_bounds = lp.lower_bounds.copy()
    upper_bounds = lp.upper_bounds.copy()

    c_norm = np.max(np.abs(c)) or 1.0

    # Build [[A, b], [c^T/c_norm, 0]] as a sparse (m+1) x (n+1) matrix
    b_col = sp.csc_matrix(b.reshape(-1, 1))
    c_row = sp.csc_matrix((c / c_norm).reshape(1, -1))
    zero = sp.csc_matrix((1, 1))
    M = sp.bmat([[A, b_col], [c_row, zero]]).tocsc()

    row_scale = np.ones(m + 1)
    col_scale = np.ones(n + 1)

    # Precompute |M| once. L1 row/col norms of D_r @ M @ D_c factor as
    # row_scale * (|M| @ col_scale) and col_scale * (|M|^T @ row_scale), so we
    # avoid rematerialising the scaled matrix on every iteration.
    absM = M.copy()
    absM.data = np.abs(absM.data)
    absM_csr = absM.tocsr()

    for _ in range(max_iter):
        row_norms = (absM_csr @ col_scale) * row_scale
        row_norms = np.where(row_norms <= threshold, 1.0, row_norms)
        row_s = np.clip(1.0 / np.sqrt(row_norms), clip_bounds[0], clip_bounds[1])
        row_scale *= row_s

        col_norms = (absM_csr.T @ row_scale) * col_scale
        col_norms = np.where(col_norms <= threshold, 1.0, col_norms)
        col_s = np.clip(1.0 / np.sqrt(col_norms), clip_bounds[0], clip_bounds[1])
        col_scale *= col_s

    dr = row_scale[:m]
    dc = col_scale[:n]

    A_final = sp.diags(dr) @ A @ sp.diags(dc)
    b_scaled = dr * b
    c_scaled = c * dc

    A_eq_scaled = A_final[: lp.A_eq.shape[0], :]
    A_ineq_scaled = A_final[lp.A_eq.shape[0] :, :]
    b_eq_scaled = b_scaled[: lp.A_eq.shape[0]]
    b_ineq_scaled = b_scaled[lp.A_eq.shape[0] :]
    lower_bounds_scaled = lower_bounds / dc
    upper_bounds_scaled = upper_bounds / dc

    lp_scaled = LP(
        c_scaled,
        A_eq_scaled,
        b_eq_scaled,
        A_ineq_scaled,
        b_ineq_scaled,
        lower_bounds_scaled,
        upper_bounds_scaled,
    )

    return lp_scaled, dr, dc


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
