# %%
import jax
import jax.numpy as jnp
from optax.projections import projection_non_negative, projection_box
import optax
from typing import NamedTuple
import time
from jaddle.jaddle_basic_types import JaddleCP, SaddleState
import jaddle.jaddle_optimisers as jo
import numpy as np

# Epoch budget for a chunk that should run until convergence: large enough never
# to bind, small enough that `count + n` cannot overflow an int32 counter.
_UNBOUNDED_EPOCHS = 2**30
# Target wall time of one chunk of epochs between host check-ins.
_CHUNK_TARGET_SECONDS = 1.0
# Buffered verbose-log rows / restart events per chunk.
_LOG_CAP = 64
_EVT_CAP = 64

_ADAPTIVE_MODES = ("extragradient", "forward_reflected")


def _init_k_slot(k, eta, state, update_mode):
    # The k-slot rides in opt_state next to the optimiser state:
    #   - k-scaling only (eta None):        k
    #   - adaptive extragradient:           (k, eta)
    #   - adaptive forward-reflected:       (k, eta, eta_prev, F_cur, F_prev, valid)
    # F_cur/F_prev are the saddle operator at the current/previous iterate and
    # eta_prev the previous accepted step, which the reflection term needs.
    # valid=False makes the next epoch recompute F_cur = F_prev = F(state) (fresh
    # start or restart), so the first step is a plain forward-backward step.
    dtype = state.primal.dtype
    k = jnp.asarray(k, dtype)
    if eta is None:
        return k
    eta = jnp.asarray(eta, dtype)
    if update_mode != "forward_reflected":
        return (k, eta)
    zeros = jax.tree.map(jnp.zeros_like, state)
    zeros2 = jax.tree.map(jnp.zeros_like, state)
    return (k, eta, eta + 0, zeros, zeros2, jnp.asarray(False))


def _make_epoch_fn(
    cp: JaddleCP,
    optimiser,
    weight_function=lambda _: 1.0,
    average=True,
    update_mode="alternating",
    k_scaling=False,
    adaptive_step=False,
):
    # Returns `run_epoch(max_iter, start_iter, state, average_state, opt_state,
    # total_weight)`, which runs one epoch of `max_iter` iterations and returns
    # (i, state, average_state, opt_state, total_weight). It is a plain
    # traceable function, not jitted: `solve` calls it inside its device-side
    # epoch loop (`run_chunk`), which is jitted as a whole. `max_iter` must be a
    # Python int (it is the scan length).
    #
    # Per-iteration adaptive step size (Malitsky-Tam local-Lipschitz line
    # search). When `adaptive_eta` is not None the extragradient scheme replaces
    # the optimiser's fixed learning rate with a single scalar base step eta,
    # line-searched every iteration: the look-ahead and corrector already
    # evaluate grad twice, so the local Lipschitz estimate
    #     L_hat = ‖g_half - g‖_w / ‖z_half - z‖_w
    # (w = the k-weighted norm) comes free, and the step is admissible while
    # eta · L_hat <= 1/sqrt(2). The primal/dual steps are tau=eta/k, sigma=eta*k.
    # eta is packed alongside k in the k-slot of opt_state. Requires k_scaling
    # (it needs k) and a contractive scheme (extragradient or
    # forward-reflected); alternating is not contractive on the saddle and a
    # line search cannot fix that, so it is excluded.
    if adaptive_step and not k_scaling:
        raise ValueError("adaptive_eta requires k_scaling (primal weight k)")
    # k-scaling is an orthogonal option (any update_mode): a primal weight k
    # rescales the primal/dual gradients by (1/k, k) before opt_update, so the
    # dual/primal step ratio is k**2. When on, k is packed into opt_state and
    # rebalanced at each restart in `solve` (PDLP-style); constant within an
    # epoch.

    def projection_primal(primal_state):
        return projection_box(primal_state, cp.lower_bounds, cp.upper_bounds)

    # Hand-written saddle gradient of the Lagrangian
    #   L = obj(x) + d_ineq·g_ineq(x) + d_eq·g_eq(x).
    # Its structure lets each half be computed independently:
    #   - dual partials are just the constraint residuals g(x) — a forward eval,
    #     no autodiff;
    #   - the primal partial is grad(obj)(x) + J_ineqᵀ·d_ineq + J_eqᵀ·d_eq,
    #     a single VJP of the constraint map against the duals plus the objective
    #     gradient.
    # The negation on the dual partials (descent on x / ascent on the duals) is
    # folded into the returned dual fields. Splitting `grad` into one-sided
    # variants lets the alternating/extragradient schemes pay for only the half
    # they actually consume.

    def constraints(primal):
        return (cp.constraints_ineq(primal), cp.constraints_eq(primal))

    def lagrangian_map(primal):
        # Stacks the objective and both constraint maps into one function so a
        # single reverse pass covers all of them. The primal partial of the
        # Lagrangian is the VJP of this map seeded with cotangents
        # (1.0, dual_ineq, dual_eq): the 1.0 on the objective output reproduces
        # grad(obj), and the dual cotangents reproduce Jᵀ·dual — in one
        # traversal of the user's graph instead of grad(obj) + a separate
        # constraints VJP.
        return (
            cp.objective(primal),
            cp.constraints_ineq(primal),
            cp.constraints_eq(primal),
        )

    def grad_primal_only(state):
        # Primal partial only: grad(obj) + Jᵀ·dual, via one fused VJP. The
        # objective/residual outputs are produced by the VJP's forward pass and
        # discarded here.
        one = jnp.ones((), state.primal.dtype)
        _, vjp_fn = jax.vjp(lagrangian_map, state.primal)
        (primal_grad,) = vjp_fn((one, state.dual_ineq, state.dual_eq))
        return primal_grad

    def grad_dual_only(state):
        # Dual partials only: the (negated) constraint residuals. A plain
        # forward eval — no objective gradient, no VJP.
        res_ineq, res_eq = constraints(state.primal)
        return -res_ineq, -res_eq

    def grad(state):
        # Full saddle gradient in a single reverse pass: one fused VJP over
        # (objective, constraints_ineq, constraints_eq) seeded with
        # (1.0, dual_ineq, dual_eq). The forward pass yields the objective value
        # (discarded) and the residuals (reused for the dual partials), so the
        # whole saddle gradient costs one VJP instead of grad(obj) + a separate
        # constraints VJP.
        one = jnp.ones((), state.primal.dtype)
        (_, res_ineq, res_eq), vjp_fn = jax.vjp(lagrangian_map, state.primal)
        (primal_grad,) = vjp_fn((one, state.dual_ineq, state.dual_eq))
        return SaddleState(
            primal=primal_grad,
            dual_ineq=-res_ineq,
            dual_eq=-res_eq,
        )

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
    # helpers keep the step bodies agnostic to whether k is packed or not. In
    # adaptive_step mode the packed k-slot is a (k, eta) pair so the
    # per-iteration step size eta rides alongside the primal weight k; otherwise
    # it is just k (or absent when k_scaling is off).
    def unpack_k(opt_state):
        if k_scaling:
            if adaptive_step:
                inner, (k, eta) = opt_state
                return inner, k, eta
            return opt_state
        return opt_state, None

    def pack_k(opt_state, k, eta=None):
        if k_scaling:
            if adaptive_step:
                return (opt_state, (k, eta))
            return (opt_state, k)
        return opt_state


    def run_epoch(
        max_iter,
        start_iter,
        state,
        average_state,
        opt_state,
        total_weight,
    ):
        # --- Adaptive extragradient step (Malitsky-Tam line search) ---
        # Mirrors the linear solver's adaptive extragradient path. Each
        # iteration takes a raw look-ahead + corrector at base step eta,
        # estimates the local Lipschitz constant from the two gradient
        # evaluations, and shrinks eta below eta_bar = (1/sqrt2)/L_hat if the
        # trial overshot. No optimiser learning rate is consumed here — eta
        # is the only step control.
        if adaptive_step:
            _MT = 1.0 / jnp.sqrt(2.0)

            # Local Lipschitz estimates are taken in the k-weighted geometry
            # the steps live in (tau = eta/k, sigma = eta*k): displacements
            # in the M-norm k‖dx‖² + ‖dy‖²/k, operator differences in the
            # dual M⁻¹-norm ‖dg_x‖²/k + k‖dg_y‖².
            def _znorm2(p, de, di, k):
                return k * jnp.vdot(p, p) + (1.0 / k) * (
                    jnp.vdot(de, de) + jnp.vdot(di, di)
                )

            def _gnorm2(p, de, di, k):
                return (1.0 / k) * jnp.vdot(p, p) + k * (
                    jnp.vdot(de, de) + jnp.vdot(di, di)
                )

            def _advance_eta(eta_bar, eta0, ip1):
                # Advance eta for the next iterate (growth allowed once the
                # step is accepted). When eta_bar is +inf the step did not
                # move: hold eta rather than letting growth run to NaN.
                eta_next = jnp.minimum(
                    (1.0 - ip1 ** (-0.3)) * eta_bar,
                    (1.0 + ip1 ** (-0.6)) * eta0,
                )
                eta_next = jnp.where(jnp.isfinite(eta_bar), eta_next, eta0)
                eta_next = jnp.where(jnp.isfinite(eta_next), eta_next, eta0)
                return jnp.maximum(eta_next, 1e-12)

            def _line_search(trial, eta, ip1, keep):
                # Branch-free line search: one trial per iteration. If it
                # overshot its admissible bound (eta > eta_bar), the trial
                # is discarded — the outputs fall back to `keep` (the
                # unchanged iterate) — and eta shrinks to just under
                # eta_bar, so the next iteration retries from the same point.
                # A retry loop (lax.while_loop) would force a device->host
                # sync on its predicate every iteration, which dominates the
                # per-iteration cost on GPU for cheap problems (~8x/epoch on
                # isotonic n=10k); a rejection here only costs one iteration.
                out, eta_bar = trial(eta)
                accept = eta <= eta_bar
                out = jax.tree.map(
                    lambda a, b: jnp.where(accept, a, b), out, keep
                )
                eta_next = jnp.where(
                    accept,
                    _advance_eta(eta_bar, eta, ip1),
                    jnp.maximum(
                        jnp.minimum((1.0 - ip1 ** (-0.3)) * eta_bar, eta), 1e-12
                    ),
                )
                return out, eta_next, accept

            def _accumulate(i, new_state, average_state, total_weight, accept):
                # Rejected trials leave the iterate unchanged and carry zero
                # averaging weight.
                if average:
                    w = jnp.where(accept, weight_function(i), 0.0)
                    total_weight = total_weight + w
                    frac = jnp.where(total_weight > 0, w / total_weight, 0.0)
                    average_state = optax.incremental_update(
                        new_state, average_state, frac
                    )
                return average_state, total_weight

            if update_mode == "forward_reflected":
                # --- Adaptive forward-reflected-backward (Malitsky-Tam) ---
                #   z+ = P(z - eta F(z) - eta_prev (F(z) - F(z_prev)))
                # One operator evaluation per accepted step (F(z+) is the
                # next iteration's F(z)), versus two for extragradient. The
                # admissible step is eta · L_hat <= _FRB with L_hat measured
                # between z and z+.
                _FRB = 0.45

                inner0, (k0, eta0, eta_prev0, F_cur0, F_prev0, valid0) = opt_state

                # Fresh start / restart: seed F at the current iterate once
                # per epoch (outside the scan), so the reflection vanishes.
                def _seed():
                    g = grad(state)
                    return g, jax.tree.map(lambda x: x + 0, g)

                F_cur0, F_prev0 = jax.lax.cond(
                    valid0, lambda: (F_cur0, F_prev0), _seed
                )

                def step(carry, _):
                    (
                        i,
                        state,
                        average_state,
                        eta,
                        eta_prev,
                        F_cur,
                        F_prev,
                        total_weight,
                    ) = carry
                    ip1 = jnp.asarray(i + 1, eta.dtype)
                    k = k0
                    # Reflection term is fixed across line-search retries.
                    refl = jax.tree.map(
                        lambda a, b: eta_prev * (a - b), F_cur, F_prev
                    )

                    def trial(eta_t):
                        x_new = projection_primal(
                            state.primal
                            - (eta_t * F_cur.primal + refl.primal) / k
                        )
                        dual_ineq = projection_non_negative(
                            state.dual_ineq
                            - k * (eta_t * F_cur.dual_ineq + refl.dual_ineq)
                        )
                        dual_eq = state.dual_eq - k * (
                            eta_t * F_cur.dual_eq + refl.dual_eq
                        )
                        cand = SaddleState(
                            primal=x_new, dual_ineq=dual_ineq, dual_eq=dual_eq
                        )
                        F_new = grad(cand)
                        dg2 = _gnorm2(
                            F_new.primal - F_cur.primal,
                            F_new.dual_eq - F_cur.dual_eq,
                            F_new.dual_ineq - F_cur.dual_ineq,
                            k,
                        )
                        dz2 = _znorm2(
                            x_new - state.primal,
                            dual_eq - state.dual_eq,
                            dual_ineq - state.dual_ineq,
                            k,
                        )
                        eta_bar = jnp.where(
                            dg2 > 0.0, _FRB * jnp.sqrt(dz2 / dg2), jnp.inf
                        )
                        # Accepted: shift the F history and record the step.
                        return (cand, F_new, F_cur, eta_t), eta_bar

                    # On rejection the whole FRB state (iterate, F history,
                    # previous step) stays put; only eta shrinks.
                    (new_state, F_new, F_old, eta_prev_new), eta_next, accept = (
                        _line_search(
                            trial, eta, ip1, (state, F_cur, F_prev, eta_prev)
                        )
                    )
                    average_state, total_weight = _accumulate(
                        i, new_state, average_state, total_weight, accept
                    )
                    return (
                        i + 1,
                        new_state,
                        average_state,
                        eta_next,
                        eta_prev_new,
                        F_new,
                        F_old,
                        total_weight,
                    ), None

                (
                    i,
                    state,
                    average_state,
                    eta,
                    eta_prev,
                    F_cur,
                    F_prev,
                    total_weight,
                ), _ = jax.lax.scan(
                    step,
                    (
                        start_iter,
                        state,
                        average_state,
                        eta0,
                        eta_prev0,
                        F_cur0,
                        F_prev0,
                        total_weight,
                    ),
                    None,
                    length=max_iter,
                )
                opt_state = (
                    inner0,
                    (k0, eta, eta_prev, F_cur, F_prev, jnp.asarray(True)),
                )
                return i, state, average_state, opt_state, total_weight

            def _trial(eta, state, k, g):
                tau = eta / k
                sigma = eta * k
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

                # Local Lipschitz estimate, measured on the look-ahead
                # displacement.
                dg2 = _gnorm2(
                    g_half.primal - g.primal,
                    g_half.dual_eq - g.dual_eq,
                    g_half.dual_ineq - g.dual_ineq,
                    k,
                )
                dz2 = _znorm2(
                    xh - state.primal,
                    yh_eq - state.dual_eq,
                    yh_ineq - state.dual_ineq,
                    k,
                )
                eta_bar = jnp.where(dg2 > 0.0, _MT * jnp.sqrt(dz2 / dg2), jnp.inf)
                return cand, eta_bar

            def step(carry, _):
                i, state, average_state, opt_state, total_weight = carry
                opt_state, k, eta = unpack_k(opt_state)
                ip1 = jnp.asarray(i + 1, eta.dtype)

                g = grad(state)
                new_state, eta_next, accept = _line_search(
                    lambda eta_t: _trial(eta_t, state, k, g), eta, ip1, state
                )
                opt_state = pack_k(opt_state, k, eta_next)
                average_state, total_weight = _accumulate(
                    i, new_state, average_state, total_weight, accept
                )

                return (
                    i + 1,
                    new_state,
                    average_state,
                    opt_state,
                    total_weight,
                ), None

            (i, state, average_state, opt_state, total_weight), _ = jax.lax.scan(
                step,
                (
                    start_iter,
                    state,
                    average_state,
                    opt_state,
                    total_weight,
                ),
                None,
                length=max_iter,
            )
            return i, state, average_state, opt_state, total_weight

        def step(carry, _):
            (
                i,
                state,
                average_state,
                opt_state,
                total_weight,
            ) = carry

            opt_state, k = unpack_k(opt_state)

            if update_mode == "alternating":
                # Only the primal half of the start gradient and the dual
                # half of the post-primal gradient are ever consumed, so
                # compute just those: a VJP+obj-grad for the primal, and a
                # plain residual eval for the dual (no VJP, no obj grad).
                grad_primal_start = grad_primal_only(state)
                if k_scaling:
                    grad_primal_start = grad_primal_start / k

                primal_gradient = SaddleState(
                    primal=grad_primal_start,
                    dual_ineq=jnp.zeros_like(state.dual_ineq),
                    dual_eq=jnp.zeros_like(state.dual_eq),
                )
                primal_updates, _ = opt_update(
                    primal_gradient,
                    opt_state,
                    state,
                )
                primal_updates = keep_only_primal(primal_updates)
                state = optax.apply_updates(state, primal_updates)
                state = SaddleState(
                    primal=projection_primal(state.primal),
                    dual_ineq=state.dual_ineq,
                    dual_eq=state.dual_eq,
                )

                dual_ineq_g, dual_eq_g = grad_dual_only(state)
                if k_scaling:
                    dual_ineq_g = dual_ineq_g * k
                    dual_eq_g = dual_eq_g * k
                combined_gradient = SaddleState(
                    primal=grad_primal_start,
                    dual_ineq=dual_ineq_g,
                    dual_eq=dual_eq_g,
                )
                combined_updates, opt_state = opt_update(
                    combined_gradient,
                    opt_state,
                    state,
                )
                dual_updates = keep_only_dual(combined_updates)
                state = optax.apply_updates(state, dual_updates)
                state = SaddleState(
                    primal=state.primal,
                    dual_ineq=projection_non_negative(state.dual_ineq),
                    dual_eq=state.dual_eq,
                )
            else:
                # extragradient
                # --- Look-ahead gradient ---
                g = grad(state)

                scaled_g = scale_by_k(g, k) if k_scaling else g
                la_updates, _ = opt_update(scaled_g, opt_state, state)
                state_half = optax.apply_updates(state, la_updates)
                state_half = SaddleState(
                    primal=projection_primal(state_half.primal),
                    dual_ineq=projection_non_negative(state_half.dual_ineq),
                    dual_eq=state_half.dual_eq,
                )

                # Corrector: gradient at look-ahead point, applied from original state
                g_half = grad(state_half)
                scaled_g_half = scale_by_k(g_half, k) if k_scaling else g_half
                corr_updates, opt_state = opt_update(
                    scaled_g_half, opt_state, state
                )
                state = optax.apply_updates(state, corr_updates)
                state = SaddleState(
                    primal=projection_primal(state.primal),
                    dual_ineq=projection_non_negative(state.dual_ineq),
                    dual_eq=state.dual_eq,
                )

            opt_state = pack_k(opt_state, k)

            # `average` is a Python-static bool, so branch on it at trace
            # time rather than threading a per-step lax.cond (which would
            # force XLA to evaluate the predicate and the running-mean AXPY
            # every iteration even when averaging is off — the same class of
            # issue fixed at the epoch level for the metrics path).
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
            ), None

        (i, state, average_state, opt_state, total_weight), _ = jax.lax.scan(
            step,
            (start_iter, state, average_state, opt_state, total_weight),
            None,
            length=max_iter,
        )

        return i, state, average_state, opt_state, total_weight

    return run_epoch


def _check_settings(optimiser, adaptive_eta, update_mode, k_scale):
    """Resolve and validate the settings ``solve()`` and ``make_solver`` share.
    Returns ``(optimiser, adaptive_eta, adaptive_step, k_scaling, k_lo, k_hi)``."""
    if adaptive_eta == "auto":
        if update_mode == "forward_reflected" or (
            update_mode in _ADAPTIVE_MODES and k_scale is not None and optimiser is None
        ):
            adaptive_eta = 1.0
        else:
            adaptive_eta = None

    if optimiser is None:
        optimiser = jo.gd(1 / 2)

    valid_update_modes = ["alternating", "extragradient", "forward_reflected"]
    if update_mode not in valid_update_modes:
        raise ValueError(f"update_mode must be one of {valid_update_modes}")

    # Per-iteration adaptive step size (Malitsky-Tam line search). Only the
    # contractive schemes (extragradient, forward-reflected) support it; it also
    # needs the primal weight k, hence k_scale. forward_reflected has no
    # fixed-step (optimiser) form here, so it requires the line search.
    adaptive_step = adaptive_eta is not None and update_mode in _ADAPTIVE_MODES
    if adaptive_eta is not None:
        if update_mode not in _ADAPTIVE_MODES:
            raise ValueError(
                f"adaptive_eta is only supported with update_mode in {_ADAPTIVE_MODES}"
            )
        if k_scale is None:
            raise ValueError("adaptive_eta requires k_scale (primal weight k)")
    elif update_mode == "forward_reflected":
        raise ValueError("update_mode='forward_reflected' requires adaptive_eta")

    # ``k_scale`` is the public knob for primal-weight scaling: ``None`` disables
    # it, otherwise it sets a symmetric clamp band ``[1/k_scale, k_scale]``.
    k_scaling = k_scale is not None
    if k_scaling:
        k_lo, k_hi = 1.0 / k_scale, k_scale
    else:
        k_lo, k_hi = None, None
    return optimiser, adaptive_eta, adaptive_step, k_scaling, k_lo, k_hi


def _device_solver(
    cp,
    *,
    optimiser,
    initial_solution,
    initial_opt_state,
    iterations_per_epoch,
    primal_grad_norm_tolerance,
    primal_feasibility_tolerance,
    complementarity_slack_tolerance,
    weight_function,
    verbose,
    log_every,
    average,
    update_mode,
    k_scaling,
    k_lo,
    k_hi,
    k_theta,
    k_init,
    adaptive_eta,
    adaptive_step,
    restarts,
    epochs_per_restart,
    restart_multiplier,
    restart_decay,
    iterations_per_epoch_decay,
    iterations_per_epoch_min,
):
    """The device side of ``solve()`` for one problem: the jitted epoch loop
    (iterations, metrics, convergence test, restarts and the primal weight)
    and its initial state. ``cp``'s functions may close over traced values, so
    this also runs inside ``jit`` / ``vmap`` / ``grad`` (``make_solver``).

    Returns ``(build_chunk, carry)``: ``build_chunk(ipe)`` gives the jitted
    ``run_chunk(carry, n_epochs)`` for an epoch length, and ``carry`` is the
    initial loop state.
    """
    def projection_primal(primal_state):
        return projection_box(primal_state, cp.lower_bounds, cp.upper_bounds)

    def langrangian(state):
        return (
            cp.objective(state.primal)
            + state.dual_ineq @ cp.constraints_ineq(state.primal)
            + state.dual_eq @ cp.constraints_eq(state.primal)
        )

    def langrangian_with_obj(state):
        # Returns (lagrangian, objective) as (value, aux) so value_and_grad can
        # retrieve the objective without an extra forward pass.
        obj = cp.objective(state.primal)
        lagrangian = (
            obj
            + state.dual_ineq @ cp.constraints_ineq(state.primal)
            + state.dual_eq @ cp.constraints_eq(state.primal)
        )
        return lagrangian, obj

    def grad(state):
        gradient = jax.grad(langrangian)(state)
        return SaddleState(
            primal=gradient.primal,
            dual_ineq=gradient.dual_ineq,
            dual_eq=gradient.dual_eq,
        )

    has_dual_bound = cp.dual_bound is not None

    @jax.jit
    def compute_epoch_metrics(average_state):
        # value_and_grad with has_aux avoids a second cp.objective forward pass.
        (_, objective_value), gradient_raw = jax.value_and_grad(
            langrangian_with_obj, has_aux=True
        )(average_state)
        gradient = SaddleState(
            primal=gradient_raw.primal,
            dual_ineq=gradient_raw.dual_ineq,
            dual_eq=gradient_raw.dual_eq,
        )

        grad_primal = gradient.primal
        grad_dual_ineq = gradient.dual_ineq
        grad_dual_eq = gradient.dual_eq

        # Unscale constraint violations to original space

        projected_primal = projection_primal(average_state.primal - grad_primal)
        projected_gradient_residual = average_state.primal - projected_primal
        primal_grad_norm = jnp.max(jnp.abs(projected_gradient_residual))

        ineq_violations = jnp.maximum(grad_dual_ineq, 0.0)
        max_ineq_violation = (
            jnp.max(ineq_violations) if ineq_violations.size > 0 else jnp.zeros(())
        )

        eq_violations = jnp.abs(grad_dual_eq)
        max_eq_violation = (
            jnp.max(eq_violations) if eq_violations.size > 0 else jnp.zeros(())
        )

        complementarity_slack = (
            jnp.max(jnp.abs(average_state.dual_ineq * grad_dual_ineq))
            if average_state.dual_ineq.size > 0
            else jnp.zeros(())
        ) / (1.0 + jnp.abs(objective_value))

        constraint_bound = jnp.maximum(max_ineq_violation, max_eq_violation)

        if getattr(cp, "residuals", None) is not None:
            primal_grad_norm, constraint_bound, complementarity_slack = cp.residuals(
                average_state
            )

        if has_dual_bound:
            dual_bound = cp.dual_bound(
                average_state.dual_ineq,
                average_state.dual_eq,
            )
            duality_gap = objective_value - dual_bound
        else:
            duality_gap = jnp.nan
        dual_gap_is_finite = has_dual_bound & jnp.isfinite(duality_gap)

        return (
            objective_value,
            primal_grad_norm,
            complementarity_slack,
            constraint_bound,
            duality_gap,
            dual_gap_is_finite,
        )

    def check_convergence(
        primal_grad_norm,
        complementarity_slack,
        constraint_bound,
    ):
        # True while NOT yet converged.
        return (
            (primal_grad_norm > primal_grad_norm_tolerance)
            | (complementarity_slack > complementarity_slack_tolerance)
            | (constraint_bound > primal_feasibility_tolerance)
        )

    if initial_solution is None:
        initial_solution = cp.initial_solution()

    # PDLP-style primal-weight initialisation. When k-scaling is on and k_init is
    # left as None we derive it from the objective/RHS norms ||c|| / ||b||, which
    # puts the primal/dual step ratio in the right order of magnitude before
    # iteration 1 instead of starting symmetric. For a linear objective
    # c = grad(obj)(0), and for constraints of the form Ax - b, evaluating at
    # x = 0 gives -b, so b = -[c_eq(0); c_ineq(0)]. This is generic over
    # JaddleCP/LP/JaddleLP since it only uses the objective/constraint callables.
    if k_scaling and k_init is None:
        zero = jnp.zeros_like(initial_solution.primal)

        # Fuse into one JIT-compiled call: objective forward+grad and both
        # constraint evaluations in a single function to reduce dispatch overhead.
        def _k_init_fn(x):
            obj, c = jax.value_and_grad(cp.objective)(x)
            b = jnp.concatenate([cp.constraints_eq(x), cp.constraints_ineq(x)])
            return c, b

        c, b = jax.jit(_k_init_fn)(zero)
        norm_c2 = jnp.vdot(c, c) + 1e-60
        norm_b2 = jnp.vdot(b, b) + 1e-60
        # Kept on the device (no float()) so this runs under jit / vmap.
        k_init = jnp.clip(jnp.sqrt(norm_c2 / norm_b2), k_lo, k_hi)
    elif k_init is None:
        k_init = 1.0

    if initial_opt_state is not None:
        opt_state = initial_opt_state
    elif k_scaling:
        k_slot = _init_k_slot(
            k_init,
            adaptive_eta if adaptive_step else None,
            initial_solution,
            update_mode,
        )
        opt_state = (optimiser.init(initial_solution), k_slot)
    else:
        opt_state = optimiser.init(initial_solution)

    def kkt_merit(primal_grad_norm, complementarity_slack, constraint_bound):
        # Normalised KKT merit for the restart trigger: the maximum of the three
        # residuals. It is only compared with itself at successive restarts, so
        # a fixed normaliser of 1 is enough.
        return jnp.maximum(
            jnp.maximum(primal_grad_norm, complementarity_slack),
            constraint_bound,
        )

    def _k_of(opt_state):
        # The k-slot is a tuple led by k in adaptive_step mode, plain k otherwise.
        return opt_state[1][0] if adaptive_step else opt_state[1]

    def _rebalance_k(new_state, ref_state, k_prev):
        # PDLP-style primal-weight rebalance: drive k from the primal-vs-dual
        # *movement* over the just-finished cycle (distance between iterates),
        # not per-step gradient norms. omega = ||dy|| / ||dx|| under tau = eta/k,
        # sigma = eta*k (matches the linear solver), blended with the current
        # weight in log space (k_theta), then clamped. Squared norms avoid two
        # sqrts; the ratio is preserved.
        dp = new_state.primal - ref_state.primal
        dd = jnp.concatenate(
            [
                new_state.dual_eq - ref_state.dual_eq,
                new_state.dual_ineq - ref_state.dual_ineq,
            ]
        )
        move_p2 = jnp.vdot(dp, dp) + 1e-60
        move_d2 = jnp.vdot(dd, dd) + 1e-60
        k_target = jnp.sqrt(move_d2 / move_p2)
        log_k = k_theta * jnp.log(k_target) + (1.0 - k_theta) * jnp.log(k_prev)
        return jnp.clip(jnp.exp(log_k), k_lo, k_hi)

    def _select(pred, a, b):
        return jax.tree.map(lambda x, y: jnp.where(pred, x, y), a, b)

    def _strong(tree):
        # Strip JAX's weak typing (from Python-scalar inputs like jnp.asarray(0))
        # so the carry handed to run_chunk always has exactly the types run_chunk
        # returns; otherwise the second call retraces and recompiles the chunk.
        return jax.tree.map(
            lambda x: jax.lax.convert_element_type(x, jnp.asarray(x).dtype), tree
        )

    # --- Device-resident epoch loop -------------------------------------------
    # Epochs run in chunks: one jitted lax.while_loop (`run_chunk`) executes many
    # epochs back to back on the device -- the iterations, the end-of-epoch
    # metrics, the convergence test and the restart / primal-weight logic -- so
    # the host never waits on the device between epochs. Python regains control
    # only between chunks, to enforce max_epochs / max_seconds, print the
    # buffered epoch log and restart messages, and catch Ctrl-C. A chunk also
    # ends early when a restart decays `iterations_per_epoch` (the epoch length
    # is a static scan length, so the next chunk is compiled for the new length)
    # or when the verbose log buffers fill.
    _metric_shapes = jax.eval_shape(compute_epoch_metrics, initial_solution)
    _merit_dtype = _metric_shapes[1].dtype

    def _initial_metrics():
        # +inf everywhere (and gap finiteness False) so the convergence test
        # can't pass before the first epoch has computed real metrics.
        return tuple(
            jnp.zeros(m.shape, m.dtype)
            if m.dtype == jnp.bool_
            else jnp.full(m.shape, jnp.inf, m.dtype)
            for m in _metric_shapes
        )

    def _build_chunk(ipe):
        run_epoch = _make_epoch_fn(
            cp,
            optimiser,
            weight_function,
            average,
            update_mode,
            k_scaling,
            adaptive_step,
        )
        ipe_after_restart = max(
            iterations_per_epoch_min, int(ipe * iterations_per_epoch_decay)
        )

        def epoch(c):
            # The epoch runs on the restart-shifted index, so a restart re-zeros
            # the iteration counter the step schedules and weights see.
            start = c["i"] - c["restart_i_offset"]
            shifted_i, state, avg, opt, total_weight = run_epoch(
                ipe, start, c["state"], c["avg"], c["opt"], c["total_weight"]
            )
            i = shifted_i + c["restart_i_offset"]
            count = c["count"] + 1

            metrics = compute_epoch_metrics(avg if average else state)
            objective_value, primal_grad_norm, complementarity_slack, constraint_bound = (
                metrics[:4]
            )

            log, n_log, evt, n_evt = c["log"], c["n_log"], c["evt"], c["n_evt"]
            if verbose:
                write = (count == 1) | (count % log_every == 0)
                row = jnp.stack(
                    [
                        count,
                        objective_value,
                        primal_grad_norm,
                        complementarity_slack,
                        constraint_bound,
                    ]
                ).astype(log.dtype)
                log = jnp.where(write, log.at[n_log].set(row), log)
                n_log = n_log + write

            # --- Adaptive restart decision ---
            restart_i_offset = c["restart_i_offset"]
            at_restart = c["state_at_last_restart"]
            mal = c["merit_at_last_restart"]
            epochs_since = c["epochs_since_restart"]
            cycle_cap = c["cycle_cap"]
            ipe_next = c["ipe"]
            if restarts:
                epochs_since = epochs_since + 1
                merit = kkt_merit(
                    primal_grad_norm, complementarity_slack, constraint_bound
                )

                # Two-point restart: pick the better of average and iterate.
                restart_point = avg if average else state
                restart_merit = merit
                restart_used_avg = jnp.asarray(bool(average))
                if average:
                    st = compute_epoch_metrics(state)
                    state_merit = kkt_merit(st[1], st[2], st[3])
                    iterate_better = state_merit < merit
                    restart_point = _select(iterate_better, state, restart_point)
                    restart_merit = jnp.where(iterate_better, state_merit, merit)
                    restart_used_avg = ~iterate_better

                mal = jnp.where(jnp.isfinite(mal), mal, restart_merit)
                sufficient_progress = restart_merit <= restart_decay * mal
                cycle_exhausted = epochs_since >= cycle_cap
                restarted = sufficient_progress | cycle_exhausted

                if k_scaling:
                    k_new = _rebalance_k(restart_point, at_restart, _k_of(opt))
                    # Carry the learned step size across the restart (PDLP
                    # convention, as in the linear solver): re-seeding forced
                    # the adaptive rule to re-climb from adaptive_eta after
                    # every restart.
                    k_slot = _init_k_slot(
                        k_new,
                        opt[1][1] if adaptive_step else None,
                        restart_point,
                        update_mode,
                    )
                    opt_restart = (optimiser.init(restart_point), k_slot)
                else:
                    k_new = jnp.asarray(jnp.nan, _merit_dtype)
                    opt_restart = optimiser.init(restart_point)

                # Warm-start from the restart point; reset the optimiser state,
                # averaging, the weight accumulator and the iteration offset.
                opt = _select(restarted, opt_restart, opt)
                state = _select(restarted, restart_point, state)
                avg = _select(restarted, restart_point, avg)
                at_restart = _select(restarted, restart_point, at_restart)
                total_weight = jnp.where(restarted, 0.0, total_weight)
                restart_i_offset = jnp.where(restarted, i - 1, restart_i_offset)
                mal = jnp.where(restarted, restart_merit, mal)
                epochs_since = jnp.where(restarted, 0, epochs_since)
                cycle_cap = jnp.where(
                    restarted, cycle_cap * restart_multiplier, cycle_cap
                )
                ipe_next = jnp.where(restarted, ipe_after_restart, ipe_next)
                if verbose:
                    row = jnp.stack(
                        [
                            count,
                            jnp.where(sufficient_progress, 0, 1),
                            restart_merit,
                            restart_used_avg,
                            k_new,
                            cycle_cap,
                            ipe_next,
                        ]
                    ).astype(evt.dtype)
                    evt = jnp.where(restarted, evt.at[n_evt].set(row), evt)
                    n_evt = n_evt + restarted

            new = dict(
                state=state,
                avg=avg,
                opt=opt,
                state_at_last_restart=at_restart,
                metrics=metrics,
                count=count,
                i=i,
                restart_i_offset=restart_i_offset,
                total_weight=total_weight,
                merit_at_last_restart=mal,
                epochs_since_restart=epochs_since,
                cycle_cap=cycle_cap,
                ipe=ipe_next,
                done=~check_convergence(
                    primal_grad_norm, complementarity_slack, constraint_bound
                ),
                log=log,
                n_log=n_log,
                evt=evt,
                n_evt=n_evt,
            )
            # Keep the loop carry type-stable.
            return jax.tree.map(
                lambda n, o: jnp.asarray(n).astype(jnp.asarray(o).dtype), new, c
            )

        @jax.jit
        def run_chunk(c, n_epochs):
            end = c["count"] + n_epochs

            def cond(c):
                go = (~c["done"]) & (c["count"] < end) & (c["ipe"] == ipe)
                if verbose:
                    go = go & (c["n_log"] < _LOG_CAP) & (c["n_evt"] < _EVT_CAP)
                return go

            return jax.lax.while_loop(cond, epoch, c)

        return run_chunk

    carry = _strong(
        dict(
            state=initial_solution,
            avg=initial_solution,
            opt=opt_state,
            state_at_last_restart=initial_solution,
            metrics=_initial_metrics(),
            count=jnp.asarray(0),
            i=jnp.asarray(1),
            restart_i_offset=jnp.asarray(0),
            total_weight=jnp.asarray(0.0),
            merit_at_last_restart=jnp.asarray(jnp.inf, _merit_dtype),
            epochs_since_restart=jnp.asarray(0),
            cycle_cap=jnp.asarray(
                jnp.inf if epochs_per_restart is None else float(epochs_per_restart)
            ),
            ipe=jnp.asarray(iterations_per_epoch),
            done=jnp.asarray(False),
            log=jnp.zeros((_LOG_CAP, 5), _merit_dtype) if verbose else None,
            n_log=jnp.asarray(0),
            evt=jnp.zeros((_EVT_CAP, 7), _merit_dtype) if verbose else None,
            n_evt=jnp.asarray(0),
        )
    )
    return _build_chunk, carry


def solve(
    cp: JaddleCP,
    optimiser=None,
    max_epochs=None,
    max_seconds=None,
    initial_solution=None,
    initial_opt_state=None,
    iterations_per_epoch=int(1e3),
    primal_grad_norm_tolerance=1e-2,
    primal_feasibility_tolerance=1e-3,
    complementarity_slack_tolerance=1e-3,
    weight_function=lambda _: 1.0,
    verbose=False,
    log_every=1,
    average=False,
    update_mode="extragradient",
    k_scale=10.0,
    k_theta=0.5,
    k_init=None,
    adaptive_eta="auto",
    restarts=False,
    epochs_per_restart=None,
    restart_multiplier=1.0,
    restart_decay=0.2,
    iterations_per_epoch_decay=1.0,
    iterations_per_epoch_min=100,
    precompile=True,
):
    """
    Solve a convex saddle-point problem via saddle-point optimisation.

    Runs primal descent / dual ascent on the Lagrangian
    ``f(x) + λᵀg(x) + μᵀh(x)`` (``λ >= 0``), with the primal projected onto the
    box ``[lower_bounds, upper_bounds]``; all gradients come from ``jax.grad``.
    Iterations run in compiled epochs of ``iterations_per_epoch`` steps. After
    each epoch the convergence test is evaluated, and the solve stops once all
    three of the following are within tolerance:

    * stationarity, ``‖x - proj(x - ∇ₓL)‖∞ <= primal_grad_norm_tolerance``;
    * primal feasibility, the largest violation ``max(g(x)⁺, |h(x)|) <=
      primal_feasibility_tolerance``;
    * complementary slackness, ``max|λ ⊙ g(x)| / (1 + |f(x)|) <=
      complementarity_slack_tolerance``.

    The epoch loop runs on the device: each call into JAX executes a chunk of
    many epochs, including the metrics, convergence test and restart logic, with
    no host synchronisation between them. Python regains control between chunks
    (sized to take about a second) to enforce ``max_epochs`` / ``max_seconds``,
    print the verbose log and handle Ctrl-C; with ``verbose=True`` the per-epoch
    ``Time`` is the average over the epoch's chunk. When ``verbose``,
    ``max_epochs`` and ``max_seconds`` are all unset, the whole solve runs as a
    single device call (only an ``iterations_per_epoch_decay`` restart, which
    changes the compiled epoch length, returns to Python); Ctrl-C then takes
    effect only once the call returns.

    Args:
        optimiser: Optax ``GradientTransformation`` for the primal and dual
            players (build one with
            ``jaddle_optimisers.create_saddle_optimiser``, or use a ready-made
            one such as ``jo.gd`` or ``jo.optimistic_gd``). It drives the steps
            whenever the adaptive step is off: always in ``"alternating"``
            mode, and in ``"extragradient"`` mode when ``adaptive_eta=None``.
            Passing one makes ``adaptive_eta="auto"`` resolve to ``None``.
            ``None`` (default) uses plain gradient descent, ``jo.gd(0.5)``.
        max_epochs: Epoch budget (default ``None`` = no limit). The solve stops
            with ``stop_reason="max_epochs"`` once it is reached.
        max_seconds: Wall-clock budget in seconds (default ``None`` = no limit).
            Measured from entry into ``solve()``, so setup and ``precompile``
            count against it. Checked between chunks of epochs, and each chunk is
            sized from the measured epoch time to fit the remaining budget; once
            the budget is spent the current point is returned with
            ``stop_reason="time_limit"``. The solve can overrun by about one
            epoch (shrink ``iterations_per_epoch`` for a tighter cutoff).
        initial_solution: Starting ``SaddleState`` (default ``None`` = zeros
            for the primal and both duals), e.g. the ``"solution"`` of a
            previous solve.
        initial_opt_state: Starting step-size / optimiser state (default
            ``None`` = fresh), e.g. the ``"opt_state"`` of a previous solve.
            Pass it together with ``initial_solution`` to resume a solve.
        iterations_per_epoch: Iterations per compiled epoch (default 1000).
            Convergence, budgets, logging and restarts are checked between
            epochs, so smaller values react faster at some per-epoch cost.
        primal_grad_norm_tolerance: Stationarity tolerance (default 1e-2); see
            the convergence test above.
        primal_feasibility_tolerance: Constraint-violation tolerance (default
            1e-3).
        complementarity_slack_tolerance: Relative complementary-slackness
            tolerance (default 1e-3).
        weight_function: Weight ``w(i)`` given to iterate ``i`` in the running
            average (default uniform, ``lambda _: 1.0``; see also
            ``jaddle_optimisers.tail_average``). Only used when
            ``average=True``.
        verbose: Print per-epoch progress (default ``False``).
        log_every: With ``verbose``, print every this many epochs (default 1).
        average: Report and test convergence on the weighted running average
            of the iterates rather than the last iterate (default ``False``);
            restarts then resume from whichever of the two has the better merit. Averaging helps non-contractive schemes such as
            ``"alternating"``; the adaptive schemes converge in the last iterate.
        update_mode: Selects the stepping scheme (single source of truth):
            ``"extragradient"`` (default; Korpelevich two-call),
            ``"alternating"`` (primal step, then dual step at the new primal) or
            ``"forward_reflected"`` (Malitsky-Tam forward-reflected-backward:
            one gradient evaluation per iteration; requires ``adaptive_eta``).
        k_scale: Primal-weight (k) scaling control. ``None`` disables it;
            otherwise a float sets a symmetric clamp band ``[1/k_scale,
            k_scale]`` for ``k`` (default ``10`` → ``[0.1, 10]``). When enabled —
            orthogonal to ``update_mode``, so it composes with every scheme
            — a primal weight ``k`` rescales the primal/dual gradients by
            ``(1/k, k)`` before each ``opt_update``, making the dual/primal step
            ratio ``k**2``. ``k`` is initialised from ``k_init`` and rebalanced
            at each restart (PDLP-style) from primal-vs-dual iterate movement; it
            is constant within an epoch (not adapted per iteration). Tuned by
            ``k_theta``/``k_scale`` and ``k_init``.
        k_theta: Smoothing coefficient for the log-space primal-weight update at
            each restart (default 0.5 = geometric mean of the movement-based
            target and the current weight, matching PDLP). Smaller = slower
            adaptation. Only used when ``k_scale`` is set.
        k_init: Initial primal weight ``k``. ``None`` (default) initialises it to
            the PDLP heuristic ``||c|| / ||b||`` (objective vs RHS norms), where
            ``c = grad(objective)(0)`` and ``b = -[c_eq(0); c_ineq(0)]``. Pass a
            float to override (``1.0`` = symmetric steps). Only used when
            ``k_scale`` is set.
        adaptive_eta: Enables a per-iteration adaptive step size with a
            Malitsky-Tam local-Lipschitz line search. ``"auto"`` (default)
            enables it with seed ``1.0`` when ``update_mode`` supports it,
            ``k_scale`` is set and no ``optimiser`` was passed (otherwise it
            behaves like ``None``; ``forward_reflected`` always enables it — the
            seed barely matters, the line search corrects it within a few
            iterations). ``None`` keeps the optimiser's fixed learning rate.
            A float seeds a single scalar
            base step ``eta`` driving the primal step ``tau = eta / k`` and dual
            step ``sigma = eta * k``. The extragradient look-ahead and corrector
            already evaluate the gradient twice, so the local Lipschitz estimate
            ``L_hat = ‖g_half - g‖_w / ‖z_half - z‖_w`` (the k-weighted norm)
            comes free; the step is admissible while ``eta · L_hat <= 1/sqrt2``,
            and ``eta`` is rejected + shrunk if the trial overshot, then advanced
            with a two-sided guard. Only supported with ``update_mode`` in
            ``('extragradient', 'forward_reflected')`` (the contractive schemes
            here) and requires ``k_scale`` (the primal weight k). The learned
            ``eta`` is carried across restarts. The optimiser's learning rate is
            bypassed in the hot loop.
        restarts: Enable adaptive warm restarts (default ``False``). There is
            no cap on how many fire; the triggers alone decide. Each restart
            resets the optimiser momentum and averaging while keeping the
            current iterate as a warm start. A restart fires when the normalised
            KKT merit drops below ``restart_decay`` × the merit at the last
            restart (sufficient-progress restart) or the cycle-length cap is
            exhausted (no-progress restart).
        epochs_per_restart: Length cap (epochs) of the first restart cycle, or
            ``None`` (default) for no cap — restarts then fire only on the
            sufficient-progress trigger. Subsequent caps grow by
            ``restart_multiplier``.
        restart_multiplier: Geometric growth factor for cycle-length caps
            (default 1.0 = fixed length, 2.0 = doubling).
        restart_decay: Sufficient-progress threshold (default 0.2). A restart
            fires early when the merit drops below this fraction of the merit at
            the last restart.
        iterations_per_epoch_decay: Multiplicative decay applied to
            ``iterations_per_epoch`` after each restart (default 1.0 = no
            decay). Values < 1 shrink the epoch length at each restart to spend
            more time checking convergence.
        iterations_per_epoch_min: Floor for the decayed epoch length (default
            100). Only used when ``iterations_per_epoch_decay < 1``.
        precompile: Compile the device loop before the timed loop starts
            (default ``True``), so ``"solve_seconds"`` measures iteration time
            rather than XLA compilation. The compile still counts against
            ``max_seconds``. A restart that changes ``iterations_per_epoch``
            compiles a new loop inside the timed region.

    Returns:
        dict: The solution together with diagnostics. Keys:
            * ``"solution"``: the ``SaddleState`` (primal/dual iterate).
            * ``"converged"``: ``bool``, whether the solve met the convergence
              criteria (``False`` if the epoch / time budget was exhausted or the
              solve was interrupted).
            * ``"stop_reason"``: ``str``, why the solve terminated:
              ``"converged"``, ``"max_epochs"``, ``"time_limit"`` (``max_seconds``
              exhausted) or ``"interrupted"`` (KeyboardInterrupt).
            * ``"opt_state"``: the final optimiser state, for warm-starting a
              subsequent solve via ``initial_opt_state``.
            * ``"solve_seconds"``: ``float`` wall time of the epoch loop,
              excluding setup. With ``precompile=True`` (default) it also
              excludes XLA compilation; with ``precompile=False`` it includes
              the first-epoch compile.
            * ``"epochs"``: ``int``, number of epochs run.
    """

    # max_seconds is a wall-clock budget for the whole call, setup included.
    solve_entry_time = time.time()
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be > 0 (or None for no limit)")

    if log_every < 1:
        raise ValueError("log_every must be >= 1")

    if verbose:
        print("----------------------------------------------")

    optimiser, adaptive_eta, adaptive_step, k_scaling, k_lo, k_hi = _check_settings(
        optimiser, adaptive_eta, update_mode, k_scale
    )

    if verbose:
        print("====Starting Solve====")
        print("----------------------------------------------")

    _build_chunk, carry = _device_solver(
        cp,
        optimiser=optimiser,
        initial_solution=initial_solution,
        initial_opt_state=initial_opt_state,
        iterations_per_epoch=iterations_per_epoch,
        primal_grad_norm_tolerance=primal_grad_norm_tolerance,
        primal_feasibility_tolerance=primal_feasibility_tolerance,
        complementarity_slack_tolerance=complementarity_slack_tolerance,
        weight_function=weight_function,
        verbose=verbose,
        log_every=log_every,
        average=average,
        update_mode=update_mode,
        k_scaling=k_scaling,
        k_lo=k_lo,
        k_hi=k_hi,
        k_theta=k_theta,
        k_init=k_init,
        adaptive_eta=adaptive_eta,
        adaptive_step=adaptive_step,
        restarts=restarts,
        epochs_per_restart=epochs_per_restart,
        restart_multiplier=restart_multiplier,
        restart_decay=restart_decay,
        iterations_per_epoch_decay=iterations_per_epoch_decay,
        iterations_per_epoch_min=iterations_per_epoch_min,
    )

    def _epochs_arg(n):
        # Same type on every call, so the (possibly AOT-compiled) chunk accepts it.
        return jnp.asarray(n, carry["count"].dtype)

    chunk_fns = {}
    current_ipe = int(iterations_per_epoch)
    if precompile:
        # Compile the first chunk function ahead of time, so the first epoch
        # doesn't pay its compile inside the timed measurement.
        chunk_fns[current_ipe] = (
            _build_chunk(current_ipe).lower(carry, _epochs_arg(1)).compile()
        )

    # With no logging and no budgets there is nothing for the host to do between
    # epochs, so the whole solve is one call: the loop only returns once the
    # convergence test passes (or a restart changes iterations_per_epoch, whose
    # new scan length needs a new compile).
    single_call = not verbose and not max_epochs and max_seconds is None
    chunk_epochs = _UNBOUNDED_EPOCHS if single_call else 1
    seconds_per_epoch = None
    count = 0
    done = False
    restarts_printed = 0
    is_converged = True
    stop_reason = "converged"

    def print_chunk_log(host, epoch_time):
        nonlocal restarts_printed
        entries = [(int(r[0]), 0, r) for r in host["log"][: int(host["n_log"])]]
        entries += [(int(r[0]), 1, r) for r in host["evt"][: int(host["n_evt"])]]
        for epoch_no, kind, r in sorted(entries, key=lambda e: e[:2]):
            if kind == 0:
                print(
                    f"|Epoch {epoch_no}|"
                    f"|Obj{float(r[1]):.2e}|"
                    f"|PGN {float(r[2]):.2e}|"
                    f"|CS {float(r[3]):.2e}|"
                    f"|PFR {float(r[4]):.2e}|"
                    f"|Time {epoch_time:.2f}s|"
                )
                print("----------------------------------------------")
                continue
            restarts_printed += 1
            reason = "sufficient-progress" if r[1] == 0 else "cycle-cap"
            which = "avg" if r[3] else "iterate"
            k_msg = f", k={float(r[4]):.3e}" if k_scaling else ""
            print(
                f"Restart {restarts_printed} at epoch {epoch_no} "
                f"({reason}, merit={float(r[2]):.2e} "
                f"[{which}], next cap={float(r[5]):.0f} epochs, "
                f"iters/epoch={int(r[6])}{k_msg})"
            )
            print("----------------------------------------------")

    start_time = time.time()

    try:
        while True:
            if done:
                break
            if max_epochs and count >= max_epochs:
                is_converged = False
                stop_reason = "max_epochs"
                print(f"Reached maximum epochs: {max_epochs}. Stopping.")
                print("----------------------------------------------")
                break
            if max_seconds is not None:
                if time.time() - solve_entry_time >= max_seconds:
                    is_converged = False
                    stop_reason = "time_limit"
                    print(f"Reached time limit: {max_seconds}s. Stopping.")
                    print("----------------------------------------------")
                    break

            n_epochs = chunk_epochs
            if max_epochs:
                n_epochs = min(n_epochs, max_epochs - count)
            if max_seconds is not None and seconds_per_epoch:
                remaining = max_seconds - (time.time() - solve_entry_time)
                n_epochs = min(n_epochs, max(1, int(remaining / seconds_per_epoch)))

            run_chunk = chunk_fns.get(current_ipe)
            if run_chunk is None:
                run_chunk = chunk_fns[current_ipe] = _build_chunk(current_ipe)

            chunk_start = time.time()
            new_carry = run_chunk(carry, _epochs_arg(n_epochs))
            keys = ["count", "done", "ipe"]
            if verbose:
                keys += ["log", "n_log", "evt", "n_evt"]
            # The one host/device synchronisation per chunk.
            host = jax.device_get({k: new_carry[k] for k in keys})
            carry = new_carry
            chunk_seconds = time.time() - chunk_start

            epochs_run = max(int(host["count"]) - count, 1)
            count = int(host["count"])
            done = bool(host["done"])
            current_ipe = int(host["ipe"])
            if single_call:
                continue
            seconds_per_epoch = chunk_seconds / epochs_run

            if verbose:
                print_chunk_log(host, seconds_per_epoch)
                zero = jnp.zeros_like(carry["n_log"])
                carry = {**carry, "n_log": zero, "n_evt": zero}

            # Size the next chunk to take about _CHUNK_TARGET_SECONDS, growing
            # by at most 8x per chunk.
            chunk_epochs = int(
                min(
                    max(1.0, _CHUNK_TARGET_SECONDS / seconds_per_epoch),
                    8 * epochs_run,
                )
            )
    except KeyboardInterrupt:
        # `carry` still holds the last completed chunk (it is not donated).
        is_converged = False
        stop_reason = "interrupted"
        count = int(carry["count"])
        print("KeyboardInterrupt received. Returning current solution.")
        print("----------------------------------------------")

    opt_state = carry["opt"]
    output = carry["avg"] if average else carry["state"]

    output = jax.block_until_ready(output)
    end_time = time.time()
    print(f"Time to solution: {end_time - start_time:.2f} seconds")
    print("----------------------------------------------")
    print(f"Epochs to solution: {count}")
    print("----------------------------------------------")
    print(f"Objective: {cp.objective(output.primal):.5e}")
    print("----------------------------------------------")

    return {
        "solution": output,
        "converged": is_converged,
        "stop_reason": stop_reason,
        "opt_state": opt_state,
        "solve_seconds": end_time - start_time,
        "epochs": count,
    }


# %%


class SolveCoreResult(NamedTuple):
    """What a ``make_solver`` function returns; every field is a JAX array, so
    results batch under ``vmap``. ``solution`` is the reported point (the
    average when ``average=True``, else the last iterate), ``converged``
    whether the convergence test passed, and ``epochs`` the epochs run."""

    solution: SaddleState
    converged: jnp.ndarray
    epochs: jnp.ndarray


def make_solver(
    build_cp,
    max_epochs=None,
    optimiser=None,
    iterations_per_epoch=int(1e3),
    primal_grad_norm_tolerance=1e-2,
    primal_feasibility_tolerance=1e-3,
    complementarity_slack_tolerance=1e-3,
    weight_function=lambda _: 1.0,
    average=False,
    update_mode="extragradient",
    k_scale=10.0,
    k_theta=0.5,
    k_init=None,
    adaptive_eta="auto",
    restarts=False,
    epochs_per_restart=None,
    restart_multiplier=1.0,
    restart_decay=0.2,
    iterations_per_epoch_min=100,
):
    """
    Build a pure-JAX solve function for a family of convex problems.

    ``build_cp(params)`` returns an ordinary ``JaddleCP`` whose functions
    close over ``params`` (any pytree: data, weights, right-hand sides,
    bounds). ``make_solver`` returns ``solve_fn(params, initial_solution=None)
    -> SolveCoreResult``. The whole solve is traceable, so ``solve_fn`` can be
    ``jax.jit``-ed and ``jax.vmap``-ed over a batch of ``params`` whose
    problems share their shapes::

        def build_cp(a):
            return JaddleCP(n, objective=lambda x: jnp.sum((x - a) ** 2), ...)

        solve_fn = jax.jit(make_solver(build_cp, max_epochs=300))
        result = solve_fn(a)
        batch = jax.vmap(solve_fn)(a_batch)

    The algorithm is ``solve()``'s, with the same arguments and defaults.
    Called directly, a solve reproduces ``solve()`` exactly; under an outer
    ``jax.jit`` XLA fuses the whole solve into one program and may round
    differently (measured: 1e-16 to 1e-14, at most 1e-6 after the adaptive
    step amplified it, same epoch counts). What needs the host is not
    available: no ``verbose`` log, no ``max_seconds``, and the epoch length
    never changes during a solve (no ``iterations_per_epoch_decay``;
    ``iterations_per_epoch_min`` is capped at ``iterations_per_epoch``).
    ``max_epochs=None`` runs until the convergence test passes, so set a budget
    when that may not happen. Batched members iterate in lockstep until the
    last one stops (finished members are frozen).
    """
    optimiser, adaptive_eta, adaptive_step, k_scaling, k_lo, k_hi = _check_settings(
        optimiser, adaptive_eta, update_mode, k_scale
    )
    n_epochs = _UNBOUNDED_EPOCHS if not max_epochs else int(max_epochs)

    def solve_fn(params, initial_solution=None):
        cp = build_cp(params)
        build_chunk, carry = _device_solver(
            cp,
            optimiser=optimiser,
            initial_solution=initial_solution,
            initial_opt_state=None,
            iterations_per_epoch=iterations_per_epoch,
            primal_grad_norm_tolerance=primal_grad_norm_tolerance,
            primal_feasibility_tolerance=primal_feasibility_tolerance,
            complementarity_slack_tolerance=complementarity_slack_tolerance,
            weight_function=weight_function,
            verbose=False,
            log_every=1,
            average=average,
            update_mode=update_mode,
            k_scaling=k_scaling,
            k_lo=k_lo,
            k_hi=k_hi,
            k_theta=k_theta,
            k_init=k_init,
            adaptive_eta=adaptive_eta,
            adaptive_step=adaptive_step,
            restarts=restarts,
            epochs_per_restart=epochs_per_restart,
            restart_multiplier=restart_multiplier,
            restart_decay=restart_decay,
            iterations_per_epoch_decay=1.0,
            iterations_per_epoch_min=min(iterations_per_epoch_min, iterations_per_epoch),
        )
        carry = build_chunk(iterations_per_epoch)(
            carry, jnp.asarray(n_epochs, carry["count"].dtype)
        )
        return SolveCoreResult(
            solution=carry["avg"] if average else carry["state"],
            converged=carry["done"],
            epochs=carry["count"],
        )

    return solve_fn


def make_optimal_value(build_cp, return_solution=False, **settings):
    """
    The optimal value ``z*(params) = min f(x; params)`` of a family of convex
    problems, as a differentiable function of ``params``.

    ``build_cp`` is as in ``make_solver`` (``settings`` are its options).
    Returns ``value_fn(params) -> z*``. Its gradient comes from the envelope
    theorem, with no differentiation through the iterations: with the
    solution ``(x*, λ*, μ*)`` held fixed,

        ∂z*/∂params = ∂/∂params [ f(x*) + λ*ᵀ g(x*) + μ*ᵀ h(x*)
                                  + ν_lᵀ lower − ν_uᵀ upper ],

    differentiated through ``build_cp``, so data, weights, right-hand sides
    and bounds can all depend on ``params``. The bound multipliers come from
    the stationarity residual ``r = ∇ₓL`` at the solution: ``ν_l = max(r, 0)``
    on finite lower bounds and ``ν_u = max(−r, 0)`` on finite upper bounds.

    Exact where the optimal multipliers are unique; elsewhere ``z*`` has a
    kink and this is the subgradient at the solution found. Only as accurate
    as the solve: use tight tolerances. Works under ``jit`` and ``vmap``.

    ``return_solution=True`` makes ``value_fn`` return ``(z*,
    SolveCoreResult)`` from the same solve, for ``jax.grad(value_fn,
    has_aux=True)``: each call is a fresh solve, and separate solves differ
    slightly.
    """
    solve_fn = make_solver(build_cp, **settings)

    def lagrangian(params, solution, nu_lower, nu_upper):
        cp = build_cp(params)
        x = solution.primal
        lower, upper = jnp.asarray(cp.lower_bounds), jnp.asarray(cp.upper_bounds)
        return (
            cp.objective(x)
            + solution.dual_ineq @ cp.constraints_ineq(x)
            + solution.dual_eq @ cp.constraints_eq(x)
            + nu_lower @ jnp.where(jnp.isfinite(lower), lower, 0.0)
            - nu_upper @ jnp.where(jnp.isfinite(upper), upper, 0.0)
        )

    def value_at_forward(params, solution):
        return build_cp(params).objective(solution.primal), (params, solution)

    def value_at_backward(residuals, g):
        params, solution = residuals
        cp = build_cp(params)
        # Bound multipliers from the stationarity residual r = ∇ₓL.
        r = jax.grad(
            lambda x: cp.objective(x)
            + solution.dual_ineq @ cp.constraints_ineq(x)
            + solution.dual_eq @ cp.constraints_eq(x)
        )(solution.primal)
        lower, upper = jnp.asarray(cp.lower_bounds), jnp.asarray(cp.upper_bounds)
        nu_lower = jnp.where(jnp.isfinite(lower), jnp.maximum(r, 0.0), 0.0)
        nu_upper = jnp.where(jnp.isfinite(upper), jnp.maximum(-r, 0.0), 0.0)
        _, vjp = jax.vjp(lambda p: lagrangian(p, solution, nu_lower, nu_upper), params)
        (grad_params,) = vjp(g)
        # The solution is treated as fixed: no gradient flows into the solve.
        return grad_params, jax.tree.map(jnp.zeros_like, solution)

    @jax.custom_vjp
    def value_at(params, solution):
        return value_at_forward(params, solution)[0]

    value_at.defvjp(value_at_forward, value_at_backward)

    def value_fn(params):
        result = solve_fn(jax.tree.map(jax.lax.stop_gradient, params))
        solution = jax.tree.map(jax.lax.stop_gradient, result.solution)
        z = value_at(params, solution)
        return (z, result) if return_solution else z

    return value_fn


def make_solution(build_cp, active_tol=1e-6, gmres_tol=1e-10, **settings):
    """
    The solution ``x*(params)`` of a family of convex problems, differentiable
    by implicit differentiation of the KKT conditions on the active set.

    ``build_cp`` and ``settings`` are as in ``make_solver``. Returns
    ``x_fn(params) -> x*``. At the solution the active set is frozen:
    variables within ``active_tol`` (relative) of a finite bound, and
    inequality rows with ``g(x*) >= -active_tol``. The KKT conditions then form
    a square system ``F(x, μ, λ; params) = 0``: stationarity ``∇ₓL = 0`` on
    the free variables, ``x = bound(params)`` on the bound ones, ``h(x) = 0``,
    ``g(x) = 0`` on the active rows and ``λ = 0`` on the inactive ones. A VJP
    solves ``(∂F/∂z)ᵀ w = (g, 0, 0)`` matrix-free with GMRES (using
    ``jax.vjp``, so all on the device and fine under ``vmap``) and returns
    ``−(∂F/∂params)ᵀ w``.

    This needs a nondegenerate solution: strict complementarity, independent
    active constraints, and a Hessian of the Lagrangian positive definite on
    their null space (e.g. a strictly convex objective). Where GMRES cannot
    solve the system to ``gmres_tol`` (relative residual), the gradient is
    returned as NaN rather than silently wrong: a jitted function cannot raise.
    Solve to tolerances well below ``active_tol``.
    """
    from jax.scipy.sparse.linalg import gmres

    solve_fn = make_solver(build_cp, **settings)

    def kkt(z, params, masks):
        x, mu, lam = z
        free, at_lower, active = masks
        cp = build_cp(params)
        lower, upper = jnp.asarray(cp.lower_bounds), jnp.asarray(cp.upper_bounds)

        def lag(x):
            return (
                cp.objective(x)
                + lam @ cp.constraints_ineq(x)
                + mu @ cp.constraints_eq(x)
            )

        bound = jnp.where(at_lower, lower, upper)
        bound = jnp.where(jnp.isfinite(bound), bound, 0.0)
        return (
            jnp.where(free, jax.grad(lag)(x), x - bound),
            cp.constraints_eq(x),
            jnp.where(active, cp.constraints_ineq(x), lam),
        )

    def forward(params):
        solution = solve_fn(jax.tree.map(jax.lax.stop_gradient, params)).solution
        return solution.primal, (params, solution)

    def backward(residuals, g):
        params, solution = residuals
        cp = build_cp(params)
        x = solution.primal
        lower, upper = jnp.asarray(cp.lower_bounds), jnp.asarray(cp.upper_bounds)
        at_lower = jnp.isfinite(lower) & (
            jnp.abs(x - lower) <= active_tol * (1 + jnp.abs(lower))
        )
        at_upper = (
            jnp.isfinite(upper)
            & (jnp.abs(x - upper) <= active_tol * (1 + jnp.abs(upper)))
            & ~at_lower
        )
        g_ineq = cp.constraints_ineq(x)
        active = g_ineq >= -active_tol * (1 + jnp.abs(g_ineq))
        masks = (~(at_lower | at_upper), at_lower, active)
        z = (x, solution.dual_eq, solution.dual_ineq)

        _, vjp_z = jax.vjp(lambda zz: kkt(zz, params, masks), z)
        rhs = (g, jnp.zeros_like(solution.dual_eq), jnp.zeros_like(solution.dual_ineq))

        def transpose_jacobian(w):
            return vjp_z(w)[0]

        w, _ = gmres(transpose_jacobian, rhs, tol=gmres_tol, atol=0.0, restart=50, maxiter=50)
        residual = jax.tree.map(lambda a, b: a - b, transpose_jacobian(w), rhs)
        flat_norm = lambda t: jnp.sqrt(sum(jnp.vdot(a, a) for a in jax.tree.leaves(t)))
        ok = flat_norm(residual) <= 1e3 * gmres_tol * (1 + flat_norm(rhs))

        _, vjp_params = jax.vjp(lambda p: kkt(z, p, masks), params)
        (grad_params,) = vjp_params(w)
        return (
            jax.tree.map(lambda t: jnp.where(ok, -t, jnp.nan), grad_params),
        )

    @jax.custom_vjp
    def solution_map(params):
        return forward(params)[0]

    solution_map.defvjp(forward, backward)
    return solution_map


# %%
def _equilibrate_kkt(Q, A, ruiz_iterations, pc_iterations):
    """Variable scales ``d`` and constraint scales ``e`` that equilibrate the
    KKT matrix ``[[Q, Aᵀ], [A, 0]]`` symmetrically: Ruiz sweeps (each row's
    largest entry to 1), then Pock-Chambolle sweeps (each row's absolute sum
    to 1). One scale per row of the KKT matrix keeps ``D Q D`` symmetric, so
    the scaling is a change of variables ``x = D x̃``. An empty row keeps
    scale 1."""
    import scipy.sparse as sp

    n = Q.shape[0]
    kkt = abs(sp.bmat([[Q, A.T], [A, None]], format="csr"))
    scale = np.ones(kkt.shape[0])

    def sweep(reduce):
        norms = scale * np.asarray(reduce(kkt @ sp.diags(scale))).ravel()
        return scale / np.sqrt(np.where(norms > 0, norms, 1.0))

    for _ in range(ruiz_iterations):
        scale = sweep(lambda m: m.max(axis=1).toarray())
    for _ in range(pc_iterations):
        scale = sweep(lambda m: m.sum(axis=1))
    return scale[:n], scale[n:]


def _qp_certificate(Q, c, A_eq, b_eq, A_ineq, b_ineq, lower, upper, solution):
    """Relative KKT residuals and duality gap of a primal-dual pair for
    ``min ½xᵀQx + cᵀx``, in the problem's own units (l2 norms)."""
    x = np.asarray(solution.primal, dtype=np.float64)
    y_eq = np.asarray(solution.dual_eq, dtype=np.float64)
    y_ineq = np.asarray(solution.dual_ineq, dtype=np.float64)
    Qx = Q @ x
    reduced_cost = Qx + c + A_eq.T @ y_eq + A_ineq.T @ y_ineq
    # A reduced cost is paid for by the bound it pushes against, if finite.
    at_lower = (reduced_cost > 0) & np.isfinite(lower)
    at_upper = (reduced_cost < 0) & np.isfinite(upper)
    bound_term = np.sum(
        reduced_cost
        * np.where(at_lower, lower, np.where(at_upper, upper, 0.0)),
    )
    primal_residual = np.linalg.norm(
        np.concatenate(
            [
                A_eq @ x - b_eq,
                np.maximum(A_ineq @ x - b_ineq, 0.0),
                np.maximum(lower - x, 0.0),
                np.maximum(x - upper, 0.0),
            ]
        )
    )
    dual_residual = np.linalg.norm(
        np.concatenate(
            [np.where(at_lower | at_upper, 0.0, reduced_cost), np.minimum(y_ineq, 0.0)]
        )
    )
    objective = 0.5 * x @ Qx + c @ x
    # Lagrangian lower bound, with the quadratic linearised at x.
    dual_objective = -0.5 * x @ Qx - b_eq @ y_eq - b_ineq @ y_ineq + bound_term
    gap = objective - dual_objective
    b_norm = np.linalg.norm(np.concatenate([b_eq, b_ineq]))
    c_norm = max(np.linalg.norm(c), np.linalg.norm(Qx))
    return {
        "objective": float(objective),
        "dual_objective": float(dual_objective),
        "duality_gap": float(gap),
        "relative_gap": float(abs(gap) / (1 + abs(objective) + abs(dual_objective))),
        "primal_feasibility_residual": float(primal_residual),
        "dual_feasibility_residual": float(dual_residual),
        "relative_primal_feasibility_residual": float(primal_residual / (1 + b_norm)),
        "relative_dual_feasibility_residual": float(dual_residual / (1 + c_norm)),
    }


def quadratic_program(
    Q,
    c,
    A_eq=None,
    b_eq=None,
    A_ineq=None,
    b_ineq=None,
    lower_bounds=None,
    upper_bounds=None,
    primal_feasibility_tolerance=1e-3,
    dual_feasibility_tolerance=1e-3,
    dual_gap_tolerance=1e-3,
    ruiz_iterations=10,
    pc_iterations=1,
    initial_solution=None,
    **solve_options,
):
    """
    Solve the convex quadratic program

        minimise    ½ xᵀQx + cᵀx
        subject to  A_eq   x  = b_eq
                    A_ineq x <= b_ineq
                    lower_bounds <= x <= upper_bounds

    with ``solve()``, after scaling it from its matrices.

    ``Q`` is symmetric positive semidefinite; it and the constraint matrices
    are NumPy arrays or SciPy sparse matrices. A constraint block or a bound
    left as ``None`` is absent. Other keyword arguments go to ``solve()``.

    ``solve()`` sees a problem only through its functions, so it cannot scale
    one; here the entries are known. The KKT matrix ``[[Q, Aᵀ], [A, 0]]`` is
    equilibrated symmetrically with ``ruiz_iterations`` Ruiz sweeps and
    ``pc_iterations`` Pock-Chambolle sweeps (0 and 0 solve the problem as
    given), the scaled problem is solved, and the solution is mapped back.

    Termination is the linear solver's, tested in the problem's own units
    whatever the scaling, with l2 norms: primal feasibility relative to
    ``1 + ‖b‖``, dual feasibility (reduced costs ``Qx + c + Aᵀy`` that no
    finite bound absorbs) relative to ``1 + max(‖c‖, ‖Qx‖)``, and the duality
    gap relative to ``1 + |primal| + |dual|``, where the dual objective is the
    Lagrangian bound with the quadratic linearised at ``x``.

    Returns ``solve()``'s dict, with these keys replaced or added:

    * ``"solution"``: a ``SaddleState`` in the problem's own units. The
      Lagrangian is ``½xᵀQx + cᵀx + y_eqᵀ(A_eq x − b_eq) + y_ineqᵀ(A_ineq x −
      b_ineq)`` with ``y_ineq >= 0``.
    * ``"objective"``: the objective there.
    * ``"certificate"``: the three residuals above, recomputed from the
      unscaled matrices (the keys of
      ``jaddle_linear.evaluate_lp_certificate``).

    ``initial_solution`` is a ``SaddleState`` in the problem's own units;
    ``initial_opt_state`` is passed to ``solve()`` unchanged.
    """
    import scipy.sparse as sp
    from jaddle.jaddle_basic_types import scipy_to_bcoo

    c = np.asarray(c, dtype=np.float64)
    n = c.shape[0]
    Q = sp.csr_matrix(Q, dtype=np.float64)
    if Q.shape != (n, n):
        raise ValueError(f"Q has shape {Q.shape}, expected {(n, n)}")
    if abs(Q - Q.T).max() > 1e-12 * max(abs(Q).max(), 1.0):
        raise ValueError("Q must be symmetric")

    def block(A, b):
        if A is None:
            return sp.csr_matrix((0, n)), np.zeros(0)
        return sp.csr_matrix(A, dtype=np.float64), np.asarray(b, dtype=np.float64)

    A_eq, b_eq = block(A_eq, b_eq)
    A_ineq, b_ineq = block(A_ineq, b_ineq)
    m_eq = A_eq.shape[0]
    lower = (
        np.full(n, -np.inf) if lower_bounds is None else np.asarray(lower_bounds, float)
    )
    upper = (
        np.full(n, np.inf) if upper_bounds is None else np.asarray(upper_bounds, float)
    )

    d, e = _equilibrate_kkt(
        Q, sp.vstack([A_eq, A_ineq]), ruiz_iterations, pc_iterations
    )
    D = sp.diags(d)
    dtype = jo.jaddle_dtype()
    A_eq_scipy = sp.diags(e[:m_eq]) @ A_eq @ D
    A_ineq_scipy = sp.diags(e[m_eq:]) @ A_ineq @ D
    Q_s = scipy_to_bcoo(D @ Q @ D, dtype)
    A_eq_s, A_eq_T = (scipy_to_bcoo(m, dtype) for m in (A_eq_scipy, A_eq_scipy.T))
    A_ineq_s, A_ineq_T = (
        scipy_to_bcoo(m, dtype) for m in (A_ineq_scipy, A_ineq_scipy.T)
    )
    c_s = jnp.asarray(d * c, dtype=dtype)
    b_eq_s = jnp.asarray(e[:m_eq] * b_eq, dtype=dtype)
    b_ineq_s = jnp.asarray(e[m_eq:] * b_ineq, dtype=dtype)
    lower_s = jnp.asarray(lower / d, dtype=dtype)
    upper_s = jnp.asarray(upper / d, dtype=dtype)
    d_j = jnp.asarray(d, dtype=dtype)
    e_eq, e_ineq = jnp.asarray(e[:m_eq], dtype=dtype), jnp.asarray(e[m_eq:], dtype=dtype)
    b_norm = float(np.linalg.norm(np.concatenate([b_eq, b_ineq])))
    c_norm = float(np.linalg.norm(c))

    def objective(x):
        # Q is symmetric, so the gradient is Qx + c: one product with Q,
        # where differentiating xᵀQx would take a second with Qᵀ.
        Qx = jax.lax.stop_gradient(Q_s @ x)
        return x @ Qx - 0.5 * jax.lax.stop_gradient(x) @ Qx + c_s @ x

    def residuals(state):
        # _qp_certificate on the scaled problem: a residual of the original
        # is the scaled one over its row's or column's scale, and the
        # objectives and bound terms are the same numbers in either.
        x = state.primal
        Qx = Q_s @ x
        reduced_cost = Qx + c_s + A_eq_T @ state.dual_eq + A_ineq_T @ state.dual_ineq
        at_lower = (reduced_cost > 0) & jnp.isfinite(lower_s)
        at_upper = (reduced_cost < 0) & jnp.isfinite(upper_s)
        bound_term = jnp.sum(
            reduced_cost
            * jnp.where(at_lower, lower_s, jnp.where(at_upper, upper_s, 0.0))
        )
        dual_residual = jnp.linalg.norm(
            jnp.where(at_lower | at_upper, 0.0, reduced_cost) / d_j
        )
        primal_residual = jnp.linalg.norm(
            jnp.concatenate(
                [
                    (A_eq_s @ x - b_eq_s) / e_eq,
                    jnp.maximum(A_ineq_s @ x - b_ineq_s, 0.0) / e_ineq,
                ]
            )
        )
        primal_objective = 0.5 * x @ Qx + c_s @ x
        dual_objective = (
            -0.5 * x @ Qx
            - b_eq_s @ state.dual_eq
            - b_ineq_s @ state.dual_ineq
            + bound_term
        )
        return (
            dual_residual / (1 + jnp.maximum(c_norm, jnp.linalg.norm(Qx / d_j))),
            primal_residual / (1 + b_norm),
            jnp.abs(primal_objective - dual_objective)
            / (1 + jnp.abs(primal_objective) + jnp.abs(dual_objective)),
        )

    cp = JaddleCP(
        num_variables=n,
        objective=objective,
        constraints_eq=lambda x: A_eq_s @ x - b_eq_s,
        constraints_ineq=lambda x: A_ineq_s @ x - b_ineq_s,
        lower_bounds=lower_s,
        upper_bounds=upper_s,
        residuals=residuals,
    )
    if initial_solution is not None:
        initial_solution = SaddleState(
            primal=jnp.asarray(np.asarray(initial_solution.primal) / d, dtype=dtype),
            dual_ineq=jnp.asarray(
                np.asarray(initial_solution.dual_ineq) / e[m_eq:], dtype=dtype
            ),
            dual_eq=jnp.asarray(
                np.asarray(initial_solution.dual_eq) / e[:m_eq], dtype=dtype
            ),
        )
    result = solve(
        cp,
        initial_solution=initial_solution,
        primal_grad_norm_tolerance=dual_feasibility_tolerance,
        primal_feasibility_tolerance=primal_feasibility_tolerance,
        complementarity_slack_tolerance=dual_gap_tolerance,
        **solve_options,
    )
    scaled = result["solution"]
    solution = SaddleState(
        primal=jnp.asarray(d * np.asarray(scaled.primal, dtype=np.float64)),
        dual_ineq=jnp.asarray(e[m_eq:] * np.asarray(scaled.dual_ineq, dtype=np.float64)),
        dual_eq=jnp.asarray(e[:m_eq] * np.asarray(scaled.dual_eq, dtype=np.float64)),
    )
    certificate = _qp_certificate(
        Q, c, A_eq, b_eq, A_ineq, b_ineq, lower, upper, solution
    )
    result.update(
        solution=solution, objective=certificate["objective"], certificate=certificate
    )
    return result
