# %%
import jax
import jax.numpy as jnp
from optax.projections import projection_non_negative, projection_box
import optax
import functools
from typing import NamedTuple
import time
from jaddle.jaddle_basic_types import JaddleCP, SaddleState
import jaddle.jaddle_optimisers as jo
import numpy as np

_CONVEX_RUN_EPOCH_CACHE = {}


def __sps(
    max_iter,
    start_iter,
    cp: JaddleCP,
    optimiser,
    initial_solution,
    initial_avg_state=None,
    initial_opt_state=None,
    weight_function=lambda _: 1.0,
    total_weight=0.0,
    average=True,
    update_mode="synchronous",
    k_scaling=False,
    k_init=1.0,
    adaptive_eta=None,
):
    # The stepping scheme is selected by `update_mode`. This derived boolean
    # keeps the per-scheme branching below readable while the string stays the
    # single source of truth.
    extragradient = update_mode == "extragradient"
    # Per-iteration adaptive step size (Malitsky-Tam local-Lipschitz line
    # search). When `adaptive_eta` is not None the extragradient scheme replaces
    # the optimiser's fixed learning rate with a single scalar base step eta,
    # line-searched every iteration: the look-ahead and corrector already
    # evaluate grad twice, so the local Lipschitz estimate
    #     L_hat = ‖g_half - g‖_w / ‖z_half - z‖_w
    # (w = the k-weighted norm) comes free, and the step is admissible while
    # eta · L_hat <= 1/sqrt(2). The primal/dual steps are tau=eta/k, sigma=eta*k.
    # eta is packed alongside k in the k-slot of opt_state. Requires k_scaling
    # (it needs k) and only extragradient (the contractive scheme here);
    # synchronous/alternating are not contractive on the saddle and a line
    # search cannot fix that, so they are excluded.
    adaptive_step = adaptive_eta is not None and update_mode == "extragradient"
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

    cache_key = (
        id(cp),
        id(optimiser),
        id(weight_function),
        bool(average),
        update_mode,
        bool(k_scaling),
        adaptive_step,
    )
    run_epoch = _CONVEX_RUN_EPOCH_CACHE.get(cache_key)

    if run_epoch is None:

        @functools.partial(
            jax.jit,
            static_argnames=("max_iter",),
            # `average_state` is deliberately not donated: when averaging is off
            # the caller passes the same buffer for both `state` and
            # `average_state`, and XLA rejects donating one buffer twice.
            donate_argnames=(
                "state",
                "opt_state",
                "total_weight",
            ),
        )
        def run_epoch(
            max_iter,
            start_iter,
            state,
            average_state,
            opt_state,
            total_weight=0.0,
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

                def _trial(eta, state, k):
                    tau = eta / k
                    sigma = eta * k
                    g = grad(state)
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

                    # Local Lipschitz estimate in the k-weighted norm, measured
                    # on the look-ahead displacement.
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
                    eta_bar = jnp.where(dg2 > 0.0, _MT * jnp.sqrt(dz2 / dg2), jnp.inf)
                    return cand, eta_bar

                def step(carry, _):
                    i, state, average_state, opt_state, total_weight = carry
                    opt_state, k, eta = unpack_k(opt_state)
                    ip1 = jnp.asarray(i + 1, eta.dtype)

                    # Retry: while the trial step exceeds its admissible bound,
                    # shrink eta to just under eta_bar and re-trial. The
                    # (1-(i+1)^-0.3) factor < 1 guarantees strict decrease, so
                    # the loop terminates. Carry (eta, cand, eta_bar).
                    def cond(c):
                        eta_c, _, eta_bar_c = c
                        return eta_c > eta_bar_c

                    def body(c):
                        eta_c, _, eta_bar_c = c
                        eta_s = jnp.minimum((1.0 - ip1 ** (-0.3)) * eta_bar_c, eta_c)
                        cand_s, eta_bar_s = _trial(eta_s, state, k)
                        return (eta_s, cand_s, eta_bar_s)

                    cand0, eta_bar0 = _trial(eta, state, k)
                    eta0, cand, eta_bar = jax.lax.while_loop(
                        cond, body, (eta, cand0, eta_bar0)
                    )
                    new_state = cand

                    # Advance eta for the next iterate (growth allowed once the
                    # step is accepted). When eta_bar is +inf the step did not
                    # move: hold eta rather than letting growth run to NaN.
                    eta_next = jnp.minimum(
                        (1.0 - ip1 ** (-0.3)) * eta_bar,
                        (1.0 + ip1 ** (-0.6)) * eta0,
                    )
                    eta_next = jnp.where(jnp.isfinite(eta_bar), eta_next, eta0)
                    eta_next = jnp.where(jnp.isfinite(eta_next), eta_next, eta0)
                    eta_next = jnp.maximum(eta_next, 1e-12)
                    opt_state = pack_k(opt_state, k, eta_next)

                    if average:
                        w = weight_function(i)
                        total_weight = total_weight + w
                        average_state = optax.incremental_update(
                            new_state, average_state, w / total_weight
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
                elif extragradient:
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
                else:
                    gradient = grad(state)
                    if k_scaling:
                        gradient = scale_by_k(gradient, k)
                    updates, opt_state = opt_update(gradient, opt_state, state)
                    state = optax.apply_updates(state, updates)
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

        _CONVEX_RUN_EPOCH_CACHE[cache_key] = run_epoch

    state = initial_solution

    if initial_avg_state is not None:
        average_state = initial_avg_state
    else:
        average_state = initial_solution

    # `state` is donated to run_epoch; if `average_state` aliases the same
    # buffer (the common `average=False` case, where the caller passes one
    # buffer for both) XLA rejects the call (`f(donate(a), a)`). Give
    # `average_state` its own buffer so donation of `state` is safe. The copy is
    # one-per-epoch, off the hot path.
    if average_state is state:
        average_state = jax.tree.map(lambda x: x + 0, average_state)

    if initial_opt_state is not None:
        opt_state = initial_opt_state
    elif k_scaling:
        dtype = initial_solution.primal.dtype
        if adaptive_step:
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


def solve(
    cp: JaddleCP,
    optimiser=None,
    max_epochs=None,
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
    update_mode="alternating",
    output_opt_state=False,
    k_scale=10.0,
    k_theta=0.5,
    k_init=None,
    adaptive_eta=None,
    restarts=0,
    epochs_per_restart=10,
    restart_multiplier=1.0,
    restart_decay=0.2,
    iterations_per_epoch_decay=1.0,
    iterations_per_epoch_min=100,
    precompile=True,
):
    """
    Solve a convex saddle-point problem via saddle-point optimisation.

    Args:
        update_mode: Selects the stepping scheme (single source of truth):
            ``"synchronous"`` (default), ``"alternating"``, or
            ``"extragradient"`` (Korpelevich two-call).
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
            Malitsky-Tam local-Lipschitz line search. ``None`` (default) keeps
            the optimiser's fixed learning rate. A float seeds a single scalar
            base step ``eta`` driving the primal step ``tau = eta / k`` and dual
            step ``sigma = eta * k``. The extragradient look-ahead and corrector
            already evaluate the gradient twice, so the local Lipschitz estimate
            ``L_hat = ‖g_half - g‖_w / ‖z_half - z‖_w`` (the k-weighted norm)
            comes free; the step is admissible while ``eta · L_hat <= 1/sqrt2``,
            and ``eta`` is rejected + shrunk if the trial overshot, then advanced
            with a two-sided guard. Only supported with
            ``update_mode='extragradient'`` (the contractive scheme here) and
            requires ``k_scale`` (the primal weight k). ``eta`` is reset to its
            seed at each restart. The optimiser's learning rate is bypassed in
            the hot loop.
        restarts: Maximum number of warm restarts (default 0 = disabled). Each
            restart resets the optimiser momentum and averaging while keeping the
            current iterate as a warm start. A restart fires when the normalised
            KKT merit drops below ``restart_decay`` × the merit at the last
            restart (sufficient-progress restart) or the cycle-length cap is
            exhausted (no-progress restart).
        epochs_per_restart: Length cap (epochs) of the first restart cycle
            (default 10). Subsequent caps grow by ``restart_multiplier``.
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
    """

    if optimiser is None:
        optimiser = jo.gd(1 / 2)

    if log_every < 1:
        raise ValueError("log_every must be >= 1")

    if verbose:
        print("----------------------------------------------")

    valid_update_modes = [
        "synchronous",
        "alternating",
        "extragradient",
    ]
    if update_mode not in valid_update_modes:
        raise ValueError(f"update_mode must be one of {valid_update_modes}")

    # Per-iteration adaptive step size (Malitsky-Tam line search). Only the
    # extragradient scheme is contractive on the saddle, so the line search is
    # restricted to it; it also needs the primal weight k, hence k_scale.
    adaptive_step = adaptive_eta is not None and update_mode == "extragradient"
    if adaptive_eta is not None:
        if update_mode != "extragradient":
            raise ValueError(
                "adaptive_eta is only supported with update_mode='extragradient'"
            )
        if k_scale is None:
            raise ValueError("adaptive_eta requires k_scale (primal weight k)")

    # ``k_scale`` is the public knob for primal-weight scaling: ``None`` disables
    # it, otherwise it sets a symmetric clamp band ``[1/k_scale, k_scale]``.
    k_scaling = k_scale is not None
    if k_scaling:
        k_lo, k_hi = 1.0 / k_scale, k_scale
    else:
        k_lo, k_hi = None, None

    if verbose:
        print("====Starting Solve====")
        print("----------------------------------------------")

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
        return (
            (primal_grad_norm > primal_grad_norm_tolerance)
            | (complementarity_slack > complementarity_slack_tolerance)
            | (constraint_bound > primal_feasibility_tolerance)
        )

    def check_max_epochs(count):
        return count >= max_epochs

    def cond_fun(loop_vars):
        (
            i,
            state,
            average_state,
            opt_state,
            previous_objective,
            primal_grad_norm,
            complementarity_slack,
            constraint_bound,
            count,
            objective_value,
            duality_gap,
            dual_gap_is_finite,
            total_weight,
        ) = loop_vars

        return check_convergence(
            primal_grad_norm, complementarity_slack, constraint_bound
        )

    def body_fun(loop_vars):
        (
            i,
            state,
            average_state,
            opt_state,
            previous_objective,
            primal_grad_norm,
            complementarity_slack,
            constraint_bound,
            count,
            objective_value,
            duality_gap,
            dual_gap_is_finite,
            total_weight,
        ) = loop_vars

        (
            i,
            state,
            average_state,
            opt_state,
            total_weight,
        ) = __sps(
            iterations_per_epoch,
            i,
            cp,
            optimiser,
            state,
            average_state,
            opt_state,
            weight_function,
            total_weight,
            average,
            update_mode,
            k_scaling,
            k_init,
            adaptive_eta,
        )

        (
            objective_value,
            primal_grad_norm,
            complementarity_slack,
            constraint_bound,
            duality_gap,
            dual_gap_is_finite,
        ) = compute_epoch_metrics(average_state if average else state)

        count += 1

        return (
            i,
            state,
            average_state,
            opt_state,
            previous_objective,
            primal_grad_norm,
            complementarity_slack,
            constraint_bound,
            count,
            objective_value,
            duality_gap,
            dual_gap_is_finite,
            total_weight,
        )

    i = 1

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
        k_init = float(jnp.clip(jnp.sqrt(norm_c2 / norm_b2), k_lo, k_hi))
    elif k_init is None:
        k_init = 1.0

    is_converged = True
    state = initial_solution
    average_state = initial_solution
    if initial_opt_state is not None:
        opt_state = initial_opt_state
    elif k_scaling:
        dtype = initial_solution.primal.dtype
        if adaptive_step:
            k_slot = (jnp.asarray(k_init, dtype), jnp.asarray(adaptive_eta, dtype))
        else:
            k_slot = jnp.asarray(k_init, dtype)
        opt_state = (optimiser.init(initial_solution), k_slot)
    else:
        opt_state = optimiser.init(initial_solution)
    primal_grad_norm = jnp.inf
    complementarity_slack = jnp.inf
    constraint_bound = jnp.inf
    objective_value = jnp.inf
    duality_gap = jnp.inf
    dual_gap_is_finite = False
    count = 0
    total_weight = 0.0
    current_iterations_per_epoch = iterations_per_epoch

    restarts_done = 0
    restart_i_offset = 0
    epochs_since_restart = 0
    current_cycle_cap = float(epochs_per_restart)
    merit_at_last_restart = jnp.inf
    # Iterate at the last restart (or the start), used to rebalance the primal
    # weight k from the primal-vs-dual movement over the restart cycle.
    # Independent copy: `state` (== initial_solution) is donated to __sps each
    # epoch, so an alias here would read as deleted at the first k-rebalance.
    state_at_last_restart = jax.tree.map(lambda x: x + 0, initial_solution)

    def kkt_merit(primal_grad_norm, complementarity_slack, constraint_bound):
        # Normalised KKT merit for the restart trigger. Each term is divided by
        # (1 + its initial value) so all three are O(1) at the start. Since we
        # don't track initial values, just use 1.0 as the normaliser (the merit
        # is only compared with itself at successive restarts, so the scale
        # cancels). The maximum of the three terms drives the trigger.
        return jnp.maximum(
            jnp.maximum(primal_grad_norm, complementarity_slack),
            constraint_bound,
        )

    if precompile:
        # Warm both jitted functions the hot loop uses (the scan body via
        # __sps, and the end-of-epoch metrics) with the exact argument
        # signatures the timed loop will feed them, so epoch 1 doesn't pay
        # their first-call compile inside the timed measurement.
        #
        # NOTE: convex's run_epoch takes max_iter as a *static* argname (the
        # scan length is baked into the compile), so it must be warmed with the
        # real `current_iterations_per_epoch` value — warming with max_iter=1
        # would compile a throwaway length-1 executable and leave the real one
        # to compile cold in epoch 1. (Linear can use max_iter=1 because there
        # it's a traced while_loop bound, not a static scan length.)
        # run_epoch donates its state/opt_state buffers, so feed the warm-up
        # *copies* — the real state/opt_state the timed loop uses must survive.
        _warm_state = jax.tree.map(lambda x: x + 0, state)
        _warm_opt_state = jax.tree.map(lambda x: x + 0, opt_state)
        _precompile_result = __sps(
            current_iterations_per_epoch,
            i - restart_i_offset,
            cp,
            optimiser,
            _warm_state,
            average_state,
            _warm_opt_state,
            weight_function,
            total_weight,
            average,
            update_mode,
            k_scaling,
            k_init,
            adaptive_eta,
        )
        _precompile_metrics = compute_epoch_metrics(average_state if average else state)
        jax.block_until_ready((_precompile_result, _precompile_metrics))

    start_time = time.time()

    try:
        while check_convergence(
            primal_grad_norm,
            complementarity_slack,
            constraint_bound,
        ):
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
                cp,
                optimiser,
                state,
                average_state,
                opt_state,
                weight_function,
                total_weight,
                average,
                update_mode,
                k_scaling,
                k_init,
                adaptive_eta,
            )
            i = shifted_i + restart_i_offset

            (
                objective_value,
                primal_grad_norm,
                complementarity_slack,
                constraint_bound,
                duality_gap,
                dual_gap_is_finite,
            ) = compute_epoch_metrics(average_state if average else state)

            finish_epoch_time = time.time()
            count += 1

            if verbose and (count == 1 or count % log_every == 0):
                print(
                    f"|Epoch {count}|"
                    f"|Obj{objective_value:.2e}|"
                    f"|PGN {primal_grad_norm:.2e}|"
                    f"|CS {complementarity_slack:.2e}|"
                    f"|PFR {constraint_bound:.2e}|"
                    f"|Time {finish_epoch_time - start_epoch_time:.2f}s|"
                )
                print("----------------------------------------------")

            # --- Adaptive restart decision ---
            if restarts and restarts_done < restarts:
                epochs_since_restart += 1

                merit = kkt_merit(
                    primal_grad_norm, complementarity_slack, constraint_bound
                )

                # Two-point restart: pick the better of average and iterate.
                restart_point = average_state if average else state
                restart_merit = merit
                restart_used_avg = bool(average)
                if average:
                    (
                        _obj,
                        st_pgn,
                        st_cs,
                        st_cb,
                        _dg,
                        _dgf,
                    ) = compute_epoch_metrics(state)
                    state_merit = kkt_merit(st_pgn, st_cs, st_cb)
                    if bool(state_merit < merit):
                        restart_point = state
                        restart_merit = state_merit
                        restart_used_avg = False

                if not jnp.isfinite(merit_at_last_restart):
                    merit_at_last_restart = restart_merit

                sufficient_progress = bool(
                    restart_merit <= restart_decay * merit_at_last_restart
                )
                cycle_exhausted = epochs_since_restart >= current_cycle_cap

                if sufficient_progress or cycle_exhausted:
                    state = restart_point
                    if k_scaling:
                        # PDLP-style primal-weight rebalance: drive k from the
                        # primal-vs-dual *movement* over the just-finished cycle
                        # (distance between iterates), not per-step gradient
                        # norms. log-space geometric-mean blend with the current
                        # weight (k_theta), then clamp.
                        dp = state.primal - state_at_last_restart.primal
                        dd = jnp.concatenate(
                            [
                                state.dual_eq - state_at_last_restart.dual_eq,
                                state.dual_ineq - state_at_last_restart.dual_ineq,
                            ]
                        )
                        # Use squared norms to avoid two sqrt ops; ratio is preserved.
                        move_p2 = jnp.vdot(dp, dp) + 1e-60
                        move_d2 = jnp.vdot(dd, dd) + 1e-60
                        k_target = jnp.sqrt(move_p2 / move_d2)
                        # The k-slot is (k, eta) in adaptive_step mode, plain k
                        # otherwise. Read k accordingly.
                        k_prev = opt_state[1][0] if adaptive_step else opt_state[1]
                        log_k = k_theta * jnp.log(k_target) + (1.0 - k_theta) * jnp.log(
                            k_prev
                        )
                        k_new = jnp.clip(jnp.exp(log_k), k_lo, k_hi)
                        if adaptive_step:
                            # Reset eta to its seed alongside the momentum reset,
                            # so the adaptive rule re-warms in the new cycle.
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
                    # deleted at the next restart's k-rebalance.
                    state_at_last_restart = jax.tree.map(lambda x: x + 0, state)
                    total_weight = 0.0
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
                            _k_val = opt_state[1][0] if adaptive_step else opt_state[1]
                            k_msg = f", k={float(_k_val):.3e}"
                        else:
                            k_msg = ""
                        print(
                            f"Restart {restarts_done}/{restarts} at epoch {count} "
                            f"({reason}, merit={float(restart_merit):.2e} "
                            f"[{which}], next cap={current_cycle_cap:.0f} epochs, "
                            f"iters/epoch={current_iterations_per_epoch}{k_msg})"
                        )
                        print("----------------------------------------------")

        if average:
            output = average_state
        else:
            output = state
    except KeyboardInterrupt:
        is_converged = False
        if average:
            output = average_state
        else:
            output = state
        print("KeyboardInterrupt received. Returning current solution.")
        print("----------------------------------------------")

    output = jax.block_until_ready(output)
    end_time = time.time()
    print(f"Time to solution: {end_time - start_time:.2f} seconds")
    print("----------------------------------------------")
    print(f"Epochs to solution: {count}")
    print("----------------------------------------------")
    print(f"Objective: {cp.objective(output.primal):.5e}")
    print("----------------------------------------------")

    if output_opt_state:
        return output, is_converged, opt_state
    else:
        return output, is_converged


# %%
