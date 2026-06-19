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
    exponential_weight=0.01,
    total_weight=0.0,
    primal_damping=0.0,
    dual_damping_ineq=0.0,
    dual_damping_eq=0.0,
    average="off",
    update_mode="synchronous",
    pdhg_state=None,
    pdhg_eta_init=1.0,
    pdhg_omega_init=1.0,
    pdhg_reduce=0.3,
    pdhg_grow=0.5,
    per_iterate_k=False,
    per_iterate_k_theta=0.1,
    per_iterate_k_lo=0.1,
    per_iterate_k_hi=10.0,
    k_init=1.0,
    eta_init=1.0,
    extragradient=False,
):

    def projection_primal(primal_state):
        return projection_box(primal_state, lp.lower_bounds, lp.upper_bounds)

    def grad_with_Ax(state):
        # Fused matvecs: 2 sparse ops instead of 4. Returns `Ax` alongside the
        # gradient so the per-iterate magnitude controller can form `A·dx` for
        # free as `Ax_k - Ax_{k-1}` (no extra matvec).
        dual = jnp.concatenate([state.dual_eq, state.dual_ineq])
        Ax = lp.A @ state.primal  # shape: (n_eq + n_ineq,)
        ATd = lp.A_T @ dual  # shape: (n_vars,)
        grad_primal = lp.c + ATd + primal_damping * state.primal
        residual = lp.b - Ax
        grad_dual_eq = residual[: lp.n_eq] + dual_damping_eq * state.dual_eq
        grad_dual_ineq = residual[lp.n_eq :] + dual_damping_ineq * state.dual_ineq
        return (
            SaddleState(
                primal=grad_primal,
                dual_ineq=grad_dual_ineq,
                dual_eq=grad_dual_eq,
            ),
            Ax,
        )

    def grad(state):
        return grad_with_Ax(state)[0]

    def opt_update(gradient, opt_state, state):
        return optimiser.update(gradient, opt_state, state)

    def zero_dual(gradient):
        return SaddleState(
            primal=gradient.primal,
            dual_ineq=jnp.zeros_like(gradient.dual_ineq),
            dual_eq=jnp.zeros_like(gradient.dual_eq),
        )

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

    cache_key = (
        id(lp),
        id(optimiser),
        id(weight_function),
        float(exponential_weight),
        float(primal_damping),
        float(dual_damping_ineq),
        float(dual_damping_eq),
        average,
        update_mode,
        float(pdhg_reduce),
        float(pdhg_grow),
        bool(per_iterate_k),
        float(per_iterate_k_theta),
        float(per_iterate_k_lo),
        float(per_iterate_k_hi),
        bool(extragradient),
    )
    run_epoch = _LINEAR_RUN_EPOCH_CACHE.get(cache_key)

    if run_epoch is None:

        @jax.jit
        def run_epoch(
            max_iter,
            start_iter,
            state,
            average_state,
            opt_state,
            total_weight=0.0,
        ):
            apply_updates = optax.apply_updates

            if update_mode == "pdhg":
                # True PDHG (Chambolle–Pock) with PDLP-style per-iteration
                # adaptive step size and primal weight. The optax `optimiser` /
                # `opt_state` are bypassed entirely: PDHG carries its own scalar
                # step `eta` and primal weight `omega` in `opt_state` (as a
                # length-2 array `[eta, omega]`), since the step sizes are state
                # adapted from iterate movement, not an optax transform.
                #
                # Parameterisation (PDLP): tau = eta / omega (primal step),
                # sigma = eta * omega (dual step). The omega-weighted norm is
                #   ||z||^2_w = omega ||x||^2 + (1/omega) ||y||^2.
                # The adaptive rule accepts a step when
                #   eta <= eta_bar := ||z_{k+1}-z_k||^2_w
                #                     / ( 2 (dy)^T A (dx) ),
                # then proposes the next trial step by interpolating toward
                # eta_bar (grow) while backtracking (reduce) on rejection. This
                # is the mechanism that lets PDHG track the local interaction
                # norm of A instead of a fixed worst-case 1/||A|| step.

                def pdhg_iter(i, state, eta, omega):
                    x = state.primal
                    y = jnp.concatenate([state.dual_eq, state.dual_ineq])
                    ATy = lp.A_T @ y  # reused in the primal grad

                    def attempt(carry):
                        eta_try, *_ = carry
                        tau = eta_try / omega
                        sigma = eta_try * omega

                        # True Chambolle–Pock (primal-first form with
                        # over-relaxation). The extrapolated primal
                        # x_bar = 2 x_new - x feeds the dual step; this is the
                        # extrapolation that makes PDHG *contract* on the coupled
                        # rotation instead of orbiting it like Arrow–Hurwicz/GDA.
                        grad_primal = lp.c + ATy
                        x_new = projection_primal(x - tau * grad_primal)

                        x_bar = 2.0 * x_new - x
                        residual = lp.A @ x_bar - lp.b
                        y_new = y + sigma * residual
                        # eq duals are free; ineq duals project to nonneg.
                        y_new = jnp.concatenate(
                            [
                                y_new[: lp.n_eq],
                                projection_non_negative(y_new[lp.n_eq :]),
                            ]
                        )

                        dx = x_new - x
                        dy = y_new - y
                        # omega-weighted movement and the interaction term.
                        norm_w = omega * (dx @ dx) + (dy @ dy) / omega
                        interaction = dy @ (lp.A @ dx)
                        # eta_bar: largest step keeping the CP descent
                        # inequality valid. Guard the (near-)zero interaction
                        # case (no coupling movement) with a large bound so the
                        # step is accepted and grows.
                        eta_bar = jnp.where(
                            jnp.abs(interaction) > 1e-30,
                            norm_w / (2.0 * jnp.abs(interaction)),
                            jnp.inf,
                        )
                        accepted = eta_try <= eta_bar
                        # On accept, next trial grows toward eta_bar; on reject,
                        # backtrack. Both clamp against eta_bar for safety.
                        eta_grow = jnp.minimum((1.0 + pdhg_grow) * eta_try, eta_bar)
                        eta_shrink = jnp.minimum(
                            (1.0 - pdhg_reduce) * eta_bar, pdhg_reduce * eta_try
                        )
                        eta_next = jnp.where(accepted, eta_grow, eta_shrink)
                        return (eta_next, x_new, y_new, accepted)

                    def cond(carry):
                        return jnp.logical_not(carry[3])

                    eta_next, x_new, y_new, _ = jax.lax.while_loop(
                        cond,
                        attempt,
                        attempt((eta, x, y, False)),
                    )

                    new_state = SaddleState(
                        primal=x_new,
                        dual_eq=y_new[: lp.n_eq],
                        dual_ineq=y_new[lp.n_eq :],
                    )
                    return new_state, eta_next

                def step(carry, _):
                    i, state, average_state, opt_state, total_weight = carry
                    eta, omega = opt_state[0], opt_state[1]

                    state, eta = pdhg_iter(i, state, eta, omega)
                    opt_state = jnp.array([eta, omega])

                    if average == "polyak":
                        w = weight_function(i)
                        total_weight = total_weight + w
                        average_state = optax.incremental_update(
                            state, average_state, w / total_weight
                        )
                    elif average == "exponential":
                        average_state = optax.incremental_update(
                            state, average_state, exponential_weight
                        )

                    return (i + 1, state, average_state, opt_state, total_weight), None

            elif update_mode == "alternating":

                def step(carry, _):
                    i, state, average_state, opt_state, total_weight = carry

                    # 1) Primal-only update
                    g0 = grad(state)
                    primal_updates, _ = opt_update(zero_dual(g0), opt_state, state)
                    state = apply_updates(state, keep_only_primal(primal_updates))
                    state = SaddleState(
                        primal=projection_primal(state.primal),
                        dual_ineq=state.dual_ineq,
                        dual_eq=state.dual_eq,
                    )

                    # 2) Dual-only update (using post-primal dual gradients)
                    g1 = grad(state)
                    combined_gradient = SaddleState(
                        primal=g0.primal,
                        dual_ineq=g1.dual_ineq,
                        dual_eq=g1.dual_eq,
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

                    # `average` is a Python-level static, so dispatch the
                    # averaging mode at trace time. When averaging is "off" the
                    # incremental_update (a full read/write pass over the state
                    # tree) is dropped from the hot loop entirely, and the
                    # polyak-only weight_function call is skipped otherwise.
                    if average == "polyak":
                        w = weight_function(i)
                        total_weight = total_weight + w
                        average_state = optax.incremental_update(
                            state, average_state, w / total_weight
                        )
                    elif average == "exponential":
                        average_state = optax.incremental_update(
                            state, average_state, exponential_weight
                        )

                    return (i + 1, state, average_state, opt_state, total_weight), None

            elif per_iterate_k:
                # Per-iterate primal/dual step-ratio control. The split scales
                # applied updates by (eta/k, eta*k) for (primal, dual), so the
                # dual/primal step ratio is k**2. `k` (ratio) and `eta`
                # (magnitude, set by lr_scale on restarts) are carried alongside
                # the optax state as (opt_state, k, eta).
                #
                #   k_target = sqrt(||u_primal|| / ||u_dual||)   (movement balance)
                #
                # log-space smoothed and clamped.
                def step(carry, _):
                    i, state, average_state, opt_state, total_weight = carry
                    opt_state, k, eta = opt_state

                    g = grad(state)
                    updates, opt_state = opt_update(g, opt_state, state)

                    # --- ratio control (k) ---
                    # Use raw optimizer update norms (before split and projection).
                    # A large primal update that gets clipped by box constraints
                    # correctly signals the dual is under-powered: the primal is
                    # already at its bound and the dual needs more force to move it.
                    up = updates.primal
                    ud = jnp.concatenate([updates.dual_eq, updates.dual_ineq])
                    norm_p = jnp.sqrt(up @ up) + 1e-30
                    norm_d = jnp.sqrt(ud @ ud) + 1e-30
                    k_target = jnp.sqrt(norm_p / norm_d)
                    log_k = per_iterate_k_theta * jnp.log(k_target) + (
                        1.0 - per_iterate_k_theta
                    ) * jnp.log(k)
                    k = jnp.clip(jnp.exp(log_k), per_iterate_k_lo, per_iterate_k_hi)

                    # Apply the (eta/k, eta*k) split to the updates.
                    updates = SaddleState(
                        primal=updates.primal * (eta / k),
                        dual_ineq=updates.dual_ineq * (eta * k),
                        dual_eq=updates.dual_eq * (eta * k),
                    )
                    state = apply_updates(state, updates)
                    state = SaddleState(
                        primal=projection_primal(state.primal),
                        dual_ineq=projection_non_negative(state.dual_ineq),
                        dual_eq=state.dual_eq,
                    )
                    opt_state = (opt_state, k, eta)

                    if average == "polyak":
                        w = weight_function(i)
                        total_weight = total_weight + w
                        average_state = optax.incremental_update(
                            state, average_state, w / total_weight
                        )
                    elif average == "exponential":
                        average_state = optax.incremental_update(
                            state, average_state, exponential_weight
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
                # The (eta/k, eta*k) split is applied to each gradient before
                # passing it to opt_update, exactly as in the per_iterate_k path.
                # k adapts from the look-ahead gradient norms.
                def step(carry, _):
                    i, state, average_state, opt_state, total_weight = carry
                    opt_state, k, eta = opt_state

                    # --- Look-ahead gradient ---
                    g = grad(state)

                    # --- per-iterate k (ratio) from look-ahead gradient norms ---
                    up = g.primal
                    ud = jnp.concatenate([g.dual_eq, g.dual_ineq])
                    norm_p = jnp.sqrt(up @ up) + 1e-30
                    norm_d = jnp.sqrt(ud @ ud) + 1e-30
                    k_target = jnp.sqrt(norm_p / norm_d)
                    log_k = per_iterate_k_theta * jnp.log(k_target) + (
                        1.0 - per_iterate_k_theta
                    ) * jnp.log(k)
                    k = jnp.clip(jnp.exp(log_k), per_iterate_k_lo, per_iterate_k_hi)

                    # Look-ahead: run the user's optimiser on g at state to
                    # get the look-ahead point. la_opt_state is NOT committed —
                    # we discard it and reuse the original opt_state for the
                    # corrector so that momentum/statistics only advance once.
                    scaled_g = SaddleState(
                        primal=g.primal * (eta / k),
                        dual_ineq=g.dual_ineq * (eta * k),
                        dual_eq=g.dual_eq * (eta * k),
                    )
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
                    scaled_g_half = SaddleState(
                        primal=g_half.primal * (eta / k),
                        dual_ineq=g_half.dual_ineq * (eta * k),
                        dual_eq=g_half.dual_eq * (eta * k),
                    )
                    corr_updates, opt_state = opt_update(
                        scaled_g_half, opt_state, state
                    )
                    state = apply_updates(state, corr_updates)
                    state = SaddleState(
                        primal=projection_primal(state.primal),
                        dual_ineq=projection_non_negative(state.dual_ineq),
                        dual_eq=state.dual_eq,
                    )
                    opt_state = (opt_state, k, eta)

                    if average == "polyak":
                        w = weight_function(i)
                        total_weight = total_weight + w
                        average_state = optax.incremental_update(
                            state, average_state, w / total_weight
                        )
                    elif average == "exponential":
                        average_state = optax.incremental_update(
                            state, average_state, exponential_weight
                        )

                    return (i + 1, state, average_state, opt_state, total_weight), None

            else:

                def step(carry, _):
                    i, state, average_state, opt_state, total_weight = carry

                    g = grad(state)
                    updates, opt_state = opt_update(g, opt_state, state)
                    state = apply_updates(state, updates)
                    state = SaddleState(
                        primal=projection_primal(state.primal),
                        dual_ineq=projection_non_negative(state.dual_ineq),
                        dual_eq=state.dual_eq,
                    )

                    # `average` is a Python-level static, so dispatch the
                    # averaging mode at trace time. When averaging is "off" the
                    # incremental_update (a full read/write pass over the state
                    # tree) is dropped from the hot loop entirely, and the
                    # polyak-only weight_function call is skipped otherwise.
                    if average == "polyak":
                        w = weight_function(i)
                        total_weight = total_weight + w
                        average_state = optax.incremental_update(
                            state, average_state, w / total_weight
                        )
                    elif average == "exponential":
                        average_state = optax.incremental_update(
                            state, average_state, exponential_weight
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

    if initial_opt_state is not None:
        opt_state = initial_opt_state
    elif update_mode == "pdhg":
        # PDHG carries its scalar step + primal weight in lieu of an optax state.
        opt_state = jnp.array([pdhg_eta_init, pdhg_omega_init])
    elif per_iterate_k or extragradient:
        # Pack the running step-ratio k and magnitude eta alongside the optax state.
        dtype = initial_solution.primal.dtype
        opt_state = (
            optimiser.init(initial_solution),
            jnp.asarray(k_init, dtype),
            jnp.asarray(eta_init, dtype),
        )
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
    dual_gap_tolerance=1e-2,
    primal_grad_norm_tolerance=1e-3,
    complementarity_slack_tolerance=1e-3,
    weight_function=lambda _: 1.0,
    exponential_weight=0.01,
    verbose=False,
    log_every=10,
    average="off",
    update_mode="synchronous",
    pdhg_eta_init=1.0,
    pdhg_omega_init=1.0,
    pdhg_reduce=0.3,
    pdhg_grow=0.5,
    per_iterate_k=False,
    per_iterate_k_theta=0.1,
    per_iterate_k_lo=0.1,
    per_iterate_k_hi=10.0,
    k_init=1.0,
    eta_init=1.0,
    extragradient=False,
    scale=None,
    expert_diagnostics=False,
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
    polish_optimiser=None,
    polish_lr_scale_threshold=1e-2,
    polish_merit_threshold=None,
    eq_projection_threshold=None,
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

    ``update_mode='pdhg'`` selects true PDHG (Chambolle–Pock) with a PDLP-style
    *per-iteration* adaptive step size and primal weight, bypassing the optax
    ``optimiser`` entirely (it may be left ``None``). PDHG carries a scalar step
    ``eta`` and primal weight ``omega`` (primal step ``tau = eta/omega``, dual
    step ``sigma = eta*omega``) and, each iteration, accepts the largest step
    consistent with the Chambolle–Pock descent inequality measured on the
    *actual* iterate movement (``pdhg_grow`` controls the optimistic growth,
    ``pdhg_reduce`` the backtracking on rejection). This tracks the local
    interaction norm of ``A`` instead of a fixed ``1/||A||`` step, which is the
    main reason PDLP outruns plain GDA on ill-balanced / rotational instances.
    ``restarts`` and ``average`` still apply and are recommended.

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
        k_init: Initial value of ``k`` (default 1.0 = symmetric steps, the
            PC/Ruiz-scaled baseline). Only used when ``per_iterate_k=True`` or
            ``extragradient=True``.
        eta_init: Initial value of the step magnitude ``eta`` (default 1.0).
            On restarts, ``eta`` is scaled by ``lr_scale`` to track the
            magnitude decay. Only used when ``per_iterate_k=True`` or
            ``extragradient=True``.
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
        polish_optimiser: Optional optax optimiser to cross over to once the
            base learning rate has decayed sufficiently. When ``lr_scale`` drops
            below ``polish_lr_scale_threshold`` on a restart, the main optimiser
            is replaced by this one and ``lr_scale`` is reset to 1.0 so the
            polisher's own learning rate is unaffected by the main solver's
            accumulated decay. The polisher warm-starts from the current restart
            point and runs for the remainder of the solve budget. Default
            ``None`` disables crossover.
        polish_lr_scale_threshold: ``lr_scale`` value below which crossover to
            ``polish_optimiser`` fires (default 1e-2). Only meaningful when
            ``polish_optimiser`` is set.
        eq_projection_threshold: When set, after each epoch the unscaled equality
            residual is checked; if it exceeds this value the primal (and average)
            are projected onto the equality manifold ``A_eq x = b_eq`` via the
            precomputed factorisation of ``A_eq A_eq^T``. Default ``None``
            disables projection. Only useful when equality feasibility is the
            bottleneck; has no effect when there are no equality constraints.
    """

    if log_every < 1:
        raise ValueError("log_every must be >= 1")

    if verbose:
        print("----------------------------------------------")

    if update_mode not in ["synchronous", "alternating", "pdhg"]:
        raise ValueError(
            "update_mode must be one of ['synchronous', 'alternating', 'pdhg']"
        )
    if per_iterate_k and update_mode != "synchronous":
        raise ValueError(
            "per_iterate_k is only implemented for update_mode='synchronous'."
        )
    if extragradient and update_mode != "synchronous":
        raise ValueError(
            "extragradient is only implemented for update_mode='synchronous'."
        )
    if extragradient and per_iterate_k:
        raise ValueError(
            "extragradient and per_iterate_k both control per-iterate stepping; "
            "use extragradient=True to get the look-ahead/corrector step with "
            "built-in k/eta adaptation."
        )
    if verbose:
        print("====Starting Solve====")
        print("----------------------------------------------")

    if scale == "ruiz":
        lp, row_scale, col_scale = ruiz_scaling(lp)
        if verbose:
            print("Applied Ruiz scaling to the LP.")
            print("----------------------------------------------")

    elif scale == "pc":
        lp, row_scale, col_scale = pc_scaling(lp)
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

        if verbose:
            print("Applied combined Ruiz + PC scaling to the LP.")
            print("----------------------------------------------")

    else:
        row_scale = np.ones(lp.A_eq.shape[0] + lp.A_ineq.shape[0])
        col_scale = np.ones(lp.c.shape[0])

    if initial_solution is None:
        initial_solution = lp.initial_solution()

    row_scale_ineq = row_scale[len(lp.b_eq) :]
    row_scale_eq = row_scale[: len(lp.b_eq)]

    # Convert to jax arrays for use inside jitted functions
    jnp_row_scale_ineq = jnp.array(row_scale_ineq)
    jnp_row_scale_eq = jnp.array(row_scale_eq)

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

    @jax.jit
    def compute_epoch_metrics(average_state):
        objective_value = lp.objective(average_state.primal) * c_max

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
    ):
        # Standard LP optimality certificate — all three conditions must hold:
        #   * primal feasibility: constraint_bound within tolerance
        #   * dual feasibility: reduced-cost residual within tolerance
        #   * a finite, small duality gap
        # The gap is computed in scaled space (Ruiz+PC) and the objective is in
        # true units (column scaling cancels in c^T x), so normalising by
        # objective_value would mix units. Instead we normalise by
        # (1 + |gap_at_start|) — but since we don't track that, we just use an
        # absolute tolerance on the scaled gap. The kkt_merit used for restarts
        # already does the right relative normalisation; this gate just needs
        # to be loose enough to not block when the gap is genuinely small.
        standard = (
            (constraint_bound <= primal_feasibility_tolerance)
            & (dual_feasibility_residual <= dual_feasibility_threshold)
            & dual_gap_is_finite
            & (jnp.abs(duality_gap) <= dual_gap_tolerance)
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

    i = 1
    state = initial_solution
    average_state = initial_solution
    if initial_opt_state is not None:
        opt_state = initial_opt_state
    elif update_mode == "pdhg":
        # PDHG carries its own [eta, omega] step-state; the optax optimiser is
        # unused in this mode.
        opt_state = jnp.array([pdhg_eta_init, pdhg_omega_init])
    elif per_iterate_k or extragradient:
        # Pack the per-iterate step-ratio k and magnitude eta with the optax state.
        _pik_dtype = initial_solution.primal.dtype
        opt_state = (
            optimiser.init(initial_solution),
            jnp.asarray(k_init, _pik_dtype),
            jnp.asarray(eta_init, _pik_dtype),
        )
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
    is_converged = True
    current_iterations_per_epoch = iterations_per_epoch
    active_extragradient = extragradient
    active_per_iterate_k = per_iterate_k

    # Rolling window of recent objective values for the opt-in primal_stop rule
    # (oldest first). Seeded with inf so the window is not "full" until enough
    # real epochs have elapsed.
    obj_window = jnp.full((max(int(primal_stop_window), 1),), jnp.inf)

    lr_scale = 1.0

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

    start_time = time.time()
    missing_expert_state_warning_printed = False

    def print_expert_weights(epoch_count, state_for_weights):
        nonlocal missing_expert_state_warning_printed
        if not expert_diagnostics:
            return

        extracted = jo.hedge_diagnostics_from_state(state_for_weights)
        if extracted is None:
            extracted = jo.hedge_weights_from_state(state_for_weights)
        if extracted is None:
            if (not missing_expert_state_warning_printed) and verbose:
                print(
                    "Expert diagnostics requested, but optimiser state has no hedge weights."
                )
                print("----------------------------------------------")
                missing_expert_state_warning_printed = True
            return

        if isinstance(extracted, dict):
            weights = extracted["weights"]
            losses = extracted["clipped_losses"]
            centered_losses = extracted["centered_losses"]
            hedge_eta = extracted["eta"]
        else:
            # hedge_weights_from_state fallback: bare weight vector.
            weights = extracted
            losses = centered_losses = hedge_eta = None

        if verbose:
            print(f"Player Weights (epoch {epoch_count}): {np.asarray(weights)}")
            if losses is not None:
                print(f"Player Losses (epoch {epoch_count}): {np.asarray(losses)}")
                print(
                    f"Centered Losses (epoch {epoch_count}): "
                    f"{np.asarray(centered_losses)}"
                )
            if hedge_eta is not None:
                print(f"Hedge Eta (epoch {epoch_count}): {float(hedge_eta):.3e}")
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
            )
        )

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
                exponential_weight,
                total_weight,
                primal_damping,
                dual_damping_ineq,
                dual_damping_eq,
                average,
                update_mode,
                pdhg_eta_init=pdhg_eta_init,
                pdhg_omega_init=pdhg_omega_init,
                pdhg_reduce=pdhg_reduce,
                pdhg_grow=pdhg_grow,
                per_iterate_k=active_per_iterate_k,
                per_iterate_k_theta=per_iterate_k_theta,
                per_iterate_k_lo=per_iterate_k_lo,
                per_iterate_k_hi=per_iterate_k_hi,
                k_init=k_init,
                eta_init=eta_init,
                extragradient=active_extragradient,
            )
            # __sps increments the (restart-shifted) counter; restore global i.
            i = shifted_i + restart_i_offset

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
            ) = jax.lax.cond(
                average == "exponential" or average == "polyak",
                lambda: compute_epoch_metrics(average_state),
                lambda: compute_epoch_metrics(state),
            )

            finish_epoch_time = time.time()
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
                    if average != "off":
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

            if verbose and (count == 1 or count % log_every == 0):
                dual_gap_status = (
                    "finite" if bool(dual_gap_is_finite) else "dual-infeasible"
                )
                print(
                    f"|Epoch {count}|"
                    f"|Obj{objective_value:.2e}|"
                    f"|PGN {primal_grad_norm:.2e}|"
                    f"|CS {complementarity_slack:.2e}|"
                    f"|PFR {constraint_bound:.2e}|"
                    f"|DFR {dual_feasibility_residual:.2e}|"
                    f"|DG {duality_gap:.2e} ({dual_gap_status})|"
                    f"|Time {finish_epoch_time - start_epoch_time:.2f}s|"
                )
                print("----------------------------------------------")

                # Per-iterate k/eta carried in the packed opt_state as
                # (optax_state, k, eta); surface the values settled on this epoch.
                if active_per_iterate_k or active_extragradient:
                    print(
                        f"  per-iterate k={float(opt_state[1]):.3f}"
                        f" eta={float(opt_state[2]):.3e}"
                        f" (τ={float(opt_state[2]) / float(opt_state[1]):.2e},"
                        f" σ={float(opt_state[2]) * float(opt_state[1]):.2e})"
                    )
                    print("----------------------------------------------")

                print_expert_weights(count, opt_state)

            # --- Adaptive restart decision ---
            if (
                restarts
                and restarts_done >= restarts
                and polish_optimiser is not None
                and optimiser is not polish_optimiser
            ):
                optimiser = polish_optimiser
                state = average_state if average != "off" else state
                opt_state = opt_state = (
                    (optimiser.init(state), 1.0, lr_scale)
                    if (active_per_iterate_k or active_extragradient)
                    else optimiser.init(state)
                )
                active_per_iterate_k = True
                active_extragradient = False
                # total_weight = 0.0
                if verbose:
                    print(f"  → Crossing over to polish optimiser (restarts exhausted)")
                    print("----------------------------------------------")

            if (
                restarts
                and restarts_done < restarts
                and optimiser is not polish_optimiser
            ):
                epochs_since_restart += 1

                # `merit` is the metric of the point the epoch metrics were
                # computed on: the average when averaging is on, else the last
                # iterate. This drives the *trigger*.
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
                # start. Compute the *other* point's merit and restart to
                # whichever is better, instead of always restarting to the last
                # iterate (which discarded a frequently-better average).
                restart_point = average_state if average != "off" else state
                restart_merit = merit
                restart_used_avg = average != "off"
                if average != "off":
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
                cycle_exhausted = epochs_since_restart >= current_cycle_cap

                if sufficient_progress or cycle_exhausted:
                    # Warm-start restart from the better of {average, iterate};
                    # reset momentum, averaging, weight accumulation and the LR /
                    # weight_function schedule (via the iteration offset).
                    state = restart_point
                    if update_mode == "pdhg":
                        # Reset PDHG's step/primal-weight to its initial guess;
                        # the iterate (warm start) is preserved.
                        opt_state = jnp.array([pdhg_eta_init, pdhg_omega_init])
                    elif active_per_iterate_k or active_extragradient:
                        # Reset momentum but carry the learned k; apply lr_scale to eta.
                        opt_state = (
                            optimiser.init(state),
                            1.0,
                            lr_scale,
                        )
                    else:
                        opt_state = optimiser.init(state)
                    average_state = state
                    # total_weight = 0.0
                    restart_i_offset = i - 1
                    merit_at_last_restart = restart_merit
                    # Crossover: if lr_scale has decayed below threshold and a
                    # polishing optimiser was provided, swap to it once. The
                    # polisher warm-starts from the current restart point and
                    # runs the rest of the solve budget with lr_scale reset to 1
                    # (so the polisher's own learning rate is unaffected by the
                    # main solver's accumulated decay).
                    crossover_lr = (
                        polish_optimiser is not None
                        and optimiser is not polish_optimiser
                        and lr_scale < polish_lr_scale_threshold
                    )
                    crossover_merit = (
                        polish_optimiser is not None
                        and optimiser is not polish_optimiser
                        and polish_merit_threshold is not None
                        and bool(restart_merit < polish_merit_threshold)
                    )
                    if crossover_lr or crossover_merit:
                        optimiser = polish_optimiser
                        opt_state = (
                            (optimiser.init(state), opt_state[1], lr_scale)
                            if (active_per_iterate_k or active_extragradient)
                            else optimiser.init(state)
                        )
                        active_per_iterate_k = True
                        active_extragradient = False
                        if verbose:
                            reason = (
                                f"lr_scale crossed {polish_lr_scale_threshold:.2e}"
                                if crossover_lr
                                else f"merit crossed {polish_merit_threshold:.2e}"
                            )
                            print(f"  → Crossing over to polish optimiser ({reason})")
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
                        print(
                            f"Restart {restarts_done}/{restarts} at epoch {count} "
                            f"({reason}, merit={float(restart_merit):.2e} "
                            f"[{which}], lr_scale={lr_scale:.3f}, "
                            f"next cap={current_cycle_cap:.0f} epochs, "
                            f"iters/epoch={current_iterations_per_epoch})"
                        )
                        print("----------------------------------------------")

        if average != "off":
            output = average_state
        else:
            output = state
    except KeyboardInterrupt:
        is_converged = False
        if average != "off":
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
    print(f"Objective: {lp.objective(output.primal):.5e}")
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
    # Resolve to the active JAX float width: float64 when x64 is enabled (the
    # "x64" profile, PDLP-style double precision), float32 otherwise. Hardcoding
    # float64 with x64 disabled silently truncated and spammed warnings.
    float_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    np_float = np.float64 if jax.config.jax_enable_x64 else np.float32

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


def spectral_norm(lp: JaddleLP, n_iter: int = 30) -> float:
    """Estimate ‖[A_eq; A_ineq]‖₂ via power iteration.

    Returns the spectral norm, which gives the theoretically safe step size
    bound ``η ≤ 1 / spectral_norm(lp)`` for PDHG-style solvers.
    """
    key = jax.random.PRNGKey(0)
    v = jax.random.normal(key, (lp.A_eq.shape[1],))
    v = v / jnp.linalg.norm(v)

    A_eq, A_ineq = lp.A_eq, lp.A_ineq
    for _ in range(n_iter):
        v2 = A_eq.T @ (A_eq @ v) + A_ineq.T @ (A_ineq @ v)
        sigma = jnp.linalg.norm(v2)
        v = v2 / sigma

    return float(jnp.sqrt(sigma))


def ruiz_scaling(lp: LP, max_iter=20, threshold=1e-8, clip_bounds=(1e-6, 1e6)):
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

    A_eq = __convert_to_scipy(lp.A_eq)
    A_ineq = __convert_to_scipy(lp.A_ineq)
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

    for _ in range(max_iter):
        M_cur = sp.diags(row_scale) @ M @ sp.diags(col_scale)

        row_norms = np.abs(M_cur).max(axis=1).todense().A.flatten()
        row_norms = np.where(row_norms <= threshold, 1.0, row_norms)
        row_s = np.clip(1.0 / np.sqrt(row_norms), clip_bounds[0], clip_bounds[1])
        row_scale *= row_s

        M_cur = sp.diags(row_scale) @ M @ sp.diags(col_scale)

        col_norms = np.abs(M_cur).max(axis=0).todense().A.flatten()
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

    lp_scaled = to_jaddle_sparse(lp_scaled)

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

    A_eq = __convert_to_scipy(lp.A_eq)
    A_ineq = __convert_to_scipy(lp.A_ineq)
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

    for _ in range(max_iter):
        M_cur = sp.diags(row_scale) @ M @ sp.diags(col_scale)

        row_norms = np.abs(M_cur).sum(axis=1).A.flatten()
        row_norms = np.where(row_norms <= threshold, 1.0, row_norms)
        row_s = np.clip(1.0 / np.sqrt(row_norms), clip_bounds[0], clip_bounds[1])
        row_scale *= row_s

        M_cur = sp.diags(row_scale) @ M @ sp.diags(col_scale)

        col_norms = np.abs(M_cur).sum(axis=0).A.flatten()
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

    lp_scaled = to_jaddle_sparse(lp_scaled)

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


# %%
