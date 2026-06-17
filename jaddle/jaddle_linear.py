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
    per_iterate_eta=False,
    per_iterate_eta_theta=0.1,
    per_iterate_eta_lo=1e-4,
    per_iterate_eta_hi=1e4,
    eta_init=1.0,
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
        bool(per_iterate_eta),
        float(per_iterate_eta_theta),
        float(per_iterate_eta_lo),
        float(per_iterate_eta_hi),
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
                # Per-iterate primal/dual step-ratio control, the matvec-free
                # lesson borrowed from PDHG/PDLP. PDLP's value is not its line
                # search (which is what makes update_mode="pdhg" slow — a
                # data-dependent while_loop that breaks scan fusion and costs an
                # extra A@dx matvec per trial); its value is the per-iterate
                # *primal-weight* signal. We reconstruct that signal for free
                # from the optimiser updates already computed each step, with no
                # extra matvec and no inner loop, so the epoch stays a single
                # fused lax.scan.
                #
                # `k` (ratio) and, optionally, `eta` (magnitude) are carried
                # alongside the optax state as (opt_state, k, eta, prev_Ax,
                # prev_dual). The combined split scales the applied updates by
                # (eta/k, eta*k) for (primal, dual), so the dual/primal step
                # ratio is k**2 (same convention as the epoch-level update_k) and
                # eta sets the shared magnitude.
                #
                #   k_target = sqrt(||u_primal|| / ||u_dual||)   (movement balance)
                #
                # The magnitude target is PDLP's adaptive-step bound, made
                # matvec-free: the descent bound is
                #   eta_bar = ||dz||^2_omega / (2 |dy^T (A dx)|),
                # and A·dx is recovered for free as Ax_k - Ax_{k-1} (Ax is already
                # computed every iteration by grad_with_Ax), so no extra matvec
                # and no while_loop are needed. Movement is measured from the
                # *previous* iterate (a one-step lag), then both controls are
                # log-space smoothed and clamped.
                def step(carry, _):
                    i, state, average_state, opt_state, total_weight = carry
                    opt_state, k, eta, prev_Ax, prev_dual = opt_state

                    g, Ax = grad_with_Ax(state)
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

                    # --- magnitude control (eta) ---
                    if per_iterate_eta:
                        dual_now = jnp.concatenate([state.dual_eq, state.dual_ineq])
                        # A·dx from cached Ax (free); dy from cached dual.
                        A_dx = Ax - prev_Ax
                        dy = dual_now - prev_dual
                        # omega-weighted movement norm with omega = k**2 (so the
                        # weighting matches the (1/k, k) split we apply).
                        omega = k * k
                        norm_w = omega * (norm_p * norm_p) + (dy @ dy) / omega
                        interaction = jnp.abs(dy @ A_dx)
                        eta_bar = jnp.where(
                            interaction > 1e-30,
                            norm_w / (2.0 * interaction),
                            eta,  # no coupling movement → keep current eta
                        )
                        log_eta = per_iterate_eta_theta * jnp.log(eta_bar) + (
                            1.0 - per_iterate_eta_theta
                        ) * jnp.log(eta)
                        eta = jnp.clip(
                            jnp.exp(log_eta),
                            per_iterate_eta_lo,
                            per_iterate_eta_hi,
                        )
                        prev_Ax = Ax
                        prev_dual = dual_now

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
                    opt_state = (opt_state, k, eta, prev_Ax, prev_dual)

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
    elif per_iterate_k:
        # Pack the running step-ratio k, magnitude eta, and the previous-iterate
        # caches (prev_Ax, prev_dual) alongside the optax state. `solve` threads
        # this packed tuple opaquely across epochs.
        dtype = initial_solution.primal.dtype
        prev_Ax = lp.A @ initial_solution.primal
        prev_dual = jnp.concatenate(
            [initial_solution.dual_eq, initial_solution.dual_ineq]
        )
        opt_state = (
            optimiser.init(initial_solution),
            jnp.asarray(k_init, dtype),
            jnp.asarray(eta_init, dtype),
            prev_Ax,
            prev_dual,
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


def update_k(k, primal_feas, dual_feas, theta=0.5, lo=0.3, hi=3.0, deadband=2.0):
    """Residual-balancing update for the primal/dual step-ratio scalar ``k``,
    where the dual/primal step ratio is ``k**2`` (primal step ``∝ 1/k``, dual
    step ``∝ k``).

    The mapping respects which *player resolves* which residual, not which
    residual is named after which player:

      * ``primal_feas`` = ‖Ax − b‖ (primal-feasibility / constraint residual) is
        resolved by the **dual** — the dual prices constraint violation, and that
        price ``Aᵀy`` is the only force pulling x toward feasibility. So a large
        primal-feasibility residual means the dual is under-powered → grow ``k``.
      * ``dual_feas`` = ‖c + Aᵀy − z‖ (dual-feasibility / reduced-cost residual)
        is resolved by the **primal** — x must respond to the prices. A large
        dual-feasibility residual means the primal is under-powered → shrink
        ``k``.

    Hence ``k ← k · (primal_feas / dual_feas)**(theta/2)``: constraint
    infeasibility drives the dual, reduced-cost infeasibility drives the primal.

    Both residuals must be supplied in the *scaled* space the solver iterates in
    (not un-scaled certificate units) so the controller reacts to primal/dual
    dynamics rather than the conditioning PC/Ruiz scaling has already removed.

    The update is damped (``theta`` < 1), suppressed inside a multiplicative
    ``deadband`` around balance, and clamped to ``[lo, hi]``. Hitting a clamp is
    the signal that step-size rebalancing alone cannot fix the iteration (the
    failure is directional GDA instability, not an imbalance) and the fix is
    averaging / alternating updates / optimistic-EG correction, not more ``k``.
    """
    ratio = (primal_feas + 1e-30) / (dual_feas + 1e-30)
    in_band = (ratio < deadband) & (ratio > 1.0 / deadband)
    factor = jnp.where(in_band, 1.0, ratio ** (0.5 * theta))
    return jnp.clip(k * factor, lo, hi)


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
    per_iterate_eta=False,
    per_iterate_eta_theta=0.1,
    per_iterate_eta_lo=1e-4,
    per_iterate_eta_hi=1e4,
    eta_init=1.0,
    scale=None,
    expert_diagnostics=False,
    output_opt_state=False,
    scaled_objective=False,
    restarts=0,
    epochs_per_restart=10,
    restart_multiplier=1.0,
    restart_decay=0.2,
    adaptive_k=False,
    base_lr=1e-1,
    k_init=1.0,
    k_theta=0.5,
    k_lo=0.3,
    k_hi=3.0,
    k_deadband=2.0,
    primal_stop=False,
    primal_stop_window=5,
    primal_stop_obj_tol=1e-4,
    iterations_per_epoch_decay=1.0,
    iterations_per_epoch_min=100,
):
    """
    Solve a linear program via saddle-point optimisation.

    Termination uses the standard LP optimality certificate: primal feasibility
    (``primal_feasibility_tolerance``), dual feasibility
    (``dual_feasibility_tolerance``), and a finite duality gap within
    ``dual_gap_tolerance``. ``primal_grad_norm`` and ``complementarity_slack``
    are reported as diagnostics only — they are redundant with these three
    conditions and do not gate termination.

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
    ``adaptive_k`` is incompatible with this mode (PDHG adapts the ratio
    itself); ``restarts`` and ``average`` still apply and are recommended.

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
        adaptive_k: Enable the per-epoch residual-balancing controller for the
            primal/dual step ratio (dual/primal step ratio is ``k**2``). The two
            sub-optimisers MUST be built with
            ``optax.inject_hyperparams(...)(learning_rate=...)`` so their learning
            rates are live array leaves; otherwise this raises. Default ``False``
            leaves the optimiser learning rates untouched.
        base_lr: Geometric-mean learning-rate *magnitude* the controller splits
            as ``(base_lr / k, base_lr * k)`` for (primal, dual). Either a float
            (constant) or a callable ``base_lr(epoch) -> float`` for an annealing
            schedule, evaluated once per epoch. The controller adapts the ratio
            ``k`` independently of this magnitude decay. Only used when
            ``adaptive_k=True``.
        k_init: Initial value of ``k`` (default 1.0 = symmetric steps, the
            PC/Ruiz-scaled baseline).
        k_theta: Damping on the controller update (default 0.5). The effective
            per-epoch exponent on the residual ratio is ``k_theta / 2``; smaller
            is slower/safer.
        k_lo, k_hi: Clamp band for ``k`` (default ``[0.3, 3.0]``). Sitting at a
            clamp is the signal that step-size rebalancing alone cannot fix the
            iteration (directional GDA instability) — the fix is optimistic/
            extragradient correction on both players, not widening the band.
        k_deadband: Multiplicative deadband around balance (default 2.0); the
            controller does not react while ``r_dual/r_primal`` is within
            ``[1/k_deadband, k_deadband]``, preventing limit-cycle hunting.
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
    """

    if log_every < 1:
        raise ValueError("log_every must be >= 1")

    if verbose:
        print("----------------------------------------------")

    if update_mode not in ["synchronous", "alternating", "pdhg"]:
        raise ValueError(
            "update_mode must be one of ['synchronous', 'alternating', 'pdhg']"
        )
    if update_mode == "pdhg" and adaptive_k:
        raise ValueError(
            "adaptive_k is incompatible with update_mode='pdhg': PDHG adapts its "
            "own step size (eta) and primal weight (omega) per iteration, so the "
            "epoch-level k controller and set_saddle_lrs do not apply."
        )
    if per_iterate_k and update_mode != "synchronous":
        raise ValueError(
            "per_iterate_k is only implemented for update_mode='synchronous'."
        )
    if per_iterate_k and adaptive_k:
        raise ValueError(
            "per_iterate_k and adaptive_k both control the primal/dual step ratio; "
            "use one. per_iterate_k adapts k every iteration (matvec-free, inside "
            "the scan); adaptive_k adapts it once per epoch from the certificate "
            "residuals."
        )
    if per_iterate_eta and not per_iterate_k:
        raise ValueError(
            "per_iterate_eta (magnitude control) requires per_iterate_k=True: the "
            "two share the same packed step-state and the (eta/k, eta*k) split."
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

        # Scaled-space constraint residual (∞-norm), used by the adaptive-k
        # controller. Kept in scaled units (unlike the certificate residuals
        # below) so it is comparable with `primal_grad_norm` and reflects the
        # primal/dual dynamics rather than the conditioning scaling removed.
        constraint_residual_scaled = jnp.max(jnp.abs(Ax_minus_b))

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

        dual_feasibility_violation = jnp.where(
            has_only_lower,
            jnp.maximum(-reduced_cost, 0.0),
            jnp.where(
                has_only_upper,
                jnp.maximum(reduced_cost, 0.0),
                jnp.where(has_no_bounds, jnp.abs(reduced_cost), 0.0),
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
            constraint_residual_scaled,
        )

    def converged(
        constraint_bound,
        dual_feasibility_residual,
        duality_gap,
        dual_gap_is_finite,
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
        return (
            (constraint_bound <= primal_feasibility_tolerance)
            & (dual_feasibility_residual <= dual_feasibility_threshold)
            & dual_gap_is_finite
            & (jnp.abs(duality_gap) <= dual_gap_tolerance)
        )

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
    elif per_iterate_k:
        # Pack the per-iterate step-ratio k, magnitude eta, and previous-iterate
        # caches (prev_Ax, prev_dual) with the optax state.
        _pik_dtype = initial_solution.primal.dtype
        opt_state = (
            optimiser.init(initial_solution),
            jnp.asarray(k_init, _pik_dtype),
            jnp.asarray(eta_init, _pik_dtype),
            lp.A @ initial_solution.primal,
            jnp.concatenate([initial_solution.dual_eq, initial_solution.dual_ineq]),
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

    # Rolling window of recent objective values for the opt-in primal_stop rule
    # (oldest first). Seeded with inf so the window is not "full" until enough
    # real epochs have elapsed.
    obj_window = jnp.full((max(int(primal_stop_window), 1),), jnp.inf)

    # Adaptive-k state. k carries across epochs; on each epoch the controller
    # nudges the *ratio* (k) while `base_lr` sets the *magnitude*. The injected
    # learning rates are rewritten as (base_lr(t)/k, base_lr(t)*k), decoupling
    # ratio-adaptation from magnitude-decay. `base_lr` may be a constant or a
    # callable of the epoch count for an annealing schedule.
    def base_lr_at(epoch):
        return base_lr(epoch) if callable(base_lr) else base_lr

    k = jnp.asarray(k_init)
    if adaptive_k:
        lr0 = base_lr_at(0)
        opt_state = set_saddle_lrs(opt_state, lr0 / k, lr0 * k)

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
                per_iterate_k=per_iterate_k,
                per_iterate_k_theta=per_iterate_k_theta,
                per_iterate_k_lo=per_iterate_k_lo,
                per_iterate_k_hi=per_iterate_k_hi,
                k_init=k_init,
                per_iterate_eta=per_iterate_eta,
                per_iterate_eta_theta=per_iterate_eta_theta,
                per_iterate_eta_lo=per_iterate_eta_lo,
                per_iterate_eta_hi=per_iterate_eta_hi,
                eta_init=eta_init,
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
                constraint_residual_scaled,
            ) = jax.lax.cond(
                average == "exponential" or average == "polyak",
                lambda: compute_epoch_metrics(average_state),
                lambda: compute_epoch_metrics(state),
            )

            finish_epoch_time = time.time()
            count += 1

            # Roll the latest objective into the primal_stop window (oldest first).
            obj_window = jnp.concatenate(
                [obj_window[1:], jnp.reshape(objective_value, (1,))]
            )

            # --- Adaptive k (primal/dual step-ratio) update ---
            # Run on the outer (per-epoch) cadence the damping assumes. Only the
            # learning-rate *values* in opt_state are overwritten, so the jitted
            # epoch loop is not retraced.
            if adaptive_k:
                # primal_feas (‖Ax−b‖, scaled) is resolved by the dual → drives
                # k up; dual_feas (‖c+Aᵀy−z‖, already scaled) is resolved by the
                # primal → drives k down.
                k = update_k(
                    k,
                    primal_feas=constraint_residual_scaled,
                    dual_feas=dual_feasibility_residual,
                    theta=k_theta,
                    lo=k_lo,
                    hi=k_hi,
                    deadband=k_deadband,
                )
                lr_t = base_lr_at(count)
                opt_state = set_saddle_lrs(opt_state, lr_t / k, lr_t * k)
                if verbose and (count == 1 or count % log_every == 0):
                    at_rail = bool((k <= k_lo + 1e-6) | (k >= k_hi - 1e-6))
                    print(
                        f"  adaptive k={float(k):.3f} base_lr={lr_t:.2e}"
                        f" (τ={lr_t / float(k):.2e}, σ={lr_t * float(k):.2e})"
                        + (
                            "  [AT CLAMP — k cannot fix it; see EG/optimism]"
                            if at_rail
                            else ""
                        )
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

                # Per-iterate k/eta carry in the packed opt_state as
                # (optax_state, k, eta, prev_Ax, prev_dual); surface the values
                # the controller settled on at the end of this epoch.
                if per_iterate_k:
                    print(
                        f"  per-iterate k={float(opt_state[1]):.3f}"
                        f" eta={float(opt_state[2]):.3e}"
                        f" (τ={float(opt_state[2]) / float(opt_state[1]):.2e},"
                        f" σ={float(opt_state[2]) * float(opt_state[1]):.2e})"
                    )
                    print("----------------------------------------------")

                print_expert_weights(count, opt_state)

            # --- Adaptive restart decision ---
            if restarts and restarts_done < restarts:
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
                    elif per_iterate_k:
                        # Reset momentum but carry the learned k and eta; reseed
                        # the previous-iterate caches at the current warm-start.
                        opt_state = (
                            optimiser.init(state),
                            opt_state[1],
                            opt_state[2],
                            lp.A @ state.primal,
                            jnp.concatenate([state.dual_eq, state.dual_ineq]),
                        )
                    else:
                        opt_state = optimiser.init(state)
                    # optimiser.init resets the injected learning rates to their
                    # construction values; restore the current adaptive k at the
                    # current point on the magnitude-decay schedule.
                    if adaptive_k:
                        lr_t = base_lr_at(count)
                        opt_state = set_saddle_lrs(opt_state, lr_t / k, lr_t * k)
                    average_state = state
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
                        print(
                            f"Restart {restarts_done}/{restarts} at epoch {count} "
                            f"({reason}, merit={float(restart_merit):.2e} "
                            f"[{which}], next cap={current_cycle_cap:.0f} epochs, "
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
    print("-----------------------------------------------")


# %%


def __convert_to_scipy(jsp_mat: jsp.BCOO) -> sp.csc_matrix:
    data = np.array(jsp_mat.data)
    indices = np.array(jsp_mat.indices)
    row, col = indices[:, 0], indices[:, 1]

    return sp.csc_matrix((data, (row, col)), shape=jsp_mat.shape)


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


def project_onto_eq(lp: LP, primal: jnp.ndarray, tol: 1e-6) -> jnp.ndarray:
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
    residual = lp.b_eq - lp.A_eq @ primal

    def matvec(v):
        return lp.A_eq_T @ (lp.A_eq @ v)

    delta, info = gmres(matvec, lp.A_eq_T @ residual, tol=tol)

    if info != 0:
        print(f"GMRES did not converge (info={info})")

    return primal + delta


# %%
