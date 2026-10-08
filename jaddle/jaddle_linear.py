# %%
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsp
from optax.projections import projection_non_negative, projection_box
import optax
import numpy as np
import copy
import functools
from typing import NamedTuple
import time
from scipy import sparse as sp
from jaddle.jaddle_basic_types import LP, JaddleLP, SaddleState
from scipy.sparse.linalg import gmres
from jax.scipy.sparse.linalg import gmres
import jaddle.jaddle_optimisers as jo

np.set_printoptions(precision=2, suppress=True)


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
    initial_solution,
    initial_avg_state=None,
    initial_opt_state=None,
    total_weight=0.0,
    primal_damping=0.0,
    dual_damping_ineq=0.0,
    dual_damping_eq=0.0,
    average=True,
    update_mode="pdhg",
    k_init=1.0,
    adaptive_eta=1.0,
    merit_fn=None,
    check_every=None,
    merit_threshold=-jnp.inf,
    stall_threshold=-jnp.inf,
    epoch_cache=None,
):
    # `epoch_cache` is a dict owned by the caller (one per `solve` call) that
    # memoises the jitted epoch runner across epochs. It must NOT be module-global:
    # each runner closes over `lp` and `merit_fn`, so a global cache pins every
    # solved LP's scaled matrices and compiled executables for the life of the
    # process (~250 MB host + GPU per MIPLIB instance; OOMs the benchmark sweep).
    #
    # In-epoch restart check: when `merit_fn` and `check_every` are given, the
    # epoch runs in chunks of `check_every` iterations and exits early once
    # min(merit_fn(average), merit_fn(state)) <= `merit_threshold` (the caller's
    # sufficient-progress threshold), so `solve`'s restart fires without waiting
    # for the epoch boundary. It also exits on cuPDLP's condition (ii): merit <=
    # `stall_threshold` (necessary decay) AND rising vs the previous chunk; the
    # returned `stalled` flag tells `solve` to restart on it. The epoch length
    # rounds down to a whole number of chunks.
    #
    # Every mode is cuPDLP-style adaptive PDHG: a single scalar base step eta is
    # line-searched every iteration (a trial step is taken, the largest
    # admissible step eta_bar = move / (2|interaction|) is formed from the trial
    # movement, and the step is rejected + shrunk if it overshot eta_bar). The
    # primal/dual steps are tau=eta/k, sigma=eta*k, with k the primal weight
    # (rebalanced at restarts in `solve`). The modes differ only in the dual's
    # extrapolation and an optional anchor:
    #   * "pdhg":        dual reads x_bar = 2 x_new - x_old (theta=1).
    #   * "alternating": dual reads x_new (theta=0, Gauss-Seidel). Not
    #                    contractive in general; relies on averaging/restarts.
    #   * "halpern":     pdhg operator T(z), anchored back toward z_0 (the
    #                    iterate at the start of the restart cycle):
    #                        z_{k+1} = lambda_k z_0 + (1 - lambda_k) T(z_k),
    #                    lambda_k = 1/(k+1) with the restart-shifted index. The
    #                    convex combination of two feasible iterates stays
    #                    feasible, so no re-projection is needed.
    halpern = update_mode == "halpern"
    theta = 0.0 if update_mode == "alternating" else 1.0

    def projection_primal(primal_state):
        return projection_box(primal_state, lp.lower_bounds, lp.upper_bounds)

    def grad_primal_only(state):
        # Primal partial: c + Aᵀd (+ damping). One sparse matvec (Aᵀ @ d).
        dual = jnp.concatenate([state.dual_eq, state.dual_ineq])
        ATd = lp.A_T @ dual
        return lp.c + ATd + primal_damping * state.primal

    def grad_dual_only_from_Ax(Ax, state):
        # Dual partials b - Ax (+ damping) from a pre-computed A @ x.
        residual = lp.b - Ax
        grad_dual_eq = residual[: lp.n_eq] + dual_damping_eq * state.dual_eq
        grad_dual_ineq = residual[lp.n_eq :] + dual_damping_ineq * state.dual_ineq
        return grad_dual_ineq, grad_dual_eq

    # opt_state is the step-size state: (k, eta) for pdhg/alternating and
    # (k, eta, anchor) for halpern. Inside the epoch it is extended with a
    # carried A @ x so each iteration reuses last iteration's A @ x_new instead
    # of recomputing it — a 3->2 matvec cut. pdhg/alternating carry
    # (k, eta, Ax). Halpern's anchor blend changes the primal after the matvec,
    # but A is linear, so the blended iterate's matvec is one axpy away:
    #     A @ (lam z_0 + (1-lam) cand) = lam (A @ z_0) + (1-lam) Ax_new.
    # The anchor z_0 is constant within an epoch (restarts / re-anchors only
    # fire at epoch boundaries in `solve`), so A @ z_0 is loop-invariant and
    # halpern carries (k, eta, anchor, Ax_anchor, Ax_state).
    cache_key = (
        id(lp),
        float(primal_damping),
        float(dual_damping_ineq),
        float(dual_damping_eq),
        average,
        update_mode,
        id(merit_fn),
        check_every,
    )
    if epoch_cache is None:
        epoch_cache = {}
    run_epoch = epoch_cache.get(cache_key)

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
            merit_threshold=-jnp.inf,
            stall_threshold=-jnp.inf,
            *,
            max_iter,
        ):
            def _descent_bound(state, cand, k, interaction):
                # eta_bar = move / (2 |interaction|), move = k‖dx‖² + (1/k)‖dy‖².
                dx = cand.primal - state.primal
                dy_eq = cand.dual_eq - state.dual_eq
                dy_ineq = cand.dual_ineq - state.dual_ineq
                move = k * jnp.vdot(dx, dx) + (1.0 / k) * (
                    jnp.vdot(dy_eq, dy_eq) + jnp.vdot(dy_ineq, dy_ineq)
                )
                # No movement => any step is fine (avoid 0/0); flag with +inf so
                # the retry loop accepts and the eta-growth branch is suppressed.
                return jnp.where(interaction > 0.0, move / (2.0 * interaction), jnp.inf)

            def trial(eta, state, k, Ax_old, gp):
                # One raw PDHG step at base step eta. `gp` = c + Aᵀy at state
                # (eta-independent, computed once per iteration so line-search
                # retries don't redo its matvec). Ax_old = A @ state.primal is
                # carried; only Ax_new = A @ x_new is computed here, and the
                # dual reads x_bar = x_new + theta (x_new - x_old), so
                # A @ x_bar = Ax_old + (1 + theta) A_dx.
                tau = eta / k
                sigma = eta * k
                x_new = projection_primal(state.primal - tau * gp)
                Ax_new = lp.A @ x_new
                A_dx = Ax_new - Ax_old
                Ax_bar = Ax_old + (1.0 + theta) * A_dx
                gd_ineq, gd_eq = grad_dual_only_from_Ax(Ax_bar, state)
                dual_ineq = projection_non_negative(state.dual_ineq - sigma * gd_ineq)
                dual_eq = state.dual_eq - sigma * gd_eq
                cand = SaddleState(primal=x_new, dual_ineq=dual_ineq, dual_eq=dual_eq)
                dy = jnp.concatenate(
                    [dual_eq - state.dual_eq, dual_ineq - state.dual_ineq]
                )
                interaction = jnp.abs(jnp.vdot(dy, A_dx))
                return cand, _descent_bound(state, cand, k, interaction), Ax_new

            def step(carry, _):
                i, state, average_state, opt_state, total_weight = carry
                if halpern:
                    k, eta, anchor, Ax_anchor, Ax_old = opt_state
                else:
                    k, eta, Ax_old = opt_state
                ip1 = jnp.asarray(i + 1, eta.dtype)

                # Retry: while the trial step exceeds its admissible bound,
                # shrink eta to just under eta_bar and re-trial. The
                # (1-(i+1)^-0.3) factor < 1 guarantees strict decrease, so the
                # loop terminates. Carry (eta, cand, eta_bar, Ax_new).
                def cond(c):
                    eta_c, _, eta_bar_c, _ = c
                    return eta_c > eta_bar_c

                def body(c):
                    eta_c, _, eta_bar_c, _ = c
                    eta_s = jnp.minimum((1.0 - ip1 ** (-0.3)) * eta_bar_c, eta_c)
                    cand_s, eta_bar_s, Ax_new_s = trial(eta_s, state, k, Ax_old, gp)
                    return (eta_s, cand_s, eta_bar_s, Ax_new_s)

                gp = grad_primal_only(state)
                cand0, eta_bar0, Ax_new0 = trial(eta, state, k, Ax_old, gp)
                eta0, cand, eta_bar, Ax_new = jax.lax.while_loop(
                    cond, body, (eta, cand0, eta_bar0, Ax_new0)
                )

                if halpern:
                    # Halpern anchor: blend T(z_k)=cand back toward z_0.
                    # lambda_k = 1/(k_local+1) where k_local is the
                    # restart-shifted iteration index `i` (== i_global -
                    # restart_i_offset, reset to ~1 each cycle by `solve`), so
                    # lambda decays as the cycle progresses and re-warms toward
                    # 1/2 at each restart.
                    k_local = jnp.asarray(i, eta.dtype)
                    lam = 1.0 / (k_local + 1.0)
                    new_state = jax.tree.map(
                        lambda z0, tz: lam * z0 + (1.0 - lam) * tz,
                        anchor,
                        cand,
                    )
                    # Advance the matvec carry through the blend by linearity
                    # of A: A @ new_primal = lam Ax_anchor + (1-lam) Ax_new.
                    Ax_next = lam * Ax_anchor + (1.0 - lam) * Ax_new
                else:
                    new_state = cand
                    Ax_next = Ax_new

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
                    opt_state = (k, eta_next, anchor, Ax_anchor, Ax_next)
                else:
                    opt_state = (k, eta_next, Ax_next)

                if average:
                    # PDLP-style step-size-weighted average (Applegate et al.,
                    # "Practical LP using PDHG", the restart-average
                    # z̄ⁿ = Σ ηₖ zᵏ / Σ ηₖ): weight each iterate by the eta
                    # ACTUALLY used to produce it (eta0, the accepted
                    # post-retry-shrink step).
                    w = eta0
                    total_weight = total_weight + w
                    average_state = optax.incremental_update(
                        new_state, average_state, w / total_weight
                    )

                return (i + 1, new_state, average_state, opt_state, total_weight)

            # Fixed iteration count per epoch: lax.scan (static `max_iter`) lets
            # XLA pipeline the loop body better than a while_loop whose only exit
            # condition is `i < end_iter`. `max_iter` is a static_argname, so a
            # changing iterations_per_epoch (restart decay) triggers a recompile.
            #
            # The caller-facing opt_state is (k, eta[, anchor]); seed the Ax
            # carry here (one or two matvecs/epoch — negligible) and strip it
            # from the returned opt_state. Re-seeding every epoch also resets any
            # rounding drift the blended carry accumulated within the prior epoch.
            if halpern:
                k0, eta0, anchor0 = opt_state
                opt_state = (
                    k0,
                    eta0,
                    anchor0,
                    lp.A @ anchor0.primal,
                    lp.A @ state.primal,
                )
            else:
                k0, eta0 = opt_state
                opt_state = (k0, eta0, lp.A @ state.primal)

            step_carry = (start_iter, state, average_state, opt_state, total_weight)
            # scan requires the carry's output dtypes to match its input dtypes
            # exactly (e.g. a Python-int start_iter); cast each step's output
            # carry back to the input carry's dtypes so the loop is type-stable.
            _carry_dtypes = jax.tree.map(lambda x: jnp.asarray(x).dtype, step_carry)

            def step_typed(carry, _):
                new_carry = step(carry, _)
                new_carry = jax.tree.map(
                    lambda v, dt: v.astype(dt), new_carry, _carry_dtypes
                )
                return new_carry, None

            stalled = jnp.asarray(False)
            if merit_fn is None or check_every is None or check_every >= max_iter:
                scan_out, _ = jax.lax.scan(
                    step_typed,
                    step_carry,
                    None,
                    length=max_iter,
                )
            else:
                n_chunks = max_iter // check_every

                def chunk_cond(c):
                    chunk, _, done, _, _ = c
                    return (chunk < n_chunks) & (~done)

                def chunk_body(c):
                    chunk, carry, _, _, prev_m = c
                    carry, _ = jax.lax.scan(step_typed, carry, None, length=check_every)
                    _, s, avg, _, _ = carry
                    m = merit_fn(s)
                    if average:
                        m = jnp.minimum(m, merit_fn(avg))
                    # Condition (ii) needs a previous chunk in this epoch, so the
                    # first chunk (prev_m = inf) can only fire condition (i).
                    stalled = (
                        (m <= stall_threshold) & jnp.isfinite(prev_m) & (m > prev_m)
                    )
                    done = (m <= merit_threshold) | stalled
                    return chunk + 1, carry, done, stalled, m

                _, scan_out, _, stalled, _ = jax.lax.while_loop(
                    chunk_cond,
                    chunk_body,
                    (
                        jnp.asarray(0),
                        step_carry,
                        jnp.asarray(False),
                        jnp.asarray(False),
                        jnp.asarray(jnp.inf, state.primal.dtype),
                    ),
                )
            i, state, average_state, opt_state, total_weight = scan_out

            if halpern:
                opt_state = opt_state[:3]
            else:
                opt_state = opt_state[:2]

            return i, state, average_state, opt_state, total_weight, stalled

        epoch_cache[cache_key] = run_epoch

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
    else:
        dtype = initial_solution.primal.dtype
        opt_state = (jnp.asarray(k_init, dtype), jnp.asarray(adaptive_eta, dtype))
        if halpern:
            # On a bare __sps call the anchor seeds from the incoming state;
            # across a restart cycle `solve` threads it via opt_state.
            opt_state = opt_state + (jax.tree.map(lambda x: x + 0, initial_solution),)

    # run_epoch DONATES `state` and `opt_state`. The caller's restart path may
    # hand us a `state`/`opt_state` pair that shares buffers (e.g. `state =
    # restart_point` aliasing the halpern anchor). Donating two args that alias
    # one buffer double-frees it, so break any aliasing with an independent copy
    # of each donated tree — one cheap pass per epoch, off the hot path.
    state = jax.tree.map(lambda x: x + 0, state)
    opt_state = jax.tree.map(jnp.copy, opt_state)

    return run_epoch(
        start_iter,
        state,
        average_state,
        opt_state,
        total_weight,
        merit_threshold,
        stall_threshold,
        max_iter=max_iter,
    )


def _vector_norm(v, norm):
    """‖v‖∞ (``norm="inf"``) or ‖v‖₂ (``"l2"``); 0 for an empty vector."""
    if norm == "l2":
        return jnp.linalg.norm(v)
    return jnp.max(jnp.abs(v), initial=0.0)


def solve(
    lp: JaddleLP,
    max_epochs=None,
    max_seconds=None,
    initial_solution=None,
    initial_opt_state=None,
    iterations_per_epoch=256,
    dual_damping_ineq=0.0,
    dual_damping_eq=0.0,
    primal_damping=0.0,
    primal_feasibility_tolerance=1e-3,
    dual_feasibility_tolerance=1e-3,
    dual_gap_tolerance=1e-3,
    dual_residual="pdlp",
    termination_norm="inf",
    restart_norm="inf",
    verbose=False,
    log_every=1,
    average=True,
    report_best=True,
    update_mode="pdhg",
    k_scale=1e8,
    k_theta=0.5,
    k_init=None,
    k_update_per_epoch=False,
    adaptive_eta=0.0,
    scale=True,
    scaled_objective=True,
    scaled_rhs=True,
    scaled_augmented=True,
    augmented_weight=1.0,
    ruiz_iterations=10,
    pc_iterations=1,
    cost_col_floor=1e-2,
    restarts=False,
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
    restart_check_every="auto",
    eq_projection_threshold=None,
    vertex_bias=0.0,
    vertex_bias_seed=0,
    reference_objective=None,
):
    """
    Solve a linear program via saddle-point optimisation.

    Termination uses the standard LP optimality certificate, all tested
    RELATIVELY (PDLP/HiGHS convention): primal feasibility
    (``primal_feasibility_tolerance``, normalised by 1+‖b‖), dual feasibility
    (``dual_feasibility_tolerance``, normalised by 1+‖c‖), and a finite duality
    gap within ``dual_gap_tolerance`` (normalised by 1+|primal_obj|+|dual_obj|,
    the PDLP/cuPDLP convention, so RDG is directly comparable to PDLP).

    Every update_mode is cuPDLP-style adaptive PDHG: plain projected
    primal-descent / dual-ascent steps with a per-iteration line-searched step
    size ``eta`` and a primal weight ``k`` (no optimiser plug-in).

    Adaptive restarts (PDLP-style) accelerate ill-conditioned problems. A
    restart resets the averaging (and the halpern anchor) while keeping the current
    iterate as a warm start, which prevents the saddle iteration from settling
    into slow rotational orbits. A restart fires when either the normalised KKT
    merit decays past ``restart_decay`` of its value at the last restart
    (sufficient-progress restart) or the current cycle reaches its length cap
    (no-progress restart). Enable with ``restarts=True``.

    Args:
        max_seconds: Wall-clock budget in seconds (default ``None`` = no limit).
            Measured from entry into ``solve()``, so scaling / setup and the
            first-epoch XLA compile count against it. Checked at each epoch
            boundary: once the budget is spent no further epoch starts and the
            current point is returned with ``stop_reason="time_limit"``, so the
            solve can overrun by up to one epoch (shrink
            ``iterations_per_epoch`` for a tighter cutoff).
        dual_residual: How reduced costs split between the dual residual (DFR)
            and the dual objective (hence the gap). ``"pdlp"`` (default) follows
            PDLP with ``handle_some_primal_gradients_on_finite_bounds_as_residuals``:
            rᵢ > 0 is absorbed by a finite lower bound and rᵢ < 0 by a finite
            upper bound when xᵢ is near it (|xᵢ - bound| <= |xᵢ|), adding rᵢ·bound
            to the dual objective; every other component is a residual |rᵢ|.
            Complementarity errors on boxed variables therefore show in the gap,
            and far finite bounds (e.g. [0, 1e6] boxes) can't swamp the dual
            objective. ``"projected"`` is the older per-variable projected-gradient
            residual |x - proj(x - r)| for boxed variables, with every finite bound
            always in the dual objective.
        termination_norm: Norm used for the primal / dual feasibility
            stopping tests (and the printed PFR / DFR): ``"inf"`` (default)
            tests ‖r_p‖∞/(1+‖b‖∞) and ‖r_d‖∞/(1+‖c‖∞); ``"l2"`` tests
            ‖r_p‖₂/(1+‖b‖₂) and ‖r_d‖₂/(1+‖c‖₂), the cuPDLP-C / PDLP default,
            for like-for-like comparisons. Termination only: the restart merit
            follows ``restart_norm``, so changing this alone leaves the iterate
            trajectory unchanged and only moves the epoch at which it stops.
        restart_norm: Norm (``"inf"`` default, or ``"l2"``) for the primal /
            dual feasibility terms of the restart KKT merit and the
            dual-infeasible ``still_improving`` cycle-cap guard, normalised by
            1+‖b‖ / 1+‖c‖ in the same norm. ``"l2"`` matches cuPDLP-C's restart
            criterion. Unlike ``termination_norm`` this changes the trajectory.
        restarts: Enable adaptive warm restarts (default ``False``). There is
            no cap on how many fire; the triggers alone decide. Each restart resets the averaging (and the halpern anchor / lambda
            counter) while keeping the current iterate as a warm start.
        epochs_per_restart: Length cap of the first restart cycle, expressed in
            epochs AT THE DEFAULT ``iterations_per_epoch`` (default 10) but
            internally converted to and tracked in ITERATIONS
            (``epochs_per_restart * iterations_per_epoch``), so the cycle-cap
            restart fires at the same point in the optimisation trajectory
            regardless of ``iterations_per_epoch``. This matters because a
            restart is destructive (it wipes the PDHG averaging): tying the cap to a raw epoch count made the
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
            * ``"pdhg"`` (default): Chambolle–Pock PDHG (primal step then dual
              step on the extrapolated primal x_bar = 2x^{k+1} − x^k).
            * ``"alternating"``: primal step, then a dual step at the new primal
              x^{k+1} (Gauss–Seidel, no extrapolation). Not contractive in
              general; relies on averaging/restarts. Experimental.
            * ``"halpern"``: restarted Halpern-anchored PDHG. Each iterate is the
              adaptive PDHG step T(z) blended back toward an anchor z_0:
              ``z_{k+1} = lambda_k z_0 + (1−lambda_k) T(z_k)``, ``lambda_k =
              1/(k+1)`` (cycle-local k). The anchor z_0 and lambda counter reset
              to the current iterate at each restart, giving last-iterate
              acceleration. Best paired with ``restarts=True``.
        k_scale: Clamp band ``[1/k_scale, k_scale]`` for the primal weight ``k``
            (default ``1e8``); ``None`` leaves ``k`` unclamped. The primal and
            dual steps are ``eta / k`` and ``eta * k``, so the dual/primal step
            ratio is ``k**2``. ``k`` is initialised from ``k_init`` and rebalanced
            at each restart (PDLP-style) from primal-vs-dual iterate movement; it
            is constant within an epoch (not adapted per iteration). Tuned by
            ``k_theta``/``k_scale`` and ``k_init``.
        k_init: Initial primal weight ``k``. ``None`` (default) initialises it to
            the PDLP heuristic ``||c|| / ||b||`` (objective vs RHS norms, in the
            scaled space the solver iterates in). Pass a float to override
            (``1.0`` = symmetric steps, the PC/Ruiz-scaled baseline).
        k_update_per_epoch: When ``True``, the primal weight ``k`` is
            rebalanced at every epoch boundary (not only at restarts) using the
            primal-vs-dual iterate movement over the just-finished epoch — same
            log-space geometric-mean blend (``k_theta``) and ``[1/k_scale,
            k_scale]`` clamp as the restart rebalance. Unlike a restart it does
            NOT reset averaging or (for halpern) re-anchor ``z_0`` / reset
            ``eta``; only ``k`` changes, so the step split tracks the local
            primal/dual progress within a restart cycle.
            ``False`` (default, the PDLP convention) keeps k frozen between
            restarts: under a wide ``k_scale`` clamp the per-epoch update let k
            run away on mzzv11 (k→1e8, gap 0.94).
        adaptive_eta: Seed for the cuPDLP-style per-iteration adaptive step
            ``eta``, which drives the primal step ``tau = eta / k`` and dual step
            ``sigma = eta * k``. Each iteration takes a trial step, forms the
            largest admissible step from the interaction term
            ``(y^{k+1}-y^k)ᵀ A (x^{k+1}-x^k)``, and rejects + shrinks ``eta`` if
            the trial overshot; ``eta`` is then advanced with a two-sided guard.
            ``0.0`` (default) seeds it at ``1/||[[A,b],[c,0]]||_2``. The learned
            ``eta`` is carried across restarts.
        k_theta: Smoothing coefficient for the log-space primal-weight update at
            each restart / epoch. A float fixes it (PDLP uses 0.5; smaller =
            slower adaptation).
            ``"adaptive"`` (default) sets it from data as a trust region on log k:
            starting at 0.5, each restart doubles theta (capped at 1) if the
            restart merit fell over the cycle since the previous k move, else
            halves it (floored at 0.05) and reverts k to its value before that
            move. The movement ratio alone can't tell a correct k move from a
            runaway (mzzv11's per-epoch runaway was monotone), but the merit can.
            Epochs to certify, restart-only vs fixed 0.5: barwon 41 vs 166,
            binschedule2 39 vs 61, plus gains on stp3d and mzzv11.        primal_stop: Opt-in, dual-free termination (default ``False``). When
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
        cost_col_floor: Scaling floor for cost-pinned columns: a costed column
            whose largest scaled matrix entry is below it is rescaled so that
            entry becomes 1. Fixes epigraph objectives (``min z, z >= a_k.x``
            over dense rows, fhnw-binschedule0) where augmented Ruiz leaves the
            variable huge in scaled units. Default ``1e-2``; ``0`` disables. See
            ``scale_problem``.
        eq_projection_threshold: When set, after each epoch the unscaled equality
            residual is checked; if it exceeds this value the primal (and average)
            are projected onto the equality manifold ``A_eq x = b_eq`` via the
            precomputed factorisation of ``A_eq A_eq^T``. Default ``None``
            disables projection. Only useful when equality feasibility is the
            bottleneck; has no effect when there are no equality constraints.
        restart_check_every: Evaluate the restart merit every this many
            iterations inside an epoch and end the epoch early once the
            sufficient-progress test (``restart_decay``) would fire, so restarts
            aren't delayed to the epoch boundary. Costs ~2 matvec pairs per check.
            Also exits on the condition-(ii) stall (merit within
            ``necessary_decay`` of the last restart and rising chunk-to-chunk).
            Epoch lengths round down to a multiple of it. Default ``"auto"``
            (a tenth of the current epoch length); ``None`` checks only at epoch
            boundaries.
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

    Returns:
        dict: The solution together with diagnostics. Keys:
            * ``"solution"``: the ``SaddleState`` (primal/dual iterate), unscaled
              back to the original problem's units.
            * ``"converged"``: ``bool``, whether the solve terminated by meeting
              the LP optimality certificate or a ``primal_stop`` heuristic stop
              (see ``"stop_reason"`` to disambiguate).
            * ``"opt_state"``: the final step-size state ``(k, eta)`` (plus the
              anchor for halpern), for warm-starting a subsequent solve via
              ``initial_opt_state``.
            * ``"stop_reason"``: ``str`` recording *why* the solve terminated:
              ``"certificate"`` (full LP optimality certificate met),
              ``"primal_stall"`` (the ``primal_stop`` heuristic fired — feasible
              but not certified optimal, so the objective may be suboptimal even
              though ``"converged"`` is ``True``), ``"max_epochs"`` (epoch budget
              exhausted), ``"time_limit"`` (``max_seconds`` exhausted), or
              ``"interrupted"`` (KeyboardInterrupt).
            * ``"solve_seconds"``: ``float`` wall time of the epoch loop (incl. the
              first-epoch XLA compile but not the scaling / sparse-setup phase).
            * ``"corrected_seconds"``: ``float`` steady-state runtime with the
              one-off first-epoch XLA compile amortised out:
              ``n * (solve_seconds - first_epoch_seconds) / (n - 1)`` where ``n``
              is the epoch count. Falls back to ``solve_seconds`` when it can't be
              formed (fewer than two epochs).
            * ``"epochs"``: ``int``, number of epochs run.
    """

    # max_seconds is a wall-clock budget for the whole call, setup included.
    solve_entry_time = time.time()
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be > 0 (or None for no limit)")

    lp = __pad_empty_blocks(lp)

    if log_every < 1:
        raise ValueError("log_every must be >= 1")
    if dual_residual not in ("pdlp", "projected"):
        raise ValueError(
            f"dual_residual must be 'pdlp' or 'projected', got {dual_residual!r}"
        )
    if termination_norm not in ("inf", "l2"):
        raise ValueError(
            f"termination_norm must be 'inf' or 'l2', got {termination_norm!r}"
        )
    if restart_norm not in ("inf", "l2"):
        raise ValueError(f"restart_norm must be 'inf' or 'l2', got {restart_norm!r}")

    if verbose:
        print("----------------------------------------------")

    valid_update_modes = ["pdhg", "alternating", "halpern"]
    if update_mode not in valid_update_modes:
        raise ValueError(f"update_mode must be one of {valid_update_modes}")

    # ``k_scale`` sets the clamp band ``[1/k_scale, k_scale]`` for the primal
    # weight k; ``None`` leaves k unclamped.
    if k_scale is not None:
        k_lo, k_hi = 1.0 / k_scale, k_scale
    else:
        k_lo, k_hi = 0.0, np.inf
    if adaptive_eta is None or adaptive_eta < 0.0:
        raise ValueError("adaptive_eta must be a float >= 0 (0.0 = auto seed)")
    halpern = update_mode == "halpern"

    if verbose:
        print("====Starting Solve====")
        print("----------------------------------------------")

    # solve() takes a JaddleLP (the JAX-native, device-side representation) or a
    # raw scipy LP (e.g. hand-built in examples). scale_problem() accepts either
    # and runs on device, returning the JaddleLP the solver iterates on.
    if scale:
        # Augmented Ruiz: equilibrate [[A,b],[c,0]] so cost and RHS information also
        # drive the equilibration. Conditions the constraint (esp. equality) block
        # better on cost/RHS-dominated problems (momentum1: A-only Ruiz froze the
        # primal at a far-from-optimal point; augmented converges in ~15 epochs).
        # A-only Ruiz (scale_problem(augmented=False)) is available as a knob but is
        # not the default — it broke both momentum1 and boeing once the relative
        # convergence test + true-units norm fixes were in place. PC then applies
        # its single Pock-Chambolle finishing pass.
        lp, row_scale, col_scale, c_max = scale_problem(
            lp,
            scaled_objective=scaled_objective,
            scaled_rhs=scaled_rhs,
            augmented=scaled_augmented,
            augmented_weight=augmented_weight,
            ruiz_iter=ruiz_iterations,
            pc_iter=pc_iterations,
            cost_col_floor=cost_col_floor,
        )

        if verbose:
            print("Applied combined Ruiz + PC scaling to the LP.")
            print("----------------------------------------------")

    else:
        row_scale = np.ones(lp.A_eq.shape[0] + lp.A_ineq.shape[0])
        col_scale = np.ones(lp.c.shape[0])
        c_max = 1.0
        # solve() reassigns lp.c below (vertex_bias, c_max unscaling), so never
        # iterate on the caller's JaddleLP itself. Its arrays are immutable, so a
        # shallow copy suffices.
        lp = copy.copy(lp) if isinstance(lp, JaddleLP) else to_jaddle_sparse(lp)

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

        # Primal residual in a given norm. `constraint_bound` (restart_norm)
        # feeds only the restart merit; `termination_pfr` (termination_norm)
        # feeds the stopping test and printed PFR. When both norms match, XLA
        # CSEs the duplicate reduction.
        def primal_residual(norm):
            if norm == "l2":
                return jnp.sqrt(
                    jnp.sum(ineq_violations**2) + jnp.sum(eq_violations**2)
                )
            return jnp.maximum(max_ineq_violation, max_eq_violation)

        constraint_bound = primal_residual(restart_norm)
        termination_pfr = primal_residual(termination_norm)

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

        lower_term = reduced_cost * lower_bounds
        upper_term = reduced_cost * upper_bounds

        if dual_residual == "pdlp":
            # PDLP convention (handle_some_primal_gradients_on_finite_bounds_as_
            # residuals): a positive reduced cost is absorbed by a finite lower
            # bound, a negative one by a finite upper bound, but only when the
            # primal sits NEAR that bound (|x - bound| <= |x|). An absorbed rᵢ
            # contributes rᵢ·boundᵢ to the dual objective and nothing to the dual
            # residual; anything else is a dual residual |rᵢ| and contributes
            # nothing to the dual objective. Without the "near" test a far finite
            # bound (wairoa's [0, 1e6] boxes) multiplies a tiny reduced-cost error
            # by 1e6 in the dual objective and the gap never closes. The test is
            # scale-invariant (x and the bounds share col_scale), so the scaled
            # masks serve the true-unit residual below too.
            x_s = average_state.primal
            pos, neg = reduced_cost > 0.0, reduced_cost < 0.0
            lb_ok = finite_lower & (jnp.abs(x_s - lower_bounds) <= jnp.abs(x_s))
            ub_ok = finite_upper & (jnp.abs(x_s - upper_bounds) <= jnp.abs(x_s))
            absorb_lower = pos & lb_ok
            absorb_upper = neg & ub_ok
            box_infimum = jnp.where(
                absorb_lower, lower_term, jnp.where(absorb_upper, upper_term, 0.0)
            )
            dual_feasibility_violation = jnp.where(
                absorb_lower | absorb_upper, 0.0, jnp.abs(reduced_cost_true)
            )
            dual_feasibility_residual = _vector_norm(
                dual_feasibility_violation, restart_norm
            )
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
                termination_pfr,
                _vector_norm(dual_feasibility_violation, termination_norm),
            )

        # box_infimum = inf over the box of rᵢ·xᵢ — the dual contribution of each
        # variable's bound. Taken as the finite bound·r product unconditionally,
        # with NO per-term sign guard on the reduced cost: matching PDLP, a small
        # (or cancelling) duality gap is trusted as small regardless of which
        # individual dual multipliers are momentarily the "wrong" sign. Dual
        # feasibility is enforced separately below via `dual_feasibility_residual`
        # / DFR, so a wrong-sign reduced cost is still caught — just not by
        # blocking the gap to +∞. (Bound-class masks are scale-invariant.)
        box_infimum = jnp.where(
            has_both_bounds,
            jnp.minimum(lower_term, upper_term),
            jnp.where(
                has_only_lower,
                lower_term,
                jnp.where(has_only_upper, upper_term, 0.0),
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
        dual_feasibility_residual = _vector_norm(
            dual_feasibility_violation, restart_norm
        )

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
            termination_pfr,
            _vector_norm(dual_feasibility_violation, termination_norm),
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
        # If duality_gap is ever non-finite (nan/inf inputs), report +inf rather
        # than the raw inf/inf = nan ratio, so the log and any |gap|-based
        # comparison read sensibly. Convergence still gates on the separate
        # `dual_gap_is_finite` flag regardless.
        dual_bound = objective_value - duality_gap
        return jnp.where(
            jnp.isfinite(duality_gap),
            jnp.abs(duality_gap)
            / (1.0 + jnp.abs(objective_value) + jnp.abs(dual_bound)),
            jnp.inf,
        )

    def relative_gap_abs(
        objective_value, duality_gap, gap_bound_comp, gap_ineq_comp, gap_eq_comp
    ):
        # No-cancellation companion to `relative_gap`: sum of |component|s instead
        # of the signed sum, so a gap that reads small only through cancellation
        # between e.g. gap_bound_comp and gap_eq_comp (qap10) doesn't pass. Shared
        # by `converged()` (the actual gate) and `print_epoch_metrics` (so the
        # printed diagnostics can't show a converged-looking RDG while this hidden
        # condition is still what's blocking termination).
        dual_bound = objective_value - duality_gap
        gap_denom = 1.0 + jnp.abs(objective_value) + jnp.abs(dual_bound)
        return (
            jnp.abs(gap_bound_comp) + jnp.abs(gap_ineq_comp) + jnp.abs(gap_eq_comp)
        ) / gap_denom

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
        # closure. The residuals passed in, and the termination_*_norm
        # normalisers, are in `termination_norm`.
        relative_primal_residual = constraint_bound / (1.0 + termination_b_norm)
        relative_dual_residual = dual_feasibility_residual / (1.0 + termination_c_norm)

        # No-cancellation guard, and the ONLY gap test needed. `duality_gap` is
        # the SIGNED sum of three complementarity terms (bound, inequality,
        # equality coupling): duality_gap = gap_bound_comp + gap_ineq_comp +
        # gap_eq_comp exactly. At a true KKT point each term is ~0, so the sum is
        # ~0 honestly. But on degenerate problems with large dual multipliers the
        # equality-coupling term `y_eqᵀ(b−A_eq x)` can be large and negative —
        # (large y)·(small residual) — and CANCEL a large positive
        # bound-complementarity term, making the signed sum tiny while the point
        # is nowhere near complementary. qap10 (QAP assignment relaxation)
        # certified ~1.4–2.4% suboptimal this way: gap_bound≈+5.2, gap_eq≈−5.2,
        # sum≈0.04 → RDG 6.5e-5, yet the recovered dual_obj sat BELOW the true
        # optimum (weak-duality violation). Testing the sum of |component|s (not
        # their signed sum) closes this: cancellation can shrink the signed sum
        # but not the absolute sum. By the triangle inequality
        # |duality_gap| <= |gap_bound_comp|+|gap_ineq_comp|+|gap_eq_comp| always
        # (same denominator on both), so `relative_gap(duality_gap, ...)` (the
        # plain, cancellation-prone RDG) is NEVER larger than `gap_abs` — testing
        # gap_abs alone subsumes it; a separate RDG <= tol test can never reject
        # a point gap_abs already accepted.
        gap_abs = relative_gap_abs(
            objective_value, duality_gap, gap_bound_comp, gap_ineq_comp, gap_eq_comp
        )
        return (
            (relative_primal_residual <= primal_feasibility_tolerance)
            & (relative_dual_residual <= dual_feasibility_threshold)
            & dual_gap_is_finite
            & (gap_abs <= dual_gap_tolerance)
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
        relative_primal_residual = constraint_bound / (1.0 + termination_b_norm)
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
    if k_init is None:
        norm_c = float(jnp.linalg.norm(lp.c)) + 1e-30
        norm_b = float(jnp.linalg.norm(lp.b)) + 1e-30
        k_init = float(np.clip(norm_c / norm_b, k_lo, k_hi))

    i = 1
    state = initial_solution
    average_state = initial_solution
    if initial_opt_state is not None:
        opt_state = initial_opt_state
    else:
        # Step-size state (k, eta); halpern also carries the anchor z_0
        # (cycle-start iterate), seeded from the initial solution and reset at
        # each restart below.
        _pik_dtype = initial_solution.primal.dtype
        opt_state = (
            jnp.asarray(k_init, _pik_dtype),
            jnp.asarray(adaptive_eta, _pik_dtype),
        )
        if halpern:
            opt_state = opt_state + (jax.tree.map(lambda x: x + 0, initial_solution),)
    primal_grad_norm = jnp.inf
    complementarity_slack = jnp.inf
    constraint_bound = jnp.inf
    dual_feasibility_residual = jnp.inf
    termination_pfr = jnp.inf
    termination_dfr = jnp.inf
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
    # buffer-donation reason as state_at_last_restart.
    k_per_epoch = bool(k_update_per_epoch)
    state_at_last_epoch = jax.tree.map(lambda x: x + 0, initial_solution)

    # Rolling window of recent objective values for the opt-in primal_stop rule
    # (oldest first). Seeded with inf so the window is not "full" until enough
    # real epochs have elapsed.
    obj_window = jnp.full((max(int(primal_stop_window), 1),), jnp.inf)

    # Adaptive restart bookkeeping. `restart_i_offset` is subtracted from the
    # global iteration counter `i` before it is handed to __sps, so a restart
    # re-zeros the halpern lambda counter and the eta growth schedule without
    # disturbing the running epoch/iteration accounting.
    restarts_done = 0
    restart_i_offset = 0
    epochs_since_restart = 0
    # The cycle-exhaustion cap is tracked in ITERATIONS, not epochs, so that
    # `cycle_exhausted` fires at the same point in the optimisation trajectory
    # regardless of `iterations_per_epoch`. A restart is a destructive reset
    # (it wipes the PDHG averaging), so its cadence must
    # not depend on how the same iteration budget happens to be chopped into
    # epochs. Seed the cap in iterations from the epoch-count knob so the
    # default (epochs_per_restart=10) means the same thing it always has at the
    # default iterations_per_epoch; iterations_since_restart accumulates the
    # ACTUAL per-epoch iteration count, so it stays correct even under
    # iterations_per_epoch_decay.
    iterations_since_restart = 0
    current_cycle_cap_iters = float(epochs_per_restart) * float(iterations_per_epoch)
    merit_at_last_restart = jnp.inf
    # k_theta="adaptive": live smoothing coefficient, and k before the last
    # restart rebalance (the revert target when that move made the merit worse).
    if isinstance(k_theta, str):
        if k_theta != "adaptive":
            raise ValueError(f"k_theta must be a float or 'adaptive', got {k_theta!r}")
        adaptive_theta, theta_live = True, 0.5
    else:
        adaptive_theta, theta_live = False, float(k_theta)
    k_before_last_move = None
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
    # b_norm / c_norm normalise the restart merit (restart_norm); the
    # termination_* pair normalises the stopping test / printed PFR, DFR.
    _b_true = lp.b / _row_scale_all
    _c_true = lp.c / col_scale
    b_norm = float(_vector_norm(_b_true, restart_norm))
    c_norm = float(_vector_norm(_c_true, restart_norm))
    termination_b_norm = float(_vector_norm(_b_true, termination_norm))
    termination_c_norm = float(_vector_norm(_c_true, termination_norm))

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
        # Surface the live primal weight k and adaptive step eta for the epoch
        # trace. opt_state is (k, eta[, anchor]).
        return float(opt_state[0]), float(opt_state[1])

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
        relative_pfr = termination_pfr / (1.0 + termination_b_norm)
        relative_dfr = termination_dfr / (1.0 + termination_c_norm)
        # RDGABS: the no-cancellation companion to RDG (see relative_gap_abs).
        # `converged()` requires BOTH RDG and RDGABS <= dual_gap_tolerance;
        # without printing RDGABS a run can show PFR/DFR/RDG all comfortably
        # inside tolerance yet never certify because RDGABS alone is still open
        # (the gap components are cancelling rather than genuinely small) — this
        # was invisible in the log before RDGABS was added here.
        rdg_abs = relative_gap_abs(
            objective_value, duality_gap, gap_bound_comp, gap_ineq_comp, gap_eq_comp
        )
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
        print(
            f"|Epoch {count}|"
            f"|Obj{objective_value:.2e}|"
            f"|PFR {relative_pfr:.2e}|"
            f"|DFR {relative_dfr:.2e}|"
            f"|RDG {relative_gap(duality_gap, objective_value):.2e}|"
            f"|RDGABS {rdg_abs:.2e}|"
            f"{objerr_str}"
            f"{ke_str}"
            f"{time_str}"
        )
        print("----------------------------------------------")

    def is_done():
        nonlocal stop_reason
        certificate_met = bool(
            converged(
                termination_pfr,
                termination_dfr,
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
        if primal_stop and bool(converged_primal(termination_pfr, obj_window)):
            stop_reason = "primal_stall"
            return True
        return False

    # Per-solve memo of jitted epoch runners (see __sps); freed when solve returns.
    _epoch_cache = {}

    def _inloop_merit(s):
        # Same restart merit `solve` computes at the epoch boundary, traced into
        # __sps for the in-epoch sufficient-progress check.
        obj, _pgn, _cs, cb, dfr, dg, dgf, *_rest = compute_epoch_metrics(s)
        return kkt_merit(cb, dfr, dg, dgf, obj)

    def _check_every():
        # "auto": ten in-epoch checks per epoch (tracks iterations_per_epoch
        # decay). None disables the in-epoch check.
        if restart_check_every == "auto":
            return max(1, current_iterations_per_epoch // 10)
        return restart_check_every

    def _restart_thresholds():
        # (sufficient-progress, necessary-decay) thresholds for the in-epoch
        # check; -inf disables both (no finite baseline yet, restarts off, or
        # the check is off).
        if (
            _check_every() is None
            or not restarts
            or not bool(jnp.isfinite(merit_at_last_restart))
        ):
            return -float("inf"), -float("inf")
        return (
            float(restart_decay * merit_at_last_restart),
            float(necessary_decay * merit_at_last_restart),
        )

    # PDLP-style primal-weight rebalance from the primal-vs-dual *movement*
    # between two iterates (distance, not per-step gradient norms). Shared by the
    # restart rebalance and the per-epoch rebalance: log-space geometric-mean
    # blend of the movement-ratio target with the current weight (k_theta), then
    # clamp to [k_lo, k_hi]. Squared norms avoid two sqrts; the ratio is preserved.
    def _rebalance_k(new_state, ref_state, k_prev, theta):
        dp = new_state.primal - ref_state.primal
        dd = jnp.concatenate(
            [
                new_state.dual_eq - ref_state.dual_eq,
                new_state.dual_ineq - ref_state.dual_ineq,
            ]
        )
        move_p2 = jnp.vdot(dp, dp) + 1e-60
        move_d2 = jnp.vdot(dd, dd) + 1e-60
        # PDLP primal-weight update: omega = ||dy|| / ||dx|| under tau = eta/k,
        # sigma = eta*k, balancing k||dx||^2 against ||dy||^2 / k.
        k_target = jnp.sqrt(move_d2 / move_p2)
        log_k = theta * jnp.log(k_target) + (1.0 - theta) * jnp.log(k_prev)
        return jnp.clip(jnp.exp(log_k), k_lo, k_hi)

    start_time = time.time()

    try:
        while not is_done():
            if max_epochs:
                if check_max_epochs(count):
                    is_converged = False
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

            start_epoch_time = time.time()
            (
                shifted_i,
                state,
                average_state,
                opt_state,
                total_weight,
                inloop_stalled,
            ) = __sps(
                current_iterations_per_epoch,
                i - restart_i_offset,
                lp,
                state,
                average_state,
                opt_state,
                total_weight,
                primal_damping,
                dual_damping_ineq,
                dual_damping_eq,
                average,
                update_mode,
                k_init=k_init,
                adaptive_eta=adaptive_eta,
                merit_fn=_inloop_merit if _check_every() else None,
                check_every=_check_every(),
                merit_threshold=_restart_thresholds()[0],
                stall_threshold=_restart_thresholds()[1],
                epoch_cache=_epoch_cache,
            )
            inloop_stalled = bool(inloop_stalled)
            # Iterations actually run (an in-epoch restart check may exit early).
            iters_this_epoch = int(shifted_i) - (i - restart_i_offset)
            # __sps increments the (restart-shifted) counter; restore global i.
            # `shifted_i` comes back as a JAX array (it is the scan-carried loop
            # index). Coerce to a Python int so the `start_iter` argument fed to
            # __sps next epoch (i - restart_i_offset) stays a Python int. Otherwise
            # it flips int -> ArrayImpl after epoch 1, and since start_iter is a
            # traced (non-static) argument that type change retraces run_epoch —
            # a second ~0.6s XLA compile billed to epoch 2.
            i = int(shifted_i) + restart_i_offset
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
                termination_pfr,
                termination_dfr,
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
            if restarts:
                epochs_since_restart += 1
                iterations_since_restart += iters_this_epoch

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
                # → True, firing a restart EVERY epoch (e.g. neos-3754480-nidda: 10
                # restarts in 10 epochs, all on merit=inf) and wiping the averaging
                # long before the feasibility tail where restarts actually help. Only the length-based
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
                # epoch; the restart reset wiped the trajectory and it
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
                # The in-epoch check already saw the turnaround chunk-to-chunk
                # (`inloop_stalled`); the epoch-level comparison can't, since the
                # epoch ended mid-rise.
                stalling_restart = (
                    merit_is_finite
                    and bool(restart_merit <= necessary_decay * merit_at_last_restart)
                    and (
                        inloop_stalled
                        or (
                            bool(jnp.isfinite(prev_epoch_merit))
                            and bool(restart_merit > prev_epoch_merit)
                        )
                    )
                )

                if sufficient_progress or stalling_restart or cycle_exhausted:
                    restarted_this_epoch = True
                    # Warm-start restart from the better of {average, iterate};
                    # reset averaging, weight accumulation and the iteration
                    # offset (halpern lambda / eta-growth schedule).
                    state = restart_point
                    # PDLP-style primal-weight rebalance: drive k from the
                    # primal-vs-dual *movement* over the just-finished cycle
                    # (distance between iterates), not per-step gradient norms.
                    # log-space geometric-mean blend with the current weight
                    # (k_theta), then clamp.
                    k_prev = opt_state[0]
                    if (
                        adaptive_theta
                        and k_before_last_move is not None
                        and merit_is_finite
                        and bool(jnp.isfinite(merit_at_last_restart))
                    ):
                        # Trust region on log k: the cycle just finished ran
                        # under the last k move, so judge that move by it.
                        if bool(restart_merit < merit_at_last_restart):
                            theta_live = min(1.0, 2.0 * theta_live)
                        else:
                            theta_live = max(0.05, 0.5 * theta_live)
                            k_prev = k_before_last_move
                    k_before_last_move = k_prev
                    k_new = _rebalance_k(
                        state, state_at_last_restart, k_prev, theta_live
                    )
                    if halpern:
                        # Restarted Halpern: reset eta AND re-anchor z_0 to the
                        # cycle-start iterate `state`. The lambda counter resets
                        # via restart_i_offset below. Independent anchor copy —
                        # `state` is donated to __sps next epoch.
                        _eta_dtype = state.primal.dtype
                        opt_state = (
                            k_new,
                            jnp.asarray(adaptive_eta, _eta_dtype),
                            jax.tree.map(lambda x: x + 0, state),
                        )
                    else:
                        # Carry the learned step size across the restart.
                        # The adaptive rule grows eta ~100-250x above its
                        # 1/||A|| seed over a cycle; re-seeding here forced
                        # the rule to re-climb from scratch after every
                        # restart (a stretch of tiny, conservative steps).
                        # PDLP convention resets averaging at a restart but
                        # keeps the step size.
                        opt_state = (k_new, opt_state[1])
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
                        k_msg = f", k={float(opt_state[0]):.3e}"
                        if adaptive_theta:
                            k_msg += f", k_theta={theta_live:.3g}"
                        print(
                            f"Restart {restarts_done} at epoch {count} "
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
            # (and reset averaging/anchor/eta), so only adjust k on epochs where no
            # restart fired. Unlike the restart path this leaves averaging, the
            # halpern anchor and eta untouched — it only rewrites k, tracking
            # primal/dual progress within the current restart cycle.
            if k_per_epoch and not restarted_this_epoch:
                k_new = _rebalance_k(
                    state, state_at_last_epoch, opt_state[0], theta_live
                )
                opt_state = (k_new, *opt_state[1:])
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
                # opt_state is (k, eta, anchor); rewrite only the anchor with an
                # independent copy (state is donated to __sps next epoch).
                opt_state = (
                    opt_state[0],
                    opt_state[1],
                    jax.tree.map(lambda x: x + 0, state),
                )
                # Reset the cycle-local index so lambda_k = 1/(k_local+1) re-warms
                # toward 1/2 next epoch; without this lambda would stay ~0 and the
                # fresh anchor would carry no weight (a silent no-op).
                restart_i_offset = i - 1

        # The while-loop exits the iteration *after* the converging epoch, so its
        # metrics were computed but only printed if it landed on a log_every
        # boundary. Print the final converged epoch's criteria here (skip when we
        # broke out via max_epochs / max_seconds, which print their own message
        # and leave is_converged False).
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
    if verbose:
        print(f"Time to solution: {end_time - start_time:.2f} seconds")
        print("----------------------------------------------")
        print(f"Epochs to solution: {count}")
        print("----------------------------------------------")
        # Report against the TRUE cost (lp.c may carry the vertex-bias perturbation).
        print(f"Objective: {float((c_true * c_max) @ output.primal):.5e}")
        print("----------------------------------------------")

    if scale:
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
    }


# %%
def to_jaddle_sparse(lp: LP):
    float_dtype, _ = __profile_dtypes()

    lp_jax = JaddleLP(
        jnp.array(lp.c, dtype=float_dtype),
        __scipy_to_bcoo(lp.A_eq, float_dtype),
        jnp.array(lp.b_eq, dtype=float_dtype),
        __scipy_to_bcoo(lp.A_ineq, float_dtype),
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


def __profile_dtypes():
    """Resolve the active precision profile's float width.

    Returns ``(float_dtype, compute_dtype)``. ``float_dtype`` is the profile
    width: float64 (x64, PDLP-style double precision), float32, or float16;
    float64 requires x64 to be enabled (otherwise JAX silently truncates and
    spams warnings), so it falls back to float32 when x64 is off.
    ``compute_dtype`` is at least float32 -- scaling never runs in half
    precision, and scipy.sparse cannot hold float16 anyway.
    """
    float_dtype = jo.jaddle_dtype()
    if float_dtype == jnp.float64 and not jax.config.jax_enable_x64:
        float_dtype = jnp.float32
    return float_dtype, jnp.promote_types(float_dtype, jnp.float32)


def __pad_empty_blocks(lp):
    """Give an empty constraint block a single all-zero row (``0 = 0`` /
    ``0 <= 0``) so downstream code never sees a zero-size constraint class.

    Returns a new LP of the caller's kind (scipy ``LP`` or ``JaddleLP``) when
    padding is needed, else ``lp`` itself; the caller's object is never modified.
    """
    if lp.A_eq.shape[0] > 0 and lp.A_ineq.shape[0] > 0:
        return lp
    n = lp.c.shape[0]

    if isinstance(lp, JaddleLP):

        def zero_row(like):
            return jsp.BCOO(
                (
                    jnp.zeros(0, like.data.dtype),
                    jnp.zeros((0, 2), like.indices.dtype),
                ),
                shape=(1, n),
                indices_sorted=True,
                unique_indices=True,
            )

        def zero_rhs(like):
            return jnp.zeros(1, like.dtype)

    else:

        def zero_row(like):
            return sp.csr_matrix((1, n), dtype=like.dtype)

        def zero_rhs(like):
            return np.zeros(1, np.asarray(like).dtype)

    A_eq, b_eq, A_ineq, b_ineq = lp.A_eq, lp.b_eq, lp.A_ineq, lp.b_ineq
    if A_eq.shape[0] == 0:
        A_eq, b_eq = zero_row(A_eq), zero_rhs(b_eq)
    if A_ineq.shape[0] == 0:
        A_ineq, b_ineq = zero_row(A_ineq), zero_rhs(b_ineq)
    return type(lp)(
        lp.c, A_eq, b_eq, A_ineq, b_ineq, lp.lower_bounds, lp.upper_bounds
    )


def __scipy_to_bcoo(A, dtype):
    """Sorted, deduplicated BCOO of the scipy matrix ``A`` with ``dtype`` data.

    Accepts any scipy sparse format. The scipy side is built at the nearest
    scipy-supported width and only the on-device data is cast, so a float16
    ``dtype`` lives in JAX alone.
    """
    np_float = np.float64 if jnp.dtype(dtype).itemsize == 8 else np.float32
    # sorted_indices() copies, so sum_duplicates() never touches the caller's A.
    A = sp.csr_matrix(A, dtype=np_float).sorted_indices()
    A.sum_duplicates()
    bcoo = jsp.BCOO.from_scipy_sparse(A.tocoo()).sort_indices()
    return jsp.BCOO((bcoo.data.astype(dtype), bcoo.indices), shape=bcoo.shape)


def __equilibrate(
    absdata,
    row_idx,
    col_idx,
    row_scale,
    col_scale,
    max_iter,
    use_max,
    clip_bounds,
    threshold,
):
    """Run ``max_iter`` Sinkhorn-Ruiz equilibration sweeps from the given scales.

    ``use_max=True`` gives the L-infinity (Ruiz) norm via ``segment_max``;
    ``use_max=False`` gives the L1 (Pock-Chambolle) norm via ``segment_sum``.
    Row/col norms of ``D_r M D_c`` factor as ``row_scale * reduce(|M| * col_scale)``
    so the scaled matrix is never rematerialised -- each sweep is two gathers and
    two segmented reductions over the nnz. Starting from non-unit scales
    therefore continues an earlier pass on the already-scaled matrix, returning
    the product of the incoming scales and this pass's updates. The
    empty-row/col guard is implicit: absent segments yield the reduction
    identity (0 for ``segment_sum``, -inf for ``segment_max``), which the
    ``<= threshold -> 1.0`` clamp maps to a unit (no-op) scale either way.
    """
    lo, hi = clip_bounds
    n_rows, n_cols = row_scale.shape[0], col_scale.shape[0]

    def reduce_segments(vals, seg, num):
        if use_max:
            return jax.ops.segment_max(vals, seg, num_segments=num)
        return jax.ops.segment_sum(vals, seg, num_segments=num)

    def body(_, carry):
        row_scale, col_scale = carry
        # Simultaneous (Jacobi) update: both norms are taken from the SAME
        # incoming (row_scale, col_scale), matching Ruiz/Pock-Chambolle and
        # PDLP. A sequential (Gauss-Seidel) update — col norms of the already
        # row-rescaled matrix — voids the PC single-pass guarantee
        # ||D_r^1/2 A D_c^1/2||_2 <= 1 (e.g. [[2]] -> 2^(1/4)).
        row_norms = reduce_segments(absdata * col_scale[col_idx], row_idx, n_rows)
        row_norms = row_norms * row_scale
        row_norms = jnp.where(row_norms <= threshold, 1.0, row_norms)

        col_norms = reduce_segments(absdata * row_scale[row_idx], col_idx, n_cols)
        col_norms = col_norms * col_scale
        col_norms = jnp.where(col_norms <= threshold, 1.0, col_norms)

        row_scale = row_scale * jnp.clip(1.0 / jnp.sqrt(row_norms), lo, hi)
        col_scale = col_scale * jnp.clip(1.0 / jnp.sqrt(col_norms), lo, hi)
        return row_scale, col_scale

    return jax.lax.fori_loop(0, max_iter, body, (row_scale, col_scale))


def __nonzero_or_one(x):
    return jnp.where(x > 0, x, 1.0)


@functools.partial(
    jax.jit,
    static_argnames=(
        "ruiz_iter",
        "pc_iter",
        "clip_bounds",
        "augmented",
        "scaled_objective",
        "scaled_rhs",
    ),
)
def __scale_problem_jax(
    eq_data,
    eq_idx,
    ineq_data,
    ineq_idx,
    c,
    b_eq,
    b_ineq,
    lower_bounds,
    upper_bounds,
    threshold,
    augmented_weight,
    cost_col_floor,
    *,
    ruiz_iter,
    pc_iter,
    clip_bounds,
    augmented,
    scaled_objective,
    scaled_rhs,
):
    """Device-side body of ``scale_problem``: equilibrate, then apply the scales.

    Takes the BCOO data/indices of ``A_eq``/``A_ineq`` plus the LP vectors and
    returns ``(row_scale, col_scale, eq_data, ineq_data, c, b_eq, b_ineq,
    lower_bounds, upper_bounds, c_max)``, all scaled. The sparsity pattern is
    unchanged by diagonal scaling, so the callers reuse the input indices.
    """
    n_eq = b_eq.shape[0]
    m, n = n_eq + b_ineq.shape[0], c.shape[0]
    b = jnp.concatenate([b_eq, b_ineq])

    # [A_eq; A_ineq] as flat COO, ineq rows offset below the eq block.
    a_row = jnp.concatenate([eq_idx[:, 0], ineq_idx[:, 0] + n_eq])
    a_col = jnp.concatenate([eq_idx[:, 1], ineq_idx[:, 1]])
    a_data = jnp.concatenate([eq_data, ineq_data])

    if augmented:
        # |[[A, b], [w c^T/||c||_inf, 0]]|: b is column n, the cost row is row m.
        # ``augmented_weight`` (w) scales the cost row's participation,
        # interpolating between the two endpoints for a single-dense-big-M-row
        # instance like germanrr: the big-M magnitude is conserved and Ruiz can
        # only decide whether it lands in the matrix or in the cost vector
        # (their product ``matrix_ratio * c_max`` is invariant ~= sqrt of the
        # raw big-M ratio). 1.0 is full augmented (cost stays O(1), matrix
        # keeps the residual big-M spread); ->0 approaches A-only Ruiz (matrix
        # flattens, cost blows up). An intermediate value (~1e-3..1e-2 on
        # germanrr) balances the two. No effect without a big-M row.
        # Zero b/c entries are stored but inert: they add 0 to an L1 sum, can't
        # lift an L-inf max above a real entry, and an all-zero row/col still
        # hits the <= threshold -> 1.0 clamp.
        c_norm = __nonzero_or_one(jnp.max(jnp.abs(c), initial=0.0))
        absdata = jnp.abs(
            jnp.concatenate([a_data, b, augmented_weight * c / c_norm])
        )
        row_idx = jnp.concatenate(
            [a_row, jnp.arange(m, dtype=a_row.dtype), jnp.full(n, m, a_row.dtype)]
        )
        col_idx = jnp.concatenate(
            [a_col, jnp.full(m, n, a_col.dtype), jnp.arange(n, dtype=a_col.dtype)]
        )
        n_rows, n_cols = m + 1, n + 1
    else:
        absdata, row_idx, col_idx = jnp.abs(a_data), a_row, a_col
        n_rows, n_cols = m, n

    ones_r = jnp.ones(n_rows, dtype=absdata.dtype)
    ones_c = jnp.ones(n_cols, dtype=absdata.dtype)
    row_scale, col_scale = __equilibrate(
        absdata, row_idx, col_idx, ones_r, ones_c, ruiz_iter, True, clip_bounds,
        threshold,
    )

    if augmented:
        # PC equilibrates the augmented matrix of the Ruiz-SCALED LP, whose b
        # column and cost row are rebuilt from the scaled vectors rather than
        # carrying Ruiz's appended row/col scales: the b column is D_r b (col
        # scale 1) and the cost row is re-normalised to w (c∘d_c)/||c∘d_c||_inf.
        # Continuing from the Ruiz scales reproduces that by resetting the
        # appended col scale to 1 and choosing the appended row scale so that
        # |w c_j / c_norm| · row_scale[m] · d_c,j equals the re-normalised row.
        c_ruiz = __nonzero_or_one(jnp.max(jnp.abs(c * col_scale[:n]), initial=0.0))
        row_scale = row_scale.at[m].set(c_norm / c_ruiz)
        col_scale = col_scale.at[n].set(1.0)

    row_scale, col_scale = __equilibrate(
        absdata, row_idx, col_idx, row_scale, col_scale, pc_iter, False,
        clip_bounds, threshold,
    )
    dr, dc = row_scale[:m], col_scale[:n]

    # Apply D_r A D_c directly to the stored nonzeros.
    eq_data = eq_data * dr[eq_idx[:, 0]] * dc[eq_idx[:, 1]]
    ineq_data = ineq_data * dr[n_eq + ineq_idx[:, 0]] * dc[ineq_idx[:, 1]]
    c = c * dc
    b = b * dr
    lower_bounds = lower_bounds / dc
    upper_bounds = upper_bounds / dc

    # --- Objective/RHS constant normalisation second (on equilibrated c/b) ----
    if scaled_objective:
        # Normalise by the ||c||_inf of the EQUILIBRATED cost so the constant is
        # measured in scaled space. Returned as `c_max` and multiplied back
        # through at every unscaling site (objective, duals).
        c_max = __nonzero_or_one(jnp.max(jnp.abs(c), initial=0.0))
        c = c / c_max
    else:
        c_max = jnp.ones((), dtype=c.dtype)

    if scaled_rhs:
        # Rescale every row (both A's row and b's entry) by the SAME global
        # scalar ||b||_inf of the EQUILIBRATED RHS. Dividing a whole row of Ax=b
        # by a nonzero constant doesn't change its feasible set, so this is free
        # — unlike scaling b alone, which would change the constraint. Folded
        # into dr (not a separate c_max-style constant) so every existing
        # true-units unscaling site (b_norm, constraint_bound, the final dual
        # output rescale) picks it up automatically with no further threading.
        b_max = __nonzero_or_one(jnp.max(jnp.abs(b), initial=0.0))
        eq_data = eq_data / b_max
        ineq_data = ineq_data / b_max
        b = b / b_max
        dr = dr / b_max

    # --- Cost-pinned column floor (after both normalisations) -----------------
    # Augmented Ruiz lets the cost entry set a costed column's scale. When the
    # column's matrix entries are tiny next to its cost, the scaled variable
    # must take a huge value at the optimum and PDHG crawls towards it
    # (fhnw-binschedule0: min z over z >= load_k(x), dense load rows squashed to
    # ~1e-4, z_s* ~ 1.7e4 at 99.8% of ||x_s*||). Rescale such columns so their
    # largest matrix entry is 1; c and the bounds ride along, c_max is
    # untouched. Costed columns only: the same floor on every column fires on
    # half of binschedule0's columns and collapses the step size.
    a_col_all = jnp.concatenate([eq_idx[:, 1], ineq_idx[:, 1]])
    col_max = (
        jnp.zeros(n, dtype=c.dtype)
        .at[a_col_all]
        .max(jnp.abs(jnp.concatenate([eq_data, ineq_data])))
    )
    pinned = (c != 0.0) & (col_max > threshold) & (col_max < cost_col_floor)
    col_boost = jnp.where(
        pinned,
        jnp.minimum(1.0 / jnp.where(pinned, col_max, 1.0), clip_bounds[1]),
        1.0,
    )
    eq_data = eq_data * col_boost[eq_idx[:, 1]]
    ineq_data = ineq_data * col_boost[ineq_idx[:, 1]]
    c = c * col_boost
    lower_bounds = lower_bounds / col_boost
    upper_bounds = upper_bounds / col_boost
    dc = dc * col_boost

    return (
        dr,
        dc,
        eq_data,
        ineq_data,
        c,
        b[:n_eq],
        b[n_eq:],
        lower_bounds,
        upper_bounds,
        c_max,
    )


def scale_problem(
    lp,
    ruiz_iter=40,
    pc_iter=1,
    threshold=1e-8,
    clip_bounds=(1e-6, 1e6),
    augmented=True,
    augmented_weight=1.0,
    scaled_objective=True,
    scaled_rhs=True,
    cost_col_floor=1e-2,
):
    """
    Applies Ruiz+PC scaling to an LP in standard form with sparse matrices:
        min c^T x
        s.t. A_eq x = b_eq
             A_ineq x <= b_ineq
             lower_bounds <= x <= upper_bounds
    ``lp`` is a ``JaddleLP`` or a scipy-backed ``LP``. Returns the scaled
    ``JaddleLP``, row_scaling (length m), col_scaling (length n) as device
    arrays, and the objective constant ``c_max``.

    ``augmented`` selects what is equilibrated:
      * ``True`` (default): the augmented ``[[A, b], [c^T, 0]]`` so cost and RHS
        information also drive the equilibration.
      * ``False`` (PDLP-style): ``A`` alone, with ``b``/``c`` riding the
        resulting row/col scales. It froze momentum1's primal and broke boeing
        once the relative convergence test + true-units norm fixes were in.

    ``ruiz_iter`` L-infinity (Ruiz) sweeps run first, then ``pc_iter`` L1
    (Pock-Chambolle) sweeps continue from the Ruiz scales.

    Ordering (PDLP convention): equilibration (Ruiz + PC) runs FIRST, on the raw
    ``c``/``b``, so the appended cost row / RHS column of the augmented matrix
    carry their true magnitudes into the sweeps. The objective/RHS constant
    normalisation (``c_max``/``b_max``) is applied AFTERWARDS, to the already
    equilibrated ``c``/``b``. Doing the constant normalisation first would feed
    pre-flattened ``c``/``b`` into the augmented equilibration and change the
    resulting row/col scales; PDLP equilibrates, then rescales objective and RHS.

    ``cost_col_floor``: a costed column whose largest scaled matrix entry is
    below this floor is rescaled so that entry becomes 1 (its cost and bounds
    rescale with it). Under augmented Ruiz such a column's scale is pinned by
    its cost, which leaves the variable huge in scaled units (an epigraph
    ``min z, z >= a_k.x`` over dense rows, fhnw-binschedule0). ``0`` disables.

    Everything runs on device in one jitted call: the matrix never round-trips
    through scipy (that plumbing was ~95% of the old host-side pipeline's time,
    ~1 s on scpm1 vs ~50 ms for the sweeps themselves).
    """
    float_dtype, compute_dtype = __profile_dtypes()
    if isinstance(lp, JaddleLP):
        A_eq, A_ineq = lp.A_eq, lp.A_ineq
    else:
        # Convert at the compute width rather than the profile width, so a
        # float16 profile can't overflow raw big-M coefficients before scaling
        # brings them down.
        A_eq = __scipy_to_bcoo(lp.A_eq, compute_dtype)
        A_ineq = __scipy_to_bcoo(lp.A_ineq, compute_dtype)

    def vec(v):
        return jnp.asarray(v, dtype=compute_dtype)

    dr, dc, eq_data, ineq_data, c, b_eq, b_ineq, lb, ub, c_max = __scale_problem_jax(
        A_eq.data.astype(compute_dtype),
        A_eq.indices,
        A_ineq.data.astype(compute_dtype),
        A_ineq.indices,
        vec(lp.c),
        vec(lp.b_eq),
        vec(lp.b_ineq),
        vec(lp.lower_bounds),
        vec(lp.upper_bounds),
        threshold,
        augmented_weight,
        cost_col_floor,
        ruiz_iter=ruiz_iter,
        pc_iter=pc_iter,
        clip_bounds=tuple(clip_bounds),
        augmented=augmented,
        scaled_objective=scaled_objective,
        scaled_rhs=scaled_rhs,
    )

    def rebuild(data, like):
        return jsp.BCOO(
            (data.astype(float_dtype), like.indices),
            shape=like.shape,
            indices_sorted=like.indices_sorted,
            unique_indices=like.unique_indices,
        )

    lp_scaled = JaddleLP(
        c.astype(float_dtype),
        rebuild(eq_data, A_eq),
        b_eq.astype(float_dtype),
        rebuild(ineq_data, A_ineq),
        b_ineq.astype(float_dtype),
        lb.astype(float_dtype),
        ub.astype(float_dtype),
    )
    return lp_scaled, dr, dc, float(c_max)


def project_onto_eq(lp: JaddleLP, primal: jnp.ndarray, tol=1e-6) -> jnp.ndarray:
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


def lp_to_dual(lp: JaddleLP) -> JaddleLP:
    """
    Builds the true dual LP of ``lp``, adapted to THIS codebase's dual sign
    convention.

    The paper's (Applegate et al., arXiv:2501.07018) (2)/(3a) primal update is
    ``x' = proj_X(x - τ(c - Aᵀy))``, i.e. its dual stationarity is
    ``c - Aᵀy = r``. This codebase's primal update (`grad_primal` in
    `solve`'s `grad`/`grad_primal_only`, ~line 205/220) is instead
    ``x' = proj_X(x - τ(c + Aᵀy))`` — the OPPOSITE sign on the ``Aᵀy`` term,
    so its stationarity condition is ``c + Aᵀy = r``, i.e. this codebase's
    ``dual_eq``/``dual_ineq`` are the negative of the paper's ``y``. Using
    ``Aᵀy + r = c`` verbatim would silently solve for the paper's ``y``, which
    has the wrong sign relative to every other dual quantity this codebase
    produces (`solve()`'s own `dual_eq`/`dual_ineq`, `compute_epoch_metrics`'s
    reduced cost, etc.) — the resulting `dual_eq`/`dual_ineq` would fail
    `evaluate_lp_certificate`'s feasibility check despite solving *a* valid LP.

        maximize_{y, r}  b_eqᵀy_eq + b_ineqᵀy_ineq + Σ box_infimum(r; l, u)
        subject to:      Aᵀy - r = -c,  y ∈ Y,  r ∈ R

    which is exactly `evaluate_lp_certificate`'s own dual-objective/duality-gap
    decomposition (jaddle_linear.py ~2943), so a (dual_eq, dual_ineq, r) that
    solves this dual LP to optimality is a genuine LP dual solution, not just a
    feasible point of Eq (7).

    ``box_infimum(r; l, u)`` is ``min(r·l, r·u)`` when x has two finite bounds
    (r free there — see below), ``r·l``/``r·u`` when only one bound is finite,
    and 0 when x is free. The two-sided case makes the objective PIECEWISE
    LINEAR in a free r, which a plain JaddleLP (linear c) can't represent
    directly. Fix: split each two-sided column's r into ``r = r_plus - r_minus``
    with ``r_plus, r_minus ≥ 0`` — then ``min(r·l, r·u) = l·r_plus - u·r_minus``
    exactly (maximizing pushes whichever of r_plus/r_minus is slack to 0, which
    is the same thing the min() would have picked). Columns with only one
    finite bound (or none) keep a single ``r`` entry as before.

    The dual LP's primal variable is the stacked
    ``z = [y_eq; y_ineq; r_rest; r_plus; r_minus]`` — ``r_rest`` holds one
    entry per column that is NOT two-sided-bounded (free/lower-only/upper-only,
    same bounds rule as before), ``r_plus``/``r_minus`` hold one entry each per
    two-sided-bounded column. `solve()`'s own primal update IS the y/r update
    of the dual LP — no vestigial x, no vestigial objective coupling.

    ``y ∈ Y``: y_eq is free (equality rows dualize with no sign restriction),
    y_ineq ≥ 0 (matching this codebase's `A_ineq x ≤ b_ineq` convention) —
    encoded as bounds on the y-block of z, not as inequality rows.

    ``r ∈ R``: the per-variable normal-cone sign pattern implied by each
    original variable's bounds on x (same rule as `compute_epoch_metrics`'s
    dual-feasibility check): two-sided variables admit any sign, split into
    r_plus, r_minus ≥ 0; lower-only need r ≥ 0; upper-only need r ≤ 0; free
    variables need r = 0 — encoded as bounds on the r-block of z.

    The single equality block ``Aᵀy - r = -c`` (n rows, one per original
    variable, with the two-sided columns' ``-r`` split into ``-r_plus + r_minus``)
    is exactly this codebase's dual-feasibility constraint; there are no
    inequality rows since all sign structure now lives in the bounds on z.
    """
    m = lp.A_eq.shape[0] + lp.A_ineq.shape[0]
    n = lp.c.shape[0]

    lower_bounds_np = np.asarray(lp.lower_bounds)
    upper_bounds_np = np.asarray(lp.upper_bounds)
    finite_lower_np = np.isfinite(lower_bounds_np)
    finite_upper_np = np.isfinite(upper_bounds_np)
    has_both_bounds_np = finite_lower_np & finite_upper_np
    has_only_lower_np = finite_lower_np & (~finite_upper_np)
    has_only_upper_np = (~finite_lower_np) & finite_upper_np

    two_sided_idx = np.flatnonzero(has_both_bounds_np)
    rest_idx = np.flatnonzero(~has_both_bounds_np)
    n_rest = rest_idx.shape[0]
    n_two_sided = two_sided_idx.shape[0]

    # z = [y_eq; y_ineq; r_rest; r_plus; r_minus]
    rest_offset = m
    plus_offset = m + n_rest
    minus_offset = m + n_rest + n_two_sided
    z_size = m + n_rest + 2 * n_two_sided

    # [A_eq^T | A_ineq^T] acting on [y_eq; y_ineq], padded into the full
    # (n, z_size) constraint block.
    A_T_block = lp.A_T  # (n, m), rows = original variables, cols = [eq; ineq] duals
    index_dtype = A_T_block.indices.dtype
    AT_padded = jsp.BCOO((A_T_block.data, A_T_block.indices), shape=(n, z_size))

    # -r_rest on the rest columns' own rows.
    rest_rows = jnp.asarray(rest_idx, dtype=index_dtype)
    rest_cols = jnp.arange(n_rest, dtype=index_dtype) + rest_offset
    rest_data = -jnp.ones(n_rest, dtype=lp.c.dtype)

    # -r_plus + r_minus on the two-sided columns' own rows.
    two_sided_rows = jnp.asarray(two_sided_idx, dtype=index_dtype)
    plus_cols = jnp.arange(n_two_sided, dtype=index_dtype) + plus_offset
    minus_cols = jnp.arange(n_two_sided, dtype=index_dtype) + minus_offset

    r_rows = jnp.concatenate([rest_rows, two_sided_rows, two_sided_rows])
    r_cols = jnp.concatenate([rest_cols, plus_cols, minus_cols])
    r_data = jnp.concatenate(
        [
            rest_data,
            -jnp.ones(n_two_sided, dtype=lp.c.dtype),
            jnp.ones(n_two_sided, dtype=lp.c.dtype),
        ]
    )
    r_bcoo = jsp.BCOO((r_data, jnp.stack([r_rows, r_cols], axis=1)), shape=(n, z_size))

    # AT_padded and r_bcoo occupy disjoint columns (duals vs r), so summing
    # them just places both sets of entries into one (n, z_size) block.
    constraint_block = (AT_padded + r_bcoo).sum_duplicates()

    A_eq_new = constraint_block
    b_eq_new = -lp.c
    A_ineq_new = jsp.BCOO(
        (jnp.zeros(0, dtype=lp.c.dtype), jnp.zeros((0, 2), dtype=index_dtype)),
        shape=(0, z_size),
    )
    b_ineq_new = jnp.zeros(0, dtype=lp.c.dtype)

    y_eq_lower = jnp.full(lp.n_eq, -jnp.inf, dtype=lp.c.dtype)
    y_eq_upper = jnp.full(lp.n_eq, jnp.inf, dtype=lp.c.dtype)
    n_ineq = lp.A_ineq.shape[0]
    y_ineq_lower = jnp.zeros(n_ineq, dtype=lp.c.dtype)
    y_ineq_upper = jnp.full(n_ineq, jnp.inf, dtype=lp.c.dtype)

    # r_rest ∈ R: lower-only → [0, inf), upper-only → (-inf, 0], free → {0}.
    has_only_lower_rest = jnp.asarray(has_only_lower_np[rest_idx])
    has_only_upper_rest = jnp.asarray(has_only_upper_np[rest_idx])
    r_rest_lower = jnp.where(
        has_only_lower_rest, 0.0, jnp.where(has_only_upper_rest, -jnp.inf, 0.0)
    )
    r_rest_upper = jnp.where(
        has_only_lower_rest, jnp.inf, jnp.where(has_only_upper_rest, 0.0, 0.0)
    )

    # r_plus, r_minus ≥ 0 for two-sided columns.
    r_plus_lower = jnp.zeros(n_two_sided, dtype=lp.c.dtype)
    r_plus_upper = jnp.full(n_two_sided, jnp.inf, dtype=lp.c.dtype)
    r_minus_lower = jnp.zeros(n_two_sided, dtype=lp.c.dtype)
    r_minus_upper = jnp.full(n_two_sided, jnp.inf, dtype=lp.c.dtype)

    lower_new = jnp.concatenate(
        [y_eq_lower, y_ineq_lower, r_rest_lower, r_plus_lower, r_minus_lower]
    )
    upper_new = jnp.concatenate(
        [y_eq_upper, y_ineq_upper, r_rest_upper, r_plus_upper, r_minus_upper]
    )

    # solve() minimizes c_new @ z; the true dual maximizes
    # b_eq^T y_eq + b_ineq^T y_ineq + l^T r_plus - u^T r_minus (two-sided cols)
    # + box_infimum terms for the single-bound r_rest columns, so c_new is the
    # negative of all of that.
    lower_rest = jnp.asarray(lower_bounds_np[rest_idx], dtype=lp.c.dtype)
    upper_rest = jnp.asarray(upper_bounds_np[rest_idx], dtype=lp.c.dtype)
    # Only one of lower_rest/upper_rest is finite per rest column (has_both is
    # excluded from rest); the other is ±inf but multiplies a r bound pinned to
    # 0 in that regime, so replace the non-finite side with 0 to avoid inf*0.
    c_r_rest = -jnp.where(has_only_lower_rest, lower_rest, 0.0) - jnp.where(
        has_only_upper_rest, upper_rest, 0.0
    )
    lower_two_sided = jnp.asarray(lower_bounds_np[two_sided_idx], dtype=lp.c.dtype)
    upper_two_sided = jnp.asarray(upper_bounds_np[two_sided_idx], dtype=lp.c.dtype)
    c_r_plus = -lower_two_sided
    c_r_minus = upper_two_sided

    c_new = jnp.concatenate(
        [
            -lp.b_eq,
            -lp.b_ineq,
            c_r_rest,
            c_r_plus,
            c_r_minus,
        ]
    )

    dual_lp = JaddleLP(
        c=c_new,
        A_eq=A_eq_new,
        b_eq=b_eq_new,
        A_ineq=A_ineq_new,
        b_ineq=b_ineq_new,
        lower_bounds=lower_new,
        upper_bounds=upper_new,
    )
    # Layout needed to reconstruct r (length n) and unpack z downstream.
    dual_lp.rest_idx = rest_idx
    dual_lp.two_sided_idx = two_sided_idx
    dual_lp.rest_offset = rest_offset
    dual_lp.plus_offset = plus_offset
    dual_lp.minus_offset = minus_offset
    return dual_lp


def build_dual_feasibility_lp(lp: JaddleLP) -> JaddleLP:
    """
    Builds Equation (7) of Applegate et al. (arXiv:2501.07018) as its own LP:
    `lp_to_dual`'s true dual LP with its objective zeroed out (a feasibility
    problem, so it doesn't require the piecewise-linear box term of the true
    dual to be optimized — only for `z` to satisfy `Aᵀy - r = -c` within the
    y/r bounds `lp_to_dual` already encodes).

        maximize_{y, r}  0
        subject to:      Aᵀy - r = -c,  y ∈ Y,  r ∈ R

    (equivalently, minimizing 0 subject to the same constraint — same feasible
    set, same PDHG dynamics either way). This is exactly the stationarity
    condition ``c + Aᵀy = r`` this codebase's own gradient uses, so a
    (dual_eq, dual_ineq) solving this problem is directly comparable to (and
    interchangeable with) any dual iterate `solve()` itself produces.
    """
    dual_lp = lp_to_dual(lp)
    zeroed = JaddleLP(
        c=jnp.zeros_like(dual_lp.c),
        A_eq=dual_lp.A_eq,
        b_eq=dual_lp.b_eq,
        A_ineq=dual_lp.A_ineq,
        b_ineq=dual_lp.b_ineq,
        lower_bounds=dual_lp.lower_bounds,
        upper_bounds=dual_lp.upper_bounds,
    )
    zeroed.rest_idx = dual_lp.rest_idx
    zeroed.two_sided_idx = dual_lp.two_sided_idx
    zeroed.rest_offset = dual_lp.rest_offset
    zeroed.plus_offset = dual_lp.plus_offset
    zeroed.minus_offset = dual_lp.minus_offset
    return zeroed


def evaluate_lp_certificate(lp: JaddleLP, primal, dual_eq, dual_ineq):
    """
    Standalone, TRUE-units reimplementation of `solve()`'s internal
    `compute_epoch_metrics`/`relative_gap` (jaddle_linear.py ~1458), for judging
    the quality of an (x, y) pair produced OUTSIDE a `solve()` call (e.g. from
    `solve_dual_feasibility`), where none of `solve()`'s internal scale factors
    (`col_scale`, `row_scale`, `c_max`) are available. `compute_epoch_metrics`
    itself can't be called directly for this: it's a closure over those scaled-
    space factors and operates on the solver's internal scaled iterate, not on
    true-unit (x, y).

    This runs the identical box_infimum / duality-gap decomposition with every
    scale factor set to its identity (col_scale=row_scale=c_max=1), which is
    algebraically equivalent to compute_epoch_metrics when `primal`/`dual_eq`/
    `dual_ineq` are already in true (unscaled) units — the case here, since
    `solve()` always returns `result["solution"]` unscaled back to true units.

    Returns a dict: ``objective``, ``dual_objective``, ``duality_gap``,
    ``relative_gap``, ``primal_feasibility_residual``, ``dual_feasibility_residual``.
    """
    dual = jnp.concatenate([dual_eq, dual_ineq])
    Ax = lp.A @ primal
    reduced_cost = lp.c + lp.A_T @ dual
    Ax_minus_b = Ax - lp.b
    grad_dual_eq = Ax_minus_b[: lp.n_eq]
    grad_dual_ineq = Ax_minus_b[lp.n_eq :]

    objective_value = lp.objective(primal)

    lower_bounds = lp.lower_bounds
    upper_bounds = lp.upper_bounds
    finite_lower = jnp.isfinite(lower_bounds)
    finite_upper = jnp.isfinite(upper_bounds)
    has_both_bounds = finite_lower & finite_upper
    has_only_lower = finite_lower & (~finite_upper)
    has_only_upper = (~finite_lower) & finite_upper

    lower_term = reduced_cost * lower_bounds
    upper_term = reduced_cost * upper_bounds
    box_infimum = jnp.where(
        has_both_bounds,
        jnp.minimum(lower_term, upper_term),
        jnp.where(
            has_only_lower, lower_term, jnp.where(has_only_upper, upper_term, 0.0)
        ),
    )

    proj_box = projection_box(primal - reduced_cost, lower_bounds, upper_bounds)
    dual_feasibility_violation = jnp.where(
        has_both_bounds,
        jnp.abs(primal - proj_box),
        jnp.where(
            has_only_lower,
            jnp.maximum(-reduced_cost, 0.0),
            jnp.where(
                has_only_upper,
                jnp.maximum(reduced_cost, 0.0),
                jnp.abs(reduced_cost),
            ),
        ),
    )
    dual_feasibility_residual = jnp.max(dual_feasibility_violation, initial=0.0)

    ineq_violations = jnp.maximum(grad_dual_ineq, 0.0)
    eq_violations = jnp.abs(grad_dual_eq)
    primal_feasibility_residual = jnp.maximum(
        jnp.max(ineq_violations, initial=0.0), jnp.max(eq_violations, initial=0.0)
    )

    relative_primal_feasibility_residual = primal_feasibility_residual / (
        1.0 + jnp.max(jnp.abs(lp.b), initial=0.0)
    )
    relative_dual_feasibility_residual = dual_feasibility_residual / (
        1.0 + jnp.max(jnp.abs(lp.c), initial=0.0)
    )

    gap_bound_comp = reduced_cost @ primal - jnp.sum(box_infimum)
    gap_ineq_comp = -(dual_ineq @ grad_dual_ineq)
    gap_eq_comp = -(dual_eq @ grad_dual_eq)
    duality_gap = gap_bound_comp + gap_ineq_comp + gap_eq_comp

    dual_objective = objective_value - duality_gap
    relative_gap = jnp.abs(duality_gap) / (
        1.0 + jnp.abs(objective_value) + jnp.abs(dual_objective)
    )

    return {
        "objective": objective_value,
        "dual_objective": dual_objective,
        "duality_gap": duality_gap,
        "relative_gap": relative_gap,
        "primal_feasibility_residual": primal_feasibility_residual,
        "dual_feasibility_residual": dual_feasibility_residual,
        "relative_primal_feasibility_residual": relative_primal_feasibility_residual,
        "relative_dual_feasibility_residual": relative_dual_feasibility_residual,
    }


def solve_dual_feasibility(
    lp: JaddleLP,
    initial_dual_eq=None,
    initial_dual_ineq=None,
    **kwargs,
):
    """
    Solves the dual feasibility problem, Equation (7) of Applegate et al.
    (arXiv:2501.07018), by handing `build_dual_feasibility_lp`'s LP (the true
    dual LP from `lp_to_dual`, with its objective zeroed) to `solve()`
    unmodified — this is the literal PDLP-on-(7) construction from Algorithm 4
    step 3, using this codebase's own primal-dual saddle solver rather than a
    bespoke update rule. The imported (`initial_dual_eq`, `initial_dual_ineq`)
    dual — typically the outer solve's own warm-started dual — seeds `y`; `r`
    is seeded consistently from it rather than left at zero.

    Returns `solve()`'s result dict, with `solution.primal` split back into
    `dual_eq`/`dual_ineq` (the y found for the original LP) and `r` (the
    recovered reduced costs, length n, reassembled from the LP's r_rest/
    r_plus/r_minus split), added as extra keys.
    """
    dual_feasibility_lp = build_dual_feasibility_lp(lp)
    rest_idx = dual_feasibility_lp.rest_idx
    two_sided_idx = dual_feasibility_lp.two_sided_idx
    rest_offset = dual_feasibility_lp.rest_offset
    plus_offset = dual_feasibility_lp.plus_offset
    minus_offset = dual_feasibility_lp.minus_offset

    n_eq = lp.n_eq
    n_ineq = lp.A_ineq.shape[0]
    n = lp.c.shape[0]
    n_rest = rest_idx.shape[0]
    n_two_sided = two_sided_idx.shape[0]

    initial_solution = None
    if initial_dual_eq is not None or initial_dual_ineq is not None:
        y_eq0 = (
            jnp.zeros(n_eq, dtype=lp.c.dtype)
            if initial_dual_eq is None
            else initial_dual_eq
        )
        y_ineq0 = (
            jnp.zeros(n_ineq, dtype=lp.c.dtype)
            if initial_dual_ineq is None
            else initial_dual_ineq
        )
        # Seed r at the warm start's OWN reduced cost c + A^Ty0, not zero: r is
        # completely unconstrained (free) for any two-sided-box variable, so a
        # zero-seeded r has nothing pulling it back if PDHG's first few steps
        # overshoot while it catches up to the true reduced cost — starting it
        # already at the right value removes that transient entirely (measured
        # on stp3d: r=0 seed corrupted the polish within single-digit PDHG
        # iterations even though the warm y0 was already near-optimal).
        y0 = jnp.concatenate([y_eq0, y_ineq0])
        r0 = lp.c + lp.A_T @ y0
        r0_rest = r0[rest_idx]
        r0_two_sided = r0[two_sided_idx]
        # r = r_plus - r_minus with both >= 0: seed the positive part on
        # whichever side r0 actually sits, zero on the other, so z0 already
        # satisfies the split's sign constraints exactly (no projection jolt).
        r0_plus = jnp.maximum(r0_two_sided, 0.0)
        r0_minus = jnp.maximum(-r0_two_sided, 0.0)
        z0 = jnp.concatenate([y_eq0, y_ineq0, r0_rest, r0_plus, r0_minus])
        # dual_ineq/dual_eq here are the DUAL-FEASIBILITY LP's own duals (for
        # its equality block `A^Ty - r = -c`, which has n rows) — not y_eq0/
        # y_ineq0, which are its primal block. The dual-feasibility LP has 0
        # inequality rows, but `solve()` pads any zero-row A_ineq to a single
        # dummy row (jaddle_linear.py ~1188) without a matching primal LP
        # constructed here, so dual_ineq must be length 1, not 0, to match.
        initial_solution = SaddleState(
            primal=z0,
            dual_ineq=jnp.zeros(1, dtype=lp.c.dtype),
            dual_eq=jnp.zeros(n, dtype=lp.c.dtype),
        )

    result = solve(
        dual_feasibility_lp,
        initial_solution=initial_solution,
        **kwargs,
    )

    z = result["solution"].primal
    result["dual_eq"] = z[:n_eq]
    result["dual_ineq"] = z[n_eq : n_eq + n_ineq]

    r_rest = z[rest_offset : rest_offset + n_rest]
    r_plus = z[plus_offset : plus_offset + n_two_sided]
    r_minus = z[minus_offset : minus_offset + n_two_sided]
    r = jnp.zeros(n, dtype=lp.c.dtype)
    r = r.at[rest_idx].set(r_rest)
    r = r.at[two_sided_idx].set(r_plus - r_minus)
    result["r"] = r

    return result


# %%


def primal_polish(
    lp: JaddleLP,
    warm_start: SaddleState,
    **kwargs,
):
    """
    Runs Algorithm 4 step 2 (Applegate et al., arXiv:2501.07018): PDHG on the
    primal feasibility problem (their Eq. 6), warm-started from `warm_start`.

    Since Eq. 6 has a zero objective, its dual has no notion of "the" solution
    — an unconstrained solve wanders freely through the feasible set and can
    drift arbitrarily far from `warm_start` (measured on stp3d: 8+ iterations
    already destroys the polish, since gradient steps toward feasibility can
    move a long way in objective terms while barely moving the residual on an
    underdetermined problem). The paper's fix is to cap this to a short burst
    (`k/8` iterations, `k` = outer solve's iteration count so far) rather than
    solving to convergence — `max_iters` here plays that role, passed through
    as `iterations_per_epoch` with a single epoch so it caps the actual PDHG
    iteration count, not an epoch count (whose default iteration budget of
    256 is far too long for this warm-start use).
    """
    initial_solution = SaddleState(
        primal=warm_start.primal,
        dual_ineq=jnp.zeros_like(warm_start.dual_ineq),
        dual_eq=jnp.zeros_like(warm_start.dual_eq),
    )

    lp_feasible = JaddleLP(
        c=jnp.zeros_like(lp.c),
        A_ineq=lp.A_ineq,
        b_ineq=lp.b_ineq,
        A_eq=lp.A_eq,
        b_eq=lp.b_eq,
        lower_bounds=lp.lower_bounds,
        upper_bounds=lp.upper_bounds,
    )

    kwargs.setdefault("max_epochs", 1)
    kwargs.setdefault("iterations_per_epoch", 8)

    return solve(
        lp_feasible,
        initial_solution=initial_solution,
        **kwargs,
    )["solution"].primal


def dual_polish(
    lp: JaddleLP,
    warm_start: SaddleState,
    **kwargs,
):
    """
    Runs Algorithm 4 step 3: PDHG on the dual feasibility problem (Eq. 7),
    warm-started from `warm_start`. See `primal_polish`'s docstring for why
    `max_iters` (a short capped burst, not a full solve) is essential here —
    the exact same unconstrained-drift failure mode was measured on stp3d:
    dual_polish warm-started from a near-optimal y (relative_gap 2.3e-4)
    degraded to relative_gap ~1.0 after just 8 iterations of unconstrained
    PDHG on Eq. 7, because the dual feasibility problem's zero objective gives
    the solver no reason to stay near the warm start.
    """

    kwargs.setdefault("max_epochs", 1)
    kwargs.setdefault("iterations_per_epoch", 1)

    result = solve_dual_feasibility(
        lp,
        initial_dual_eq=warm_start.dual_eq,
        initial_dual_ineq=warm_start.dual_ineq,
        **kwargs,
    )

    return result["dual_eq"], result["dual_ineq"]


def solve_with_polishing(
    lp: JaddleLP,
    max_rounds: int = 5,
    max_epochs=None,
    tol=1e-6,
    **kwargs,
):

    if kwargs.get("verbose", False):
        print("Solving original problem...")

    # primal_polish/dual_polish/evaluate_lp_certificate all require a JaddleLP
    # (BCOO blocks) — solve() itself accepts a scipy-backed LP too and converts
    # it internally, but that conversion is local to solve() and never visible
    # here. Normalise once so the polishing loop below always has BCOO to work
    # with, regardless of which form the caller passed in.
    if not isinstance(lp, JaddleLP):
        lp = to_jaddle_sparse(lp)

    rounds = 0
    # First, solve the original problem
    result = solve(
        lp,
        primal_feasibility_tolerance=tol,
        dual_feasibility_tolerance=tol,
        dual_gap_tolerance=tol,
        max_epochs=max_epochs,
        **kwargs,
    )

    # If the solver converged, perform primal and dual polishing
    while not result["converged"] and rounds <= max_rounds:
        rounds += 1

        if kwargs.get("verbose", False):
            print(f"Polishing round {rounds}...")
        polished_primal = primal_polish(
            lp,
            warm_start=result["solution"],
            **kwargs,
        )

        solution = SaddleState(
            primal=polished_primal,
            dual_ineq=result["solution"].dual_ineq,
            dual_eq=result["solution"].dual_eq,
        )

        lp_cert = evaluate_lp_certificate(
            lp,
            polished_primal,
            result["solution"].dual_eq,
            result["solution"].dual_ineq,
        )

        if kwargs.get("verbose", False):
            print(
                f"Relative primal feasibility residual: {lp_cert['relative_primal_feasibility_residual']}"
            )
            print(
                f"Relative dual feasibility residual: {lp_cert['relative_dual_feasibility_residual']}"
            )
            print(f"Relative gap: {lp_cert['relative_gap']}")

        if (
            lp_cert["relative_gap"] < tol
            and lp_cert["relative_primal_feasibility_residual"] < tol
            and lp_cert["relative_dual_feasibility_residual"] < tol
        ):
            if kwargs.get("verbose", False):
                print("Polished solution is feasible and optimal. Stopping polishing.")
            break

        if kwargs.get("verbose", False):
            print(f"Re-solving with polished solution as warm start...")

        result = solve(
            lp,
            initial_solution=solution,
            primal_feasibility_tolerance=tol,
            dual_feasibility_tolerance=tol,
            dual_gap_tolerance=tol,
            max_epochs=max_epochs,
            **kwargs,
        )
    return result
