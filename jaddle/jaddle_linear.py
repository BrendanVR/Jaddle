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
from jaddle.jaddle_basic_types import LP, JaddleLP, LPValues, SaddleState, scipy_to_bcoo
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
def _make_epoch_fn(
    lp: JaddleLP,
    primal_damping=0.0,
    dual_damping_ineq=0.0,
    dual_damping_eq=0.0,
    average=True,
    update_mode="pdhg",
    merit_fn=None,
    check_every=None,
):
    # Returns `run_epoch(start_iter, state, average_state, opt_state,
    # total_weight, merit_threshold, stall_threshold, *, max_iter)`, which runs
    # one epoch of `max_iter` iterations and returns
    # (i, state, average_state, opt_state, total_weight, stalled). It is a plain
    # traceable function, not jitted: `solve` calls it inside its device-side
    # epoch loop (`run_chunk`), which is jitted as a whole. `max_iter` must be a
    # Python int (it is the scan length).
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

    def run_epoch(
        start_iter,
        state,
        average_state,
        opt_state,
        total_weight,
        merit_threshold,
        stall_threshold,
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

    return run_epoch


# Epoch budget for a chunk that should run until convergence: large enough never
# to bind, small enough that `count + n` cannot overflow an int32 counter.
_UNBOUNDED_EPOCHS = 2**30


def _vector_norm(v, norm):
    """‖v‖∞ (``norm="inf"``) or ‖v‖₂ (``"l2"``); 0 for an empty vector."""
    if norm == "l2":
        return jnp.linalg.norm(v)
    return jnp.max(jnp.abs(v), initial=0.0)


@functools.partial(jax.jit, static_argnames=("scale", "scaled_objective"))
def _unscale_output(
    output, col_scale, row_scale_ineq, row_scale_eq, c_max, *, scale, scaled_objective
):
    """Map the scaled-space solution back to true units (primal *= col_scale,
    duals *= row_scale, then duals *= c_max), as ONE compile instead of an
    eager per-op kernel each."""
    if scale:
        output = SaddleState(
            primal=output.primal * col_scale,
            dual_ineq=output.dual_ineq * row_scale_ineq,
            dual_eq=output.dual_eq * row_scale_eq,
        )
    if scaled_objective:
        output = SaddleState(
            primal=output.primal,
            dual_ineq=output.dual_ineq * c_max,
            dual_eq=output.dual_eq * c_max,
        )
    return output


def _check_settings(
    dual_residual, termination_norm, restart_norm, update_mode, k_scale, adaptive_eta
):
    """Validate the settings ``solve()`` and ``make_solver`` share. Returns
    ``(k_lo, k_hi, adaptive_eta, halpern)``, with ``adaptive_eta`` either a
    float or ``"auto"`` (resolved after scaling)."""
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

    valid_update_modes = ["pdhg", "alternating", "halpern"]
    if update_mode not in valid_update_modes:
        raise ValueError(f"update_mode must be one of {valid_update_modes}")

    # ``k_scale`` sets the clamp band ``[1/k_scale, k_scale]`` for the primal
    # weight k; ``None`` leaves k unclamped.
    if k_scale is not None:
        k_lo, k_hi = 1.0 / k_scale, k_scale
    else:
        k_lo, k_hi = 0.0, np.inf
    # "auto" (or 0.0) seeds eta from the scaled LP's augmented spectral norm,
    # resolved after scaling.
    if isinstance(adaptive_eta, str):
        if adaptive_eta != "auto":
            raise ValueError(
                f"adaptive_eta must be a float >= 0 or 'auto', got {adaptive_eta!r}"
            )
    elif adaptive_eta is None or adaptive_eta < 0.0:
        raise ValueError("adaptive_eta must be a float >= 0 or 'auto' (0.0 = auto)")
    elif adaptive_eta == 0.0:
        adaptive_eta = "auto"
    return k_lo, k_hi, adaptive_eta, update_mode == "halpern"


def _device_solver(
    lp,
    *,
    c_true,
    c_max,
    col_scale,
    jnp_row_scale_ineq,
    jnp_row_scale_eq,
    initial_solution,
    initial_opt_state,
    k_init,
    k_lo,
    k_hi,
    adaptive_eta,
    halpern,
    vertex_bias,
    primal_damping,
    dual_damping_ineq,
    dual_damping_eq,
    primal_feasibility_tolerance,
    dual_feasibility_tolerance,
    dual_gap_tolerance,
    dual_residual,
    termination_norm,
    restart_norm,
    verbose,
    log_every,
    average,
    report_best,
    update_mode,
    k_theta,
    k_update_per_epoch,
    iterations_per_epoch,
    restarts,
    epochs_per_restart,
    restart_multiplier,
    restart_decay,
    necessary_decay,
    primal_stop,
    primal_stop_window,
    primal_stop_obj_tol,
    halpern_reanchor_per_epoch,
    iterations_per_epoch_decay,
    iterations_per_epoch_min,
    restart_check_every,
    reference_objective,
):
    """The device side of ``solve()``: everything that runs inside its jitted
    epoch loop, built for one scaled problem.

    ``lp`` is the scaled ``JaddleLP``; ``c_true`` the (scaled) cost the metrics
    report when ``vertex_bias`` perturbs ``lp.c``; ``c_max``, ``col_scale`` and
    the row scales map residuals back to true units; ``initial_solution`` is
    in scaled space. The remaining arguments are ``solve()``'s settings.

    Returns ``(build_chunk, carry, print_epoch_metrics, adaptive_theta,
    chunk_target_seconds)``: ``build_chunk(ipe)`` gives the jitted
    ``run_chunk(carry, n_epochs)`` for an epoch length, ``carry`` is the
    initial loop state, and the rest serve ``solve()``'s host loop.
    """
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
                return jnp.sqrt(jnp.sum(ineq_violations**2) + jnp.sum(eq_violations**2))
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

    # PDLP-style primal-weight initialisation. When k-scaling is on and k_init is
    # left as None we derive it from the objective/RHS norms ||c|| / ||b|| (in
    # the scaled space the solver iterates in), which puts the primal/dual step
    # ratio in the right order of magnitude before iteration 1 instead of
    # starting symmetric.
    if k_init is None:
        # Kept on the device (no float()) so this runs under jit / vmap.
        norm_c = jnp.linalg.norm(lp.c) + 1e-30
        norm_b = jnp.linalg.norm(lp.b) + 1e-30
        k_init = jnp.clip(norm_c / norm_b, k_lo, k_hi)

    if initial_opt_state is not None:
        opt_state = initial_opt_state
    else:
        # Step-size state (k, eta); halpern also carries the anchor z_0
        # (cycle-start iterate), seeded from the initial solution and reset at
        # each restart.
        _pik_dtype = initial_solution.primal.dtype
        opt_state = (
            jnp.asarray(k_init, _pik_dtype),
            jnp.asarray(adaptive_eta, _pik_dtype),
        )
        if halpern:
            opt_state = opt_state + (initial_solution,)

    # k_theta="adaptive": live smoothing coefficient, judged at each restart
    # (see the restart logic in `epoch` below).
    if isinstance(k_theta, str):
        if k_theta != "adaptive":
            raise ValueError(f"k_theta must be a float or 'adaptive', got {k_theta!r}")
        adaptive_theta, theta_init = True, 0.5
    else:
        adaptive_theta, theta_init = False, float(k_theta)
    k_per_epoch = bool(k_update_per_epoch)

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
    b_norm = _vector_norm(_b_true, restart_norm)
    c_norm = _vector_norm(_c_true, restart_norm)
    termination_b_norm = _vector_norm(_b_true, termination_norm)
    termination_c_norm = _vector_norm(_c_true, termination_norm)

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

    def print_epoch_metrics(
        epoch,
        objective_value,
        termination_pfr,
        termination_dfr,
        duality_gap,
        gap_bound_comp,
        gap_ineq_comp,
        gap_eq_comp,
        epoch_time=None,
        k_val=None,
        eta_val=None,
    ):
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
        relative_pfr = termination_pfr / (1.0 + float(termination_b_norm))
        relative_dfr = termination_dfr / (1.0 + float(termination_c_norm))
        # RDGABS: the no-cancellation companion to RDG (see relative_gap_abs).
        # `converged()` requires RDGABS <= dual_gap_tolerance; without printing
        # it a run can show PFR/DFR/RDG all comfortably inside tolerance yet
        # never certify because the gap components are cancelling rather than
        # genuinely small.
        rdg = float(relative_gap(duality_gap, objective_value))
        rdg_abs = float(
            relative_gap_abs(
                objective_value, duality_gap, gap_bound_comp, gap_ineq_comp, gap_eq_comp
            )
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
            f"|Epoch {epoch}|"
            f"|Obj{objective_value:.2e}|"
            f"|PFR {relative_pfr:.2e}|"
            f"|DFR {relative_dfr:.2e}|"
            f"|RDG {rdg:.2e}|"
            f"|RDGABS {rdg_abs:.2e}|"
            f"{objerr_str}"
            f"{ke_str}"
            f"{time_str}"
        )
        print("----------------------------------------------")

    def _inloop_merit(s):
        # Same restart merit `solve` computes at the epoch boundary, traced into
        # the epoch for the in-epoch sufficient-progress check.
        obj, _pgn, _cs, cb, dfr, dg, dgf, *_rest = compute_epoch_metrics(s)
        return kkt_merit(cb, dfr, dg, dgf, obj)

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

    # --- Device-resident epoch loop -------------------------------------------
    # Epochs run in chunks: one jitted lax.while_loop (`run_chunk`) executes many
    # epochs back to back on the device -- the iterations, the end-of-epoch
    # metrics, the termination test, and the restart / primal-weight logic -- so
    # the host never waits on the device between epochs. Python regains control
    # only between chunks, to enforce max_epochs / max_seconds, print the
    # buffered epoch log and restart messages, and catch Ctrl-C. Chunks are sized
    # to take about `_CHUNK_TARGET_SECONDS`, so those checks stay responsive.
    #
    # All loop state lives in the `carry` dict below. A chunk also ends early
    # when a restart decays `iterations_per_epoch` (the epoch length is a static
    # scan length, so the next chunk is compiled for the new length) or when the
    # verbose log buffers fill.
    _CHUNK_TARGET_SECONDS = 1.0
    _LOG_CAP = 64  # buffered epoch-log rows per chunk
    _EVT_CAP = 64  # buffered restart events per chunk
    _metric_shapes = jax.eval_shape(compute_epoch_metrics, initial_solution)
    _merit_dtype = _metric_shapes[3].dtype

    def _initial_metrics():
        # +inf everywhere (and gap finiteness False) so the convergence tests
        # can't fire before the first epoch has computed real metrics.
        return tuple(
            jnp.zeros(m.shape, m.dtype)
            if m.dtype == jnp.bool_
            else jnp.full(m.shape, jnp.inf, m.dtype)
            for m in _metric_shapes
        )

    def _merit_of(m):
        return kkt_merit(m[3], m[4], m[5], m[6], m[0])

    def _select(pred, a, b):
        return jax.tree.map(lambda x, y: jnp.where(pred, x, y), a, b)

    def _strong(tree):
        # Strip JAX's weak typing (from Python-scalar inputs like jnp.asarray(0))
        # so the carry handed to run_chunk always has exactly the types run_chunk
        # returns; otherwise the second call retraces and recompiles the chunk.
        return jax.tree.map(
            lambda x: jax.lax.convert_element_type(x, jnp.asarray(x).dtype), tree
        )

    def _check_every_for(ipe):
        # "auto": ten in-epoch checks per epoch. None disables the check.
        if restart_check_every == "auto":
            return max(1, ipe // 10)
        return restart_check_every

    def _build_chunk(ipe):
        check_every = _check_every_for(ipe)
        run_epoch = _make_epoch_fn(
            lp,
            primal_damping,
            dual_damping_ineq,
            dual_damping_eq,
            average,
            update_mode,
            merit_fn=_inloop_merit if check_every else None,
            check_every=check_every,
        )
        inloop_restart_check = bool(check_every) and restarts
        ipe_after_restart = max(
            iterations_per_epoch_min, int(ipe * iterations_per_epoch_decay)
        )

        def epoch(c):
            mal = c["merit_at_last_restart"]
            # In-epoch restart thresholds: sufficient progress and necessary
            # decay relative to the merit at the last restart; -inf (never
            # fires) until a finite baseline exists, or when the check is off.
            if inloop_restart_check:
                finite = jnp.isfinite(mal)
                thr_sufficient = jnp.where(finite, restart_decay * mal, -jnp.inf)
                thr_stall = jnp.where(finite, necessary_decay * mal, -jnp.inf)
            else:
                thr_sufficient = thr_stall = jnp.asarray(-jnp.inf, _merit_dtype)

            # The epoch runs on the restart-shifted index, so a restart re-zeros
            # the halpern lambda counter and the eta growth schedule.
            start = c["i"] - c["restart_i_offset"]
            shifted_i, state, avg, opt, total_weight, stalled = run_epoch(
                start,
                c["state"],
                c["avg"],
                c["opt"],
                c["total_weight"],
                thr_sufficient,
                thr_stall,
                max_iter=ipe,
            )
            i = shifted_i + c["restart_i_offset"]
            # Iterations actually run (an in-epoch restart check may exit early).
            iters_this_epoch = shifted_i - start
            count = c["count"] + 1

            metrics = compute_epoch_metrics(avg if average else state)
            # `used_avg` tracks which point the metrics describe.
            used_avg = jnp.asarray(average)
            # report_best: when averaging is on, the average and the last iterate
            # are different points and either can be the better solution
            # (averaging stabilises rotational problems but lags the last iterate
            # when it is already contracting). Report/converge on whichever has
            # the lower KKT merit, at the cost of a second matvec pair per epoch.
            if report_best and average:
                state_metrics = compute_epoch_metrics(state)
                iterate_better = _merit_of(state_metrics) < _merit_of(metrics)
                metrics = _select(iterate_better, state_metrics, metrics)
                used_avg = ~iterate_better
            objective_value = metrics[0]

            # Roll the latest objective into the primal_stop window (oldest first).
            obj_window = jnp.concatenate(
                [c["obj_window"][1:], jnp.reshape(objective_value, (1,))]
            )

            log, n_log, evt, n_evt = c["log"], c["n_log"], c["evt"], c["n_evt"]
            if verbose:
                write = (count == 1) | (count % log_every == 0)
                row = jnp.stack(
                    [
                        count,
                        objective_value,
                        metrics[10],
                        metrics[11],
                        metrics[5],
                        metrics[7],
                        metrics[8],
                        metrics[9],
                        opt[0],
                        opt[1],
                    ]
                ).astype(log.dtype)
                log = jnp.where(write, log.at[n_log].set(row), log)
                n_log = n_log + write

            # --- Adaptive restart decision ---
            restarted = jnp.asarray(False)
            theta = c["theta"]
            k_before, has_k_before = c["k_before_last_move"], c["has_k_before"]
            at_restart = c["state_at_last_restart"]
            restart_i_offset = c["restart_i_offset"]
            iterations_since = c["iterations_since_restart"]
            cycle_cap = c["cycle_cap"]
            ipe_next = c["ipe"]
            restarts_done = c["restarts_done"]
            prev_pfr = c["prev_relative_pfr"]
            prev_dfr = c["prev_relative_dfr"]
            prev_epoch_merit = c["prev_epoch_merit"]
            if restarts:
                iterations_since = iterations_since + iters_this_epoch
                # `merit` is the metric of the *reported* point: with report_best
                # it is already the better of {average, iterate}; otherwise it is
                # the average (averaging on) or the last iterate.
                merit = _merit_of(metrics)
                cycle_exhausted = iterations_since >= cycle_cap

                # --- Two-point restart candidate (PDLP-style) ---
                # Restart to whichever of {average, iterate} has the lower merit
                # instead of always discarding a frequently-better average.
                restart_used_avg = used_avg if average else jnp.asarray(False)
                restart_point = _select(restart_used_avg, avg, state)
                restart_merit = merit
                near_threshold = merit <= restart_decay * mal
                if average and not report_best:
                    # report_best is off, so the iterate's metrics were not
                    # computed yet. Only pay for them on epochs where a restart
                    # can actually fire (cycle exhausted, or the average already
                    # near the sufficient-progress threshold).
                    need = cycle_exhausted | near_threshold
                    state_merit = jax.lax.cond(
                        need,
                        lambda s: _merit_of(compute_epoch_metrics(s)).astype(
                            merit.dtype
                        ),
                        lambda s: jnp.full((), jnp.inf, merit.dtype),
                        state,
                    )
                    iterate_better = need & (state_merit < merit)
                    restart_point = _select(iterate_better, state, restart_point)
                    restart_merit = jnp.where(iterate_better, state_merit, restart_merit)
                    restart_used_avg = restart_used_avg & ~iterate_better

                # A non-finite merit carries no progress signal — it just means
                # the iterate is dual-infeasible so the duality gap (hence the KKT
                # merit) is +∞. Without this gate the progress tests below
                # degenerate to `inf <= restart_decay*inf` → True, firing a
                # restart EVERY epoch (neos-3754480-nidda: 10 restarts in 10
                # epochs, all on merit=inf). Only the length-based
                # `cycle_exhausted` path may fire on an inf merit.
                merit_is_finite = jnp.isfinite(restart_merit)

                # While still dual-infeasible, BOTH relative feasibility residuals
                # decreasing epoch-over-epoch means the run is mid-flight on a
                # good trajectory, not stuck — a cycle-cap restart there only
                # destroys momentum (momentum1). Suppress just the exhaustion path.
                relative_pfr = metrics[3] / (1.0 + b_norm)
                relative_dfr = metrics[4] / (1.0 + c_norm)
                still_improving = (
                    (~merit_is_finite)
                    & (relative_pfr < prev_pfr)
                    & (relative_dfr < prev_dfr)
                )
                cycle_exhausted = cycle_exhausted & ~still_improving
                prev_pfr, prev_dfr = relative_pfr, relative_dfr

                # Seed the baseline on the first finite merit so the
                # sufficient-progress test has something real to compare against.
                mal = jnp.where(
                    ~jnp.isfinite(mal) & merit_is_finite, restart_merit, mal
                )

                sufficient_progress = merit_is_finite & (
                    restart_merit <= restart_decay * mal
                )
                # cuPDLP condition (ii) — "necessary decay + stalling": restart
                # once the merit has decayed to <= necessary_decay of its
                # cycle-start value AND has started rising again (the in-epoch
                # check saw it chunk-to-chunk, or vs the previous epoch).
                stalling_restart = (
                    merit_is_finite
                    & (restart_merit <= necessary_decay * mal)
                    & (
                        stalled
                        | (
                            jnp.isfinite(prev_epoch_merit)
                            & (restart_merit > prev_epoch_merit)
                        )
                    )
                )
                restarted = sufficient_progress | stalling_restart | cycle_exhausted

                # PDLP-style primal-weight rebalance from the movement over the
                # just-finished cycle, blended in log space (k_theta), clamped.
                k_prev = opt[0]
                theta_new = theta
                if adaptive_theta:
                    # Trust region on log k: the cycle just finished ran under
                    # the last k move, so judge that move by it.
                    judge = has_k_before & merit_is_finite & jnp.isfinite(mal)
                    improved = restart_merit < mal
                    theta_new = jnp.where(
                        judge,
                        jnp.where(
                            improved,
                            jnp.minimum(1.0, 2.0 * theta),
                            jnp.maximum(0.05, 0.5 * theta),
                        ),
                        theta,
                    )
                    k_prev = jnp.where(judge & ~improved, k_before, k_prev)
                k_new = _rebalance_k(restart_point, at_restart, k_prev, theta_new)
                if halpern:
                    # Restarted Halpern: reset eta and re-anchor z_0 to the
                    # cycle-start iterate.
                    opt_restart = (
                        k_new,
                        jnp.asarray(adaptive_eta, opt[1].dtype),
                        restart_point,
                    )
                else:
                    # Carry the learned step size across the restart (PDLP
                    # convention): re-seeding forced the adaptive rule to
                    # re-climb from scratch after every restart.
                    opt_restart = (k_new, opt[1])

                # Warm-start from the restart point; reset averaging, the weight
                # accumulator (else the new cycle's average stays frozen near the
                # restart point) and the iteration offset.
                theta = jnp.where(restarted, theta_new, theta)
                k_before = jnp.where(restarted, k_prev, k_before)
                has_k_before = has_k_before | restarted
                opt = _select(restarted, opt_restart, opt)
                state = _select(restarted, restart_point, state)
                avg = _select(restarted, restart_point, avg)
                at_restart = _select(restarted, restart_point, at_restart)
                total_weight = jnp.where(restarted, 0.0, total_weight)
                restart_i_offset = jnp.where(restarted, i - 1, restart_i_offset)
                mal = jnp.where(restarted, restart_merit, mal)
                # New cycle: no previous-epoch merit yet for condition (ii).
                prev_epoch_merit = jnp.where(restarted, jnp.inf, restart_merit)
                iterations_since = jnp.where(restarted, 0, iterations_since)
                cycle_cap = jnp.where(
                    restarted, cycle_cap * restart_multiplier, cycle_cap
                )
                ipe_next = jnp.where(restarted, ipe_after_restart, ipe_next)
                restarts_done = restarts_done + restarted
                if verbose:
                    reason = jnp.where(
                        sufficient_progress, 0, jnp.where(stalling_restart, 1, 2)
                    )
                    row = jnp.stack(
                        [
                            count,
                            reason,
                            restart_merit,
                            restart_used_avg,
                            k_new,
                            theta_new,
                            cycle_cap,
                            ipe_next,
                        ]
                    ).astype(evt.dtype)
                    evt = jnp.where(restarted, evt.at[n_evt].set(row), evt)
                    n_evt = n_evt + restarted

            # Per-epoch primal-weight rebalance (k_update_per_epoch): only on
            # epochs where no restart fired (a restart already rebalanced k).
            # Leaves averaging, the halpern anchor and eta untouched.
            at_epoch = c["state_at_last_epoch"]
            if k_per_epoch:
                k_epoch = _rebalance_k(state, at_epoch, opt[0], theta)
                opt = (jnp.where(restarted, opt[0], k_epoch),) + tuple(opt[1:])
                at_epoch = state

            # Optional per-epoch Halpern re-anchor: start a fresh Halpern cycle at
            # every epoch boundary (anchor := current iterate, lambda counter
            # reset), unless a real restart already re-anchored this epoch.
            if halpern and halpern_reanchor_per_epoch:
                opt = (opt[0], opt[1], _select(restarted, opt[2], state))
                restart_i_offset = jnp.where(restarted, restart_i_offset, i - 1)

            # Termination. Check the certificate first: when both it and the
            # primal-stall heuristic fire on the same epoch, attribute the stop
            # to the certificate — the stronger reason.
            certificate = converged(
                metrics[10],
                metrics[11],
                metrics[5],
                metrics[6],
                metrics[0],
                metrics[7],
                metrics[8],
                metrics[9],
            )
            if primal_stop:
                primal_stall = converged_primal(metrics[10], obj_window)
            else:
                primal_stall = jnp.asarray(False)

            new = dict(
                state=state,
                avg=avg,
                opt=opt,
                state_at_last_restart=at_restart,
                state_at_last_epoch=at_epoch,
                metrics=metrics,
                used_avg=used_avg,
                count=count,
                i=i,
                restart_i_offset=restart_i_offset,
                total_weight=total_weight,
                obj_window=obj_window,
                iterations_since_restart=iterations_since,
                cycle_cap=cycle_cap,
                merit_at_last_restart=mal,
                theta=theta,
                k_before_last_move=k_before,
                has_k_before=has_k_before,
                prev_relative_pfr=prev_pfr,
                prev_relative_dfr=prev_dfr,
                prev_epoch_merit=prev_epoch_merit,
                restarts_done=restarts_done,
                ipe=ipe_next,
                done=certificate | primal_stall,
                stop_code=jnp.where(certificate, 1, jnp.where(primal_stall, 2, 0)),
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

    carry = dict(
        state=initial_solution,
        avg=initial_solution,
        opt=opt_state,
        state_at_last_restart=initial_solution,
        state_at_last_epoch=initial_solution if k_per_epoch else None,
        metrics=_initial_metrics(),
        used_avg=jnp.asarray(average),
        count=jnp.asarray(0),
        i=jnp.asarray(1),
        restart_i_offset=jnp.asarray(0),
        total_weight=jnp.asarray(0.0),
        obj_window=jnp.full((max(int(primal_stop_window), 1),), jnp.inf),
        iterations_since_restart=jnp.asarray(0),
        # The cycle cap is tracked in ITERATIONS so `cycle_exhausted` fires at the
        # same point in the trajectory regardless of how iterations are chopped
        # into epochs; iterations_since_restart accumulates the ACTUAL count.
        # None = no cap: inf never exhausts and stays inf under restart_multiplier.
        cycle_cap=jnp.asarray(
            jnp.inf
            if epochs_per_restart is None
            else float(epochs_per_restart) * float(iterations_per_epoch)
        ),
        merit_at_last_restart=jnp.asarray(jnp.inf, _merit_dtype),
        theta=jnp.asarray(theta_init),
        k_before_last_move=jnp.asarray(1.0, opt_state[0].dtype),
        has_k_before=jnp.asarray(False),
        # Previous epoch's relative residuals (for the `still_improving` guard)
        # and merit (for condition (ii)); inf until the first epoch sets them.
        prev_relative_pfr=jnp.asarray(jnp.inf, _merit_dtype),
        prev_relative_dfr=jnp.asarray(jnp.inf, _merit_dtype),
        prev_epoch_merit=jnp.asarray(jnp.inf, _merit_dtype),
        restarts_done=jnp.asarray(0),
        ipe=jnp.asarray(iterations_per_epoch),
        done=jnp.asarray(False),
        stop_code=jnp.asarray(0),
        log=jnp.zeros((_LOG_CAP, 10), _merit_dtype) if verbose else None,
        n_log=jnp.asarray(0),
        evt=jnp.zeros((_EVT_CAP, 8), _merit_dtype) if verbose else None,
        n_evt=jnp.asarray(0),
    )
    carry = _strong(carry)
    return _build_chunk, carry, print_epoch_metrics, adaptive_theta, _CHUNK_TARGET_SECONDS


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
    termination_norm="l2",
    restart_norm="l2",
    verbose=False,
    log_every=1,
    average=True,
    report_best=True,
    update_mode="alternating",
    k_scale=1e8,
    k_theta=0.5,
    k_init=None,
    k_update_per_epoch=False,
    adaptive_eta=1.0,
    scale=True,
    scaled_objective=True,
    scaled_rhs=True,
    scaled_augmented=True,
    augmented_weight=1.0,
    ruiz_iterations=10,
    pc_iterations=1,
    cost_col_floor=0,
    restarts=True,
    epochs_per_restart=None,
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
    (no-progress restart). On by default; disable with ``restarts=False``.

    The epoch loop runs on the device: each call into JAX executes a chunk of
    many epochs, including the metrics, convergence test and restart logic, with
    no host synchronisation between them. Python regains control between
    chunks (sized to take about a second) to enforce ``max_epochs`` /
    ``max_seconds``, print the verbose log and handle Ctrl-C. With
    ``verbose=True`` the per-epoch ``Time`` is therefore the average over the
    epoch's chunk. When ``verbose``, ``max_epochs`` and ``max_seconds`` are all
    unset, nothing needs the host between epochs, so the whole solve runs as a
    single device call (only an ``iterations_per_epoch_decay`` restart, which
    changes the compiled epoch length, returns to Python). In that mode Ctrl-C
    takes effect only once the call returns, and ``"corrected_seconds"`` equals
    ``"solve_seconds"``.

    Args:
        max_seconds: Wall-clock budget in seconds (default ``None`` = no limit).
            Measured from entry into ``solve()``, so scaling / setup and the
            first-epoch XLA compile count against it. Checked between chunks
            of epochs, and each chunk is sized from the measured epoch time to
            fit the remaining budget; once the budget is spent the current
            point is returned with ``stop_reason="time_limit"``. The solve can
            overrun by about one epoch (shrink ``iterations_per_epoch`` for a
            tighter cutoff).
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
            stopping tests (and the printed PFR / DFR): ``"l2"`` (default)
            tests ‖r_p‖₂/(1+‖b‖₂) and ‖r_d‖₂/(1+‖c‖₂), the cuPDLP-C / PDLP
            default, for like-for-like comparisons; ``"inf"`` tests
            ‖r_p‖∞/(1+‖b‖∞) and ‖r_d‖∞/(1+‖c‖∞). Termination only: the restart merit
            follows ``restart_norm``, so changing this alone leaves the iterate
            trajectory unchanged and only moves the epoch at which it stops.
        restart_norm: Norm (``"l2"`` default, or ``"inf"``) for the primal /
            dual feasibility terms of the restart KKT merit and the
            dual-infeasible ``still_improving`` cycle-cap guard, normalised by
            1+‖b‖ / 1+‖c‖ in the same norm. ``"l2"`` matches cuPDLP-C's restart
            criterion. Unlike ``termination_norm`` this changes the trajectory.
        restarts: Enable adaptive warm restarts (default ``True``). There is
            no cap on how many fire; the triggers alone decide. Each restart
            resets the averaging (and the halpern anchor / lambda counter) while
            keeping the current iterate as a warm start.
        epochs_per_restart: Length cap of the first restart cycle, or ``None``
            (default) for no cap — restarts then fire only on the
            sufficient-progress / stalling triggers. When set, expressed in
            epochs AT THE DEFAULT ``iterations_per_epoch`` but
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
            * ``"alternating"`` (default): primal step, then a dual step at the
              new primal x^{k+1} (Gauss–Seidel / alternating GDA, no
              extrapolation). Not contractive in general; relies on
              averaging/restarts.
            * ``"pdhg"``: Chambolle–Pock PDHG (primal step then dual step on
              the extrapolated primal x_bar = 2x^{k+1} − x^k).
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
            A float > 0 sets the seed directly (default ``1.0``, a natural scale
            once Ruiz/PC scaling has brought ``||A||`` to O(1)); ``"auto"`` (or
            ``0.0``) seeds it at ``1/||[[A,b],[c,0]]||_2`` of the scaled LP,
            estimated by power iteration. The line search corrects a poor seed
            within a few iterations. The learned ``eta`` is carried across restarts.
        k_theta: Smoothing coefficient for the log-space primal-weight update at
            each restart / epoch. A float fixes it (default ``0.5``, as in PDLP;
            smaller = slower adaptation).
            ``"adaptive"`` sets it from data as a trust region on log k:
            starting at 0.5, each restart doubles theta (capped at 1) if the
            restart merit fell over the cycle since the previous k move, else
            halves it (floored at 0.05) and reverts k to its value before that
            move. The movement ratio alone can't tell a correct k move from a
            runaway (mzzv11's per-epoch runaway was monotone), but the merit can.
            Epochs to certify, restart-only vs fixed 0.5: barwon 41 vs 166,
            binschedule2 39 vs 61, plus gains on stp3d and mzzv11. It was the
            default briefly, but fixed 0.5 proved safer across instances
            (hgms30 ratchets k under ``"adaptive"``).
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
        cost_col_floor: Scaling floor for cost-pinned columns: a costed column
            whose largest scaled matrix entry is below it is rescaled so that
            entry becomes 1. Fixes epigraph objectives (``min z, z >= a_k.x``
            over dense rows, fhnw-binschedule0) where augmented Ruiz leaves the
            variable huge in scaled units. Default ``0`` (disabled); pass e.g.
            ``1e-2`` to enable. See ``scale_problem``.
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
              formed (fewer than two epochs, or a single-call solve).
            * ``"epochs"``: ``int``, number of epochs run.
    """

    # max_seconds is a wall-clock budget for the whole call, setup included.
    solve_entry_time = time.time()
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be > 0 (or None for no limit)")

    lp = __pad_empty_blocks(lp)

    if log_every < 1:
        raise ValueError("log_every must be >= 1")

    if verbose:
        print("----------------------------------------------")

    k_lo, k_hi, adaptive_eta, halpern = _check_settings(
        dual_residual, termination_norm, restart_norm, update_mode, k_scale, adaptive_eta
    )

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

    if adaptive_eta == "auto":
        adaptive_eta = 1 / float(estimate_augmented_spectral_norm(lp))
        if verbose:
            print(f"Adaptive step size seed set to 1/||A||_2 = {adaptive_eta:.3e}")
            print("----------------------------------------------")

    # A user-supplied initial_solution is given in the LP's original (unscaled)
    # space, so it must be mapped into the scaled space the solver iterates in.
    # The default from lp.initial_solution() is already built from the scaled lp
    # and must NOT be rescaled again.
    user_supplied_initial = initial_solution is not None
    if initial_solution is None:
        initial_solution = lp.initial_solution()
    else:
        # A warm start for the caller's LP has no dual for the zero row that
        # __pad_empty_blocks gives an empty block; give it a zero dual.
        def fit(dual, rows):
            dual = jnp.asarray(dual)
            if dual.shape[0] == rows:
                return dual
            if dual.shape[0] == 0 and rows == 1:
                return jnp.zeros(1, dual.dtype)
            raise ValueError(
                f"initial_solution has {dual.shape[0]} duals for {rows} rows"
            )

        initial_solution = SaddleState(
            primal=jnp.asarray(initial_solution.primal),
            dual_ineq=fit(initial_solution.dual_ineq, lp.A_ineq.shape[0]),
            dual_eq=fit(initial_solution.dual_eq, lp.A_eq.shape[0]),
        )

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

    # Convert to jax arrays for use inside jitted functions. Match the state
    # dtype so dividing by the scales doesn't upcast the state back out of the
    # profile precision (numpy float64 scale * jax float16 -> float64). Sliced
    # and cast on the host, then uploaded: each eager device slice/cast would
    # compile its own per-shape kernel.
    _row_scale_host = np.asarray(row_scale)
    _state_np_dtype = jnp.dtype(_state_dtype)
    jnp_row_scale_ineq = jax.device_put(
        _row_scale_host[len(lp.b_eq) :].astype(_state_np_dtype)
    )
    jnp_row_scale_eq = jax.device_put(
        _row_scale_host[: len(lp.b_eq)].astype(_state_np_dtype)
    )
    col_scale = jax.device_put(np.asarray(col_scale).astype(_state_np_dtype))

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
        # Map the user's original-space solution into scaled space: the exact
        # inverse of _unscale_output (primal *= col_scale, dual *= row_scale *
        # c_max). c_max is 1 unless the objective was normalised; leaving it
        # out warm-started every dual off by that factor.
        initial_solution = SaddleState(
            primal=initial_solution.primal / col_scale,
            dual_ineq=initial_solution.dual_ineq / (jnp_row_scale_ineq * c_max),
            dual_eq=initial_solution.dual_eq / (jnp_row_scale_eq * c_max),
        )

    (
        _build_chunk,
        carry,
        print_epoch_metrics,
        adaptive_theta,
        _CHUNK_TARGET_SECONDS,
    ) = _device_solver(
        lp,
        c_true=c_true,
        c_max=c_max,
        col_scale=col_scale,
        jnp_row_scale_ineq=jnp_row_scale_ineq,
        jnp_row_scale_eq=jnp_row_scale_eq,
        initial_solution=initial_solution,
        initial_opt_state=initial_opt_state,
        k_init=k_init,
        k_lo=k_lo,
        k_hi=k_hi,
        adaptive_eta=adaptive_eta,
        halpern=halpern,
        vertex_bias=vertex_bias,
        primal_damping=primal_damping,
        dual_damping_ineq=dual_damping_ineq,
        dual_damping_eq=dual_damping_eq,
        primal_feasibility_tolerance=primal_feasibility_tolerance,
        dual_feasibility_tolerance=dual_feasibility_tolerance,
        dual_gap_tolerance=dual_gap_tolerance,
        dual_residual=dual_residual,
        termination_norm=termination_norm,
        restart_norm=restart_norm,
        verbose=verbose,
        log_every=log_every,
        average=average,
        report_best=report_best,
        update_mode=update_mode,
        k_theta=k_theta,
        k_update_per_epoch=k_update_per_epoch,
        iterations_per_epoch=iterations_per_epoch,
        restarts=restarts,
        epochs_per_restart=epochs_per_restart,
        restart_multiplier=restart_multiplier,
        restart_decay=restart_decay,
        necessary_decay=necessary_decay,
        primal_stop=primal_stop,
        primal_stop_window=primal_stop_window,
        primal_stop_obj_tol=primal_stop_obj_tol,
        halpern_reanchor_per_epoch=halpern_reanchor_per_epoch,
        iterations_per_epoch_decay=iterations_per_epoch_decay,
        iterations_per_epoch_min=iterations_per_epoch_min,
        restart_check_every=restart_check_every,
        reference_objective=reference_objective,
    )

    chunk_fns = {}
    count = 0
    done = False
    stop_code = 0
    current_ipe = int(iterations_per_epoch)
    restarts_printed = 0
    # With no logging and no budgets there is nothing for the host to do between
    # epochs, so the whole solve is one call: the loop only returns when the
    # certificate is met (or a restart changes iterations_per_epoch, whose new
    # scan length needs a new compile).
    single_call = not verbose and not max_epochs and max_seconds is None
    # Wall time of the very first epoch (which pays the one-off XLA compile).
    # Outside single-call mode the first chunk is exactly one epoch, so this
    # keeps its meaning; callers use it to form a "corrected" runtime.
    first_epoch_seconds = None
    seconds_per_epoch = None
    chunk_epochs = _UNBOUNDED_EPOCHS if single_call else 1
    is_converged = True
    # Why the loop terminated; overwritten below by the reason that fired.
    stop_reason = "max_epochs"

    def print_chunk_log(host, epoch_time):
        nonlocal restarts_printed
        entries = [(int(r[0]), 0, r) for r in host["log"][: int(host["n_log"])]]
        entries += [(int(r[0]), 1, r) for r in host["evt"][: int(host["n_evt"])]]
        for epoch_no, kind, r in sorted(entries, key=lambda e: e[:2]):
            if kind == 0:
                print_epoch_metrics(
                    epoch_no,
                    *(float(v) for v in r[1:8]),
                    epoch_time=epoch_time,
                    k_val=float(r[8]),
                    eta_val=float(r[9]),
                )
                continue
            restarts_printed += 1
            reason = ("sufficient-progress", "stalling", "cycle-cap")[int(r[1])]
            which = "avg" if r[3] else "iterate"
            k_msg = f", k={float(r[4]):.3e}"
            if adaptive_theta:
                k_msg += f", k_theta={float(r[5]):.3g}"
            print(
                f"Restart {restarts_printed} at epoch {epoch_no} "
                f"({reason}, merit={float(r[2]):.2e} "
                f"[{which}], next cap={float(r[6]):.0f} iters, "
                f"iters/epoch={int(r[7])}{k_msg})"
            )
            print("----------------------------------------------")

    start_time = time.time()

    try:
        while True:
            if done:
                stop_reason = "certificate" if stop_code == 1 else "primal_stall"
                break
            if max_epochs and count >= max_epochs:
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
            new_carry = run_chunk(carry, n_epochs)
            keys = ["count", "done", "stop_code", "ipe"]
            if verbose:
                keys += ["log", "n_log", "evt", "n_evt"]
            # The one host/device synchronisation per chunk.
            host = jax.device_get({k: new_carry[k] for k in keys})
            carry = new_carry
            chunk_seconds = time.time() - chunk_start

            epochs_run = max(int(host["count"]) - count, 1)
            count = int(host["count"])
            done = bool(host["done"])
            stop_code = int(host["stop_code"])
            current_ipe = int(host["ipe"])
            if single_call:
                continue
            if first_epoch_seconds is None:
                first_epoch_seconds = chunk_seconds
            else:
                seconds_per_epoch = chunk_seconds / epochs_run

            if verbose:
                print_chunk_log(host, chunk_seconds / epochs_run)
                zero = jnp.zeros_like(carry["n_log"])
                carry = {**carry, "n_log": zero, "n_evt": zero}

            # Size the next chunk to take about _CHUNK_TARGET_SECONDS, growing
            # by at most 8x per chunk (the first chunk's time includes compile).
            per_epoch = chunk_seconds / epochs_run
            chunk_epochs = int(
                min(max(1.0, _CHUNK_TARGET_SECONDS / per_epoch), 8 * epochs_run)
            )

        # Print the final converged epoch's criteria (skip when we stopped on a
        # budget, which prints its own message and leaves is_converged False).
        if verbose and is_converged and count > 0:
            m = jax.device_get(carry["metrics"])
            print("Convergence criteria met.")
            if report_best and average:
                print(
                    f"Reported point: "
                    f"{'average' if bool(carry['used_avg']) else 'iterate'}"
                )
            print("----------------------------------------------")
            print_epoch_metrics(
                count,
                float(m[0]),
                float(m[10]),
                float(m[11]),
                float(m[5]),
                float(m[7]),
                float(m[8]),
                float(m[9]),
            )
    except KeyboardInterrupt:
        # `carry` still holds the last completed chunk (it is not donated).
        is_converged = False
        stop_reason = "interrupted"
        count = int(carry["count"])
        print("KeyboardInterrupt received. Returning current solution.")
        print("----------------------------------------------")

    state, average_state, opt_state = carry["state"], carry["avg"], carry["opt"]
    reported_used_avg = bool(carry["used_avg"])
    if report_best and average:
        output = average_state if reported_used_avg else state
    elif average:
        output = average_state
    else:
        output = state

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

    output = _unscale_output(
        output,
        col_scale,
        jnp_row_scale_ineq,
        jnp_row_scale_eq,
        c_max,
        scale=bool(scale),
        scaled_objective=bool(scaled_objective),
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


class SolveCoreResult(NamedTuple):
    """What a ``make_solver`` function returns; every field is a JAX array, so
    results batch under ``vmap``.

    ``solution`` is the reported point in true units (as in ``solve()``),
    ``converged`` whether the LP certificate (or ``primal_stop``) fired,
    ``stop_code`` 1 for the certificate, 2 for ``primal_stop`` and 0 for the
    epoch budget, and ``epochs`` the epochs run."""

    solution: SaddleState
    converged: jnp.ndarray
    stop_code: jnp.ndarray
    epochs: jnp.ndarray


def make_solver(
    lp,
    max_epochs=None,
    iterations_per_epoch=256,
    dual_damping_ineq=0.0,
    dual_damping_eq=0.0,
    primal_damping=0.0,
    primal_feasibility_tolerance=1e-3,
    dual_feasibility_tolerance=1e-3,
    dual_gap_tolerance=1e-3,
    dual_residual="pdlp",
    termination_norm="l2",
    restart_norm="l2",
    average=True,
    report_best=True,
    update_mode="alternating",
    k_scale=1e8,
    k_theta=0.5,
    k_init=None,
    k_update_per_epoch=False,
    adaptive_eta=1.0,
    scale=True,
    scaled_objective=True,
    scaled_rhs=True,
    scaled_augmented=True,
    augmented_weight=1.0,
    ruiz_iterations=10,
    pc_iterations=1,
    cost_col_floor=0,
    restarts=True,
    epochs_per_restart=None,
    restart_multiplier=1.0,
    restart_decay=0.2,
    necessary_decay=0.8,
    primal_stop=False,
    primal_stop_window=5,
    primal_stop_obj_tol=1e-4,
    halpern_reanchor_per_epoch=False,
    iterations_per_epoch_min=100,
    restart_check_every="auto",
):
    """
    Build a pure-JAX solve function for LPs with ``lp``'s sparsity pattern.

    Returns ``solve_fn(values=None, initial_solution=None) -> SolveCoreResult``.
    ``values`` is an ``LPValues`` (``lp.values()`` when omitted): the cost,
    the constraint matrices' stored nonzeros in ``lp``'s BCOO order, the
    right-hand sides and the bounds. ``initial_solution`` is an optional warm
    start in true units. The whole solve (scaling, the epoch loop to the
    certificate or ``max_epochs``, unscaling) is traceable, so ``solve_fn``
    can be ``jax.jit``-ed and ``jax.vmap``-ed over a batch of values.

    The algorithm is ``solve()``'s, with the same arguments and defaults, and
    a single solve reproduces ``solve()`` exactly. What needs the host is not
    available: no ``verbose`` log, no ``max_seconds``, no ``vertex_bias``,
    and the epoch length never changes during a solve (no
    ``iterations_per_epoch_decay``; ``iterations_per_epoch_min`` is capped at
    ``iterations_per_epoch``). ``max_epochs=None`` runs to the certificate,
    which never comes on an infeasible LP: set a budget when that can happen.
    """
    if not isinstance(lp, JaddleLP):
        lp = to_jaddle_sparse(lp)
    pattern = __pad_empty_blocks(lp)
    k_lo, k_hi, adaptive_eta_setting, halpern = _check_settings(
        dual_residual, termination_norm, restart_norm, update_mode, k_scale, adaptive_eta
    )
    float_dtype, compute_dtype = __profile_dtypes()
    n_eq = pattern.n_eq
    n_epochs = _UNBOUNDED_EPOCHS if not max_epochs else int(max_epochs)
    base_values = pattern.values()

    def solve_fn(values=None, initial_solution=None):
        v = base_values if values is None else _pad_values(values, pattern)
        if scale:
            dr, dc, eq_data, ineq_data, c, b_eq, b_ineq, lb, ub, c_max = (
                __scale_problem_jax(
                    jnp.asarray(v.A_eq_data, compute_dtype),
                    pattern.A_eq.indices,
                    jnp.asarray(v.A_ineq_data, compute_dtype),
                    pattern.A_ineq.indices,
                    jnp.asarray(v.c, compute_dtype),
                    jnp.asarray(v.b_eq, compute_dtype),
                    jnp.asarray(v.b_ineq, compute_dtype),
                    jnp.asarray(v.lower_bounds, compute_dtype),
                    jnp.asarray(v.upper_bounds, compute_dtype),
                    1e-8,
                    augmented_weight,
                    cost_col_floor,
                    ruiz_iter=ruiz_iterations,
                    pc_iter=pc_iterations,
                    clip_bounds=(1e-6, 1e6),
                    augmented=scaled_augmented,
                    scaled_objective=scaled_objective,
                    scaled_rhs=scaled_rhs,
                )
            )
            scaled = pattern.with_values(
                LPValues(
                    *(
                        x.astype(float_dtype)
                        for x in (c, eq_data, b_eq, ineq_data, b_ineq, lb, ub)
                    )
                )
            )
        else:
            scaled = pattern.with_values(v)
            dr = jnp.ones(scaled.b.shape[0], scaled.c.dtype)
            dc = jnp.ones(scaled.c.shape[0], scaled.c.dtype)
            c_max = jnp.ones((), scaled.c.dtype)

        eta = adaptive_eta_setting
        if eta == "auto":
            eta = 1 / estimate_augmented_spectral_norm(scaled)

        state_dtype = scaled.c.dtype
        row_scale_eq = dr[:n_eq].astype(state_dtype)
        row_scale_ineq = dr[n_eq:].astype(state_dtype)
        col_scale = dc.astype(state_dtype)
        c_max = jnp.asarray(c_max, state_dtype)
        if initial_solution is None:
            # lp.initial_solution(), on the device: the box-projected zero.
            start = SaddleState(
                primal=jnp.clip(
                    jnp.zeros_like(scaled.c), scaled.lower_bounds, scaled.upper_bounds
                ),
                dual_ineq=jnp.zeros(scaled.b.shape[0] - n_eq, state_dtype),
                dual_eq=jnp.zeros(n_eq, state_dtype),
            )
        else:
            start = SaddleState(
                primal=initial_solution.primal.astype(state_dtype) / col_scale,
                dual_ineq=initial_solution.dual_ineq.astype(state_dtype)
                / (row_scale_ineq * c_max),
                dual_eq=initial_solution.dual_eq.astype(state_dtype)
                / (row_scale_eq * c_max),
            )

        build_chunk, carry, *_ = _device_solver(
            scaled,
            c_true=scaled.c,
            c_max=c_max,
            col_scale=col_scale,
            jnp_row_scale_ineq=row_scale_ineq,
            jnp_row_scale_eq=row_scale_eq,
            initial_solution=start,
            initial_opt_state=None,
            k_init=k_init,
            k_lo=k_lo,
            k_hi=k_hi,
            adaptive_eta=eta,
            halpern=halpern,
            vertex_bias=0.0,
            primal_damping=primal_damping,
            dual_damping_ineq=dual_damping_ineq,
            dual_damping_eq=dual_damping_eq,
            primal_feasibility_tolerance=primal_feasibility_tolerance,
            dual_feasibility_tolerance=dual_feasibility_tolerance,
            dual_gap_tolerance=dual_gap_tolerance,
            dual_residual=dual_residual,
            termination_norm=termination_norm,
            restart_norm=restart_norm,
            verbose=False,
            log_every=1,
            average=average,
            report_best=report_best,
            update_mode=update_mode,
            k_theta=k_theta,
            k_update_per_epoch=k_update_per_epoch,
            iterations_per_epoch=iterations_per_epoch,
            restarts=restarts,
            epochs_per_restart=epochs_per_restart,
            restart_multiplier=restart_multiplier,
            restart_decay=restart_decay,
            necessary_decay=necessary_decay,
            primal_stop=primal_stop,
            primal_stop_window=primal_stop_window,
            primal_stop_obj_tol=primal_stop_obj_tol,
            halpern_reanchor_per_epoch=halpern_reanchor_per_epoch,
            iterations_per_epoch_decay=1.0,
            iterations_per_epoch_min=min(iterations_per_epoch_min, iterations_per_epoch),
            restart_check_every=restart_check_every,
            reference_objective=None,
        )
        carry = build_chunk(iterations_per_epoch)(carry, n_epochs)

        if report_best and average:
            output = _select_state(carry["used_avg"], carry["avg"], carry["state"])
        elif average:
            output = carry["avg"]
        else:
            output = carry["state"]
        output = _unscale_output(
            output,
            col_scale,
            row_scale_ineq,
            row_scale_eq,
            c_max,
            scale=bool(scale),
            scaled_objective=bool(scaled_objective),
        )
        stop_code = carry["stop_code"]
        return SolveCoreResult(
            solution=output,
            converged=stop_code > 0,
            stop_code=stop_code,
            epochs=carry["count"],
        )

    solve_fn.pattern = pattern
    return solve_fn


def _select_state(pred, a, b):
    return jax.tree.map(lambda x, y: jnp.where(pred, x, y), a, b)


def _pad_values(values, pattern):
    """Give an empty equality / inequality block the single zero row that
    ``__pad_empty_blocks`` adds to the pattern, so values taken from the
    original (unpadded) LP fit. Works on batched values too (the row is
    appended along the last axis)."""

    def pad(b, rows):
        if b.shape[-1] == 0 and rows == 1:
            return jnp.zeros(b.shape[:-1] + (1,), b.dtype)
        return b

    return values._replace(
        b_eq=pad(values.b_eq, pattern.n_eq),
        b_ineq=pad(values.b_ineq, pattern.A_ineq.shape[0]),
    )


def solve_batch(lp, values, **settings):
    """
    Solve a batch of LPs sharing ``lp``'s sparsity pattern: ``jax.vmap`` of
    ``make_solver(lp, **settings)``, jitted.

    ``values`` is an ``LPValues`` whose fields carry a leading batch axis;
    a field given without one (e.g. bounds shared by every member) is
    broadcast. Each member is scaled on its own, exactly as ``solve()`` would.
    Members iterate in lockstep until the last one stops (a finished member
    is frozen), so one hard member sets the batch's run time; ``max_epochs``
    bounds it.

    Batched members are not bit-identical to separate solves: XLA rounds
    batched reductions slightly differently, and the adaptive step size
    amplifies that over the iterations. Each member still certifies to the
    requested tolerance.

    Each call re-traces and compiles. To solve many batches of the same shape,
    jit ``jax.vmap(make_solver(lp, **settings))`` once and reuse it.

    Returns a ``SolveCoreResult`` with a leading batch axis on every field.
    """
    solve_fn = make_solver(lp, **settings)
    values = LPValues(*(jnp.asarray(v) for v in values))
    # A field is batched when it has one more axis than the LP's own field.
    axes = LPValues(
        *(
            0 if v.ndim == r.ndim + 1 else None
            for v, r in zip(values, solve_fn.pattern.values())
        )
    )
    if all(a is None for a in axes):
        raise ValueError("solve_batch: no field of `values` has a batch axis")
    return jax.jit(jax.vmap(solve_fn, in_axes=(axes,)))(values)


def make_optimal_value(lp, **settings):
    """
    The LP's optimal value as a differentiable function of its numbers.

    Returns ``value_fn(values) -> z*``, the objective ``cᵀx*`` of the solution
    ``make_solver(lp, **settings)`` finds for ``values`` (an ``LPValues``; ``lp``'s
    own when omitted). Its gradient comes from the envelope theorem, read off
    the primal-dual solution ``(x*, y*)`` with no differentiation through the
    iterations. With Jaddle's Lagrangian ``cᵀx + yᵀ(Ax − b)`` (``y_ineq >= 0``)
    and reduced cost ``r = c + Aᵀy*``:

    * ``∂z*/∂c = x*``, ``∂z*/∂b = −y*``;
    * ``∂z*/∂A_ij = y*_i x*_j`` on the stored nonzeros;
    * ``∂z*/∂lower_j = max(r_j, 0)`` and ``∂z*/∂upper_j = min(r_j, 0)`` where
      the bound is finite, else 0.

    These are exact where the optimal primal and dual solutions are unique.
    At a degenerate optimum ``z*`` is not differentiable and this returns one
    element of the subdifferential, the one at the solution found. The
    gradient is only as accurate as the solve: use tight tolerances, and check
    ``make_solver``'s ``converged`` when it matters. Works under ``jit`` and
    ``vmap``.
    """
    solve_fn = make_solver(lp, **settings)
    pattern = solve_fn.pattern
    eq_rows, eq_cols = pattern.A_eq.indices[:, 0], pattern.A_eq.indices[:, 1]
    ineq_rows, ineq_cols = pattern.A_ineq.indices[:, 0], pattern.A_ineq.indices[:, 1]
    base_values = pattern.values()

    def forward(values):
        padded = _pad_values(values, pattern)
        solution = solve_fn(padded).solution
        return padded.c @ solution.primal, (values, padded, solution)

    def backward(residuals, g):
        values, padded, solution = residuals
        x, y_eq, y_ineq = solution.primal, solution.dual_eq, solution.dual_ineq
        reduced_cost = padded.c + pattern.with_values(padded).A_T @ jnp.concatenate(
            [y_eq, y_ineq]
        )
        finite_lower = jnp.isfinite(padded.lower_bounds)
        finite_upper = jnp.isfinite(padded.upper_bounds)

        def unpad(grad_b, original):
            # The padded zero row is not one of the caller's values.
            return grad_b[..., : original.shape[-1]]

        grad = LPValues(
            c=x,
            A_eq_data=y_eq[eq_rows] * x[eq_cols],
            b_eq=unpad(-y_eq, values.b_eq),
            A_ineq_data=y_ineq[ineq_rows] * x[ineq_cols],
            b_ineq=unpad(-y_ineq, values.b_ineq),
            lower_bounds=jnp.where(finite_lower, jnp.maximum(reduced_cost, 0.0), 0.0),
            upper_bounds=jnp.where(finite_upper, jnp.minimum(reduced_cost, 0.0), 0.0),
        )
        return (jax.tree.map(lambda t, v: (g * t).astype(jnp.asarray(v).dtype), grad, values),)

    @jax.custom_vjp
    def value(values):
        return forward(values)[0]

    value.defvjp(forward, backward)

    def value_fn(values=None):
        return value(base_values if values is None else values)

    value_fn.pattern = pattern
    return value_fn


def make_perturbed_solution(
    lp, sigma=0.1, num_samples=16, antithetic=True, **settings
):
    """
    A smoothed, differentiable solution map ``c -> x*_σ(c)`` (the perturbed
    optimizer of Berthet et al., "Learning with Differentiable Perturbed
    Optimizers", 2020).

    An LP's solution is piecewise constant in its cost, so ``dx*/dc`` is zero
    almost everywhere. The perturbed solution ``x*_σ(c) = E[x*(c + σZ)]``
    (``Z`` standard normal) is smooth, and its Jacobian has the unbiased
    estimate ``E[x*(c + σZ) Zᵀ] / σ``. Both are estimated from
    ``num_samples`` solves, run as one batch with ``vmap``.

    Returns ``x_fn(c, key, values=None)``: ``c`` is the cost to perturb (its
    gradient is the only one computed), ``key`` a ``jax.random`` key, and
    ``values`` the LP's other numbers (``lp``'s own when omitted). With
    ``antithetic=True`` the samples come in ``±Z`` pairs, which cancels the
    estimate's odd-order noise (``num_samples`` must then be even). ``sigma``
    is in the units of ``c``: larger is smoother and more biased.

    The samples are solved to ``settings``' tolerances; a sample that does not
    converge within ``max_epochs`` still contributes its last point. The
    feasible region must be bounded: a perturbed cost can otherwise make a
    sample unbounded (measured: on ``min 3x₁ + 2x₂, x₁ + x₂ >= 4, x >= 0``
    samples with ``c₂ + σZ₂ < 0`` drove the mean of ``x₂`` to ~500).
    """
    if num_samples < 1 or (antithetic and num_samples % 2):
        raise ValueError(
            "num_samples must be >= 1, and even when antithetic=True"
        )
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    solve_fn = make_solver(lp, **settings)
    pattern = solve_fn.pattern
    batched = jax.vmap(
        lambda c, v: solve_fn(v._replace(c=c)).solution.primal, in_axes=(0, None)
    )

    def noise(key, c):
        if antithetic:
            half = jax.random.normal(key, (num_samples // 2,) + c.shape, c.dtype)
            return jnp.concatenate([half, -half])
        return jax.random.normal(key, (num_samples,) + c.shape, c.dtype)

    def forward(c, key, values):
        z = noise(key, c)
        xs = batched(c + sigma * z, values)
        return xs.mean(axis=0), (xs, z)

    def backward(residuals, g):
        xs, z = residuals
        # gᵀ J = E[(g · x*(c + σZ)) Z] / σ. No gradient for the key or the
        # LP's other numbers.
        grad_c = jnp.mean((xs @ g)[:, None] * z, axis=0) / sigma
        return grad_c, None, None

    @jax.custom_vjp
    def perturbed(c, key, values):
        return forward(c, key, values)[0]

    perturbed.defvjp(forward, backward)

    def x_fn(c, key, values=None):
        v = pattern.values() if values is None else _pad_values(values, pattern)
        return perturbed(jnp.asarray(c, v.c.dtype), key, v)

    x_fn.pattern = pattern
    return x_fn


def make_solution(lp, active_tol=1e-6, **settings):
    """
    The LP's solution ``x*`` as a function of its numbers, differentiable by
    implicit differentiation of the active constraints.

    Returns ``x_fn(values=None) -> x*`` (``values`` an ``LPValues``, ``lp``'s
    own when omitted). At a nondegenerate vertex the active constraints
    determine ``x*``: the equality rows and the inequality rows within
    ``active_tol`` of tight, restricted to the variables not within
    ``active_tol`` of a bound, form a square matrix ``M``. A VJP solves
    ``Mᵀλ = g`` once and gives

    * ``b``: ``λ`` on the active rows, 0 elsewhere;
    * ``A_ij``: ``−λ_i x*_j``;
    * the bound each variable sits at: ``(g − Aᵀλ)_j``;
    * ``c``: 0. An LP's solution is piecewise constant in its cost, so this is
      exact but rarely useful; ``make_perturbed_solution`` gives a smoothed
      ``dx*/dc``.

    ``active_tol`` is relative (``|x − bound| <= active_tol·(1 + |bound|)``,
    likewise for row slacks), so solve to tolerances well below it. The
    derivative is undefined unless the solution is a nondegenerate vertex;
    the VJP fails when the active set is not square (a degenerate vertex, or
    a point inside an optimal face, which first-order methods often return on
    degenerate LPs) or ``M`` is singular. The ``ValueError`` explaining which
    is raised in a host callback, so JAX surfaces it wrapped in a
    ``JaxRuntimeError`` whose message contains the original. The solve
    of ``Mᵀλ = g`` runs on the host with scipy (``jax.pure_callback``), so
    under ``vmap`` the members' VJPs run one after another.
    """
    import scipy.sparse.linalg as spla

    solve_fn = make_solver(lp, **settings)
    pattern = solve_fn.pattern
    n_eq = pattern.n_eq
    m, n = pattern.A.shape
    a_rows = np.asarray(pattern.A.indices[:, 0])
    a_cols = np.asarray(pattern.A.indices[:, 1])
    # A row with no stored entries (e.g. the 0 = 0 row padding an empty
    # block) never constrains x, so it is never active.
    row_has_entries = jnp.asarray(np.bincount(a_rows, minlength=m) > 0)
    eq_rows, eq_cols = pattern.A_eq.indices[:, 0], pattern.A_eq.indices[:, 1]
    ineq_rows, ineq_cols = pattern.A_ineq.indices[:, 0], pattern.A_ineq.indices[:, 1]

    def host_lambda(a_data, basic, active_rows, g):
        basic = np.asarray(basic, bool)
        active_rows = np.asarray(active_rows, bool)
        n_basic, n_active = int(basic.sum()), int(active_rows.sum())
        if n_basic != n_active:
            raise ValueError(
                "make_solution: the solution is not a nondegenerate vertex "
                f"({n_basic} variables off their bounds, {n_active} active rows), "
                "so dx*/dθ is undefined there. Solve to tighter tolerances, "
                "adjust active_tol, or use make_perturbed_solution."
            )
        lam = np.zeros(m, np.asarray(g).dtype)
        if n_basic:
            A = sp.csr_matrix((np.asarray(a_data), (a_rows, a_cols)), shape=(m, n))
            M = A[np.flatnonzero(active_rows)][:, np.flatnonzero(basic)].tocsc()
            try:
                lam_active = spla.splu(M.T.tocsc()).solve(np.asarray(g)[basic])
            except RuntimeError as exc:
                raise ValueError(
                    "make_solution: the active-constraint matrix is singular"
                ) from exc
            lam[active_rows] = lam_active
        return lam

    def forward(values):
        padded = _pad_values(values, pattern)
        x = solve_fn(padded).solution.primal
        return x, (values, padded, x)

    def backward(residuals, g):
        values, padded, x = residuals
        lp_v = pattern.with_values(padded)
        lower, upper = padded.lower_bounds, padded.upper_bounds
        at_lower = jnp.isfinite(lower) & (
            jnp.abs(x - lower) <= active_tol * (1 + jnp.abs(lower))
        )
        at_upper = (
            jnp.isfinite(upper)
            & (jnp.abs(x - upper) <= active_tol * (1 + jnp.abs(upper)))
            & ~at_lower
        )
        basic = ~(at_lower | at_upper)
        slack = lp_v.A @ x - lp_v.b
        active_rows = row_has_entries & (
            (jnp.arange(m) < n_eq)
            | (jnp.abs(slack) <= active_tol * (1 + jnp.abs(lp_v.b)))
        )
        lam = jax.pure_callback(
            host_lambda,
            jax.ShapeDtypeStruct((m,), x.dtype),
            lp_v.A.data,
            basic,
            active_rows,
            g,
            vmap_method="sequential",
        )
        w = g - lp_v.A_T @ lam
        lam_eq, lam_ineq = lam[:n_eq], lam[n_eq:]
        grad = LPValues(
            c=jnp.zeros_like(x),
            A_eq_data=-lam_eq[eq_rows] * x[eq_cols],
            b_eq=lam_eq[..., : values.b_eq.shape[-1]],
            A_ineq_data=-lam_ineq[ineq_rows] * x[ineq_cols],
            b_ineq=lam_ineq[..., : values.b_ineq.shape[-1]],
            lower_bounds=jnp.where(at_lower, w, 0.0),
            upper_bounds=jnp.where(at_upper, w, 0.0),
        )
        return (jax.tree.map(lambda t, v: t.astype(jnp.asarray(v).dtype), grad, values),)

    @jax.custom_vjp
    def solution(values):
        return forward(values)[0]

    solution.defvjp(forward, backward)

    def x_fn(values=None):
        return solution(pattern.values() if values is None else values)

    x_fn.pattern = pattern
    return x_fn


def solve_with_presolve(
    model,
    highs_options=None,
    finish_epochs=50,
    eliminate_defined_vars=False,
    reduced_solver=None,
    **solve_kwargs,
):
    """
    Presolve with HiGHS, solve the reduced LP with ``solve()``, and map the
    primal-dual solution back to the original problem with HiGHS's postsolve.

    ``model`` is a path to a file HiGHS reads (MPS, LP, ...), a
    ``highspy.Highs`` with a model loaded, a ``highspy.HighsLp``, or a Jaddle
    ``LP`` / ``JaddleLP``. Integer variables are relaxed: Jaddle solves the
    LP relaxation. ``highs_options`` (a dict) is applied to the HiGHS instance
    that presolves; other keyword arguments go to ``solve()``.

    ``eliminate_defined_vars=True`` also runs
    ``presolve.eliminate_defined_variables`` on HiGHS's reduced LP (dense
    rows defining an aggregate or hiding the objective, which HiGHS keeps but
    which stall first-order methods: gmut-*, proteindesign*, radiation*). Its
    own postsolve maps the primal and dual back before HiGHS's does.

    ``reduced_solver`` replaces ``solve()`` for the reduced LP (e.g.
    ``solve_with_polishing``); it is called with ``solve_kwargs`` and must
    return ``solve()``'s result keys.

    HiGHS's postsolve maps the duals as well as the primal values (no basis
    is needed), so the returned point carries a certificate for the problem
    as given. On neos-1593097 the postsolved pair certified on the original
    LP at the reduced solve's 1e-8 tolerance, with the objective matching
    HiGHS's optimum to 2e-10.

    Postsolve rebuilds eliminated rows' and columns' duals from the reduced
    ones, which is exact at a vertex but can enlarge a first-order solution's
    small dual errors (neos-1593097 at 1e-6: some runs certified on the
    original, one had dual residual 6.6e-5). When the postsolved point fails
    the original certificate, up to ``finish_epochs`` epochs of ``solve()``
    on the original LP, warm-started from it, finish the job (2 epochs on
    neos-1593097, against 23 from cold); the finished point is kept only if
    it certifies. ``finish_epochs=0`` disables this.

    Returns ``solve()``'s dict for the reduced solve, with these keys
    replaced or added:

    * ``"solution"``: a ``SaddleState`` for the original LP in Jaddle's
      standard form (``highs_helpers.rows_to_standard_form`` of the original
      rows; for an ``LP`` input, exactly that LP's rows), or ``None`` when
      presolve settled the problem as infeasible or unbounded.
    * ``"objective"``: the original objective at that point, constant offset
      included.
    * ``"highs_solution"``: the same point in HiGHS's row form for the
      original model, ``col_value``, ``col_dual``, ``row_value``, ``row_dual``.
    * ``"finish"``: the finishing solve's ``epochs`` and whether it
      ``certified`` (0 and False when it was not needed).
    * ``"certificate"``: ``evaluate_lp_certificate`` of the original LP at
      the postsolved point (l2 norm, PDLP dual residual, as ``solve()``
      tests).
    * ``"converged"``: whether that certificate meets the requested
      tolerances, i.e. whether the point is certified for the problem as
      given; ``"reduced_converged"`` is the reduced solve's own verdict and
      ``"stop_reason"`` its reason for stopping.
    * ``"presolve"``: HiGHS's presolve status and the sizes before and after.

    The two verdicts can differ, because the tolerances are relative and
    presolve can change the problem's scale. On leo1 presolve substitutes a
    dense objective row (coefficients ~1e7) into the cost, so ‖c‖ grows from
    1 to 3e9: the reduced solve certifies at 1e-6 with reduced costs ~10 in
    absolute terms, which against the original ‖c‖ = 1 is a relative dual
    residual of ~50. The objective still matches HiGHS's optimum to ~1e-8;
    only the dual certificate fails on the original.
    * ``"stop_reason"``: also ``"presolve_infeasible"`` or
      ``"presolve_unbounded_or_infeasible"`` when presolve decides the problem
      (then ``"converged"`` is False), and ``"presolve_solved"`` when it reduces
      the problem to nothing.
    """
    import highspy
    from jaddle import highs_helpers as hh

    highs = model if isinstance(model, highspy.Highs) else highspy.Highs()
    highs.setOptionValue("output_flag", False)
    for key, value in (highs_options or {}).items():
        highs.setOptionValue(key, value)
    if isinstance(model, str):
        status = highs.readModel(model)
        if status == highspy.HighsStatus.kError:
            raise ValueError(f"HiGHS could not read {model!r}")
    elif isinstance(model, highspy.HighsLp):
        highs.passModel(model)
    elif isinstance(model, (LP, JaddleLP)):
        highs.passModel(hh.jaddle_lp_to_highs(model))
    elif not isinstance(model, highspy.Highs):
        raise TypeError(f"unsupported model type {type(model).__name__}")
    n = highs.getNumCol()
    if n:
        highs.changeColsIntegrality(
            n, np.arange(n, dtype=np.int32), np.zeros(n, dtype=np.uint8)
        )
    original = highs.getLp()

    start = time.time()
    highs.presolve()
    presolve_seconds = time.time() - start
    presolve_status = highs.getModelPresolveStatus()
    PS = highspy.HighsPresolveStatus
    if presolve_status in (PS.kNullError, PS.kOptionsError):
        raise RuntimeError(f"HiGHS presolve failed: {presolve_status.name}")
    reduced = highs.getPresolvedLp()
    presolve = {
        "status": presolve_status.name,
        "rows": (original.num_row_, reduced.num_row_),
        "cols": (original.num_col_, reduced.num_col_),
        "seconds": presolve_seconds,
    }

    lp_original = hh.highs_to_standard_form_sparse(original)
    if presolve_status in (PS.kInfeasible, PS.kUnboundedOrInfeasible):
        return {
            "solution": None,
            "converged": False,
            "stop_reason": (
                "presolve_infeasible"
                if presolve_status == PS.kInfeasible
                else "presolve_unbounded_or_infeasible"
            ),
            "objective": None,
            "highs_solution": None,
            "certificate": None,
            "presolve": presolve,
            "epochs": 0,
        }

    if presolve_status == PS.kReducedToEmpty:
        result = {"converged": True, "stop_reason": "presolve_solved", "epochs": 0}
        reduced_solution = highspy.HighsSolution()
        reduced_solution.value_valid = True
        reduced_solution.dual_valid = True
    else:
        lp_reduced = hh.highs_to_standard_form_sparse(reduced)
        lp_solved, elimination = lp_reduced, None
        if eliminate_defined_vars:
            from jaddle import presolve as jaddle_presolve

            lp_solved, _, elimination = jaddle_presolve.eliminate_defined_variables(
                lp_reduced
            )
        result = (reduced_solver or solve)(lp_solved, **solve_kwargs)
        s = result["solution"]
        x = np.asarray(s.primal, dtype=np.float64)
        # solve() may have padded an empty block with one zero row; that row
        # is not one of the solved LP's rows.
        y_eq = np.asarray(s.dual_eq, dtype=np.float64)[: lp_solved.A_eq.shape[0]]
        y_ineq = np.asarray(s.dual_ineq, dtype=np.float64)[: lp_solved.A_ineq.shape[0]]
        if elimination is not None:
            x = elimination.primal(x)
            y_eq, y_ineq = elimination.dual(y_eq, y_ineq)
        presolve["eliminated_defined_vars"] = (
            0 if elimination is None else lp_reduced.c.shape[0] - lp_solved.c.shape[0]
        )
        A = sp.csc_matrix(
            (
                reduced.a_matrix_.value_,
                reduced.a_matrix_.index_,
                reduced.a_matrix_.start_,
            ),
            shape=(reduced.num_row_, reduced.num_col_),
        )
        row_dual = hh.jaddle_duals_to_highs(
            y_eq, y_ineq, np.asarray(reduced.row_lower_), np.asarray(reduced.row_upper_)
        )
        reduced_solution = highspy.HighsSolution()
        reduced_solution.col_value = x
        reduced_solution.row_value = A @ x
        reduced_solution.row_dual = row_dual
        reduced_solution.col_dual = np.asarray(reduced.col_cost_) - A.T @ row_dual
        reduced_solution.value_valid = True
        reduced_solution.dual_valid = True

    if highs.postsolve(reduced_solution) == highspy.HighsStatus.kError:
        raise RuntimeError("HiGHS postsolve failed")
    full = highs.getSolution()
    highs_solution = {
        "col_value": np.asarray(full.col_value),
        "col_dual": np.asarray(full.col_dual),
        "row_value": np.asarray(full.row_value),
        "row_dual": np.asarray(full.row_dual),
    }
    dual_eq, dual_ineq = hh.highs_duals_to_jaddle(
        highs_solution["row_dual"],
        np.asarray(original.row_lower_),
        np.asarray(original.row_upper_),
    )
    solution = SaddleState(
        primal=jnp.asarray(highs_solution["col_value"]),
        dual_ineq=jnp.asarray(dual_ineq),
        dual_eq=jnp.asarray(dual_eq),
    )
    lp_check = __pad_empty_blocks(to_jaddle_sparse(lp_original))

    def original_certificate(state):
        # The point padded to lp_check's rows (an empty block has one zero row).
        cert = evaluate_lp_certificate(
            lp_check,
            state.primal,
            jnp.zeros(lp_check.n_eq).at[: state.dual_eq.size].set(state.dual_eq),
            jnp.zeros(lp_check.A_ineq.shape[0])
            .at[: state.dual_ineq.size]
            .set(state.dual_ineq),
            norm=solve_kwargs.get("termination_norm", "l2"),
            dual_residual=solve_kwargs.get("dual_residual", "pdlp"),
        )
        cert = {key: float(value) for key, value in cert.items()}
        # Judged with solve()'s test, on the problem as given.
        certified = (
            cert["relative_primal_feasibility_residual"]
            <= solve_kwargs.get("primal_feasibility_tolerance", 1e-3)
            and cert["relative_dual_feasibility_residual"]
            <= solve_kwargs.get("dual_feasibility_tolerance", 1e-3)
            and np.isfinite(cert["duality_gap"])
            and cert["relative_gap_abs"]
            <= solve_kwargs.get("dual_gap_tolerance", 1e-3)
        )
        return cert, bool(certified)

    certificate, original_certified = original_certificate(solution)
    finish = {"epochs": 0, "certified": False}
    if not original_certified and finish_epochs:
        finish_kwargs = dict(solve_kwargs)
        if solve_kwargs.get("max_seconds") is not None:
            spent = time.time() - start
            finish_kwargs["max_seconds"] = max(solve_kwargs["max_seconds"] - spent, 1e-6)
        finished = solve(
            lp_original,
            initial_solution=solution,
            **{**finish_kwargs, "max_epochs": int(finish_epochs)},
        )
        finish["epochs"] = finished["epochs"]
        f = finished["solution"]
        # Drop the zero row solve() pads an empty block with.
        f = SaddleState(
            primal=f.primal,
            dual_ineq=f.dual_ineq[: lp_original.A_ineq.shape[0]],
            dual_eq=f.dual_eq[: lp_original.A_eq.shape[0]],
        )
        finished_cert, finished_certified = original_certificate(f)
        if finished_certified:
            finish["certified"] = True
            solution, certificate, original_certified = f, finished_cert, True
            # The same point in HiGHS's row form.
            A_orig = sp.csc_matrix(
                (
                    original.a_matrix_.value_,
                    original.a_matrix_.index_,
                    original.a_matrix_.start_,
                ),
                shape=(original.num_row_, original.num_col_),
            )
            x = np.asarray(f.primal, dtype=np.float64)
            row_dual = hh.jaddle_duals_to_highs(
                np.asarray(f.dual_eq),
                np.asarray(f.dual_ineq),
                np.asarray(original.row_lower_),
                np.asarray(original.row_upper_),
            )
            highs_solution = {
                "col_value": x,
                "col_dual": np.asarray(original.col_cost_) - A_orig.T @ row_dual,
                "row_value": A_orig @ x,
                "row_dual": row_dual,
            }
    result.update(
        solution=solution,
        converged=original_certified,
        reduced_converged=bool(result["converged"]),
        objective=float(np.asarray(original.col_cost_) @ np.asarray(solution.primal))
        + float(original.offset_),
        highs_solution=highs_solution,
        certificate=certificate,
        presolve=presolve,
        finish=finish,
    )
    return result


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
    return type(lp)(lp.c, A_eq, b_eq, A_ineq, b_ineq, lp.lower_bounds, lp.upper_bounds)


def __scipy_to_bcoo(A, dtype):
    """Sorted, deduplicated BCOO of the scipy matrix ``A`` with ``dtype`` data
    (see ``jaddle_basic_types.scipy_to_bcoo``: built on the host, no compiles)."""
    return scipy_to_bcoo(A, dtype)


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
        absdata = jnp.abs(jnp.concatenate([a_data, b, augmented_weight * c / c_norm]))
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
        absdata,
        row_idx,
        col_idx,
        ones_r,
        ones_c,
        ruiz_iter,
        True,
        clip_bounds,
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
        absdata,
        row_idx,
        col_idx,
        row_scale,
        col_scale,
        pc_iter,
        False,
        clip_bounds,
        threshold,
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


def evaluate_lp_certificate(
    lp: JaddleLP, primal, dual_eq, dual_ineq, norm="inf", dual_residual="projected"
):
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

    ``norm`` (``"inf"`` or ``"l2"``) and ``dual_residual`` (``"projected"`` or
    ``"pdlp"``) select the residual norm and reduced-cost split exactly as
    ``solve()``'s ``termination_norm`` / ``dual_residual`` do. The defaults keep
    this function's historical ∞-norm / projected certificate; pass
    ``norm="l2", dual_residual="pdlp"`` to reproduce ``solve()``'s default
    stopping test, which gates the gap on ``relative_gap_abs``.

    Returns a dict: ``objective``, ``dual_objective``, ``duality_gap``,
    ``relative_gap``, ``relative_gap_abs`` (the no-cancellation gap ``solve()``
    terminates on), ``primal_feasibility_residual``,
    ``dual_feasibility_residual`` and their ``relative_*`` forms.
    """
    if norm not in ("inf", "l2"):
        raise ValueError(f"norm must be 'inf' or 'l2', got {norm!r}")
    if dual_residual not in ("pdlp", "projected"):
        raise ValueError(
            f"dual_residual must be 'pdlp' or 'projected', got {dual_residual!r}"
        )
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
    if dual_residual == "pdlp":
        # Same absorption rule as solve()'s compute_epoch_metrics: a reduced
        # cost is absorbed by a finite bound of the matching sign only when the
        # primal sits near that bound, else it is a dual residual.
        lb_ok = finite_lower & (jnp.abs(primal - lower_bounds) <= jnp.abs(primal))
        ub_ok = finite_upper & (jnp.abs(primal - upper_bounds) <= jnp.abs(primal))
        absorb_lower = (reduced_cost > 0.0) & lb_ok
        absorb_upper = (reduced_cost < 0.0) & ub_ok
        box_infimum = jnp.where(
            absorb_lower, lower_term, jnp.where(absorb_upper, upper_term, 0.0)
        )
        dual_feasibility_violation = jnp.where(
            absorb_lower | absorb_upper, 0.0, jnp.abs(reduced_cost)
        )
    else:
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
    dual_feasibility_residual = _vector_norm(dual_feasibility_violation, norm)

    ineq_violations = jnp.maximum(grad_dual_ineq, 0.0)
    eq_violations = jnp.abs(grad_dual_eq)
    primal_feasibility_residual = _vector_norm(
        jnp.concatenate([eq_violations, ineq_violations]), norm
    )

    relative_primal_feasibility_residual = primal_feasibility_residual / (
        1.0 + _vector_norm(lp.b, norm)
    )
    relative_dual_feasibility_residual = dual_feasibility_residual / (
        1.0 + _vector_norm(lp.c, norm)
    )

    gap_bound_comp = reduced_cost @ primal - jnp.sum(box_infimum)
    gap_ineq_comp = -(dual_ineq @ grad_dual_ineq)
    gap_eq_comp = -(dual_eq @ grad_dual_eq)
    duality_gap = gap_bound_comp + gap_ineq_comp + gap_eq_comp

    dual_objective = objective_value - duality_gap
    gap_denom = 1.0 + jnp.abs(objective_value) + jnp.abs(dual_objective)
    relative_gap = jnp.abs(duality_gap) / gap_denom
    relative_gap_abs = (
        jnp.abs(gap_bound_comp) + jnp.abs(gap_ineq_comp) + jnp.abs(gap_eq_comp)
    ) / gap_denom

    return {
        "objective": objective_value,
        "dual_objective": dual_objective,
        "duality_gap": duality_gap,
        "relative_gap": relative_gap,
        "relative_gap_abs": relative_gap_abs,
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


def _with_vectors(lp: JaddleLP, c=None, b_eq=None, b_ineq=None, lower=None, upper=None):
    """Shallow copy of ``lp`` with some of its vectors replaced. The constraint
    matrices (and their sorted BCOO forms) are shared, not rebuilt."""
    out = copy.copy(lp)
    if c is not None:
        out.c = c
    if b_eq is not None:
        out.b_eq = b_eq
    if b_ineq is not None:
        out.b_ineq = b_ineq
    if lower is not None:
        out.lower_bounds = lower
    if upper is not None:
        out.upper_bounds = upper
    out.b = jnp.concatenate([out.b_eq, out.b_ineq])
    return out


def primal_feasibility_lp(lp: JaddleLP) -> JaddleLP:
    """The primal feasibility problem (Eq. 6 of Applegate et al.,
    arXiv:2501.07018): ``lp`` with its objective set to zero."""
    return _with_vectors(lp, c=jnp.zeros_like(lp.c))


def homogenised_dual_lp(lp: JaddleLP, primal=None) -> JaddleLP:
    """The LP whose saddle solve yields a dual-feasible point of ``lp`` (Eq. 7
    of Applegate et al., arXiv:2501.07018, in PDLP's form): ``b = 0`` and
    bounds homogenised to 0 or ±inf. Its optimal primal is 0, and its dual
    optima ``y`` are exactly those whose reduced cost ``c + Aᵀy`` every kept
    bound absorbs.

    Without ``primal`` every finite bound is kept (PDLP). A boxed variable
    then becomes fixed at 0 and absorbs a reduced cost of either sign, which
    ``solve()``'s ``dual_residual="pdlp"`` certificate does not accept unless
    x is near the matching bound: measured on a random LP, a dual polished
    this way went from DFR 1.5e-3 to 7.7e-2. With ``primal`` a bound is kept
    only where that certificate would absorb at ``primal`` (finite and
    ``|x - bound| <= |x|``), so a dual optimum has zero DFR at ``primal``."""
    lower, upper = lp.lower_bounds, lp.upper_bounds
    keep_lower, keep_upper = jnp.isfinite(lower), jnp.isfinite(upper)
    if primal is not None:
        keep_lower &= jnp.abs(primal - lower) <= jnp.abs(primal)
        keep_upper &= jnp.abs(primal - upper) <= jnp.abs(primal)
    return _with_vectors(
        lp,
        b_eq=jnp.zeros_like(lp.b_eq),
        b_ineq=jnp.zeros_like(lp.b_ineq),
        lower=jnp.where(keep_lower, 0.0, -jnp.inf).astype(lower.dtype),
        upper=jnp.where(keep_upper, 0.0, jnp.inf).astype(upper.dtype),
    )


def _primal_polish_solve(lp: JaddleLP, warm_start: SaddleState, **kwargs):
    # Optimal dual of a zero-objective problem is 0, so start the dual there.
    # k_init is left to the ||c||/||b|| derivation: c = 0 sends it to the low
    # clamp, i.e. a large primal step and no objective force — what a
    # feasibility projection wants (seeding the main solve's learned k left
    # mzzv11's 2-epoch feasibility problem unconverged at 20 epochs).
    kwargs.setdefault("max_epochs", 1)
    init = SaddleState(
        primal=warm_start.primal,
        dual_ineq=jnp.zeros_like(warm_start.dual_ineq),
        dual_eq=jnp.zeros_like(warm_start.dual_eq),
    )
    return solve(primal_feasibility_lp(lp), initial_solution=init, **kwargs)


def _dual_polish_solve(lp: JaddleLP, warm_start: SaddleState, **kwargs):
    # Mirror image: the homogenised problem's optimal primal is 0. Its bounds
    # follow the warm start's primal (see homogenised_dual_lp).
    kwargs.setdefault("max_epochs", 1)
    init = SaddleState(
        primal=jnp.zeros_like(warm_start.primal),
        dual_ineq=warm_start.dual_ineq,
        dual_eq=warm_start.dual_eq,
    )
    return solve(
        homogenised_dual_lp(lp, warm_start.primal), initial_solution=init, **kwargs
    )


def primal_polish(
    lp: JaddleLP,
    warm_start: SaddleState,
    **kwargs,
):
    """
    Algorithm 4's primal step (Applegate et al., arXiv:2501.07018): a short
    ``solve()`` of the primal feasibility problem (``c = 0``) warm-started at
    ``warm_start.primal`` with a zero dual. Returns the polished primal.

    The zero objective leaves the solver free to wander anywhere in the
    feasible set, so this must be a short burst from a point that is already
    nearly feasible and nearly optimal — not a solve to convergence. The burst
    is ``max_epochs`` epochs (default 1) of ``iterations_per_epoch``; PDLP uses
    an eighth of the main solve's iterations so far. Other keyword arguments
    go to ``solve()``.
    """
    if not isinstance(lp, JaddleLP):
        lp = to_jaddle_sparse(lp)
    return _primal_polish_solve(lp, warm_start, **kwargs)["solution"].primal


def dual_polish(
    lp: JaddleLP,
    warm_start: SaddleState,
    **kwargs,
):
    """
    Algorithm 4's dual step: a short ``solve()`` of the homogenised problem
    (``b = 0``, bounds moved to 0 or ±inf; see ``homogenised_dual_lp``)
    warm-started at the dual of ``warm_start`` with a zero primal. Returns the
    polished ``(dual_eq, dual_ineq)``. Same burst semantics as
    ``primal_polish``.

    The homogenised bounds are chosen from ``warm_start.primal``: only bounds
    the primal is near can absorb a reduced cost, matching ``solve()``'s
    default ``dual_residual="pdlp"`` certificate. Pair the result with that
    same primal.
    """
    if not isinstance(lp, JaddleLP):
        lp = to_jaddle_sparse(lp)
    sol = _dual_polish_solve(lp, warm_start, **kwargs)["solution"]
    return sol.dual_eq, sol.dual_ineq


# Main-solve arguments that must not reach the polishing sub-solves: they
# describe the main problem's start point, objective or heuristic stop.
_POLISH_DROPPED_KWARGS = frozenset(
    {
        "initial_solution",
        "initial_opt_state",
        "k_init",
        "reference_objective",
        "primal_stop",
        "primal_stop_window",
        "primal_stop_obj_tol",
        "vertex_bias",
        "vertex_bias_seed",
    }
)


def solve_with_polishing(
    lp: JaddleLP,
    tol=1e-6,
    primal_feasibility_tolerance=None,
    dual_feasibility_tolerance=None,
    dual_gap_tolerance=None,
    max_epochs=None,
    max_seconds=None,
    first_polish_epoch=16,
    polish_fraction=0.125,
    **kwargs,
):
    """
    ``solve()`` with PDLP feasibility polishing (Applegate et al.,
    arXiv:2501.07018, Algorithm 4).

    First-order LP solvers often close the duality gap well before the primal
    and dual residuals reach a tight tolerance. Polishing targets that tail.
    The main solve runs in chunks whose total length doubles (``first_polish_epoch``,
    then 2×, 4×, … epochs). After each chunk that ends uncertified with the
    gap already within ``dual_gap_tolerance``, each side that is still
    infeasible is polished by a short sub-solve of ``polish_fraction`` times
    the main epochs so far:

    * primal: ``primal_polish`` (``c = 0``) from the current primal;
    * dual: ``dual_polish`` (homogenised ``b = 0`` problem) from the current dual.

    The recombined pair is checked against the same certificate ``solve()``
    uses; if it passes it is returned, otherwise it is discarded and the main
    solve resumes from its own (unpolished) point, so polishing never makes the
    result worse. Polishing costs at most about ``2 * polish_fraction`` of the
    main solve's epochs.

    Polishing is a finishing tool. It cannot rescue a solve that is stuck far
    from tolerance: a feasibility sub-solve warm-started there inherits the
    stall, and a cold-started feasible point ignores the objective.

    Each chunk boundary is a warm restart (``initial_solution`` /
    ``initial_opt_state`` carry the point, ``k`` and ``eta``; the averaging
    resets), and each ``solve()`` call repeats the scaling setup.

    Args:
        tol: Default for the three tolerances below.
        primal_feasibility_tolerance, dual_feasibility_tolerance,
            dual_gap_tolerance: As in ``solve()``; ``None`` means ``tol``. Set
            the gap tolerance looser than the feasibility ones to get PDLP's
            "feasible, approximately optimal" mode.
        max_epochs: Main-solve epoch budget across all chunks (polishing
            epochs are not counted). ``None`` = no limit.
        max_seconds: Wall-clock budget for the whole call, polishing included.
        first_polish_epoch: Length of the first main chunk, i.e. the earliest
            epoch at which polishing can fire.
        polish_fraction: Per-side polishing budget as a fraction of the main
            epochs so far (PDLP: 1/8).
        **kwargs: Passed to ``solve()``. The sub-solves get them too, except
            the start point, ``k_init``, ``vertex_bias``, ``primal_stop`` and
            ``reference_objective``.

    Returns:
        ``solve()``'s result dict, with ``"epochs"`` summed over the main
        chunks, ``"solve_seconds"`` and ``"corrected_seconds"`` set to the
        whole call's wall time, and a ``"polish"`` dict: ``attempts``,
        ``epochs`` (sub-solve epochs, both sides) and ``polished`` (whether the
        returned point came from polishing).
    """
    entry_time = time.time()
    if not isinstance(lp, JaddleLP):
        lp = to_jaddle_sparse(lp)
    # solve() pads an empty constraint class with a zero row and returns duals
    # of that padded size; pad here so the certificate below sees the same LP.
    lp = __pad_empty_blocks(lp)
    if first_polish_epoch < 1:
        raise ValueError("first_polish_epoch must be >= 1")
    if not 0.0 < polish_fraction:
        raise ValueError("polish_fraction must be > 0")

    tolerances = {
        "primal_feasibility_tolerance": (
            tol if primal_feasibility_tolerance is None else primal_feasibility_tolerance
        ),
        "dual_feasibility_tolerance": (
            tol if dual_feasibility_tolerance is None else dual_feasibility_tolerance
        ),
        "dual_gap_tolerance": tol if dual_gap_tolerance is None else dual_gap_tolerance,
    }
    p_tol = tolerances["primal_feasibility_tolerance"]
    d_tol = tolerances["dual_feasibility_tolerance"]
    g_tol = tolerances["dual_gap_tolerance"]
    # Judge points with solve()'s own stopping test.
    cert_options = {
        "norm": kwargs.get("termination_norm", "l2"),
        "dual_residual": kwargs.get("dual_residual", "pdlp"),
    }
    verbose = kwargs.get("verbose", False)
    sub_kwargs = {k: v for k, v in kwargs.items() if k not in _POLISH_DROPPED_KWARGS}
    state = kwargs.pop("initial_solution", None)
    opt_state = kwargs.pop("initial_opt_state", None)

    def seconds_left():
        if max_seconds is None:
            return None
        return max_seconds - (time.time() - entry_time)

    main_epochs = 0
    polish_epochs = 0
    attempts = 0
    polished = False
    chunk = int(first_polish_epoch)
    result = None

    while True:
        if max_epochs is not None:
            chunk = min(chunk, int(max_epochs) - main_epochs)
        left = seconds_left()
        if left is not None and left <= 0:
            if result is not None:
                result["stop_reason"] = "time_limit"
                break
            left = 1e-6  # spent before the first chunk: solve() reports it
        result = solve(
            lp,
            max_epochs=chunk,
            max_seconds=left,
            initial_solution=state,
            initial_opt_state=opt_state,
            **tolerances,
            **kwargs,
        )
        main_epochs += result["epochs"]
        state, opt_state = result["solution"], result["opt_state"]
        # Anything but an exhausted chunk (certified, primal_stall, time limit,
        # interrupted) ends the solve, as does the overall epoch budget.
        if result["stop_reason"] != "max_epochs" or (
            max_epochs is not None and main_epochs >= max_epochs
        ):
            break

        cert = evaluate_lp_certificate(
            lp, state.primal, state.dual_eq, state.dual_ineq, **cert_options
        )
        # PDLP polishes only once the gap has closed: polishing does not
        # improve the gap, so an earlier candidate cannot certify.
        if float(cert["relative_gap_abs"]) <= g_tol:
            attempts += 1
            budget = max(1, int(np.ceil(polish_fraction * main_epochs)))
            if verbose:
                print(
                    f"Feasibility polish {attempts} after {main_epochs} epochs "
                    f"(PFR {float(cert['relative_primal_feasibility_residual']):.2e}, "
                    f"DFR {float(cert['relative_dual_feasibility_residual']):.2e}), "
                    f"{budget} epochs per side"
                )
            primal = state.primal
            dual_eq, dual_ineq = state.dual_eq, state.dual_ineq
            interrupted = False
            current = cert
            for side, residual, limit in (
                ("primal", "relative_primal_feasibility_residual", p_tol),
                ("dual", "relative_dual_feasibility_residual", d_tol),
            ):
                left = seconds_left()
                if float(current[residual]) <= limit or (
                    left is not None and left <= 0
                ):
                    continue
                polish_solve = (
                    _primal_polish_solve if side == "primal" else _dual_polish_solve
                )
                # The dual side runs second and homogenises its bounds at the
                # already-polished primal, the one it will be paired with.
                sub = polish_solve(
                    lp,
                    SaddleState(
                        primal=primal, dual_ineq=state.dual_ineq, dual_eq=state.dual_eq
                    ),
                    **{
                        **sub_kwargs,
                        **tolerances,
                        "max_epochs": budget,
                        "max_seconds": left,
                    },
                )
                polish_epochs += sub["epochs"]
                if sub["stop_reason"] == "interrupted":
                    interrupted = True
                    break
                if side == "primal":
                    primal = sub["solution"].primal
                    # Moving x changes which bounds absorb reduced costs, so
                    # re-test the dual side at the polished primal.
                    current = evaluate_lp_certificate(
                        lp, primal, dual_eq, dual_ineq, **cert_options
                    )
                else:
                    dual_eq = sub["solution"].dual_eq
                    dual_ineq = sub["solution"].dual_ineq
            if interrupted:
                result["stop_reason"] = "interrupted"
                break

            candidate = evaluate_lp_certificate(
                lp, primal, dual_eq, dual_ineq, **cert_options
            )
            if (
                float(candidate["relative_primal_feasibility_residual"]) <= p_tol
                and float(candidate["relative_dual_feasibility_residual"]) <= d_tol
                and bool(jnp.isfinite(candidate["duality_gap"]))
                and float(candidate["relative_gap_abs"]) <= g_tol
            ):
                if verbose:
                    print("Polished point is certified.")
                polished = True
                result["solution"] = SaddleState(
                    primal=primal, dual_ineq=dual_ineq, dual_eq=dual_eq
                )
                result["converged"] = True
                result["stop_reason"] = "certificate"
                break
            if verbose:
                print(
                    "Polished point not certified "
                    f"(PFR {float(candidate['relative_primal_feasibility_residual']):.2e}, "
                    f"DFR {float(candidate['relative_dual_feasibility_residual']):.2e}, "
                    f"gap {float(candidate['relative_gap_abs']):.2e}); resuming."
                )

        # Double the main solve's total length.
        chunk = main_epochs

    wall = time.time() - entry_time
    result["epochs"] = main_epochs
    result["solve_seconds"] = wall
    result["corrected_seconds"] = wall
    result["polish"] = {
        "attempts": attempts,
        "epochs": polish_epochs,
        "polished": polished,
    }
    return result


# %%


def _project_rays(lp: JaddleLP, primal_ray, dual_ray_eq, dual_ray_ineq):
    """Project candidate rays onto the cones a certificate needs: the primal
    ray onto the recession cone of the bounds (d_j >= 0 under a finite lower
    bound, <= 0 under a finite upper one, so boxed variables get 0) and the
    inequality part of the dual ray onto y >= 0. A certificate for the
    projected ray is a certificate, so projecting only discards noise."""
    d = jnp.where(jnp.isfinite(lp.lower_bounds), jnp.maximum(primal_ray, 0.0), primal_ray)
    d = jnp.where(jnp.isfinite(lp.upper_bounds), jnp.minimum(d, 0.0), d)
    return d, dual_ray_eq, jnp.maximum(dual_ray_ineq, 0.0)


def evaluate_infeasibility_certificate(
    lp: JaddleLP, primal_ray, dual_ray_eq, dual_ray_ineq
):
    """
    PDLP's infeasibility tests (Applegate et al., "Infeasibility detection
    with primal-dual hybrid gradient for large-scale linear programming",
    2021) for candidate rays, in true units and ∞-norms. The rays are first
    projected onto their cones (see ``_project_rays``); both tests are
    invariant to the rays' scale.

    Primal infeasibility (Farkas): a dual ray ``y`` (``y_ineq >= 0``) with
    ``q = Aᵀy`` and dual ray objective ``F = Σ_j inf_{l_j<=x_j<=u_j} q_j x_j
    − bᵀy``, where only the components of ``q`` some finite bound absorbs
    count towards the infimum and the rest are its infeasibility. If the
    infeasibility were 0 and ``F > 0``, every x in the box would have
    ``yᵀ(Ax − b) > 0``, which no feasible x can. The test is
    ``F > 0`` and ``infeasibility / F <= tol``; it then holds for no LP with a
    feasible point of ‖x‖₁ < 1/tol.

    Dual infeasibility (the primal is unbounded if feasible): a primal ray
    ``d`` in the bounds' recession cone with ``cᵀd < 0`` and infeasibility
    ``max(‖A_eq d‖∞, ‖max(A_ineq d, 0)‖∞)``. The test is ``cᵀd < 0`` and
    ``infeasibility / (−cᵀd) <= tol``; it then holds for no LP with a dual
    feasible point of ‖y‖₁ < 1/tol.

    A ray objective counts as positive (negative) only when it exceeds
    ``1000·eps`` times the sum of the magnitudes it is computed from, so
    rounding noise in a ray with zero infeasibility cannot certify.

    Returns a dict: ``dual_ray_objective``, ``dual_ray_infeasibility``,
    ``primal_infeasibility_ratio`` (the Farkas test's ratio, ``inf`` unless
    ``F > 0``), ``primal_ray_objective``, ``primal_ray_infeasibility`` and
    ``dual_infeasibility_ratio`` (``inf`` unless ``cᵀd < 0``).
    """
    d, y_eq, y_ineq = _project_rays(lp, primal_ray, dual_ray_eq, dual_ray_ineq)
    finite_lower = jnp.isfinite(lp.lower_bounds)
    finite_upper = jnp.isfinite(lp.upper_bounds)
    lower = jnp.where(finite_lower, lp.lower_bounds, 0.0)
    upper = jnp.where(finite_upper, lp.upper_bounds, 0.0)

    # Farkas ray. A positive q_j is bounded below only by a finite lower
    # bound, a negative one only by a finite upper bound.
    y = jnp.concatenate([y_eq, y_ineq])
    q = lp.A_T @ y
    absorb_lower = (q > 0.0) & finite_lower
    absorb_upper = (q < 0.0) & finite_upper
    box_infimum = jnp.where(
        absorb_lower, q * lower, jnp.where(absorb_upper, q * upper, 0.0)
    )
    dual_ray_objective = jnp.sum(box_infimum) - lp.b @ y
    dual_ray_infeasibility = jnp.max(
        jnp.where(absorb_lower | absorb_upper, 0.0, jnp.abs(q)), initial=0.0
    )
    # An objective within rounding of zero is no evidence: a ray with zero
    # infeasibility would otherwise certify on noise alone.
    noise = 1e3 * jnp.finfo(q.dtype).eps
    dual_significant = dual_ray_objective > noise * (
        jnp.sum(jnp.abs(box_infimum)) + jnp.abs(lp.b) @ jnp.abs(y)
    )
    primal_infeasibility_ratio = jnp.where(
        dual_significant,
        dual_ray_infeasibility
        / jnp.where(dual_significant, dual_ray_objective, 1.0),
        jnp.inf,
    )

    # Unbounded ray.
    Ad = lp.A @ d
    primal_ray_objective = lp.c @ d
    primal_ray_infeasibility = jnp.maximum(
        jnp.max(jnp.abs(Ad[: lp.n_eq]), initial=0.0),
        jnp.max(jnp.maximum(Ad[lp.n_eq :], 0.0), initial=0.0),
    )
    primal_significant = -primal_ray_objective > noise * (jnp.abs(lp.c) @ jnp.abs(d))
    dual_infeasibility_ratio = jnp.where(
        primal_significant,
        primal_ray_infeasibility
        / jnp.where(primal_significant, -primal_ray_objective, 1.0),
        jnp.inf,
    )

    return {
        "dual_ray_objective": dual_ray_objective,
        "dual_ray_infeasibility": dual_ray_infeasibility,
        "primal_infeasibility_ratio": primal_infeasibility_ratio,
        "primal_ray_objective": primal_ray_objective,
        "primal_ray_infeasibility": primal_ray_infeasibility,
        "dual_infeasibility_ratio": dual_infeasibility_ratio,
    }


def _min_norm_correction(M, rhs, iter_lim=2000):
    """Minimum-norm δ with M δ = rhs (least squares if inconsistent). The
    iteration cap bounds the cost of a check on a large LP; an unconverged
    correction just fails the certificate test that follows."""
    from scipy.sparse.linalg import lsqr

    if M.shape[0] == 0 or M.shape[1] == 0:
        return np.zeros(M.shape[1])
    return lsqr(M, rhs, atol=1e-15, btol=1e-15, iter_lim=iter_lim)[0]


def _refine_rays(A, n_eq, lower, upper, d, y_eq, y_ineq):
    """
    Remove the small infeasibility of nearly-certifying rays while keeping
    their sign pattern, on the host with scipy.

    Farkas ray: the components of ``q = Aᵀy`` no bound absorbs are set to 0
    exactly by the minimum-norm change to the rows ``y`` already uses
    (equality rows, and inequality rows with ``y > 0``). Unbounded ray: the
    equality rows and the violated inequality rows of ``A d`` are set to 0 by
    the minimum-norm change to the columns ``d`` already moves (plus free
    columns). PDHG's iterate differences reach the right pattern long before
    their infeasibility falls to a tight tolerance (it decays like 1/k), so
    this turns an early, rough ray into a certificate.
    The result must still be checked with ``evaluate_infeasibility_certificate``.
    """
    finite_lower, finite_upper = np.isfinite(lower), np.isfinite(upper)

    y = np.concatenate([y_eq, y_ineq])
    q = A.T @ y
    absorbed = ((q > 0) & finite_lower) | ((q < 0) & finite_upper)
    cols = np.flatnonzero(~absorbed & (q != 0))
    rows = np.concatenate([np.arange(n_eq), n_eq + np.flatnonzero(y_ineq > 0)])
    if cols.size:
        y = y.copy()
        y[rows] += _min_norm_correction(A[rows][:, cols].T.tocsr(), -q[cols])

    Ad = A @ d
    rows = np.concatenate(
        [np.arange(n_eq), n_eq + np.flatnonzero(Ad[n_eq:] > 0)]
    )
    cols = np.flatnonzero((d != 0) | (~finite_lower & ~finite_upper))
    if rows.size:
        d = d.copy()
        d[cols] += _min_norm_correction(A[rows][:, cols].tocsr(), -Ad[rows])

    return d, y[:n_eq], y[n_eq:]


def detect_infeasibility(
    lp: JaddleLP,
    tol=1e-8,
    max_epochs=None,
    max_seconds=None,
    first_check_epoch=4,
    refine=True,
    **kwargs,
):
    """
    Detect an infeasible or unbounded LP the PDLP way: on such a problem the
    PDHG iterates diverge, and their per-iteration difference converges to a
    direction (the infimal displacement vector) whose dual part is a Farkas
    ray when the primal is infeasible and whose primal part is an unbounded
    ray when the dual is infeasible.

    ``solve()`` runs in chunks whose total length doubles (``first_check_epoch``,
    then 2×, 4×, … epochs), each warm-started from the last iterate with its
    step-size state. After each chunk, the difference between the chunk's end
    and start points is tested with ``evaluate_infeasibility_certificate``
    against ``tol``. If ``solve()`` certifies optimality instead, the LP is
    feasible and bounded.

    The raw difference converges slowly: its infeasibility ratio falls like
    1/iterations (measured on a random LP with a contradictory row pair: 1e-4
    after ~100 epochs, still ~1e-6 after 4000). Its sign pattern settles much
    earlier, so with ``refine=True`` (default) a ray that fails the test is
    corrected by ``_refine_rays`` (a minimum-norm least-squares fix on the
    host, with scipy) and re-tested; on that LP the refined ray certified to
    1e-14 from 64 epochs.

    The certificates are sound up to scale: a primal-infeasibility result
    rules out any feasible point with ‖x‖₁ < 1/tol, a dual-infeasibility
    result any dual feasible point with ‖y‖₁ < 1/tol (see
    ``evaluate_infeasibility_certificate``). Run in float64 for tight
    ``tol``.

    The solve keeps ``solve()``'s restarted averaging and ``k`` rebalancing
    (PDLP detects on restarted PDHG too) but sets ``report_best=False``, since
    a best-merit point can freeze on a diverging trajectory. Measured on six
    netlib ``infeas`` problems (30 s each): last-iterate PDHG with restarts
    off and ``k`` frozen detected none; restarted averaging detected bgdbg1
    and reactor; adding ``k`` rebalancing (these defaults) also detected box1
    and ex72a, and left forest6 at 2.5e-8.

    Args:
        tol: Relative tolerance for both infeasibility tests (PDLP's default
            is 1e-8).
        max_epochs: Epoch budget across all chunks. ``None`` = no limit.
        max_seconds: Wall-clock budget for the whole call.
        first_check_epoch: Length of the first chunk, i.e. the earliest check.
        refine: Try the least-squares ray refinement when the raw ray fails.
        **kwargs: Passed to ``solve()``, including its optimality tolerances,
            which decide the ``"optimal"`` outcome. They default to ``tol``
            (not ``solve()``'s 1e-3), so an almost-feasible LP is not
            reported optimal.

    Returns:
        dict with
            * ``"status"``: ``"primal_infeasible"``, ``"dual_infeasible"``,
              ``"optimal"`` (``solve()`` certified optimality) or
              ``"undetermined"`` (budget spent, interrupted, or ``solve()``
              stopped on ``primal_stop``). When both tests pass,
              ``"primal_infeasible"`` wins.
            * ``"primal_ray"``, ``"dual_ray_eq"``, ``"dual_ray_ineq"``: the last
              candidate rays, projected onto their cones and scaled to unit
              ∞-norm (``None`` before the first check).
            * ``"certificate"``: ``evaluate_infeasibility_certificate``'s dict
              for those rays, as floats (``None`` before the first check).
            * ``"epochs"``, ``"seconds"``: epochs run and the whole call's wall
              time.
            * ``"result"``: the last ``solve()`` result.
    """
    entry_time = time.time()
    if not isinstance(lp, JaddleLP):
        lp = to_jaddle_sparse(lp)
    lp = __pad_empty_blocks(lp)
    if first_check_epoch < 1:
        raise ValueError("first_check_epoch must be >= 1")
    for key, value in (
        ("report_best", False),
        # "optimal" must be as strict as an infeasibility claim: at solve()'s
        # default 1e-3, netlib's almost-feasible cplex2 certified as optimal.
        ("primal_feasibility_tolerance", tol),
        ("dual_feasibility_tolerance", tol),
        ("dual_gap_tolerance", tol),
    ):
        kwargs.setdefault(key, value)
    verbose = kwargs.get("verbose", False)

    state = kwargs.pop("initial_solution", None)
    if state is None:
        state = lp.initial_solution()
    opt_state = kwargs.pop("initial_opt_state", None)

    if refine:
        A_host = __convert_to_scipy(lp.A).tocsr()
        lower_host = np.asarray(lp.lower_bounds)
        upper_host = np.asarray(lp.upper_bounds)

    def check(d, y_eq, y_ineq):
        # The rays scaled to unit ∞-norm (the dual ray as a whole), and their
        # certificate as floats.
        cert = {
            key: float(value)
            for key, value in evaluate_infeasibility_certificate(
                lp, d, y_eq, y_ineq
            ).items()
        }
        d_scale = float(jnp.max(jnp.abs(d), initial=0.0))
        y_scale = float(
            jnp.max(jnp.abs(jnp.concatenate([y_eq, y_ineq])), initial=0.0)
        )
        rays = (
            d / d_scale if d_scale > 0.0 else d,
            y_eq / y_scale if y_scale > 0.0 else y_eq,
            y_ineq / y_scale if y_scale > 0.0 else y_ineq,
        )
        return rays, cert

    def passes(cert):
        return (
            cert["primal_infeasibility_ratio"] <= tol
            or cert["dual_infeasibility_ratio"] <= tol
        )

    status = "undetermined"
    rays = (None, None, None)
    cert = None
    epochs = 0
    chunk = int(first_check_epoch)
    result = None

    while True:
        if max_epochs is not None:
            chunk = min(chunk, int(max_epochs) - epochs)
        left = None
        if max_seconds is not None:
            left = max_seconds - (time.time() - entry_time)
            if left <= 0:
                if result is not None:
                    break
                left = 1e-6  # spent before the first chunk: solve() reports it
        result = solve(
            lp,
            max_epochs=chunk,
            max_seconds=left,
            initial_solution=state,
            initial_opt_state=opt_state,
            **kwargs,
        )
        epochs += result["epochs"]
        new = result["solution"]
        if result["stop_reason"] == "certificate":
            status = "optimal"
            break
        if result["stop_reason"] not in ("max_epochs", "time_limit"):
            break  # interrupted, or primal_stop's heuristic stop

        candidate = _project_rays(
            lp,
            new.primal - state.primal,
            new.dual_eq - state.dual_eq,
            new.dual_ineq - state.dual_ineq,
        )
        rays, cert = check(*candidate)
        if verbose:
            print(
                f"Infeasibility check after {epochs} epochs: primal-infeasibility "
                f"ratio {cert['primal_infeasibility_ratio']:.2e}, "
                f"dual-infeasibility ratio {cert['dual_infeasibility_ratio']:.2e}"
            )
        if refine and not passes(cert):
            refined = _refine_rays(
                A_host, lp.n_eq, lower_host, upper_host,
                *(np.asarray(v) for v in candidate),
            )
            refined_rays, refined_cert = check(
                *_project_rays(lp, *(jnp.asarray(v, lp.c.dtype) for v in refined))
            )
            if verbose:
                print(
                    "  refined: primal-infeasibility ratio "
                    f"{refined_cert['primal_infeasibility_ratio']:.2e}, "
                    "dual-infeasibility ratio "
                    f"{refined_cert['dual_infeasibility_ratio']:.2e}"
                )
            if passes(refined_cert):
                rays, cert = refined_rays, refined_cert
        if cert["primal_infeasibility_ratio"] <= tol:
            status = "primal_infeasible"
            break
        if cert["dual_infeasibility_ratio"] <= tol:
            status = "dual_infeasible"
            break
        if result["stop_reason"] == "time_limit" or (
            max_epochs is not None and epochs >= max_epochs
        ):
            break

        state, opt_state = new, result["opt_state"]
        chunk = epochs  # double the total length

    return {
        "status": status,
        "primal_ray": rays[0],
        "dual_ray_eq": rays[1],
        "dual_ray_ineq": rays[2],
        "certificate": cert,
        "epochs": epochs,
        "seconds": time.time() - entry_time,
        "result": result,
    }
