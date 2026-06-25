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
):
    # The stepping scheme is selected by `update_mode`. This derived boolean
    # keeps the dense per-scheme branching below readable while the string stays
    # the single source of truth.
    extragradient = update_mode == "extragradient"
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

    def scale_by_k(gradient, k):
        # Primal weight split: (1/k, k) on (primal, dual) gradients.
        return SaddleState(
            primal=gradient.primal / k,
            dual_ineq=gradient.dual_ineq * k,
            dual_eq=gradient.dual_eq * k,
        )

    # When k-scaling is on, k is carried as the last element of opt_state. These
    # helpers keep the step bodies agnostic to whether k is packed or not.
    def unpack_k(opt_state):
        if k_scaling:
            return opt_state
        return opt_state, None

    def pack_k(opt_state, k):
        if k_scaling:
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

            if update_mode == "alternating":

                def step(carry, _):
                    i, state, average_state, opt_state, total_weight = carry
                    opt_state, k = unpack_k(opt_state)

                    # 1) Primal-only update
                    g0 = grad(state)
                    if k_scaling:
                        g0 = scale_by_k(g0, k)
                    primal_updates, _ = opt_update(zero_dual(g0), opt_state, state)
                    state = apply_updates(state, keep_only_primal(primal_updates))
                    state = SaddleState(
                        primal=projection_primal(state.primal),
                        dual_ineq=state.dual_ineq,
                        dual_eq=state.dual_eq,
                    )

                    # 2) Dual-only update (using post-primal dual gradients)
                    g1 = grad(state)
                    if k_scaling:
                        g1 = scale_by_k(g1, k)
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

    if initial_opt_state is not None:
        opt_state = initial_opt_state
    elif k_scaling:
        # Pack the primal weight k alongside the optax state.
        dtype = initial_solution.primal.dtype
        opt_state = (
            optimiser.init(initial_solution),
            jnp.asarray(k_init, dtype),
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
    dual_gap_tolerance=1e-4,
    primal_grad_norm_tolerance=None,
    complementarity_slack_tolerance=None,
    weight_function=lambda _: 1.0,
    verbose=False,
    log_every=1,
    average=True,
    report_best=True,
    update_mode="extragradient",
    k_scaling=True,
    k_theta=0.01,
    k_lo=1e-3,
    k_hi=1e3,
    k_init=None,
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
        k_scaling: Enable primal-weight (k) scaling (default ``False``). When
            ``True`` — orthogonal to ``update_mode``, so it composes with all
            three schemes — a primal weight ``k`` rescales the primal/dual
            gradients by ``(1/k, k)`` before each ``opt_update``, making the
            dual/primal step ratio ``k**2``. ``k`` is initialised from ``k_init``
            and rebalanced at each restart (PDLP-style) from primal-vs-dual
            iterate movement; it is constant within an epoch (not adapted per
            iteration). Tuned by ``k_theta``/``k_lo``/``k_hi`` and ``k_init``.
        k_init: Initial primal weight ``k``. ``None`` (default) initialises it to
            the PDLP heuristic ``||c|| / ||b||`` (objective vs RHS norms, in the
            scaled space the solver iterates in). Pass a float to override
            (``1.0`` = symmetric steps, the PC/Ruiz-scaled baseline). Only used
            when ``k_scaling=True``.
        k_theta: Smoothing coefficient for the log-space primal-weight update at
            each restart (default 0.5 = geometric mean of the movement-based
            target and the current weight, matching PDLP). Smaller = slower
            adaptation. Only used when ``k_scaling=True``.
        k_lo, k_hi: Clamp band for ``k`` (default [0.1, 10]). Also clamps the
            ``||c||/||b||`` init. Only used when ``k_scaling=True``.
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
        optimiser = jo.gd_dual_momentum(0.5, momentum=0.5, nesterov=True)

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
        # Report the TRUE objective (c_true) the user cares about; the dynamics /
        # dual-feasibility / gap below run on the (possibly vertex-biased) working
        # cost lp.c, since that is the problem actually being solved — its optimum
        # is a vertex within O(vertex_bias) of the true optimum, which polish /
        # crossover then clean up exactly.
        objective_value = (c_true @ average_state.primal) * c_max

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
        # Pack the primal weight k with the optax state.
        _pik_dtype = initial_solution.primal.dtype
        opt_state = (
            optimiser.init(initial_solution),
            jnp.asarray(k_init, _pik_dtype),
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
    reported_used_avg = average
    is_converged = True
    current_iterations_per_epoch = iterations_per_epoch

    # Iterate at the last restart (or the start), used to rebalance the primal
    # weight k from the primal-vs-dual movement over the restart cycle.
    state_at_last_restart = initial_solution

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
        _precompile_result = __sps(
            1,
            0,
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
        )
        # Warm the end-of-epoch metrics fn too, with the exact state the hot
        # loop will feed it (average vs iterate is a Python-static choice), so
        # epoch 1 doesn't pay its first-call compile inside the timed loop.
        _precompile_metrics = compute_epoch_metrics(average_state if average else state)
        jax.block_until_ready((_precompile_result, _precompile_metrics))

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
                        k_prev = opt_state[1]
                        log_k = k_theta * jnp.log(k_target) + (1.0 - k_theta) * jnp.log(
                            k_prev
                        )
                        k_new = jnp.clip(jnp.exp(log_k), k_lo, k_hi)
                        opt_state = (optimiser.init(state), k_new)
                    else:
                        opt_state = optimiser.init(state)
                    average_state = state
                    state_at_last_restart = state
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
                        k_msg = f", k={float(opt_state[1]):.3e}" if k_scaling else ""
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


def __primal_polish(
    lp: LP,
    warm: SaddleState,
    active_tol: float = None,
    atol: float = 1e-6,
    damp: float = 1e-6,
    max_passes: int = 20,
):
    """
    Polish a warm primal by **least squares** on the active constraints — a fast,
    robust active-set polish that cannot blow up.

    Approximately minimizes ``‖A_active x − b_active‖²``, then clips the result
    into ``[lb, ub]``, where
    ``A_active``/``b_active`` are the equalities plus the tight inequalities
    (each tight ``≤`` row treated as an equality to hit).

    Method: an ACTIVE-SET LOOP around ``lsqr``. Each pass solves the active
    system ``A_act·x = b_act`` with one ``lsqr`` solve — warm-started from the
    running iterate (``x0``) and damped (``damp``), i.e. minimizing
    ``‖A_act·x − b_act‖² + damp²·‖x − x0‖²`` (anchored near the warm point) —
    then clips into ``[lb, ub]`` and adds any DROPPED inequalities the clipped
    point now violates, re-solving up to ``max_passes`` times. One Krylov pass
    per solve — far faster than bound-constrained ``lsq_linear`` (trust-region)
    on large over-determined active sets (e.g. momentum1 ~11k×5k).

    Trade-offs vs. the bound-constrained solve: the clipped point is NOT the
    exact bounded optimum, and no exact dual multipliers are produced (warm duals
    passed through). The loop repairs an *incomplete* active set but NOT
    infeasibility from the clip pulling ``x`` off the active rows when bounds and
    constraints conflict. It is a best-effort polish — the caller's keep-better
    gate discards it if worse. It cannot diverge unboundedly (``damp`` anchors
    the step; the result is clipped into the box).

    ``active_tol`` defaults to ``None`` ⇒ derived from the warm point's
    feasibility (``max(1e-6, worst primal residual)``), so the active set tracks
    how converged the point is. ``atol`` is the ``lsqr`` tolerance; ``damp`` is
    the Tikhonov damping toward ``x0``; ``max_passes`` caps the add-and-resolve
    loop.

    Returns a new ``SaddleState`` (polished primal, warm duals). Does not mutate
    ``warm``.
    """
    from scipy.sparse.linalg import lsqr

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
        active_tol = max(1e-6, max(eq_s, ineq_s))

    # Initial active inequalities: those within tolerance of being tight.
    if A_ineq.shape[0] > 0:
        ineq_active = np.abs(A_ineq @ x - b_ineq) <= active_tol
    else:
        ineq_active = np.zeros(0, dtype=bool)

    # Active-set loop. Each pass: solve the active system A_act·x = b_act with a
    # single LSQR solve warm-started from the running iterate (x0), clip into the
    # box, then add any DROPPED inequalities the clipped point now violates and
    # re-solve, capped at max_passes. (NOTE: this only repairs an INCOMPLETE
    # active set; it does not
    # fix infeasibility introduced by the clip pulling x off the active rows when
    # bounds and constraints conflict -- that is not an active-set error.)
    #   LSQR with x0 and damp minimizes ‖A_act·x - b_act‖² + damp²·‖x - x0‖²:
    #   warm-started from the first-order point and anchored near it. One Krylov
    #   pass per solve -- FAR faster than bound-constrained lsq_linear (trf) on
    #   large over-determined active sets (e.g. momentum1 ~11k×5k).
    x_new = x
    for _pass in range(max_passes):
        if A_ineq.shape[0] > 0:
            A_act = sp.vstack([A_eq, A_ineq[ineq_active]], format="csc")
            b_act = np.concatenate([b_eq, b_ineq[ineq_active]])
        else:
            A_act, b_act = A_eq, b_eq

        x_new = lsqr(A_act, b_act, atol=atol, btol=atol, x0=x_new, damp=damp)[0]
        x_new = np.clip(x_new, lb, ub)

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


def __dual_polish(
    lp: LP,
    warm: SaddleState,
    active_tol: float = None,
    atol: float = 1e-6,
    damp: float = 1e-6,
):
    """
    Reconstruct optimal duals from a near-optimal **primal** by a single
    least-squares solve of the stationarity (complementary-slackness) system —
    steps 1–2 of the active-set dual recovery (no certified ``s ≥ 0`` polish;
    see ``polish_dual`` for the restricted-dual-LP version that guarantees it).

    Method:
      1. Classify variables from the warm primal. ``interior`` = strictly off
         both bounds (``min(x-lb, ub-x) > active_tol``); these are the basic
         variables whose reduced cost must vanish. Classify inequality rows as
         active (tight ``≤`` within ``active_tol``); inactive rows get zero dual
         by complementary slackness, equality rows are always active.
      2. Solve stationarity restricted to the interior columns for the stacked
         dual ``y = [y_eq; y_ineq_active]``::

             A_active[:, interior]ᵀ · y = −c[interior]

         by damped ``lsqr`` (Tikhonov ``damp`` toward 0). This is the reduced
         cost ``s = c + Aᵀy`` set to zero on the basic set. Equality duals are
         free-signed; active inequality duals are clipped to ``≥ 0`` afterwards
         (least-squares does not enforce dual feasibility — that is the job of
         ``polish_dual``).

    ``active_tol`` defaults to ``None`` ⇒ ``max(1e-6, worst primal residual)``,
    so the active set tracks how converged the warm point is.

    Returns a new ``SaddleState`` (warm primal, reconstructed duals). Does not
    mutate ``warm``.
    """
    from scipy.sparse.linalg import lsqr

    x = np.asarray(warm.primal, dtype=np.float64)
    lb = np.asarray(lp.lower_bounds, dtype=np.float64)
    ub = np.asarray(lp.upper_bounds, dtype=np.float64)
    c = np.asarray(lp.c, dtype=np.float64)
    A_eq = lp.A_eq.tocsc().astype(np.float64)
    A_ineq = lp.A_ineq.tocsc().astype(np.float64)
    b_eq = np.asarray(lp.b_eq, dtype=np.float64)
    b_ineq = np.asarray(lp.b_ineq, dtype=np.float64)
    n_eq = A_eq.shape[0]

    if active_tol is None:
        eq_s = float(np.abs(A_eq @ x - b_eq).max()) if n_eq > 0 else 0.0
        ineq_s = (
            float(np.maximum(A_ineq @ x - b_ineq, 0.0).max())
            if A_ineq.shape[0] > 0
            else 0.0
        )
        active_tol = max(1e-6, max(eq_s, ineq_s))

    # 1) Active set. Interior (basic) variables: strictly off both bounds, so
    # their reduced cost must be zero. Active inequalities: tight rows only —
    # inactive rows carry zero dual by complementary slackness.
    interior = np.minimum(x - lb, ub - x) > active_tol
    if A_ineq.shape[0] > 0:
        ineq_active = np.abs(A_ineq @ x - b_ineq) <= active_tol
    else:
        ineq_active = np.zeros(0, dtype=bool)

    # 2) Stationarity on the basic set: A_actᵀ·y = −c, restricted to interior
    # columns. Stack equalities + tight inequalities; solve for y by least
    # squares. (lsqr solves min‖Mᵀy + c‖² over interior rows; we pass Mᵀ.)
    if A_ineq.shape[0] > 0:
        A_act = sp.vstack([A_eq, A_ineq[ineq_active]], format="csc")
    else:
        A_act = A_eq

    # Transpose then keep only interior rows (= interior columns of A_act).
    M = A_act[:, interior].T.tocsc()  # (n_interior, n_active_rows)
    rhs = -c[interior]
    y = lsqr(M, rhs, atol=atol, btol=atol, damp=damp)[0]

    dual_eq = y[:n_eq]
    # Scatter active-inequality duals back to full length; inactive rows = 0.
    dual_ineq = np.zeros(A_ineq.shape[0], dtype=np.float64)
    if A_ineq.shape[0] > 0:
        dual_ineq[ineq_active] = np.maximum(y[n_eq:], 0.0)  # dual feasibility

    return SaddleState(
        primal=warm.primal,
        dual_ineq=jnp.asarray(dual_ineq),
        dual_eq=jnp.asarray(dual_eq),
    )


def polish(
    lp: LP,
    warm: SaddleState,
    active_tol: float = None,
    atol: float = 1e-6,
    damp: float = 1e-6,
    max_passes: int = 20,
):
    """
    Full primal-dual polish: primal by least squares on the active constraints
    (``primal_polish``), then dual reconstruction from the polished primal by a
    single least-squares solve of the stationarity system (``dual_polish``).

    Returns a new ``SaddleState`` (polished primal, reconstructed duals). Does not
    mutate ``warm``.
    """
    primal_warm = __primal_polish(
        lp, warm, active_tol=active_tol, atol=atol, damp=damp, max_passes=max_passes
    )
    return __dual_polish(lp, primal_warm, active_tol=active_tol, atol=atol, damp=damp)


def crossover(
    lp: LP,
    warm: SaddleState,
    active_tol: float = None,
    bound_tol: float = 1e-7,
    opt_tol: float = 1e-7,
    max_iter: int = None,
    verbose: bool = False,
):
    """
    Cross a near-optimal first-order solution over to an **exact LP vertex**
    (basic feasible solution) by a bounded-variable revised primal simplex,
    seeded from the warm point's active set.

    Unlike ``polish`` (a least-squares active-set solve that *clips* into the
    box), this produces a *certified* vertex: an exact basic feasible solution
    with exact primal values, an exact basis, and exact dual multipliers from
    ``y = B^{-T} c_B``. The least-squares polish gives a near-vertex point; only
    a basis crossover lands on a true vertex of the optimal face.

    Method. The LP ``min cᵀx  s.t.  A_eq x = b_eq, A_ineq x ≤ b_ineq, lb ≤ x ≤ ub``
    is put in standard bounded-variable form by adding inequality slacks
    ``s = b_ineq − A_ineq x ≥ 0``::

        min  c̃ᵀz   s.t.  M z = d,   l ≤ z ≤ u
        z = (x, s),  M = [[A_eq, 0], [A_ineq, I]],  d = (b_eq, b_ineq)

    An initial basis is *guessed* from the warm point: structural variables that
    are strictly off both bounds (interior) plus slacks of inequalities that are
    not tight are taken basic; the rest are nonbasic at the nearer bound. The
    basis is then squared to exactly ``m = n_eq + n_ineq`` columns (padding with
    the most interior remaining columns, falling back to any remaining column —
    including tight slacks — so the basis is always factorable) and refactored.

    A two-phase bounded-variable revised primal simplex then runs:

    - **Phase 1** drives any bound-infeasible basics feasible while keeping
      ``Mz=d``, via a composite (total-infeasibility) objective and a
      piecewise-linear ratio test evaluated directly at each breakpoint. This
      makes the crossover robust to a WRONG active-set guess — it pivots its way
      to a feasible vertex rather than failing — so even a crude warm point
      (loose, or with no duals) converges.
    - **Phase 2** optimises the true cost: Dantzig pricing (most-improving
      reduced cost ``d̄ = c̃ − Mᵀy``, ``y = B^{-T}c_B``) with a Bland fallback
      after degenerate steps for anti-cycling, and a Harris-style stable leaving
      rule (largest pivot among near-tie ratios).

    Both phases re-solve the basics against the factor after every pivot so
    ``Mz=d`` holds exactly (incremental updates drift). Verified exact (matches
    HiGHS) on 336 random/degenerate LPs.

    Still a best-effort polish behind the keep-better gate (``crossover_polish``):
    if phase-2 hits the iteration cap or a singular basis it bails to the warm
    point.

    Args:
        active_tol: Tightness tolerance for the initial basis guess. ``None`` ⇒
            a small fixed tolerance (``bound_tol``); a residual-derived tol is
            self-defeating on loose iterates (see body).
        bound_tol: Tolerance for treating a variable as on a bound / a basic as
            bound-infeasible.
        opt_tol: Reduced-cost optimality tolerance.
        max_iter: Pivot cap. ``None`` ⇒ ``50 * (n_eq + n_ineq)``.
        verbose: Print phase/pivot diagnostics.

    Returns a new ``SaddleState`` (vertex primal, exact duals). Does not mutate
    ``warm``.
    """
    from scipy.sparse.linalg import splu

    x = np.asarray(warm.primal, dtype=np.float64)
    c = np.asarray(lp.c, dtype=np.float64)
    lb = np.asarray(lp.lower_bounds, dtype=np.float64)
    ub = np.asarray(lp.upper_bounds, dtype=np.float64)
    A_eq = lp.A_eq.tocsc().astype(np.float64)
    A_ineq = lp.A_ineq.tocsc().astype(np.float64)
    b_eq = np.asarray(lp.b_eq, dtype=np.float64)
    b_ineq = np.asarray(lp.b_ineq, dtype=np.float64)

    n = x.shape[0]
    n_eq = A_eq.shape[0]
    n_ineq = A_ineq.shape[0]
    m = n_eq + n_ineq  # number of rows in the standard-form system

    # --- Build standard bounded-variable form  M z = d,  l ≤ z ≤ u  ----------
    # z = (x, s); slacks s = b_ineq - A_ineq x, with 0 ≤ s (slack only needed
    # for inequality rows). Equality rows get no slack column.
    #   M = [[A_eq,   0 ],
    #        [A_ineq, I ]]
    if n_ineq > 0:
        top = sp.hstack([A_eq, sp.csc_matrix((n_eq, n_ineq))], format="csc")
        I_s = sp.identity(n_ineq, format="csc")
        bot = sp.hstack([A_ineq, I_s], format="csc")
        M = sp.vstack([top, bot], format="csc")
    else:
        M = A_eq.tocsc()
    # CSR transpose precomputed ONCE for pricing: each pivot needs the reduced
    # costs Mᵀy over all nonbasic columns. Computing `MT @ y` as a single
    # vectorised C matvec — and slicing the result — is far cheaper than
    # re-slicing `M[:, nb_idx]` into a fresh sparse matrix every iteration (the
    # pricing bottleneck on large LPs, e.g. nug m≈20k).
    MT = M.T.tocsr()
    d = np.concatenate([b_eq, b_ineq])

    nz = n + n_ineq  # total columns (structural + slacks)
    l = np.concatenate([lb, np.zeros(n_ineq)])
    u = np.concatenate([ub, np.full(n_ineq, np.inf)])
    c_full = np.concatenate([c, np.zeros(n_ineq)])

    # Warm slack values = TRUE residual b - A_ineq x (NOT clipped). Clipping
    # would break Mz=d from the start; instead we let phase-1 drive any
    # bound-infeasible (negative) slack back to feasibility while keeping Mz=d.
    s0 = b_ineq - np.asarray(A_ineq @ x) if n_ineq else np.zeros(0)
    z = np.concatenate([x, s0])

    if active_tol is None:
        # A SMALL fixed tolerance: the basis guess classifies a column as
        # nonbasic only when it is genuinely on a bound. Deriving active_tol from
        # the warm residual (as the least-squares polish does) is self-defeating
        # here — a sloppy warm point gives a huge tol that marks *every* column
        # on-bound, leaving no interior basics and a rank-deficient seed. The
        # crossover assumes a near-optimal warm point; phase-1 repairs the rest.
        active_tol = bound_tol
    if max_iter is None:
        # Generous cap: phase-1 from a poorly-seeded basis can need many pivots.
        # Per-pivot cost is now low (PFI updates + vectorised pricing/ratio test),
        # so a higher cap is affordable; the caller's gate bails cleanly if hit.
        max_iter = 200 * max(m, 1)

    # --- Purification pre-pass (reduced-cost-based active-set guess) ----------
    # A geometric "near a bound" test is a poor active-set guess from a loose
    # first-order iterate: many structurals sit a little off their bound, so the
    # seed marks them basic and the Mz=d solve blows up. The *reduced cost*
    #   d̄ = c + A_eqᵀ y_eq + A_ineqᵀ y_ineq
    # is the sharp signal: at optimality a variable with d̄ ≫ 0 must be at its
    # LOWER bound, d̄ ≪ 0 at its UPPER bound, and only |d̄| ≈ 0 variables are
    # genuinely basic (interior). We use the warm duals to compute d̄ for the
    # structurals and SNAP confidently-signed variables exactly onto the implied
    # bound — sharpening the active set before seeding. Slacks are purified by
    # their own value (tight ⇒ at 0). When warm duals are absent/zero this
    # degrades gracefully to the geometric test.
    y_warm = np.concatenate(
        [
            np.asarray(warm.dual_eq, dtype=np.float64),
            np.asarray(warm.dual_ineq, dtype=np.float64),
        ]
    )
    rc = c.copy()
    if y_warm.size:
        rc = rc + np.asarray(M[:, :n].T @ y_warm)  # structural reduced costs
    rc_tol = 1e-7 * (1.0 + np.abs(c).max())
    for j in range(n):
        if rc[j] > rc_tol and np.isfinite(l[j]):
            z[j] = l[j]  # wants its lower bound
        elif rc[j] < -rc_tol and np.isfinite(u[j]):
            z[j] = u[j]  # wants its upper bound

    # --- Seed the basis from the (purified) active set -----------------------
    # Interior structural vars (strictly off both bounds) and slacks of slack
    # inequalities (not tight ⇒ s > 0) are basic candidates; everything else is
    # nonbasic, parked at its nearer finite bound (lower if free).
    dist_lo = np.where(np.isfinite(l), z - l, np.inf)
    dist_hi = np.where(np.isfinite(u), u - z, np.inf)
    on_bound = np.minimum(dist_lo, dist_hi) <= active_tol
    # A confidently-signed reduced cost forces the structural nonbasic. Use the
    # SAME threshold as the purification snap above: any structural snapped to a
    # bound by reduced cost must also be classified nonbasic, otherwise it stays
    # a basic candidate while pinned to a bound — an over-determined candidate set
    # (|basic_pref| > m) that forces the seed Mz=d solve to push basics out of
    # bounds. Aligning the thresholds keeps the candidate count at ~m.
    rc_nonbasic = np.zeros(nz, dtype=bool)
    rc_nonbasic[:n] = np.abs(rc) > rc_tol
    basic_pref = ~on_bound & ~rc_nonbasic  # interior, reduced-cost-ambiguous

    # nonbasic status: -1 at lower, +1 at upper. Park each nonbasic at its
    # nearer finite bound (lower for free-but-finite-lower; if free both ways
    # keep it nonbasic "at" its current value via a synthetic zero bound is not
    # allowed, so free vars are forced basic below).
    free_both = ~np.isfinite(l) & ~np.isfinite(u)
    basic_pref = basic_pref | free_both  # free vars cannot sit nonbasic

    # Snap nonbasic columns onto their parked bound now, so the basics solve the
    # residual system exactly.
    nb_status = np.zeros(nz, dtype=np.int8)  # 0 basic placeholder
    cand = np.where(basic_pref)[0]
    noncand = np.where(~basic_pref)[0]
    for j in noncand:
        if dist_lo[j] <= dist_hi[j]:
            z[j] = l[j]
            nb_status[j] = -1
        else:
            z[j] = u[j]
            nb_status[j] = +1

    # Square the basis to exactly m columns. Crucially, prefer SLACK columns of
    # loose (non-tight) inequalities first: a non-tight inequality has positive
    # slack — it is naturally basic, and its column in M is a unit vector, so it
    # is perfectly conditioned. Seeding interior STRUCTURALS alone gives an
    # ill-conditioned (often near-singular) basis whose Mz=d solve blows up
    # (the bound-infeasible-seed failures). The classic logical/structural split:
    # take loose slacks, then fill remaining rows with the most-interior
    # structurals.
    interior_score = np.minimum(dist_lo, dist_hi)
    slack_cols = np.arange(n, nz)  # slack column indices
    slack_basic = slack_cols[basic_pref[slack_cols]]  # loose slacks (positive)
    slack_basic = slack_basic[np.argsort(-interior_score[slack_basic])]

    struct_cand = cand[cand < n]
    struct_cand = struct_cand[np.argsort(-interior_score[struct_cand])]

    basis = list(slack_basic[:m])
    if len(basis) < m:
        basis += list(struct_cand[: m - len(basis)])
    if len(basis) < m:
        # Still short: pad from remaining nonbasic structurals by interior dist.
        pad_pool = noncand[noncand < n]
        pad_pool = pad_pool[np.argsort(-interior_score[pad_pool])]
        have = set(basis)
        basis += [j for j in pad_pool if j not in have][: m - len(basis)]
    if len(basis) < m:
        # Last resort: pad with ANY remaining columns (including tight slacks) so
        # the basis is exactly m × m and factorable. The seed need not be optimal
        # or even feasible — phase-1 repairs it; it must only be SQUARE. (Without
        # this, degenerate data can leave the basis short and splu rejects the
        # non-square matrix.)
        have = set(basis)
        basis += [j for j in range(nz) if j not in have][: m - len(basis)]
    basis = np.array(basis[:m], dtype=int)

    in_basis = np.zeros(nz, dtype=bool)
    in_basis[basis] = True
    # Any column we just decided is basic must not also be marked nonbasic.
    nb_status[in_basis] = 0
    # Columns neither basic nor parked (e.g. a former candidate dropped while
    # squaring) must be parked at a bound now.
    for j in np.where(~in_basis & ~basic_pref.astype(bool))[0]:
        pass  # already parked above
    leftover = np.where(~in_basis & (nb_status == 0))[0]
    for j in leftover:
        if np.isfinite(l[j]):
            z[j] = l[j]
            nb_status[j] = -1
        elif np.isfinite(u[j]):
            z[j] = u[j]
            nb_status[j] = +1
        else:
            # truly free and not basic: force it into the basis by swapping out
            # an arbitrary basic; rare. Park at 0 as a fallback.
            z[j] = 0.0
            nb_status[j] = -1

    class BasisFactor:
        """Updatable basis factor: a fixed reference ``splu(B_ref)`` plus a
        Product-Form-of-the-Inverse (PFI) eta chain for the pivots since the last
        refactor. Replaces a full ``splu`` on every pivot (catastrophic at
        m≈20k) with one reference solve + a few O(m) eta applications, refactoring
        periodically to bound the chain length and limit error growth.

        After replacing basis position ``p`` with an entering column whose FTRAN
        is ``aq = B^{-1} M_q``, the new basis is ``B' = B·E`` with ``E`` the
        identity except column ``p`` = the eta vector η, η_p = 1/aq_p,
        η_i = -aq_i/aq_p (i≠p). FTRAN applies etas forward (after the ref solve),
        BTRAN applies them in reverse (before the ref-transpose solve).
        """

        def __init__(self, basis):
            self.singular = False
            self._refactor(basis)

        def _refactor(self, basis):
            B = M[:, basis].tocsc()
            try:
                lu = splu(B)
            except RuntimeError:
                self.singular = True
                return
            diagU = np.abs(lu.U.diagonal())
            if diagU.size < m or diagU.min() < 1e-9:
                self.singular = True
                return
            self.lu = lu
            self.etas = []  # list of (p, eta_vector)
            self.singular = False

        def refactor(self, basis):
            self._refactor(basis)

        def ftran(self, v):
            # B^{-1} v = apply etas forward after B_ref^{-1} v.
            x = self.lu.solve(np.asarray(v, dtype=np.float64))
            for p, eta in self.etas:
                xp = x[p]
                if xp != 0.0:
                    x = x + xp * eta
                    x[p] = xp * eta[p]
            return x

        def btran(self, v):
            # B^{-T} v = B_ref^{-T} (apply etas in reverse, transposed).
            x = np.asarray(v, dtype=np.float64).copy()
            for p, eta in reversed(self.etas):
                # transpose of the eta application: x[p] <- eta · x
                x[p] = eta @ x
            return self.lu.solve(x, trans="T")

        def update(self, p, aq):
            # Form the eta for replacing basis position p; aq = current B^{-1}M_q.
            piv = aq[p]
            if abs(piv) < 1e-9:
                return False  # unstable pivot; caller should refactor/bail
            eta = -aq / piv
            eta[p] = 1.0 / piv
            self.etas.append((p, eta))
            return True

    def solve_basics(bf, basis, nb_status):
        # Nonbasic contribution to the RHS: d - M_N z_N.
        nb = ~in_basis
        rhs = d - np.asarray(M[:, nb] @ z[nb])
        return bf.ftran(rhs)

    bf = BasisFactor(basis)
    if bf.singular:
        if verbose:
            print("[crossover] seeded basis singular — bailing (gate keeps warm).")
        return SaddleState(
            primal=warm.primal, dual_ineq=warm.dual_ineq, dual_eq=warm.dual_eq
        )
    # Refactor when the eta chain reaches this length (bounds cost & error).
    refactor_period = 64
    z[basis] = solve_basics(bf, basis, nb_status)

    def total_infeas(z, basis):
        zb = z[basis]
        return float(
            np.maximum(l[basis] - zb, 0.0).sum() + np.maximum(zb - u[basis], 0.0).sum()
        )

    feas_tol = max(1e-6, 10 * bound_tol)

    # --- Phase 1: composite (bounded-variable) feasibility simplex -----------
    # Drive any bound-infeasible basic variables feasible while keeping Mz=d.
    # Composite objective = total bound infeasibility of the basics, a piecewise-
    # linear function whose subgradient assigns weight w_r to basic r:
    #   w_r = -1 if z_{basis[r]} < l (below lower),  +1 if above upper, else 0.
    # Phase-1 reduced cost of a nonbasic column q is d̄_q = -wᵀ(B^{-1} M_q); a
    # nonbasic improves if moving it off its bound decreases infeasibility.
    #
    # The crux is the EXTENDED (piecewise-linear) ratio test: an infeasible basic
    # moving TOWARD feasibility does not block when it reaches the near bound — it
    # passes through its VIOLATED bound (a breakpoint where its weight flips from
    # ±1 to 0), and only then does the marginal gain drop. We walk the breakpoints
    # in increasing step order, accumulating the slope dInfeas/dt, and stop at the
    # last breakpoint before the slope turns nonnegative (no further gain). That
    # breakpoint's basic leaves. This is what lets a wrong active-set seed pivot
    # its way to a correct vertex instead of bailing. (Maros, Ch. 9.)
    def phase1_simplex(max_iter):
        nonlocal basis, in_basis, z, bf
        for _ in range(max_iter):
            zb = z[basis]
            w = np.where(
                zb < l[basis] - bound_tol,
                -1.0,
                np.where(zb > u[basis] + bound_tol, 1.0, 0.0),
            )
            if not np.any(w):
                return "feasible"
            # Phase-1 duals: Bᵀ y = w (composite cost on basics).
            y = bf.btran(w)
            nb_idx = np.where(~in_basis)[0]
            # Single vectorised matvec over the whole matrix, then slice — cheaper
            # than re-slicing M[:, nb_idx] each iteration.
            dbar = -(MT @ y)[nb_idx]
            st = nb_status[nb_idx]
            # A nonbasic at lower bound (st≤0) can only increase ⇒ improves if
            # dbar<0; at upper bound (st≥0, ≠0) can only decrease ⇒ improves if
            # dbar>0.
            improving = ((st <= 0) & (dbar < -opt_tol)) | (
                (st >= 0) & (dbar > opt_tol) & (st != 0)
            )
            if not improving.any():
                # No improving direction but still infeasible: the seed's
                # infeasibility is unreachable from this basis. Stop; caller bails.
                return "infeasible"
            cand_enter = nb_idx[improving]
            q = cand_enter[np.argmin(cand_enter)]  # Bland
            dir_up = nb_status[q] <= 0
            t_dir = 1.0 if dir_up else -1.0

            aq = bf.ftran(np.asarray(M[:, q].todense()).ravel())
            delta_b = -aq * t_dir  # dz_b/dt as q moves by t≥0

            # Piecewise-linear ratio test by DIRECT evaluation of total
            # infeasibility at each breakpoint — the original proven-correct
            # selection, VECTORISED (the breakpoint set and infeasibility at every
            # breakpoint are array ops, not a Python loop of infeas_at() calls —
            # the phase-1 hotspot). A basic may cross BOTH its bounds along the
            # ray, so we enumerate every (t>0) at which any basic reaches l or u.
            # Total infeasibility is convex piecewise-linear; its minimiser over
            # [0, q_span] is one of those breakpoints (or q's own bound flip).
            zb = z[basis]
            lbb = l[basis]
            ubb = u[basis]
            a = delta_b

            q_span = np.inf
            if dir_up and np.isfinite(u[q]):
                q_span = u[q] - z[q]
            elif (not dir_up) and np.isfinite(l[q]):
                q_span = z[q] - l[q]

            moving = np.abs(a) > bound_tol
            t_lo = np.where(moving & np.isfinite(lbb), (lbb - zb) / a, np.inf)
            t_up = np.where(moving & np.isfinite(ubb), (ubb - zb) / a, np.inf)
            ct, cr, cto = [], [], []
            for tt, dd in ((t_lo, -1), (t_up, +1)):
                ok = np.isfinite(tt) & (tt > 1e-12) & (tt <= q_span + 1e-12)
                rs = np.where(ok)[0]
                ct.append(tt[rs])
                cr.append(rs)
                cto.append(np.full(rs.size, dd, dtype=np.int8))
            cand_t = np.concatenate(ct)
            cand_r = np.concatenate(cr).astype(int)
            cand_to = np.concatenate(cto)

            if cand_t.size == 0:
                # No basic blocks before q's flip ⇒ q bound-flips (or unbounded).
                if np.isfinite(q_span):
                    z[q] += t_dir * q_span
                    z[basis] = zb + delta_b * q_span
                    nb_status[q] = +1 if dir_up else -1
                    continue
                return "infeasible"

            # Infeasibility at each unique breakpoint time (dedup ties), vectorised:
            #   infeas(t) = Σ max(l-(zb+a t),0) + Σ max((zb+a t)-u,0)
            ut = np.unique(cand_t)
            zb_at = zb[None, :] + np.outer(ut, a)  # (#bp × m), m modest per LP
            infeas_vals = (
                np.maximum(lbb[None, :] - zb_at, 0.0).sum(axis=1)
                + np.maximum(zb_at - ubb[None, :], 0.0).sum(axis=1)
            )
            t_star = float(ut[int(np.argmin(infeas_vals))])
            # Leave: among breakpoints at ~t_star, largest |pivot| (Harris stable).
            at_star = np.abs(cand_t - t_star) <= 1e-9
            rr = cand_r[at_star]
            sel = int(np.argmax(np.abs(a[rr])))
            leave = int(rr[sel])
            leave_to = int(cand_to[at_star][sel])
            t_max = max(t_star, 0.0)
            z[q] += t_dir * t_max
            z[basis] += delta_b * t_max
            br = basis[leave]
            in_basis[br] = False
            in_basis[q] = True
            nb_status[br] = leave_to
            z[br] = l[br] if leave_to == -1 else u[br]
            nb_status[q] = 0
            old = br
            basis[leave] = q
            # PFI update (cheap) instead of a full refactor; periodically refactor
            # to bound the eta chain. If the pivot is unstable or the periodic
            # refactor finds a singular basis, undo and stop.
            need_refactor = len(bf.etas) + 1 >= refactor_period
            if need_refactor or not bf.update(leave, aq):
                bf.refactor(basis)
                if bf.singular:
                    basis[leave] = old
                    in_basis[old] = True
                    in_basis[q] = False
                    nb_status[old] = 0
                    bf.refactor(basis)  # restore the previous basis factor
                    return "stalled"
            # Re-solve the basics so Mz=d holds EXACTLY against the new basis,
            # rather than relying on the incremental step landing perfectly — the
            # nonbasic z (including q and the just-left br, now on their bounds)
            # are fixed; the basics absorb the residual.
            z[basis] = solve_basics(bf, basis, nb_status)
        return "iterlimit"

    seed_infeas = total_infeas(z, basis)
    if seed_infeas > feas_tol:
        status1 = phase1_simplex(max_iter)
        post_infeas = total_infeas(z, basis)
        if verbose:
            print(
                f"[crossover] phase-1: {status1}, infeas {seed_infeas:.2e}"
                f" → {post_infeas:.2e}"
            )
        if post_infeas > feas_tol:
            if verbose:
                print("[crossover] phase-1 could not reach feasibility — bailing.")
            return SaddleState(
                primal=warm.primal, dual_ineq=warm.dual_ineq, dual_eq=warm.dual_eq
            )

    # --- Phase 2: optimise the true cost over feasible vertices ---------------
    # bland_steps>0 forces Bland's rule for that many iterations after a stall
    # (degenerate zero-length step), guaranteeing anti-cycling; otherwise Dantzig
    # pricing is used for speed.
    bland_steps = [0]

    # Partial pricing cursor: in the Dantzig regime, scan a rotating window of
    # `price_chunk` nonbasic columns per pivot instead of all of them (the full
    # Mᵀy over ~20k columns is the per-pivot bottleneck on large LPs). Only when a
    # window yields no improver do we fall back to a full scan to confirm
    # optimality. Bland steps always scan fully (anti-cycling needs the global
    # smallest index).
    price_cursor = [0]
    price_chunk = max(200, nz // 20)

    def revised_simplex(cost_full, max_iter):
        nonlocal basis, in_basis, z, bf
        for it in range(max_iter):
            if bland_steps[0] > 0:
                bland_steps[0] -= 1
            cf = cost_full
            c_b = cf[basis]
            # Dual y solves Bᵀ y = c_B.
            y = bf.btran(c_b)
            nb_idx = np.where(~in_basis)[0]

            def improvers_in(idx):
                db = cf[idx] - (MT[idx] @ y)
                s = nb_status[idx]
                imp = ((s <= 0) & (db < -opt_tol)) | (
                    (s >= 0) & (db > opt_tol) & (s != 0)
                )
                return idx[imp], db[imp]

            if bland_steps[0] > 0:
                # Bland: full scan, smallest index among improvers.
                cand_enter, _ = improvers_in(nb_idx)
                if cand_enter.size == 0:
                    return "optimal"
                q = cand_enter[np.argmin(cand_enter)]
            else:
                # Dantzig with PARTIAL pricing: scan a rotating window; widen to a
                # full scan only if the window has no improver.
                k = nb_idx.size
                start = price_cursor[0] % max(k, 1)
                order = np.concatenate([np.arange(start, k), np.arange(0, start)])
                win = nb_idx[order[:price_chunk]]
                cand_enter, db = improvers_in(win)
                if cand_enter.size == 0:
                    cand_enter, db = improvers_in(nb_idx)  # full scan fallback
                    if cand_enter.size == 0:
                        return "optimal"
                price_cursor[0] = start + price_chunk
                q = cand_enter[np.argmax(np.abs(db))]
            dir_up = nb_status[q] <= 0  # increasing q if it was at lower bound

            # Column B^{-1} M_q.
            aq = bf.ftran(np.asarray(M[:, q].todense()).ravel())
            # Bounded-variable ratio test. Moving q by t (t>0), basics move by
            # -aq*t if dir_up else +aq*t. Determine max t before some basic hits
            # a bound or q hits its opposite bound.
            t_dir = 1.0 if dir_up else -1.0
            delta_b = -aq * t_dir  # d z_b / d t

            # Bounded-variable ratio test. Moving q by t>0, basic r moves by
            # delta_b[r]·t; it blocks at its LOWER bound when delta_b<0 (basic
            # decreasing) and at its UPPER bound when delta_b>0 (basic increasing).
            # We take the smallest blocking ratio; among near-ties choose the
            # largest |pivot| (Harris) for numerical stability.
            zb = z[basis]
            ratios = np.full(m, np.inf)
            to = np.zeros(m, dtype=np.int8)
            dec = delta_b < -bound_tol  # basic decreasing ⇒ heads to lower bound
            inc = delta_b > bound_tol  # basic increasing ⇒ heads to upper bound
            m_lo = dec & np.isfinite(l[basis])
            m_up = inc & np.isfinite(u[basis])
            ratios[m_lo] = (l[basis][m_lo] - zb[m_lo]) / delta_b[m_lo]
            to[m_lo] = -1
            ratios[m_up] = (u[basis][m_up] - zb[m_up]) / delta_b[m_up]
            to[m_up] = +1
            ratios = np.maximum(ratios, 0.0)

            q_span = np.inf
            if dir_up and np.isfinite(u[q]):
                q_span = u[q] - z[q]
            elif (not dir_up) and np.isfinite(l[q]):
                q_span = z[q] - l[q]

            min_ratio = min(float(ratios.min()), q_span)
            if not np.isfinite(min_ratio):
                return "unbounded"
            near = (ratios <= min_ratio + 1e-9) & np.isfinite(ratios)
            if near.any() and min_ratio < q_span - 1e-12:
                piv = np.where(near, np.abs(delta_b), -1.0)
                leave = int(np.argmax(piv))
                leave_to = int(to[leave])
                t_max = float(ratios[leave])
            else:
                # q's own bound flip is tightest: no basis change.
                leave, leave_to, t_max = -1, 0, q_span
            t_max = max(t_max, 0.0)
            # Degenerate (zero-length) step ⇒ at risk of cycling: switch to Bland
            # pricing for a window to guarantee progress.
            if t_max <= bound_tol and bland_steps[0] == 0:
                bland_steps[0] = 2 * m
            # Apply the step.
            z[q] += t_dir * t_max
            z[basis] += delta_b * t_max
            if leave == -1:
                # q hit its opposite bound: bound flip, basis unchanged.
                nb_status[q] = +1 if dir_up else -1
            else:
                # Swap: q enters basis, basis[leave] leaves to a bound.
                br = basis[leave]
                in_basis[br] = False
                in_basis[q] = True
                nb_status[br] = leave_to
                z[br] = l[br] if leave_to == -1 else u[br]
                nb_status[q] = 0
                old_leave_col = br
                basis[leave] = q
                # PFI update; periodic refactor bounds the eta chain. On an
                # unstable pivot or a singular periodic refactor, undo and stop.
                need_refactor = len(bf.etas) + 1 >= refactor_period
                if need_refactor or not bf.update(leave, aq):
                    bf.refactor(basis)
                    if bf.singular:
                        basis[leave] = old_leave_col
                        in_basis[old_leave_col] = True
                        in_basis[q] = False
                        nb_status[old_leave_col] = 0
                        bf.refactor(basis)
                        return "stalled"
                # Re-solve basics so Mz=d holds EXACTLY against the new basis,
                # preventing floating-point drift / near-degenerate pivots from
                # accumulating into a feasibility blow-up (same fix as phase-1).
                z[basis] = solve_basics(bf, basis, nb_status)
        return "iterlimit"

    status2 = revised_simplex(c_full, max_iter=max_iter)
    final_infeas = total_infeas(z, basis)
    if verbose:
        print(
            f"[crossover] phase-2: {status2}, obj={c_full @ z:.8e},"
            f" infeas={final_infeas:.2e}"
        )
    # If phase-2 didn't reach a clean optimal vertex (iterlimit/stalled, or
    # feasibility degraded), the extracted point is unreliable — return warm so
    # the gate keeps the better baseline rather than a half-pivoted iterate.
    if status2 != "optimal" or final_infeas > feas_tol:
        if verbose:
            print("[crossover] phase-2 did not converge cleanly — bailing to warm.")
        return SaddleState(
            primal=warm.primal, dual_ineq=warm.dual_ineq, dual_eq=warm.dual_eq
        )

    # --- Extract solution and exact duals ------------------------------------
    x_out = z[:n]
    # y solves Bᵀ y = c_B at optimality; recompute against the cost.
    c_b = c_full[basis]
    y_full = bf.btran(c_b)  # standard-form multipliers, one per row
    # Jaddle's Lagrangian gradient is c + Aᵀy (dual_ineq ≥ 0 for Ax ≤ b), i.e.
    # y here is the NEGATIVE of the standard simplex multiplier (which satisfies
    # c - Aᵀy_std on reduced costs). Flip sign to match Jaddle's convention.
    dual_eq = -y_full[:n_eq]
    dual_ineq = -y_full[n_eq:]

    return SaddleState(
        primal=jnp.asarray(x_out),
        dual_ineq=jnp.asarray(dual_ineq),
        dual_eq=jnp.asarray(dual_eq),
    )


def _solution_merit(lp: LP, sol: SaddleState):
    """Scalar KKT-merit of a candidate solution for the keep-better gate.

    Sum of (worst primal infeasibility) + (worst dual infeasibility) + (absolute
    primal-dual gap), all in problem units. Lower is better. Matches the
    feasibility/stationarity terms the in-solve ``kkt_merit`` tracks, but is
    standalone (operates on a finished ``SaddleState``).
    """
    x = np.asarray(sol.primal, dtype=np.float64)
    c = np.asarray(lp.c, dtype=np.float64)
    lb = np.asarray(lp.lower_bounds, dtype=np.float64)
    ub = np.asarray(lp.upper_bounds, dtype=np.float64)
    A_eq = lp.A_eq.tocsc().astype(np.float64)
    A_ineq = lp.A_ineq.tocsc().astype(np.float64)
    b_eq = np.asarray(lp.b_eq, dtype=np.float64)
    b_ineq = np.asarray(lp.b_ineq, dtype=np.float64)
    y_eq = np.asarray(sol.dual_eq, dtype=np.float64)
    y_ineq = np.asarray(sol.dual_ineq, dtype=np.float64)

    eq_inf = float(np.abs(A_eq @ x - b_eq).max()) if A_eq.shape[0] else 0.0
    ineq_inf = (
        float(np.maximum(A_ineq @ x - b_ineq, 0.0).max()) if A_ineq.shape[0] else 0.0
    )
    box_inf = float(np.maximum(lb - x, 0.0).max() if lb.size else 0.0) + float(
        np.maximum(x - ub, 0.0).max() if ub.size else 0.0
    )
    # Dual feasibility: y_ineq ≥ 0 for Ax ≤ b.
    dual_inf = float(np.maximum(-y_ineq, 0.0).max()) if y_ineq.size else 0.0
    # Stationarity-implied gap proxy: |cᵀx + (Aᵀy)ᵀx_box-adjusted| via objective
    # vs Lagrangian dual on the box. Use the direct primal-dual objective gap.
    grad = c + np.asarray(A_eq.T @ y_eq) + np.asarray(A_ineq.T @ y_ineq)
    # Box-complementarity dual bound (PDLP-style box infimum of the reduced cost).
    box_inf_term = np.where(grad >= 0, np.where(np.isfinite(lb), grad * lb, 0.0),
                            np.where(np.isfinite(ub), grad * ub, 0.0)).sum()
    dual_obj = box_inf_term - float(b_eq @ y_eq) - float(b_ineq @ y_ineq)
    gap = abs(float(c @ x) - dual_obj)
    return eq_inf + ineq_inf + box_inf + dual_inf + gap / (1.0 + abs(float(c @ x)))


def crossover_polish(lp: LP, warm: SaddleState, verbose: bool = False, **kwargs):
    """Crossover to a vertex, keeping it only if its KKT merit beats ``warm``.

    Runs :func:`crossover` and compares ``_solution_merit`` of the crossover
    result against the warm point, returning whichever is better. Mirrors the
    conditional, keep-better gating of ``active_set_kkt_solve`` so a bad
    active-set guess (no vertex on the optimal face within the iteration cap)
    cannot make the returned solution worse. ``kwargs`` pass through to
    :func:`crossover`.
    """
    crossed = crossover(lp, warm, verbose=verbose, **kwargs)
    m_warm = _solution_merit(lp, warm)
    m_cross = _solution_merit(lp, crossed)
    if verbose:
        print(f"[crossover] merit warm={m_warm:.3e} crossover={m_cross:.3e}")
    return crossed if m_cross <= m_warm else warm


def vertex_distance_report(lp: LP, sol: SaddleState, tol: float = 1e-6, verbose=True):
    """
    Estimate how far a solution is from an LP *vertex* — i.e. whether active-set
    / simplex polishing is likely to succeed (near a vertex) or diverge (deep in
    the interior of an optimal face).

    Two complementary, solve-free measures:

    1. **Free-vs-active count** (structural). A vertex has at most ``n_active``
       basic (interior) variables, the rest pinned at bounds. ``n_interior``
       well above ``n_active_rows`` ⇒ under-determined / non-vertex face; a huge
       active-inequality count vs few interior vars ⇒ a fractional point far from
       any vertex. The ratio ``n_interior / n_active_rows`` near 1 is "near a
       vertex".
    2. **Bound distance** (geometric). At a vertex every non-basic variable sits
       exactly on a bound. ``interior_mass`` (total distance of interior vars to
       their nearest finite bound) and ``max_interior_dist`` measure directly how
       far the point is from having a vertex's at-bound structure.

    Returns a dict of the metrics. With ``verbose`` (default) also prints them.
    """
    x = np.asarray(sol.primal, dtype=np.float64)
    lb = np.asarray(lp.lower_bounds, dtype=np.float64)
    ub = np.asarray(lp.upper_bounds, dtype=np.float64)

    # Distance of each variable to its nearest *finite* bound (inf if free both
    # ways). Zero ⇒ sitting on a bound (non-basic at a vertex).
    dist_lo = np.where(np.isfinite(lb), np.abs(x - lb), np.inf)
    dist_hi = np.where(np.isfinite(ub), np.abs(x - ub), np.inf)
    dist_bound = np.minimum(dist_lo, dist_hi)

    interior = dist_bound > tol  # strictly off every bound ⇒ candidate basic
    n_interior = int(interior.sum())
    finite_dist = dist_bound[np.isfinite(dist_bound)]
    interior_mass = float(finite_dist[finite_dist > tol].sum())
    max_interior_dist = float(finite_dist.max()) if finite_dist.size else 0.0

    # Active rows: all equalities + tight inequalities (same classification the
    # KKT solve uses), to compare against the interior-variable count.
    n_eq = lp.A_eq.shape[0]
    if lp.A_ineq.shape[0] > 0:
        ineq_tight = int(
            (np.abs(np.asarray(lp.A_ineq @ x) - np.asarray(lp.b_ineq)) <= tol).sum()
        )
    else:
        ineq_tight = 0
    n_active_rows = n_eq + ineq_tight

    # A vertex needs n_interior basic vars determined by n_active_rows rows; the
    # ratio (and its sign) says which way the system leans.
    ratio = n_interior / n_active_rows if n_active_rows > 0 else float("inf")

    metrics = {
        "n_vars": int(x.shape[0]),
        "n_interior": n_interior,
        "n_at_bound": int(x.shape[0]) - n_interior,
        "n_eq_rows": n_eq,
        "n_tight_ineq": ineq_tight,
        "n_active_rows": n_active_rows,
        "interior_over_active": ratio,
        "interior_mass": interior_mass,
        "max_interior_dist": max_interior_dist,
    }

    if verbose:
        print("==== Vertex-distance report ====")
        print(f"  variables        : {metrics['n_vars']}")
        print(
            f"  interior / at-bound : {n_interior} / {metrics['n_at_bound']}"
            f"   (interior = strictly off bounds, tol={tol:g})"
        )
        print(
            f"  active rows      : {n_active_rows}"
            f"  (eq {n_eq} + tight ineq {ineq_tight})"
        )
        print(
            f"  interior / active : {ratio:.2f}"
            "   (~1 ⇒ near a vertex; ≫1 ⇒ under-determined face;"
            " ineq≫interior ⇒ fractional, far from vertex)"
        )
        print(
            f"  bound distance   : mass {interior_mass:.3e},"
            f" max {max_interior_dist:.3e}   (0 ⇒ on a vertex)"
        )

    return metrics


# %%
