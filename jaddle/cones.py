"""Second-order, positive semidefinite and exponential cones on the LP
solver's variables.

A ``Cones`` spec puts cone constraints on a tail of the primal vector, in the
style of SCS's cone dimensions: second-order cones first, then PSD cones, then
exponential cones. With ``n`` variables and
``Cones(soc=(3, 5), psd=(4,), exp=2)`` the first ``n - 24`` variables are
ordinary (boxed) LP variables, the next 3 lie in one second-order cone, the
next 5 in another, the next 10 are a 4 x 4 PSD matrix in ``svec`` form, and the
last 6 are two exponential cones:

- **Second-order cone** of size ``m``: ``{(t, x) in R x R^(m-1) : ‖x‖ <= t}``.
- **PSD cone** of side ``d``: the upper triangle of a symmetric ``d x d``
  matrix, row by row (equivalently the lower triangle column by column, SCS's
  order), with off-diagonal entries multiplied by ``√2``, so that the vector
  inner product equals the matrix (trace) inner product. Its length is
  ``d (d + 1) / 2``.
- **Exponential cone**, 3 variables:
  ``closure{(x, y, z) : y exp(x / y) <= z, y > 0}`` (CVXPY's and SCS's
  convention). Projected with Friberg's root-finding method (H. Friberg,
  "Projection onto the exponential cone: a univariate root-finding problem",
  2021), ported from SCS's implementation.

The dual condition on a cone block is that its reduced cost ``r`` lies in the
dual cone ``K*``, and ``r - Π_{K*}(r) = -Π_K(-r)`` (Moreau), so only the
projection onto ``K`` is needed. The second-order and PSD cones are self-dual;
the exponential cone is not. Cone variables must have infinite bounds: the
primal projection is then the box on the head and the cones on the tail, with
no intersection to project onto. The exponential cone's projection is built
for float64; float32 works but is less accurate near the cone's edges.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class Cones:
    """Cone blocks on the tail of the primal vector (see the module docstring).

    Args:
        soc: Sizes of the second-order cones, each at least 1.
        psd: Side lengths of the PSD cones, each at least 1.
        exp: Number of exponential cones (3 variables each).
    """

    soc: tuple = ()
    psd: tuple = ()
    exp: int = 0

    def __post_init__(self):
        object.__setattr__(self, "soc", tuple(int(m) for m in self.soc))
        object.__setattr__(self, "psd", tuple(int(d) for d in self.psd))
        object.__setattr__(self, "exp", int(self.exp))
        if any(m < 1 for m in self.soc) or any(d < 1 for d in self.psd):
            raise ValueError(f"cone sizes must be positive, got {self}")
        if self.exp < 0:
            raise ValueError(f"exp must be nonnegative, got {self.exp}")

    @property
    def soc_size(self):
        return sum(self.soc)

    @property
    def psd_size(self):
        return sum(d * (d + 1) // 2 for d in self.psd)

    @property
    def size(self):
        """Number of variables in cone blocks."""
        return self.soc_size + self.psd_size + 3 * self.exp

    def __bool__(self):
        return self.size > 0

    def block_sizes(self):
        """The length of each cone block, in tail order."""
        return (
            list(self.soc)
            + [d * (d + 1) // 2 for d in self.psd]
            + [3] * self.exp
        )

    # -- host-side index structure (static; computed once per spec)

    @functools.cached_property
    def _soc_index(self):
        # seg[j]: which cone tail entry j belongs to; head: its first entry.
        seg = np.repeat(np.arange(len(self.soc), dtype=np.int32), self.soc)
        starts = np.concatenate([[0], np.cumsum(self.soc)[:-1]]).astype(np.int32)
        head = np.zeros(self.soc_size, bool)
        head[starts] = True
        return seg, starts, head

    @functools.cached_property
    def _psd_groups(self):
        # PSD blocks grouped by side, so each group projects with one batched
        # eigh: {d: (positions (count, d(d+1)/2) into the PSD part of the
        # tail, rows, cols)}.
        groups = {}
        offset = 0
        for d in self.psd:
            length = d * (d + 1) // 2
            groups.setdefault(d, []).append(np.arange(offset, offset + length))
            offset += length
        out = {}
        for d, positions in groups.items():
            rows, cols = np.triu_indices(d)
            out[d] = (np.stack(positions).astype(np.int32), rows, cols)
        return out

    def column_groups(self, n):
        """A group id per column of an ``n``-variable problem: each linear
        column its own group, each cone block one group. Scales that must be
        uniform on a cone block are tied by group."""
        n_lin = n - self.size
        if n_lin < 0:
            raise ValueError(f"{n} variables cannot hold cones of total size {self.size}")
        sizes = self.block_sizes()
        block = np.repeat(np.arange(len(sizes), dtype=np.int32), sizes)
        return np.concatenate([np.arange(n_lin, dtype=np.int32), n_lin + block])

    # -- projections (traceable)

    def project(self, v, guess=None):
        """Euclidean projection of a tail vector ``v`` (length ``size``) onto
        the product of the cones.

        With ``guess`` (one root per exponential cone, see ``project_exp``)
        the result is ``(projection, roots)``, the roots to pass as the next
        guess."""
        psd_end = self.soc_size + self.psd_size
        parts = []
        if self.soc:
            parts.append(self._project_soc(v[: self.soc_size]))
        if self.psd:
            parts.append(self._project_psd(v[self.soc_size : psd_end]))
        if self.exp:
            exp_part = project_exp(v[psd_end:].reshape(-1, 3), guess)
            if guess is not None:
                exp_part, guess = exp_part
            parts.append(exp_part.reshape(-1))
        out = jnp.concatenate(parts) if len(parts) > 1 else parts[0]
        return out if guess is None else (out, guess)

    def dual_residual(self, r):
        """``r - Π_{K*}(r) = -Π_K(-r)``, the projection of ``r`` onto the
        polar of the dual cone (Moreau). Its norm is the distance from ``r`` to
        ``K*``: the dual infeasibility of a reduced cost ``r`` on the cone
        blocks."""
        return -self.project(-r)

    def tie(self, d):
        """Replace each cone block of a per-column vector ``d`` (length ``n``,
        cones in the tail) by the block's geometric mean, so a diagonal
        scaling by ``d`` maps each cone to itself."""
        if not self:
            return d
        n = d.shape[0]
        n_lin = n - self.size
        groups = jnp.asarray(self.column_groups(n)[n_lin:] - n_lin)
        num = len(self.block_sizes())
        log_tail = jnp.log(d[n_lin:])
        mean = jax.ops.segment_sum(log_tail, groups, num_segments=num) / jnp.asarray(
            self.block_sizes(), d.dtype
        )
        return jnp.concatenate([d[:n_lin], jnp.exp(mean)[groups]])

    def _project_soc(self, v):
        seg, starts, head = self._soc_index
        num = len(self.soc)
        t = v[starts]
        sq = jax.ops.segment_sum(jnp.where(head, 0.0, v * v), seg, num_segments=num)
        nx = jnp.sqrt(sq)
        inside = nx <= t
        zero = nx <= -t
        alpha = 0.5 * (t + nx)
        x_scale = jnp.where(
            inside, 1.0, jnp.where(zero, 0.0, alpha / jnp.where(nx > 0, nx, 1.0))
        )
        new_t = jnp.where(inside, t, jnp.where(zero, 0.0, alpha))
        return jnp.where(head, new_t[seg], v * x_scale[seg])

    def _project_psd(self, v):
        # `v` is the PSD part of the tail.
        out = v
        for d, (pos, rows, cols) in self._psd_groups.items():
            out = out.at[pos].set(_project_svec(v[pos], d, rows, cols))
        return out


def _project_svec(blocks, d, rows, cols):
    """Project a batch of svec vectors (``count x d(d+1)/2``) onto the PSD
    cone."""
    off = rows != cols
    w = jnp.where(off, 1.0 / np.sqrt(2.0), 1.0).astype(blocks.dtype)
    vals = blocks * w
    mat = jnp.zeros((blocks.shape[0], d, d), blocks.dtype)
    mat = mat.at[:, rows, cols].set(vals).at[:, cols, rows].set(vals)
    lam, U = jnp.linalg.eigh(mat)
    proj = (U * jnp.maximum(lam, 0.0)[:, None, :]) @ jnp.swapaxes(U, -1, -2)
    return proj[:, rows, cols] / w


# %%
# Exponential cone: Friberg's method, ported from SCS (src/exp_cone.c, MIT).
# Each loop runs a fixed number of iterations with a `done` mask in place of
# SCS's early exits, so the projection traces under jit and vmap. The cone is
# (r, s, t) = (x, y, z) with s exp(r / s) <= t.

_EXP_INF = 1e15
_EXP_TOL = 1e-8
_NEWTON_ITERS = 20
# Newton steps run unconditionally before checking for convergence. From the
# previous iteration's root (the solver's warm start) the search almost always
# finishes within these steps. On GPU a rolled loop step is its own kernel
# launch (~23 us) and a step that tests whether to continue costs ~90 us (the
# test goes back to the host), so off CPU the prefix is unrolled: it fuses
# into one kernel, and the `cond` after it is the only host round trip. On
# the 100-asset Kelly problem that took a GPU epoch from 69 to 44 ms. On CPU
# unrolling made an epoch slower (143 -> 169 ms), so there the loop stays
# rolled.
_NEWTON_FIXED = 4
_BISECTION_ITERS = 40


def _hfun(r0, s0, t0, rho):
    exprho = jnp.exp(rho)
    expnegrho = 1.0 / exprho
    f = (
        ((rho - 1.0) * r0 + s0) * exprho
        - (r0 - rho * s0) * expnegrho
        - (rho * (rho - 1.0) + 1.0) * t0
    )
    df = (
        (rho * r0 + s0) * exprho
        + (r0 - (rho - 1.0) * s0) * expnegrho
        - (2.0 * rho - 1.0) * t0
    )
    return f, df


def _safe(x):
    return jnp.where(x == 0.0, 1.0, x)


def _sq_dist(a, b):
    return sum((ai - bi) ** 2 for ai, bi in zip(a, b))


def _exp_heuristics(r0, s0, t0):
    """SCS's cheap candidates for the projections onto the cone (``vp``) and
    its polar (``vd``), with their squared distances."""
    # Primal: the perspective boundary, or raise t on the interior.
    vp = (jnp.minimum(r0, 0.0), jnp.zeros_like(s0), jnp.maximum(t0, 0.0))
    pdist = _sq_dist((r0, s0, t0), vp)
    tp = jnp.maximum(t0, s0 * jnp.exp(r0 / jnp.where(s0 > 0.0, s0, 1.0)))
    p_int = (s0 > 0.0) & ((tp - t0) ** 2 < pdist)
    vp = tuple(jnp.where(p_int, a, b) for a, b in zip((r0, s0, tp), vp))
    pdist = jnp.where(p_int, (tp - t0) ** 2, pdist)
    # Polar: its boundary, or lower t on its interior.
    vd = (jnp.zeros_like(r0), jnp.minimum(s0, 0.0), jnp.minimum(t0, 0.0))
    ddist = _sq_dist((r0, s0, t0), vd)
    td = jnp.minimum(t0, -r0 * jnp.exp(s0 / jnp.where(r0 > 0.0, r0, 1.0) - 1.0))
    d_int = (r0 > 0.0) & ((t0 - td) ** 2 < ddist)
    vd = tuple(jnp.where(d_int, a, b) for a, b in zip((r0, s0, td), vd))
    ddist = jnp.where(d_int, (t0 - td) ** 2, ddist)
    return vp, pdist, vd, ddist


def _exp_bracket(r0, s0, t0, pdist, ddist):
    """SCS's ``exp_search_bracket``: bounds on the root of ``_hfun``."""
    disc = jnp.sqrt(jnp.maximum(r0 * r0 + s0 * s0 - r0 * s0, 0.0))
    psi_p = jnp.where(
        r0 > s0, (r0 - s0 + disc) / _safe(r0), -s0 / _safe(r0 - s0 - disc)
    )
    ppsi = ((psi_p - 1.0) * r0 + s0) / (psi_p * (psi_p - 1.0) + 1.0)
    psi_d = jnp.where(
        s0 > r0, (r0 - disc) / _safe(s0), (r0 - s0) / _safe(r0 + disc)
    )
    dpsi = (r0 - psi_d * s0) / (psi_d * (psi_d - 1.0) + 1.0)

    def pomega(rho):
        val = jnp.exp(rho) / (rho * (rho - 1.0) + 1.0)
        return jnp.where(rho < 2.0, jnp.minimum(val, np.exp(2.0) / 3.0), val)

    def domega(rho):
        val = -jnp.exp(-rho) / (rho * (rho - 1.0) + 1.0)
        return jnp.where(rho > -1.0, jnp.maximum(val, -np.exp(1.0) / 3.0), val)

    def safediv_pos(x, y):
        return x / jnp.maximum(y, 1e-18)

    Dp = jnp.sqrt(jnp.maximum(pdist - jnp.minimum(s0, 0.0) ** 2, 0.0))
    Dd = jnp.sqrt(jnp.maximum(ddist - jnp.minimum(r0, 0.0) ** 2, 0.0))
    low = jnp.full_like(r0, -_EXP_INF)
    upr = jnp.full_like(r0, _EXP_INF)
    low = jnp.where(
        t0 > 0.0,
        jnp.maximum(low, jnp.log(jnp.where(t0 > 0.0, t0, 1.0) / _safe(ppsi))),
        low,
    )
    upr = jnp.where(
        t0 < 0.0,
        jnp.minimum(upr, -jnp.log(jnp.where(t0 < 0.0, -t0, 1.0) / _safe(dpsi))),
        upr,
    )

    baselow = jnp.where(r0 > 0.0, 1.0 - s0 / _safe(r0), -_EXP_INF)
    low = jnp.where(r0 > 0.0, jnp.maximum(low, baselow), low)
    tpu = jnp.maximum(1e-12, jnp.minimum(Dd, Dp + t0))
    val = r0 * pomega(low)
    sgn = jnp.where(val < 0.0, -1.0, 1.0)
    curbnd = jnp.maximum(low, baselow + safediv_pos(tpu, jnp.abs(val)) * sgn)
    upr = jnp.where(r0 > 0.0, jnp.minimum(upr, curbnd), upr)

    baseupr = jnp.where(s0 > 0.0, r0 / _safe(s0), _EXP_INF)
    upr = jnp.where(s0 > 0.0, jnp.minimum(upr, baseupr), upr)
    tdl = -jnp.maximum(1e-12, jnp.minimum(Dp, Dd - t0))
    val = s0 * domega(upr)
    sgn = jnp.where(val < 0.0, -1.0, 1.0)
    curbnd = jnp.minimum(upr, baseupr - safediv_pos(tdl, jnp.abs(val)) * sgn)
    low = jnp.where(s0 > 0.0, jnp.maximum(low, curbnd), low)

    # Guarantee a valid bracket (rounding can flip or overshoot the bounds).
    low = jnp.clip(jnp.minimum(low, upr), baselow, baseupr)
    upr = jnp.clip(jnp.maximum(low, upr), baselow, baseupr)
    fl, _ = _hfun(r0, s0, t0, low)
    fu, _ = _hfun(r0, s0, t0, upr)
    same_sign = (low != upr) & (fl * fu > 0.0)
    keep_low = jnp.abs(fl) < jnp.abs(fu)
    upr = jnp.where(same_sign & keep_low, low, upr)
    low = jnp.where(same_sign & ~keep_low, upr, low)
    return low, upr


def _exp_root(r0, s0, t0, xl, xu, x0=None):
    """SCS's damped Newton root search on ``_hfun``, with its bisection
    fallback, at fixed iteration counts. ``x0`` is a starting guess (clipped
    to the bracket; NaN or ``None`` starts at the bracket's midpoint, as SCS
    does)."""
    eps, dftol, lodamp, hidamp = 1e-15, 1e-13, 0.05, 0.95
    # SCS stops when a step falls below 1e-15 and discards that step. Newton
    # converges quadratically, so a step below 1e-9 leaves an error of about
    # 1e-18: take it and stop. That saves the ~3 steps SCS spends confirming,
    # which is most of the work from a warm start.
    step_tol = 1e-9

    def newton(_, c):
        x, xl, xu, done = c
        f, df = _hfun(r0, s0, t0, x)
        found = jnp.abs(f) <= eps
        stop = done | found
        xl_n = jnp.where(f < 0.0, x, xl)
        xu_n = jnp.where(f < 0.0, xu, x)
        collapsed = xu_n <= xl_n
        mid = 0.5 * (xu_n + xl_n)
        xl_n = jnp.where(collapsed, mid, xl_n)
        xu_n = jnp.where(collapsed, mid, xu_n)
        flat = (jnp.abs(f) >= _EXP_INF) | ~jnp.isfinite(f) | (df < dftol)
        x_plus = x - f / jnp.where(df == 0.0, 1.0, df)
        small = jnp.abs(x_plus - x) <= step_tol * jnp.maximum(1.0, jnp.abs(x_plus))
        x_new = jnp.where(
            x_plus >= xu_n,
            jnp.minimum(lodamp * x + hidamp * xu_n, xu_n),
            jnp.where(
                x_plus <= xl_n, jnp.maximum(lodamp * x + hidamp * xl_n, xl_n), x_plus
            ),
        )
        moves = ~(stop | collapsed | flat)
        return (
            jnp.where(moves, jnp.where(small, jnp.clip(x_plus, xl_n, xu_n), x_new), x),
            jnp.where(stop, xl, xl_n),
            jnp.where(stop, xu, xu_n),
            stop | ~moves | small,
        )

    # A few unconditional steps, then continue only while some search is
    # unfinished (see _NEWTON_FIXED).
    mid = 0.5 * (xl + xu)
    x0 = mid if x0 is None else jnp.where(jnp.isfinite(x0), jnp.clip(x0, xl, xu), mid)
    state = jax.lax.fori_loop(
        0,
        _NEWTON_FIXED,
        newton,
        (x0, xl, xu, jnp.zeros_like(x0, bool)),
        # Decided when traced, so a platform chosen after import counts.
        unroll=jax.default_backend() != "cpu",
    )

    def finish(state):
        return jax.lax.while_loop(
            lambda c: (c[0] < _NEWTON_ITERS) & ~jnp.all(c[4]),
            lambda c: (c[0] + 1, *newton(c[0], c[1:])),
            (_NEWTON_FIXED, *state),
        )[1:]

    x, xl, xu, converged = jax.lax.cond(
        jnp.all(state[3]), lambda state: state, finish, state
    )
    newton_root = jnp.clip(x, xl, xu)

    def bisection(_):
        def step(_, c):
            x, xl, xu, x_plus, done = c
            f, _ = _hfun(r0, s0, t0, x)
            xl_n = jnp.where(f < 0.0, x, xl)
            xu_n = jnp.where(f < 0.0, xu, x)
            xp = 0.5 * (xl_n + xu_n)
            stop = (
                (jnp.abs(xp - x) <= 1e-12 * jnp.maximum(1.0, jnp.abs(xp)))
                | (xp == xl_n)
                | (xp == xu_n)
            )
            keep = done
            return (
                jnp.where(keep | stop, x, xp),
                jnp.where(keep, xl, xl_n),
                jnp.where(keep, xu, xu_n),
                jnp.where(keep, x_plus, xp),
                done | stop,
            )

        out = jax.lax.fori_loop(
            0,
            _BISECTION_ITERS,
            step,
            (x, xl, xu, x, jnp.zeros_like(x, bool)),
        )
        return out[3]

    # The bisection fallback only runs when some Newton search failed.
    fallback = jax.lax.cond(jnp.all(converged), lambda _: newton_root, bisection, None)
    return jnp.where(converged, newton_root, fallback)


def project_exp(v, guess=None):
    """Euclidean projection of each row of ``v`` (``k x 3``) onto the
    exponential cone ``closure{(x, y, z) : y exp(x / y) <= z, y > 0}``.

    With ``guess`` (one root ``ρ`` per row, NaN for none) the root search
    starts there, and the result is ``(projection, roots)``: pass the roots
    back as the next guess when projecting a nearby point, as the solver does
    from one iteration to the next. ``ρ = x / y`` does not depend on scale."""
    # The projection commutes with positive scaling, so each point is
    # projected at unit norm: SCS's tolerances are absolute, and at a norm of
    # 1e-3 they accepted heuristic points 40% off the true projection.
    norm = jnp.linalg.norm(v, axis=1, keepdims=True)
    norm = jnp.where(norm > 0.0, norm, 1.0)
    p, rho = _project_exp_unit(v / norm, guess)
    if guess is None:
        return p * norm
    return p * norm, rho


def _project_exp_unit(v, guess=None):
    r0, s0, t0 = v[:, 0], v[:, 1], v[:, 2]
    vp, pdist, vd, ddist = _exp_heuristics(r0, s0, t0)

    # Skip the root search where the heuristics already solve it.
    err = jnp.maximum(
        jnp.maximum(jnp.abs(vp[0] + vd[0] - r0), jnp.abs(vp[1] + vd[1] - s0)),
        jnp.abs(vp[2] + vd[2] - t0),
    )
    dot = vp[0] * vd[0] + vp[1] * vd[1] + vp[2] * vd[2]
    trivial = (
        ((s0 <= 0.0) & (r0 <= 0.0))
        | (jnp.minimum(pdist, ddist) <= _EXP_TOL**2)
        | ((err <= _EXP_TOL) & (dot <= _EXP_TOL))
    )

    # Elsewhere solve for rho on a bracket; where trivial, a dummy bracket
    # keeps the arithmetic finite (its result is discarded).
    xl, xu = _exp_bracket(r0, s0, t0, pdist, ddist)
    xl = jnp.where(trivial, 0.0, xl)
    xu = jnp.where(trivial, 0.0, xu)
    rho = _exp_root(r0, s0, t0, xl, xu, guess)

    linrho = (rho - 1.0) * r0 + s0
    exprho = jnp.exp(rho)
    quadrho = rho * (rho - 1.0) + 1.0
    valid = (linrho > 0.0) & (jnp.abs(exprho) < _EXP_INF)
    scale = jnp.where(valid, linrho / quadrho, 0.0)
    v_hat = (rho * scale, scale, jnp.where(valid, exprho, 0.0) * scale)
    hat_dist = jnp.where(valid, _sq_dist(v_hat, (r0, s0, t0)), _EXP_INF)
    use_hat = ~trivial & (hat_dist <= pdist)
    out = [jnp.where(use_hat, a, b) for a, b in zip(v_hat, vp)]
    # A trivial cone's dummy root is no guess: keep the one it came with.
    if guess is not None:
        rho = jnp.where(trivial, guess, rho)
    return jnp.stack(out, axis=1), rho


def svec(M):
    """The ``svec`` of a symmetric matrix (NumPy): upper triangle row by row,
    off-diagonals times ``√2``."""
    M = np.asarray(M)
    rows, cols = np.triu_indices(M.shape[0])
    return M[rows, cols] * np.where(rows == cols, 1.0, np.sqrt(2.0))


def smat(v):
    """Inverse of ``svec`` (NumPy)."""
    v = np.asarray(v)
    d = int(round((np.sqrt(8 * v.size + 1) - 1) / 2))
    rows, cols = np.triu_indices(d)
    vals = v / np.where(rows == cols, 1.0, np.sqrt(2.0))
    M = np.zeros((d, d), v.dtype)
    M[rows, cols] = vals
    M[cols, rows] = vals
    return M


def project_primal(cones, x, lower, upper, guess=None):
    """Project a primal vector: box on the linear head, cones on the tail.
    ``cones=None`` (or empty) is the plain box projection. With ``guess``
    (exponential-cone roots, see ``Cones.project``) the result is
    ``(projection, roots)``."""
    from optax.projections import projection_box

    if not cones:
        return projection_box(x, lower, upper)
    n_lin = x.shape[0] - cones.size
    tail = cones.project(x[n_lin:], guess)
    if guess is not None:
        tail, guess = tail
    out = jnp.concatenate(
        [projection_box(x[:n_lin], lower[:n_lin], upper[:n_lin]), tail]
    )
    return out if guess is None else (out, guess)
