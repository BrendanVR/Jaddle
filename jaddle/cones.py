"""Second-order and positive semidefinite cones on the LP solver's variables.

A ``Cones`` spec puts cone constraints on a tail of the primal vector, in the
style of SCS's cone dimensions. With ``n`` variables and
``Cones(soc=(3, 5), psd=(4,))`` the first ``n - 18`` variables are ordinary
(boxed) LP variables, the next 3 lie in one second-order cone, the next 5 in
another, and the last 10 are a 4 x 4 PSD matrix in ``svec`` form:

- **Second-order cone** of size ``m``: ``{(t, x) in R x R^(m-1) : ‖x‖ <= t}``.
- **PSD cone** of side ``d``: the upper triangle of a symmetric ``d x d``
  matrix, row by row (equivalently the lower triangle column by column, SCS's
  order), with off-diagonal entries multiplied by ``√2``, so that the vector
  inner product equals the matrix (trace) inner product. Its length is
  ``d (d + 1) / 2``.

Both cones are self-dual, so the dual condition on a cone block is that its
reduced cost lies in the same cone. Cone variables must have infinite bounds:
the primal projection is then the box on the head and the cone on the tail,
with no intersection to project onto.
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
    """

    soc: tuple = ()
    psd: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "soc", tuple(int(m) for m in self.soc))
        object.__setattr__(self, "psd", tuple(int(d) for d in self.psd))
        if any(m < 1 for m in self.soc) or any(d < 1 for d in self.psd):
            raise ValueError(f"cone sizes must be positive, got {self}")

    @property
    def soc_size(self):
        return sum(self.soc)

    @property
    def size(self):
        """Number of variables in cone blocks."""
        return self.soc_size + sum(d * (d + 1) // 2 for d in self.psd)

    def __bool__(self):
        return self.size > 0

    def block_sizes(self):
        """The length of each cone block, in tail order."""
        return list(self.soc) + [d * (d + 1) // 2 for d in self.psd]

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
        # eigh: {d: (positions (count, d(d+1)/2) into the tail, rows, cols)}.
        groups = {}
        offset = self.soc_size
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

    def project(self, v):
        """Euclidean projection of a tail vector ``v`` (length ``size``) onto
        the product of the cones."""
        parts = []
        if self.soc:
            parts.append(self._project_soc(v[: self.soc_size]))
        if self.psd:
            parts.append(self._project_psd(v)[self.soc_size :])
        return jnp.concatenate(parts) if len(parts) > 1 else parts[0]

    def polar(self, v):
        """``v - project(v)``: the projection of ``v`` onto the polar cone
        ``-K`` (Moreau). Its norm is the distance from ``v`` to ``K``, which is
        the dual infeasibility of a reduced cost ``v`` on the cone block."""
        return v - self.project(v)

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
        # Works on the whole tail so the group positions index it directly;
        # the SOC part passes through and is dropped by `project`.
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


def project_primal(cones, x, lower, upper):
    """Project a primal vector: box on the linear head, cones on the tail.
    ``cones=None`` (or empty) is the plain box projection."""
    from optax.projections import projection_box

    if not cones:
        return projection_box(x, lower, upper)
    n_lin = x.shape[0] - cones.size
    return jnp.concatenate(
        [
            projection_box(x[:n_lin], lower[:n_lin], upper[:n_lin]),
            cones.project(x[n_lin:]),
        ]
    )
