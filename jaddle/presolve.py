"""Jaddle-specific presolve reductions, applied on top of HiGHS presolve.

Standard presolvers are tuned for simplex/IPM, where fill-in hurts the
factorisation and a dense row costs little. PDHG has the opposite trade-off:
extra nonzeros are cheap (matvec-bound) but some row structures wreck the
conditioning. The reductions here target those structures.
"""

import numpy as np
import scipy.sparse as sp

from jaddle.jaddle_basic_types import LP


class DefinedVariablePostsolve:
    """Recovers eliminated variables: ``x_full[J] = E @ x + const``."""

    def __init__(self, n_full, keep, elim, E, const):
        self.n_full = n_full
        self.keep = keep
        self.elim = elim
        self.E = E
        self.const = const

    def primal(self, x):
        x = np.asarray(x, dtype=np.float64)
        x_full = np.zeros(self.n_full)
        x_full[self.keep] = x
        # E is indexed by original columns and is zero on the eliminated ones.
        x_full[self.elim] = self.E @ x_full + self.const
        return x_full


class ChainedPostsolve:
    """Composes the postsolves of successive elimination passes."""

    def __init__(self, steps):
        self.steps = steps

    def primal(self, x):
        # Each pass maps its reduced space back to the previous pass's space.
        for step in reversed(self.steps):
            x = step.primal(x)
        return x


def eliminate_defined_variables(
    lp: LP,
    min_row_nnz=100,
    min_costed_row_nnz=2,
    max_col_uses=4,
    max_fill=1.0,
    max_passes=10,
    verbose=False,
):
    """Substitute out variables defined by an equality row.

    Targets rows ``a*z + r.x = b`` where ``z`` has zero cost and appears in no
    other equality row, e.g. ``z = sum_j a_j x_j`` aggregates whose coefficients mirror the
    objective (gmut-*, leo1). Also targets the objective-row variant where ``z``
    is the row's only costed entry (``min z, z = sum_j a_j x_j``, proteindesign*):
    the substitution then turns the dense row into a dense cost vector. HiGHS/SCIP leave these in place: ``z`` is not a
    column singleton (it is used by inequality rows too), not free (so the
    substitution needs an explicit bound row), and substituting a dense row
    exceeds their fill-in limits. For PDHG the dense row is the problem -- it
    hides the objective in a constraint and the primal drifts off feasibility.

    With ``z = E_z.x + const_z`` (``E_z = -r/a``, ``const_z = b/a``):
      * cost:        c += c_z * E_z, objective offset += c_z * const_z
      * inequality:  rows using z get w * E_z added, rhs -= w * const_z
      * bounds of z: become rows ``lb_z <= E_z.x + const_z <= ub_z``, skipped
                     when already implied by the bounds on x.

    Zero-cost aggregate rows are eligible only with at least ``min_row_nnz``
    nonzeros (the dense aggregates that cause trouble). Objective rows are
    eligible from ``min_costed_row_nnz`` nonzeros: a SHORT objective row hides
    the objective just as well (radiationm40-10-02: ``min z, z = 1601 N + T``,
    3 nnz, with N and T each defined by an 11-nnz row). A row shorter than
    ``min_row_nnz`` additionally needs ``z`` to appear in no inequality row, so
    the substitution only relocates the cost. Further passes, for objective
    rows only, repeat until nothing more is eliminated (at most
    ``max_passes``), so the cost is pushed down such chains onto the decision
    variables. ``z`` may appear in at most
    ``max_col_uses`` inequality rows, and total added nonzeros over all passes
    are capped at ``max_fill`` times the original nnz.

    Returns ``(reduced_lp, offset, postsolve)``; add ``offset`` to the reduced
    objective. ``postsolve.primal(x)`` maps a reduced solution back. Returns
    ``(lp, 0.0, None)`` when nothing is eliminated.
    """
    budget = max_fill * (lp.A_eq.nnz + lp.A_ineq.nnz)
    offset, steps = 0.0, []
    for p in range(max_passes):
        # Later passes follow objective chains only: repeating the zero-cost
        # rule picks up second-level aggregates that slow ns1760995 down.
        lp, pass_offset, step, fill = _eliminate_pass(
            lp, min_row_nnz, min_costed_row_nnz, max_col_uses, budget, verbose,
            costed_only=p > 0,
        )
        if step is None:
            break
        offset += pass_offset
        budget -= fill
        steps.append(step)
    if not steps:
        return lp, 0.0, None
    return lp, offset, steps[0] if len(steps) == 1 else ChainedPostsolve(steps)


def _eliminate_pass(
    lp, min_row_nnz, min_costed_row_nnz, max_col_uses, budget, verbose, costed_only
):
    """One elimination pass of ``eliminate_defined_variables``.

    Returns ``(reduced_lp, offset, postsolve, fill)``, with ``postsolve`` None
    when nothing is eliminated; ``fill`` is the nonzero budget consumed.
    """
    c = np.asarray(lp.c, dtype=np.float64)
    lb = np.asarray(lp.lower_bounds, dtype=np.float64)
    ub = np.asarray(lp.upper_bounds, dtype=np.float64)
    b_eq = np.asarray(lp.b_eq, dtype=np.float64)
    b_ineq = np.asarray(lp.b_ineq, dtype=np.float64)
    A_eq = sp.csr_matrix(lp.A_eq, dtype=np.float64)
    A_ineq = sp.csc_matrix(lp.A_ineq, dtype=np.float64)
    n = len(c)
    if A_eq.shape[0] == 0:
        return lp, 0.0, None, 0

    col_eq_uses = np.diff(sp.csc_matrix(A_eq).indptr)
    col_ineq_uses = np.diff(A_ineq.indptr)

    rows, cols, fill = [], [], 0
    for i in range(A_eq.shape[0]):
        lo, hi = A_eq.indptr[i], A_eq.indptr[i + 1]
        nnz = hi - lo
        if nnz < 2 or nnz < min(min_row_nnz, min_costed_row_nnz):
            continue
        idx, val = A_eq.indices[lo:hi], A_eq.data[lo:hi]
        ok = (col_eq_uses[idx] == 1) & (col_ineq_uses[idx] <= max_col_uses)
        ok &= np.abs(val) > 1e-9
        costed = c[idx] != 0.0
        # A short objective row only qualifies when its aggregate is used by no
        # inequality row, so the substitution just relocates the cost (no
        # matrix fill). Without this guard, short rows like ``x = y`` with a
        # costed ``x`` are substituted all over the set (mzzv11, neos-827175).
        # It also needs b = 0: otherwise the objective moves into the offset,
        # which solve() never sees, so the relative gap is normalised by a
        # spuriously large objective (seqsolve2short4288: offset ~ -1e4 on an
        # optimum near 0, falsely certified).
        if (
            nnz >= min_costed_row_nnz
            and np.count_nonzero(costed) == 1
            and ok[costed][0]
            and (
                nnz >= min_row_nnz
                or (col_ineq_uses[idx[costed][0]] == 0 and b_eq[i] == 0.0)
            )
        ):
            # Objective row ``min z, z = sum_j a_j x_j`` (proteindesign*): the
            # row's only costed entry IS the aggregate, so substitute it and its
            # cost moves onto the zero-cost summands.
            ok = costed
        elif costed_only or nnz < min_row_nnz:
            continue
        else:
            # The aggregate carries no cost of its own (the cost is spread over
            # the summands). Without this, a costed summand x_j would be
            # substituted instead, leaving the aggregate -- and the pathology --
            # in place.
            ok &= ~costed
        if not ok.any():
            continue
        # Fewest inequality uses (least fill -- the aggregate variable itself,
        # not an ordinary summand), then largest pivot for stability.
        cand = np.flatnonzero(ok)
        k = cand[np.lexsort((-np.abs(val[cand]), col_ineq_uses[idx[cand]]))[0]]
        j = idx[k]
        # Substituted rows plus up to two bound rows.
        cost = (hi - lo - 1) * (col_ineq_uses[j] + 2)
        if fill + cost > budget:
            continue
        fill += cost
        rows.append(i)
        cols.append(j)

    if not rows:
        return lp, 0.0, None, 0

    rows, cols = np.array(rows), np.array(cols)
    piv = np.asarray(A_eq[rows, cols]).ravel()
    R = A_eq[rows].tolil()
    R[np.arange(len(rows)), cols] = 0.0
    # z_J = E @ x + const (E has zero columns at J: each z_j sits in one eq row).
    E = sp.csr_matrix(-sp.diags(1.0 / piv) @ R.tocsr())
    E.eliminate_zeros()
    const = b_eq[rows] / piv

    c_J = c[cols]
    c_new = c + E.T @ c_J
    offset = float(c_J @ const)

    W = A_ineq[:, cols]
    A_ineq_new = (A_ineq + W @ E).tocsr()
    b_ineq_new = b_ineq - W @ const

    # Bound rows for z, skipped when implied by the bounds on x.
    Ep, En = E.maximum(0), E.minimum(0)
    lb_safe, ub_safe = np.where(np.isfinite(lb), lb, 0.0), np.where(np.isfinite(ub), ub, 0.0)
    has_lb, has_ub = np.isfinite(lb), np.isfinite(ub)
    no_lb, no_ub = (~has_lb).astype(np.float64), (~has_ub).astype(np.float64)
    # Activity bounds of E.x: +inf/-inf wherever an unbounded var contributes.
    act_min = Ep @ lb_safe + En @ ub_safe
    act_min[(Ep @ no_lb - En @ no_ub) > 0] = -np.inf
    act_max = Ep @ ub_safe + En @ lb_safe
    act_max[(Ep @ no_ub - En @ no_lb) > 0] = np.inf
    lbz, ubz = lb[cols] - const, ub[cols] - const
    need_lb = np.isfinite(lbz) & (act_min < lbz)
    need_ub = np.isfinite(ubz) & (act_max > ubz)
    extra_A = [-E[need_lb], E[need_ub]]
    extra_b = [-lbz[need_lb], ubz[need_ub]]

    keep = np.setdiff1d(np.arange(n), cols)
    keep_rows = np.setdiff1d(np.arange(A_eq.shape[0]), rows)
    A_ineq_out = sp.vstack([A_ineq_new] + extra_A, format="csc")[:, keep]
    A_ineq_out.eliminate_zeros()
    reduced = LP(
        c_new[keep],
        A_eq[keep_rows][:, keep].tocsc(),
        b_eq[keep_rows],
        A_ineq_out,
        np.concatenate([b_ineq_new] + extra_b),
        lb[keep],
        ub[keep],
    )
    if verbose:
        print(
            f"Defined-variable presolve: eliminated {len(rows)} eq rows/cols, "
            f"added {int(need_lb.sum() + need_ub.sum())} bound rows, "
            f"nnz {A_eq.nnz + A_ineq.nnz} -> "
            f"{reduced.A_eq.nnz + reduced.A_ineq.nnz}"
        )
    return reduced, offset, DefinedVariablePostsolve(n, keep, cols, E, const), fill
