import numpy as np
import scipy.sparse as sp
import pyscipopt as scip
from pyscipopt import SCIP_PARAMSETTING
from jaddle.jaddle_basic_types import LP


def try_set(model, name, value):
    try:
        model.setParam(name, value)
        print(f"set {name} = {value}")
    except KeyError:
        print(f"skipped (not in this build): {name}")


def read_relaxed_model(path: str, presolve: bool = False, quiet: bool = True):
    """
    Reads an LP/MPS file into a PySCIPOpt Model and relaxes integrality, so the
    model is the LP relaxation. With ``presolve=True`` SCIP's presolver is run
    and the TRANSFORMED (presolved) problem is what ``scip_to_standard_form_sparse``
    will export.

    Presolve settings: constraint upgrading is disabled so every row stays a
    linear-type constraint, and objective scaling is disabled so the transformed
    objective is ``c^T x + offset`` with no hidden scale factor.
    """
    model = scip.Model()
    if quiet:
        model.hideOutput()
    model.readProblem(path)
    for v in model.getVars():
        if v.vtype() != "CONTINUOUS":
            model.chgVarType(v, "C")
    if presolve:
        # Symmetry detection (graph automorphisms) can hang for minutes on large
        # instances (neos-4260495-otere) and is useless for an LP relaxation --
        # it only serves integer branching.
        # PaPILO (milp presolver)
        model.setPresolve(SCIP_PARAMSETTING.AGGRESSIVE)
        try_set(model, "misc/usesymmetry", 0)
        try_set(model, "presolving/milp/maxrounds", -1)  # no cap on rounds
        try_set(model, "presolving/milp/threads", 0)  # 0 = auto; or set your core count
        try_set(model, "presolving/milp/enableprobing", True)
        try_set(model, "presolving/milp/enabledualinfer", True)
        try_set(model, "presolving/milp/enabledomcol", True)
        try_set(model, "presolving/milp/enableparallelrows", True)
        try_set(model, "presolving/milp/enablemultiaggr", True)  # off by default
        try_set(model, "presolving/milp/enablesparsify", True)  # off by default
        try_set(
            model, "presolving/milp/modifyconsfac", 1.0
        )  # allow more constraint modification
        model.presolve()
    return model


def scip_to_standard_form_sparse(model: scip.Model, transformed: bool = None):
    """
    Converts a PySCIPOpt Model to standard form matrices:
        min c^T x
        s.t. A_eq x = b_eq
             A_ineq x <= b_ineq
             x >= lower_bounds
             x <= upper_bounds
    Integrality is ignored (the LP relaxation is exported). Maximisation
    problems are negated into minimisation form.

    ``transformed`` selects the presolved problem; by default it is used when
    the model has been presolved/transformed.

    Returns: (LP, offset) where ``offset`` is the constant objective term that
    lives outside ``c`` (in minimisation form, i.e. min c^T x + offset). For a
    maximisation model the original objective is ``-(c^T x + offset)``. This is
    the analogue of HiGHS' ``HighsLp.offset_``.
    """
    if transformed is None:
        transformed = model.getStage() >= scip.SCIP_STAGE.TRANSFORMED

    inf = model.infinity()

    def _clean(a):
        a = np.asarray(a, dtype=np.float64)
        a[a >= inf] = np.inf
        a[a <= -inf] = -np.inf
        return a

    vars_ = model.getVars(transformed=transformed)
    num_col = len(vars_)
    col_of = {v.ptr(): j for j, v in enumerate(vars_)}

    if transformed:
        # SCIP's transformed problem is always a minimisation.
        sense = 1.0
        offset = model.getObjoffset(original=False) + model.getObjoffset(original=True)
        lower_bounds = _clean([v.getLbGlobal() for v in vars_])
        upper_bounds = _clean([v.getUbGlobal() for v in vars_])
    else:
        sense = -1.0 if model.getObjectiveSense() == "maximize" else 1.0
        offset = sense * model.getObjoffset(original=True)
        lower_bounds = _clean([v.getLbOriginal() for v in vars_])
        upper_bounds = _clean([v.getUbOriginal() for v in vars_])
    c = sense * np.array([v.getObj() for v in vars_], dtype=np.float64)

    # Build A in COO form from the linear-type constraints
    rows, cols, vals = [], [], []
    row_lower, row_upper = [], []
    for i, cons in enumerate(model.getConss(transformed=transformed)):
        if not cons.isLinearType():
            raise ValueError(
                f"constraint {cons.name!r} of type {cons.getConshdlrName()!r} "
                "is not linear; cannot export to a Jaddle LP"
            )
        cvars = model.getConsVars(cons)
        cvals = model.getConsVals(cons)
        rows.extend([i] * len(cvars))
        cols.extend(col_of[v.ptr()] for v in cvars)
        vals.extend(cvals)
        row_lower.append(model.getLhs(cons))
        row_upper.append(model.getRhs(cons))

    num_row = len(row_lower)
    A = sp.csr_matrix(
        (np.asarray(vals, dtype=np.float64), (rows, cols)),
        shape=(num_row, num_col),
        dtype=np.float64,
    )
    row_lower = _clean(row_lower)
    row_upper = _clean(row_upper)

    # Equality constraints: use a numeric tolerance and require finite bounds
    eps = 1e-8
    finite_lower_all = np.isfinite(row_lower)
    finite_upper_all = np.isfinite(row_upper)
    eq_mask = (
        finite_lower_all & finite_upper_all & (np.abs(row_lower - row_upper) <= eps)
    )

    A_eq = A[eq_mask, :].tocsc().astype(np.float64)
    b_eq = row_lower[eq_mask].astype(np.float64)

    # Inequality constraints (rows not treated as equalities)
    ineq_mask = ~eq_mask
    A_ineq_rows = A[ineq_mask, :]
    row_lower_ineq = row_lower[ineq_mask]
    row_upper_ineq = row_upper[ineq_mask]

    finite_upper = np.isfinite(row_upper_ineq)
    finite_lower = np.isfinite(row_lower_ineq)

    matrices = []
    vectors = []

    if finite_upper.any():
        matrices.append(A_ineq_rows[finite_upper].tocsc().astype(np.float64))
        vectors.append(row_upper_ineq[finite_upper].astype(np.float64))

    if finite_lower.any():
        matrices.append((-A_ineq_rows[finite_lower]).tocsc().astype(np.float64))
        vectors.append((-row_lower_ineq[finite_lower]).astype(np.float64))

    if len(matrices) > 0:
        A_ineq = sp.vstack(matrices, format="csc")
        b_ineq = np.concatenate(vectors)
    else:
        # No inequality rows: return empty (0 x num_col) matrix and empty RHS
        A_ineq = sp.csc_matrix((0, num_col), dtype=np.float64)
        b_ineq = np.empty(0, dtype=np.float64)

    return LP(c, A_eq, b_eq, A_ineq, b_ineq, lower_bounds, upper_bounds), offset
