import numpy as np
import scipy.sparse as sp

import jaddle.jaddle_linear as jl
from jaddle.jaddle_basic_types import SaddleState


def _to_numpy_1d(values, dtype=np.float32):
    arr = np.asarray(values)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr.astype(dtype)


def _parse_infinity_array(values, dtype=np.float32):
    parsed = []
    for value in values:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"inf", "+inf", "infinity", "+infinity"}:
                parsed.append(np.inf)
                continue
            if lowered in {"-inf", "-infinity"}:
                parsed.append(-np.inf)
                continue
        parsed.append(value)
    return _to_numpy_1d(parsed, dtype=dtype)


def _matrix_to_csr(matrix) -> sp.csr_matrix:
    if sp.issparse(matrix):
        return matrix.tocsr().astype(np.float32)

    # jax.experimental.sparse.BCOO objects expose .todense() and often .shape
    if hasattr(matrix, "todense") and hasattr(matrix, "shape"):
        dense = np.asarray(matrix.todense(), dtype=np.float32)
        return sp.csr_matrix(dense)

    return sp.csr_matrix(np.asarray(matrix, dtype=np.float32))


def _extract_cuopt_problem_dict(cuopt_problem) -> dict:
    if isinstance(cuopt_problem, dict):
        return cuopt_problem

    if hasattr(cuopt_problem, "to_dict") and callable(cuopt_problem.to_dict):
        return cuopt_problem.to_dict()

    if hasattr(cuopt_problem, "model_dump") and callable(cuopt_problem.model_dump):
        return cuopt_problem.model_dump()

    required_attrs = [
        "csr_constraint_matrix",
        "constraint_bounds",
        "objective_data",
        "variable_bounds",
    ]
    if all(hasattr(cuopt_problem, attr) for attr in required_attrs):
        return {
            "csr_constraint_matrix": cuopt_problem.csr_constraint_matrix,
            "constraint_bounds": cuopt_problem.constraint_bounds,
            "objective_data": cuopt_problem.objective_data,
            "variable_bounds": cuopt_problem.variable_bounds,
            "maximize": getattr(cuopt_problem, "maximize", False),
            "variable_types": getattr(cuopt_problem, "variable_types", None),
            "variable_names": getattr(cuopt_problem, "variable_names", None),
        }

    raise TypeError(
        "Unsupported cuOpt problem format. Provide a cuOpt JSON dict, object with "
        "to_dict()/model_dump(), or object exposing CSR/bounds/objective fields."
    )


def cuopt_to_standard_form_sparse(cuopt_problem) -> jl.LP:
    """
    Converts cuOpt LP JSON-style problem data to Jaddle standard form:
        min c^T x
        s.t. A_eq x = b_eq
             A_ineq x <= b_ineq
             lower_bounds <= x <= upper_bounds

    Accepts either:
    - dict with cuOpt JSON keys
    - object exposing to_dict()/model_dump() returning that schema
    """
    data = _extract_cuopt_problem_dict(cuopt_problem)

    csr = data["csr_constraint_matrix"]
    offsets = np.asarray(csr["offsets"], dtype=np.int32)
    indices = np.asarray(csr["indices"], dtype=np.int32)
    values = np.asarray(csr["values"], dtype=np.float32)

    num_rows = offsets.size - 1

    variable_bounds = data["variable_bounds"]
    lower_bounds = _parse_infinity_array(variable_bounds["lower_bounds"])
    upper_bounds = _parse_infinity_array(variable_bounds["upper_bounds"])
    num_cols = lower_bounds.size

    A = sp.csr_matrix((values, indices, offsets), shape=(num_rows, num_cols))

    objective_coeffs = data["objective_data"]["coefficients"]
    c = _to_numpy_1d(objective_coeffs)

    maximize = bool(data.get("maximize", False))
    if maximize:
        c = -c

    bounds = data["constraint_bounds"]
    row_lower = _parse_infinity_array(bounds["lower_bounds"])
    row_upper = _parse_infinity_array(bounds["upper_bounds"])

    eps = 1e-8
    finite_lower = np.isfinite(row_lower)
    finite_upper = np.isfinite(row_upper)
    eq_mask = finite_lower & finite_upper & (np.abs(row_lower - row_upper) <= eps)

    A_eq = A[eq_mask, :].tocsc().astype(np.float32)
    b_eq = row_lower[eq_mask].astype(np.float32)

    ineq_mask = ~eq_mask
    A_rows = A[ineq_mask, :]
    row_lower_ineq = row_lower[ineq_mask]
    row_upper_ineq = row_upper[ineq_mask]

    matrices = []
    vectors = []

    finite_upper_ineq = np.isfinite(row_upper_ineq)
    if finite_upper_ineq.any():
        matrices.append(A_rows[finite_upper_ineq].tocsc().astype(np.float32))
        vectors.append(row_upper_ineq[finite_upper_ineq].astype(np.float32))

    finite_lower_ineq = np.isfinite(row_lower_ineq)
    if finite_lower_ineq.any():
        matrices.append((-A_rows[finite_lower_ineq]).tocsc().astype(np.float32))
        vectors.append((-row_lower_ineq[finite_lower_ineq]).astype(np.float32))

    if matrices:
        A_ineq = sp.vstack(matrices, format="csc")
        b_ineq = np.concatenate(vectors)
    else:
        A_ineq = sp.csc_matrix((0, num_cols), dtype=np.float32)
        b_ineq = np.empty(0, dtype=np.float32)

    return jl.LP(c, A_eq, b_eq, A_ineq, b_ineq, lower_bounds, upper_bounds)


def jaddle_to_cuopt_dict(
    lp: jl.LP,
    maximize: bool = False,
    objective_offset: float = 0.0,
    variable_types=None,
    variable_names=None,
) -> dict:
    """
    Converts a Jaddle LP to cuOpt JSON-compatible dictionary format.

    Args:
        lp: Jaddle LP in standard form.
        maximize: If True, emit an equivalent maximization problem for cuOpt.
        objective_offset: Optional objective constant term.
        variable_types: Optional iterable (e.g. ['C', 'C', ...]).
        variable_names: Optional iterable of variable names.
    """
    A_eq = _matrix_to_csr(lp.A_eq)
    A_ineq = _matrix_to_csr(lp.A_ineq)

    A_all = sp.vstack([A_eq, A_ineq], format="csr")

    b_eq = _to_numpy_1d(lp.b_eq)
    b_ineq = _to_numpy_1d(lp.b_ineq)

    eq_rows = b_eq.size
    ineq_rows = b_ineq.size

    row_lower = np.concatenate(
        [
            b_eq,
            np.full((ineq_rows,), -np.inf, dtype=np.float32),
        ]
    )
    row_upper = np.concatenate([b_eq, b_ineq])

    c = _to_numpy_1d(lp.c)
    out_c = -c if maximize else c

    lower_bounds = _to_numpy_1d(lp.lower_bounds)
    upper_bounds = _to_numpy_1d(lp.upper_bounds)

    n = c.size
    if variable_types is None:
        variable_types = ["C"] * n
    if variable_names is None:
        variable_names = [f"x_{i}" for i in range(n)]

    return {
        "csr_constraint_matrix": {
            "offsets": A_all.indptr.astype(np.int32).tolist(),
            "indices": A_all.indices.astype(np.int32).tolist(),
            "values": A_all.data.astype(np.float32).tolist(),
        },
        "constraint_bounds": {
            "lower_bounds": row_lower.astype(np.float32).tolist(),
            "upper_bounds": row_upper.astype(np.float32).tolist(),
        },
        "objective_data": {
            "coefficients": out_c.astype(np.float32).tolist(),
            "scalability_factor": 1.0,
            "offset": float(objective_offset),
        },
        "variable_bounds": {
            "lower_bounds": lower_bounds.astype(np.float32).tolist(),
            "upper_bounds": upper_bounds.astype(np.float32).tolist(),
        },
        "maximize": bool(maximize),
        "variable_types": list(variable_types),
        "variable_names": list(variable_names),
    }


def jaddle_to_cuopt_problem(lp: jl.LP, name: str = "jaddle_lp", **kwargs):
    """
    Builds a cuOpt Problem object from a Jaddle LP.

    This helper uses cuOpt's Python Problem API when available. It falls back
    to an informative ImportError if cuOpt is not installed.
    """
    try:
        from cuopt.linear_programming.problem import (
            Problem,
            VType,
            CType,
            Constraint,
            LinearExpression,
            sense,
        )
    except Exception as exc:
        raise ImportError(
            "cuOpt Python package is required for jaddle_to_cuopt_problem. "
            "Install cuOpt and retry."
        ) from exc

    data = jaddle_to_cuopt_dict(lp, **kwargs)

    problem = Problem(name)

    var_types = data.get("variable_types", [])
    variable_names = data.get("variable_names", [])
    obj_coeffs = np.asarray(data["objective_data"]["coefficients"], dtype=np.float32)
    lb = np.asarray(data["variable_bounds"]["lower_bounds"], dtype=np.float32)
    ub = np.asarray(data["variable_bounds"]["upper_bounds"], dtype=np.float32)

    vars_added = []
    for i in range(lb.size):
        vtype = (
            VType.INTEGER
            if i < len(var_types) and var_types[i] == "I"
            else VType.CONTINUOUS
        )
        var_name = variable_names[i] if i < len(variable_names) else f"x_{i}"
        vars_added.append(
            problem.addVariable(
                lb=float(lb[i]),
                ub=float(ub[i]),
                obj=float(obj_coeffs[i]) if i < obj_coeffs.size else 0.0,
                vtype=vtype,
                name=var_name,
            )
        )

    csr = data["csr_constraint_matrix"]
    offsets = np.asarray(csr["offsets"], dtype=np.int32)
    indices = np.asarray(csr["indices"], dtype=np.int32)
    values = np.asarray(csr["values"], dtype=np.float32)

    row_lower = np.asarray(data["constraint_bounds"]["lower_bounds"], dtype=np.float32)
    row_upper = np.asarray(data["constraint_bounds"]["upper_bounds"], dtype=np.float32)

    eps = 1e-8
    for i in range(offsets.size - 1):
        start = offsets[i]
        end = offsets[i + 1]

        row_vars = [vars_added[int(j)] for j in indices[start:end]]
        row_coeffs = [float(v) for v in values[start:end]]
        expr = LinearExpression(row_vars, row_coeffs, 0.0)

        lo = row_lower[i]
        up = row_upper[i]

        if np.isfinite(lo) and np.isfinite(up) and abs(float(lo) - float(up)) <= eps:
            problem.addConstraint(
                Constraint(expr, CType.EQ, float(lo)), name=f"c_{i}_eq"
            )
        else:
            if np.isfinite(up):
                problem.addConstraint(
                    Constraint(expr, CType.LE, float(up)), name=f"c_{i}_ub"
                )
            if np.isfinite(lo):
                problem.addConstraint(
                    Constraint(expr, CType.GE, float(lo)), name=f"c_{i}_lb"
                )

    obj_sense = sense.MAXIMIZE if data.get("maximize", False) else sense.MINIMIZE
    obj_offset = float(data["objective_data"].get("offset", 0.0))
    problem.setObjective(
        LinearExpression(vars_added, [float(v) for v in obj_coeffs], obj_offset),
        sense=obj_sense,
    )

    return problem


def warmstart_cuopt_problem(cuopt_problem, jaddle_solution: SaddleState) -> bool:
    """
    Warm starts a cuOpt LP object with a Jaddle SaddleState primal solution.

    Returns True if a known warm-start method was found and invoked.
    """
    primal = np.asarray(jaddle_solution.primal, dtype=np.float32).reshape(-1)

    # Common API names across solver libraries.
    candidate_methods = [
        "setInitialPrimalSolution",
        "setPrimalStart",
        "setWarmStart",
        "set_start",
    ]

    for method_name in candidate_methods:
        if hasattr(cuopt_problem, method_name):
            method = getattr(cuopt_problem, method_name)
            try:
                method(primal.tolist())
                return True
            except TypeError:
                method(primal)
                return True

    # cuOpt internals callback fallback when available.
    try:
        from cuopt.linear_programming.internals import SetSolutionCallback

        if hasattr(cuopt_problem, "setSolutionCallback"):
            callback = SetSolutionCallback(primal.tolist())
            cuopt_problem.setSolutionCallback(callback)
            return True
    except Exception:
        pass

    return False
