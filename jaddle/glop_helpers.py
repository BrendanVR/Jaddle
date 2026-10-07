import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import scipy.sparse as sp
from ortools.linear_solver import linear_solver_pb2

from jaddle.highs_helpers import rows_to_standard_form

# The OR-Tools Python bindings don't expose glop's presolver, so it runs in a
# small C++ helper (tools/glop_presolve, built with tools/glop_presolve/build.sh).
# Override its location with the JADDLE_GLOP_PRESOLVE environment variable.
GLOP_PRESOLVE_BINARY = os.environ.get(
    "JADDLE_GLOP_PRESOLVE",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tools",
        "glop_presolve",
        "build",
        "glop_presolve",
    ),
)


def glop_presolve(path: str, verbose: bool = False):
    """
    Reads an MPS file, relaxes integrality and presolves the LP relaxation with
    OR-Tools' glop presolver (MainLpPreprocessor, with PDLP's settings: no
    dualization, no implied-free preprocessor, no scaling).

    Returns: (LP, offset, info) where ``offset`` is the constant objective term
    outside ``c`` in minimisation form (min c^T x + offset; for a maximisation
    model the original objective is ``-(c^T x + offset)``, as in
    ``sciopt_helpers.scip_to_standard_form_sparse``) and ``info`` is the
    helper's report (status, original/presolved sizes, read/presolve seconds).

    Raises RuntimeError if glop's presolve already decided the problem (status
    other than INIT, e.g. PRIMAL_INFEASIBLE), since no LP is left to solve.
    """
    if not os.path.exists(GLOP_PRESOLVE_BINARY):
        raise FileNotFoundError(
            f"glop_presolve helper not found at {GLOP_PRESOLVE_BINARY}; "
            "build it with tools/glop_presolve/build.sh"
        )
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "presolved.pb")
        cmd = [GLOP_PRESOLVE_BINARY, path, out_path]
        if verbose:
            cmd.insert(1, "--verbose")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"glop_presolve failed on {path} (exit {proc.returncode}): "
                f"{proc.stderr.strip()}"
            )
        lines = proc.stdout.strip().splitlines()
        if verbose:
            print("\n".join(lines[:-1]))
            sys.stdout.flush()
        info = json.loads(lines[-1])
        if info["status"] != "INIT":
            raise RuntimeError(
                f"glop presolve decided {path} outright (status {info['status']})"
            )
        model = linear_solver_pb2.MPModelProto()
        with open(out_path, "rb") as f:
            model.ParseFromString(f.read())

    lp, offset = mpmodel_to_standard_form_sparse(
        model, scaling_factor=info["objective_scaling_factor"]
    )
    return lp, offset, info


def mpmodel_to_standard_form_sparse(model, scaling_factor: float = 1.0):
    """
    Converts an MPModelProto (linear constraints only) to Jaddle's standard
    form. Maximisation is negated into minimisation. ``scaling_factor`` is the
    glop objective scaling factor (true objective = factor * (c^T x + offset)),
    which MPModelProto can't store; it is folded into ``c`` and the offset.

    Returns: (LP, offset).
    """
    if len(model.general_constraint) or model.HasField("quadratic_objective"):
        raise ValueError("model has non-linear parts; cannot export to a Jaddle LP")

    sense = -1.0 if model.maximize else 1.0
    scale = sense * scaling_factor
    variables = model.variable
    c = scale * np.array([v.objective_coefficient for v in variables], dtype=np.float64)
    lower_bounds = np.array([v.lower_bound for v in variables], dtype=np.float64)
    upper_bounds = np.array([v.upper_bound for v in variables], dtype=np.float64)
    offset = scale * model.objective_offset

    cons = model.constraint
    row_nnz = np.array([len(k.var_index) for k in cons], dtype=np.int64)
    indptr = np.concatenate([[0], np.cumsum(row_nnz)])
    indices = np.fromiter(
        (j for k in cons for j in k.var_index), dtype=np.int64, count=indptr[-1]
    )
    data = np.fromiter(
        (a for k in cons for a in k.coefficient), dtype=np.float64, count=indptr[-1]
    )
    A = sp.csr_matrix((data, indices, indptr), shape=(len(cons), len(variables)))
    row_lower = np.array([k.lower_bound for k in cons], dtype=np.float64)
    row_upper = np.array([k.upper_bound for k in cons], dtype=np.float64)

    lp = rows_to_standard_form(c, A, row_lower, row_upper, lower_bounds, upper_bounds)
    return lp, float(offset)
