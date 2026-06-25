# %% [markdown]
# # Solve a Scaled LP using Jaddle Saddle Point Optimisation
# This example demonstrates how to load and solve a MIPLIB linear program (LP) using saddle point optimisation methods implemented in Jaddle.
# We will use the `highspy` library to load a MIPLIB LP from an MPS file.

# %%
import os

# Suppress INFO and WARNING logs from XLA/JAX
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import optax
import highspy as hspy
import jaddle.jaddle_optimisers as jo
import jaddle.jaddle_linear as jl
import jaddle.highs_helpers as hh

# %%
jo.configure_jax("float64")

# %% [markdown]
# ## Load the LP
# We load a MIPLIB LP from an MPS file using the `highspy` library.
PROBLEM_NAME = "nug"  # name of MIPLIB problem (without .mps extension)
highs = hspy.Highs()
highs.readModel(
    f"/home/brendanvr/python/Jaddle/data/{PROBLEM_NAME}.mps"
)  # path to MPS file

# %%
# Relax integrality
for col in range(highs.numVariables):
    highs.changeColIntegrality(col, hspy.HighsVarType.kContinuous)

# %%
highs.setOptionValue("presolve", "off")
highs.setOptionValue("primal_feasibility_tolerance", 1e-3)
highs.setOptionValue("dual_feasibility_tolerance", 1e-3)
highs.setOptionValue("pdlp_optimality_tolerance", 1e-4)
highs.setOptionValue("solver", "pdlp")
highs.solve()


# %% [markdown]
# We convert the LP to Jaddle's sparse format, before applying the selected scaling strategy.
highs_lp = highs.getLp()
lp = hh.highs_to_standard_form_sparse(highs_lp)


# %% [markdown]
# ## Solve the presolved LP using Jaddle's saddle point solver
print("Problem:", PROBLEM_NAME)
jl.lp_summary_statistics(lp)


# %% [markdown]
# ## Stage 1 — crude solve
# A first-order saddle solve to loose tolerances. Cheap; gets us to ~1e-3.

solve_args = {
    "verbose": True,
    "log_every": 1,
    "vertex_bias": 1e-3,
    "dual_feasibility_tolerance": 1e-2,
    "dual_gap_tolerance": 1e-2,
}

solution_crude, _ = jl.solve(lp, **solve_args)

# %% [markdown]
# ## Stage 3 (optional) — least-squares active-set polish
# A fast active-set polish: each pass does one warm-started, damped LSQR solve on
# the active constraints, clips into the box, and adds any violated inequalities
# before re-solving. Cannot diverge; keep its result only if it improves
# feasibility over the iterate (the keep-better gate, applied below).
#
# Polish the *returned* (fully-unscaled) solution against the *same* unscaled
# ``lp`` — never a scaled iterate.
solution_polished = jl.polish(
    lp,
    warm=solution_crude,
    damp=1e0,
    atol=1e-12,
)


# %%
def report(label, sol):
    """Print objective and primal-feasibility residuals for a solution."""
    x = sol.primal
    print(
        f"  {label}: obj={float(lp.objective(x)):.8e} "
        f"eq_res={float(lp.eq_slack(x)):.2e} "
        f"ineq_res={float(lp.ineq_slack(x)):.2e}"
    )


print("\n==== Polish summary ====")
report("crude   ", solution_crude)
report("polished", solution_polished)
# %%

print(lp.complementarity_slack(solution_crude.primal, solution_crude.dual_ineq))
print(lp.complementarity_slack(solution_polished.primal, solution_polished.dual_ineq))
# %%
