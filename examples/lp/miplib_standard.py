# %% [markdown]
# # Solve a Scaled LP using Jaddle Saddle Point Optimisation
# This example demonstrates how to load and solve a MIPLIB linear program (LP) using saddle point optimisation methods implemented in Jaddle.
# We will use the `highspy` library to load a MIPLIB LP from an MPS file.

# %%
import os

# Suppress INFO and WARNING logs from XLA/JAX
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import highspy as hspy
import jaddle.jaddle_optimisers as jo
import jaddle.jaddle_linear as jl
import jaddle.highs_helpers as hh
import jax.numpy as jnp

# %%
jo.configure_jax("float64")

# %% [markdown]
# ## Load the LP
# We load a MIPLIB LP from an MPS file using the `highspy` library.
PROBLEM_NAME = "momentum1"  # name of MIPLIB problem (without .mps extension)
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
highs.setOptionValue("pdlp_optimality_tolerance", 1e-5)
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
solution_jaddle, _ = jl.solve(
    lp,
    verbose=True,
    k_scale=1e3,
    dual_feasibility_tolerance=1e-1,
)

# %%
solution = jl.primal_polish(
    lp,
    solution_jaddle,
    atol=1e-12,
    bound_tol=1e-12,
    active_tol=1e-6,
    primal_damp=1e0,
    dual_damp=1e-12,
)

# %%
print("Jaddle solution:")
print(f"Objective value: {lp.objective(solution_jaddle.primal):.6f}")
print(f"Equality feasibility: {lp.eq_slack(solution_jaddle.primal):.6e}")
print(f"Inequality feasibility: {lp.ineq_slack(solution_jaddle.primal):.6e}")
print(
    f"Complementary Slackness: {lp.complementarity_slack(solution_jaddle.primal, solution_jaddle.dual_ineq):.6e}"
)

# %%
print("Polished solution:")
print(f"Objective value: {lp.objective(solution.primal):.6f}")
print(f"Equality feasibility: {lp.eq_slack(solution.primal):.6e}")
print(f"Inequality feasibility: {lp.ineq_slack(solution.primal):.6e}")
print(
    f"Complementary Slackness: {lp.complementarity_slack(solution.primal, solution.dual_ineq):.6e}"
)

# %%
upper = (solution.primal <= lp.upper_bounds).all()
lower = (solution.primal >= lp.lower_bounds).all()

print(f"Primal solution within bounds: {upper and lower}")

# %%
jl.solve(lp, verbose=True, initial_solution=solution)


# %%
