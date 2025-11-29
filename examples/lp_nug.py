# %% [markdown]
# # Solve a real-world linear program with saddle point optimisation
# This notebook demonstrates how to load and solve a real-world linear program (LP) using saddle point optimisation methods implemented in Jaddle.
# We will use the `highspy` library to load a real-world LP from an MPS file.
# The LP will then be presolved to simplify the problem before applying saddle point optimisation.

# %%
import time
import jaddle.jaddle_linear as jl
import jaddle.highs_helpers as hh
import highspy as hspy

# %% [markdown]
# ## Load and presolve the LP
highs = hspy.Highs()
highs.readModel("../data/nug.mps")
highs.presolve()
highs_lp = highs.getPresolvedLp()
jaddle_lp = hh.highs_to_standard_form_sparse(highs_lp)

# %% [markdown]
# We solve the presolved LP using Highs' pdlp solver for comparison.
start_time = time.time()
highs_solution = hh.highs_linear_solver(
    highs_lp, method="pdlp", feasibility_tolerance=1e-5
)
print("--------------------------------")
print("Highs pdlp solver objective:", jaddle_lp.objective(highs_solution))
print("--------------------------------")

# %% [markdown]
# ## Solve the scaled, presolved LP using Jaddle's saddle point solver
start_time = time.time()
solution = jl.solve(
    iterations_per_epoch=int(1e4),
    lp=jaddle_lp,
    initial_solution=jaddle_lp.initial_solution(),
    constraint_tolerance=1e-3,
    progress_tolerance=1e-4,
    complementarity_tolerance=1e-4,
    exponential_weighting=0.1,
    scale_rc=True,
    scale_objective=True,
)
print("--------------------------------")
print("Saddle point solver objective:", jaddle_lp.objective(solution["primal"]))
print("Inequality violation:", jaddle_lp.ineq_slack(solution["primal"]))
print("Equality violation:", jaddle_lp.eq_slack(solution["primal"]))
print("--------------------------------")

# %%
