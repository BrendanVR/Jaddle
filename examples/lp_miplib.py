# %% [markdown]
# # Solve a MIPLIB LP using Jaddle Saddle Point Optimisation
# This example demonstrates how to load and solve a MIPLIB linear program (LP) using saddle point optimisation methods implemented in Jaddle.
# We will use the `highspy` library to load a MIPLIB LP from an MPS file.
# The LP will then be presolved to simplify the problem before applying saddle point optimisation.

# %%
import os

# Suppress INFO and WARNING logs from XLA/JAX
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import jaddle.jaddle_linear as jl
import jaddle.jaddle_optimisers as jo
import jaddle.highs_helpers as hh
import highspy as hspy
import optax

# %% [markdown]
# ## Load and presolve the LP
# We load a MIPLIB LP from an MPS file using the `highspy` library.
# The LP is then presolved to reduce its size and complexity.
# Finally, we convert the presolved LP into a format compatible with Jaddle.
highs = hspy.Highs()
highs.readModel("<PATH_TO_MPS_FILE>")  # path to MPS file
highs.presolve()
highs_lp = highs.getPresolvedLp()
jaddle_lp = hh.highs_to_standard_form_sparse(highs_lp)

# %% [markdown]
# We solve the presolved LP using HiGHs PDLP solver for comparison.
start_time = time.time()
highs_solution = hh.highs_linear_solver(
    highs_lp, method="pdlp", feasibility_tolerance=1e-5
)
# %%
print("--------------------------------")
print("HiGHs PDLP solver objective:", jaddle_lp.objective(highs_solution))
print("Inequality violation:", jaddle_lp.ineq_slack(highs_solution))
print("Equality violation:", jaddle_lp.eq_slack(highs_solution))
print("--------------------------------")

# %%
lr = optax.cosine_decay_schedule(
    init_value=1e-2,
    decay_steps=int(1e5),
    alpha=1e-4,
    exponent=1.5,
)

optimiser = jo.adamdelta_saddle(lr_primal=lr, lr_dual=1.0)

# %% [markdown]
# ## Solve the scaled, presolved LP using Jaddle's saddle point solver
start_time = time.time()
solution = jl.solve(
    optimiser=optimiser,
    lp=jaddle_lp,
    scale_A=True,
    scale_b=True,
    scale_c=True,
)
print("--------------------------------")
print("Saddle point solver objective:", jaddle_lp.objective(solution["primal"]))
print("Inequality violation:", jaddle_lp.ineq_slack(solution["primal"]))
print("Equality violation:", jaddle_lp.eq_slack(solution["primal"]))
print("Time to solution:", time.time() - start_time)
print("--------------------------------")

# %%
jl.lp_summary_statistics(jaddle_lp)
# %%
