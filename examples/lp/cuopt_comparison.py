# %% [markdown]
# # Solve a Scaled LP using Jaddle Saddle Point Optimisation
# This example demonstrates how to load and solve a MIPLIB linear program (LP) using saddle point optimisation methods implemented in Jaddle.
# We will use the `cuopt` library to load a MIPLIB LP from an MPS file.

# %%
import cuopt

# %% [markdown]
# ## Load the LP
# We load a MIPLIB LP from an MPS file using the `cuopt` library.
PROBLEM_NAME = "momentum1"  # name of MIPLIB problem (without .mps extension)
problem = cuopt.linear_programming.Problem.read(
    f"/home/brendanvr/python/Jaddle/data/{PROBLEM_NAME}.mps"
).relax()  # path to MPS file

problem.solve()

# %%
