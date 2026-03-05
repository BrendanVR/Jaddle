# %% [markdown]
# # Solve a Scaled LP using Jaddle Saddle Point Optimisation
# This example demonstrates how to load and solve a MIPLIB linear program (LP) using saddle point optimisation methods implemented in Jaddle.
# We will use the `cuopt` library to load a MIPLIB LP from an MPS file.

# %%
import cuopt

# %%
PROBLEM_NAME = input("Enter the MIPLIB problem name: ")

# %% [markdown]
# ## Load the LP
# We load a MIPLIB LP from an MPS file using the `cuopt` library.
solver = cuopt.linear_programming.Problem.readMPS(
    f"/home/brendanvr/python/Jaddle/data/{PROBLEM_NAME}.mps"
).relax()  # path to MPS file

solver.solve()

# %%
