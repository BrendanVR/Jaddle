# %% [markdown]
# # Solve a Scaled LP using Jaddle Saddle Point Optimisation
# This example demonstrates how to load and solve a MIPLIB linear program (LP) using saddle point optimisation methods implemented in Jaddle.
# We will use the `highspy` library to load a MIPLIB LP from an MPS file.

# %%
import highspy as hspy

# %%
PROBLEM_NAME = input("Enter the MIPLIB problem name: ")
presolve = input("Presolve the problem using HiGHS? (y/n): ").lower() == "y"

# %% [markdown]
# ## Load the LP
# We load a MIPLIB LP from an MPS file using the `highspy` library.
highs = hspy.Highs()
highs.readModel(
    f"/home/brendanvr/python/Jaddle/data/{PROBLEM_NAME}.mps"
)  # path to MPS file


# Relax integrality
info = highs.getInfo()

for col in range(highs.numVariables):
    highs.changeColIntegrality(col, hspy.HighsVarType.kContinuous)

if not presolve:
    highs.setOptionValue("presolve", "off")

highs.setOptionValue("solver", "pdlp")
highs.solve()
