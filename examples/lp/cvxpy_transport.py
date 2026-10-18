# %% [markdown]
# # A linear program in CVXPY, solved by Jaddle
# Three plants make one product and ship it to four markets. The model has
# three variables, each with its own shape and bounds:
#
# * `produce[i]`, the output of plant `i`, between 0 and the plant's capacity;
# * `ship[i, j]`, the amount sent from plant `i` to market `j`, nonnegative;
# * `unmet[j]`, the demand of market `j` left unserved, nonnegative and
#   penalised in the objective.
#
# Everything a plant makes is shipped, every market's demand is served or
# counted as unmet, and the plants' total emissions are capped. The objective is
# the cost of production, shipping and unmet demand. The cheapest plant (South)
# is also the dirtiest, so the cap binds and shifts output to Central.
#
# CVXPY turns the model into an LP, and `jcv.JaddleSolver` solves it with
# `jl.solve_with_presolve`: HiGHS presolve, a Jaddle solve, then HiGHS postsolve.
# The results are checked against HiGHS's own simplex solve through CVXPY.

# %%
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cvxpy as cp
import numpy as np

import jaddle.jaddle_optimisers as jo

jo.configure_jax("float64")

import jaddle.jaddle_cvxpy as jcv  # noqa: E402

# %% [markdown]
# ## Data

# %%
plants = ["North", "Central", "South"]
markets = ["A", "B", "C", "D"]

capacity = np.array([500.0, 400.0, 450.0])  # units per day
production_cost = np.array([10.0, 12.0, 9.0])  # $ per unit
emissions = np.array([1.2, 0.6, 1.5])  # tonnes CO2 per unit
emissions_cap = 1250.0  # tonnes CO2 per day

demand = np.array([300.0, 250.0, 350.0, 200.0])  # units per day
shipping_cost = np.array(  # $ per unit, plant x market
    [
        [2.0, 4.0, 5.0, 3.0],
        [3.0, 1.0, 3.0, 2.0],
        [5.0, 3.0, 1.0, 4.0],
    ]
)
unmet_penalty = 40.0  # $ per unit of demand not served

# %% [markdown]
# ## Model
# `bounds=[0, capacity]` and `nonneg=True` become Jaddle's variable bounds, so
# they cost no constraint rows.

# %%
produce = cp.Variable(len(plants), bounds=[0, capacity], name="produce")
ship = cp.Variable((len(plants), len(markets)), nonneg=True, name="ship")
unmet = cp.Variable(len(markets), nonneg=True, name="unmet")

ship_all = cp.sum(ship, axis=1) == produce
serve_demand = cp.sum(ship, axis=0) + unmet >= demand
emissions_limit = emissions @ produce <= emissions_cap
constraints = [ship_all, serve_demand, emissions_limit]

cost = (
    production_cost @ produce
    + cp.sum(cp.multiply(shipping_cost, ship))
    + unmet_penalty * cp.sum(unmet)
)
problem = cp.Problem(cp.Minimize(cost), constraints)

# %% [markdown]
# ## Reference: HiGHS's simplex method

# %%
problem.solve(solver="HIGHS")
reference_cost = problem.value
reference_produce = produce.value.copy()
reference_prices = serve_demand.dual_value.copy()
reference_carbon_price = float(emissions_limit.dual_value)

# %% [markdown]
# ## Solve with Jaddle
# `tol` sets Jaddle's relative primal, dual and gap tolerances. Other keyword
# arguments of `problem.solve` go to `jl.solve_with_presolve`.

# %%
problem.solve(solver=jcv.JaddleSolver(), tol=1e-8)
stats = problem.solver_stats.extra_stats
print(
    f"Status {problem.status}: daily cost ${problem.value:,.2f} "
    f"(HiGHS ${reference_cost:,.2f}), {stats['epochs']} epochs, presolve "
    f"{stats['presolve']['rows'][0]}x{stats['presolve']['cols'][0]} -> "
    f"{stats['presolve']['rows'][1]}x{stats['presolve']['cols'][1]}"
)

# %% [markdown]
# ## The plan
# Rows are plants, columns markets.

# %%
print("\nShipments (units per day)")
print(f"{'':>8s}" + "".join(f"{m:>8s}" for m in markets) + f"{'total':>8s}{'cap':>8s}")
for i, plant in enumerate(plants):
    row = "".join(f"{v:8.1f}" for v in ship.value[i])
    print(f"{plant:>8s}{row}{produce.value[i]:8.1f}{capacity[i]:8.0f}")
print(f"{'unmet':>8s}" + "".join(f"{v:8.1f}" for v in unmet.value))

# %% [markdown]
# ## What the duals say
# A demand row's dual is the marginal cost of serving one more unit in that
# market. The emissions row's dual is the cost saved by one more tonne of
# allowance, an implicit carbon price.

# %%
print("\nMarginal cost of demand ($ per unit)")
for market, price, reference in zip(markets, serve_demand.dual_value, reference_prices):
    print(f"  {market}: {price:7.3f}  (HiGHS {reference:7.3f})")
print(
    f"Carbon price: ${float(emissions_limit.dual_value):.3f} per tonne "
    f"(HiGHS ${reference_carbon_price:.3f})"
)

assert problem.status == cp.OPTIMAL
assert abs(problem.value - reference_cost) <= 1e-6 * abs(reference_cost)
assert np.allclose(produce.value, reference_produce, atol=1e-3)
assert np.allclose(serve_demand.dual_value, reference_prices, atol=1e-4)

# %%
