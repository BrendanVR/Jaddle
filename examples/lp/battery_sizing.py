# %% [markdown]
# # Sizing a Battery with Differentiable Linear Programs
# How big a battery should a site with solar panels buy? The answer depends on how
# the battery would be operated, and operating it well is itself an optimisation
# problem: one linear program per day. This example solves a batch of those LPs at
# once with `jax.vmap`, differentiates their optimal costs with respect to the
# battery's size, and runs gradient descent on the size.

# %%
import jax
import jax.numpy as jnp
import numpy as np
import optax
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import jaddle.jaddle_linear as jl
import jaddle.jaddle_optimisers as jo

jo.configure_jax("float64")
jax.config.update("jax_platform_name", "cpu")  # Using CPU for small problems

# %% [markdown]
# ## Generate Scenarios
# Each scenario is one day in hourly steps: an electricity price with a morning and
# an evening peak, a demand that follows the same pattern, and solar output that
# depends on how cloudy the day is.
rng = np.random.default_rng(0)
T, num_days = 24, 32
hours = np.arange(T)


def bump(centre, width):
    return np.exp(-0.5 * ((hours - centre) / width) ** 2)


prices = (0.12 + 0.10 * bump(8, 1.5) + 0.25 * bump(19, 2.0)) * rng.uniform(
    0.7, 1.3, (num_days, 1)
)  # $/kWh
demand = (1.0 + 0.8 * bump(8, 1.5) + 2.0 * bump(19, 2.0)) * rng.uniform(
    0.8, 1.2, (num_days, 1)
)  # kW
solar = 5.0 * bump(12.5, 2.5) * rng.uniform(0.1, 1.0, (num_days, 1))  # kW
net_load = demand - solar

# %% [markdown]
# ## The Operating Problem
# Given a battery with `energy` kWh of storage and `power` kW of charge and
# discharge rate, one day's cheapest operation is the LP
#
#     minimise    sum_t price[t] * grid[t]
#     subject to  soc[t] = soc[t-1] + efficiency * charge[t] - discharge[t] / efficiency
#                 grid[t] + discharge[t] - charge[t] >= demand[t] - solar[t]
#                 0 <= charge[t], discharge[t] <= power
#                 0 <= soc[t] <= energy,   grid[t] >= 0
#
# The state of charge wraps around (`soc[-1]` is `soc[T-1]`), so the battery ends
# the day as it started. Surplus solar is simply not used. The battery's size
# appears only in the upper bounds.
efficiency = 0.95
identity = sp.identity(T, format="csr")
previous = sp.csr_matrix(np.roll(np.eye(T), -1, axis=1))
zero = sp.csr_matrix((T, T))
# Variables, in order: charge, discharge, grid, soc (T of each).
A_eq = sp.hstack(
    [-efficiency * identity, identity / efficiency, zero, identity - previous]
)
A_ineq = sp.hstack([identity, -identity, -identity, zero])


def upper_bounds(energy, power, xp=np):
    return xp.concatenate(
        [xp.full(2 * T, power), xp.full(T, xp.inf), xp.full(T, energy)]
    )


def operating_lp(price, net_load, energy, power):
    return jl.LP(
        c=np.concatenate([np.zeros(2 * T), price, np.zeros(T)]),
        A_eq=A_eq.tocsc(),
        b_eq=np.zeros(T),
        A_ineq=A_ineq.tocsc(),
        b_ineq=-net_load,
        lower_bounds=np.zeros(4 * T),
        upper_bounds=upper_bounds(energy, power),
    )


# %% [markdown]
# ## The Sizing Problem
# The battery costs a fixed amount per day for each kWh of storage and each kW of
# power. The total daily cost of a design is that plus the average optimal
# operating cost over the scenarios.
#
# `jl.make_optimal_value` returns an LP's optimal value as a differentiable
# function of its numbers. Every day's LP has the same sparsity pattern, so one
# function serves them all: `jax.vmap` solves the whole batch together, and
# `jax.grad` returns the gradient with respect to the design. No derivative is
# taken through the solver's iterations. The gradient of an LP's optimal value
# with respect to a bound is the reduced cost at that bound, which the solver
# has already computed.
energy_cost, power_cost = 0.08, 0.03  # $/kWh/day and $/kW/day
tolerance = dict(
    primal_feasibility_tolerance=1e-4,
    dual_feasibility_tolerance=1e-4,
    dual_gap_tolerance=1e-4,
)
template = operating_lp(prices[0], net_load[0], 1.0, 1.0)
values = jl.to_jaddle_sparse(template).values()
operating_cost = jl.make_optimal_value(template, max_epochs=200, **tolerance)


def day_cost(design, price, net_load):
    energy, power = design
    return operating_cost(
        values._replace(
            c=jnp.concatenate([jnp.zeros(2 * T), price, jnp.zeros(T)]),
            b_ineq=-net_load,
            upper_bounds=upper_bounds(energy, power, xp=jnp),
        )
    )


def total_cost(design):
    investment = energy_cost * design[0] + power_cost * design[1]
    return investment + jnp.mean(
        jax.vmap(day_cost, in_axes=(None, 0, 0))(design, prices, net_load)
    )


# %% [markdown]
# ## Optimise the Design
# Projected gradient descent with Adam and a decaying step, from a small battery.
# Every step solves all the scenarios' LPs.
cost_and_gradient = jax.jit(jax.value_and_grad(total_cost))
num_steps = 40
optimiser = optax.adam(optax.exponential_decay(2.5, num_steps, 0.01), b1=0.5)
design = jnp.array([1.0, 0.5])  # energy (kWh), power (kW)
opt_state = optimiser.init(design)
history = []
for step in range(num_steps):
    cost, gradient = cost_and_gradient(design)
    history.append((float(cost), *np.asarray(design)))
    updates, opt_state = optimiser.update(gradient, opt_state)
    design = jnp.maximum(design + updates, 0.0)
history = np.array(history)
energy, power = (float(v) for v in design)
print(f"No battery:      ${float(total_cost(jnp.zeros(2))):.4f} per day")
print(
    f"Gradient design: {energy:.2f} kWh, {power:.2f} kW, ${float(total_cost(design)):.4f} per day"
)

# %% [markdown]
# ## Check Against One Large LP
# The sizing problem is also a single LP: make the size two extra variables and
# stack every scenario's operating problem beside them. It grows with the number
# of scenarios, where the batched approach above stays one small LP solved many
# times, but here it is small enough for HiGHS to give the exact answer. The cost
# is flat near the optimum, so the two designs agree more closely in cost than in
# size.
n = 4 * T
is_power, is_soc = np.arange(n) < 2 * T, np.arange(n) >= 3 * T
bounded = is_power | is_soc
# One row per bounded variable: x[j] <= energy or x[j] <= power.
size_columns = -np.column_stack([is_soc[bounded], is_power[bounded]]).astype(float)
select_bounded = sp.identity(n, format="csr")[bounded]
per_day = sp.identity(num_days, format="csr")
no_size = sp.csr_matrix((num_days * T, 2))
day_costs = np.hstack([np.zeros((num_days, 2 * T)), prices, np.zeros((num_days, T))])
exact = linprog(
    c=np.concatenate([[energy_cost, power_cost], day_costs.ravel() / num_days]),
    A_eq=sp.hstack([no_size, sp.kron(per_day, A_eq)]),
    b_eq=np.zeros(num_days * T),
    A_ub=sp.vstack(
        [
            sp.hstack([no_size, sp.kron(per_day, A_ineq)]),
            sp.hstack(
                [
                    sp.kron(np.ones((num_days, 1)), size_columns),
                    sp.kron(per_day, select_bounded),
                ]
            ),
        ]
    ),
    b_ub=np.concatenate([-net_load.ravel(), np.zeros(num_days * bounded.sum())]),
    bounds=(0, None),
    method="highs",
)
print(
    f"Exact design:    {exact.x[0]:.2f} kWh, {exact.x[1]:.2f} kW, ${exact.fun:.4f} per day"
)

# %%
day = int(np.argmax(solar.sum(axis=1)))  # the sunniest day
schedule = jl.solve(
    operating_lp(prices[day], net_load[day], energy, power), **tolerance
)
charge, discharge, grid, soc = np.split(np.asarray(schedule["solution"].primal), 4)

figure, (left, right) = plt.subplots(1, 2, figsize=(14, 5))
left.plot(history[:, 1], label="Energy (kWh)")
left.plot(history[:, 2], label="Power (kW)")
left.axhline(exact.x[0], color="C0", linestyle="--", alpha=0.6)
left.axhline(exact.x[1], color="C1", linestyle="--", alpha=0.6)
left.set_xlabel("Gradient step")
left.set_title("Battery size (dashed: exact)")
left.legend()

right.step(hours, soc, where="post", label="State of charge (kWh)")
right.step(hours, solar[day], where="post", label="Solar (kW)")
right.step(hours, demand[day], where="post", label="Demand (kW)")
right.step(hours, grid, where="post", label="Grid import (kW)")
right.set_xlabel("Hour")
right.set_title("Operation on the sunniest day")
right.legend(loc="upper left")
price_axis = right.twinx()
price_axis.step(hours, prices[day], where="post", color="grey", linestyle=":")
price_axis.set_ylabel("Price ($/kWh, dotted)")
plt.tight_layout()
plt.show()

# %%
