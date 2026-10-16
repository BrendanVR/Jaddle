# %% [markdown]
# # Sizing a Battery with Differentiable Linear Programs
# How big a battery should a site with solar panels buy? The answer depends on how
# the battery would be operated, and operating it well is itself an optimisation
# problem: one linear program per day. This example solves a batch of those LPs at
# once with `jax.vmap`, differentiates their optimal costs with respect to the
# battery's size, and runs gradient descent on the size.
#
# The battery's price has economies of scale, so the sizing problem is not itself
# a linear program. That is what makes the decomposition worthwhile: with a price
# proportional to size, everything below would collapse into one LP, and solving
# that LP directly would be far quicker.

# %%
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=2"
os.environ["JAX_BCOO_CUSPARSE_LOWERING"] = "true"


import jax
import jax.numpy as jnp
import numpy as np
import optax
import scipy.sparse as sp
import matplotlib.pyplot as plt
import jaddle.jaddle_linear as jl
import jaddle.jaddle_optimisers as jo

jo.configure_jax("float32")

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
# The battery's daily cost grows as the 0.8 power of its storage and of its power
# rating, so doubling either costs about 1.74 times as much. The total daily cost
# of a design is that plus the average optimal operating cost over the scenarios.
#
# `jl.make_optimal_value` returns an LP's optimal value as a differentiable
# function of its numbers. Every day's LP has the same sparsity pattern, so one
# function serves them all: `jax.vmap` solves the whole batch together, and
# `jax.grad` returns the gradient with respect to the design. No derivative is
# taken through the solver's iterations. The gradient of an LP's optimal value
# with respect to a bound is the reduced cost at that bound, which the solver
# has already computed.
energy_cost, power_cost = 0.17, 0.05  # $/day for 1 kWh and for 1 kW
scale_exponent = 0.8
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
    investment = (
        energy_cost * design[0] ** scale_exponent
        + power_cost * design[1] ** scale_exponent
    )
    return investment + jnp.mean(
        jax.vmap(day_cost, in_axes=(None, 0, 0))(design, prices, net_load)
    )


# ## Optimise the Design
# Projected gradient descent with Adam and a decaying step. Every step solves all
# the scenarios' LPs. The size is kept slightly above zero, where the investment
# cost's slope is unbounded.
cost_and_gradient = jax.jit(jax.value_and_grad(total_cost))
num_steps = 40
smallest_size = 1e-3


def descend(start):
    optimiser = optax.adam(optax.exponential_decay(2.5, num_steps, 0.01), b1=0.5)
    design = jnp.array(start)
    opt_state = optimiser.init(design)
    history = []
    for step in range(num_steps):
        cost, gradient = cost_and_gradient(design)
        history.append((float(cost), *np.asarray(design)))
        updates, opt_state = optimiser.update(gradient, opt_state)
        design = jnp.maximum(design + updates, smallest_size)
    return design, float(total_cost(design)), np.array(history)


design, cost, history = descend([1.0, 0.5])  # energy (kWh), power (kW)
energy, power = (float(v) for v in design)
print(f"No battery:        ${float(total_cost(jnp.zeros(2))):.4f} per day")
print(f"From a small one:  {energy:.2f} kWh, {power:.2f} kW, ${cost:.4f} per day")

# %% [markdown]
# ## Check From a Second Start
# Economies of scale make the investment cost concave, so the sizing problem is
# not convex and gradient descent only promises a local optimum. There is no
# exact solver to compare against, so the check is a second run from an oversized
# battery, which should arrive at the same design. The cost is flat near the
# optimum, so the two runs agree more closely in cost than in size.
large_design, large_cost, large_history = descend([30.0, 8.0])
print(
    f"From a large one:  {float(large_design[0]):.2f} kWh, "
    f"{float(large_design[1]):.2f} kW, ${large_cost:.4f} per day"
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
left.plot(large_history[:, 1], color="C0", linestyle="--", alpha=0.6)
left.plot(large_history[:, 2], color="C1", linestyle="--", alpha=0.6)
left.set_xlabel("Gradient step")
left.set_title("Battery size (dashed: from a large battery)")
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
