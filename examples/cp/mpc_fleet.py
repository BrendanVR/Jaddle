# %% [markdown]
# # Model Predictive Control for a Fleet of 1000 Systems
# Model predictive control (MPC) picks each input by solving a small quadratic
# program. The QP plans the next `T` inputs from the measured state, the first
# input is applied, and the QP is solved again at the next time step. Each QP is
# tiny: here it has 105 variables and 120 constraint rows. An interior-point or
# ADMM solver on a CPU solves one in well under a millisecond, which is hard
# to beat one problem at a time.
#
# Now control a fleet of 1000 identical systems, each from its own state. The
# QPs share their matrices, and only the measured state, which enters the
# linear term and the right-hand side, differs. `jc.make_qp_solver` scales the
# matrices once and returns a pure-JAX solve function. `jax.vmap` turns it into
# one GPU program that solves the whole fleet's QPs in lockstep, warm-started
# from the previous time step's plans.
#
# We compare it with SCS, an operator-splitting solver that, like Jaddle, is a
# first-order method and supports warm starts. Each system gets its own SCS
# solver, which factorises its matrix once. Then at each step it updates only
# the vectors (`solver.update`) and warm-starts from that system's previous
# solution.

# %%
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import jax

import time
import numpy as np
import scipy.linalg
import scipy.sparse as sp
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scs
import jaddle.jaddle_convex as jc
import jaddle.jaddle_optimisers as jo
from jaddle.jaddle_basic_types import SaddleState

jo.configure_jax("float32")

# %% [markdown]
# ## The system: oscillating masses
# This is the benchmark from Wang and Boyd, "Fast model predictive control
# using online optimization" (2010). A chain of 4 unit masses is joined by
# unit springs, with the end masses also tied to walls. 3 actuators push
# adjacent masses apart. The state is the 4 positions and 4 velocities. The
# dynamics are discretised with a zero-order hold every `dt` seconds:
#
#     x[t+1] = A x[t] + B u[t] + w[t]
#
# Here `w` is a random force disturbance on the velocities. The inputs are
# limited to `|u| <= 0.5` and the positions to `|p| <= 3`.
masses, horizon, dt = 4, 15, 0.5
nx, nu = 2 * masses, masses - 1
u_max, p_max, w_max = 0.5, 3.0, 0.5
penalty = 20.0  # on position-limit violations; see below

stiffness = 2 * np.eye(masses) - np.eye(masses, k=1) - np.eye(masses, k=-1)
actuators = np.eye(masses, nu) - np.eye(masses, nu, k=-1)
A_c = np.block(
    [[np.zeros((masses, masses)), np.eye(masses)], [-stiffness, np.zeros((masses, masses))]]
)
B_c = np.vstack([np.zeros((masses, nu)), actuators])
expm = scipy.linalg.expm(np.block([[A_c, B_c], [np.zeros((nu, nx + nu))]]) * dt)
A, B = expm[:nx, :nx], expm[:nx, nx:]

# %% [markdown]
# ## The MPC problem as a condensed QP
# Over the horizon, the predicted states are a linear function of the
# measured state `x0` and the planned inputs `u = (u[0], ..., u[T-1])`:
#
#     (x[1], ..., x[T]) = Φ x0 + Γ u
#
# Substituting this eliminates the states. The input limits are hard. The
# position limits are soft: a disturbance can push a system to where no input
# keeps it within `p_max`, and a QP with hard position limits would then be
# infeasible. Each position limit gets a slack `e >= 0`, with an exact
# penalty: when the limits can be met, a large enough `penalty` gives the
# same plan as hard limits. The MPC problem
#
#     minimise    ½ Σ_t ||x[t+1]||² + ½ Σ_t ||u[t]||² + penalty Σ_t 1ᵀe[t+1]
#     subject to  |u[t]| <= u_max,  |p[t+1]| <= p_max + e[t+1],  e >= 0
#
# becomes a QP in `z = (u, e)`:
#
#     minimise    ½ uᵀ(ΓᵀΓ + I)u + (ΓᵀΦ x0)ᵀu + penalty 1ᵀe
#     subject to  [ S Γ, −I; −S Γ, −I] z <= [p_max − S Φ x0; p_max + S Φ x0]
#                 −u_max <= u <= u_max,  e >= 0
#
# Here `S` picks the positions out of each state. The QP's matrices are the
# same for every system and every time step. Only `c` and `b_ineq` depend on
# `x0`, which is the case `jc.make_qp_solver` is for.
Phi = np.vstack([np.linalg.matrix_power(A, t + 1) for t in range(horizon)])
Gamma = np.zeros((nx * horizon, nu * horizon))
for t in range(horizon):
    for s in range(t + 1):
        Gamma[t * nx : (t + 1) * nx, s * nu : (s + 1) * nu] = (
            np.linalg.matrix_power(A, t - s) @ B
        )
positions = np.kron(np.eye(horizon), np.eye(masses, nx))  # S

n_u, n_e = nu * horizon, masses * horizon
Q = scipy.linalg.block_diag(Gamma.T @ Gamma + np.eye(n_u), np.zeros((n_e, n_e)))
G = np.block(
    [[positions @ Gamma, -np.eye(n_e)], [-positions @ Gamma, -np.eye(n_e)]]
)
lower = np.r_[np.full(n_u, -u_max), np.zeros(n_e)]
upper = np.r_[np.full(n_u, u_max), np.full(n_e, np.inf)]
print(f"QP: {Q.shape[0]} variables, {G.shape[0]} inequality rows, plus bounds")


def qp_vectors(x0):
    """`c` and `b_ineq` for a batch of measured states `x0` (rows)."""
    predicted = x0 @ Phi.T
    p = predicted @ positions.T
    c = np.concatenate(
        [predicted @ Gamma, np.full(predicted.shape[:-1] + (n_e,), penalty)], axis=-1
    )
    return c, np.concatenate([p_max - p, p_max + p], axis=-1)


def plan_cost(x0, z):
    """The MPC objective of plans `z` (rows), with the constant term."""
    u, e = z[..., :n_u], z[..., n_u:]
    predicted = x0 @ Phi.T + u @ Gamma.T
    return (
        0.5 * np.sum(predicted**2, axis=-1)
        + 0.5 * np.sum(u**2, axis=-1)
        + penalty * np.sum(e, axis=-1)
    )


def shift(v, layout, xp=np):
    """Move a plan one step forward in time: each segment `(length, block)`
    of the last axis of `v` holds `length // block` steps of `block` entries;
    drop the first step and repeat the last. `xp` is `np` or `jnp`."""
    pieces, start = [], 0
    for length, block in layout:
        segment = v[..., start : start + length]
        segment = segment.reshape(v.shape[:-1] + (length // block, block))
        segment = xp.concatenate([segment[..., 1:, :], segment[..., -1:, :]], axis=-2)
        pieces.append(segment.reshape(v.shape[:-1] + (length,)))
        start += length
    return xp.concatenate(pieces, axis=-1)


# The variables are (u, e), and the inequality rows are the upper position
# limits then the lower ones.
primal_layout = [(n_u, nu), (n_e, masses)]
row_layout = [(n_e, masses), (n_e, masses)]


# %% [markdown]
# ## The fleet
# 1000 systems start from random positions and velocities. All of them see
# the same draw of disturbances under both controllers.
fleet, steps = 1000, 30
rng = np.random.default_rng(0)


def random_fleet(n, steps):
    x0 = rng.uniform(-1, 1, (n, nx)) * np.r_[np.full(masses, 2.0), np.full(masses, 0.5)]
    w = rng.uniform(-w_max, w_max, (steps, n, nx)) * np.r_[np.zeros(masses), np.ones(masses)]
    return x0, w


x0, disturbances = random_fleet(fleet, steps)

# %% [markdown]
# ## Jaddle: the whole fleet in one batched solve
# `make_qp_solver` returns `solve_fn(c, b_eq, b_ineq, initial_solution)`, and
# `jax.vmap` batches it over the fleet. At each step we warm-start from the
# previous step's plans, shifted one step forward in time. That means dropping
# the first input and repeating the last one, and doing the same with the
# duals of the position rows. The whole closed-loop step is one jitted
# function: build the QP vectors, solve, apply the first inputs, simulate.
#
# The batch runs in lockstep until its slowest member converges. A
# warm-started QP needs only a few dozen iterations, so a short epoch
# (`iterations_per_epoch=32`, against the default 256) lets the batch stop
# soon after the last member converges. A sweep over 16, 32, 64 and 128 put
# 32 fastest here. Both solvers stop at a relative accuracy of 1e-3.
TOL = 1e-3


def make_controller(max_epochs=100):
    """One jitted closed-loop step for a fleet of any size, on the default
    device: (states, warm start, disturbances) -> (next states, next warm
    start, plans, converged, epochs)."""
    solve = jax.vmap(
        jc.make_qp_solver(
            Q,
            A_ineq=G,
            lower_bounds=lower,
            upper_bounds=upper,
            primal_feasibility_tolerance=TOL,
            dual_feasibility_tolerance=TOL,
            dual_gap_tolerance=TOL,
            iterations_per_epoch=32,
            max_epochs=max_epochs,
        )
    )
    Phi_j, Gamma_j, S_j, A_j, B_j = (jnp.asarray(m) for m in (Phi, Gamma, positions, A, B))

    @jax.jit
    def step(x, warm, w):
        # Full-precision products: the float32 profile runs matrix-matrix
        # products in TF32 (about 3 significant digits).
        with jax.default_matmul_precision("highest"):
            predicted = x @ Phi_j.T
            p = predicted @ S_j.T
            c = jnp.concatenate([predicted @ Gamma_j, jnp.full((x.shape[0], n_e), penalty)], 1)
            b_ineq = jnp.concatenate([p_max - p, p_max + p], axis=1)
            result = solve(c, b_ineq=b_ineq, initial_solution=warm)
            z = result.solution.primal
            warm = SaddleState(
                primal=shift(z, primal_layout, jnp),
                dual_ineq=shift(result.solution.dual_ineq, row_layout, jnp),
                dual_eq=result.solution.dual_eq,
            )
            x_next = x @ A_j.T + z[:, :nu] @ B_j.T + w
        return x_next, warm, z, result.converged, result.epochs

    return step


def run_jaddle(step, x0, disturbances):
    """Run the closed loop. Returns the states and plans of every step, the
    time of the first step (which compiles) and of the rest, and the
    converged count and largest epoch count over all the solves."""
    n = x0.shape[0]
    x = jnp.asarray(x0, jnp.float32)
    warm = SaddleState(
        primal=jnp.zeros((n, Q.shape[0])),
        dual_ineq=jnp.zeros((n, G.shape[0])),
        dual_eq=jnp.zeros((n, 0)),
    )
    w = jnp.asarray(disturbances, jnp.float32)
    states, plans, converged, epochs = [x], [], 0, 0
    start = time.perf_counter()
    for k in range(len(disturbances)):
        x, warm, u, ok, ep = step(x, warm, w[k])
        states.append(x)
        plans.append(u)
        converged, epochs = converged + ok.sum(), jnp.maximum(epochs, ep.max())
        if k == 0:
            jax.block_until_ready(x)
            first, start = time.perf_counter() - start, time.perf_counter()
    jax.block_until_ready(x)
    rest = time.perf_counter() - start
    return dict(
        states=np.asarray(jnp.stack(states)),
        plans=np.asarray(jnp.stack(plans)),
        first_step_seconds=first,
        step_seconds=rest / (len(disturbances) - 1),
        converged=int(converged),
        max_epochs=int(epochs),
    )


gpu = jax.devices()[0]
with jax.default_device(gpu):
    jaddle_gpu = run_jaddle(make_controller(), x0, disturbances)
print(
    f"Jaddle on {gpu.device_kind}: first step (with compile)"
    f" {jaddle_gpu['first_step_seconds']:.2f} s, then"
    f" {1e3 * jaddle_gpu['step_seconds']:.1f} ms per step"
    f" = {1e6 * jaddle_gpu['step_seconds'] / fleet:.1f} us per QP"
)
print(
    f"Converged: {jaddle_gpu['converged']} / {fleet * steps},"
    f" at most {jaddle_gpu['max_epochs']} epochs of 32 iterations"
)

# %% [markdown]
# ## SCS: one QP at a time, warm-started
# SCS solves `min ½xᵀPx + cᵀx` subject to `Ax + s = b` with `s` in a cone.
# The bounds become extra rows (`u <= u_max`, `−u <= u_max`, `−e <= 0`), and
# all the rows go in the nonnegative cone.
# Each system has its own solver object, built (and factorised) before the
# loop. One object shared by the fleet would also work, since `update` changes
# only the vectors, but SCS adapts its internal step scale during a solve and
# keeps it for the next one. A scale tuned to one system then slows the next:
# a shared solver took 4x the iterations here. Each QP updates `b` and `c` and
# warm-starts from that system's shifted previous solution. SCS runs
# single-threaded on the CPU. Its `eps_abs` and `eps_rel` are the closest
# match to Jaddle's relative tolerance.
bound_rows = np.vstack(
    [np.eye(n_u, n_u + n_e), -np.eye(n_u, n_u + n_e), -np.eye(n_e, n_u + n_e, k=n_u)]
)
bound_rhs = np.r_[np.full(2 * n_u, u_max), np.zeros(n_e)]
A_scs = sp.csc_matrix(np.vstack([G, bound_rows]))
# SCS's dual and slack vectors follow the rows: the position rows, then the
# input bounds, then e >= 0.
scs_row_layout = row_layout + [(n_u, nu), (n_u, nu), (n_e, masses)]
P_scs = sp.triu(sp.csc_matrix(Q), format="csc")


def scs_solver(c, b_ineq):
    return scs.SCS(
        dict(P=P_scs, A=A_scs, b=np.concatenate([b_ineq, bound_rhs]), c=c),
        dict(l=A_scs.shape[0]),
        eps_abs=TOL,
        eps_rel=TOL,
        verbose=False,
    )


def run_scs(x0, disturbances):
    n = x0.shape[0]
    start = time.perf_counter()
    solvers = [scs_solver(*v) for v in zip(*qp_vectors(x0))]
    setup_seconds = time.perf_counter() - start
    x, warm = x0.copy(), [None] * n
    states, plans, statuses, iterations = [x.copy()], [], {}, []
    solver_seconds = 0.0
    start = time.perf_counter()
    for w in disturbances:
        c, b_ineq = qp_vectors(x)
        z = np.empty((n, Q.shape[0]))
        for i in range(n):
            solvers[i].update(b=np.concatenate([b_ineq[i], bound_rhs]), c=c[i])
            if warm[i] is None:
                solution = solvers[i].solve()
            else:
                solution = solvers[i].solve(warm_start=True, **warm[i])
            info = solution["info"]
            statuses[info["status"]] = statuses.get(info["status"], 0) + 1
            iterations.append(info["iter"])
            solver_seconds += info["solve_time"] / 1e3
            z[i] = solution["x"]
            warm[i] = dict(
                x=shift(z[i], primal_layout),
                y=shift(solution["y"], scs_row_layout),
                s=shift(solution["s"], scs_row_layout),
            )
        x = x @ A.T + z[:, :nu] @ B.T + w
        states.append(x.copy())
        plans.append(z)
    return dict(
        states=np.stack(states),
        plans=np.stack(plans),
        setup_seconds=setup_seconds,
        seconds=time.perf_counter() - start,
        solver_seconds=solver_seconds,
        statuses=statuses,
        iterations=np.array(iterations),
    )


scs_run = run_scs(x0, disturbances)
scs_per_qp = scs_run["seconds"] / (fleet * steps)
print(
    f"SCS: {scs_run['seconds']:.1f} s for {fleet * steps} QPs"
    f" = {1e6 * scs_per_qp:.0f} us per QP"
    f" ({1e6 * scs_run['solver_seconds'] / (fleet * steps):.0f} us inside SCS)"
)
print(
    f"SCS: setup {scs_run['setup_seconds']:.2f} s for {fleet} solvers,"
    f" median {np.median(scs_run['iterations']):.0f} iterations,"
    f" statuses {scs_run['statuses']}"
)

# %% [markdown]
# ## Do the two controllers agree?
# At the first step both solvers see the same states, so their plans can be
# compared with a reference: SCS run to a tolerance of 1e-9 on the first 100
# systems. The two solvers' stopping tests differ, so the same nominal 1e-3
# need not mean the same accuracy. After the first step the two closed loops
# drift apart slightly, since each solver stops at its own point, but the
# total closed-loop cost, `Σ ||x||² + ||u||²` over the run, should agree.
reference = np.array(
    [
        scs.SCS(
            dict(P=P_scs, A=A_scs, b=np.concatenate([b, bound_rhs]), c=c),
            dict(l=A_scs.shape[0]),
            eps_abs=1e-9,
            eps_rel=1e-9,
            verbose=False,
        ).solve()["x"]
        for c, b in zip(*qp_vectors(x0[:100]))
    ]
)
reference_cost = plan_cost(x0[:100], reference)
for run, name in [(jaddle_gpu, "Jaddle"), (scs_run, "SCS")]:
    error = np.abs(plan_cost(x0[:100], run["plans"][0, :100]) - reference_cost)
    error /= reference_cost
    print(
        f"{name} first-step plan cost, relative error:"
        f" median {np.median(error):.1e}, largest {error.max():.1e}"
    )


def closed_loop_cost(run):
    u0 = run["plans"][:, :, :nu]
    return np.sum(run["states"][1:] ** 2) + np.sum(u0**2)


cost_jaddle, cost_scs = closed_loop_cost(jaddle_gpu), closed_loop_cost(scs_run)
print(
    f"Closed-loop cost: Jaddle {cost_jaddle:.1f}, SCS {cost_scs:.1f}"
    f" (relative difference {abs(cost_jaddle - cost_scs) / cost_scs:.1e})"
)
print(
    "Largest planned input:",
    f"Jaddle {np.abs(jaddle_gpu['plans'][..., :n_u]).max():.4f},"
    f" SCS {np.abs(scs_run['plans'][..., :n_u]).max():.4f} (limit {u_max})",
)
for run, name in [(jaddle_gpu, "Jaddle"), (scs_run, "SCS")]:
    excess = np.abs(run["states"][1:, :, :masses]).max(axis=(0, 2)) - p_max
    print(
        f"{name}: largest planned slack {run['plans'][..., n_u:].max():.1e},"
        f" {np.sum(excess > 1e-3)} / {fleet} systems went past p_max"
    )

# %% [markdown]
# ## How the batch size matters
# SCS's time per QP does not depend on the fleet size: it solves one QP at a
# time. Jaddle's batched solve has a fixed cost per step (kernel launches,
# and the lockstep iterations of the slowest member), which a large batch
# spreads over many QPs. Below we time the closed loop for fleets from 1 to
# 10,000 systems, on the GPU and on the CPU. The first step of each size
# compiles and is left out.
sizes = [1, 10, 100, 1000, 10000]
sweep_steps = 10
sweep_fleets = {n: random_fleet(n, sweep_steps) for n in sizes}
cpu = jax.devices("cpu")[0]
per_qp = {"GPU": [], "CPU": []}
for device_name, device in [("GPU", gpu), ("CPU", cpu)]:
    with jax.default_device(device):
        step = make_controller()
        for n in sizes:
            run = run_jaddle(step, *sweep_fleets[n])
            per_qp[device_name].append(run["step_seconds"] / n)
            print(
                f"Jaddle {device_name}, fleet {n:6d}:"
                f" {1e3 * run['step_seconds']:8.2f} ms per step,"
                f" {1e6 * run['step_seconds'] / n:8.1f} us per QP,"
                f" at most {run['max_epochs']} epochs"
            )
print(f"SCS, any fleet: {1e6 * scs_per_qp:.0f} us per QP")

# %% [markdown]
# ## A real-time budget
# Lockstep has a cost: the batch stops only when its slowest member has
# converged. In a large fleet a few systems are always in a hard spot, such as
# being pushed past a position limit so that a slack is active, and those few
# can set the time for everyone. A controller has a deadline anyway, so the
# usual remedy is a fixed iteration budget. Members that run out of budget
# return their current plan, which already respects the input limits, since
# Jaddle keeps every iterate inside the bounds.
with jax.default_device(gpu):
    budget_run = run_jaddle(make_controller(max_epochs=10), *sweep_fleets[10000])
print(
    f"Jaddle GPU, fleet 10000, at most 10 epochs:"
    f" {1e6 * budget_run['step_seconds'] / 10000:.1f} us per QP,"
    f" {budget_run['converged']} / {10000 * sweep_steps} converged"
)

# %%
fig, (ax_time, ax_path) = plt.subplots(1, 2, figsize=(14, 5))
ax_time.plot(sizes, 1e6 * np.array(per_qp["GPU"]), "o-", label="Jaddle, GPU")
ax_time.plot(sizes, 1e6 * np.array(per_qp["CPU"]), "s-", label="Jaddle, CPU")
ax_time.plot(
    10000,
    1e6 * budget_run["step_seconds"] / 10000,
    "*",
    color="C0",
    markersize=12,
    label="Jaddle, GPU, 10-epoch budget",
)
ax_time.axhline(1e6 * scs_per_qp, color="C3", linestyle="--", label="SCS, warm-started")
ax_time.set_xscale("log")
ax_time.set_yscale("log")
ax_time.set_xlabel("Fleet size (QPs per step)")
ax_time.set_ylabel("Time per QP (us)")
ax_time.set_title("Time per MPC solve")
ax_time.legend()

time_axis = dt * np.arange(steps + 1)
for run, name, style in [(jaddle_gpu, "Jaddle", "-"), (scs_run, "SCS", "--")]:
    for j in range(masses):
        ax_path.plot(time_axis, run["states"][:, 0, j], style, color=f"C{j}")
    ax_path.plot([], [], "k" + style, label=name)
ax_path.set_xlabel("Time (s)")
ax_path.set_ylabel("Mass position")
ax_path.set_title("System 0 of the fleet, closed loop")
ax_path.legend()
plt.show()

# %% [markdown]
# On the laptop GPU this was written on, the picture is:
#
# * **One system:** SCS is faster. A lone tiny QP leaves a GPU idle, and the
#   time goes on launching the solver's kernels.
# * **Tens of systems:** the batched solve catches up with SCS.
# * **A thousand systems:** each QP takes about 1/40 of SCS's time, and
#   Jaddle's plans are also closer to the reference. The CPU, batched, is
#   still about 3x faster than looping over SCS.
# * **Ten thousand systems:** lockstep shows. The rare hard member sets the
#   pace unless the solve has an epoch budget, and with one the per-QP time
#   is back near its best.
#
# This is a comparison of throughput. To control one system with a hard
# sub-millisecond deadline, use a CPU solver such as SCS or OSQP. To run many
# controllers at once, as in a fleet, a Monte-Carlo study of a controller, or
# a scenario tree, batch them.
