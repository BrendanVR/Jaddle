# %% [markdown]
# # Robust Kalman filtering with CVXPY and Jaddle
# A vehicle moves in the plane with linear dynamics driven by a random force,
# and its position is measured with noise:
#
#     x[t+1] = A x[t] + B w[t],      y[t] = C x[t] + v[t].
#
# The state `x[t]` holds position and velocity, `w[t]` is the unknown force
# and `v[t]` the measurement noise. Smoothing estimates the whole trajectory
# from the measurements `y` by choosing the forces and noise that best explain
# them. With Gaussian noise that is least squares, the Kalman smoother:
#
#     minimize  Σ ‖w[t]‖² + τ Σ ‖v[t]‖²    subject to the dynamics.
#
# Here a fifth of the measurements are outliers, with 20 times the usual
# noise. Least squares lets them pull the estimate off course. The robust
# filter replaces `‖v[t]‖²` with the Huber function of `‖v[t]‖`, which grows
# only linearly for large residuals:
#
#     minimize  Σ ‖w[t]‖² + τ Σ huber(‖v[t]‖, ρ)    subject to the dynamics.
#
# This is CVXPY's "robust Kalman filtering" example. The least-squares
# filter is a QP, which `jcv.solve` hands to `jcv.JaddleSolver` (CVXPY's QP
# form, solved by `jc.quadratic_program`). The robust one is not: CVXPY
# would turn `huber(norm(v))` into second-order cones, which Jaddle does not
# support. `jcv.solve` therefore compiles it into JAX functions for `jc.solve`
# instead, where the Huber of the norm stays a smooth function. Both are
# checked against Clarabel, the interior-point solver CVXPY ships with.

# %%
import os
import time

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import optax

import jaddle.jaddle_optimisers as jo

jo.configure_jax("float64")

import jaddle.jaddle_cvxpy as jcv  # noqa: E402

# %% [markdown]
# ## The system
# 1000 time steps over 50 seconds. The state is `(position, velocity)` in two
# dimensions, with light damping `γ` on the velocity. The force is standard
# Gaussian; each measurement is an outlier with probability `p`, and then its
# noise has standard deviation `σ` instead of 1.

# %%
n, T = 1000, 50
ts, dt = np.linspace(0, T, n, endpoint=True, retstep=True)
gamma = 0.05

A = np.zeros((4, 4))
B = np.zeros((4, 2))
C = np.zeros((2, 4))
A[0, 0] = A[1, 1] = 1
A[0, 2] = A[1, 3] = (1 - gamma * dt / 2) * dt
A[2, 2] = A[3, 3] = 1 - gamma * dt
B[0, 0] = B[1, 1] = dt**2 / 2
B[2, 0] = B[3, 1] = dt
C[0, 0] = C[1, 1] = 1

sigma, p = 20, 0.20
rng = np.random.default_rng(6)
w_true = rng.standard_normal((2, n))
v_true = rng.standard_normal((2, n))
outliers = rng.random(n) <= p
v_true[:, outliers] = sigma * rng.standard_normal((2, outliers.sum()))

x_true = np.zeros((4, n + 1))
y = np.zeros((2, n))
for t in range(n):
    y[:, t] = C @ x_true[:, t] + v_true[:, t]
    x_true[:, t + 1] = A @ x_true[:, t] + B @ w_true[:, t]

# %% [markdown]
# ## The two filters
# CVXPY's page adds the dynamics one time step at a time. Written over whole
# slices of the variables, the model is 2 constraints instead of 2000 and
# CVXPY builds it much faster.


# %%
def kalman(robust, tau, rho=None):
    x = cp.Variable((4, n + 1))
    w = cp.Variable((2, n))
    v = cp.Variable((2, n))
    if robust:
        noise = cp.sum(cp.huber(cp.norm(v, 2, axis=0), rho))
    else:
        noise = cp.sum_squares(v)
    constraints = [x[:, 1:] == A @ x[:, :-1] + B @ w, y == C @ x[:, :-1] + v]
    return cp.Problem(cp.Minimize(cp.sum_squares(w) + tau * noise), constraints), x


def rms_error(x):
    return np.sqrt(np.mean(np.sum((x[:2] - x_true[:2]) ** 2, axis=0)))


estimates = {}
for name, robust, tau, rho in [
    ("least squares", False, 0.08, None),
    ("robust", True, 2.0, 2.0),
]:
    problem, x = kalman(robust, tau, rho)
    start = time.perf_counter()
    problem.solve()
    clarabel_time = time.perf_counter() - start
    reference, reference_x = problem.value, x.value.copy()

    start = time.perf_counter()
    jcv.solve(problem, tol=1e-6)
    jaddle_time = time.perf_counter() - start
    estimates[name] = x.value
    print(
        f"{name}: {problem.status}, objective {problem.value:.4f} "
        f"(Clarabel {reference:.4f}), max |x - x_Clarabel| "
        f"{np.max(np.abs(x.value - reference_x)):.1e}, "
        f"{problem.solver_stats.num_iters} epochs, RMS position error "
        f"{rms_error(x.value):.3f}"
    )
    print(
        f"  runtime: Clarabel {clarabel_time:.3f}s, "
        f"Jaddle {jaddle_time:.3f}s (includes JAX compilation)"
    )

# %% [markdown]
# ## The estimated trajectories
# The measurements are grey, with the outliers far from the path. The
# least-squares estimate follows them; the robust one stays on the true
# trajectory.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
for ax, (name, x) in zip(axes, estimates.items()):
    ax.scatter(y[0], y[1], s=4, color="0.7", label="measurements")
    ax.plot(x_true[0], x_true[1], color="black", lw=1.5, label="true trajectory")
    ax.plot(x[0], x[1], color="tab:red", lw=1.2, label="estimate")
    ax.set_title(f"{name} (RMS position error {rms_error(x):.2f})")
# Zoomed to the true path; many outliers fall outside the frame.
lo, hi = x_true[:2].min(axis=1), x_true[:2].max(axis=1)
pad = 0.15 * (hi - lo)
axes[0].set_xlim(lo[0] - pad[0], hi[0] + pad[0])
axes[0].set_ylim(lo[1] - pad[1], hi[1] + pad[1])
axes[0].legend(loc="upper right")
fig.tight_layout()
plt.show()

# %%
