# %% [markdown]
# # Solve a MIPLIB LP using Jaddle Saddle Point Optimisation
# This example demonstrates how to load and solve a MIPLIB linear program (LP) using saddle point optimisation methods implemented in Jaddle.
# %%
import os

# Runtime performance profile for JAX/XLA.
# Set JADDLE_JAX_PROFILE to one of: "safe", "balanced", "max_speed".
JAX_PROFILE = os.environ.get("JADDLE_JAX_PROFILE", "max_speed").lower()


def _append_xla_flag(flag: str):
    current = os.environ.get("XLA_FLAGS", "")
    if flag not in current:
        os.environ["XLA_FLAGS"] = f"{current} {flag}".strip()


# Suppress INFO and WARNING logs from XLA/JAX
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

if JAX_PROFILE in ["balanced", "max_speed"]:
    os.environ.setdefault("JAX_ENABLE_X64", "0")
    os.environ.setdefault(
        "JAX_COMPILATION_CACHE_DIR",
        os.path.expanduser("~/.cache/jaddle_jax"),
    )
    _append_xla_flag("--xla_disable_hlo_passes=slow-operation-alarm")

if JAX_PROFILE == "max_speed":
    os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "tensorfloat32")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
    _append_xla_flag("--xla_gpu_autotune_level=4")
elif JAX_PROFILE == "balanced":
    os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "high")

print(
    "[JAX Profile] "
    f"mode={JAX_PROFILE}, "
    f"x64={os.environ.get('JAX_ENABLE_X64', 'default')}, "
    f"matmul_precision={os.environ.get('JAX_DEFAULT_MATMUL_PRECISION', 'default')}, "
    f"cache={os.environ.get('JAX_COMPILATION_CACHE_DIR', 'disabled')}, "
    f"xla_flags='{os.environ.get('XLA_FLAGS', '')}'"
)

import jax
import jax.numpy as jnp
import jaddle.jaddle_linear as jl
import jaddle.jaddle_optimisers as jo
import jaddle.highs_helpers as hh
import highspy as hspy
import optax

# %%
PROBLEM_NAME = "ns1758913"  # name of the MIPLIB problem to load

# %% [markdown]
# ## Load and presolve the LP
# We load a MIPLIB LP from an MPS file using the `highspy` library.
highs = hspy.Highs()
highs.readModel(
    f"/home/brendanvr/python/Jaddle/data/{PROBLEM_NAME}.mps"
)  # path to MPS file

# %% [markdown]
# We convert the LP to Jaddle's sparse format.
highs_lp = highs.getLp()
jaddle_lp = jl.to_jaddle_sparse(hh.highs_to_standard_form_sparse(highs_lp))

# %%
learning_rates = [
    optax.exponential_decay(
        init_value=init_value,
        transition_steps=int(1e4),
        decay_rate=decay_rate,
        end_value=end_value,
    )
    for decay_rate in [0.9, 0.99]
    for init_value in [1e0]
    for end_value in [1e-7]
]
alphas = [0.05]
betas = [1.0]

HEDGE_ETA = 0.01
LOSS_CLIP = 20.0
EXPLORATION_RATE = 0.0
PRUNE_THRESHOLD = 5e-3
PRUNE_MIN_KEEP = 1
PRUNE_MIN_STEP = int(8e4)

primal_experts = [
    optax.optimistic_adam_v2(learning_rate=lr, alpha=alpha, beta=beta)
    for lr in learning_rates
    for alpha in alphas
    for beta in betas
]

dual_experts = [
    optax.optimistic_adam_v2(learning_rate=lr, alpha=alpha, beta=beta)
    for lr in learning_rates
    for alpha in alphas
    for beta in betas
]

ensemble_optimiser = jo.hedge_ensemble_saddle(
    lp=jaddle_lp,
    primal_experts=primal_experts,
    dual_experts=dual_experts,
    primal_eta=HEDGE_ETA,
    dual_eta=HEDGE_ETA,
    loss_clip=LOSS_CLIP,
    exploration_rate=EXPLORATION_RATE,
    center_losses=True,
)
is_converged = False
solution = jaddle_lp.initial_solution()
opt_state = None

while (len(primal_experts) > 1 or len(dual_experts) > 1) and (not is_converged):
    solution, is_converged, opt_state = jl.solve(
        lp=jaddle_lp,
        optimiser=ensemble_optimiser,
        initial_opt_state=opt_state,
        initial_solution=solution,
        verbose=True,
        expert_diagnostics=True,
        iterations_per_epoch=int(1e4),
        output_opt_state=True,
        max_epochs=10,
        weight_function=lambda i: jax.lax.select(i <= int(5e4), 0.5, 1.0),
        scale="ruiz+pc",
        update_mode="alternating",
    )

    if is_converged:
        break

    opt_state, primal_idx, dual_idx = opt_state.prune(
        threshold=PRUNE_THRESHOLD,
        min_keep=PRUNE_MIN_KEEP,
        min_step=PRUNE_MIN_STEP,
    )
    primal_experts = [primal_experts[i] for i in primal_idx]
    dual_experts = [dual_experts[i] for i in dual_idx]

    print(
        f"Pruned to {len(primal_experts)} primal experts and {len(dual_experts)} dual experts."
    )

    ensemble_optimiser = jo.hedge_ensemble_saddle(
        lp=jaddle_lp,
        primal_experts=primal_experts,
        dual_experts=dual_experts,
        primal_eta=HEDGE_ETA,
        dual_eta=HEDGE_ETA,
        loss_clip=LOSS_CLIP,
        exploration_rate=EXPLORATION_RATE,
        center_losses=True,
    )

if not is_converged:
    solution, is_converged = jl.solve(
        lp=jaddle_lp,
        optimiser=ensemble_optimiser,
        initial_solution=solution,
        initial_opt_state=opt_state,
        verbose=True,
        iterations_per_epoch=int(1e4),
        weight_function=lambda i: jax.lax.select(i <= int(5e4), 1e-8, 1.0),
        scale="ruiz+pc",
    )

# %%
print(f"Primal Equality Residual: {jaddle_lp.eq_slack(solution.primal)}")
print(f"Primal Inequality Residual: {jaddle_lp.ineq_slack(solution.primal)}")
print("----------------------------------------------")

# %%
