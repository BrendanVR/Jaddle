# %% [markdown]
# Hybrid primal LR: per-step Optax schedule + epoch-level controller

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import jax
import jax.numpy as jnp
import optax
import jaddle.jaddle_linear as jl
import jaddle.jaddle_optimisers as jo
import jaddle.highs_helpers as hh
import highspy as hspy


def optimiser_builder(lr_state):
    """Build a saddle optimiser using lr_state parameters.

    lr_state keys used:
      - base_lr: float
      - decay_steps: int
      - decay_rate: float
    """
    base_lr = lr_state.get("base_lr", 1.0)
    decay_steps = int(lr_state.get("decay_steps", 1000))
    decay_rate = float(lr_state.get("decay_rate", 0.99))
    # Use a shifted schedule so when we rebuild the optimiser each epoch
    # the schedule continues from the last epoch's step rather than restarting.
    step_offset = int(lr_state.get("step_offset", 0))

    base_schedule = optax.exponential_decay(
        init_value=base_lr,
        transition_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=False,
    )

    def shifted_schedule(step):
        return base_schedule(step + step_offset)

    schedule = shifted_schedule

    primal_opt = optax.optimistic_adam_v2(schedule, alpha=0.05)
    dual_opt = optax.adadelta(learning_rate=1.0)

    return jo.create_saddle_optimiser(primal_opt, dual_optimizer=dual_opt)


def lr_controller(diagnostics, count, lr_state):
    """Epoch-level controller that reduces base_lr when primal progress stalls.

    Policy:
      - Keep track of best `primal_grad_norm` seen.
      - If `primal_grad_norm` does not improve by `min_rel_improve`
        for `patience` epochs, reduce `base_lr *= decay_factor` and reset.
    Returns: (new_lr_state, reset_opt_state_flag)
    """
    # Pull diagnostics (they may be numpy/JAX scalars)
    primal_grad_norm = float(diagnostics.get("primal_grad_norm", jnp.inf))

    best = lr_state.get("best_primal_grad_norm", float("inf"))
    stalled = lr_state.get("stalled_epochs", 0)
    patience = int(lr_state.get("patience", 3))
    decay_factor = float(lr_state.get("decay_factor", 0.5))
    min_rel_improve = float(lr_state.get("min_rel_improve", 0.01))

    improved = primal_grad_norm < best * (1.0 - min_rel_improve)

    if improved:
        lr_state["best_primal_grad_norm"] = primal_grad_norm
        lr_state["stalled_epochs"] = 0
    else:
        # not improved
        stalled += 1
        lr_state["stalled_epochs"] = stalled

        if stalled >= patience:
            # reduce learning rate
            old_lr = lr_state.get("base_lr", 1.0)
            new_lr = float(old_lr) * decay_factor
            lr_state["base_lr"] = new_lr
            lr_state["stalled_epochs"] = 0
            lr_state["best_primal_grad_norm"] = primal_grad_norm
            print(
                f"[lr_controller] Stalled {stalled} epochs — reducing base_lr {old_lr} -> {new_lr}"
            )
            # increment step offset for the epoch we just finished
            iterations_per_epoch = int(lr_state.get("iterations_per_epoch", 1000))
            lr_state["step_offset"] = (
                int(lr_state.get("step_offset", 0)) + iterations_per_epoch
            )
            return lr_state, True

    # increment step offset for the epoch we just finished (no LR change)
    iterations_per_epoch = int(lr_state.get("iterations_per_epoch", 1000))
    lr_state["step_offset"] = int(lr_state.get("step_offset", 0)) + iterations_per_epoch
    return lr_state


def main():
    highs = hspy.Highs()
    highs.readModel("/home/brendanvr/python/Jaddle/data/nug.mps")

    highs_lp = highs.getLp()
    jaddle_lp = jl.to_jaddle_sparse(hh.highs_to_standard_form_sparse(highs_lp))

    initial_lr_state = {
        "base_lr": 1e-1,
        "decay_steps": 5000,
        "decay_rate": 0.9,
        "patience": 2,
        "decay_factor": 0.9,
        "min_rel_improve": 1e-3,
        "best_primal_grad_norm": float("inf"),
        "stalled_epochs": 0,
        "epoch": 0,
        "step_offset": 0,
        "iterations_per_epoch": 5000,
    }

    iterations_per_epoch = int(initial_lr_state["iterations_per_epoch"])

    solution = jl.solve(
        lp=jaddle_lp,
        optimiser_builder=optimiser_builder,
        lr_controller=lr_controller,
        lr_state=initial_lr_state,
        reset_opt_state_on_lr_change=True,
        verbose=True,
        iterations_per_epoch=iterations_per_epoch,
        max_epochs=100,
        weight_function=lambda i: jax.lax.select(i < int(5e4), 1e-16, 1.0),
    )

    print("--------------------------------")
    print("Saddle point solver objective:", jaddle_lp.objective(solution.primal))
    print("Inequality violation:", jaddle_lp.ineq_slack(solution.primal))
    print("Equality violation:", jaddle_lp.eq_slack(solution.primal))
    print("--------------------------------")


if __name__ == "__main__":
    main()

# %%
