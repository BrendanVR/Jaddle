import optax


def adamdelta_saddle(
    lr_primal=None,
    lr_dual=None,
    alpha: float = 5e-2,
):

    if lr_primal is None:
        lr_primal = optax.cosine_decay_schedule(1e0, 5e4, alpha=1e-3, exponent=10.0)
    if lr_dual is None:
        lr_dual = 1.0
    optimiser = optax.partition(
        {
            "primal_opt": optax.optimistic_adam_v2(
                learning_rate=lr_primal,
                alpha=alpha,
            ),
            "dual_opt": optax.adadelta(
                learning_rate=lr_dual,
            ),
        },
        param_labels={
            "primal": "primal_opt",
            "dual_ineq": "dual_opt",
            "dual_eq": "dual_opt",
        },
    )

    return optimiser
