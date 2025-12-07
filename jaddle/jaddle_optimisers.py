import optax


def adamdelta_saddle(
    lr_primal=1e-3,
    lr_dual=1.0,
    alpha: float = 5e-2,
    nesterov=True,
):
    optimiser = optax.partition(
        {
            "primal_opt": optax.optimistic_adam_v2(
                learning_rate=lr_primal,
                alpha=alpha,
                nesterov=nesterov,
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


def adam2max_saddle(
    lr_primal=1e-3,
    lr_dual=1e-3,
    alpha: float = 5e-2,
    nesterov=True,
):
    optimiser = optax.partition(
        {
            "primal_opt": optax.optimistic_adam_v2(
                learning_rate=lr_primal,
                alpha=alpha,
                nesterov=nesterov,
            ),
            "dual_opt": optax.adamax(
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


def adamgrad_saddle(
    lr_primal=1e-3,
    lr_dual=1.0,
    alpha: float = 5e-2,
    nesterov=True,
):
    optimiser = optax.partition(
        {
            "primal_opt": optax.optimistic_adam_v2(
                learning_rate=lr_primal,
                alpha=alpha,
                nesterov=nesterov,
            ),
            "dual_opt": optax.adagrad(
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


def sgd_saddle(
    lr_primal=1e-3,
    lr_dual=1e-3,
    momentum_primal=0.1,
    momentum_dual=0.1,
    nesterov=True,
):
    optimiser = optax.partition(
        {
            "primal_opt": optax.sgd(
                learning_rate=lr_primal,
                momentum=momentum_primal,
                nesterov=nesterov,
            ),
            "dual_opt": optax.sgd(
                learning_rate=lr_dual,
                momentum=momentum_dual,
                nesterov=nesterov,
            ),
        },
        param_labels={
            "primal": "primal_opt",
            "dual_ineq": "dual_opt",
            "dual_eq": "dual_opt",
        },
    )

    return optimiser


def optimistic_sgd_saddle(
    lr_primal=1e-3,
    lr_dual=1e-3,
):
    optimiser = optax.partition(
        {
            "primal_opt": optax.optimistic_gradient_descent(learning_rate=lr_primal),
            "dual_opt": optax.optimistic_gradient_descent(learning_rate=lr_dual),
        },
        param_labels={
            "primal": "primal_opt",
            "dual_ineq": "dual_opt",
            "dual_eq": "dual_opt",
        },
    )

    return optimiser
