import optax
import jax
import jax.numpy as jnp
import jaddle.jaddle_basic_types as jt


def _saddle_param_labels(params):
    if hasattr(params, "_fields"):
        return type(params)(
            primal="primal_opt",
            dual_ineq="dual_opt",
            dual_eq="dual_opt",
        )
    return {
        "primal": "primal_opt",
        "dual_ineq": "dual_opt",
        "dual_eq": "dual_opt",
    }


def _saddle_param_labels_granular(params):
    if hasattr(params, "_fields"):
        return type(params)(
            primal="primal_opt",
            dual_ineq="dual_ineq_opt",
            dual_eq="dual_eq_opt",
        )
    return {
        "primal": "primal_opt",
        "dual_ineq": "dual_ineq_opt",
        "dual_eq": "dual_eq_opt",
    }


def scale_by_inverse_metric(metric, epsilon: float = 1e-8):
    if metric is None:
        metric_array = None
        metric_tree = None
    else:
        try:
            metric_array = jnp.asarray(metric)
            metric_tree = None
        except TypeError:
            metric_array = None
            metric_tree = jax.tree_util.tree_map(lambda m: jnp.asarray(m), metric)

    def _is_array_leaf(value):
        return (
            hasattr(value, "dtype")
            and hasattr(value, "shape")
            and not isinstance(value, optax.MaskedNode)
        )

    def _scale_with_array(grad_leaf, metric_leaf):
        if not _is_array_leaf(grad_leaf):
            return grad_leaf
        return grad_leaf / (jnp.abs(metric_leaf) + epsilon)

    def _scale_with_tree(grad_leaf, metric_leaf):
        if not _is_array_leaf(grad_leaf):
            return grad_leaf
        if isinstance(metric_leaf, optax.MaskedNode):
            return grad_leaf
        return grad_leaf / (jnp.abs(metric_leaf) + epsilon)

    def init_fn(_):
        return ()

    def update_fn(updates, state, params=None):
        del params
        if metric_array is None and metric_tree is None:
            return updates, state

        if metric_array is not None:
            scaled_updates = jax.tree_util.tree_map(
                lambda g: _scale_with_array(g, metric_array), updates
            )
            return scaled_updates, state

        scaled_updates = jax.tree_util.tree_map(_scale_with_tree, updates, metric_tree)

        return scaled_updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def optimistic_adam_saddle(
    lr_primal=1e0,
    lr_dual=1e0,
):
    primal_opt = optax.inject_hyperparams(optax.optimistic_adam_v2)(
        learning_rate=lr_primal
    )
    dual_opt = optax.inject_hyperparams(optax.optimistic_adam_v2)(learning_rate=lr_dual)

    optimiser = optax.partition(
        {
            "primal_opt": primal_opt,
            "dual_opt": dual_opt,
        },
        param_labels=_saddle_param_labels,
    )

    return optimiser


def compute_static_metrics(lp: jt.LP):
    eq_abs = jnp.abs(lp.A_eq.data)
    eq_rows = lp.A_eq.indices[:, 0]
    eq_cols = lp.A_eq.indices[:, 1]

    ineq_abs = jnp.abs(lp.A_ineq.data)
    ineq_rows = lp.A_ineq.indices[:, 0]
    ineq_cols = lp.A_ineq.indices[:, 1]

    primal_metric = jnp.ones(lp.num_variables())
    primal_metric = primal_metric.at[eq_cols].add(eq_abs)
    primal_metric = primal_metric.at[ineq_cols].add(ineq_abs)

    dual_eq_metric = jnp.ones(lp.num_eq_constraints())
    dual_eq_metric = dual_eq_metric.at[eq_rows].add(eq_abs)

    dual_ineq_metric = jnp.ones(lp.num_ineq_constraints())
    dual_ineq_metric = dual_ineq_metric.at[ineq_rows].add(ineq_abs)

    return primal_metric, dual_ineq_metric, dual_eq_metric


def create_saddle_optimiser(
    primal_optimizer: optax.GradientTransformation,
    dual_optimizer: optax.GradientTransformation,
):
    optimiser = optax.partition(
        {
            "primal_opt": primal_optimizer,
            "dual_opt": dual_optimizer,
        },
        param_labels=_saddle_param_labels,
    )

    return optimiser


def create_metric_preconditioned_saddle_optimiser(
    primal_optimizer: optax.GradientTransformation,
    dual_ineq_optimizer: optax.GradientTransformation,
    dual_eq_optimizer: optax.GradientTransformation = None,
    primal_metric=None,
    dual_ineq_metric=None,
    dual_eq_metric=None,
    epsilon: float = 1e-8,
):
    if dual_eq_optimizer is None:
        dual_eq_optimizer = dual_ineq_optimizer

    if dual_eq_metric is None:
        dual_eq_metric = dual_ineq_metric

    primal_opt = optax.chain(
        scale_by_inverse_metric(primal_metric, epsilon=epsilon),
        primal_optimizer,
    )
    dual_ineq_opt = optax.chain(
        scale_by_inverse_metric(dual_ineq_metric, epsilon=epsilon),
        dual_ineq_optimizer,
    )
    dual_eq_opt = optax.chain(
        scale_by_inverse_metric(dual_eq_metric, epsilon=epsilon),
        dual_eq_optimizer,
    )

    optimiser = optax.partition(
        {
            "primal_opt": primal_opt,
            "dual_ineq_opt": dual_ineq_opt,
            "dual_eq_opt": dual_eq_opt,
        },
        param_labels=_saddle_param_labels_granular,
    )

    return optimiser


def optimistic_adam_metric_saddle(
    lr_primal=1e-3,
    lr_dual_ineq=1e-3,
    lr_dual_eq=None,
    alpha: float = 5e-2,
    nesterov=True,
    primal_metric=None,
    dual_ineq_metric=None,
    dual_eq_metric=None,
    epsilon: float = 1e-8,
):
    if lr_dual_eq is None:
        lr_dual_eq = lr_dual_ineq

    primal_optimizer = optax.inject_hyperparams(optax.optimistic_adam_v2)(
        learning_rate=lr_primal,
        alpha=alpha,
        nesterov=nesterov,
    )
    dual_ineq_optimizer = optax.inject_hyperparams(optax.optimistic_adam_v2)(
        learning_rate=lr_dual_ineq,
        alpha=alpha,
        nesterov=nesterov,
    )
    dual_eq_optimizer = optax.inject_hyperparams(optax.optimistic_adam_v2)(
        learning_rate=lr_dual_eq,
        alpha=alpha,
        nesterov=nesterov,
    )

    return create_metric_preconditioned_saddle_optimiser(
        primal_optimizer=primal_optimizer,
        dual_ineq_optimizer=dual_ineq_optimizer,
        dual_eq_optimizer=dual_eq_optimizer,
        primal_metric=primal_metric,
        dual_ineq_metric=dual_ineq_metric,
        dual_eq_metric=dual_eq_metric,
        epsilon=epsilon,
    )
