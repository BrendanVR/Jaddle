# %% [markdown]
# # Fairness-Constrained Logistic Regression with Jaddle
# A lender trains a classifier on its past decisions. Those decisions were
# biased against one group of applicants, and a model fitted to them learns
# the bias, even when the group attribute is left out of the features. This
# example adds a fairness constraint to logistic regression (Zafar et al.,
# "Fairness Constraints: Mechanisms for Fair Classification", AISTATS 2017),
# traces the whole fairness/accuracy trade-off in one batched solve, and
# reads off the price of fairness from the constraint's dual.
#
# The objective is a log-likelihood, not a quadratic, so this is a job for a
# general `JaddleCP` rather than `jc.quadratic_program`.

# %%
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import jaddle.jaddle_convex as jc
import jaddle.jaddle_optimisers as jo

jo.configure_jax("float32")

# %% [markdown]
# ## A biased world
# Each applicant belongs to group A or group B (30% of applicants), and has
# three skill features drawn from the same distribution in both groups. Their
# merit is a weighted sum of the skills, and in a fair world they would repay
# (`y_fair = 1`) when merit plus noise is positive. The historical labels `y`
# we train on instead knock a fixed amount off every group-B applicant.
#
# The group attribute is not a feature, but two proxy features (think
# postcode) are shifted for group B, so the model can still tell the groups
# apart. Leaving the sensitive attribute out of the features is not enough.
rng = np.random.default_rng(0)
skill_weights = np.array([1.0, 0.7, 0.5])
bias = 1.0


def sample(m):
    group_b = rng.random(m) < 0.3
    skills = rng.standard_normal((m, 3))
    proxies = group_b[:, None] + 0.7 * rng.standard_normal((m, 2))
    noise = 0.5 * rng.standard_normal(m)
    merit = skills @ skill_weights + noise
    X = np.hstack([skills, proxies])
    return X, (merit - bias * group_b > 0), (merit > 0), group_b


X, y, y_fair, group_b = sample(4000)
X_test, y_test, y_fair_test, group_b_test = sample(20000)
num_features = X.shape[1]
print(
    f"Positive rate in the training labels: A {y[~group_b].mean():.2f}, "
    f"B {y[group_b].mean():.2f}"
)


# %% [markdown]
# ## The convex program
# The model scores an applicant as `x·θ + b` and approves when the score is
# positive. Logistic regression minimises the mean log-loss, plus a small ridge
# penalty that keeps the problem strictly convex:
#
#     minimise    mean(log(1 + exp(−ỹ (Xθ + b)))) + λ ||θ||²      (ỹ = ±1)
#
# A natural fairness target is demographic parity: equal approval rates in
# both groups. Approval rates are step functions of `θ`, so Zafar et al.
# constrain a convex stand-in, the covariance between group membership `s`
# and the score:
#
#     |mean((s − s̄)(Xθ + b))| ≤ c
#
# For a binary `s` this is `s̄(1 − s̄)` times the gap between the groups'
# mean scores. It is linear in `(θ, b)`, so it becomes two inequality
# constraints, `cov − c ≤ 0` and `−cov − c ≤ 0`. `c = ∞` gives plain logistic
# regression and `c = 0` equal mean scores.
#
# We write the problem as a function of `c`, so that the next section can
# solve it for many values of `c` at once.
X_jax = jnp.array(X)
y_signed = jnp.where(jnp.array(y), 1.0, -1.0)
s_centred = jnp.array(group_b, dtype=jnp.float32) - group_b.mean()
ridge = 1e-3


def build_cp(c):
    def scores(params):
        theta, intercept = params[:-1], params[-1]
        return jnp.matmul(X_jax, theta, precision="highest") + intercept

    def objective(params):
        log_loss = jnp.mean(jax.nn.softplus(-y_signed * scores(params)))
        return log_loss + ridge * jnp.sum(params[:-1] ** 2)

    def fairness(params):
        covariance = jnp.mean(s_centred * scores(params))
        return jnp.stack([covariance - c, -covariance - c])

    return jc.JaddleCP(
        num_variables=num_features + 1,
        objective=objective,
        constraints_eq=lambda params: jnp.zeros(0),
        constraints_ineq=fairness,
        lower_bounds=jnp.full(num_features + 1, -jnp.inf),
        upper_bounds=jnp.full(num_features + 1, jnp.inf),
    )


settings = dict(
    primal_grad_norm_tolerance=1e-5,
    primal_feasibility_tolerance=1e-5,
    complementarity_slack_tolerance=1e-5,
)


# %% [markdown]
# ## How a model performs
# For each model we report its accuracy against the biased labels it was
# trained on, its accuracy against the fair labels (which we only know because
# we made the data), and the demographic parity gap: the difference between
# the groups' approval rates. All three are measured on the test set.
def evaluate(params):
    params = np.asarray(params)
    approve = X_test @ params[:-1] + params[-1] > 0
    return dict(
        accuracy=np.mean(approve == y_test),
        fair_accuracy=np.mean(approve == y_fair_test),
        gap=approve[~group_b_test].mean() - approve[group_b_test].mean(),
    )


def report(name, params):
    m = evaluate(params)
    print(
        f"{name:28s} accuracy {m['accuracy']:.3f}  "
        f"fair accuracy {m['fair_accuracy']:.3f}  gap {m['gap']:+.3f}"
    )


# %% [markdown]
# ## Unconstrained and fully fair models
# First plain logistic regression (`c = ∞`), then the fully fair model
# (`c = 0`).
unconstrained = jc.solve(build_cp(jnp.inf), **settings)
fair = jc.solve(build_cp(0.0), **settings)
print("Converged:", unconstrained["converged"], fair["converged"])

theta_unconstrained = np.asarray(unconstrained["solution"].primal)
theta_fair = np.asarray(fair["solution"].primal)
report("Unconstrained (c = ∞)", theta_unconstrained)
report("Fair (c = 0)", theta_fair)
print("Weights on the proxies, unconstrained:", np.round(theta_unconstrained[3:5], 3))
print("Weights on the proxies, fair:         ", np.round(theta_fair[3:5], 3))

# %% [markdown]
# The unconstrained model puts negative weight on the proxies and approves
# group B much less often, reproducing the bias in its training data. The fair
# model closes most of the gap. It agrees less with the biased labels, but
# more with the fair ones, because here the bias is in the labels, not in the
# applicants.

# %% [markdown]
# ## The trade-off curve in one batched solve
# `jc.make_solver(build_cp)` turns the problem into a pure JAX function of
# `c`, which we `vmap` over 40 thresholds between 0 and the unconstrained
# model's covariance.
#
# Each solution also carries the duals `λ₊, λ₋` of the two fairness
# constraints, and `|λ₊ − λ₋|` is the price of fairness: by how much the
# optimal training loss rises per unit tightening of `c`. Usually only one
# constraint is active and this is just its dual. At `c = 0` both are, and
# then only the difference of the duals is determined.
covariance_unconstrained = abs(
    float(
        jnp.mean(
            s_centred * (X_jax @ theta_unconstrained[:-1] + theta_unconstrained[-1])
        )
    )
)
thresholds = jnp.linspace(0.0, covariance_unconstrained, 40)
solve_curve = jax.jit(jax.vmap(jc.make_solver(build_cp, **settings)))
curve = solve_curve(thresholds)
print("Converged:", int(curve.converged.sum()), "/", len(thresholds))

metrics = [evaluate(params) for params in np.asarray(curve.solution.primal)]
gaps = np.array([m["gap"] for m in metrics])
accuracy = np.array([m["accuracy"] for m in metrics])
fair_accuracy = np.array([m["fair_accuracy"] for m in metrics])
duals = np.asarray(curve.solution.dual_ineq)
price = np.abs(duals[:, 0] - duals[:, 1])

# %%
fig, (ax_tradeoff, ax_price) = plt.subplots(1, 2, figsize=(14, 5))
ax_tradeoff.plot(gaps, accuracy, "o-", color="C0", label="Against the biased labels")
ax_tradeoff.plot(gaps, fair_accuracy, "o-", color="C2", label="Against the fair labels")
ax_tradeoff.set_xlabel("Demographic parity gap (approval rate A − B)")
ax_tradeoff.set_ylabel("Test accuracy")
ax_tradeoff.set_title("Fairness/accuracy trade-off")
ax_tradeoff.legend()

ax_price.plot(thresholds, price, "o-", color="C3")
ax_price.set_xlabel("Covariance bound c")
ax_price.set_ylabel("Training loss per unit of c")
ax_price.set_title("Price of fairness")
plt.show()

# %% [markdown]
# Moving left along the curve (fairer models), accuracy against the biased
# labels falls, while accuracy against the fair labels rises. The price of
# fairness grows steadily as `c` tightens, so each further step towards
# parity costs more training loss than the last. It is zero at the
# unconstrained model's covariance, where the constraint stops binding.

# %%
