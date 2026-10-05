# %% [markdown]
# # Linear Regression with Random Fourier Features
# This notebook demonstrates how to perform linear regression using Random Fourier Features (RFF) to approximate a non-linear function. We will use Jaddle's convex optimization framework to solve the regression problem with L2 regularization.
import optax
import numpy as np
import jax.numpy as jnp
import jaddle.jaddle_convex as jc
import jaddle.jaddle_optimisers as jo
from sklearn.kernel_approximation import RBFSampler
import matplotlib.pyplot as plt

# %%
# Set random seed for reproducibility
np.random.seed(42)

# %%
# Generate synthetic data
n_samples = 200
X = np.linspace(0, 2 * np.pi, n_samples)
y = 3 * np.sin(X) + 2 * np.cos(5 * X) + np.random.normal(0, 0.5, n_samples)

# %%
# Apply Random Fourier Features
gamma = 10.0
rbf_sampler = RBFSampler(gamma=gamma, n_components=50, random_state=42)
X_transformed = rbf_sampler.fit_transform(X.reshape(-1, 1))

# Convert to JAX arrays
X_jax = jnp.array(X_transformed, dtype=jnp.float32)
y_jax = jnp.array(y, dtype=jnp.float32)


# %%
# Define linear regression model with L2 regularization
def objective(w):
    losses = jnp.square(X_jax @ w - y_jax)
    return jnp.mean(losses)


def constraints_ineq(w):
    return jnp.array([jnp.dot(w, w) - 200.0])  # L2 norm constraint (||w||^2 <= 100)


def constraints_eq(w):
    return jnp.array([])  # No equality constraints in this example


# %%
# Define the convex problem
cp = jc.JaddleCP(
    num_variables=X_jax.shape[1],
    objective=objective,
    constraints_eq=constraints_eq,
    constraints_ineq=constraints_ineq,
    lower_bounds=-jnp.inf * jnp.ones(X_jax.shape[1]),
    upper_bounds=jnp.inf * jnp.ones(X_jax.shape[1]),
)

# %%
# Solve the problem using Jaddle Convex SPS optimizer
solution = jc.solve(cp)["solution"]

l2_norm = jnp.linalg.norm(solution.primal, ord=2)
print("L2 norm of solution.primal:", l2_norm)

# %%
y_pred = jnp.dot(X_jax, solution.primal)

# Plot predictions vs actual values
plt.figure(figsize=(10, 6))
plt.plot(X, y_jax, "o", label="Actual", alpha=0.6)
plt.plot(X, y_pred, "-", label="Predicted", linewidth=2, color="red")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression with RBF Features")
plt.grid(True, alpha=0.3)
plt.show()

# %%

# %%
