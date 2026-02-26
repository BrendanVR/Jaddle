<img width="1536" height="1024" alt="jaddle_logo_full" src="https://github.com/user-attachments/assets/d4e61518-3557-4273-b4d0-0d89510e0f3f" />

# Jaddle: The JAX Saddle Solver
*Saddle up with JAX to solve large scale linear and convex programs*

## 🚀 Introduction

Jaddle isn’t here to replace your industrial solver. It’s here to show that primal–dual optimization can be elegant, lightweight, and fun. Built in ~1000 lines of JAX.

Jaddle performs admirably on many hard linear and convex benchmarks, while remaining simple enough for rapid experimentation. Utilizing the basic building blocks that Optax provides, one can specify many complicated variants of primal–dual optimizers all with only 10–15 lines of code.


## ✨ Why Jaddle?

- 🪶 Minimalist: (~1000 lines of code)
- 🧩 Modular: built on JAX + Optax primitives
- 🤸‍♀️ Flexible: swap optimizers in ~10 lines
- 🚀 Portable: CPU / GPU / TPU

## 🧪 Example Optimizer

With Optax partitioning, even complicated primal–dual variants collapse into a few lines:

```python
import optax

optimiser = jo.create_saddle_optimiser(
    optax.optimistic_adam_v2(learning_rate=1e-3, alpha=0.05),
    dual_optimizer=optax.adadelta(learning_rate=1.0),
)

```

You can also add static optimizer-side metric preconditioning without changing the LP matrices:

```python
import jax.numpy as jnp

num_vars = jaddle_lp.num_variables()
num_eq = jaddle_lp.num_eq_constraints()
num_ineq = jaddle_lp.num_ineq_constraints()

optimiser = jo.optimistic_adam_metric_saddle(
    lr_primal=1e-2,
    lr_dual_ineq=1e-2,
    lr_dual_eq=1e-2,
    primal_metric=jnp.ones(num_vars),
    dual_ineq_metric=jnp.ones(num_ineq),
    dual_eq_metric=jnp.ones(num_eq),
)
```

---

## 📊 Benchmarks

Jaddle has been tested on challenging MIPLIB relaxations.  While not a production solver, it performs competitively with PDLP‑style methods. Here we compare **Jaddle** against **cuPDLP-C** with the appropriate scaling. We also make use of HiGHs' pre-solve functionality. We quote the size of the presolved system and not of the original problem. We also only quote solve time, and not time taken to scale the system (which is done in the same fashion by Jaddle and cuPDLP-C). Running with the above adamdelta_saddle optimiser on a GPU we have the following benchmark results:

| Instance    | Variables | Constraints| cuPDLP-C Runtime (s) | Jaddle Runtime (s) | 
|-------------|------| ------|-----------|--------|
| `nug`         | 20446   |   18268  |     1      |   3 |
| `stp3d`       | 136924  |  97457    |    35      | 9 |        
| `ns1758913`   |  17684  |  26760 |    39       | 11 |
| `buildingenergy`   |  154978   |  277594  |    142       | 92 |

The point is not to beat cuPDLP‑C outright, but to show that ~1000 lines of JAX can hang in the same ballpark.

## 📦 Installation

Clone the repo and install locally:
```bash
pip install -e .
```
Jaddle has been tested to work the the WSL 2.

## 📖 Examples

Along with source code, we provide examples to start your Jaddle journey, showcasing both the linear and convex capabilities of the library.
