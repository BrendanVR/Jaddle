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

```

This modular code produces a family of optimizers for each primal, dual learning rate and each input extra gradient parameter alpha. JAX allows you to succinctly specify what the optimiser is doing.

---

## 📊 Benchmarks

Jaddle has been tested on challenging MIPLIB relaxations.  While not a production solver, it performs competitively with PDLP‑style methods. Here we compare **Jaddle** against **cuPDLP-C** with the appropriate scaling. We also make use of HiGHs' pre-solve functionality. We quote the size of the presolved system and not of the original problem. We also only quote solve time, and not time taken to scale the system (which is done in the same fashion by Jaddle and cuPDLP-C). Running with the above adamdelta_saddle optimiser on a GPU we have the following benchmark results:

| Instance    | Variables | Constraints| cuPDLP-C Runtime (s) | Jaddle Runtime (s) | 
|-------------|------| ------|-----------|--------|
| `nug`         | 20446   |   18268  |     1      |   3 |
| `stp3d`       | 136924  |  97457    |    35      | 9 |        
| `ns1758913`   |  17684  |  26760 |    39       | 11 |
| `buildingenergy`   |  154978   |  277594  |    142       | 92 |

The point is not to beat cuPDLP‑C outright, but to show that ~1000 lines of JAX can hang in the same ballpark. The hope is that by utilizing increased algorithmic smarts, Jaddle can reduce the oscillatory nature of PDLP.

## 📦 Installation

Clone the repo and install locally:
```bash
pip install -e .
```
Jaddle has been tested to work the the WSL 2.

## 📖 Examples

Along with source code, we provide four examples to start your Jaddle journey, showcasing both the linear and convex capabilities of the library.
