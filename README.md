<img width="1536" height="1024" alt="jaddle_logo_full" src="https://github.com/user-attachments/assets/d4e61518-3557-4273-b4d0-0d89510e0f3f" />

# Jaddle: The JAX Saddle Solver
*Saddle up with JAX to solve large scale linear and convex programs*

## 🚀 Introduction

Jaddle isn’t here to replace your industrial solver. It’s here to show that primal–dual optimization can be elegant, lightweight, and fun. Built in ~3000 lines of JAX.

Jaddle performs admirably on many hard linear and convex benchmarks, while remaining simple enough for rapid experimentation. Utilizing the basic building blocks that Optax provides, one can specify many complicated variants of primal–dual optimizers all with only 5-10 lines of code.


## ✨ Why Jaddle?

- 🪶 Minimalist: (~3000 lines of code)
- 🧩 Modular: built on JAX + Optax primitives
- 🤸‍♀️ Flexible: swap optimizers in ~10 lines
- 🚀 Portable: CPU / GPU / TPU

## 🧪 Example Optimizer

With Optax partitioning, even complicated primal–dual variants collapse into a few lines:

```python
import optax

optimiser = jo.create_saddle_optimiser(
    optax.optimistic_adam_v2(learning_rate=1e-3, alpha=0.05),
    optax.adadelta(learning_rate=1.0),
)
```

## 📊 Benchmarks

Jaddle has been tested on challenging MIPLIB relaxations.  While not a production solver, it performs competitively with PDLP‑style methods. 

## 📦 Installation

Clone the repo and install locally:
```bash
pip install -e .
```
Jaddle has been tested to work the the WSL 2.

## 📖 Examples

Along with source code, we provide examples to start your Jaddle journey, showcasing both the linear and convex capabilities of the library.
