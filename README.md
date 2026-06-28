<img width="1536" height="1024" alt="jaddle_logo_full" src="https://github.com/user-attachments/assets/d4e61518-3557-4273-b4d0-0d89510e0f3f" />

# Jaddle: The JAX Saddle Solver
*Saddle up with JAX to solve large scale linear and convex programs*

## 🚀 Introduction

Jaddle isn’t here to replace your industrial solver. It’s here to show that primal–dual optimization can be elegant, lightweight, and fun. Built in ~4000 lines of JAX.

Jaddle performs admirably on many hard linear and convex benchmarks, while remaining simple enough for rapid experimentation. Utilizing the basic building blocks that Optax provides, one can specify many complicated variants of primal–dual optimizers all with only 5-10 lines of code.


## ✨ Why Jaddle?

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

Jaddle has been tested on challenging MIPLIB relaxations.  While not a production solver, it performs competitively with PDLP‑style methods. Benchmark using my laptops NVIDIA GeForce RT 5070 Ti GPU. Optimum is HiGHS's exact solver (simplex/IPM) solved to optimality, used as a ground-truth objective oracle. Jaddle time is solve-only (iterate loop incl. first-epoch XLA compile, excl. setup/scaling). Crucially it includes XLA compile time, roughly 0.8 seconds of overhead.

| Problem | Vars | Cons | Optimum | Jaddle obj | Jaddle solve (s) | Converged | Rel. gap to opt |
|---|---:|---:|---:|---:|---:|:---:|---:|
| acc-tight5 | 1339 | 3052 | 0 | 0 | 1.05 | ✅ | 0.00e+00 |
| adlittle | 97 | 56 | 2.255e+05 | 2.255e+05 | 1.01 | ✅ | 1.82e-04 |
| app1-2 | 26871 | 53467 | -264.6 | -264.6 | 4.15 | ✅ | 2.83e-05 |
| boeing | 384 | 440 | -335.2 | -335.2 | 4.32 | ✅ | 1.23e-04 |
| csched010 | 1758 | 351 | 332.4 | 332.9 | 18.63 | ✅ | 1.54e-03 |
| ex9 | 10404 | 40962 | 81 | 81 | 1.07 | ✅ | 6.93e-16 |
| lectsched-4-obj | 7901 | 14163 | 0 | 0 | 1.13 | ✅ | 0.00e+00 |
| lotsize | 2985 | 1920 | 3.484e+05 | 3.485e+05 | 1.22 | ✅ | 1.92e-04 |
| momentum1 | 5174 | 42680 | 7.279e+04 | 7.279e+04 | 3.22 | ✅ | 6.94e-05 |
| n9-3 | 7644 | 2364 | 7890 | 7890 | 1.28 | ✅ | 2.18e-08 |
| net12 | 14115 | 14021 | 17.25 | 17.23 | 1.33 | ✅ | 1.23e-03 |
| nug | 20448 | 19728 | 214 | 214 | 1.14 | ✅ | 7.10e-07 |
| sing2 | 31630 | 28891 | 1.715e+07 | 1.689e+07 | 3.86 | ✅ | 1.49e-02 |

## 📦 Installation

Clone the repo and install locally:
```bash
pip install -e .
```
Jaddle has been tested to work the the WSL 2.

## 📖 Examples

Along with source code, we provide examples to start your Jaddle journey, showcasing both the linear and convex capabilities of the library.
