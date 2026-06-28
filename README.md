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

Jaddle has been tested on challenging MIPLIB relaxations.  While not a production solver, it performs competitively with PDLP‑style methods. 

| Problem | Vars | Cons | HiGHS-PDLP obj | HiGHS (s) | Jaddle obj | Jaddle (s) | Converged | Rel. obj gap |
|---|---:|---:|---:|---:|---:|---:|:---:|---:|
| acc-tight5 | 1339 | 3052 | 0 | 0.01 | 0 | 2.95 | ✅ | 0.00e+00 |
| adlittle | 97 | 56 | 2.255e+05 | 0.00 | 2.255e+05 | 2.56 | ✅ | 6.10e-05 |
| app1-2 | 26871 | 53467 | -265.9 | 1.27 | -264.6 | 5.35 | ✅ | 4.90e-03 |
| boeing | 384 | 440 | -335.2 | 0.04 | -335.2 | 5.15 | ✅ | 1.08e-04 |
| csched010 | 1758 | 351 | 323.9 | 0.43 | 333.1 | 19.89 | ✅ | 2.81e-02 |
| ex9 | 10404 | 40962 | 81 | 0.28 | 81 | 3.05 | ✅ | 2.60e-05 |
| lectsched-4-obj | 7901 | 14163 | 0 | 0.08 | 0 | 3.25 | ✅ | 0.00e+00 |
| lotsize | 2985 | 1920 | 3.484e+05 | 0.14 | 3.484e+05 | 3.60 | ✅ | 1.02e-04 |
| momentum1 | 5174 | 42680 | 7.285e+04 | 1.42 | 7.279e+04 | 4.64 | ✅ | 8.75e-04 |
| n9-3 | 7644 | 2364 | 7890 | 0.06 | 7890 | 3.15 | ✅ | 1.20e-05 |
| net12 | 14115 | 14021 | 17.15 | 0.44 | 17.23 | 3.32 | ✅ | 4.29e-03 |
| nug | 20448 | 19728 | 214.3 | 0.31 | 214 | 3.28 | ✅ | 1.54e-03 |
| sing2 | 31630 | 28891 | 1.691e+07 | 0.62 | 1.689e+07 | 6.01 | ✅ | 1.04e-03 |
| stp3d | 204880 | 159488 | 480.9 | 12.64 | 481.9 | 5.54 | ✅ | 2.04e-03 |

## 📦 Installation

Clone the repo and install locally:
```bash
pip install -e .
```
Jaddle has been tested to work the the WSL 2.

## 📖 Examples

Along with source code, we provide examples to start your Jaddle journey, showcasing both the linear and convex capabilities of the library.
