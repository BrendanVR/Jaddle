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

Jaddle has been tested on challenging MIPLIB relaxations. While not a production solver, it performs competitively with PDLP‑style methods. Here we benchmark MIPLIB problems between 30 and 50mb in size using my laptop's NVIDIA GeForce RT 5070 Ti GPU. We make use of Jaddles in-built PDHG solver, with 10 rounds of Ruiz scaling followed by 1 round of PC scaling. Jaddle solve (s) is solve-only.

| Problem | Vars | Cons | Jaddle obj | Jaddle solve (s) | Converged |
|---|---:|---:|---:|---:|:---:|
| eilA101-2 | 65832 | 100 | 803.4 | 22.06 | ✅ |
| ex10 | 15895 | 62931 | 100 | 3.04 | ✅ |
| map10 | 26617 | 49894 | -602.2 | 57.75 | ✅ |
| map16715-04 | 26617 | 49894 | -296.3 | 67.80 | ✅ |
| n3div36 | 22120 | 4453 | 1.143e+05 | 22.69 | ✅ |
| neos-3555904-turama | 22233 | 67836 | -41.45 | 2.94 | ✅ |
| neos-5049753-cuanza | 242736 | 313956 | 464 | 4.54 | ✅ |
| physiciansched3-3 | 23572 | 85819 | 2.432e+06 | 69.45 | ✅ |
| sorrell3 | 1024 | 169162 | -512 | 7.83 | ✅ |
| thor50dday | 53130 | 230 | 4174 | 3.31 | ✅ |
| triptim1 | 24010 | 14593 | 22.86 | 14.46 | ✅ |
| uccase12 | 40327 | 92414 | 1.151e+04 | 6.28 | ✅ |
| uccase9 | 21361 | 32295 | 1.082e+04 | 26.05 | ✅ |


## 📦 Installation

Clone the repo and install locally:
```bash
pip install -e .
```
Jaddle has been tested to work the the WSL 2.

## 📖 Examples

Along with source code, we provide examples to start your Jaddle journey, showcasing both the linear and convex capabilities of the library.