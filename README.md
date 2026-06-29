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

Jaddle has been tested on challenging MIPLIB relaxations. While not a production solver, it performs competitively with PDLP‑style methods. Here we benchmark MIPLIB problems between 30 and 50mb in size using my laptop's NVIDIA GeForce RT 5070 Ti GPU. We make use of Jaddles in-built PDHG solver, with 10 rounds of Ruiz scaling followed by 1 round of PC scaling. Jaddle time is solve-only. Crucially it includes XLA compile time, roughly 0.8 seconds of overhead. cuPDLP-c also failed to converge in 2 minutes on proteindesign121hz512p9 and radiationm40-10-02.

| Problem | Vars | Cons | Jaddle obj | Jaddle solve (s) | Converged |
|---|---:|---:|---:|---:|:---:|
| buildingenergy | 145237 | 267853  | 3.325e+04 | 59.86 | ✅  |
| eilA101-2 | 65832 | 101  | 803.2 | 16.46 | ✅  |
| ex10 | 15895 | 62931  | 100 | 3.17 | ✅  |
| map10 | 26617 | 49895  | -602.2 | 49.26 | ✅  |
| map16715-04 | 26617 | 49895  | -296.3 | 58.51 | ✅  |
| n3div36 | 22120 | 4454  | 1.144e+05 | 19.57 | ✅  |
| neos-3555904-turama | 22233 | 67836  | -41.45 | 3.38 | ✅  |
| neos-5049753-cuanza | 242736 | 313956  | 464 | 4.63 | ✅  |
| neos-848589 | 550539 | 1484  | 0 | 96.23 | ✅  |
| netdiversion | 129174 | 99787  | 230.8 | 13.93 | ✅  |
| physiciansched3-3 | 23572 | 85819  | 2.427e+06 | 63.52 | ✅ |
| proteindesign121hz512p9 | 159067 | 224  | 0 | 79.56 | ❌  |
| radiationm40-10-02 | 42613 | 44087  | 0 | 42.42 | ❌  |
| rd-rplusc-21 | 543 | 54182  | 100 | 14.02 | ✅  |
| sorrell3 | 1024 | 169163  | -512 | 8.13 | ✅  |
| thor50dday | 53130 | 231  | 4174 | 3.37 | ✅  |
| triptim1 | 24010 | 14593  | 22.86 | 10.62 | ✅  |
| uccase12 | 40327 | 92414  | 1.151e+04 | 6.25 | ✅  |
| uccase9 | 21361 | 32295  | 1.082e+04 | 9.66 | ✅  |

## 📦 Installation

Clone the repo and install locally:
```bash
pip install -e .
```
Jaddle has been tested to work the the WSL 2.

## 📖 Examples

Along with source code, we provide examples to start your Jaddle journey, showcasing both the linear and convex capabilities of the library.