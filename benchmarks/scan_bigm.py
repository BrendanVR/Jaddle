# %% [markdown]
# # Big-M cost scanner
#
# Scans every `.mps` file in a directory and flags instances whose **objective**
# has a big-M / penalty structure: a wide dynamic range between the largest and
# the median nonzero cost coefficient, plus a large absolute coefficient.
#
# This is the structure that defeats Jaddle's saddle-point LP solver (e.g.
# `glass4`, `binkar10_1`): the optimum is dominated by a handful of ~1e6 penalty
# costs, so a constraint residual that is tiny relative to the RHS norm still
# leaves an enormous *objective* gap. The solver parks at a primal/dual-feasible
# point with a frozen relative duality gap, far above the true optimum.
#
# We deliberately key on the COST vector only. Large *constraint-matrix*
# coefficients are a much noisier signal -- they are often just units/scaling
# that Ruiz scaling handles -- so they are reported for context but do not drive
# the flag.
#
# Usage:
#     python benchmarks/scan_bigm.py
#     python benchmarks/scan_bigm.py --data-dir data --ratio 1e3 --abs 1e4

# %%
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import glob

import numpy as np
import highspy as hspy

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")

# Default big-M COST thresholds: flag when the objective spans a wide dynamic
# range (max/median over nonzero costs) AND has a large absolute coefficient,
# so a wide ratio over tiny costs is not flagged. Shared with benchmark.py's
# --skip-bigm so both use one definition.
DEFAULT_RATIO = 1e3
DEFAULT_ABS = 1e4


def parse_args():
    p = argparse.ArgumentParser(
        description="Flag MPS instances with a big-M / penalty cost structure."
    )
    p.add_argument("--data-dir", default=DATA_DIR, help="Directory of .mps files.")
    p.add_argument(
        "--ratio",
        type=float,
        default=DEFAULT_RATIO,
        help="Flag when max|c| / median|c| over nonzero costs exceeds this "
        f"(default {DEFAULT_RATIO:g}).",
    )
    p.add_argument(
        "--abs",
        type=float,
        default=DEFAULT_ABS,
        help="Require max|c| to also exceed this absolute value, so a wide ratio "
        f"over tiny costs is not flagged (default {DEFAULT_ABS:g}).",
    )
    return p.parse_args()


def cost_stats(path):
    """Return (max|c|, median|c|, ratio, max|A|) for the original (un-relaxed)
    model at ``path``. Stats are over NONZERO coefficients."""
    highs = hspy.Highs()
    highs.setOptionValue("output_flag", "false")
    highs.readModel(path)
    lp = highs.getLp()

    c = np.abs(np.asarray(lp.col_cost_))
    c = c[c != 0]
    cost_max = float(c.max()) if c.size else 0.0
    cost_med = float(np.median(c)) if c.size else 0.0
    ratio = (cost_max / cost_med) if cost_med > 0 else 0.0

    a = np.abs(np.asarray(lp.a_matrix_.value_))
    a = a[a != 0]
    a_max = float(a.max()) if a.size else 0.0

    return cost_max, cost_med, ratio, a_max


def is_bigm_cost(path, ratio=DEFAULT_RATIO, abs_thresh=DEFAULT_ABS):
    """True if the instance at ``path`` has a big-M / penalty COST structure:
    max|c|/median|c| > ``ratio`` AND max|c| > ``abs_thresh``."""
    cost_max, _cost_med, cost_ratio, _a_max = cost_stats(path)
    return cost_ratio > ratio and cost_max > abs_thresh


def main():
    args = parse_args()
    paths = sorted(glob.glob(os.path.join(args.data_dir, "*.mps")))
    if not paths:
        print(f"No .mps files in {args.data_dir}.")
        return

    print(
        f"Scanning {len(paths)} instance(s) in {args.data_dir} "
        f"(flag: max|c|/median|c| > {args.ratio:g} AND max|c| > {args.abs:g})\n"
    )

    flagged = []
    errors = []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            cost_max, cost_med, ratio, a_max = cost_stats(path)
        except Exception as exc:
            errors.append((name, repr(exc)))
            continue
        if ratio > args.ratio and cost_max > args.abs:
            flagged.append((name, cost_max, cost_med, ratio, a_max))

    flagged.sort(key=lambda r: r[3], reverse=True)

    print(f"{'instance':28s} {'max|c|':>10s} {'med|c|':>10s} {'c-ratio':>10s} {'max|A|':>10s}")
    print("-" * 72)
    for name, cmax, cmed, ratio, amax in flagged:
        print(f"{name:28s} {cmax:10.2e} {cmed:10.2e} {ratio:10.2e} {amax:10.2e}")

    print(f"\n=== {len(flagged)} COST big-M instance(s) ===")
    print("  " + "  ".join(name for name, *_ in flagged))

    if errors:
        print(f"\n{len(errors)} file(s) failed to read:")
        for name, err in errors:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()

# %%
