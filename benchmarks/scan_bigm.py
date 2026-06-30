# %% [markdown]
# # Big-M scanner (cost and matrix)
#
# Scans every `.mps` file in a directory and flags instances with a big-M /
# penalty structure. Two distinct structures, two distinct failure modes:
#
# 1. **Cost big-M** (`is_bigm_cost`): the *objective* spans a wide dynamic range
#    between the largest and the median nonzero cost coefficient, plus a large
#    absolute coefficient. This defeats Jaddle's saddle-point LP solver (e.g.
#    `glass4`, `binkar10_1`): the optimum is dominated by a handful of ~1e6
#    penalty costs, so a constraint residual that is tiny relative to the RHS
#    norm still leaves an enormous *objective* gap. The solver parks at a
#    primal/dual-feasible point with a frozen relative duality gap, far above
#    the true optimum.
#
# 2. **Matrix big-M** (`is_bigm_matrix`): the *constraint matrix* has rows where
#    one coefficient dwarfs its rowmates (e.g. `sp150x300d`: 300 rows of the
#    form [1, 1, ..., 1, 3050]). This is the constraint-side mirror of cost
#    big-M and stalls the saddle solver on a primal-feasibility plateau: Ruiz
#    scaling equilibrates whole rows/columns but cannot flatten the *within-row*
#    anisotropy, so the step geometry stays wildly skewed and a residual that is
#    tiny relative to ||b|| can still leave a binary far from {0,1}.
#
# A raw `max|A|` threshold is a much noisier signal -- a single large entry is
# often just units/scaling that Ruiz handles. So the matrix flag keys on the
# *per-row* spread (max/min within a row) being large across a *high fraction*
# of rows: a pervasive, systematic pattern that scaling cannot absorb, not a
# lone outlier. The two flags are independent; an instance may trip either,
# both, or neither, and `benchmark.py` exposes them as separate skip flags.
#
# Usage:
#     python benchmarks/scan_bigm.py
#     python benchmarks/scan_bigm.py --data-dir data --ratio 1e3 --abs 1e4
#     python benchmarks/scan_bigm.py --row-spread 1e3 --row-frac 0.5

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
# range (max/median over nonzero costs) AND the penalty is significant -- either
# in absolute terms (max|c| large, e.g. glass4's 1e6) OR relative to a tiny bulk
# of costs (median|c| below `DEFAULT_MED_FLOOR`, e.g. mas74/mas76 where max|c|=1
# but median|c|=1e-5, so cost-1 columns dominate a sea of 1e-5 costs). The
# absolute gate alone misses the tiny-median family, which slips under any
# reasonable abs threshold yet has the same penalty structure that stalls the
# saddle solver. Shared with benchmark.py's --skip-bigm so both use one
# definition.
DEFAULT_RATIO = 1e3
DEFAULT_ABS = 1e3
DEFAULT_MED_FLOOR = 1e-3

# Default big-M MATRIX thresholds: flag when the per-row coefficient spread
# (max|A_row| / min|A_row| over nonzero entries) has a large MEDIAN across rows
# AND a large FRACTION of rows are themselves wide. Keying on the median (not
# the max) row spread demands the wide-coefficient pattern be PERVASIVE rather
# than a single outlier row -- a lone skewed row is usually benign units that
# Ruiz scaling absorbs, whereas sp150x300d's 300 identical [1,...,1,3050] rows
# are a systematic big-M coupling that survives equilibration. Calibrated on
# data/: this pair flags sp150x300d (median spread 3050, 67% of rows wide) and
# neos-3754480-nidda (1280, 74%) along with ~19 other known-hard instances,
# without firing on benign mild-scaling models. Shared with benchmark.py's
# --skip-bigm-matrix so both use one definition.
DEFAULT_ROW_SPREAD = 1e3
DEFAULT_ROW_FRAC = 0.5
DEFAULT_WIDE_ROW = 1e2


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
        help="With a wide ratio, flag when max|c| exceeds this absolute value "
        f"(default {DEFAULT_ABS:g}). A wide ratio over uniformly tiny costs is "
        "not flagged by this path alone -- see --med-floor for the "
        "relative-penalty (tiny-median) path.",
    )
    p.add_argument(
        "--med-floor",
        type=float,
        default=DEFAULT_MED_FLOOR,
        help="With a wide ratio, also flag when median|c| is below this value "
        f"(default {DEFAULT_MED_FLOOR:g}), catching the relative-penalty family "
        "(e.g. mas74/mas76: max|c|=1, median|c|=1e-5) that slips under --abs.",
    )
    p.add_argument(
        "--row-spread",
        type=float,
        default=DEFAULT_ROW_SPREAD,
        help="Matrix big-M: flag when the MEDIAN per-row coefficient spread "
        "(max|A_row|/min|A_row|) over rows exceeds this "
        f"(default {DEFAULT_ROW_SPREAD:g}).",
    )
    p.add_argument(
        "--row-frac",
        type=float,
        default=DEFAULT_ROW_FRAC,
        help="Matrix big-M: with a wide median row spread, also require this "
        f"fraction of rows to be wide (default {DEFAULT_ROW_FRAC:g}), i.e. the "
        "skew is pervasive rather than a single outlier row.",
    )
    p.add_argument(
        "--wide-row",
        type=float,
        default=DEFAULT_WIDE_ROW,
        help="Matrix big-M: a row counts as 'wide' for --row-frac when its "
        f"spread exceeds this (default {DEFAULT_WIDE_ROW:g}).",
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


def is_bigm_cost(
    path, ratio=DEFAULT_RATIO, abs_thresh=DEFAULT_ABS, med_floor=DEFAULT_MED_FLOOR
):
    """True if the instance at ``path`` has a big-M / penalty COST structure:
    a wide dynamic range (max|c|/median|c| > ``ratio``) together with a
    significant penalty -- either max|c| > ``abs_thresh`` (absolutely large,
    e.g. glass4) OR median|c| < ``med_floor`` (the penalty dwarfs a tiny bulk of
    costs, e.g. mas74/mas76)."""
    cost_max, cost_med, cost_ratio, _a_max = cost_stats(path)
    return cost_ratio > ratio and (cost_max > abs_thresh or cost_med < med_floor)


def matrix_stats(path, wide_row=DEFAULT_WIDE_ROW):
    """Return (median_row_spread, max_row_spread, wide_row_frac) for the model at
    ``path``. Row spread is max|A_row| / min|A_row| over a row's NONZERO entries;
    only rows with at least two nonzeros contribute (a single-entry row has no
    spread). ``wide_row_frac`` is the fraction of contributing rows whose spread
    exceeds ``wide_row``."""
    highs = hspy.Highs()
    highs.setOptionValue("output_flag", "false")
    highs.readModel(path)
    lp = highs.getLp()
    nr = lp.num_row_

    am = lp.a_matrix_
    val = np.abs(np.asarray(am.value_))
    # a_matrix_ is column-wise (CSC): index_ holds the ROW of each nonzero, so we
    # can reduce per-row max/min/count by scattering over the row indices without
    # materialising the matrix or looping in Python.
    row = np.asarray(am.index_)
    nz = val > 0
    row, val = row[nz], val[nz]

    row_max = np.zeros(nr)
    row_min = np.full(nr, np.inf)
    row_cnt = np.zeros(nr, dtype=np.int64)
    np.maximum.at(row_max, row, val)
    np.minimum.at(row_min, row, val)
    np.add.at(row_cnt, row, 1)

    has_spread = (row_cnt >= 2) & (row_min > 0)
    spreads = row_max[has_spread] / row_min[has_spread]
    if spreads.size == 0:
        return 1.0, 1.0, 0.0
    return (
        float(np.median(spreads)),
        float(spreads.max()),
        float(np.mean(spreads > wide_row)),
    )


def is_bigm_matrix(
    path,
    row_spread=DEFAULT_ROW_SPREAD,
    row_frac=DEFAULT_ROW_FRAC,
    wide_row=DEFAULT_WIDE_ROW,
):
    """True if the instance at ``path`` has a big-M / penalty MATRIX structure:
    the median per-row coefficient spread exceeds ``row_spread`` AND at least
    ``row_frac`` of rows are individually wide (spread > ``wide_row``). The
    median + fraction gate demands a pervasive within-row skew (e.g.
    sp150x300d's [1,...,1,3050] rows) rather than a single outlier row that Ruiz
    scaling would absorb."""
    med_spread, _max_spread, wide_frac = matrix_stats(path, wide_row=wide_row)
    return med_spread > row_spread and wide_frac >= row_frac


def main():
    args = parse_args()
    paths = sorted(glob.glob(os.path.join(args.data_dir, "*.mps")))
    if not paths:
        print(f"No .mps files in {args.data_dir}.")
        return

    print(
        f"Scanning {len(paths)} instance(s) in {args.data_dir}\n"
        f"  COST flag:   max|c|/median|c| > {args.ratio:g} AND "
        f"[max|c| > {args.abs:g} OR median|c| < {args.med_floor:g}]\n"
        f"  MATRIX flag: median row spread > {args.row_spread:g} AND "
        f">={args.row_frac:g} of rows wide (spread > {args.wide_row:g})\n"
    )

    cost_flagged = []
    matrix_flagged = []
    errors = []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            cost_max, cost_med, ratio, a_max = cost_stats(path)
            med_spread, max_spread, wide_frac = matrix_stats(path, args.wide_row)
        except Exception as exc:
            errors.append((name, repr(exc)))
            continue
        if ratio > args.ratio and (cost_max > args.abs or cost_med < args.med_floor):
            cost_flagged.append((name, cost_max, cost_med, ratio, a_max))
        if med_spread > args.row_spread and wide_frac >= args.row_frac:
            matrix_flagged.append((name, med_spread, max_spread, wide_frac))

    cost_flagged.sort(key=lambda r: r[3], reverse=True)
    matrix_flagged.sort(key=lambda r: r[1], reverse=True)

    print(
        f"{'instance':28s} {'max|c|':>10s} {'med|c|':>10s} {'c-ratio':>10s} {'max|A|':>10s}"
    )
    print("-" * 72)
    for name, cmax, cmed, ratio, amax in cost_flagged:
        print(f"{name:28s} {cmax:10.2e} {cmed:10.2e} {ratio:10.2e} {amax:10.2e}")
    print(f"\n=== {len(cost_flagged)} COST big-M instance(s) ===")
    print("  " + "  ".join(name for name, *_ in cost_flagged))

    print(
        f"\n{'instance':28s} {'med-spread':>11s} {'max-spread':>11s} {'wide-frac':>10s}"
    )
    print("-" * 72)
    for name, med_spread, max_spread, wide_frac in matrix_flagged:
        print(f"{name:28s} {med_spread:11.2e} {max_spread:11.2e} {wide_frac:10.2f}")
    print(f"\n=== {len(matrix_flagged)} MATRIX big-M instance(s) ===")
    print("  " + "  ".join(name for name, *_ in matrix_flagged))

    if errors:
        print(f"\n{len(errors)} file(s) failed to read:")
        for name, err in errors:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()

# %%
