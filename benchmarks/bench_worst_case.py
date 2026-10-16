"""Worst case for the exact search: just under max_unique unique values, many periods.

Also measures parallel scaling over several features.
Usage: python benchmarks/bench_worst_case.py [n_rows] [n_periods] [n_features] [n_jobs]
"""

import resource
import sys
import time
import warnings

import numpy as np
import pandas as pd

from tide import TIDE

n = int(sys.argv[1]) if len(sys.argv) > 1 else 5_000_000
n_per = int(sys.argv[2]) if len(sys.argv) > 2 else 24
n_feat = int(sys.argv[3]) if len(sys.argv) > 3 else 4
n_jobs = int(sys.argv[4]) if len(sys.argv) > 4 else 2

rng = np.random.default_rng(0)
per = rng.integers(0, n_per, n)
cols, z = {}, rng.normal(size=n)
y = (rng.random(n) < 1 / (1 + np.exp(-(-2.5 + 0.8 * z)))).astype(np.int8)
for i in range(n_feat):
    v = z + rng.normal(scale=0.7, size=n)
    ranks = np.argsort(np.argsort(v))
    cols[f"f{i}"] = (ranks * 990_000 // n).astype(np.float64)  # ~990k unique values
X = pd.DataFrame(cols)
print(f"{n:,} rows, {n_per} periods, {n_feat} features, unique/feature={X['f0'].nunique():,}")

for jobs in (1, n_jobs):
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        t = TIDE(n_jobs=jobs).fit(X, y, per)
    dt = time.perf_counter() - t0
    print(
        f"n_jobs={jobs}: {dt:.1f}s total, {dt / n_feat:.2f}s per feature, "
        f"bins={t.summary_['n_bins'].tolist()}"
    )
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"peak RSS of main process: {peak:.0f} MB")
