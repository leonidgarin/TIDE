"""Fit-time benchmark per feature type.

Usage: python benchmarks/bench_fit.py [n_rows] [n_periods]
"""

import sys
import time
import warnings

import numpy as np
import pandas as pd

from tide import TIDE


def make(n, n_per, seed=0):
    rng = np.random.default_rng(seed)
    per = rng.integers(0, n_per, n)
    z = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-2.5 + 0.8 * z)))).astype(np.int8)
    noise = rng.normal(scale=0.5, size=n)
    cols = {
        "float_all_unique": z + noise,  # ~n unique -> undersampled (> max_unique)
        "float_900k_unique": np.round(z + noise, 3) + rng.integers(0, 90, n) * 10,
        "float_100k_unique": np.round(z + noise, 2) * 7,
        "int_100_unique": np.clip(np.round((z + noise) * 15 + 50), 0, 99),
    }
    mixed = np.round((z + noise) * 10, 1).astype(object)
    r = rng.random(n)
    mixed[r < 0.05] = np.nan
    mixed[(r >= 0.05) & (r < 0.08)] = "no_hit"
    mixed[(r >= 0.08) & (r < 0.10)] = -999
    cols["mixed_object"] = mixed
    return pd.DataFrame(cols), y, per


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5_000_000
    n_per = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    t0 = time.perf_counter()
    X, y, per = make(n, n_per)
    print(
        f"data: {n:,} rows x {X.shape[1]} features, {n_per} periods "
        f"({time.perf_counter() - t0:.1f}s to generate)"
    )
    for col in X.columns:
        print(f"  {col:20s} unique={X[col].nunique():>10,}", end="", flush=True)
        t = TIDE(cat_values=[-999])
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            t.fit(X[[col]], y, per)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        t.transform(X[[col]])
        tr_s = time.perf_counter() - t0
        s = t.summary_.iloc[0]
        print(
            f"  fit={fit_s:6.2f}s  transform={tr_s:5.2f}s  bins={s['n_bins']:>3}"
            f"  undersampled={s['undersampled']}"
        )


if __name__ == "__main__":
    main()
