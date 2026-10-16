# TIDE — Time Invariant Discretization Engine

[![CI](https://github.com/leonidgarin/TIDE/actions/workflows/ci.yml/badge.svg)](https://github.com/leonidgarin/TIDE/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tide-binning.svg)](https://pypi.org/project/tide-binning/)
[![Python](https://img.shields.io/pypi/pyversions/tide-binning.svg)](https://pypi.org/project/tide-binning/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

TIDE bins features for credit scorecards so that the bins are **stable over time**.
In every period of the training data, as hard constraints:

- the event rate is **monotonic** over the bins, in the **same direction** in all periods,
  so WoE / event-rate curves of different bins never cross;
- adjacent bins differ in event rate by at least `min_er_delta`;
- every bin holds at least `min_sample_rate` of the period's rows and at least
  `min_class_obs` events and non-events.

It works on **mixed-type** columns, e.g. `age = [18, 44, np.nan, "special_category", -999]`,
and encodes them with Weight of Evidence. It is a scikit-learn transformer.

## Installation

```bash
pip install tide-binning
```

Requires Python ≥ 3.10. All dependencies (numpy, pandas, scipy, scikit-learn, matplotlib,
joblib, tqdm) ship prebuilt wheels for Windows, macOS and Linux.

## Quick start

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tide import TIDE

X = pd.DataFrame(...)          # features, may be mixed-type
y = pd.Series(...)             # binary target, 0/1
periods = pd.Series(...)       # e.g. application month: str, datetime, Period, int, ...

binner = TIDE(cat_values=[-999])          # -999 is a category, not a number
X_woe = binner.fit_transform(X, y, periods=periods)

lr = LogisticRegression().fit(X_woe, y)
print(roc_auc_score(y, lr.predict_proba(X_woe)[:, 1]))
```

## Results

```python
binner.summary_       # one row per feature: trend, n_bins, IV, max PSI, warnings
binner.stats_         # per feature and bin: n_obs, event rate, WoE, IV
binner.stats_per_     # per period, feature and bin: n_obs, event rate, PSI
binner.bins_          # fitted bins (format below)
binner.woes_          # {feature: {bin_idx: WoE}}
binner.plot(["age"])  # bin sizes and event rates, event rate over time, PSI over time
```

## Bins format

```python
{"trend": "pos",
 "bins": {0: {(-np.inf, 25.0): "cont", "special_category": "cat"},
          1: {(25.0, 40.0): "cont"},
          2: {(40.0, np.inf): "cont"},
          3: {None: "missing"}}}       # None = NaN / None / pd.NA / NaT
```

Intervals are `(a, b]` for a positive trend and `[a, b)` for a negative trend, so a value
that is unseen in training and lies between two borders always goes to the bin with the
**higher** event rate.

## Handling special values

| You have | Use |
|---|---|
| NaN, None, pd.NA, NaT | missing automatically |
| strings, bools | categories automatically |
| numeric codes such as `-999` that mean "no data" | `missing_values=[-999]` |
| numeric codes that are a category of their own | `cat_values=[-999]` |
| different codes per feature | `{"age": [-1], "income": [-999]}` |

`missing_strategy` (`"separate_bin"`, `"worst_bin"`, `"best_bin"`) and `cat_strategy`
(`"separate_bin"`, `"chi-squared"`, `"tide"`) control how they are binned. With
`cat_strategy="tide"`, all bins — continuous, categorical and missing — are merged until the
stability constraints hold for all of them.

## Your own bins

```python
binner = TIDE(user_bins={"age": {"trend": "pos", "bins": {
    0: {(-np.inf, 25.0): "cont"},
    1: {(25.0, 60.0): "cont"},
    2: {(60.0, np.inf): "cont", None: "missing"},
}}})
```

User bins are applied as given (U-shapes are fine); a warning lists training values that
fall in no bin.

## In production

```python
import pickle
binner = pickle.load(open("binner.pkl", "rb"))
binner.set_params(handle_unknown="worst")   # change how unseen values are treated, no refit
X_woe = binner.transform(X_new)
```

`handle_unknown`: `"keep"` (leave the raw value), `"worst"` / `"best"` (lowest / highest
WoE), `"missing_worst"` / `"missing_best"` (WoE of the missing bin, else lowest / highest).

## scikit-learn

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

pipe = Pipeline([("tide", TIDE()), ("lr", LogisticRegression())])
pipe.fit(X, y, tide__periods=periods)
cross_validate(pipe, X, y, params={"tide__periods": periods}, scoring="roc_auc")
```

`get_params` / `set_params` / `clone`, `get_feature_names_out` and metadata routing
(`TIDE().set_fit_request(periods=True)`) work as usual.

## Performance

`fit` works on per-period counts of the sorted unique values, so after one sort per
feature its cost no longer depends on the number of rows. `n_jobs=-1` fits features in
parallel. Measured on a 2-core laptop, 5,000,000 rows and 24 periods, per feature:

| feature | fit | transform |
|---|---|---|
| integer, 100 unique values | ~1 s | ~0.5 s |
| float, 1,000 unique | ~1 s | ~0.5 s |
| float, all unique (undersampled to 200 quantiles) | ~5 s | ~0.5 s |
| float, 530k unique (exact search, `max_unique=10**6`) | ~5 s | ~0.5 s |
| mixed object column (numbers, strings, NaN) | ~3 s | ~2 s |

Run `python benchmarks/bench_fit.py` to measure on your machine.

## License

MIT © Leonid Garin
