# TIDE — Design (target version 0.6.0)

Versioning: the rewrite ships as **0.6.0** and stays on 0.x while the API settles. **1.0.0** comes when we
declare the API stable.

TIDE (Time Invariant Discretization Engine) is a binning library for credit scorecards with a binary target.
Its bins are stable over time: in **every** period the WoE/event-rate trend is monotonic in the same
direction, the curves do not cross, and every bin is large enough.

Distribution: `pip install tide-binning`, `import tide`. MIT license, Python >= 3.10.

---

## 1. Public API

```python
from tide import TIDE

binner = TIDE(
    # constraints (hard)
    min_sample_rate=0.05,      # min share of rows per bin, checked in EVERY period
    min_class_obs=1,           # min events and min non-events per bin, in every period
    min_er_delta=0.0,          # min event-rate gap between adjacent bins, in every period
    trend="auto",              # "auto" | "pos" | "neg" | {feature: "pos"/"neg"}
    # candidate split points
    max_unique=1000,           # if #unique numeric values > max_unique -> quantile undersampling
    n_quantiles=200,           # number of quantile candidate points used for undersampling
    n_trend_bins="log",        # bins for trend detection: "log" -> max(3, ceil(ln N)) or int >= 2
    min_bound=-np.inf,         # or "col_min"
    max_bound=np.inf,          # or "col_max"
    # special values
    missing_values=(),         # extra values treated as missing (NaN/None/pd.NA always are)
    cat_values=(),             # numeric values treated as categories, e.g. -999
                               #   both accept a list (all features) or {feature: list}
    missing_strategy="separate_bin",  # "separate_bin" | "worst_bin" | "best_bin"
    missing_rate=0.05,         # worst/best only: attach missings if their share < missing_rate
    cat_strategy="separate_bin",      # "separate_bin" | "chi-squared" | "tide"
    alpha_significance=0.05,   # chi-squared merge threshold
    # user-defined bins
    user_bins=None,            # {feature: bins_dict}, applied as-is, no search
    # transform
    handle_unknown="keep",     # "keep"|"worst"|"best"|"missing_worst"|"missing_best" (v0.5 semantics)
    # runtime
    n_jobs=None,               # joblib, parallel across features
    verbose=False,             # tqdm progress bar
)

binner.fit(X, y, periods)              # returns self; X: pd.DataFrame, y: binary, periods: array-like or None
X_woe = binner.transform(X)            # pd.DataFrame with columns "{feature}_woe"
binner.fit_transform(X, y, periods)
binner.plot(["age", "income"])         # matplotlib, always installed
```

### sklearn compliance (checked against the sklearn developer guide)

| Rule | How TIDE follows it |
|---|---|
| `__init__` only stores parameters, unchanged; no validation or logic | Yes. All checks run in `fit`. |
| Parameters are keyword args with defaults, so `get_params` / `set_params` / `clone` work | Yes, via `BaseEstimator`. |
| `fit(X, y, **fit_params)` returns `self` | `periods` is a fit parameter, like `sample_weight`. |
| Fit params work in meta-estimators | `Pipeline.fit(X, y, tide__periods=per)`. With metadata routing on: `binner.set_fit_request(periods=True)`, which `BaseEstimator` generates automatically. In `cross_validate(..., params={"periods": per})` the periods get split along with the rows. |
| `fit` refits from scratch, with no hidden state from earlier fits | Yes. That's why v0.5's `reset` / `return_bins` are removed. |
| Learned state lives in attributes with a trailing `_`; `check_is_fitted` works | `bins_`, `woes_`, `stats_`, ... |
| `transform(X)` takes no behaviour kwargs | `handle_unknown` moved from `transform()` into `__init__`, the same as `OneHotEncoder`. |
| `n_features_in_`, `feature_names_in_`, `get_feature_names_out()` | Yes. `transform` raises if the columns differ from fit. `get_feature_names_out()` returns `{feature}_woe`. |
| Output container | Returns a pandas DataFrame; sklearn passes a DataFrame through unchanged under the default `set_output` config. |
| `n_jobs`, `verbose` names | Standard. |

**Changing `handle_unknown` after fit** (e.g. production decides differently from the DS): call
`binner.set_params(handle_unknown="worst")` on the fitted (unpickled) object, or
`pipe.set_params(tide__handle_unknown="worst")` inside a pipeline. `handle_unknown` is read only in
`transform`, so changing it leaves the bins and WoEs untouched and needs no refit. A test pins this
behaviour down.

Not targeted: full `sklearn.utils.estimator_checks.check_estimator`. It feeds random ndarrays, so it is
incompatible with pandas-only, mixed-type input and a required binary `y`. We test the applicable
subset ourselves (clone, params round-trip, not-fitted error, Pipeline, cross-validation with periods).
Requires `scikit-learn>=1.4`: the first release with metadata routing through `Pipeline` and `cross_validate(params=...)` (verified against 1.4.2 in the oldest-dependencies CI job).

Fitted attributes (trailing underscore, sklearn convention):

| Attribute | Content |
|---|---|
| `bins_` | `{feature: bins_dict}` (format below) |
| `woes_` | `{feature: {bin_idx: woe}}` |
| `stats_` | per feature/bin: `variable, trend, bin, n_obs, sample_rate, n_events, event_rate, WoE, IV_contrib, IV_total` |
| `stats_per_` | per period/feature/bin: `period, variable, trend, bin, n_obs, sample_rate, n_events, event_rate, PSI_seq, PSI_base` |
| `summary_` | one row per feature: `IV, n_bins, trend, max_PSI_seq, max_PSI_base, single_bin, user_bins, warnings` |
| `periods_` | sorted unique periods seen in fit |
| `feature_names_in_`, `n_features_in_` | sklearn standard |

### Bins format (same as v0.5)

```python
{"trend": "pos",
 "bins": {0: {(-np.inf, 0.34): "cont", "special_category": "cat"},
          1: {(0.34, 0.65): "cont"},
          2: {(0.65, np.inf): "cont"},
          3: {None: "missing", -999: "missing"}}}
```

* **Bracket convention as in v0.5, set by trend:**
  * `pos` → intervals `(a, b]`, and the split point is the last value of the left bin (`u[k]`);
  * `neg` → intervals `[a, b)`, and the split point is the first value of the right bin (`u[k+1]`).

  In both cases, an unseen value between `u[k]` and `u[k+1]` goes to the bin with the **higher event
  rate** (the conservative choice in production). The `-inf` / `+inf` edges always print as `(` / `)`.
* `None` is the key for NaN/None/pd.NA/NaT. (`np.nan` is unreliable as a dict key, because `nan != nan`.)
* User bins use the same format. Their `"trend"` key is **required** whenever they contain continuous
  intervals, because it selects the bracket convention.

## 2. Input handling

* **Mixed columns** are parsed once, vectorized:
  1. missing = NaN/None/pd.NA/NaT, or a value in `missing_values`;
  2. categorical = any `str` (including `"12"`), `bool`, or a value in `cat_values`;
  3. everything else that is numeric is continuous; `±inf` falls into the edge bins.
  Fast path: a numeric dtype column needs no per-element type checks at all.
* **y**: 0/1 or bool, no NaN, both classes present.
* **periods**: any sortable dtype (str, datetime, pd.Timestamp, Period, numbers); not assumed sorted;
  NaN/NaT → error. `periods=None` means one period (plain monotonic binning).
  The user chooses the granularity and excludes broken periods before calling `fit`.

## 3. Algorithm (same logic as v0.5, faster implementation)

Per feature:

1. **Trend.** Use the forced value if given, otherwise as v0.5: `n_trend_bins` quantile bins on the
   continuous part, then the sign of the slope of event rate vs bin index (slope 0 → `pos`).
2. **Candidate axis.** Sorted unique continuous values. If there are more than `max_unique` of them,
   the axis becomes `n_quantiles` quantile edges instead (undersampling) and each value is mapped to its
   group with `searchsorted`. The groups use the trend's bracket convention (`side="left"` for `(a,b]`,
   `side="right"` for `[a,b)`), so the split points are always the quantile edges. If `missing_strategy` is worst/best and `0 < missing_share < missing_rate`,
   the missing values are appended to the worst/best end of the axis (as in v0.5); they take part in
   event rates but can never be a split point.
3. **Crosstab.** One `np.bincount` over `axis_idx * P + period_idx` gives `counts[U, P]` and `events[U, P]`.
   All later steps use these arrays only; the raw data is not touched again.
4. **Greedy scan.** Exactly the v0.5 rules: from `idx_base`, a candidate `k` is valid if in every period
   the cumulative event rate equals its suffix min (pos) / max (neg), the size is `>= min_sample_rate`
   (relative to the period's total rows), each class has `>= min_class_obs` rows, the rate gap to the
   previous bin is `>= min_er_delta`, it is not a missing point, and the remainder can still form a valid
   bin. Take the first valid `k` and repeat. Speed-up: global prefix sums instead of a fresh cumsum per
   step (exact, because the counts are integers).
   **Fix:** v0.5 does not check `min_er_delta` for the last bin. If the remainder violates it, the
   remainder is merged into the previous bin. This keeps monotonicity and makes every constraint hold.
5. **Categories:**
   * `separate_bin` (default): each category is its own bin, and categories may cross.
   * `chi-squared`: v0.5 ChiMerge, with cached counts (no rescans of the data).
   * `tide`: merges **all** kinds of bins (cont/cat/missing) until there are no crossings, then checks
     delta, size and class count, using v0.5 priorities and the closest-by-squared-ER-distance partner.
     Two refinements make P1–P5 guaranteed:
     (a) two continuous bins may merge only if they are adjacent, and then their intervals become one;
     (b) monotonicity of the bins that contain continuous intervals is checked in every period, with
     top priority.
     The merge loop updates a `[n_bins, P]` count matrix incrementally.
6. **Missing:** if not attached in step 2, all missing values go to one separate bin (if any are present).
7. **Single bin:** if no split survives, the feature gets one continuous bin and a `UserWarning`;
   `summary_.single_bin = True`.
8. **WoE / IV / PSI:** v0.5 formulas, unchanged:
   `WoE = ln((%nonevents + eps) / (%events + eps))` (symmetric eps, agreed after review: v0.5 had eps
   only in the denominator, which gave -inf WoE / inf IV for bins without non-events),
   `IV = Σ (%ne − %e)·WoE`, PSI sequential and on-base.

**User bins**: applied as given (U-, V-shapes allowed). There is a quick coverage check: a warning names
training values that fall in no bin. Stats are computed exactly as for fitted bins.

**Transform**: vectorized (`searchsorted` for intervals, with `side` chosen by the feature's trend; a dict
map for categories and missing values).
Values in no bin are found by an explicit "unassigned" mask, not by `x == x_woe` as in v0.5.
Output column names are `{feature}_woe`, as in v0.5.

## 4. Performance target

* O(N log N) per feature (one sort), then O(U·P·n_bins) for the scan.
* Features run in parallel with joblib (`n_jobs`); numeric columns go to workers via memmap.
* Target: 5M rows × 1000 features in minutes on a multi-core laptop. A benchmark script in `benchmarks/`.
* Resolved: the default was lowered from 1,000,000 to 1000. Exact search on ~1M points was slow and,
  with many periods, collapsed informative features to one bin (see docs/algorithm.md). Old note:
  with `max_unique=1_000_000`, a feature with ~1M unique values and many periods costs roughly
  1–10 s per feature in the scan. Milestone 1 benchmarks this; if needed we revisit the default.

## 5. Package layout & tooling

```
pyproject.toml            # hatchling; deps: numpy, pandas, scipy, scikit-learn>=1.4, matplotlib, joblib, tqdm
src/tide/
  __init__.py
  _estimator.py           # TIDE class
  _parse.py               # mixed-type split, period/y validation
  _crosstab.py            # candidate axis, undersampling, bincount
  _greedy.py              # continuous scan
  _categorical.py         # chi-merge, tide merge
  _metrics.py             # WoE, IV, PSI
  _bins.py                # bins format: apply, validate, coverage check, labels
  _stats.py               # stats_, stats_per_, summary_
  plotting.py
tests/                    # pytest + hypothesis
docs/                     # mkdocs-material + mkdocstrings (numpy docstrings), GitHub Pages
benchmarks/
.github/workflows/        # ci.yml, release.yml, docs.yml
```

* **Tests**:
  - unit tests per module;
  - **property tests** (hypothesis): on random drifting data, assert every hard constraint holds in
    `stats_per_`;
  - **golden tests** against v0.5.0 output on fixed datasets (identical, except for the documented fixes);
  - sklearn `check_estimator`-style checks where applicable;
  - coverage target 90%+.
* **CI**: ruff (lint and format), pytest on {ubuntu, windows, macos} × {3.10 … 3.13}, build sdist + wheel.
* **Release**: a git tag `vX.Y.Z` triggers build and upload to PyPI via Trusted Publishing (no tokens).
  Pure-Python wheel, so it installs anywhere numpy/pandas do. All dependencies, matplotlib included,
  ship prebuilt wheels for Windows, macOS and Linux, so a plain `pip install tide-binning` is enough.

## 6. Milestones

1. Skeleton, CI, parsing, crosstab, greedy scan, WoE/IV/PSI, transform, golden tests, benchmark.
2. Missing strategies, categorical strategies (separate, chi-squared, tide), user bins.
3. Stats tables, summary, plots.
4. Docs site, README, examples notebook, release **0.6.0** to TestPyPI, then PyPI.

Git and GitHub are connected later. Until then the code lives locally in `/home/user/Desktop/test`.

## 7. Changes vs v0.5 (breaking)

* `fit(X, y, per)` → `fit(X, y, periods)`; fitted attributes renamed (`exog_bins` → `bins_`, etc.).
* `reset` / `return_bins` are removed (sklearn semantics: `fit` always refits). To add features, use
  another instance or `user_bins`.
* `handle_unknown` moves from `transform()` into the constructor. `transform` no longer takes `xnames`
  or ndarray input; it needs the same columns as `fit`.
* NaN is allowed in X; the bins key for it is `None`.
* Last-bin `min_er_delta` fix; the tide categorical mode guarantees all constraints.

Unchanged: bins format, bracket convention, WoE/IV/PSI formulas, greedy rules, missing and
categorical strategies.
