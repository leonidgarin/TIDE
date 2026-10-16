# User guide

## Choosing periods

TIDE checks every constraint in **every** period, so the periods define what "stable" means.
Pick the granularity that matches how the model will be monitored (month, quarter, ...),
and drop broken or irrelevant periods before calling `fit`. More periods mean stricter
binning: with many small periods, noisy features may collapse to one bin (you get a
warning and `summary_.n_cont_bins == 1`).

## Reading the results

- `summary_` — start here: IV, number of bins, maximum PSI and warnings per feature.
- `stats_` — the classic binning table for the whole sample.
- `stats_per_` — the same per period, with PSI against the previous (`PSI_seq`) and the
  first (`PSI_base`) period.
- `plot()` — three panels per feature for reports: bin sizes and event rates, event rate of
  each bin over time, and bin shares with PSI over time.

## Parameters that matter most

| Parameter | Effect |
|---|---|
| `min_sample_rate` | Larger → fewer, larger bins. Checked per period. |
| `min_class_obs` | Larger → no bins with very few events in some period. |
| `min_er_delta` | Larger → adjacent bins must be clearly different in every period. |
| `max_unique`, `n_quantiles` | When a feature has more than `max_unique` (default 1000) unique values, only `n_quantiles` (default 200) quantiles are candidate split points. Integer features such as age keep their exact search. |
| `trend` | Force `"pos"` / `"neg"` when business logic dictates the direction. |
| `missing_strategy` | Separate bin for missing values, or join them to the worst / best bin when they are rare. |
| `cat_strategy` | Keep categories separate, ChiMerge them, or merge everything for full stability (`"tide"`). |

## Manual binning

Fit once, inspect `bins_`, edit the dict and pass it back as `user_bins` — any shape is
allowed, including U-shapes. The `"trend"` key selects the bracket convention:
`(a, b]` for `"pos"`, `[a, b)` for `"neg"`.

```python
binner = TIDE().fit(X, y, periods)
bins = binner.bins_["age"]
bins["bins"][0] = {(-np.inf, 30.0): "cont"}   # edit
binner2 = TIDE(user_bins={"age": bins}).fit(X, y, periods)
```
