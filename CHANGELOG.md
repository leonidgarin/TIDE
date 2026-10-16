# Changelog

## 0.6.0 (unreleased)

Complete rewrite. The greedy algorithm and its rules are unchanged: on the scenarios in
`tests/golden`, the bins are identical to v0.5.0.

### New
- scikit-learn estimator: `get_params` / `set_params` / `clone`, `Pipeline`,
  `cross_validate(params={"tide__periods": ...})`, metadata routing,
  `get_feature_names_out`.
- NaN / None / pd.NA / NaT are supported in features (bins key `None`).
- `cat_values` and `missing_values` turn numeric codes such as `-999` into categories or
  missing values, globally or per feature.
- `trend` can be forced (`"pos"` / `"neg"`), globally or per feature.
- `max_unique` / `n_quantiles` (defaults 1000 / 200): features with more than 1000 unique values
  are split at 200 quantiles instead of every unique value. This is faster, and with many
  periods the exact search often finds no split at all. Pass a large `max_unique` for v0.5 behaviour.
- `user_bins`: apply your own bins (any shape), with a coverage warning.
- `summary_` table; a warning when a feature collapses to one continuous bin.
- `n_jobs` (parallel over features) and `verbose` (progress bar).
- `plot()` returns the figures and accepts `show=False`.
- Periods may be any sortable type and need not be sorted.

### Fixed
- `min_er_delta` is now also guaranteed for the last bin (it is merged otherwise).
- `min_er_delta` no longer applies to the first bin (v0.5 required its event rate ≥ delta).
- With `worst_bin` / `best_bin`, no split is made between the continuous part and the
  attached missing values.
- `cat_strategy="tide"` now guarantees all constraints: continuous bins merge only with
  adjacent ones (their intervals become one), and their monotonicity is re-checked.
  The missing bin takes part in the merge.
- `transform` no longer mistakes values equal to a WoE for unknown values.
- `stats_per` no longer duplicates rows on refit.
- `"col_min"` / `"col_max"` bounds now use the next representable float instead of
  `± 1e-8`, which failed for large values.

### Changed (breaking)
- WoE uses a symmetric epsilon, `ln((%nonevents + 1e-8) / (%events + 1e-8))`: a bin without
  non-events no longer gets WoE = -inf and IV = inf. Other bins are unchanged up to ~1e-8.
- `fit(X, y, per)` → `fit(X, y, periods)`; `reset` and `return_bins` were removed.
- `handle_unknown` moved from `transform()` into the constructor (`set_params` changes it).
- `transform` requires the same columns as `fit`; `xnames` and ndarray input were removed.
- Attributes: `exog_bins` → `bins_`, `exog_woes` → `woes_`, `stats` → `stats_`,
  `stats_per` → `stats_per_`; the stats tables have a new `bin_idx` column.
- `missing_categories` → `missing_values`; `n_prebins` → `n_trend_bins`.
- Bin labels (stats tables and plots) show integer bounds without decimals (`(18,25]`) and
  other bounds with exactly 4 decimals (`(0.3400,1.1130]`). `bins_` keeps the full values.
