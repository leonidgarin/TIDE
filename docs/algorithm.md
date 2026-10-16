# Algorithm

For each feature:

1. **Split the column** into continuous values (real numbers), categories (strings, bools,
   `cat_values`) and missing values (NaN-likes, `missing_values`).
2. **Trend.** Unless forced, the sign of the slope of a linear fit of event rate against
   the index of `max(3, ceil(ln n))` quantile bins of the continuous part.
3. **Candidate axis.** The sorted unique continuous values — or `n_quantiles` quantiles
   when there are more than `max_unique` of them. With `missing_strategy="worst_bin"` /
   `"best_bin"` and a missing share below `missing_rate`, the missing values are placed at
   the worst / best end of the axis.
4. **Greedy scan.** From the start of the axis, the bin `[base..k]` is a valid candidate
   if, in every period:
     - its cumulative event rate equals the minimum (positive trend) / maximum (negative
       trend) over all its extensions `[base..j]`, `j ≥ k` — so the next bin can only have
       a higher (lower) rate;
     - it holds `≥ min_sample_rate` of the period's rows and `≥ min_class_obs` events and
       non-events;
     - its rate differs from the previous bin's by `≥ min_er_delta`;
     - and the split lies between two continuous points.

   The first valid `k` is taken if the remainder can still form a valid bin; otherwise
   the scan stops and the remainder is the last bin. If the last bin breaks
   `min_er_delta`, it is merged into its neighbour.
5. **Categories.** Separate bins, ChiMerge (Kerber, 1992), or the TIDE merge: until no two
   bins' event-rate curves cross and every constraint holds, the worst bin is merged with
   the closest one (squared event-rate distance); continuous bins only merge with adjacent
   ones.
6. **WoE / IV / PSI**:
   `WoE = ln((%nonevents + ε) / (%events + ε))`, `IV = Σ (%nonevents − %events) · WoE`,
   `PSI = Σ (%new − %old) · ln(%new / (%old + ε))`, `ε = 1e-8`.

!!! note "Exact search on many unique values"
    A candidate must be the lowest (highest) cumulative event rate from that point on in
    **every** period simultaneously. With hundreds of thousands of candidate points, each
    step adds only a few rows, the rate jitters, and with many periods such a point may
    never exist — the feature then becomes a single bin. Lowering `max_unique` makes the
    scan work on quantiles instead, which smooths the jitter.
