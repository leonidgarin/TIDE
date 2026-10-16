"""Trend detection, candidate axis and the greedy TIDE scan for the continuous part.

All functions work on the sorted unique values of the continuous part and their
counts, never on raw rows, so their cost does not grow with the number of rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-8

__all__ = ["Axis", "build_axis", "detect_trend", "greedy_cuts"]


def _value_at(uniq: np.ndarray, cum: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Values at 0-based positions of the sorted data given unique values and cumulative counts."""
    return uniq[np.searchsorted(cum, pos, side="right")]


def weighted_percentile(uniq: np.ndarray, cnt: np.ndarray, q: np.ndarray) -> np.ndarray:
    """``np.percentile(data, q)`` (linear method) computed from unique values and counts."""
    cum = np.cumsum(cnt)
    n = cum[-1]
    h = (n - 1) * np.asarray(q, dtype=np.float64) / 100.0
    lo = np.floor(h).astype(np.int64)
    hi = np.minimum(lo + 1, n - 1)
    v_lo = _value_at(uniq, cum, lo)
    v_hi = _value_at(uniq, cum, hi)
    return v_lo + (h - lo) * (v_hi - v_lo)


def detect_trend(uniq: np.ndarray, cnt: np.ndarray, ev: np.ndarray, n_bins: int) -> str:
    """Sign of the slope of event rate vs. quantile-bin index (v0.5 rule).

    ``uniq`` are the sorted unique continuous values, ``cnt`` / ``ev`` their row and
    event counts. Returns ``"pos"`` when the slope is positive or close to zero.
    """
    if uniq.size == 0:
        return "pos"
    edges = np.unique(weighted_percentile(uniq, cnt, np.linspace(0, 100, n_bins + 1)))
    k = edges.size - 1
    if k < 2:
        return "pos"
    idx = np.searchsorted(edges, uniq, side="left") - 1  # interval (e[i], e[i+1]]
    idx[idx < 0] = 0  # the minimum belongs to the first interval
    b_cnt = np.bincount(idx, weights=cnt, minlength=k)
    b_ev = np.bincount(idx, weights=ev, minlength=k)
    er = b_ev / (b_cnt + EPS)
    a1 = np.polyfit(np.arange(k), er, deg=1)[0]
    if np.isclose(a1, 0) or a1 > 0:
        return "pos"
    return "neg"


@dataclass
class Axis:
    """Ordered candidate points with their per-period counts.

    ``values[k]`` is the split value recorded when a bin ends at point ``k`` (positive
    trend) or when the next bin starts at point ``k`` (negative trend). Points that are
    not continuous (attached missing values) have NaN values.
    """

    values: np.ndarray  # float64 (U,)
    is_cont: np.ndarray  # bool (U,)
    counts: np.ndarray  # int64 (P, U)
    events: np.ndarray  # int64 (P, U)
    undersampled: bool
    point_of_unique: np.ndarray  # int64 (n_unique,): axis point of each unique value


def build_axis(
    uniq: np.ndarray,
    codes: np.ndarray,
    y_cont: np.ndarray,
    p_cont: np.ndarray,
    n_periods: int,
    trend: str,
    max_unique: int,
    n_quantiles: int,
    extra_counts: np.ndarray | None = None,
    extra_events: np.ndarray | None = None,
    extra_position: str | None = None,
) -> Axis:
    """Build the candidate axis: unique values, or quantile points when there are too many.

    ``uniq`` are the sorted unique continuous values and ``codes`` the index into
    ``uniq`` of every continuous row. ``extra_*`` (shape ``(P, m)``) are per-period counts
    of missing values attached at the ``"start"`` or ``"end"`` of the axis
    (missing_strategy worst_bin / best_bin).
    """
    n_u = uniq.size
    undersampled = n_u > max_unique
    if undersampled:
        cnt = np.bincount(codes, minlength=n_u)
        cum = np.cumsum(cnt)
        n = cum[-1]
        levels = np.linspace(0.0, 1.0, n_quantiles + 1)
        # inverted CDF: the smallest value with at least q * n rows at or below it
        pos = np.clip(np.ceil(levels * n).astype(np.int64) - 1, 0, n - 1)
        q = _value_at(uniq, cum, pos)
        if trend == "pos":
            # groups (pts[j-1], pts[j]]; the last point is the maximum
            pts = np.unique(np.append(q[1:], uniq[-1]))
            point_of_unique = np.searchsorted(pts, uniq, side="left")
        else:
            # groups [pts[j], pts[j+1]); the first point is the minimum
            pts = np.unique(np.append(uniq[0], q[:-1]))
            point_of_unique = np.searchsorted(pts, uniq, side="right") - 1
        values = pts.astype(np.float64)
    else:
        values = uniq.astype(np.float64)
        point_of_unique = np.arange(n_u)
    point_of_unique = point_of_unique.astype(np.int64)

    u = values.size
    gid = point_of_unique[codes] if undersampled else codes
    flat = p_cont * u + gid
    counts = np.bincount(flat, minlength=n_periods * u).reshape(n_periods, u)
    events = np.bincount(flat, weights=y_cont, minlength=n_periods * u)
    events = np.rint(events).astype(np.int64).reshape(n_periods, u)
    is_cont = np.ones(u, dtype=bool)

    if extra_counts is not None and extra_counts.shape[1] > 0:
        m = extra_counts.shape[1]
        nan_vals = np.full(m, np.nan)
        no = np.zeros(m, dtype=bool)
        if extra_position == "start":
            values = np.concatenate([nan_vals, values])
            is_cont = np.concatenate([no, is_cont])
            counts = np.hstack([extra_counts, counts])
            events = np.hstack([extra_events, events])
            point_of_unique = point_of_unique + m
        else:
            values = np.concatenate([values, nan_vals])
            is_cont = np.concatenate([is_cont, no])
            counts = np.hstack([counts, extra_counts])
            events = np.hstack([events, extra_events])

    return Axis(
        values,
        is_cont,
        np.ascontiguousarray(counts),
        np.ascontiguousarray(events),
        undersampled,
        point_of_unique,
    )


def _first_true(pred, lo: int, hi: int) -> int:
    """Smallest k in [lo, hi] with pred(k) True (pred monotone False..True); hi + 1 if none."""
    while lo <= hi:
        mid = (lo + hi) // 2
        if pred(mid):
            hi = mid - 1
        else:
            lo = mid + 1
    return lo


def greedy_cuts(
    axis: Axis,
    per_size: np.ndarray,
    trend: str,
    min_sample_rate: float,
    min_class_obs: int,
    min_er_delta: float,
) -> list[int]:
    """Greedy TIDE scan (v0.5 rules). Returns the axis indices where a bin ends.

    A candidate end point ``k`` for the bin starting at ``base`` is valid when, in every
    period: the cumulative event rate of ``[base..k]`` equals its suffix min (pos) / max
    (neg), so extending the bin cannot move the rate the wrong way; the bin holds at least
    ``min_sample_rate`` of the period's rows and ``min_class_obs`` of each class; its rate
    differs from the previous bin's by at least ``min_er_delta``. The split must lie between
    two continuous points. The first valid ``k`` is taken, and it is kept only if the
    remainder can still form a valid bin; otherwise the scan stops.

    Implementation: the size and class conditions are monotone in ``k``, so their limits
    are found by exact binary search, and the per-period event-rate checks run only on
    the remaining range. The formulas are the same as in v0.5, so the results are identical.
    """
    pos = trend == "pos"
    n_per, u = axis.counts.shape
    cc_all = np.zeros((n_per, u + 1))
    ce_all = np.zeros((n_per, u + 1))
    np.cumsum(axis.counts, axis=1, out=cc_all[:, 1:])
    np.cumsum(axis.events, axis=1, out=ce_all[:, 1:])
    size_den = EPS + per_size.astype(np.float64)
    splittable = axis.is_cont & np.append(axis.is_cont[1:], False)
    msr, mco, d = min_sample_rate, min_class_obs, min_er_delta

    def bin_ok(base: int, k: int) -> bool:  # size and class presence of [base..k]
        cc = cc_all[:, k + 1] - cc_all[:, base]
        ce = ce_all[:, k + 1] - ce_all[:, base]
        return bool(np.all(cc / size_den >= msr) and np.all(ce >= mco) and np.all(cc - ce >= mco))

    def rem_ok(k: int) -> bool:  # size and class presence of [k+1..end]
        rc = cc_all[:, u] - cc_all[:, k + 1]
        re = ce_all[:, u] - ce_all[:, k + 1]
        return bool(np.all(rc / size_den >= msr) and np.all(re >= mco) and np.all(rc - re >= mco))

    cuts: list[int] = []
    er_prev = np.full(n_per, -np.inf if pos else np.inf)
    base = 0
    while base <= u - 1:
        k_lo = _first_true(lambda k, b=base: bin_ok(b, k), base, u - 1)
        # remainder conditions are monotone decreasing in k: last k that keeps them
        k_hi = _first_true(lambda k: not rem_ok(k), base, u - 1) - 1
        if k_lo > u - 1:
            break  # no bin from base satisfies size / class: no candidate (v0.5 breaks too)
        hi = min(k_hi, u - 2)
        if hi < k_lo:
            break
        mask = splittable[k_lo : hi + 1].copy()
        start = k_lo
        for p in range(n_per):
            first = int(np.argmax(mask))
            if not mask[first]:
                break
            if first:
                start += first
                mask = mask[first:]
            cc = cc_all[p, start + 1 :] - cc_all[p, base]
            ce = ce_all[p, start + 1 :] - ce_all[p, base]
            er = ce / (EPS + cc)
            if pos:
                window = np.minimum.accumulate(er[::-1])[::-1]
                cond = er >= er_prev[p] + d
            else:
                window = np.maximum.accumulate(er[::-1])[::-1]
                cond = er <= er_prev[p] - d
            cond &= er == window
            mask &= cond[: mask.size]
        if not mask.any():
            break
        k = start + int(np.argmax(mask))
        cc = cc_all[:, k + 1] - cc_all[:, base]
        ce = ce_all[:, k + 1] - ce_all[:, base]
        cuts.append(k)
        er_prev = ce / (EPS + cc)
        base = k + 1

    # The last bin (the remainder) must also keep min_er_delta to its neighbour.
    while cuts and d > 0:
        last = cuts[-1]
        start_prev = cuts[-2] + 1 if len(cuts) > 1 else 0
        prev_c = cc_all[:, last + 1] - cc_all[:, start_prev]
        prev_e = ce_all[:, last + 1] - ce_all[:, start_prev]
        rem_c = cc_all[:, u] - cc_all[:, last + 1]
        rem_e = ce_all[:, u] - ce_all[:, last + 1]
        er_p = prev_e / (EPS + prev_c)
        er_r = rem_e / (EPS + rem_c)
        ok_last = np.all(er_r >= er_p + d) if pos else np.all(er_r <= er_p - d)
        if ok_last:
            break
        cuts.pop()

    return cuts
