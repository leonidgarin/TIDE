"""Fitting of a single feature: from raw column to bins and per-period counts."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ._bins import compile_bins
from ._categorical import chi_merge, tide_merge
from ._greedy import build_axis, detect_trend, greedy_cuts
from ._parse import parse_column

__all__ = ["FeatureFit", "FitParams", "fit_feature", "fit_user_bins"]


@dataclass(frozen=True)
class FitParams:
    min_sample_rate: float
    min_class_obs: int
    min_er_delta: float
    max_unique: int
    n_quantiles: int
    n_trend_bins: int
    min_bound: object
    max_bound: object
    missing_strategy: str
    missing_rate: float
    cat_strategy: str
    alpha_significance: float


@dataclass
class FeatureFit:
    name: str
    bins: dict
    counts: np.ndarray  # (n_bins, P), rows in sorted bin-index order
    events: np.ndarray
    n_cont_bins: int
    undersampled: bool = False
    user: bool = False
    warnings: list[str] = field(default_factory=list)


def sorted_unique(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted unique values and the index of each element into them.

    Hashing is faster for low-cardinality data, sorting for high-cardinality data;
    a small sample decides which one to use.
    """
    sample = x[:: max(1, x.size // 20000)]
    if np.unique(sample).size > 0.5 * sample.size:
        order = np.argsort(x)
        xs = x[order]
        new = np.empty(x.size, dtype=bool)
        new[0] = True
        np.not_equal(xs[1:], xs[:-1], out=new[1:])
        codes = np.empty(x.size, dtype=np.int64)
        codes[order] = np.cumsum(new) - 1
        return xs[new], codes
    codes, uniq = pd.factorize(x, sort=True)
    return np.asarray(uniq, dtype=np.float64), codes.astype(np.int64)


def _period_counts(codes: np.ndarray, y: np.ndarray, p: np.ndarray, n_groups: int, n_per: int):
    m = codes >= 0
    flat = codes[m] * n_per + p[m]
    c = np.bincount(flat, minlength=n_groups * n_per).reshape(n_groups, n_per)
    e = np.bincount(flat, weights=y[m], minlength=n_groups * n_per)
    return c, np.rint(e).astype(np.int64).reshape(n_groups, n_per)


def _coverage_warning(name: str, codes: np.ndarray, raw) -> str | None:
    miss = codes < 0
    if not miss.any():
        return None
    vals = np.asarray(raw, dtype=object)[miss]
    examples = list(dict.fromkeys(map(repr, vals[:1000])))[:5]
    return (
        f"Feature {name!r}: {int(miss.sum())} training values are not covered by the bins "
        f"(e.g. {', '.join(examples)}); they are excluded from the stats."
    )


def fit_user_bins(name, raw, bins, y, p, n_per, cat_values, missing_values) -> FeatureFit:
    """Apply user bins as given and count rows per bin and period."""
    parsed = parse_column(raw, cat_values, missing_values)
    comp = compile_bins(bins, name)
    codes = comp.assign(parsed)
    c, e = _period_counts(codes, y, p, len(comp.bin_ids), n_per)
    n_cont = sum(any(k == "cont" for k in bins["bins"][b].values()) for b in comp.bin_ids)
    warns = [w for w in [_coverage_warning(name, codes, raw)] if w]
    return FeatureFit(name, bins, c, e, n_cont, user=True, warnings=warns)


def fit_feature(
    name, raw, y, p, n_per, per_size, trend_param, cat_values, missing_values, prm: FitParams
) -> FeatureFit:
    """Run the TIDE algorithm on one feature."""
    parsed = parse_column(raw, cat_values, missing_values)
    n = parsed.n
    warns: list[str] = []
    cm = parsed.cont_mask
    x_cont = parsed.x_cont
    has_cont = x_cont.size > 0
    y_cont = y[cm]
    p_cont = p[cm]
    if has_cont:
        uniq, u_codes = sorted_unique(x_cont)
    else:
        u_codes, uniq = np.empty(0, dtype=np.int64), np.empty(0)

    # 1. trend
    if trend_param in ("pos", "neg"):
        trend = trend_param
    else:
        u_cnt = np.bincount(u_codes, minlength=uniq.size)
        u_ev = np.bincount(u_codes, weights=y_cont, minlength=uniq.size)
        trend = detect_trend(uniq, u_cnt, u_ev, prm.n_trend_bins)

    # 2. missing values attached to the worst / best end of the axis?
    n_miss_keys = len(parsed.miss_keys)
    miss_rows = parsed.miss_codes >= 0
    r_missing = miss_rows.sum() / n
    attach = (
        has_cont
        and prm.missing_strategy in ("worst_bin", "best_bin")
        and 0 < r_missing < prm.missing_rate
    )
    position = None
    extra_c = extra_e = None
    if attach:
        at_end = (prm.missing_strategy == "worst_bin") == (trend == "pos")
        position = "end" if at_end else "start"
        extra_c, extra_e = _period_counts(parsed.miss_codes, y, p, n_miss_keys, n_per)
        extra_c, extra_e = extra_c.T, extra_e.T

    # 3-4. continuous bins
    cont_bounds: np.ndarray = np.empty(0)
    undersampled = False
    if has_cont:
        axis = build_axis(
            uniq,
            u_codes,
            y_cont,
            p_cont,
            n_per,
            trend,
            prm.max_unique,
            prm.n_quantiles,
            extra_c,
            extra_e,
            position,
        )
        undersampled = axis.undersampled
        cuts = np.asarray(
            greedy_cuts(
                axis, per_size, trend, prm.min_sample_rate, prm.min_class_obs, prm.min_er_delta
            ),
            dtype=np.int64,
        )
        mids = axis.values[cuts] if trend == "pos" else axis.values[cuts + 1]
        lo = np.nextafter(uniq[0], -np.inf) if prm.min_bound == "col_min" else prm.min_bound
        hi = np.nextafter(uniq[-1], np.inf) if prm.max_bound == "col_max" else prm.max_bound
        lo, hi = float(lo), float(hi)
        mids = mids[(mids > lo) & (mids < hi)]
        cont_bounds = np.unique(np.r_[lo, mids, hi])
        if cont_bounds.size == 2 and uniq.size > 1:
            warns.append(
                f"Feature {name!r}: no split satisfies the constraints; "
                "the continuous part is one bin."
            )

    # atoms: continuous intervals, categories, missing (one atom unless attached)
    intervals = list(zip(cont_bounds[:-1].tolist(), cont_bounds[1:].tolist()))
    n_cont = len(intervals)
    n_cat = len(parsed.cat_keys)
    atom_bins: dict[int, dict] = {i: {iv: "cont"} for i, iv in enumerate(intervals)}
    if attach:
        target = n_cont - 1 if position == "end" else 0
        for k in parsed.miss_keys:
            atom_bins[target][k] = "missing"
    for j, k in enumerate(parsed.cat_keys):
        atom_bins[n_cont + j] = {k: "cat"}
    if n_miss_keys and not attach:
        atom_bins[n_cont + n_cat] = {k: "missing" for k in parsed.miss_keys}

    comp = compile_bins({"trend": trend, "bins": atom_bins}, name)
    # continuous rows: assign the unique values once, then gather per row
    cont_codes = comp.assign_values(uniq)[u_codes] if has_cont else None
    codes = comp.assign(parsed, cont_codes=cont_codes)
    c_atoms, e_atoms = _period_counts(codes, y, p, len(atom_bins), n_per)

    # 5. categorical strategy
    if prm.cat_strategy == "chi-squared" and n_cat > 1:
        cat_ids = np.arange(n_cont, n_cont + n_cat)
        tot_c = c_atoms[cat_ids].sum(axis=1)
        tot_e = e_atoms[cat_ids].sum(axis=1)
        if n_cat >= 100:
            warns.append(f"Feature {name!r} has {n_cat} categories; ChiMerge is O(n^2) per step.")
        cat_groups = chi_merge(tot_c - tot_e, tot_e, prm.alpha_significance)
        groups = [[i] for i in range(n_cont)] + [[n_cont + j for j in g] for g in cat_groups]
        groups += [[a] for a in range(n_cont + n_cat, len(atom_bins))]
    elif prm.cat_strategy == "tide" and len(atom_bins) > 1:
        if len(atom_bins) >= 200:
            warns.append(
                f"Feature {name!r} has {len(atom_bins)} bins before the TIDE merge; "
                "it is O(n^3) and may be slow."
            )
        groups = tide_merge(
            c_atoms,
            e_atoms,
            n_cont,
            trend,
            prm.min_sample_rate,
            prm.min_class_obs,
            prm.min_er_delta,
        )
        groups = _order_groups(groups, n_cont)
    else:
        groups = [[a] for a in range(len(atom_bins))]

    bins = {"trend": trend, "bins": {}}
    out_c = np.zeros((len(groups), n_per), dtype=np.int64)
    out_e = np.zeros_like(out_c)
    n_cont_bins = 0
    for gi, g in enumerate(groups):
        cont_atoms = sorted(a for a in g if a < n_cont)
        merged: dict = {}
        if cont_atoms:
            n_cont_bins += 1
            a0, b0 = intervals[cont_atoms[0]][0], intervals[cont_atoms[-1]][1]
            merged[(a0, b0)] = "cont"
        for a in g:
            for atom, kind in atom_bins[a].items():
                if kind != "cont":
                    merged[atom] = kind
        bins["bins"][gi] = merged
        out_c[gi] = c_atoms[g].sum(axis=0)
        out_e[gi] = e_atoms[g].sum(axis=0)

    cov = _coverage_warning(name, codes, raw)
    if cov:
        warns.append(cov)
    return FeatureFit(name, bins, out_c, out_e, n_cont_bins, undersampled, warnings=warns)


def _order_groups(groups: list[list[int]], n_cont: int) -> list[list[int]]:
    """Bins with continuous intervals first (in axis order), then the rest by first atom."""

    def key(g):
        cont = [a for a in g if a < n_cont]
        return (0, min(cont)) if cont else (1, min(g))

    return sorted(groups, key=key)
