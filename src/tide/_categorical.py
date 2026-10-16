"""Merging strategies for categorical bins: ChiMerge and the TIDE stability merge."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from ._greedy import EPS

__all__ = ["chi2_pvalues", "chi_merge", "tide_merge"]


def chi2_pvalues(a0, a1, b0, b1) -> np.ndarray:
    """Vectorized p-values of the 2x2 chi-squared test with Yates' correction.

    Matches ``scipy.stats.chi2_contingency([[a0, a1], [b0, b1]]).pvalue``. As in v0.5,
    tables with two or more zero cells are given p = 1, unless the two bins are
    perfectly separated (``[[0, +], [+, 0]]`` or ``[[+, 0], [0, +]]``).
    """
    obs = np.stack(
        [np.asarray(a0, float), np.asarray(a1, float), np.asarray(b0, float), np.asarray(b1, float)]
    )
    zeros = (obs == 0).sum(axis=0)
    diag = ((obs[0] == 0) & (obs[1] > 0) & (obs[2] > 0) & (obs[3] == 0)) | (
        (obs[0] > 0) & (obs[1] == 0) & (obs[2] == 0) & (obs[3] > 0)
    )
    force_one = (zeros >= 2) & ~diag

    ra, rb = obs[0] + obs[1], obs[2] + obs[3]
    c0, c1 = obs[0] + obs[2], obs[1] + obs[3]
    tot = ra + rb
    with np.errstate(divide="ignore", invalid="ignore"):
        exp = np.stack([ra * c0, ra * c1, rb * c0, rb * c1]) / tot
        diff = exp - obs
        adj = obs + np.sign(diff) * np.minimum(0.5, np.abs(diff))
        stat = ((adj - exp) ** 2 / exp).sum(axis=0)
    p = chi2.sf(stat, 1)
    p = np.where(force_one, 1.0, p)
    return np.atleast_1d(p)


def chi_merge(n0: np.ndarray, n1: np.ndarray, alpha: float) -> list[list[int]]:
    """ChiMerge (Kerber, 1992) over categories, as in v0.5.

    Any two bins may merge. The pair with the highest p-value merges while that
    p-value is above ``alpha``. The merged bin moves to the end of the bin list,
    and ties resolve to the pair first seen (v0.5 order).

    Returns the groups as lists of category indices, in v0.5 output order.
    """
    k = n0.shape[0]
    if k < 2:
        return [[i] for i in range(k)]
    members: dict[int, list[int]] = {i: [i] for i in range(k)}
    c0 = np.asarray(n0, float).copy()
    c1 = np.asarray(n1, float).copy()
    order = list(range(k))  # slot ids in bin-list order

    pv = np.full((k, k), -np.inf)
    rank = np.full((k, k), np.iinfo(np.int64).max, dtype=np.int64)
    iu, ju = np.triu_indices(k, 1)
    pv[iu, ju] = chi2_pvalues(c0[iu], c1[iu], c0[ju], c1[ju])
    rank[iu, ju] = np.arange(iu.size)
    next_rank = iu.size

    while len(order) > 1:
        best = pv.max()
        if best <= alpha:
            break
        cand = pv == best
        r = np.where(cand, rank, np.iinfo(np.int64).max)
        i, j = np.unravel_index(np.argmin(r), r.shape)
        i, j = int(i), int(j)
        # merge slot j into slot i; the merged bin goes to the end of the list
        members[i] = members[i] + members.pop(j)
        c0[i] += c0[j]
        c1[i] += c1[j]
        order = [s for s in order if s not in (i, j)] + [i]
        for s in (i, j):
            pv[s, :] = -np.inf
            pv[:, s] = -np.inf
        others = np.array(order[:-1], dtype=np.int64)
        if others.size:
            p_new = chi2_pvalues(
                c0[others], c1[others], np.full(others.size, c0[i]), np.full(others.size, c1[i])
            )
            lo = np.minimum(others, i)
            hi = np.maximum(others, i)
            pv[lo, hi] = p_new
            rank[lo, hi] = next_rank + np.arange(others.size)
            next_rank += others.size

    return [sorted(members[s]) for s in order]


def tide_merge(
    counts: np.ndarray,
    events: np.ndarray,
    n_cont: int,
    trend: str,
    min_sample_rate: float,
    min_class_obs: int,
    min_er_delta: float,
) -> list[list[int]]:
    """Merge atomic bins (continuous bins, categories, missing) until all constraints hold.

    Atoms ``0..n_cont-1`` are the continuous bins, in axis order. Violations are fixed
    in this priority order:

    0. monotonic event rate of the bins with continuous intervals, in every period;
    1. no crossings of event-rate curves between any two bins (v0.5);
    2. minimum event-rate gap ``min_er_delta`` between any two bins (v0.5);
    3. bin size ``>= min_sample_rate`` in every period (v0.5);
    4. ``>= min_class_obs`` events and non-events in every period (v0.5).

    For 1-4, the offending bin (the smallest by rows, on ties) merges with the bin
    closest by squared event-rate distance. Two bins with continuous intervals may merge
    only if they are adjacent, so every bin keeps one contiguous interval.

    Returns the groups as lists of atom indices.
    """
    pos = trend == "pos"
    c = counts.astype(np.float64).copy()
    e = events.astype(np.float64).copy()
    groups: list[list[int]] = [[a] for a in range(c.shape[0])]
    lo = np.array([a if a < n_cont else -1 for a in range(c.shape[0])], dtype=np.int64)
    hi = lo.copy()

    def merge(t: int, g: int) -> None:
        nonlocal c, e, lo, hi
        groups[t] = groups[t] + groups[g]
        c[t] += c[g]
        e[t] += e[g]
        if lo[g] >= 0:
            lo[t] = lo[g] if lo[t] < 0 else min(lo[t], lo[g])
            hi[t] = max(hi[t], hi[g])
        del groups[g]
        c = np.delete(c, g, axis=0)
        e = np.delete(e, g, axis=0)
        lo = np.delete(lo, g)
        hi = np.delete(hi, g)

    while len(groups) > 1:
        er = e / (c + EPS)
        tot = c.sum(axis=1)
        has_cont = lo >= 0

        # 0. monotonicity of continuous bins
        ci = np.flatnonzero(has_cont)
        if ci.size > 1:
            ci = ci[np.argsort(lo[ci])]
            d = er[ci[1:]] - er[ci[:-1]]
            viol = (d < 0).any(axis=1) if pos else (d > 0).any(axis=1)
            if viol.any():
                cand = np.flatnonzero(viol)
                pair_tot = tot[ci[cand]] + tot[ci[cand + 1]]
                q = cand[np.argmin(pair_tot)]
                t, g = int(ci[q]), int(ci[q + 1])
                merge(t, g)
                continue

        sgn = (er[:, None, :] > er[None, :, :]).astype(np.int8) - (
            er[:, None, :] < er[None, :, :]
        ).astype(np.int8)
        crossings = (sgn[:, :, :-1] != sgn[:, :, 1:]).sum(axis=2).sum(axis=1)
        dist = np.abs(er[:, None, :] - er[None, :, :]).min(axis=2)
        np.fill_diagonal(dist, np.inf)
        delta = dist.min(axis=1)
        sample_rate = (c / c.sum(axis=0)).min(axis=1)
        ev_min = e.min(axis=1)
        nev_min = (c - e).min(axis=1)

        if crossings.max() > 0:
            bad = crossings == crossings.max()
        elif delta.min() < min_er_delta:
            bad = delta == delta.min()
        elif sample_rate.min() < min_sample_rate:
            bad = sample_rate == sample_rate.min()
        elif ev_min.min() < min_class_obs:
            bad = ev_min == ev_min.min()
        elif nev_min.min() < min_class_obs:
            bad = nev_min == nev_min.min()
        else:
            break
        cand = np.flatnonzero(bad)
        t = int(cand[np.argmin(tot[cand])])

        allowed = np.ones(len(groups), dtype=bool)
        allowed[t] = False
        if has_cont[t]:
            adjacent = (hi[t] + 1 == lo) | (hi + 1 == lo[t])
            allowed &= ~has_cont | adjacent
        if not allowed.any():  # pragma: no cover - cannot happen with >= 2 groups
            break
        sq = ((er[t] - er) ** 2).sum(axis=1)
        sq[~allowed] = np.inf
        g = int(np.argmin(sq))
        merge(t, g)

    return groups
