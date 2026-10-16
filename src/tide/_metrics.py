"""Weight of Evidence, Information Value and Population Stability Index (v0.5 formulas)."""

from __future__ import annotations

import numpy as np

from ._greedy import EPS

__all__ = ["event_rates", "psi", "woe_iv"]


def event_rates(counts: np.ndarray, events: np.ndarray) -> np.ndarray:
    """``events / (counts + eps)``."""
    return events / (counts + EPS)


def woe_iv(counts: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """WoE and IV contributions per bin.

    ``WoE = ln((%nonevents + eps) / (%events + eps))`` and
    ``IV_contrib = (%nonevents - %events) * WoE``,
    where ``%events = events_bin / events_all`` and likewise for non-events. The symmetric
    ``eps`` keeps WoE finite when a bin has no events or no non-events.
    """
    counts = np.asarray(counts, dtype=np.float64)
    events = np.asarray(events, dtype=np.float64)
    nonevents = counts - events
    with np.errstate(divide="ignore", invalid="ignore"):
        ev_r = events / events.sum()
        nev_r = nonevents / nonevents.sum()
        woe = np.log((nev_r + EPS) / (ev_r + EPS))
        iv = (nev_r - ev_r) * woe
    return woe, iv


def psi(counts_per: np.ndarray, how: str = "sequential") -> np.ndarray:
    """PSI per period from bin counts of shape (n_bins, n_periods).

    ``PSI = sum((%new - %old) * ln(%new / (%old + eps)))``, where ``%`` is the bin's
    share within the period. With ``how="sequential"`` each period is compared with the
    previous one; with ``"on_base"`` every period is compared with the first.
    The first value is NaN.
    """
    c = np.asarray(counts_per, dtype=np.float64)
    shares = c / (c.sum(axis=0, keepdims=True) + EPS)
    n_per = c.shape[1]
    out = np.full(n_per, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        for t in range(1, n_per):
            old = shares[:, t - 1] if how == "sequential" else shares[:, 0]
            new = shares[:, t]
            out[t] = np.sum((new - old) * np.log(new / (old + EPS)))
    return out
