"""Stats tables built from per-period bin counts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._bins import bin_label
from ._fit import FeatureFit
from ._greedy import EPS
from ._metrics import event_rates, psi, woe_iv

STATS_COLUMNS = [
    "variable",
    "trend",
    "bin_idx",
    "bin",
    "n_obs",
    "sample_rate",
    "n_events",
    "event_rate",
    "WoE",
    "IV_contrib",
    "IV_total",
]
STATS_PER_COLUMNS = [
    "period",
    "variable",
    "trend",
    "bin_idx",
    "bin",
    "n_obs",
    "sample_rate",
    "n_events",
    "event_rate",
    "PSI_seq",
    "PSI_base",
]
SUMMARY_COLUMNS = [
    "variable",
    "trend",
    "n_bins",
    "n_cont_bins",
    "IV",
    "max_PSI_seq",
    "max_PSI_base",
    "user_bins",
    "undersampled",
    "warnings",
]


def _nanmax(a: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    return float(a.max()) if a.size else np.nan


def feature_tables(fit: FeatureFit, periods: pd.Index):
    """Return (stats, stats_per, summary_row, woes) for one fitted feature."""
    trend = fit.bins["trend"]
    idx = sorted(fit.bins["bins"])
    labels = [bin_label(fit.bins["bins"][i], trend) for i in idx]
    c, e = fit.counts, fit.events
    tot_c, tot_e = c.sum(axis=1), e.sum(axis=1)
    woe, iv = woe_iv(tot_c, tot_e)
    iv_total = float(np.nansum(iv))
    no_nonev = [b for b, c_, e_ in zip(idx, tot_c, tot_e) if c_ > 0 and c_ == e_]
    no_ev = [b for b, c_, e_ in zip(idx, tot_c, tot_e) if c_ > 0 and e_ == 0]
    empty = [b for b, c_ in zip(idx, tot_c) if c_ == 0]
    if no_nonev:
        fit.warnings.append(
            f"Feature {fit.name!r}: bins {no_nonev} have no non-events: WoE = ln(1e-8 / %events) "
            "is finite but extreme and inflates IV."
        )
    if no_ev:
        fit.warnings.append(
            f"Feature {fit.name!r}: bins {no_ev} have no events: WoE = ln(%nonevents / 1e-8) is "
            "finite but extreme and inflates IV."
        )
    if empty:
        fit.warnings.append(
            f"Feature {fit.name!r}: bins {empty} are empty: WoE = 0, IV contribution 0."
        )
    n_bins = len(idx)

    stats = pd.DataFrame(
        {
            "variable": fit.name,
            "trend": trend,
            "bin_idx": idx,
            "bin": labels,
            "n_obs": tot_c,
            "sample_rate": tot_c / (EPS + tot_c.sum()),
            "n_events": tot_e,
            "event_rate": event_rates(tot_c, tot_e),
            "WoE": woe,
            "IV_contrib": iv,
            "IV_total": iv_total,
        },
        columns=STATS_COLUMNS,
    )

    psi_seq = psi(c, "sequential")
    psi_base = psi(c, "on_base")
    n_per = len(periods)
    stats_per = pd.DataFrame(
        {
            "period": np.repeat(np.asarray(periods, dtype=object), n_bins),
            "variable": fit.name,
            "trend": trend,
            "bin_idx": np.tile(idx, n_per),
            "bin": np.tile(np.asarray(labels, dtype=object), n_per),
            "n_obs": c.T.ravel(),
            "sample_rate": (c / (EPS + c.sum(axis=0, keepdims=True))).T.ravel(),
            "n_events": e.T.ravel(),
            "event_rate": event_rates(c, e).T.ravel(),
            "PSI_seq": np.repeat(psi_seq, n_bins),
            "PSI_base": np.repeat(psi_base, n_bins),
        },
        columns=STATS_PER_COLUMNS,
    )

    summary = {
        "variable": fit.name,
        "trend": trend,
        "n_bins": n_bins,
        "n_cont_bins": fit.n_cont_bins,
        "IV": iv_total,
        "max_PSI_seq": _nanmax(psi_seq),
        "max_PSI_base": _nanmax(psi_base),
        "user_bins": fit.user,
        "undersampled": fit.undersampled,
        "warnings": " | ".join(fit.warnings),
    }
    return stats, stats_per, summary, dict(zip(idx, woe.tolist()))
