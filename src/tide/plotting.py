"""Matplotlib plots of binning results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from ._bins import bin_label

__all__ = ["period_label", "plot_feature"]


def period_label(p) -> str:
    """Short text for a period: dates without a time part print as YYYY-MM-DD."""
    if isinstance(p, pd.Timestamp) and p == p.normalize():
        return p.strftime("%Y-%m-%d")
    return str(p)


def plot_feature(binner, name, figsize=(15, 6), dpi=120, psi_threshold=0.35):
    """Three panels for one feature: bin stats, event rate over time, PSI over time."""
    st = binner.stats_[binner.stats_["variable"] == name].reset_index(drop=True)
    sp = binner.stats_per_[binner.stats_per_["variable"] == name]
    periods = [period_label(p) for p in binner.periods_]
    bins = binner.bins_[name]
    labels = [bin_label(bins["bins"][b], bins["trend"]) for b in st["bin_idx"]]
    x = np.arange(len(labels))
    xp = np.arange(len(periods))
    label_periods = len(periods) <= 12  # per-period number labels overlap beyond this
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = [cycle[i % len(cycle)] for i in range(len(labels))]

    with plt.rc_context({"font.size": 8}):
        fig, ax = plt.subplots(1, 3, figsize=figsize, dpi=dpi, layout="constrained")
        fig.suptitle(
            f"Binning results for {name}\n"
            f"IV={st['IV_total'].iloc[0]:.4f}, trend={st['trend'].iloc[0]}"
        )

        # 1. bin sizes and event rates on the whole sample
        a0 = ax[0]
        a0.set_title("Bin stats")
        a0.bar(x, st["n_obs"], color=colors, alpha=0.35, label="n_obs (% total)")
        for i, row in st.iterrows():
            a0.annotate(
                f"{row['n_obs']:,}\n({row['sample_rate']:.1%})",
                (i, row["n_obs"] / 2),
                ha="center",
                va="center",
                rotation=90,
                color="dimgray",
            )
        a0.set_xticks(x, [str(i) for i in st["bin_idx"]])
        a0.set_xlabel("Bin")
        a0.set_ylabel("Bin size")
        a0b = a0.twinx()
        a0b.plot(x, st["event_rate"], color="darkblue", alpha=0.7, marker="o", label="event_rate")
        for i, row in st.iterrows():
            a0b.annotate(
                f"{row['event_rate']:.2%}",
                (i, row["event_rate"]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                color="darkblue",
            )
        a0b.yaxis.set_major_formatter(PercentFormatter(1.0))
        a0b.set_ylabel("Event rate")
        h1, l1 = a0.get_legend_handles_labels()
        h2, l2 = a0b.get_legend_handles_labels()
        a0.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7)

        # 2. event rate of each bin over time
        a1 = ax[1]
        for b, color in zip(st["bin_idx"], colors):
            rows = sp[sp["bin_idx"] == b]
            a1.plot(xp, rows["event_rate"].to_numpy(), marker=".", color=color)
        a1.set_title("Event rate stability over time")
        a1.set_xlabel("Period")
        a1.set_ylabel("Event rate")
        a1.set_xticks(xp, periods, rotation=90)
        a1.yaxis.set_major_formatter(PercentFormatter(1.0))

        # 3. bin shares per period and PSI
        a2 = ax[2]
        bottom = np.zeros(len(periods))
        for b, color in zip(st["bin_idx"], colors):
            h = sp[sp["bin_idx"] == b]["sample_rate"].to_numpy()
            a2.bar(xp, h, width=0.95, bottom=bottom, color=color, alpha=0.35)
            for i, (v, mid) in enumerate(zip(h, bottom + h / 2)):
                if label_periods and v >= 0.04:
                    a2.annotate(f"{v:.1%}", (i, mid), color="dimgray", ha="center", va="center")
            bottom = bottom + h
        a2.set_title("Bin shares and Population Stability Index")
        a2.set_xlabel("Period")
        a2.set_ylabel("Bin size, % of rows in period")
        a2.set_xticks(xp, periods, rotation=90)
        a2.set_ylim(0, 1)
        a2.yaxis.set_major_formatter(PercentFormatter(1.0))
        a2b = a2.twinx()
        a2b.axhline(
            psi_threshold, color="red", ls="--", alpha=0.6, label=f"threshold={psi_threshold:g}"
        )
        first = sp[sp["bin_idx"] == st["bin_idx"].iloc[0]]
        for col, color, dy in (("PSI_seq", "darkgreen", 5), ("PSI_base", "darkblue", -10)):
            vals = first[col].to_numpy()
            a2b.plot(xp, vals, marker=".", color=color, alpha=0.7, label=col)
            for i, v in enumerate(vals):
                if label_periods and np.isfinite(v):
                    a2b.annotate(
                        f"{v:.3f}",
                        (i, v),
                        textcoords="offset points",
                        xytext=(0, dy),
                        ha="center",
                        color=color,
                        fontsize=6,
                    )
        psi_vals = np.r_[first["PSI_seq"].to_numpy(), first["PSI_base"].to_numpy()]
        psi_vals = psi_vals[np.isfinite(psi_vals)]
        top = max(psi_threshold, psi_vals.max() if psi_vals.size else 0.0)
        a2b.set_ylim(0, top * 1.25)
        a2b.set_ylabel("PSI")
        a2b.legend(loc="upper right", fontsize=7)

        # one legend for the bins, shared by all panels
        handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.6) for c in colors]
        fig.legend(
            handles,
            [f"{i}: {lab}" for i, lab in zip(st["bin_idx"], labels)],
            title="Bins",
            loc="outside lower center",
            ncols=min(len(labels), 4),
            fontsize=7,
        )
    return fig
