"""The TIDE estimator."""

from __future__ import annotations

import math
import numbers
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._bins import compile_bins, special_keys, validate_bins
from ._fit import FitParams, fit_feature, fit_user_bins
from ._parse import check_target, encode_periods, normalize_key, parse_column
from ._stats import STATS_COLUMNS, STATS_PER_COLUMNS, SUMMARY_COLUMNS, feature_tables

__all__ = ["TIDE"]

HANDLE_UNKNOWN = ("keep", "worst", "best", "missing_worst", "missing_best")


class TIDE(TransformerMixin, BaseEstimator):
    """TIDE - Time Invariant Discretization Engine.

    Bins mixed-type features (numbers, strings, missing values) so that the bins are
    stable over time, and encodes them with Weight of Evidence. For the continuous part
    the following hard constraints hold in **every** period:

    - the event rate is monotonic over the bins, in the same direction in all periods,
      so event-rate (WoE) curves never cross;
    - adjacent bins differ in event rate by at least ``min_er_delta``;
    - every bin holds at least ``min_sample_rate`` of the period's rows and at least
      ``min_class_obs`` events and non-events.

    By default, categories and missing values get their own bins. With
    ``cat_strategy="tide"``, all bins (continuous, categorical and missing) are merged
    until the constraints hold for all of them.

    Parameters
    ----------
    min_sample_rate : float, default=0.05
        Minimum share of a period's rows in each bin, checked in every period.
    min_class_obs : int, default=1
        Minimum number of events and of non-events in each bin, in every period.
    min_er_delta : float, default=0.0
        Minimum event-rate gap between adjacent continuous bins, in every period.
    trend : {"auto", "pos", "neg"} or dict, default="auto"
        Direction of the event rate over the continuous bins. ``"auto"`` uses the sign
        of a linear fit of event rate vs. quantile-bin index. A dict maps feature names
        to ``"pos"`` / ``"neg"`` / ``"auto"``; features that are not listed use ``"auto"``.
    max_unique : int, default=1000
        If the continuous part has more unique values than this, the candidate split
        points are ``n_quantiles`` quantiles instead of all unique values.
    n_quantiles : int, default=200
        Number of quantile candidate points used when ``max_unique`` is exceeded.
    n_trend_bins : "log" or int, default="log"
        Number of quantile bins for trend detection. ``"log"`` means
        ``max(3, ceil(ln(n_rows)))``.
    min_bound, max_bound : float or "col_min" / "col_max", default=-inf / inf
        Outer bounds of the first and last continuous bins. ``"col_min"`` / ``"col_max"``
        use the training minimum / maximum, so unseen values beyond them are unknown.
    missing_values : list or dict, default=()
        Extra values treated as missing, e.g. ``[-1, "N/A"]``. NaN, None, pd.NA and NaT
        are always missing. A dict maps feature names to lists.
    cat_values : list or dict, default=()
        Numeric values treated as categories, e.g. ``[-999]``. Strings and bools are
        always categories. A dict maps feature names to lists.
    missing_strategy : {"separate_bin", "worst_bin", "best_bin"}, default="separate_bin"
        ``"separate_bin"``: missing values get their own bin (if any are present).
        ``"worst_bin"`` / ``"best_bin"``: if the missing share is below ``missing_rate``,
        missing values join the continuous bin with the highest / lowest event rate and
        take part in the scan; otherwise they get their own bin.
    missing_rate : float, default=0.05
        Threshold for ``"worst_bin"`` / ``"best_bin"``.
    cat_strategy : {"separate_bin", "chi-squared", "tide"}, default="separate_bin"
        ``"separate_bin"``: every category is its own bin; curves may cross.
        ``"chi-squared"``: categories are merged with ChiMerge.
        ``"tide"``: all bins are merged until every constraint holds for all bins.
    alpha_significance : float, default=0.05
        p-value threshold for ChiMerge.
    user_bins : dict or None, default=None
        ``{feature: bins_dict}``. These features are not searched; their bins are
        applied as given. A warning lists training values that fall in no bin.
    handle_unknown : {"keep", "worst", "best", "missing_worst", "missing_best"}, default="keep"
        How ``transform`` treats values that fall in no bin. ``"keep"`` leaves the raw
        value, ``"worst"`` / ``"best"`` use the lowest / highest WoE, and
        ``"missing_worst"`` / ``"missing_best"`` use the WoE of the bin with missing
        values, else the lowest / highest WoE. It is read only in ``transform``, so it can
        be changed with ``set_params`` after fitting.
    n_jobs : int or None, default=None
        Number of parallel jobs over features (joblib). ``-1`` uses all cores.
    verbose : bool, default=False
        Show a progress bar during ``fit``.

    Attributes
    ----------
    bins_ : dict
        ``{feature: bins_dict}``.
    woes_ : dict
        ``{feature: {bin_idx: WoE}}``.
    stats_ : pd.DataFrame
        Per feature and bin: counts, event rate, WoE and IV.
    stats_per_ : pd.DataFrame
        Per period, feature and bin: counts, event rate and PSI.
    summary_ : pd.DataFrame
        One row per feature: trend, number of bins, IV, max PSI and warnings.
    periods_ : pd.Index
        Sorted unique periods seen in ``fit``.
    feature_names_in_ : np.ndarray
        Feature names seen in ``fit``.
    n_features_in_ : int
        Number of features seen in ``fit``.
    """

    def __init__(
        self,
        min_sample_rate=0.05,
        min_class_obs=1,
        min_er_delta=0.0,
        trend="auto",
        max_unique=1000,
        n_quantiles=200,
        n_trend_bins="log",
        min_bound=-np.inf,
        max_bound=np.inf,
        missing_values=(),
        cat_values=(),
        missing_strategy="separate_bin",
        missing_rate=0.05,
        cat_strategy="separate_bin",
        alpha_significance=0.05,
        user_bins=None,
        handle_unknown="keep",
        n_jobs=None,
        verbose=False,
    ):
        self.min_sample_rate = min_sample_rate
        self.min_class_obs = min_class_obs
        self.min_er_delta = min_er_delta
        self.trend = trend
        self.max_unique = max_unique
        self.n_quantiles = n_quantiles
        self.n_trend_bins = n_trend_bins
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.missing_values = missing_values
        self.cat_values = cat_values
        self.missing_strategy = missing_strategy
        self.missing_rate = missing_rate
        self.cat_strategy = cat_strategy
        self.alpha_significance = alpha_significance
        self.user_bins = user_bins
        self.handle_unknown = handle_unknown
        self.n_jobs = n_jobs
        self.verbose = verbose

    # ------------------------------------------------------------------ validation
    def _validate_params(self) -> None:
        def real(name, lo, hi, lo_open=True, hi_open=True):
            v = getattr(self, name)
            if not isinstance(v, numbers.Real) or isinstance(v, bool) or np.isnan(v):
                raise ValueError(f"{name} must be a number, got {v!r}")
            if (v <= lo if lo_open else v < lo) or (v >= hi if hi_open else v > hi):
                raise ValueError(f"{name}={v!r} is out of range")

        def integer(name, lo):
            v = getattr(self, name)
            if not isinstance(v, numbers.Integral) or isinstance(v, bool) or v < lo:
                raise ValueError(f"{name} must be an int >= {lo}, got {v!r}")

        def choice(name, options):
            if getattr(self, name) not in options:
                raise ValueError(f"{name} must be one of {options}, got {getattr(self, name)!r}")

        real("min_sample_rate", 0.0, 1.0)
        integer("min_class_obs", 1)
        real("min_er_delta", 0.0, 1.0, lo_open=False, hi_open=False)
        integer("max_unique", 1)
        integer("n_quantiles", 2)
        if self.n_trend_bins != "log":
            integer("n_trend_bins", 2)
        if self.min_bound != "col_min":
            real("min_bound", -np.inf, np.inf, lo_open=False)
        if self.max_bound != "col_max":
            real("max_bound", -np.inf, np.inf, hi_open=False)
        choice("missing_strategy", ("separate_bin", "worst_bin", "best_bin"))
        real("missing_rate", 0.0, 1.0, hi_open=False)
        choice("cat_strategy", ("separate_bin", "chi-squared", "tide"))
        real("alpha_significance", 0.0, 0.5)
        choice("handle_unknown", HANDLE_UNKNOWN)
        if isinstance(self.trend, dict):
            for k, v in self.trend.items():
                if v not in ("auto", "pos", "neg"):
                    raise ValueError(f"trend[{k!r}] must be 'auto', 'pos' or 'neg', got {v!r}")
        else:
            choice("trend", ("auto", "pos", "neg"))
        for name in ("missing_values", "cat_values"):
            v = getattr(self, name)
            if isinstance(v, (str, bytes)) or not (isinstance(v, dict) or np.iterable(v)):
                raise ValueError(f"{name} must be a list or a dict of lists, got {v!r}")
        if self.user_bins is not None and not isinstance(self.user_bins, dict):
            raise ValueError("user_bins must be a dict {feature: bins_dict} or None")

    @staticmethod
    def _per_feature(param, name):
        if isinstance(param, dict):
            return param.get(name, ())
        return param

    def _specials(self, name, user_bins):
        cat = [normalize_key(v) for v in self._per_feature(self.cat_values, name)]
        miss = [normalize_key(v) for v in self._per_feature(self.missing_values, name)]
        if user_bins is not None:
            ucat, umiss = special_keys(user_bins)
            cat += [v for v in ucat if v not in cat]
            miss += [v for v in umiss if v not in miss]
        both = [v for v in cat if v in miss and v is not None]
        if both:
            raise ValueError(f"Feature {name!r}: {both} are in both cat_values and missing_values")
        return tuple(cat), tuple(miss)

    # ------------------------------------------------------------------ fit
    def fit(self, X, y, periods=None):
        """Fit the bins.

        Parameters
        ----------
        X : pd.DataFrame
            Features; columns may be mixed-type (numbers, strings, missing values).
        y : array-like of shape (n_samples,)
            Binary target (0/1 or bool).
        periods : array-like of shape (n_samples,) or None
            Period of each row (str, datetime, pd.Timestamp, number, ...). It does not
            need to be sorted; its sort order defines the time order. ``None`` means one
            period.

        Returns
        -------
        self
        """
        self._validate_params()
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X).__name__}")
        if X.columns.has_duplicates:
            raise ValueError("X has duplicate column names")
        n = X.shape[0]
        if n == 0 or X.shape[1] == 0:
            raise ValueError("X is empty")
        y_arr = check_target(y, n)
        p_codes, periods_idx = encode_periods(periods, n)
        n_per = len(periods_idx)
        per_size = np.bincount(p_codes, minlength=n_per)

        user_bins = {}
        for name, b in (self.user_bins or {}).items():
            if name not in X.columns:
                raise ValueError(f"user_bins has feature {name!r}, which is not in X")
            user_bins[name] = validate_bins(b, name)

        prm = FitParams(
            min_sample_rate=float(self.min_sample_rate),
            min_class_obs=int(self.min_class_obs),
            min_er_delta=float(self.min_er_delta),
            max_unique=int(self.max_unique),
            n_quantiles=int(self.n_quantiles),
            n_trend_bins=(
                max(3, math.ceil(math.log(n)))
                if self.n_trend_bins == "log"
                else int(self.n_trend_bins)
            ),
            min_bound=self.min_bound,
            max_bound=self.max_bound,
            missing_strategy=self.missing_strategy,
            missing_rate=float(self.missing_rate),
            cat_strategy=self.cat_strategy,
            alpha_significance=float(self.alpha_significance),
        )

        specials = {name: self._specials(name, user_bins.get(name)) for name in X.columns}

        def tasks():
            for name in X.columns:
                raw = _column_payload(X[name])
                cat, miss = specials[name]
                if name in user_bins:
                    yield delayed(fit_user_bins)(
                        name, raw, user_bins[name], y_arr, p_codes, n_per, cat, miss
                    )
                else:
                    trend = self._per_feature(self.trend, name)
                    trend = trend if trend in ("pos", "neg") else "auto"
                    yield delayed(fit_feature)(
                        name, raw, y_arr, p_codes, n_per, per_size, trend, cat, miss, prm
                    )

        results = Parallel(n_jobs=self.n_jobs, return_as="generator")(tasks())
        if self.verbose:
            from tqdm.auto import tqdm

            results = tqdm(results, total=X.shape[1], desc="TIDE fit")

        bins, woes, stats, stats_per, summary = {}, {}, [], [], []
        for fit in results:
            s, sp, sm, wo = feature_tables(fit, periods_idx)
            for w in fit.warnings:
                warnings.warn(w, UserWarning, stacklevel=2)
            bins[fit.name] = fit.bins
            woes[fit.name] = wo
            stats.append(s)
            stats_per.append(sp)
            summary.append(sm)

        self.bins_ = bins
        self.woes_ = woes
        self.stats_ = pd.concat(stats, ignore_index=True)[STATS_COLUMNS]
        self.stats_per_ = (
            pd.concat(stats_per, ignore_index=True)[STATS_PER_COLUMNS]
            .sort_values("period", kind="stable", key=_period_key)
            .reset_index(drop=True)
        )
        self.summary_ = pd.DataFrame(summary, columns=SUMMARY_COLUMNS)
        self.periods_ = periods_idx
        self.specials_ = specials
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    # ------------------------------------------------------------------ transform
    def transform(self, X):
        """Replace every value with the WoE of its bin.

        Parameters
        ----------
        X : pd.DataFrame
            Must have the same columns as in ``fit``, in the same order.

        Returns
        -------
        pd.DataFrame
            Columns ``"{feature}_woe"``, same index as ``X``.
        """
        check_is_fitted(self, "bins_")
        if self.handle_unknown not in HANDLE_UNKNOWN:
            raise ValueError(f"handle_unknown must be one of {HANDLE_UNKNOWN}")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X).__name__}")
        cols = list(X.columns)
        if cols != list(self.feature_names_in_):
            missing = [c for c in self.feature_names_in_ if c not in cols]
            extra = [c for c in cols if c not in set(self.feature_names_in_)]
            raise ValueError(
                "X must have the same columns as in fit, in the same order"
                + (f"; missing: {missing}" if missing else "")
                + (f"; unexpected: {extra}" if extra else "")
            )
        out = {}
        for name in cols:
            out[f"{name}_woe"] = self._transform_one(X[name], name)
        return pd.DataFrame(out, index=X.index)

    def _transform_one(self, col: pd.Series, name):
        bins = self.bins_[name]
        cat, miss = self.specials_[name]
        parsed = parse_column(col, cat, miss)
        comp = compile_bins(bins, name)
        codes = comp.assign(parsed)
        woe_by_pos = np.array([self.woes_[name][b] for b in comp.bin_ids], dtype=np.float64)
        res = np.where(codes >= 0, woe_by_pos[np.maximum(codes, 0)], np.nan)
        unknown = codes < 0
        if not unknown.any():
            return res
        how = self.handle_unknown
        if how == "keep":
            res = res.astype(object)
            res[unknown] = col.to_numpy(dtype=object)[unknown]
        else:
            fill = None
            if how in ("missing_worst", "missing_best"):
                for pos, b in enumerate(comp.bin_ids):
                    if any(k == "missing" for k in bins["bins"][b].values()):
                        fill = woe_by_pos[pos]
                        break
            if fill is None:
                fill = np.nanmin(woe_by_pos) if how.endswith("worst") else np.nanmax(woe_by_pos)
            res[unknown] = fill
        warnings.warn(
            f'There are values in variable "{name}" that are not covered by fitted bins and '
            f'treated as "{how}" by WoE.',
            UserWarning,
            stacklevel=3,
        )
        return res

    def get_feature_names_out(self, input_features=None):
        """Output column names: ``"{feature}_woe"``."""
        check_is_fitted(self, "bins_")
        return np.asarray([f"{n}_woe" for n in self.feature_names_in_], dtype=object)

    # ------------------------------------------------------------------ plots
    def plot(self, features=None, figsize=(15, 6), dpi=120, psi_threshold=0.35, show=True):
        """Plot binning results: bin sizes and event rates, event rate over time, PSI.

        Parameters
        ----------
        features : str, list of str or None
            Features to plot; ``None`` plots all.
        figsize, dpi : matplotlib figure settings.
        psi_threshold : float
            Horizontal reference line on the PSI panel.
        show : bool
            Call ``matplotlib.pyplot.show()``.

        Returns
        -------
        dict
            ``{feature: matplotlib.figure.Figure}``.
        """
        from .plotting import plot_feature

        check_is_fitted(self, "bins_")
        if features is None:
            features = list(self.feature_names_in_)
        elif isinstance(features, str) or not np.iterable(features):
            features = [features]
        for f in features:
            if f not in self.bins_:
                raise ValueError(f"Feature {f!r} is not fitted")
        figs = {
            f: plot_feature(self, f, figsize=figsize, dpi=dpi, psi_threshold=psi_threshold)
            for f in features
        }
        if show:
            import matplotlib.pyplot as plt

            plt.show()
        return figs

    # ------------------------------------------------------------------ sklearn tags
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True
        tags.input_tags.categorical = True
        tags.target_tags.required = True
        return tags

    def _more_tags(self):  # scikit-learn < 1.6
        return {"allow_nan": True, "requires_y": True, "X_types": ["2darray", "string"]}


def _column_payload(s: pd.Series):
    """Column as a numpy array that joblib can memory-map (numeric) or pickle (object)."""
    dt = s.dtype
    if pd.api.types.is_numeric_dtype(dt) and not pd.api.types.is_bool_dtype(dt):
        return s.to_numpy(dtype=np.float64, na_value=np.nan)
    return s.to_numpy(dtype=object)


def _period_key(col: pd.Series):
    # stable sort by period keeps feature order, then bin order, within a period
    return pd.Series(pd.factorize(col, sort=True)[0], index=col.index)
