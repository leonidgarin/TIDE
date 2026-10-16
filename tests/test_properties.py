"""Property-based tests: the hard constraints hold for any data and parameters."""

import warnings

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tide import TIDE


@st.composite
def scenarios(draw):
    seed = draw(st.integers(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    n = draw(st.integers(500, 6000))
    n_per = draw(st.integers(1, 8))
    n_unique = draw(st.sampled_from([5, 50, 10**6]))
    slope = draw(st.floats(-3, 3))
    drift = draw(st.floats(-0.3, 0.3))
    miss_share = draw(st.sampled_from([0.0, 0.03, 0.2]))
    per = rng.integers(0, n_per, n)
    x = rng.normal(size=n)
    if n_unique < 10**6:
        x = np.round(x * n_unique / 6)
    logit = -1.5 + slope * x / max(1.0, n_unique / 6 if n_unique < 10**6 else 1.0) + drift * per
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    x = x.astype(object)
    x[rng.random(n) < miss_share] = np.nan
    params = dict(
        min_sample_rate=draw(st.sampled_from([0.02, 0.05, 0.1])),
        min_class_obs=draw(st.sampled_from([1, 3, 10])),
        min_er_delta=draw(st.sampled_from([0.0, 0.0, 0.01, 0.05])),
        missing_strategy=draw(st.sampled_from(["separate_bin", "worst_bin", "best_bin"])),
        missing_rate=0.1,
        max_unique=draw(st.sampled_from([10, 10**6])),
        n_quantiles=draw(st.sampled_from([5, 20])),
    )
    return pd.DataFrame({"x": x}), y, per, params


def _tables(t):
    sp = t.stats_per_
    n = sp.pivot(index="bin_idx", columns="period", values="n_obs").to_numpy()
    e = sp.pivot(index="bin_idx", columns="period", values="n_events").to_numpy()
    return n, e, e / (n + 1e-8)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(scenarios())
def test_continuous_constraints_hold(sc):
    X, y, per, params = sc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        t = TIDE(**params).fit(X, y, per)
    bins = t.bins_["x"]["bins"]
    cont_ids = [b for b, atoms in bins.items() if "cont" in atoms.values()]
    if len(cont_ids) < 2:
        return
    n, e, er = _tables(t)
    n, e, er = n[cont_ids], e[cont_ids], er[cont_ids]
    period_total = np.bincount(pd.factorize(per, sort=True)[0])
    sign = 1 if t.bins_["x"]["trend"] == "pos" else -1
    d = params["min_er_delta"]
    assert (np.diff(er, axis=0) * sign >= d - 1e-9).all()  # monotonic + gap, every period
    assert (n / period_total >= params["min_sample_rate"]).all()
    assert (e >= params["min_class_obs"]).all()
    assert (n - e >= params["min_class_obs"]).all()


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(scenarios(), st.integers(2, 6))
def test_tide_mode_constraints_hold(sc, n_cats):
    X, y, per, params = sc
    rng = np.random.default_rng(len(y))
    x = X["x"].to_numpy(dtype=object).copy()
    cat_rows = rng.random(len(x)) < 0.15
    x[cat_rows] = np.array([f"c{i}" for i in range(n_cats)])[
        rng.integers(0, n_cats, cat_rows.sum())
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        t = TIDE(cat_strategy="tide", **params).fit(pd.DataFrame({"x": x}), y, per)
    n, e, er = _tables(t)
    if n.shape[0] < 2:
        return
    # no crossings: every pair keeps its order (ties allowed only if constant)
    for i in range(er.shape[0]):
        for j in range(i + 1, er.shape[0]):
            s = np.sign(er[i] - er[j])
            assert (s == s[0]).all()
    total = n.sum(axis=0)
    assert (n / total >= params["min_sample_rate"]).all()
    assert (e >= params["min_class_obs"]).all() and (n - e >= params["min_class_obs"]).all()
