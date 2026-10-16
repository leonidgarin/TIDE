import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from tide import TIDE

from .conftest import make_data


def fit_quiet(t, *args):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return t.fit(*args)


# ------------------------------------------------------------------ basic behaviour


def test_fit_transform_shapes_and_attributes(data):
    X, y, per = data
    t = fit_quiet(TIDE(cat_values=[-999]), X, y, per)
    assert t.n_features_in_ == 3
    assert list(t.feature_names_in_) == ["x", "neg", "age"]
    assert t.bins_["x"]["trend"] == "pos" and t.bins_["neg"]["trend"] == "neg"
    assert len(t.periods_) == 6
    Xw = t.transform(X)
    assert list(Xw.columns) == ["x_woe", "neg_woe", "age_woe"]
    assert Xw.index.equals(X.index)
    assert Xw.notna().all().all()
    assert set(t.summary_["variable"]) == {"x", "neg", "age"}
    assert list(t.get_feature_names_out()) == ["x_woe", "neg_woe", "age_woe"]


def test_bins_format_and_specials(data):
    X, y, per = data
    t = fit_quiet(TIDE(cat_values=[-999]), X, y, per)
    atoms = {a: k for b in t.bins_["age"]["bins"].values() for a, k in b.items()}
    assert atoms["no_hit"] == "cat" and atoms[-999] == "cat" and atoms[None] == "missing"
    first = t.bins_["x"]["bins"][0]
    (a, b), kind = next(iter(first.items()))
    assert kind == "cont" and a == -np.inf


def test_stats_consistent_with_transform(data):
    X, y, per = data
    t = fit_quiet(TIDE(cat_values=[-999]), X, y, per)
    Xw = t.transform(X)
    for f in X.columns:
        st = t.stats_[t.stats_["variable"] == f]
        counts = Xw[f"{f}_woe"].round(12).value_counts()
        for _, row in st.iterrows():
            assert counts[round(row["WoE"], 12)] == row["n_obs"]
    sp = t.stats_per_
    tot = sp.groupby(["variable", "bin_idx"])["n_obs"].sum()
    st = t.stats_.set_index(["variable", "bin_idx"])["n_obs"]
    pd.testing.assert_series_equal(tot.sort_index(), st.sort_index(), check_names=False)


def test_periods_order_and_types(data):
    X, y, per = data
    t1 = fit_quiet(TIDE(), X, y, per)
    shuffled = np.random.default_rng(0).permutation(len(X))
    t2 = fit_quiet(TIDE(), X.iloc[shuffled], y[shuffled], per[shuffled])
    assert t1.bins_ == t2.bins_
    t3 = fit_quiet(TIDE(), X, y, per.strftime("%Y-%m"))
    assert t1.bins_ == t3.bins_
    assert list(t3.periods_) == sorted(set(per.strftime("%Y-%m")))


def test_no_periods_means_single_period(data):
    X, y, _ = data
    t = fit_quiet(TIDE(), X, y)
    assert list(t.periods_) == ["all"]
    assert t.summary_["max_PSI_seq"].isna().all()


def test_forced_trend(data):
    X, y, per = data
    t = fit_quiet(TIDE(trend={"x": "neg"}), X, y, per)
    assert t.bins_["x"]["trend"] == "neg"
    assert len(t.bins_["x"]["bins"]) == 1  # wrong direction: nothing monotonic to split
    assert t.bins_["neg"]["trend"] == "neg"


def test_single_bin_warning():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"noise": rng.normal(size=3000)})
    y = rng.integers(0, 2, 3000)
    per = rng.integers(0, 12, 3000)
    with pytest.warns(UserWarning, match="no split satisfies"):
        TIDE(min_sample_rate=0.3).fit(X, y, per)


def test_n_jobs_gives_same_result(data):
    X, y, per = data
    a = fit_quiet(TIDE(cat_values=[-999]), X, y, per)
    b = fit_quiet(TIDE(cat_values=[-999], n_jobs=2), X, y, per)
    assert a.bins_ == b.bins_
    pd.testing.assert_frame_equal(a.stats_per_, b.stats_per_)


def test_bracket_conventions_for_unseen_values():
    # pos: (a, b]; neg: [a, b). An unseen value between two borders goes to the
    # bin with the higher event rate in both cases.
    bins_pos = {"trend": "pos", "bins": {0: {(-np.inf, 1.0): "cont"}, 1: {(1.0, np.inf): "cont"}}}
    bins_neg = {"trend": "neg", "bins": {0: {(-np.inf, 2.0): "cont"}, 1: {(2.0, np.inf): "cont"}}}
    X = pd.DataFrame({"p": [0.0, 1.0, 1.5, 2.0] * 50, "n": [0.0, 1.0, 1.5, 2.0] * 50})
    y = np.array([0, 0, 1, 1] * 50)
    t = TIDE(user_bins={"p": bins_pos, "n": bins_neg}).fit(X, y)
    sp = t.stats_.set_index(["variable", "bin_idx"])["n_obs"]
    assert sp[("p", 0)] == 100 and sp[("p", 1)] == 100  # 1.0 is in (-inf, 1]
    assert sp[("n", 0)] == 150 and sp[("n", 1)] == 50  # 2.0 is in [2, inf)
    assert t.stats_.iloc[0]["bin"] == "(-inf,1]"
    assert t.stats_.iloc[3]["bin"] == "[2,inf)"


# ------------------------------------------------------------------ strategies


@pytest.mark.parametrize("strategy", ["separate_bin", "chi-squared", "tide"])
@pytest.mark.parametrize("missing", ["separate_bin", "worst_bin", "best_bin"])
def test_all_strategies_run(data, strategy, missing):
    X, y, per = data
    t = fit_quiet(
        TIDE(cat_values=[-999], cat_strategy=strategy, missing_strategy=missing, missing_rate=0.2),
        X,
        y,
        per,
    )
    Xw = t.transform(X)
    assert Xw.notna().all().all()


def test_tide_mode_guarantees_all_constraints():
    X, y, per = make_data(n=40000, n_per=4, seed=3)
    t = fit_quiet(TIDE(cat_values=[-999], cat_strategy="tide", min_class_obs=5), X, y, per)
    for f in X.columns:
        sp = t.stats_per_[t.stats_per_["variable"] == f]
        er = sp.pivot(index="bin_idx", columns="period", values="event_rate").to_numpy()
        n = sp.pivot(index="bin_idx", columns="period", values="n_obs").to_numpy()
        ev = sp.pivot(index="bin_idx", columns="period", values="n_events").to_numpy()
        order = np.argsort(er[:, 0])
        assert (np.diff(er[order], axis=0) >= 0).all(), f  # no crossings
        assert (n / n.sum(axis=0) >= 0.05).all()
        assert (ev >= 5).all() and (n - ev >= 5).all()


# ------------------------------------------------------------------ user bins


def test_user_bins_applied_as_is_with_coverage_warning():
    X = pd.DataFrame({"a": [1.0, 5.0, 9.0, -999, np.nan, 20.0] * 20})
    y = np.array([0, 1, 0, 1, 1, 0] * 20)
    ub = {
        "trend": "pos",
        "bins": {
            0: {(0.0, 5.0): "cont", -999: "cat"},
            1: {(5.0, 10.0): "cont"},
            2: {None: "missing"},
        },
    }
    with pytest.warns(UserWarning, match="not covered"):
        t = TIDE(user_bins={"a": ub}).fit(X, y)
    assert t.bins_["a"]["bins"] == ub["bins"]
    assert t.summary_.loc[0, "user_bins"]
    assert t.stats_["n_obs"].tolist() == [60, 20, 20]  # 1, 5 and -999 in bin 0; 20 uncovered


def test_user_bins_u_shape_allowed():
    X = pd.DataFrame({"a": np.tile(np.arange(9.0), 30)})
    y = np.tile([1, 0, 0, 0, 0, 0, 0, 0, 1], 30)
    ub = {
        "trend": "pos",
        "bins": {0: {(-np.inf, 0.0): "cont"}, 1: {(0.0, 7.0): "cont"}, 2: {(7.0, np.inf): "cont"}},
    }
    t = TIDE(user_bins={"a": ub}).fit(X, y)
    assert t.stats_["event_rate"].round(3).tolist() == [1.0, 0.0, 1.0]


@pytest.mark.parametrize(
    "bad",
    [
        {"bins": {0: {(1.0, 0.0): "cont"}}, "trend": "pos"},
        {"bins": {0: {(0.0, 2.0): "cont"}, 1: {(1.0, 3.0): "cont"}}, "trend": "pos"},
        {"bins": {0: {(0.0, 1.0): "cont"}}},
        {"bins": {0: {"a": "weird"}}},
        {"bins": {0: {"a": "cat"}, 1: {"a": "cat"}}},
    ],
)
def test_user_bins_validation(bad):
    X = pd.DataFrame({"a": [0.5, 1.5] * 5})
    with pytest.raises(ValueError):
        TIDE(user_bins={"a": bad}).fit(X, [0, 1] * 5)


# ------------------------------------------------------------------ transform & unknowns


def test_handle_unknown_options():
    X = pd.DataFrame({"a": np.r_[np.arange(100.0), np.full(20, np.nan)]})
    y = np.r_[(np.arange(100) > 60).astype(int), np.ones(20, int)]
    t = fit_quiet(TIDE(min_bound="col_min", max_bound="col_max"), X, y)
    new = pd.DataFrame({"a": [500.0, "unseen"]}, dtype=object)
    woes = t.woes_["a"]
    with pytest.warns(UserWarning, match="not covered"):
        assert t.transform(new)["a_woe"].tolist() == [500.0, "unseen"]
    t.set_params(handle_unknown="worst")
    with pytest.warns(UserWarning):
        assert t.transform(new)["a_woe"].tolist() == [min(woes.values())] * 2
    t.set_params(handle_unknown="best")
    with pytest.warns(UserWarning):
        assert t.transform(new)["a_woe"].tolist() == [max(woes.values())] * 2
    t.set_params(handle_unknown="missing_worst")
    miss_idx = next(b for b, atoms in t.bins_["a"]["bins"].items() if None in atoms)
    with pytest.warns(UserWarning):
        assert t.transform(new)["a_woe"].tolist() == [woes[miss_idx]] * 2


def test_set_params_after_pickle_changes_only_unknown_handling(data):
    X, y, per = data
    t = fit_quiet(TIDE(cat_values=[-999]), X, y, per)
    prod = pickle.loads(pickle.dumps(t))
    prod.set_params(handle_unknown="worst")
    assert prod.bins_ == t.bins_ and prod.woes_ == t.woes_
    pd.testing.assert_frame_equal(prod.transform(X), t.transform(X))
    new = X.head(3).copy()
    new["age"] = new["age"].astype(object)
    new.loc[new.index[0], "age"] = "never_seen"
    with pytest.warns(UserWarning):
        out = prod.transform(new)
    assert out["age_woe"].iloc[0] == min(t.woes_["age"].values())


def test_transform_column_checks(data):
    X, y, per = data
    t = fit_quiet(TIDE(), X, y, per)
    with pytest.raises(ValueError, match="same columns"):
        t.transform(X[["neg", "x", "age"]])
    with pytest.raises(ValueError, match="missing"):
        t.transform(X[["x", "neg"]])
    with pytest.raises(TypeError):
        t.transform(X.to_numpy())


# ------------------------------------------------------------------ validation


@pytest.mark.parametrize(
    "params",
    [
        {"min_sample_rate": 0},
        {"min_sample_rate": 1.5},
        {"min_class_obs": 0},
        {"trend": "up"},
        {"missing_strategy": "drop"},
        {"cat_strategy": "woe"},
        {"handle_unknown": "zero"},
        {"n_trend_bins": 1},
        {"max_bound": "max"},
        {"missing_values": "N/A"},
        {"cat_values": [-1], "missing_values": [-1]},
    ],
)
def test_invalid_params(data, params):
    X, y, per = data
    with pytest.raises(ValueError):
        TIDE(**params).fit(X, y, per)


def test_invalid_inputs(data):
    X, y, per = data
    with pytest.raises(TypeError):
        TIDE().fit(X.to_numpy(), y, per)
    with pytest.raises(ValueError):
        TIDE().fit(X, y[:-1], per)
    with pytest.raises(ValueError, match="not in X"):
        TIDE(user_bins={"zzz": {"bins": {0: {"a": "cat"}}}}).fit(X, y, per)


# ------------------------------------------------------------------ sklearn integration


def test_clone_and_params_roundtrip():
    t = TIDE(min_sample_rate=0.1, cat_values=[-999], trend={"a": "pos"})
    c = clone(t)
    assert c.get_params() == t.get_params()
    assert "min_sample_rate=0.1" in repr(t)


def test_not_fitted_error(data):
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        TIDE().transform(data[0])


def test_pipeline_with_periods(data):
    X, y, per = data
    pipe = Pipeline([("tide", TIDE(cat_values=[-999])), ("lr", LogisticRegression())])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pipe.fit(X, y, tide__periods=per)
    assert pipe.predict_proba(X).shape == (len(X), 2)
    pipe.set_params(tide__handle_unknown="worst")
    assert pipe.named_steps["tide"].handle_unknown == "worst"


def test_cross_validate_splits_periods(data):
    X, y, per = data
    pipe = Pipeline([("tide", TIDE(cat_values=[-999])), ("lr", LogisticRegression())])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res = cross_validate(pipe, X, y, cv=3, scoring="roc_auc", params={"tide__periods": per})
    assert (res["test_score"] > 0.7).all()


def test_metadata_routing(data):
    X, y, per = data
    with sklearn.config_context(enable_metadata_routing=True):
        pipe = Pipeline(
            [
                ("tide", TIDE(cat_values=[-999]).set_fit_request(periods=True)),
                ("lr", LogisticRegression()),
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            pipe.fit(X, y, periods=per)
    assert len(pipe.named_steps["tide"].periods_) == 6


def test_plot_returns_figures(data):
    X, y, per = data
    t = fit_quiet(TIDE(cat_values=[-999]), X, y, per)
    figs = t.plot(["x", "age"], show=False)
    assert set(figs) == {"x", "age"}
    assert len(figs["x"].axes) == 5
    legend = [t.get_text() for t in figs["x"].legends[0].get_texts()]
    assert legend[0].startswith("0: (-inf,") and legend[0].endswith("]")
    bound = legend[0].split(",")[1].rstrip("]")
    assert len(bound.split(".")[1]) == 4  # exactly 4 decimals
    with pytest.raises(ValueError):
        t.plot("nope", show=False)


def test_extreme_woe_warning():
    X = pd.DataFrame({"a": np.r_[np.arange(1000.0), [np.nan] * 50]})
    y = (X["a"].fillna(2000) > 600).astype(int)
    with pytest.warns(UserWarning, match="no non-events"):
        t = TIDE().fit(X, y)
    assert "no non-events" in t.summary_.loc[0, "warnings"]
    assert np.isfinite(t.summary_.loc[0, "IV"]) and np.isfinite(t.stats_["WoE"]).all()


def test_bin_labels_display_format():
    from tide._bins import bin_label

    assert bin_label({(-np.inf, 25.0): "cont"}, "pos") == "(-inf,25]"
    assert bin_label({(18.0, 25.0): "cont", None: "missing"}, "neg") == "[18,25); NaN"
    assert bin_label({(0.5, 1.123456): "cont"}, "pos") == "(0.5000,1.1235]"
    assert bin_label({(2.0, np.inf): "cont", "no_hit": "cat"}, "pos") == "(2,inf); no_hit"


def test_labels_rounded_but_bins_keep_full_values(data):
    X, y, per = data
    t = fit_quiet(TIDE(), X[["x"]], y, per)
    (a, b) = next(iter(t.bins_["x"]["bins"][0]))
    assert b != round(b, 4)  # full precision kept in bins_
    assert t.stats_["bin"].iloc[0] == f"(-inf,{b:.4f}]"
    assert (t.transform(X[["x"]])["x_woe"] == t.woes_["x"][0]).sum() == t.stats_["n_obs"].iloc[0]
