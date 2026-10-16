import numpy as np
import pytest
import scipy.stats as ss

from tide._categorical import chi2_pvalues, chi_merge, tide_merge
from tide._fit import sorted_unique
from tide._greedy import build_axis, detect_trend, greedy_cuts, weighted_percentile
from tide._metrics import psi, woe_iv

# ------------------------------------------------------------------ metrics (ported v0.5 tests)


def test_woe_iv_formulas():
    rng = np.random.default_rng(123)
    x = rng.random(1000)
    y = np.where(x < 0.5, rng.random(1000) < 0.8, rng.random(1000) < 0.2).astype(int)
    c = np.array([(x < 0.5).sum(), (x >= 0.5).sum()])
    e = np.array([y[x < 0.5].sum(), y[x >= 0.5].sum()])
    woe, iv = woe_iv(c, e)
    ev_r, nev_r = e / e.sum(), (c - e) / (c - e).sum()
    np.testing.assert_allclose(woe, np.log((nev_r + 1e-8) / (ev_r + 1e-8)))
    np.testing.assert_allclose(iv, (nev_r - ev_r) * np.log((nev_r + 1e-8) / (ev_r + 1e-8)))


def test_woe_iv_finite_for_pure_and_empty_bins():
    woe, iv = woe_iv(np.array([1000, 1000, 50, 50, 0]), np.array([100, 200, 50, 0, 0]))
    assert np.isfinite(woe).all() and np.isfinite(iv).all()
    assert woe[2] < -10 and woe[3] > 10  # only events / only non-events: extreme
    assert woe[4] == 0 and iv[4] == 0  # empty bin


def test_psi_sequential_and_base():
    c = np.array([[50, 40, 30], [50, 60, 70]])
    s = c / c.sum(axis=0)
    seq, base = psi(c, "sequential"), psi(c, "on_base")
    assert np.isnan(seq[0]) and np.isnan(base[0])
    exp1 = np.sum((s[:, 1] - s[:, 0]) * np.log(s[:, 1] / (s[:, 0] + 1e-8)))
    exp2 = np.sum((s[:, 2] - s[:, 1]) * np.log(s[:, 2] / (s[:, 1] + 1e-8)))
    exp2b = np.sum((s[:, 2] - s[:, 0]) * np.log(s[:, 2] / (s[:, 0] + 1e-8)))
    np.testing.assert_allclose(seq[1:], [exp1, exp2], rtol=1e-6)
    np.testing.assert_allclose(base[1:], [exp1, exp2b], rtol=1e-6)


# ------------------------------------------------------------------ chi-squared


@pytest.mark.parametrize("seed", range(5))
def test_chi2_matches_scipy(seed):
    rng = np.random.default_rng(seed)
    t = rng.integers(1, 200, size=(50, 4))
    p = chi2_pvalues(t[:, 0], t[:, 1], t[:, 2], t[:, 3])
    for row, pv in zip(t, p):
        ref = ss.chi2_contingency([[row[0], row[1]], [row[2], row[3]]])[1]
        assert pv == pytest.approx(ref, rel=1e-10)


def test_chi2_zero_rules():
    # two zero cells, not perfectly separated -> p = 1
    assert chi2_pvalues([0], [5], [0], [7])[0] == 1.0
    # perfectly separated -> real test
    p = chi2_pvalues([0], [50], [50], [0])[0]
    assert p == pytest.approx(ss.chi2_contingency([[0, 50], [50, 0]])[1])


def test_chi_merge_merges_similar():
    n0 = np.array([900, 905, 500, 498])
    n1 = np.array([100, 95, 500, 502])
    groups = chi_merge(n0, n1, alpha=0.05)
    assert sorted(map(tuple, groups)) == [(0, 1), (2, 3)]


# ------------------------------------------------------------------ trend & greedy


def _trend(x, y, n_bins):
    uniq, codes = sorted_unique(x)
    cnt = np.bincount(codes, minlength=uniq.size)
    ev = np.bincount(codes, weights=y, minlength=uniq.size)
    return detect_trend(uniq, cnt, ev, n_bins)


def test_detect_trend():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    y = (rng.random(5000) < 1 / (1 + np.exp(-2 * x))).astype(float)
    assert _trend(x, y, 8) == "pos"
    assert _trend(-x, y, 8) == "neg"
    assert _trend(np.ones(10), y[:10], 8) == "pos"


@pytest.mark.parametrize("card", [7, 300, 10**6])
def test_sorted_unique_and_weighted_percentile(card):
    rng = np.random.default_rng(card)
    x = np.round(rng.normal(size=20000) * card) / card
    uniq, codes = sorted_unique(x)
    ref_u, ref_c = np.unique(x, return_inverse=True)
    np.testing.assert_array_equal(uniq, ref_u)
    np.testing.assert_array_equal(codes, ref_c.ravel())
    q = np.linspace(0, 100, 13)
    cnt = np.bincount(codes)
    np.testing.assert_allclose(weighted_percentile(uniq, cnt, q), np.percentile(x, q))


def _axis(x, y, per, n_per, trend, max_unique=10**9, n_q=200, *extra):
    uniq, codes = sorted_unique(x)
    return build_axis(uniq, codes, y, per, n_per, trend, max_unique, n_q, *extra)


def test_undersampling_reduces_axis():
    rng = np.random.default_rng(1)
    x = rng.normal(size=10000)
    y = rng.integers(0, 2, 10000)
    per = np.zeros(10000, dtype=np.int64)
    for trend in ("pos", "neg"):
        ax = _axis(x, y, per, 1, trend, max_unique=1000, n_q=50)
        assert ax.undersampled and ax.values.size <= 51
        assert ax.counts.sum() == 10000
        if trend == "pos":
            assert ax.values[-1] == x.max()
        else:
            assert ax.values[0] == x.min()


def _bin_rates(ax, cuts):
    edges = [0, *[k + 1 for k in cuts], ax.counts.shape[1]]
    c = np.array([ax.counts[:, a:b].sum(axis=1) for a, b in zip(edges[:-1], edges[1:])])
    e = np.array([ax.events[:, a:b].sum(axis=1) for a, b in zip(edges[:-1], edges[1:])])
    return c, e, e / (c + 1e-8)


@pytest.mark.parametrize("trend", ["pos", "neg"])
@pytest.mark.parametrize("delta", [0.0, 0.02])
def test_greedy_constraints_hold(trend, delta):
    rng = np.random.default_rng(7)
    n, n_per = 30000, 4
    x = rng.normal(size=n)
    sign = 1 if trend == "pos" else -1
    y = (rng.random(n) < 1 / (1 + np.exp(-(-1.5 + sign * 1.5 * x)))).astype(np.int64)
    per = rng.integers(0, n_per, n)
    per_size = np.bincount(per)
    ax = _axis(x, y, per, n_per, trend)
    cuts = greedy_cuts(ax, per_size, trend, 0.05, 5, delta)
    assert len(cuts) >= 2
    c, e, er = _bin_rates(ax, cuts)
    d = np.diff(er, axis=0) * sign
    assert (d >= delta - 1e-12).all()  # monotonic with gap, every period, incl. last bin
    assert (c / per_size >= 0.05).all()
    assert (e >= 5).all() and (c - e >= 5).all()


def test_first_bin_has_no_delta_requirement():
    # v0.5 required the first bin's event rate >= min_er_delta (pos trend); v0.6 does not
    x = np.repeat(np.arange(4.0), 100)
    y = np.concatenate(
        [
            np.r_[np.ones(1), np.zeros(99)],
            np.r_[np.ones(20), np.zeros(80)],
            np.r_[np.ones(40), np.zeros(60)],
            np.r_[np.ones(60), np.zeros(40)],
        ]
    )
    per = np.zeros(400, dtype=np.int64)
    ax = _axis(x, y.astype(np.int64), per, 1, "pos")
    cuts = greedy_cuts(ax, np.array([400]), "pos", 0.05, 1, 0.1)
    assert cuts == [0, 1, 2]


def test_no_split_next_to_attached_missing():
    x = np.repeat(np.arange(3.0), 100)
    y = np.concatenate(
        [np.zeros(90), np.ones(10), np.zeros(70), np.ones(30), np.zeros(50), np.ones(50)]
    ).astype(np.int64)
    per = np.zeros(300, dtype=np.int64)
    miss_c = np.array([[100]])
    miss_e = np.array([[90]])
    ax = _axis(x, y, per, 1, "pos", 10**9, 200, miss_c, miss_e, "end")
    cuts = greedy_cuts(ax, np.array([400]), "pos", 0.05, 1, 0.0)
    assert 2 not in cuts  # never split between the last value and the missing point


# ------------------------------------------------------------------ tide merge


def test_tide_merge_removes_crossings_and_keeps_cont_contiguous():
    # 3 continuous bins (monotonic) + 2 categories that cross them over 3 periods
    counts = np.full((5, 3), 1000)
    rates = np.array(
        [
            [0.05, 0.05, 0.05],
            [0.10, 0.10, 0.10],
            [0.20, 0.20, 0.20],
            [0.04, 0.15, 0.25],  # crosses everything
            [0.11, 0.11, 0.11],
        ]
    )
    events = (counts * rates).astype(np.int64)
    groups = tide_merge(
        counts,
        events,
        n_cont=3,
        trend="pos",
        min_sample_rate=0.01,
        min_class_obs=1,
        min_er_delta=0.0,
    )
    c = np.array([counts[g].sum(axis=0) for g in groups])
    e = np.array([events[g].sum(axis=0) for g in groups])
    er = e / c
    order = np.argsort(er[:, 0])
    assert (np.diff(er[order], axis=0) >= 0).all()  # same order in every period
    for g in groups:
        cont = sorted(a for a in g if a < 3)
        assert cont == list(range(cont[0], cont[-1] + 1)) if cont else True
