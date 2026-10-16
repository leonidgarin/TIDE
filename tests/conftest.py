import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")


def make_data(n=20000, n_per=6, seed=0, drift=0.0):
    """Synthetic credit-like data: one strong continuous driver plus mixed-type columns."""
    rng = np.random.default_rng(seed)
    per = rng.integers(0, n_per, n)
    x = rng.normal(size=n)
    logit = -2 + 1.2 * x + drift * per
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    age = rng.integers(18, 75, n).astype(object)
    age[rng.random(n) < 0.04] = np.nan
    age[rng.random(n) < 0.03] = "no_hit"
    age[rng.random(n) < 0.03] = -999
    X = pd.DataFrame({"x": x, "neg": -2 * x + rng.normal(scale=0.3, size=n), "age": age})
    months = pd.date_range("2023-01-01", periods=n_per, freq="MS")
    return X, y, months[per]


@pytest.fixture
def data():
    return make_data()
