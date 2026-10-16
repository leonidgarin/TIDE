"""Generate golden fixtures from the original TIDE v0.5.0.

Usage:  python tests/golden/generate.py /path/to/TIDE-v0.5.0-checkout

Writes ``data.npz`` (datasets) and ``expected.json`` (v0.5.0 bins) next to this file.
Only scenarios where v0.6 must reproduce v0.5 exactly are included; the documented
fixes (first-bin min_er_delta, last-bin min_er_delta, splits next to attached missing
values, tide categorical mode) are not covered here.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent

SCENARIOS = {
    "cont_pos_6per": dict(data="cont_pos", params={}),
    "cont_neg_6per": dict(data="cont_neg", params={}),
    "cont_pos_1per": dict(data="cont_pos_1per", params={}),
    "int_age_12per": dict(data="int_age", params={}),
    "cont_strict": dict(data="cont_pos", params={"min_sample_rate": 0.1, "min_class_obs": 30}),
    "mixed_separate": dict(data="mixed", params={"missing_categories": ["MISSING"]}),
    "mixed_chi": dict(
        data="mixed", params={"missing_categories": ["MISSING"], "cat_strategy": "chi-squared"}
    ),
    "mixed_worst": dict(
        data="mixed",
        params={
            "missing_categories": ["MISSING"],
            "missing_strategy": "worst_bin",
            "missing_rate": 0.1,
        },
    ),
    "mixed_best": dict(
        data="mixed",
        params={
            "missing_categories": ["MISSING"],
            "missing_strategy": "best_bin",
            "missing_rate": 0.1,
        },
    ),
}


def make_data():
    rng = np.random.default_rng(20261016)
    out = {}
    n = 20000
    per = rng.integers(0, 6, n)
    x = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-2 + 0.9 * x + 0.1 * per)))).astype(np.int64)
    out["cont_pos"] = (x, y, per)
    out["cont_neg"] = (-x * 3 + 10, y, per)
    out["cont_pos_1per"] = (x, y, np.zeros(n, dtype=np.int64))
    per12 = rng.integers(0, 12, n)
    age = rng.integers(18, 75, n).astype(np.float64)
    ya = (rng.random(n) < 1 / (1 + np.exp(-(1 - 0.06 * age)))).astype(np.int64)
    out["int_age"] = (age, ya, per12)
    xm = rng.normal(size=n)
    ym = (rng.random(n) < 1 / (1 + np.exp(-(-1.5 + 1.5 * xm)))).astype(np.int64)
    kind = rng.random(n)
    cats = np.array(["A", "B", "C", "D", "E"])[rng.integers(0, 5, n)]
    # categorical / missing codes: 0 numeric, 1 category, 2 missing
    code = np.where(kind < 0.08, 1, np.where(kind < 0.12, 2, 0))
    ym = np.where(code == 1, (rng.random(n) < np.char.find("ABCDE", cats) * 0.05 + 0.05), ym)
    out["mixed"] = (xm, ym.astype(np.int64), rng.integers(0, 2, n), code, cats)
    return out


def to_mixed(xm, code, cats):
    x = xm.astype(object)
    x[code == 1] = cats[code == 1]
    x[code == 2] = "MISSING"
    return x


def bins_to_json(bins):
    res = {"trend": bins["trend"], "bins": []}
    for idx in sorted(bins["bins"]):
        atoms = []
        for a, kind in bins["bins"][idx].items():
            if kind == "cont":
                atoms.append(["cont", [float(a[0]), float(a[1])]])
            else:
                atoms.append([kind, a])
        res["bins"].append(atoms)
    return res


def main(old_repo):
    sys.path.insert(0, str(old_repo))
    from tide import TIDE as OldTIDE

    data = make_data()
    arrays = {}
    for k, v in data.items():
        for i, a in enumerate(v):
            arrays[f"{k}__{i}"] = a
    np.savez_compressed(HERE / "data.npz", **arrays)

    expected = {}
    for name, sc in SCENARIOS.items():
        d = data[sc["data"]]
        if sc["data"] == "mixed":
            x = to_mixed(d[0], d[3], d[4])
        else:
            x = d[0]
        X = pd.DataFrame({"x": pd.Series(x, dtype=object if sc["data"] == "mixed" else None)})
        t = OldTIDE(**sc["params"])
        bins = t.fit(X, pd.Series(d[1]), pd.Series(d[2]), return_bins=True, disable_tqdm=True)
        expected[name] = {
            "data": sc["data"],
            "params": sc["params"],
            "bins": bins_to_json(bins["x"]),
        }
        print(name, bins["x"])
    (HERE / "expected.json").write_text(json.dumps(expected, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
