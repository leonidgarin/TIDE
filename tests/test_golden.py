"""v0.6 must reproduce the bins of the original v0.5.0 on these scenarios.

Fixtures come from ``tests/golden/generate.py`` run against the v0.5.0 code.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tide import TIDE

HERE = Path(__file__).parent / "golden"
EXPECTED = json.loads((HERE / "expected.json").read_text())
DATA = np.load(HERE / "data.npz")

# v0.5 parameter names -> v0.6
RENAME = {"missing_categories": "missing_values"}


def _dataset(name):
    keys = sorted(
        (k for k in DATA.files if k.startswith(f"{name}__")), key=lambda k: int(k.split("__")[1])
    )
    return [DATA[k] for k in keys]


def _bins_to_json(bins):
    res = {"trend": bins["trend"], "bins": []}
    for idx in sorted(bins["bins"]):
        atoms = []
        for a, kind in bins["bins"][idx].items():
            atoms.append(["cont", [float(a[0]), float(a[1])]] if kind == "cont" else [kind, a])
        res["bins"].append(atoms)
    return res


def _canon(bins_json):
    # atom order inside a bin is not meaningful (v0.5 merged chi bins in set order)
    return {
        "trend": bins_json["trend"],
        "bins": [sorted(map(json.dumps, b)) for b in bins_json["bins"]],
    }


@pytest.mark.parametrize("scenario", sorted(EXPECTED))
def test_matches_v05(scenario):
    sc = EXPECTED[scenario]
    d = _dataset(sc["data"])
    if sc["data"] == "mixed":
        xm, y, per, code, cats = d
        x = xm.astype(object)
        x[code == 1] = cats[code == 1]
        x[code == 2] = "MISSING"
    else:
        x, y, per = d
    params = {RENAME.get(k, k): v for k, v in sc["params"].items()}
    params.setdefault("max_unique", 10**9)  # v0.5 always searched every unique value
    t = TIDE(**params).fit(pd.DataFrame({"x": x}), y, per)
    assert _canon(_bins_to_json(t.bins_["x"])) == _canon(sc["bins"])
