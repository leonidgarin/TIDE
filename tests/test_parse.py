import numpy as np
import pandas as pd
import pytest

from tide._parse import check_target, encode_periods, parse_column


def test_mixed_object_column():
    x = pd.Series([18, 44.5, np.nan, "special", None, "special", -999, "12", True], dtype=object)
    p = parse_column(x, cat_values=[-999], missing_values=[])
    assert p.cont_mask.tolist() == [True, True, False, False, False, False, False, False, False]
    assert p.x_cont.tolist() == [18.0, 44.5]
    # strings first (sorted), then bools, then numeric specials
    assert p.cat_keys == ["12", "special", True, -999]
    assert p.cat_codes.tolist() == [-1, -1, -1, 1, -1, 1, 3, 0, 2]
    assert p.miss_keys == [None]
    assert p.miss_codes.tolist() == [-1, -1, 0, -1, 0, -1, -1, -1, -1]


def test_numeric_fast_path_with_specials():
    x = pd.Series([1.0, -999.0, np.nan, -1.0, 5.0])
    p = parse_column(x, cat_values=[-999], missing_values=[-1])
    assert p.x_cont.tolist() == [1.0, 5.0]
    assert p.cat_keys == [-999]
    assert p.miss_keys == [None, -1]
    assert p.miss_codes.tolist() == [-1, -1, 0, 1, -1]


def test_string_missing_values_and_pandas_string_dtype():
    x = pd.Series(["a", "N/A", None, "b"], dtype="string")
    p = parse_column(x, missing_values=["N/A"])
    assert p.cat_keys == ["a", "b"]
    assert p.miss_keys == [None, "N/A"]
    assert not p.cont_mask.any()


def test_nullable_int_and_bool():
    p = parse_column(pd.Series([1, None, 3], dtype="Int64"))
    assert p.x_cont.tolist() == [1.0, 3.0] and p.miss_keys == [None]
    b = parse_column(pd.Series([True, False, True]))
    assert b.cat_keys == [False, True] and not b.cont_mask.any()


def test_nat_is_missing_and_inf_is_continuous():
    p = parse_column(pd.Series([np.inf, -np.inf, pd.NaT, 1], dtype=object))
    assert p.x_cont.tolist() == [np.inf, -np.inf, 1.0]
    assert p.miss_keys == [None]


def test_check_target():
    assert check_target([0, 1, True], 3).tolist() == [0, 1, 1]
    with pytest.raises(ValueError, match="binary"):
        check_target([0, 2, 1], 3)
    with pytest.raises(ValueError, match="both classes"):
        check_target([1, 1], 2)
    with pytest.raises(ValueError, match="missing"):
        check_target([0, np.nan, 1], 3)
    with pytest.raises(ValueError, match="rows"):
        check_target([0, 1], 3)


@pytest.mark.parametrize(
    "periods",
    [
        ["2024-03", "2024-01", "2024-02", "2024-01"],
        pd.to_datetime(["2024-03-01", "2024-01-01", "2024-02-01", "2024-01-01"]),
        pd.PeriodIndex(["2024-03", "2024-01", "2024-02", "2024-01"], freq="M"),
        [202403, 202401, 202402, 202401],
    ],
)
def test_periods_sorted_codes(periods):
    codes, uniq = encode_periods(periods, 4)
    assert codes.tolist() == [2, 0, 1, 0]
    assert len(uniq) == 3


def test_periods_errors():
    with pytest.raises(ValueError, match="missing"):
        encode_periods(pd.to_datetime(["2024-01-01", None]), 2)
    with pytest.raises(ValueError, match="rows"):
        encode_periods([1, 2], 3)
    codes, uniq = encode_periods(None, 3)
    assert codes.tolist() == [0, 0, 0] and list(uniq) == ["all"]
