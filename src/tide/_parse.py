"""Parsing of mixed-type feature columns, the target and the periods."""

from __future__ import annotations

import numbers
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["ParsedColumn", "check_target", "encode_periods", "normalize_key", "parse_column"]


@dataclass
class ParsedColumn:
    """A feature column split into its continuous, categorical and missing parts.

    Attributes
    ----------
    cont_mask : np.ndarray of bool, shape (n,)
        Rows that hold continuous (numeric) values.
    x_cont : np.ndarray of float64, shape (n_cont,)
        The continuous values, in row order.
    cat_codes : np.ndarray of int64, shape (n,)
        Index into ``cat_keys`` for categorical rows, -1 elsewhere.
    cat_keys : list
        Distinct categories, sorted (strings first, then bools, then numbers).
    miss_codes : np.ndarray of int64, shape (n,)
        Index into ``miss_keys`` for missing rows, -1 elsewhere.
    miss_keys : list
        Distinct missing keys. ``None`` stands for NaN / None / pd.NA / NaT.
    """

    cont_mask: np.ndarray
    x_cont: np.ndarray
    cat_codes: np.ndarray
    cat_keys: list
    miss_codes: np.ndarray
    miss_keys: list

    @property
    def n(self) -> int:
        return self.cont_mask.shape[0]


def _is_number(v) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_))


_vec_type = np.frompyfunc(type, 1, 1)


def _number_mask(values: np.ndarray) -> np.ndarray:
    """Vectorized ``_is_number``: classify each distinct Python type once."""
    codes, types = pd.factorize(_vec_type(values))
    flags = np.array(
        [issubclass(t, numbers.Real) and not issubclass(t, (bool, np.bool_)) for t in types],
        dtype=bool,
    )
    return flags[codes]


def normalize_key(k):
    """Return the canonical form of a categorical / missing key.

    NaN-like values become ``None``; numpy scalars become Python scalars.
    """
    if k is None:
        return None
    try:
        if pd.isna(k):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(k, np.generic):
        return k.item()
    return k


def _sort_rank(k):
    if isinstance(k, str):
        return (0, k, 0.0)
    if isinstance(k, bool):
        return (1, "", float(k))
    if _is_number(k):
        return (2, "", float(k))
    return (3, f"{type(k).__name__}:{k!r}", 0.0)


def _split_specials(values) -> tuple[list[tuple[float, object]], list]:
    """Split user special values into numeric ones and the rest (NaN-likes are dropped)."""
    numeric, other = [], []
    for v in values:
        v = normalize_key(v)
        if v is None:
            continue
        if _is_number(v):
            numeric.append((float(v), v))
        else:
            other.append(v)
    return numeric, other


def parse_column(x, cat_values=(), missing_values=()) -> ParsedColumn:
    """Split a mixed-type column into continuous, categorical and missing parts.

    Rules: NaN-like values and ``missing_values`` are missing; strings, bools and
    ``cat_values`` are categorical; every other real number is continuous.
    """
    s = x if isinstance(x, pd.Series) else pd.Series(x)
    n = len(s)
    num_cat, _ = _split_specials(cat_values)  # strings are categories anyway
    num_miss, obj_miss = _split_specials(missing_values)

    dtype = s.dtype
    numeric_dtype = pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)

    if numeric_dtype:
        vals = s.to_numpy(dtype=np.float64, na_value=np.nan)
        na = np.isnan(vals)
        cont = ~na
        obj_idx = np.empty(0, dtype=np.int64)
        obj_vals = np.empty(0, dtype=object)
    else:
        arr = s.to_numpy(dtype=object)
        na = np.asarray(pd.isna(arr), dtype=bool)
        cont = np.zeros(n, dtype=bool)
        not_na = np.flatnonzero(~na)
        if not_na.size:
            is_num = _number_mask(arr[not_na])
            cont[not_na[is_num]] = True
            obj_idx = not_na[~is_num]
        else:
            obj_idx = not_na
        vals = np.full(n, np.nan)
        if cont.any():
            try:
                vals[cont] = arr[cont].astype(np.float64)
            except (TypeError, ValueError) as err:  # pragma: no cover - exotic numbers
                raise TypeError(f"Cannot convert numeric values to float: {err}") from err
        obj_vals = arr[obj_idx]

    # Missing keys: NaN-like first (key None), then user missing values.
    miss_atoms: list[tuple[object, np.ndarray]] = []
    if na.any():
        miss_atoms.append((None, np.flatnonzero(na)))
    cat_atoms: list[tuple[object, np.ndarray]] = []

    for specials, target in ((num_miss, miss_atoms), (num_cat, cat_atoms)):
        for fval, key in specials:
            m = cont & (vals == fval)
            if m.any():
                target.append((key, np.flatnonzero(m)))
                cont &= ~m

    if obj_vals.size:
        obj_series = pd.Series(obj_vals, dtype=object)
        if obj_miss:
            mm = obj_series.isin(obj_miss).to_numpy()
            if mm.any():
                codes, uniques = pd.factorize(obj_series[mm], sort=False)
                rows = obj_idx[mm]
                for c, key in enumerate(uniques):
                    miss_atoms.append((normalize_key(key), rows[codes == c]))
                obj_series = obj_series[~mm]
                obj_idx = obj_idx[~mm]
        if len(obj_series):
            codes, uniques = pd.factorize(obj_series, sort=False)
            order = np.argsort(codes, kind="stable")
            bounds = np.searchsorted(codes[order], np.arange(len(uniques) + 1))
            for c, key in enumerate(uniques):
                cat_atoms.append((normalize_key(key), obj_idx[order[bounds[c] : bounds[c + 1]]]))

    cat_atoms.sort(key=lambda a: _sort_rank(a[0]))

    cat_codes = np.full(n, -1, dtype=np.int64)
    for c, (_, rows) in enumerate(cat_atoms):
        cat_codes[rows] = c
    miss_codes = np.full(n, -1, dtype=np.int64)
    for c, (_, rows) in enumerate(miss_atoms):
        miss_codes[rows] = c

    return ParsedColumn(
        cont_mask=cont,
        x_cont=vals[cont],
        cat_codes=cat_codes,
        cat_keys=[k for k, _ in cat_atoms],
        miss_codes=miss_codes,
        miss_keys=[k for k, _ in miss_atoms],
    )


def check_target(y, n: int) -> np.ndarray:
    """Validate a binary target and return it as an int8 array."""
    arr = y.to_numpy() if isinstance(y, (pd.Series, pd.DataFrame)) else np.asarray(y)
    arr = np.squeeze(arr) if arr.ndim > 1 else arr
    if arr.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if arr.shape[0] != n:
        raise ValueError(f"y has {arr.shape[0]} rows, X has {n}")
    if pd.isna(arr).any():
        raise ValueError("y contains missing values")
    try:
        yf = arr.astype(np.float64)
    except (TypeError, ValueError) as err:
        raise ValueError("y must be binary (0/1 or bool)") from err
    if not np.isin(yf, (0.0, 1.0)).all():
        raise ValueError("y must be binary (0/1 or bool)")
    yi = yf.astype(np.int8)
    if yi.sum() == 0 or yi.sum() == n:
        raise ValueError("y must contain both classes")
    return yi


def encode_periods(periods, n: int) -> tuple[np.ndarray, pd.Index]:
    """Encode periods as integer codes in sorted order of the period values."""
    if periods is None:
        return np.zeros(n, dtype=np.int64), pd.Index(["all"])
    s = periods if isinstance(periods, pd.Series) else pd.Series(periods)
    if len(s) != n:
        raise ValueError(f"periods has {len(s)} rows, X has {n}")
    if s.isna().any():
        raise ValueError("periods contain missing values")
    try:
        codes, uniques = pd.factorize(s, sort=True)
    except TypeError as err:
        raise ValueError(f"periods must be sortable values of one type: {err}") from err
    return codes.astype(np.int64), pd.Index(uniques)
