"""The bins format: validation, row assignment and labels.

A bins dict looks like::

    {"trend": "pos",
     "bins": {0: {(-np.inf, 0.34): "cont", "special_category": "cat"},
              1: {(0.34, 0.65): "cont"},
              2: {(0.65, np.inf): "cont"},
              3: {None: "missing"}}}

Continuous intervals are ``(a, b]`` for ``trend="pos"`` and ``[a, b)`` for
``trend="neg"``. Infinite bounds are always inclusive. ``None`` is the key for
NaN / None / pd.NA / NaT.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass

import numpy as np

from ._parse import ParsedColumn, normalize_key

__all__ = ["CompiledBins", "bin_label", "compile_bins", "special_keys", "validate_bins"]

ATOM_TYPES = ("cont", "cat", "missing")


def validate_bins(bins: dict, feature: str) -> dict:
    """Check a user bins dict and return a normalized copy."""
    where = f"user_bins[{feature!r}]"
    if not isinstance(bins, dict) or "bins" not in bins or not isinstance(bins["bins"], dict):
        raise ValueError(f"{where} must be a dict with a 'bins' dict")
    if not bins["bins"]:
        raise ValueError(f"{where} has no bins")
    out: dict = {}
    has_cont = False
    for idx, atoms in bins["bins"].items():
        if not isinstance(idx, (int, np.integer)) or isinstance(idx, bool):
            raise ValueError(f"{where}: bin index {idx!r} must be an int")
        if not isinstance(atoms, dict) or not atoms:
            raise ValueError(f"{where}: bin {idx} must be a non-empty dict of atom -> type")
        norm = {}
        for atom, kind in atoms.items():
            if kind not in ATOM_TYPES:
                raise ValueError(f"{where}: bin {idx} has atom type {kind!r}; use {ATOM_TYPES}")
            if kind == "cont":
                ok = (
                    isinstance(atom, tuple)
                    and len(atom) == 2
                    and all(isinstance(v, numbers.Real) and not isinstance(v, bool) for v in atom)
                    and not np.isnan(atom[0])
                    and not np.isnan(atom[1])
                )
                if not ok or not atom[0] < atom[1]:
                    raise ValueError(
                        f"{where}: bin {idx} has invalid interval {atom!r}; use (a, b) with a < b"
                    )
                has_cont = True
                norm[(float(atom[0]), float(atom[1]))] = "cont"
            else:
                norm[normalize_key(atom)] = kind
        out[int(idx)] = norm
    trend = bins.get("trend")
    if has_cont and trend not in ("pos", "neg"):
        raise ValueError(f"{where} has continuous intervals, so 'trend' must be 'pos' or 'neg'")
    result = {"trend": trend if trend in ("pos", "neg") else "pos", "bins": out}
    compile_bins(result, feature)  # checks overlaps and duplicate keys
    return result


def special_keys(bins: dict) -> tuple[list, list]:
    """Numeric keys typed 'cat' / 'missing' in a bins dict (e.g. -999)."""
    cat, miss = [], []
    for atoms in bins["bins"].values():
        for atom, kind in atoms.items():
            if kind != "cont" and isinstance(atom, numbers.Real) and not isinstance(atom, bool):
                (cat if kind == "cat" else miss).append(atom)
    return cat, miss


@dataclass
class CompiledBins:
    """A bins dict in a form ready for vectorized assignment."""

    trend: str
    bin_ids: list  # bin index (dict key) per position
    a: np.ndarray  # interval lower bounds, sorted
    b: np.ndarray  # interval upper bounds
    g: np.ndarray  # position of the bin owning each interval
    key_map: dict  # cat / missing key -> position

    def assign_values(self, x: np.ndarray) -> np.ndarray:
        """Bin position of each continuous value, or -1 if it falls in no interval."""
        if not self.a.size:
            return np.full(x.shape[0], -1, dtype=np.int64)
        if self.trend == "pos":  # (a, b]
            i = np.searchsorted(self.b, x, side="left")
            valid = i < self.b.size
            i = np.minimum(i, self.b.size - 1)
            ok = valid & ((x > self.a[i]) | np.isneginf(self.a[i]))
        else:  # [a, b)
            i = np.searchsorted(self.a, x, side="right") - 1
            valid = i >= 0
            i = np.maximum(i, 0)
            ok = valid & ((x < self.b[i]) | np.isposinf(self.b[i]))
        return np.where(ok, self.g[i], -1)

    def assign(self, parsed: ParsedColumn, cont_codes: np.ndarray | None = None) -> np.ndarray:
        """Bin position per row, or -1 for values that fall in no bin."""
        out = np.full(parsed.n, -1, dtype=np.int64)
        if parsed.x_cont.size:
            if cont_codes is None:
                cont_codes = self.assign_values(parsed.x_cont)
            out[parsed.cont_mask] = cont_codes
        for codes, keys in (
            (parsed.cat_codes, parsed.cat_keys),
            (parsed.miss_codes, parsed.miss_keys),
        ):
            if not keys:
                continue
            lut = np.array([self.key_map.get(k, -1) for k in keys] + [-1], dtype=np.int64)
            m = codes >= 0
            out[m] = lut[codes[m]]
        return out


def compile_bins(bins: dict, feature: str = "") -> CompiledBins:
    """Compile a bins dict; raises on overlapping intervals or duplicate keys."""
    bin_ids = sorted(bins["bins"])
    pos_of = {b: i for i, b in enumerate(bin_ids)}
    intervals, key_map = [], {}
    for b, atoms in bins["bins"].items():
        for atom, kind in atoms.items():
            if kind == "cont":
                intervals.append((atom[0], atom[1], pos_of[b]))
            else:
                if atom in key_map:
                    raise ValueError(f"{feature}: key {atom!r} appears in more than one bin")
                key_map[atom] = pos_of[b]
    intervals.sort()
    a = np.array([t[0] for t in intervals], dtype=np.float64)
    bb = np.array([t[1] for t in intervals], dtype=np.float64)
    g = np.array([t[2] for t in intervals], dtype=np.int64)
    if a.size > 1 and np.any(a[1:] < bb[:-1]):
        raise ValueError(f"{feature}: continuous intervals overlap")
    return CompiledBins(bins.get("trend", "pos"), bin_ids, a, bb, g, key_map)


def _fmt(v: float, decimals: int = 4) -> str:
    if np.isinf(v):
        return "inf" if v > 0 else "-inf"
    if float(v).is_integer():
        return str(int(v))
    return f"{v:.{decimals}f}"


def bin_label(atoms: dict, trend: str, decimals: int = 4) -> str:
    """Display label of a bin, e.g. ``(-inf,0.3400]; special_category``.

    Integer bounds print without decimals (``(18,25]``); other bounds print with exactly
    ``decimals`` decimal places, trailing zeros kept. Only the label is rounded: the bins
    themselves keep the full values.
    """
    parts = []
    for atom, kind in atoms.items():
        if kind == "cont":
            a, b = atom
            if trend == "pos":
                right = ")" if np.isinf(b) else "]"
                parts.append(f"({_fmt(a, decimals)},{_fmt(b, decimals)}{right}")
            else:
                left = "(" if np.isinf(a) else "["
                parts.append(f"{left}{_fmt(a, decimals)},{_fmt(b, decimals)})")
        else:
            parts.append("NaN" if atom is None else str(atom))
    return "; ".join(parts)
