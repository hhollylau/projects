from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from futcurves.core.roll import RollPolicy
from futcurves.core.schema import validate_meta, validate_panel

Holdings = dict[pd.Timestamp, dict[int, dict[str, float]]]


def build_strip_curve(
    panel: pd.DataFrame,
    universe: pd.DataFrame,
    meta: pd.DataFrame,
    n_positions: int,
    roll_policy: RollPolicy,
    ffill_by_contract: bool = True,
    drop_weight_tol: float = 1e-10,
) -> tuple[pd.DataFrame, Holdings]:
    if n_positions < 1:
        raise ValueError("n_positions must be >= 1")
    panel_v = validate_panel(panel)
    meta_v = validate_meta(meta)
    px_map = _build_price_map(panel_v, ffill_by_contract=ffill_by_contract)

    idx = pd.DatetimeIndex(universe.index)
    cols = list(range(1, n_positions + 1))
    curve_px = pd.DataFrame(index=idx, columns=cols, dtype="float64")
    holdings: Holdings = {}

    for d in idx:
        holdings[d] = {}
        front = _safe_contract(universe, d, 1)
        if front is None:
            _set_empty_day(holdings, d, n_positions)
            continue

        roll_start = roll_policy.roll_start(d, front, meta_v)
        roll_end = roll_policy.roll_end(d, front, meta_v)
        w_next = roll_policy.weight_next(d, roll_start, roll_end)
        w_cur = 1.0 - w_next

        for p in cols:
            c = _safe_contract(universe, d, p)
            if c is None:
                holdings[d][p] = {}
                curve_px.at[d, p] = float("nan")
                continue

            if w_next <= drop_weight_tol:
                holdings[d][p] = {c: 1.0}
                curve_px.at[d, p] = _lookup_price(px_map, d, c)
                continue

            c_next = _safe_contract(universe, d, p + 1)
            if c_next is None:
                holdings[d][p] = {c: 1.0}
                curve_px.at[d, p] = _lookup_price(px_map, d, c)
                continue

            px_cur = _lookup_price(px_map, d, c)
            px_next = _lookup_price(px_map, d, c_next)
            if pd.isna(px_cur) and pd.isna(px_next):
                curve_px.at[d, p] = float("nan")
            elif pd.isna(px_cur):
                curve_px.at[d, p] = px_next
            elif pd.isna(px_next):
                curve_px.at[d, p] = px_cur
            else:
                curve_px.at[d, p] = w_cur * px_cur + w_next * px_next

            w_map = {}
            if abs(w_cur) > drop_weight_tol:
                w_map[c] = w_cur
            if abs(w_next) > drop_weight_tol:
                w_map[c_next] = w_next
            holdings[d][p] = w_map

    return curve_px, holdings


def _build_price_map(panel: pd.DataFrame, ffill_by_contract: bool) -> pd.DataFrame:
    p = panel.copy()
    p["date"] = pd.to_datetime(p["ts"]).dt.normalize()
    p = p.sort_values(["contract", "date"])
    price_map = p.pivot_table(index="date", columns="contract", values="price", aggfunc="last")
    if ffill_by_contract:
        price_map = price_map.ffill()
    return price_map


def _lookup_price(price_map: pd.DataFrame, date: pd.Timestamp, contract: str) -> float:
    d = pd.Timestamp(date).normalize()
    if d not in price_map.index:
        return float("nan")
    if contract not in price_map.columns:
        return float("nan")
    v = price_map.at[d, contract]
    return float(v) if pd.notna(v) else float("nan")


def _safe_contract(universe: pd.DataFrame, date: pd.Timestamp, pos: int) -> str | None:
    if pos not in universe.columns:
        return None
    if date not in universe.index:
        return None
    val = universe.at[date, pos]
    if pd.isna(val):
        return None
    return str(val)


def _set_empty_day(holdings: Holdings, date: pd.Timestamp, n_positions: int) -> None:
    holdings[date] = {p: {} for p in range(1, n_positions + 1)}


def normalize_holdings_weights(holdings_for_position: Mapping[str, float]) -> dict[str, float]:
    s = float(sum(abs(v) for v in holdings_for_position.values()))
    if s == 0:
        return {}
    return {k: float(v / s) for k, v in holdings_for_position.items()}

