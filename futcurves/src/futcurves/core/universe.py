from __future__ import annotations

from typing import Callable

import pandas as pd
from pandas.tseries.offsets import BDay

from futcurves.core.schema import validate_meta

EligibilityFn = Callable[[pd.Series, pd.Timestamp], bool]


def default_cutoff_date(row: pd.Series, offset_bdays: int = 2) -> pd.Timestamp:
    base = row.get("last_trade_date")
    if pd.isna(base):
        base = row["expiry"]
    return pd.Timestamp(base) - BDay(offset_bdays)


def _default_eligibility(row: pd.Series, d: pd.Timestamp, offset_bdays: int) -> bool:
    expiry_ok = row["expiry"] >= d
    cutoff = default_cutoff_date(row=row, offset_bdays=offset_bdays)
    return bool(expiry_ok and d <= cutoff)


def build_rolling_universe(
    meta: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    n_positions: int,
    eligibility: EligibilityFn | None = None,
    calendar: str = "B",
    offset_bdays: int = 2,
) -> pd.DataFrame:
    if n_positions < 1:
        raise ValueError("n_positions must be >= 1")
    meta_v = validate_meta(meta)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    dates = pd.date_range(start_ts, end_ts, freq=calendar)
    cols = list(range(1, n_positions + 1))
    universe = pd.DataFrame(index=dates, columns=cols, dtype="object")
    elig_fn = eligibility or (lambda row, d: _default_eligibility(row, d, offset_bdays))
    sorted_meta = meta_v.sort_values("expiry").reset_index(drop=True)

    for d in dates:
        eligible_rows = [row for _, row in sorted_meta.iterrows() if elig_fn(row, d)]
        selected = [r["contract"] for r in eligible_rows[:n_positions]]
        for i, contract in enumerate(selected, start=1):
            universe.at[d, i] = contract
    return universe


def contracts_used(universe: pd.DataFrame) -> list[str]:
    vals = pd.unique(universe.values.ravel())
    return sorted([v for v in vals if isinstance(v, str) and v != ""])

