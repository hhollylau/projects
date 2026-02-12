from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

WeightFn = Literal["smoothstep", "linear", "logistic"]


@dataclass(frozen=True)
class RollPolicy:
    roll_end_offset_bdays: int = 0
    roll_window_bdays: int | None = None
    weight_fn: WeightFn = "linear"
    logistic_k: float = 10.0

    def roll_end(self, date: pd.Timestamp, front_contract: str, meta: pd.DataFrame) -> pd.Timestamp:
        _ = date
        row = _lookup_meta_row(front_contract, meta)
        ltd = row.get("last_trade_date")
        base = ltd if not pd.isna(ltd) else row["expiry"]
        return pd.Timestamp(base) - BDay(self.roll_end_offset_bdays)

    def roll_start(
        self,
        date: pd.Timestamp,
        front_contract: str,
        meta: pd.DataFrame,
        next_contract: str | None = None,
    ) -> pd.Timestamp:
        end = self.roll_end(date, front_contract, meta)
        if self.roll_window_bdays is not None:
            return end - BDay(self.roll_window_bdays)
        # Auto: window = calendar-day gap between front and next contract expiry
        if next_contract is None:
            return end  # no next contract → no blending
        front_row = _lookup_meta_row(front_contract, meta)
        next_row = _lookup_meta_row(next_contract, meta)
        front_base = front_row.get("last_trade_date")
        if pd.isna(front_base):
            front_base = front_row["expiry"]
        next_base = next_row.get("last_trade_date")
        if pd.isna(next_base):
            next_base = next_row["expiry"]
        gap = pd.Timestamp(next_base) - pd.Timestamp(front_base)
        return end - gap

    def weight_next(
        self,
        date: pd.Timestamp,
        roll_start: pd.Timestamp,
        roll_end: pd.Timestamp,
    ) -> float:
        d = pd.Timestamp(date)
        if d < roll_start:
            return 0.0
        if d > roll_end:
            return 1.0
        total = (roll_end - roll_start).days
        if total <= 0:
            return 1.0
        elapsed = (d - roll_start).days
        u = float(np.clip(elapsed / total, 0.0, 1.0))
        if self.weight_fn == "linear":
            return u
        if self.weight_fn == "smoothstep":
            return float(3 * u**2 - 2 * u**3)
        if self.weight_fn == "logistic":
            v = 1.0 / (1.0 + np.exp(-self.logistic_k * (u - 0.5)))
            v0 = 1.0 / (1.0 + np.exp(-self.logistic_k * (-0.5)))
            v1 = 1.0 / (1.0 + np.exp(-self.logistic_k * (0.5)))
            return float(np.clip((v - v0) / (v1 - v0), 0.0, 1.0))
        raise ValueError(f"Unsupported weight_fn: {self.weight_fn}")

    def weight_current(
        self,
        date: pd.Timestamp,
        roll_start: pd.Timestamp,
        roll_end: pd.Timestamp,
    ) -> float:
        return 1.0 - self.weight_next(date, roll_start, roll_end)


def _lookup_meta_row(contract: str, meta: pd.DataFrame) -> pd.Series:
    rows = meta.loc[meta["contract"] == contract]
    if rows.empty:
        raise KeyError(f"contract {contract!r} not found in meta")
    return rows.iloc[0]

