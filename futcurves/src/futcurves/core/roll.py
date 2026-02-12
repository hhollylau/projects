from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

WeightFn = Literal["smoothstep", "linear", "logistic"]


@dataclass(frozen=True)
class RollPolicy:
    roll_end_offset_bdays: int = 2
    roll_window_bdays: int = 7
    weight_fn: WeightFn = "smoothstep"
    logistic_k: float = 10.0

    def roll_end(self, date: pd.Timestamp, front_contract: str, meta: pd.DataFrame) -> pd.Timestamp:
        _ = date
        row = _lookup_meta_row(front_contract, meta)
        ltd = row.get("last_trade_date")
        base = ltd if not pd.isna(ltd) else row["expiry"]
        return pd.Timestamp(base) - BDay(self.roll_end_offset_bdays)

    def roll_start(self, date: pd.Timestamp, front_contract: str, meta: pd.DataFrame) -> pd.Timestamp:
        end = self.roll_end(date, front_contract, meta)
        return end - BDay(self.roll_window_bdays)

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

