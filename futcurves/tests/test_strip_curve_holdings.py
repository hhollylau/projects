from __future__ import annotations

import pandas as pd

from futcurves.core.curve import build_strip_curve
from futcurves.core.roll import RollPolicy


def test_strip_curve_holdings() -> None:
    dates = pd.to_datetime(["2024-01-01", "2024-01-06", "2024-01-11"])
    meta = pd.DataFrame(
        {
            "contract": ["C1", "C2", "C3"],
            "expiry": pd.to_datetime(["2024-03-20", "2024-06-20", "2024-09-20"]),
            "last_trade_date": pd.to_datetime(["2024-01-15", "2024-04-15", "2024-07-15"]),
        }
    )
    universe = pd.DataFrame({1: ["C1", "C1", "C1"], 2: ["C2", "C2", "C2"]}, index=dates)
    panel = pd.DataFrame(
        {
            "ts": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-06",
                "2024-01-06",
                "2024-01-11",
                "2024-01-11",
            ],
            "contract": ["C1", "C2", "C1", "C2", "C1", "C2"],
            "price": [100, 110, 101, 111, 102, 112],
        }
    )

    class FixedRollPolicy(RollPolicy):
        def roll_start(self, date: pd.Timestamp, front_contract: str, meta: pd.DataFrame, next_contract: str | None = None) -> pd.Timestamp:
            return pd.Timestamp("2024-01-01")

        def roll_end(self, date: pd.Timestamp, front_contract: str, meta: pd.DataFrame) -> pd.Timestamp:
            return pd.Timestamp("2024-01-11")

    rp = FixedRollPolicy(weight_fn="smoothstep")
    curve_px, holdings = build_strip_curve(panel, universe, meta, 2, rp)

    d_mid = pd.Timestamp("2024-01-06")
    w_next = rp.weight_next(d_mid, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-11"))
    w_cur = 1 - w_next

    assert abs(holdings[d_mid][1]["C1"] - w_cur) < 1e-12
    assert abs(holdings[d_mid][1]["C2"] - w_next) < 1e-12
    expected = w_cur * 101 + w_next * 111
    assert abs(curve_px.loc[d_mid, 1] - expected) < 1e-12
