from __future__ import annotations

import pandas as pd

from futcurves.core.universe import build_rolling_universe


def test_universe_dynamic() -> None:
    meta = pd.DataFrame(
        {
            "contract": ["C1", "C2", "C3", "C4", "C5", "C6"],
            "expiry": pd.to_datetime(
                [
                    "2024-03-20",
                    "2024-06-20",
                    "2024-09-20",
                    "2024-12-20",
                    "2025-03-20",
                    "2025-06-20",
                ]
            ),
            "last_trade_date": pd.to_datetime(
                [
                    "2024-03-18",
                    "2024-06-18",
                    "2024-09-18",
                    "2024-12-18",
                    "2025-03-18",
                    "2025-06-18",
                ]
            ),
        }
    )

    universe = build_rolling_universe(
        meta=meta,
        start="2024-03-13",
        end="2024-03-19",
        n_positions=3,
        offset_bdays=2,
    )

    assert universe.loc[pd.Timestamp("2024-03-13"), 1] == "C1"
    assert universe.loc[pd.Timestamp("2024-03-15"), 1] == "C2"
    assert (universe[1] != universe[1].iloc[0]).any()
