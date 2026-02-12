from __future__ import annotations

import os

from futcurves.sources.databento_source import DatabentoSource


if __name__ == "__main__":
    api_key = os.getenv("DATABENTO_API_KEY", "")
    if not api_key:
        raise RuntimeError("Set DATABENTO_API_KEY in your environment.")

    src = DatabentoSource(
        api_key=api_key,
        dataset="GLBX.MDP3",
        schema="ohlcv-1d",
        stype_in="continuous",
    )
    panel = src.fetch_continuous(
        product="SR3",
        start="2024-01-01",
        end="2026-01-01",
        n_positions=20,
        rule="v",
    )
    print(panel.head())
