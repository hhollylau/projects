from __future__ import annotations

import pandas as pd


class DatabentoSource:
    """Optional adapter for Databento continuous symbology.

    This is intentionally lightweight and optional so core package remains
    vendor-agnostic.
    """

    def __init__(self, api_key: str, dataset: str, schema: str = "ohlcv-1d", stype_in: str = "continuous"):
        self.api_key = api_key
        self.dataset = dataset
        self.schema = schema
        self.stype_in = stype_in

    def fetch_continuous(
        self,
        product: str,
        start: str,
        end: str,
        n_positions: int,
        rule: str = "v",
    ) -> pd.DataFrame:
        try:
            import databento as db  # type: ignore
        except ImportError as exc:
            raise ImportError("Install optional dependency: pip install futcurves[databento]") from exc

        client = db.Historical(key=self.api_key)
        symbols = [f"{product}.{rule}.{i}" for i in range(n_positions)]
        data = client.timeseries.get_range(
            dataset=self.dataset,
            symbols=symbols,
            schema=self.schema,
            stype_in=self.stype_in,
            start=start,
            end=end,
        )
        df = data.to_df().reset_index()

        ts_col = "ts_event" if "ts_event" in df.columns else "ts_recv"
        px_col = "close" if "close" in df.columns else "price"
        out = pd.DataFrame(
            {
                "ts": pd.to_datetime(df[ts_col]),
                "contract": df["symbol"].astype(str),
                "price": pd.to_numeric(df[px_col], errors="coerce"),
            }
        )
        if "volume" in df.columns:
            out["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        return out
