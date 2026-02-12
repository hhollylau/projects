from __future__ import annotations

from typing import NamedTuple

import pandas as pd
from dateutil.relativedelta import relativedelta

# ---------------------------------------------------------------------------
# Module-level helpers for parent symbology
# ---------------------------------------------------------------------------

_MONTH_MAP = {"H": 3, "M": 6, "U": 9, "Z": 12}


class ParentFetchResult(NamedTuple):
    """Result of a parent-symbology fetch: (panel, meta) DataFrames."""

    panel: pd.DataFrame
    meta: pd.DataFrame


def _third_wednesday(year: int, month: int) -> pd.Timestamp:
    """Return the 3rd Wednesday of the given month/year."""
    first = pd.Timestamp(year=year, month=month, day=1)
    days_to_wed = (2 - first.weekday()) % 7
    return first + pd.Timedelta(days=days_to_wed + 14)


def _infer_expiry(symbol: str, last_seen: pd.Timestamp) -> pd.Timestamp:
    """Infer contract expiry from symbol and last observed trade date.

    Resolves decade ambiguity by finding the smallest year whose 3rd-Wednesday
    expiry is on or after *last_seen*.

    Example: SR3H5 last seen 2024-12-01 -> Mar 2025.
             SR3H5 last seen 2025-04-01 -> Mar 2035.
    """
    month = _MONTH_MAP[symbol[3]]
    digit = int(symbol[4])
    year = 2010 + digit
    while True:
        expiry = _third_wednesday(year, month)
        if expiry >= last_seen:
            return expiry
        year += 10


def _build_meta(panel: pd.DataFrame) -> pd.DataFrame:
    """Build a meta DataFrame from a filtered panel.

    For each contract, infers expiry from the symbol and the last date
    the contract appeared in the data.
    """
    last_seen = panel.groupby("contract")["ts"].max()
    rows = []
    for contract, last_ts in last_seen.items():
        ts = pd.Timestamp(last_ts)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        expiry = _infer_expiry(str(contract), ts)
        rows.append({
            "contract": str(contract),
            "expiry": expiry,
            "last_trade_date": expiry - pd.tseries.offsets.BDay(2),
        })
    return pd.DataFrame(rows).sort_values("expiry").reset_index(drop=True)


def _date_batches(start: str, end: str, batch_years: int) -> list[tuple[str, str]]:
    """Split [start, end] into contiguous sub-ranges of *batch_years* length."""
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    batches: list[tuple[str, str]] = []
    while s < e:
        batch_end = min(s + relativedelta(years=batch_years), e)
        batches.append((s.strftime("%Y-%m-%d"), batch_end.strftime("%Y-%m-%d")))
        s = batch_end
    return batches


# ---------------------------------------------------------------------------
# DatabentoSource class
# ---------------------------------------------------------------------------


class DatabentoSource:
    """Optional adapter for Databento symbology.

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

    def fetch_parent(
        self,
        product: str,
        start: str,
        end: str,
        *,
        contract_filter: str | None = r"^SR3[HMUZ]\d$",
        batch_years: int | None = None,
    ) -> ParentFetchResult:
        """Fetch all active contracts using parent symbology.

        Parameters
        ----------
        product : str
            Root product symbol, e.g. ``"SR3"``. The method fetches
            ``f"{product}.FUT"`` with ``stype_in="parent"``.
        start, end : str
            Date range (inclusive) as ``"YYYY-MM-DD"`` strings.
        contract_filter : str or None
            Regex to keep only matching symbols. Default keeps SR3
            single-leg quarterly contracts. Pass ``None`` to skip.
        batch_years : int or None
            If set, splits the date range into *batch_years*-length chunks
            and fetches each separately. Useful for long date ranges to
            control cost and avoid API size limits.

        Returns
        -------
        ParentFetchResult
            Named tuple of ``(panel, meta)`` DataFrames.  Supports both
            attribute access (``result.panel``) and unpacking
            (``panel, meta = ...``).

        Notes
        -----
        The ``contract_filter`` default is SR3-specific. For other products,
        pass a suitable regex or ``None``.
        """
        try:
            import databento as db  # type: ignore
        except ImportError as exc:
            raise ImportError("Install optional dependency: pip install futcurves[databento]") from exc

        client = db.Historical(key=self.api_key)
        parent_symbol = f"{product}.FUT"

        if batch_years is not None:
            batches = _date_batches(start, end, batch_years)
        else:
            batches = [(start, end)]

        frames: list[pd.DataFrame] = []
        for batch_start, batch_end in batches:
            data = client.timeseries.get_range(
                dataset=self.dataset,
                symbols=[parent_symbol],
                schema=self.schema,
                stype_in="parent",
                start=batch_start,
                end=batch_end,
            )
            frames.append(data.to_df().reset_index())

        df = pd.concat(frames, ignore_index=True)

        ts_col = "ts_event" if "ts_event" in df.columns else "ts_recv"
        px_col = "close" if "close" in df.columns else "price"

        panel = pd.DataFrame({
            "ts": pd.to_datetime(df[ts_col]),
            "contract": df["symbol"].astype(str),
            "price": pd.to_numeric(df[px_col], errors="coerce"),
        })
        if "volume" in df.columns:
            panel["volume"] = pd.to_numeric(df["volume"], errors="coerce")

        if contract_filter is not None:
            mask = panel["contract"].str.match(contract_filter)
            panel = panel.loc[mask].reset_index(drop=True)

        panel = panel.drop_duplicates(subset=["ts", "contract"]).reset_index(drop=True)

        meta = _build_meta(panel)

        return ParentFetchResult(panel=panel, meta=meta)

    def estimate_cost(
        self,
        product: str,
        start: str,
        end: str,
        *,
        stype_in: str = "parent",
    ) -> float:
        """Estimate Databento cost for a fetch. Returns cost in USD."""
        try:
            import databento as db  # type: ignore
        except ImportError as exc:
            raise ImportError("Install optional dependency: pip install futcurves[databento]") from exc

        client = db.Historical(key=self.api_key)
        symbol = f"{product}.FUT" if stype_in == "parent" else product
        return client.metadata.get_cost(
            dataset=self.dataset,
            symbols=symbol,
            schema=self.schema,
            start=start,
            end=end,
            stype_in=stype_in,
        )
