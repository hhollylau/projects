from __future__ import annotations

from typing import NamedTuple

import pandas as pd
from dateutil.relativedelta import relativedelta

# ---------------------------------------------------------------------------
# Module-level helpers for parent symbology
# ---------------------------------------------------------------------------

_MONTH_MAP = {"H": 3, "M": 6, "U": 9, "Z": 12}

_MONTH_CODES = {1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
                7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"}

_QUARTERLY_MONTHS = [3, 6, 9, 12]
_MONTHLY_MONTHS = list(range(1, 13))


def _front_n_symbols(
    product: str,
    as_of: str | pd.Timestamp,
    n: int,
    frequency: str,
) -> list[str]:
    """Return the front *n* contract symbols as of a given date.

    Parameters
    ----------
    product : str
        Root symbol, e.g. ``"SR3"`` or ``"ES"``.
    as_of : str or Timestamp
        The reference date.
    n : int
        Number of front contracts to return.
    frequency : str
        ``"quarterly"`` (H/M/U/Z) or ``"monthly"`` (all 12 months).
    """
    d = pd.Timestamp(as_of)
    if frequency == "quarterly":
        eligible = _QUARTERLY_MONTHS
    elif frequency == "monthly":
        eligible = _MONTHLY_MONTHS
    else:
        raise ValueError(f"Unknown frequency: {frequency!r}. Use 'quarterly' or 'monthly'.")

    symbols: list[str] = []
    year = d.year
    month = d.month

    while len(symbols) < n:
        found = False
        for m in eligible:
            if m >= month:
                code = _MONTH_CODES[m]
                digit = year % 10
                symbols.append(f"{product}{code}{digit}")
                month = m + 1
                if month > 12:
                    month = 1
                    year += 1
                found = True
                break
        if not found:
            month = 1
            year += 1

    return symbols


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
    expiry is on or after *last_seen*, with a 7-day buffer to account for
    settlement data that may appear after the actual expiry date.

    Example: SR3H5 last seen 2024-12-01 -> Mar 2025.
             SR3H5 last seen 2025-04-01 -> Mar 2035.
    """
    month = _MONTH_MAP[symbol[3]]
    digit = int(symbol[4])
    year = 2010 + digit
    buffer = pd.Timedelta(days=7)
    while True:
        expiry = _third_wednesday(year, month)
        if expiry + buffer >= last_seen:
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
        ts = pd.to_datetime(df[ts_col])
        if ts.dt.tz is not None:
            ts = ts.dt.tz_localize(None)
        out = pd.DataFrame(
            {
                "ts": ts,
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

        ts = pd.to_datetime(df[ts_col])
        if ts.dt.tz is not None:
            ts = ts.dt.tz_localize(None)
        panel = pd.DataFrame({
            "ts": ts,
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

    def fetch_contracts(
        self,
        product: str,
        start: str,
        end: str,
        n_contracts: int = 20,
        frequency: str = "quarterly",
    ) -> ParentFetchResult:
        """Fetch front N outright contracts using windowed symbol generation.

        Iterates through the date range in steps matching the contract
        frequency (3 months for quarterly, 1 month for monthly).  For each
        window, computes the front *n_contracts* symbols as of the window
        start date, fetches those exact symbols from Databento with
        ``stype_in="raw_symbol"``, and concatenates the results.

        This avoids decade-ambiguity issues and only fetches outrights
        (no spreads or strips).

        Parameters
        ----------
        product : str
            Root symbol, e.g. ``"SR3"`` or ``"ES"``.
        start, end : str
            Date range as ``"YYYY-MM-DD"`` strings.
        n_contracts : int
            Number of front contracts to track (default 20).
        frequency : str
            ``"quarterly"`` or ``"monthly"``.

        Returns
        -------
        ParentFetchResult
            Named tuple of ``(panel, meta)`` DataFrames.
        """
        try:
            import databento as db  # type: ignore
        except ImportError as exc:
            raise ImportError("Install optional dependency: pip install futcurves[databento]") from exc

        step = relativedelta(months=3) if frequency == "quarterly" else relativedelta(months=1)
        client = db.Historical(key=self.api_key)

        s = pd.Timestamp(start)
        e = pd.Timestamp(end)

        frames: list[pd.DataFrame] = []
        window_num = 0
        while s < e:
            w_end = min(s + step, e)
            syms = _front_n_symbols(product, s, n_contracts, frequency)
            window_num += 1

            data = client.timeseries.get_range(
                dataset=self.dataset,
                symbols=syms,
                schema=self.schema,
                stype_in="raw_symbol",
                start=s.strftime("%Y-%m-%d"),
                end=w_end.strftime("%Y-%m-%d"),
            )
            df = data.to_df().reset_index()
            if len(df) > 0:
                frames.append(df)
                print(f"  window {window_num}: {s.date()} to {w_end.date()} — {len(df):,} rows, {df['symbol'].nunique()} symbols")
            else:
                print(f"  window {window_num}: {s.date()} to {w_end.date()} — no data")

            s = w_end

        if not frames:
            raise RuntimeError(f"No data returned from Databento for {product} {start} to {end}")

        df_all = pd.concat(frames, ignore_index=True)

        ts_col = "ts_event" if "ts_event" in df_all.columns else "ts_recv"
        px_col = "close" if "close" in df_all.columns else "price"

        ts = pd.to_datetime(df_all[ts_col])
        if ts.dt.tz is not None:
            ts = ts.dt.tz_localize(None)

        panel = pd.DataFrame({
            "ts": ts,
            "contract": df_all["symbol"].astype(str),
            "price": pd.to_numeric(df_all[px_col], errors="coerce"),
        })
        if "volume" in df_all.columns:
            panel["volume"] = pd.to_numeric(df_all["volume"], errors="coerce")

        panel = panel.drop_duplicates(subset=["ts", "contract"]).reset_index(drop=True)

        meta = _build_meta(panel)

        print(f"\nDone: {len(panel):,} rows, {panel['contract'].nunique()} contracts")
        return ParentFetchResult(panel=panel, meta=meta)

    def estimate_cost(
        self,
        product: str,
        start: str,
        end: str,
        n_contracts: int = 20,
        frequency: str = "quarterly",
    ) -> float:
        """Estimate Databento cost for a windowed fetch. Returns cost in USD.

        Uses the same windowing logic as ``fetch_contracts`` so the
        estimate matches the actual fetch cost.
        """
        try:
            import databento as db  # type: ignore
        except ImportError as exc:
            raise ImportError("Install optional dependency: pip install futcurves[databento]") from exc

        step = relativedelta(months=3) if frequency == "quarterly" else relativedelta(months=1)
        client = db.Historical(key=self.api_key)

        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        total = 0.0

        while s < e:
            w_end = min(s + step, e)
            syms = _front_n_symbols(product, s, n_contracts, frequency)
            cost = client.metadata.get_cost(
                dataset=self.dataset,
                symbols=syms,
                schema=self.schema,
                stype_in="raw_symbol",
                start=s.strftime("%Y-%m-%d"),
                end=w_end.strftime("%Y-%m-%d"),
            )
            total += cost
            s = w_end

        return total
