from __future__ import annotations

import pandas as pd
import re

from futcurves.sources.databento_source import (
    _third_wednesday,
    _infer_expiry,
    _build_meta,
    _date_batches,
)


def test_third_wednesday_known_dates() -> None:
    # March 2025: 1st is Saturday, 1st Wed is 5th, 3rd Wed is 19th
    assert _third_wednesday(2025, 3) == pd.Timestamp("2025-03-19")
    # June 2025: 1st is Sunday, 1st Wed is 4th, 3rd Wed is 18th
    assert _third_wednesday(2025, 6) == pd.Timestamp("2025-06-18")
    # September 2024: 1st is Sunday, 1st Wed is 4th, 3rd Wed is 18th
    assert _third_wednesday(2024, 9) == pd.Timestamp("2024-09-18")
    # December 2026: 1st is Tuesday, 1st Wed is 2nd, 3rd Wed is 16th
    assert _third_wednesday(2026, 12) == pd.Timestamp("2026-12-16")


def test_infer_expiry_same_decade() -> None:
    # SR3H5 last seen trading in Dec 2024 -> expiry March 2025
    result = _infer_expiry("SR3H5", pd.Timestamp("2024-12-01"))
    assert result == pd.Timestamp("2025-03-19")


def test_infer_expiry_crosses_decade() -> None:
    # SR3H5 last seen trading in April 2025 -> March 2025 already passed,
    # so must be March 2035
    result = _infer_expiry("SR3H5", pd.Timestamp("2025-04-01"))
    assert result == _third_wednesday(2035, 3)


def test_infer_expiry_pre_2020() -> None:
    # SR3H8 last seen trading in 2017 -> expiry March 2018, not 2028
    result = _infer_expiry("SR3H8", pd.Timestamp("2017-06-15"))
    assert result == _third_wednesday(2018, 3)


def test_infer_expiry_digit_zero() -> None:
    # SR3Z0 last seen trading in 2029 -> December 2030, not 2020
    result = _infer_expiry("SR3Z0", pd.Timestamp("2029-06-01"))
    assert result == _third_wednesday(2030, 12)


def test_date_batches_basic() -> None:
    batches = _date_batches("2015-01-01", "2026-01-01", batch_years=2)
    assert len(batches) == 6
    # First batch starts at start
    assert batches[0][0] == "2015-01-01"
    # Last batch ends at end
    assert batches[-1][1] == "2026-01-01"
    # Batches are contiguous: each end == next start
    for i in range(len(batches) - 1):
        assert batches[i][1] == batches[i + 1][0]


def test_date_batches_short_range() -> None:
    batches = _date_batches("2024-01-01", "2024-06-01", batch_years=2)
    assert len(batches) == 1
    assert batches[0] == ("2024-01-01", "2024-06-01")


def test_build_meta_synthetic() -> None:
    panel = pd.DataFrame({
        "ts": pd.to_datetime([
            "2024-06-01", "2024-09-01", "2024-12-01",
            "2024-06-01", "2024-09-01", "2025-02-01",
        ]),
        "contract": ["SR3H5", "SR3H5", "SR3H5", "SR3M5", "SR3M5", "SR3M5"],
        "price": [95.5, 95.6, 95.7, 95.3, 95.4, 95.5],
    })

    meta = _build_meta(panel)

    assert len(meta) == 2
    assert set(meta["contract"]) == {"SR3H5", "SR3M5"}

    h5_row = meta[meta["contract"] == "SR3H5"].iloc[0]
    assert h5_row["expiry"] == _third_wednesday(2025, 3)

    m5_row = meta[meta["contract"] == "SR3M5"].iloc[0]
    assert m5_row["expiry"] == _third_wednesday(2025, 6)

    # Meta should be sorted by expiry
    assert meta.iloc[0]["contract"] == "SR3H5"
    assert meta.iloc[1]["contract"] == "SR3M5"

    # last_trade_date should be 2 bdays before expiry
    assert "last_trade_date" in meta.columns


def test_contract_filter_regex() -> None:
    pattern = r"^SR3[HMUZ]\d$"
    # Should match
    assert re.match(pattern, "SR3H5")
    assert re.match(pattern, "SR3Z0")
    assert re.match(pattern, "SR3M9")
    # Should not match (spreads, packs, non-quarterly)
    assert not re.match(pattern, "SR3H5-SR3M5")
    assert not re.match(pattern, "SR3H25")
    assert not re.match(pattern, "SR3F5")  # F is not quarterly
