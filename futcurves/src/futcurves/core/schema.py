from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

META_REQUIRED_COLUMNS = {"contract", "expiry"}
PANEL_REQUIRED_COLUMNS = {"ts", "contract", "price"}


@dataclass(frozen=True)
class ValidationConfig:
    require_datetime_meta: bool = True
    require_datetime_panel: bool = True


def _missing_columns(df: pd.DataFrame, required: Iterable[str]) -> set[str]:
    return set(required).difference(df.columns)


def validate_meta(meta: pd.DataFrame, cfg: ValidationConfig | None = None) -> pd.DataFrame:
    cfg = cfg or ValidationConfig()
    missing = _missing_columns(meta, META_REQUIRED_COLUMNS)
    if missing:
        raise ValueError(f"meta missing required columns: {sorted(missing)}")
    out = meta.copy()
    out["contract"] = out["contract"].astype(str)
    out["expiry"] = pd.to_datetime(out["expiry"])
    if cfg.require_datetime_meta and out["expiry"].isna().any():
        raise ValueError("meta.expiry contains invalid datetimes")
    for col in ("last_trade_date", "first_notice_date", "first_trade_date"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col])
    return out


def validate_panel(panel: pd.DataFrame, cfg: ValidationConfig | None = None) -> pd.DataFrame:
    cfg = cfg or ValidationConfig()
    missing = _missing_columns(panel, PANEL_REQUIRED_COLUMNS)
    if missing:
        raise ValueError(f"panel missing required columns: {sorted(missing)}")
    out = panel.copy()
    out["ts"] = pd.to_datetime(out["ts"])
    out["contract"] = out["contract"].astype(str)
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    if cfg.require_datetime_panel and out["ts"].isna().any():
        raise ValueError("panel.ts contains invalid datetimes")
    return out

