# futcurves

Static "today's front 20 backfilled" history is wrong for futures curves because the historical listed set changes over time. `futcurves` builds a **date-aware dynamic rolling universe** and returns a **tradable strip curve** with a parallel **holdings map** so signals like "sell 5th position" become concrete contract weights.

## What it returns
- `curve_px`: rectangular date x position matrix (`1..N`) of strip prices
- `holdings`: date -> position -> `{contract: weight}` mapping for executable orders

## Install

```bash
pip install -e .[dev]
```

Optional Databento adapter:

```bash
pip install -e .[dev,databento]
```

## Core usage

```python
import pandas as pd
from futcurves import RollPolicy, build_rolling_universe, build_strip_curve

meta = pd.read_csv("meta.csv")      # contract, expiry, optional last_trade_date
panel = pd.read_csv("panel.csv")    # ts, contract, price

universe = build_rolling_universe(
    meta=meta,
    start="2023-01-01",
    end="2026-01-01",
    n_positions=20,
    offset_bdays=2,
)

roll_policy = RollPolicy(
    roll_end_offset_bdays=2,
    roll_window_bdays=7,
    weight_fn="smoothstep",  # or linear/logistic
)

curve_px, holdings = build_strip_curve(
    panel=panel,
    universe=universe,
    meta=meta,
    n_positions=20,
    roll_policy=roll_policy,
)
```

## Roll policy and accelerated roll

`RollPolicy` defines:
- roll end: `last_trade_date - roll_end_offset_bdays` (fallback: `expiry - offset`)
- roll start: `roll_end - roll_window_bdays`
- blending weight from current to next contract

Supported weight functions:
- `smoothstep` (default): accelerated S-curve `3u^2 - 2u^3`
- `linear`
- `logistic` (normalized sigmoid, configurable `logistic_k`)

For tradable strip construction, when front is rolling, blend is applied uniformly across positions so the strip shifts smoothly.

## Orders from position signals

```python
from futcurves import position_to_contract_orders

orders = position_to_contract_orders(
    holdings=holdings,
    date="2025-06-10",
    position=5,
    target_notional=-5_000_000,
)
# e.g. {"SR3U5": -3_000_000, "SR3Z5": -2_000_000}
```

## Databento optional quickstart

### Continuous symbology (position-based)

Databento continuous symbology supports patterns such as `ES.v.0` where index is rank by roll rule and `v` indicates a volume-based roll rule; see Databento docs on continuous symbols and symbology.

Example adapter usage is in `examples/databento_quickstart.py` and fetches symbols like `SR3.v.0 ... SR3.v.19` with `stype_in="continuous"`.

### Parent symbology (all active contracts)

`fetch_parent` uses parent symbology (`SR3.FUT`) to fetch **all** active contracts, automatically builds `meta` (with inferred expiry dates) and `panel` DataFrames, and filters to single-leg quarterly contracts.

```python
from futcurves.sources.databento_source import DatabentoSource

src = DatabentoSource(api_key=key, dataset="GLBX.MDP3")

# Check cost first
cost = src.estimate_cost("SR3", "2015-01-01", "2026-01-01")
print(f"Estimated cost: ${cost:.2f}")

# Fetch — optional batch_years splits into smaller API calls
panel, meta = src.fetch_parent(
    "SR3",
    start="2015-01-01",
    end="2026-01-01",
    batch_years=2,  # fetch in 2-year chunks (omit for single fetch)
)
```

Expiry dates are inferred from the last observed trade date for each contract, which correctly handles decade-wrapping year digits (e.g. SR3H0 = 2030, not 2020).

## Examples
- `examples/sofr_strip_pca_demo.py`
- `examples/databento_quickstart.py`

## Tests

```bash
pytest -q
```
