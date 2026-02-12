from __future__ import annotations

import numpy as np
import pandas as pd

from futcurves import RollPolicy, build_rolling_universe, build_strip_curve, position_to_contract_orders


start = "2024-01-02"
end = "2025-12-31"
n_positions = 20

meta = pd.DataFrame(
    {
        "contract": [f"SR3_{i:02d}" for i in range(1, 33)],
        "expiry": pd.date_range("2024-03-20", periods=32, freq="3MS") + pd.Timedelta(days=19),
        "last_trade_date": pd.date_range("2024-03-20", periods=32, freq="3MS") + pd.Timedelta(days=17),
        "root": "SR3",
    }
)

universe = build_rolling_universe(meta, start, end, n_positions=n_positions)

all_dates = pd.date_range(start, end, freq="B")
rows = []
rng = np.random.default_rng(42)
for c_idx, c in enumerate(meta["contract"].tolist()):
    base = 95.0 + 0.03 * c_idx
    noise = np.cumsum(rng.normal(0, 0.01, size=len(all_dates)))
    px = base + noise
    for d, p in zip(all_dates, px):
        rows.append((d, c, p))
panel = pd.DataFrame(rows, columns=["ts", "contract", "price"])

roll_policy = RollPolicy()  # linear blend over exact gap between consecutive expiries
curve_px, holdings = build_strip_curve(panel, universe, meta, n_positions=n_positions, roll_policy=roll_policy)

rets = curve_px.pct_change().dropna()
lookback = 252
signals = pd.Series(0.0, index=rets.index)
for i in range(lookback, len(rets)):
    w = rets.iloc[i - lookback : i].dropna(axis=1)
    if w.shape[1] < 5:
        continue
    cov = np.cov(w.values.T)
    evals, evecs = np.linalg.eigh(cov)
    pc1 = evecs[:, np.argmax(evals)]
    today = rets.iloc[i][w.columns].values
    signals.iloc[i] = float(np.dot(today, pc1))

threshold = signals.rolling(252).std().fillna(0) * 1.5
position_signal = pd.Series(0.0, index=signals.index)
position_signal[signals > threshold] = -1.0
position_signal[signals < -threshold] = 1.0

notional = 1_000_000.0
example_date = position_signal.index[-1]
orders = position_to_contract_orders(holdings, example_date, position=5, target_notional=notional * position_signal.loc[example_date])
print("Example date:", example_date.date())
print("Signal:", position_signal.loc[example_date])
print("Position-5 underlying orders:", orders)

pnl = []
for d in position_signal.index[:-1]:
    d_next = position_signal.index[position_signal.index.get_loc(d) + 1]
    sig = position_signal.loc[d]
    if sig == 0:
        pnl.append(0.0)
        continue
    r = rets.loc[d_next, 5] if 5 in rets.columns else 0.0
    pnl.append(sig * notional * float(r))

print("Naive cumulative PnL:", float(np.sum(pnl)))
