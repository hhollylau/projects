from __future__ import annotations

import pandas as pd

from futcurves.core.roll import RollPolicy


def test_roll_weights_smoothstep() -> None:
    rp = RollPolicy(weight_fn="smoothstep")
    rs = pd.Timestamp("2024-01-01")
    re = pd.Timestamp("2024-01-11")

    assert rp.weight_next(pd.Timestamp("2023-12-31"), rs, re) == 0.0
    assert rp.weight_next(pd.Timestamp("2024-01-12"), rs, re) == 1.0
    assert abs(rp.weight_next(pd.Timestamp("2024-01-06"), rs, re) - 0.5) < 1e-12
