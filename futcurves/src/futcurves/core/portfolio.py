from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from futcurves.core.curve import Holdings


@dataclass(frozen=True)
class Order:
    contract: str
    notional: float


def position_to_contract_orders(
    holdings: Holdings,
    date: str | pd.Timestamp,
    position: int,
    target_notional: float,
    as_dataclasses: bool = False,
) -> dict[str, float] | list[Order]:
    d = pd.Timestamp(date)
    if d not in holdings:
        raise KeyError(f"date {d.date()} not in holdings")
    pos_map = holdings[d].get(position)
    if pos_map is None:
        raise KeyError(f"position {position} not in holdings[{d.date()}]")
    orders = {c: float(w * target_notional) for c, w in pos_map.items()}
    if as_dataclasses:
        return [Order(contract=c, notional=n) for c, n in orders.items()]
    return orders
