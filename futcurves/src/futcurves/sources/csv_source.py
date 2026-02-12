from __future__ import annotations

import pandas as pd

from futcurves.core.schema import validate_meta, validate_panel


class CSVSource:
    def __init__(self, meta_path: str, panel_path: str):
        self.meta_path = meta_path
        self.panel_path = panel_path

    def load_meta(self) -> pd.DataFrame:
        df = pd.read_csv(self.meta_path)
        return validate_meta(df)

    def load_panel(self) -> pd.DataFrame:
        df = pd.read_csv(self.panel_path)
        return validate_panel(df)
