from dataclasses import dataclass
import pandas as pd
from typing import List

@dataclass
class DatasetConfig:
    """Configuration du dataset des vins."""
    data_raw: pd.DataFrame
    filename: str
    features_columns: List[str]
    target_columns: List[str]

    @property
    def data(self) -> pd.DataFrame:
        """Retourne le DataFrame."""
        return self.data_raw