"""A class which deals with individual indicator data. Essentially a wrapper around
 a Pandas DataFrame which takes care of a few basic operations that we may need to do
 on each dataset (like rescaling/normalising, getting a few descriptive stats)"""

from dataclasses import dataclass
import pandas as pd

REQUIRED_COLS: list = ["iso_code", "value"]


@dataclass
class Indicator:
    data: pd.DataFrame
    indicator_name: str

    @staticmethod
    def __check_data(df: pd.DataFrame) -> None:
        if not set(REQUIRED_COLS).issubset(set(df.columns)):
            raise KeyError(
                f"{set(REQUIRED_COLS).difference(set(df.columns))} "
                f"missing from DataFrame columns"
            )

    def __post_init__(self):
        self.__check_data(self.data)
        self.data = self.data.filter(REQUIRED_COLS, axis=1)

    def get_data(self) -> pd.DataFrame:
        """Return the stored data"""
        return self.data
