"""A class which deals with individual indicator data. Essentially a wrapper around
 a Pandas DataFrame which takes care of a few basic operations that we may need to do
 on each dataset (like rescaling/normalising, getting a few descriptive stats)"""

from dataclasses import dataclass
import pandas as pd
from index.common import keep_only_valid_iso

REQUIRED_COLS: list = ["iso_code", "value"]


@dataclass
class Indicator:
    data: pd.DataFrame
    indicator_name: str
    countries_list: list[str] | None = None

    def __check_data(self, df: pd.DataFrame) -> None:
        if not set(REQUIRED_COLS).issubset(set(df.columns)):
            raise KeyError(
                f"{set(REQUIRED_COLS).difference(set(df.columns))} "
                f"missing from DataFrame columns"
            )
        if "date" in df.columns:
            self.dates: dict = df.set_index("iso_code")["date"].to_dict()
        else:
            self.dates: dict = {}

        if df.duplicated(subset="iso_code").sum() != 0:
            raise ValueError("Duplicate iso_code values in DataFrame")

    def __post_init__(self):
        # Check that the data is structured as required/expected
        self.__check_data(self.data)

        # Keep only data with valid iso codes
        self.data = self.data.pipe(keep_only_valid_iso)

        # Optionally reindex the dataframe so only countries under study are kept
        if self.countries_list is not None:
            self.data = (
                self.data.set_index("iso_code")
                .reindex(self.countries_list)
                .reset_index(drop=False)
            )
        # Keep only the required columns for analysis
        self.data = self.data.filter(REQUIRED_COLS, axis=1)

    def get_data(self, with_date: bool = False) -> pd.DataFrame:
        """Return the stored data"""
        if with_date:
            return self.data.assign(
                date=lambda d: pd.to_datetime(d.iso_code.map(self.dates))
            )
        else:
            return self.data
