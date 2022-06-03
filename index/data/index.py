from dataclasses import dataclass, field
from index.data.dimension import Dimension
import pandas as pd


@dataclass
class Index:
    dimensions: list = field(default_factory=list[Dimension])

    def add_dimension(self, indicator: Dimension):
        """Add an indicator to the dataframe"""
        self.dimensions.append(indicator)

    def get_data(self, orient="wide") -> pd.DataFrame:
        """Return the stored data. An orientation can be passed ('wide' or 'long')"""

        if orient not in ["wide", "long"]:
            raise ValueError(f"Orientation must be 'wide' or 'long' but got {orient}")

        df = pd.DataFrame()

        for dimension in self.dimensions:
            _ = dimension.get_data(orient=orient)

            if orient == "wide":
                df = pd.concat(
                    [df, _.set_index('iso_code')], axis=1
                )
            else:
                df = pd.concat([df, _], axis=0)

        return df
