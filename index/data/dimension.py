"""dataclass which should store data for two+ indicators. It should also take care of
things like assessing the similarity/correlation between the loaded indicators,
checking for data completeness, dealing with imputations (at dimension level) etc."""


from dataclasses import dataclass, field
from index.data.indicator import Indicator
import pandas as pd


@dataclass
class Dimension:
    indicators: list = field(default_factory=list[Indicator])
    dimension_name: str | None = None

    def add_indicator(self, indicator: Indicator):
        """Add an indicator to the dataframe"""
        self.indicators.append(indicator)

    def get_data(self, orient="wide") -> pd.DataFrame:
        """Return the stored data. An orientation can be passed ('wide' or 'long')"""

        df = pd.DataFrame()

        for indicator in self.indicators:
            df = pd.concat(
                [df, indicator.get_data().assign(indicator=indicator.indicator_name)],
                ignore_index=True,
            )

        if orient == "wide":
            return df.pivot(
                index="iso_code", columns="indicator", values="value"
            ).reset_index(drop=False)

        elif orient == "long":
            return df.reset_index(drop=True)

        else:
            raise ValueError(f"Orientation must be 'wide' or 'long' but got {orient}")

    def get_indicators(self) -> list:
        """Return the stored indicators"""
        return self.indicators
