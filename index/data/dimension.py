"""dataclass which should store data for two+ indicators. It should also take care of
things like assessing the similarity/correlation between the loaded indicators,
checking for data completeness, dealing with imputations (at dimension level) etc."""


from dataclasses import dataclass, field
from index.data.indicator import Indicator
import pandas as pd
from index.data.imputers import IMPUTERS


@dataclass
class Dimension:
    indicators: list = field(default_factory=list[Indicator])
    dimension_name: str | None = None
    data: pd.DataFrame = None

    def add_indicator(self, indicator: Indicator):
        """Add an indicator to the dataframe"""
        self.indicators.append(indicator)

    def rescale(self, scaler_name: str, **kwargs) -> None:
        """Rescale the data by calling on the indicator rescale method"""
        for indicator in self.indicators:
            indicator.rescale(scaler_name, **kwargs)

    def impute_missing_data(self, method: str, **kwargs) -> None:
        """Impute missing data at the dimension level"""
        if method not in IMPUTERS:
            raise ValueError(
                f"Method {method} not found in the available imputers. "
                f"Available imputers are {IMPUTERS.keys()}"
            )

        self.data = IMPUTERS[method](self.get_data(), **kwargs)

    def get_data(self, orient="wide", with_date: bool = False) -> pd.DataFrame:
        """Return the stored data. An orientation can be passed ('wide' or 'long')"""

        if self.data is not None:
            return self.data

        df = pd.DataFrame()

        for indicator in self.indicators:
            df = pd.concat(
                [
                    df,
                    indicator.get_data(with_date=with_date).assign(
                        indicator=indicator.indicator_name
                    ),
                ],
                ignore_index=True,
            )

        if orient == "wide":
            if with_date:
                raise ValueError("Cannot use with_date with wide orientation")

            self.data = df.pivot(
                index="iso_code", columns="indicator", values="value"
            ).reset_index(drop=False)

            return self.data

        elif orient == "long":
            self.data = df.reset_index(drop=True)

            return self.data

        else:
            raise ValueError(f"Orientation must be 'wide' or 'long' but got {orient}")

    def get_indicators(self) -> list:
        """Return the stored indicators"""
        return self.indicators
