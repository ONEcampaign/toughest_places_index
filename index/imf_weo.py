from dataclasses import dataclass

import pandas as pd
from weo import WEO, all_releases, download

from index.config import PATHS


def _check_weo_parameters(
    latest_y: int | None = None, latest_r: int | None = None
) -> (int, int):
    """Check parameters and return max values or provided input"""
    if latest_y is None:
        latest_y = max(*all_releases())[0]

    # if latest release isn't provided, take max value
    if latest_r is None:
        latest_r = max(*all_releases())[1]

    return latest_y, latest_r


def _update_weo(latest_y: int = None, latest_r: int = None) -> None:
    """Update data from the World Economic Outlook, using WEO package"""

    latest_y, latest_r = _check_weo_parameters(latest_y, latest_r)

    # Download the file from the IMF website and store in directory
    download(
        latest_y,
        latest_r,
        directory=PATHS.data,
        filename=f"weo{latest_y}_{latest_r}.csv",
    )


def _clean_number(number: str) -> float:
    """Clean a number and return as float"""
    import re
    import numpy as np

    if not isinstance(number, str):
        number = str(number)

    number = re.sub(r"[^\d.]", "", number)

    if number == "":
        return np.nan

    return float(number)


@dataclass
class IMFData:
    indicator: str | None = None
    data: pd.DataFrame = None

    @staticmethod
    def update(**kwargs) -> None:
        """Update the stored WEO data, using WEO package.
        Optionally provide a WEO release year `latest_y` and release `latest_r`"""

        _update_weo(**kwargs)

    def load_data(
        self, latest_y: int | None = None, latest_r: int | None = None
    ) -> None:
        """loading WEO as a clean dataframe"""

        latest_y, latest_r = _check_weo_parameters(latest_y, latest_r)

        names = {
            "ISO": "iso_code",
            "WEO Subject Code": "indicator",
            "Subject Descriptor": "indicator_name",
            "Units": "units",
            "Scale": "scale",
        }
        to_drop = [
            "WEO Country Code",
            "Country",
            "Subject Notes",
            "Country/Series-specific Notes",
            "Estimates Start After",
        ]

        df = WEO(PATHS.data + rf"/weo{latest_y}_{latest_r}.csv").df

        self.data = (
            df.drop(to_drop, axis=1)
            .rename(columns=names)
            .melt(id_vars=names.values(), var_name="date", value_name="value")
            .assign(
                year=lambda d: pd.to_datetime(d.date, format="%Y"),
                value=lambda d: d.value.apply(_clean_number),
            )
            .dropna(subset=["value"])
            .drop("date", axis=1)
            .reset_index(drop=True)
        )

    def _get_indicator(self, indicator: str) -> pd.DataFrame:
        """Get a specified imf indicator from the downloaded WEO file"""

        if self.data is None:
            self.load_data()

        return (
            self.data.loc[lambda d: d.indicator == indicator]
            .filter(["iso_code", "year", "value"], axis=1)
            .sort_values(["iso_code", "year"])
            .reset_index(drop=True)
        )

    def get_gdp_current(self) -> pd.DataFrame:
        """Indicator NGDPD, Current"""
        return self._get_indicator("NGDPD").assign(
            indicator="Gross domestic product, current prices"
        )

    def get_general_gov_expenditure(self) -> pd.DataFrame:
        """Indicator GGX, Current"""
        return self._get_indicator("GGX").assign(
            indicator="General government total expenditure"
        )

    def get_general_gov_revenue(self) -> pd.DataFrame:
        """Indicator GGR, Current"""
        return self._get_indicator("GGR").assign(indicator="General government revenue")

    def inflation_acp(self) -> pd.DataFrame:
        """Indicator PCPI, Index"""
        return self._get_indicator("PCPI").assign(
            indicator="Inflation, average consumer prices"
        )

    def inflation_epcp(self) -> pd.DataFrame:
        """Indicator PCPIE, Index"""
        return self._get_indicator("PCPIE").assign(
            indicator="Inflation, end of period consumer prices"
        )

    def inflation_gdp(self) -> pd.DataFrame:
        return self._get_indicator("NGDP_D").assign(indicator="GDP, deflator")
