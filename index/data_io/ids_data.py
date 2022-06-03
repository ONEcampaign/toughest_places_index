import os
from dataclasses import dataclass, field

import pandas as pd
from pyjstat import pyjstat

from index import common
from index.config import PATHS, DEBT_SERVICE_IDS
from index.data_io.imf_weo import WEOdata


def _time_period(start_year: int, end_year: int) -> str:
    """Take a period range and convert it to an API compatible string"""

    time_period = ""

    for y in range(start_year, end_year + 1):
        if y < end_year:
            time_period += f"yr{y};"
        else:
            time_period += f"yr{y}"

    return time_period


def _country_list(countries: str | list[str]) -> str:
    """Take a country list amd convert it to an API compatible string"""

    country_list = ""

    if isinstance(countries, str):
        return countries

    for c in countries:
        country_list += f"{c};"

    return country_list[:-1]


def _api_url(
    indicator: str,
    countries: str | list[str],
    start_year: int,
    end_year: int,
    source: int,
) -> str:
    """Query string for API for IDS data. One indicator at a time"""

    if not isinstance(indicator, str):
        raise TypeError("Must pass single indicator (as string) at a time")

    countries = _country_list(countries)
    time_period = _time_period(start_year, end_year)

    return (
        "http://api.worldbank.org/v2/"
        f"sources/{source}/country/{countries}/"
        f"series/{indicator}/time/{time_period}/"
        f"data?format=jsonstat"
    )


@dataclass
class IDSData:
    """Class to fetch IDS data"""

    indicators: str | list[str] | dict[str, str]
    countries: str | list[str] = "all"
    start_year: int = 2017
    end_year: int = 2025
    source: int = 6
    save_as: str = "ids_clean.csv"
    data: pd.DataFrame = None
    api_urls: str | list[str] = field(default_factory=list[str])

    def __post_init__(self) -> None:
        """Populate the api_urls based on the passed parameters"""
        for _ in self.indicators:
            self.api_urls.append(
                _api_url(
                    indicator=_,
                    countries=self.countries,
                    start_year=self.start_year,
                    end_year=self.end_year,
                    source=self.source,
                )
            )

    def download_data(self):
        """Download the data from the API"""

        df = pd.DataFrame()

        for url in self.api_urls:
            _ = (
                pyjstat.Dataset.read(url)
                .write(output="dataframe")
                .loc[lambda d: d.value.notna()]
                .assign(series_code=url.split("series/")[1][:14])
                .reset_index(drop=True)
            )

            df = pd.concat([df, _], ignore_index=True)

        df.to_csv(rf"{PATHS.data}/{self.save_as}", index=False)

        return self

    def load_raw_data(self):
        """Read the data from the downloaded file"""
        self.data = pd.read_csv(rf"{PATHS.data}/{self.save_as}", parse_dates=["time"])

        return self

    def clean_ids_data(self, detail: bool = False, names_dict: None | dict = None):
        """Clean the raw IDS data"""

        if self.data is None:
            raise FileNotFoundError("Data must be downloaded first")

        if (names_dict is None) and (not isinstance(self.indicators, dict)):
            raise TypeError("Must pass names_dict if not passing dict of indicators")

        if detail:
            self.data = self.data.copy().loc[lambda d: d["counterpart-area"] != "World"]
        else:
            self.data = self.data.copy().loc[lambda d: d["counterpart-area"] == "World"]

        self.data = (
            self.data.rename(
                columns={"time": "year", "counterpart-area": "counterpart"}
            )
            .pipe(common.add_iso_codes, id_col="country")
            .loc[lambda d: d.iso_code != "not found"]
            .assign(
                indicator=lambda d: d.series_code.map(
                    names_dict if names_dict is not None else self.indicators
                )
            )
            .groupby(["iso_code", "year", "indicator", "counterpart"], as_index=False)[
                "value"
            ]
            .sum()
        )

        return self

    def get_clean_data(
        self, detail: bool = False, names_dict: None | dict = None
    ) -> pd.DataFrame:
        """Get the clean data"""

        if not os.path.exists(rf"{PATHS.data}/{self.save_as}"):
            self.download_data()
            print("Downloaded data")

        self.load_raw_data().clean_ids_data(detail=detail, names_dict=names_dict)

        return self.data


def get_service_spending_ratio(year: int) -> pd.DataFrame:
    """Get the service/spending ratio for a given year"""

    # Create and IDS data object for debt service
    ids_data = IDSData(
        indicators=DEBT_SERVICE_IDS,
        countries="all",
        start_year=year,
        end_year=year,
        source=6,
        save_as=f"ids_service_{year}-{year}.csv",
    )

    # Get the debt service data, for 'World' counterpart area
    service = ids_data.get_clean_data(detail=False)

    # Calculate total service (i.e. not by creditor type)
    service = service.groupby(["iso_code", "year"], as_index=False).sum()

    # ---spending----

    # Create WEOData object
    weo_data = WEOdata()

    # Get the implied exchange dates from WEO
    exchange = weo_data.get_exchange()

    # Get and clean the spending data (from LCU billion to USD)
    spending = (
        weo_data.get_general_gov_expenditure()
        .loc[lambda d: d.year.dt.year <= year]
        .pipe(common.get_latest, by=["iso_code"], date_col="year")
        .assign(value=lambda d: d.value * 1e9)  # from billions to units
        .merge(exchange, on=["iso_code", "year"], how="left")
        .assign(value=lambda d: d.value * d.xe)  # in USD
        .drop(columns=["xe", "indicator", "year"])
    )

    # Combine the datasets and calculate the ratio
    df = service.merge(
        spending, on=["iso_code"], how="left", suffixes=("_service", "_spending")
    ).assign(ratio=lambda d: round(100 * d.value_service / d.value_spending, 2))

    return df
