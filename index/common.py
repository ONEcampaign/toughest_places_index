"""Utility Functions"""
import country_converter
import os

import country_converter as cc
import pandas as pd
import wbgapi as wb

from index.config import PATHS


def add_short_names(
    df: pd.DataFrame,
    id_col: str = "iso_code",
    target_col: str = "country_name",
) -> pd.DataFrame:
    """Add shortnames column to DataFrame
    :param df: str
        A Pandas DataFrame with an id column (iso2, iso3, un code, etc)
    :param id_col: str
        Name of the column which contains the country id.
    :param target_col: str
        Name of the column which will store the short country name.
    :return:
        A Pandas DataFrame with a new column containing the short country name.
    """
    # Create a coco object
    coco = cc.CountryConverter()

    # Get the shortnames
    df[target_col] = coco.convert(df[id_col], to="short_name")

    return df


def add_iso_codes(
    df: pd.DataFrame,
    id_col: str = "country_name",
    target_col: str = "iso_code",
) -> pd.DataFrame:
    """Add iso_codes column to DataFrame
    :param df: str
        A Pandas DataFrame with a country name (or other id not ISO3)
    :param id_col: str
        Name of the column which contains the country name.
    :param target_col: str
        Name of the column which will store the iso_code.
    :return:
        A Pandas DataFrame with a new column containing the iso_code.
    """
    # Create a coco object
    coco = cc.CountryConverter()

    # Get the shortnames
    df[target_col] = coco.convert(df[id_col], to="ISO3")

    return df


def get_latest(
    df: pd.DataFrame, by: list | str, date_col: str = "date"
) -> pd.DataFrame:
    """Get the latest value, grouping by columns specified in 'by'"""
    if isinstance(by, str):
        by = [by]
    return df.sort_values(by=by + [date_col]).groupby(by, as_index=False).last()


def _get_wb_data(series: str, years: int):
    """Get data for an indicator, using wbgapi"""

    # get data
    return wb.data.DataFrame(
        series,
        mrnev=years,
        numericTimeKeys=True,
        labels=False,
        columns="series",
        timeColumns=True,
    ).reset_index()


def _clean_wb_data(df, series, series_name):
    """Clean a dataframe obtained through the WBG api"""

    return (
        df.rename(
            columns={
                "economy": "iso_code",
                series: "value",
                "time": "date",
                f"{series}:T": "date",
            }
        )
        .assign(
            indicator=series_name, date=lambda d: pd.to_datetime(d.date, format="%Y")
        )
        .sort_values(by=["iso_code", "date"])
        .reset_index(drop=True)
    )


def wb_data(
    series: str, series_name: str, years: int, download: bool = False
) -> pd.DataFrame:
    """Get data for a specific WB indicator"""

    # Prioritise not downloading the data, unless specified
    if not download:
        try:
            return pd.read_csv(
                rf"{PATHS.data}/wb_{series}_{years}_years.csv", parse_dates=["date"]
            )
        except FileNotFoundError:
            return wb_data(series, series_name, years, download=True)

    df = _get_wb_data(series, years).pipe(_clean_wb_data, series, series_name)

    df.to_csv(rf"{PATHS.data}/wb_{series}_{years}_years.csv", index=False)

    return df


def get_population_dict() -> dict:
    """Loads population data and creates a dictionary"""

    return (
        wb_data("SP.POP.TOTL", "Total Population", years=50)
        .filter(["iso_code", "date", "value"], axis=1)
        .sort_values(["iso_code", "date"])
        .drop_duplicates(subset=["iso_code"], keep="last")
        .set_index("iso_code")["value"]
        .astype("int32")
        .to_dict()
    )


def add_population_column(
    df: pd.DataFrame,
    id_col: str = "iso_code",
    target_col: str = "population",
) -> pd.DataFrame:
    """Add a population column to DataFrame
    :param df: str
        A Pandas DataFrame with an id column (iso2, iso3, un code, etc)
    :param id_col: str
        Name of the column which contains the country id.
    :param target_col: str
        Name of the column which will store the population data.
    :return:
        A Pandas DataFrame with a new column containing the population data column.
    """

    # Get the shortnames
    df[target_col] = df[id_col].map(get_population_dict())

    return df


def add_share_of_population(
    df: pd.DataFrame,
    id_col: str = "iso_code",
    value_col: str = "value",
    target_col: str = "population_share",
) -> pd.DataFrame:
    """Add a share of population column to DataFrame
    :param df: str
        A Pandas DataFrame with an id column (iso2, iso3, un code, etc)
    :param id_col: str
        Name of the column which contains the country id.
    :param value_col: str
        Name of the column which contains the value to be converted to share.
    :param target_col: str
        Name of the column which will store the population data.
    :return:
        A Pandas DataFrame with a new column containing the population_share column.
    """

    df = df.pipe(add_population_column, id_col=id_col)
    df[target_col] = round(100 * df[value_col] / df.population, 2)

    return df.drop(columns=["population"])


def read_and_append(
    new_data: pd.DataFrame, file_path: str, date_col: str = "date", idx: list = None
) -> pd.DataFrame:
    """Read and append data to a file. This removes duplicates by date."""

    if idx is None:
        idx = ["iso_code"]
    # Read file
    saved = pd.read_csv(rf"{file_path}", parse_dates=[date_col])

    # Append new data
    data = pd.concat([saved, new_data], ignore_index=True)

    return (
        data.sort_values(by=idx + [date_col])
        .drop_duplicates(subset=idx + [date_col], keep="last")
        .reset_index(drop=True)
    )


def _download_income_levels() -> None:
    """Downloads fresh version of income levels from WB"""
    url = "https://databank.worldbank.org/data/download/site-content/CLASS.xlsx"

    df = pd.read_excel(
        url,
        sheet_name="List of economies",
        usecols=["Code", "Income group"],
        na_values=None,
    )

    df = df.dropna(subset=["Income group"])

    df.to_csv(PATHS.data + r"/income_levels.csv", index=False)


def income_levels() -> dict:
    """Return income level dictionary"""
    path = f"{PATHS.data}/income_levels.csv"

    if not os.path.exists(path):
        _download_income_levels()

    return pd.read_csv(path, na_values=None, index_col="Code")["Income group"].to_dict()


def add_income_levels(
    df: pd.DataFrame,
    id_col: str = "iso_code",
    target_col: str = "income_level",
) -> pd.DataFrame:
    """Add an income level column to DataFrame
    :param df: str
        A Pandas DataFrame with an id column (iso2, iso3, un code, etc)
    :param id_col: str
        Name of the column which contains the country id.
    :param target_col: str
        Name of the column which will store the income level data.
    :return:
        A Pandas DataFrame with a new column containing the income level data column.
    """
    # Create a coco object
    coco = cc.CountryConverter()

    return (
        df.assign(
            id_=lambda d: coco.convert(d[id_col], to="ISO3"),
            income_level=lambda d: d["id_"].map(income_levels()),
        )
        .drop(columns=["id_"])
        .rename(columns={"income_level": target_col})
    )


def keep_only_valid_iso(df: pd.DataFrame) -> pd.DataFrame:

    valid_iso = cc.CountryConverter().get_correspondence_dict("ISO3", "name_short")

    missing = {i for i in df.iso_code.unique() if i not in valid_iso}

    if len(missing) > 0:
        print(f"The following iso_codes were dropped: \n {missing}")

    return df.loc[lambda d: d.iso_code.isin(valid_iso)].reset_index(drop=True)


def _study_countries() -> list:
    """Return a list of the countries being studied"""

    # for now, for example, no_hics
    countries = _no_hics_umics()
    [
        countries.append(c)
        for c in [
            "DOM",
            "JOR",
            "GTM",
            "ECU",
            "TUR",
            "BWA",
            "PER",
            "BRA",
            "KAZ",
            "ARG",
            "PRY",
            "ROU",
            "MDA",
            "ARM",
            "SRB",
            "THA",
            "PAN",
            "BGR",
            "CRI",
            "COL",
            "CHN",
            "ZAF",
            "BLR",
            "FJI",
        ]
    ]
    return countries

def _all_countries() -> list:
    """return all countries"""

    return list(income_levels().keys())

def _no_hics_umics() -> list:
    """Return a list of countries excluding hics and umics"""
    return [
        c
        for c, lvl in income_levels().items()
        if lvl not in ["High income", "Upper middle income"]
    ]


def _no_hics() -> list:
    """Return a list of countries excluding hics and umics"""
    return [c for c, lvl in income_levels().items() if lvl not in ["High income"]]


def _africa_countries() -> list:
    """Return a list of African countries"""
    countries = cc.CountryConverter().get_correspondence_dict("ISO3", "continent")
    return [c for c, cnt in countries.items() if cnt[0] == "Africa"]


def _lics_countries() -> list:
    """Return a list of LICS countries"""
    return [c for c, lvl in income_levels().items() if lvl == "Low income"]


def _lics_lmics_countries() -> list:
    """Return a list of LICS and LMICs countries"""
    return [
        c
        for c, lvl in income_levels().items()
        if lvl in ["Low income", "Lower middle income"]
    ]


STUDY_COUNTRIES: dict[str, list] = {
    "all_countries": _all_countries(),
    "study": _study_countries(),
    "no_hics": _no_hics(),
    "no_hics_umics": _no_hics_umics(),
    "africa": _africa_countries(),
    "lics": _lics_countries(),
    "lics_lmics": _lics_lmics_countries(),
}
