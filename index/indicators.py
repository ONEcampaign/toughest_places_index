"""Create individual indicator DataFrames"""

import pandas as pd

from index.common import get_latest, add_share_of_population
from index.config import DEBT_SERVICE_IDS
from index.data_io.economist_fs import EconomistIndex
from index.data_io.ids_data import IDSData
from index.data_io.imf_weo import WEOdata
from index.data_io.wb_data import wb_reserves, wb_wasting
from index.data_io.wfp_food_sec import (
    read_hunger_data,
    read_nutrition_data,
    create_wfp_database,
)
from index.data_io.wfp_inflation import read_inflation_data, refresh_inflation_data


# --------------------------------------------------------------------------------------
#                            Dimension 1
# --------------------------------------------------------------------------------------


def get_insufficient_food(refresh: bool = False) -> pd.DataFrame:
    """Latest insufficient food data, as share of population"""

    # If refresh, refresh the hunger and nutrition databases
    if refresh:
        create_wfp_database()

    # Read the database, get the latest values, and transform to share of population
    return (
        read_hunger_data()
        .pipe(get_latest, by="iso_code")
        .filter(["iso_code", "date", "value"], axis=1)
        .pipe(add_share_of_population, target_col="value")
    )


def get_inflation(refresh: bool = False, data_type: str = "headline") -> pd.DataFrame:
    """Latest headline inflation data from WFP. 'headline' or 'food'"""

    if refresh:
        refresh_inflation_data()

    if data_type == "headline":
        data_type = "Inflation Rate"
    elif data_type == "food":
        data_type = "Food Inflation"
    else:
        raise ValueError("Data type must be 'headline' or 'food'")

    return (
        read_inflation_data()
        .loc[lambda d: d.indicator == data_type]
        .pipe(get_latest, by="iso_code")
        .filter(["iso_code", "date", "value"], axis=1)
    )


# --------------------------------------------------------------------------------------
#                            Dimension 2
# --------------------------------------------------------------------------------------


def get_economist_index(refresh: bool = False) -> pd.DataFrame:
    """Get data from the Economist Index"""

    # Create an EconomistIndex object. Refresh the data if necessary
    econ = EconomistIndex(refresh=refresh)

    # Return just the overall index data
    return (
        econ.get_overall_data()
        .filter(["iso_code", "score"], axis=1)
        .assign(date=pd.to_datetime("2021-01-01"))
        .rename({"score": "value"}, axis=1)
    )


def get_wasting(refresh: bool = False) -> pd.DataFrame:
    """Get wasting data from WFP"""

    # Refresh the data if necessary
    if refresh:
        create_wfp_database()

    # Get the WB WDI version of the indicator
    wb_ = (
        wb_wasting(refresh=refresh)
        .pipe(get_latest, by="iso_code")
        .filter(["iso_code", "value"], axis=1)
    )

    # Get the scrapped version of the indicator
    wfp_ = (
        read_nutrition_data()
        .loc[lambda d: d.indicator == "wasting"]
        .filter(["iso_code", "value"], axis=1)
    )

    # Keep only the WB data that is missing from the WFP data
    wb_ = wb_.loc[lambda d: ~d.iso_code.isin(wfp_.loc[wfp_.value.notna()].iso_code)]

    return pd.concat([wfp_, wb_], ignore_index=True).dropna().reset_index(drop=True)


# --------------------------------------------------------------------------------------
#                            Dimension 3
# --------------------------------------------------------------------------------------


def get_fiscal_reserves(refresh: bool = False) -> pd.DataFrame:
    """Get the fiscal reserves minus gold data"""

    return (
        wb_reserves(refresh=refresh)
        .pipe(get_latest, by="iso_code")
        .filter(["iso_code", "date", "value"], axis=1)
    )


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
        .pipe(get_latest, by=["iso_code"], date_col="year")
        .assign(value=lambda d: d.value * 1e9)  # from billions to units
        .merge(exchange, on=["iso_code", "year"], how="left")
        .assign(value=lambda d: d.value * d.xe)  # in USD
        .drop(columns=["xe", "indicator", "year"])
    )

    # Combine the datasets and calculate the ratio
    return (
        service.merge(
            spending, on=["iso_code"], how="left", suffixes=("_service", "_spending")
        )
        .assign(ratio=lambda d: round(100 * d.value_service / d.value_spending, 2))
        .filter(["iso_code", "year","ratio"], axis=1)
        .rename({"ratio": "value", "year":"date"}, axis=1)
    )
