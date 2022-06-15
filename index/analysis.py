"""
This file contains the final analysis
"""
import copy

from index.data.index import Index
from index.indicators import (
    get_insufficient_food,
    get_inflation,
    get_wasting,
    get_economist_index,
    get_fiscal_reserves,
    get_service_spending_ratio,
)

from index.data.indicator import Indicator
from index.data.dimension import Dimension

from index.common import STUDY_COUNTRIES

from index.common import add_short_names
import pandas as pd

COUNTRIES = STUDY_COUNTRIES["study"]
COUNTRIES = [c for c in COUNTRIES if c not in ["KIR", "FSM", "PRK", "WSM"]]

RESCALE_PARAMETERS = {"scaler_name": "quantile"}
IMPUTE_PARAMETERS = {"method": "knn", "n_neighbors": 10}


def get_dimensions() -> tuple[Dimension, Dimension]:
    """Return a list of indicators"""

    insufficient_food = Indicator(
        get_insufficient_food(refresh=False),
        "Insufficient Food Consumption (% of population)",
        countries_list=COUNTRIES,
        more_is_worse=True,
    )

    # Impute missing hunger data with zeros.
    insufficient_food.raw_data = insufficient_food.raw_data.fillna(0)

    # Headline inflation
    inflation = Indicator(
        get_inflation(refresh=False),
        "Headline Inflation (%)",
        countries_list=COUNTRIES,
        more_is_worse=True,
    )

    wasting = Indicator(
        get_wasting(refresh=False),
        "Prevalence of Wasting (% of children under 5)",
        countries_list=COUNTRIES,
        more_is_worse=True,
    )

    dimension_1 = Dimension(indicators=[insufficient_food, inflation])
    dimension_2 = Dimension(indicators=[wasting])

    return dimension_1, dimension_2


def get_macro_dimension_data() -> pd.DataFrame:
    """Return macro dimension data"""

    # Get the macro dimensions
    economist = Indicator(
        get_economist_index(refresh=False),
        "Economist Food Security Index",
        countries_list=COUNTRIES,
        more_is_worse=False,
    )

    reserves = Indicator(
        get_fiscal_reserves(refresh=False),
        "Fiscal Reserves excluding gold (per capita)",
        countries_list=COUNTRIES,
        more_is_worse=False,
    )

    service = Indicator(
        get_service_spending_ratio(2022),
        "Debt Service to Government Spending Ratio",
        countries_list=COUNTRIES,
        more_is_worse=True,
    )

    econ_dimension = Dimension(indicators=[economist, reserves, service])

    return econ_dimension.get_data()


def run_index(dimensions: tuple) -> pd.DataFrame:
    """Run the index analysis"""

    index_detailed = Index(dimensions=copy.deepcopy(list(dimensions)))
    index_summary = Index(dimensions=copy.deepcopy(list(dimensions)))

    # Run the index analysis
    index_summary.index_data(summarised=False)
    index_summary_data = (
        index_summary.rescale_index()
        .reset_index()
        .pipe(add_short_names)
        .rename({"score": "Score", "country_name": "Country"}, axis=1)
    )

    # Get the raw data
    raw_data = index_detailed.get_data().reset_index()

    # Context data
    context_data = get_macro_dimension_data()

    # combined dataframe
    df = (
        index_summary_data.merge(raw_data, on=["iso_code"])
        .merge(context_data, on=["iso_code"])
        .round(1)
    )

    return df


if __name__ == "__main__":

    index_dimensions = get_dimensions()
    dimensions = get_dimensions()
    index_data = run_index(index_dimensions)
    index_data.to_clipboard(index=False)

