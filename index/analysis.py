"""
This file contains the final analysis
"""
import copy

from index.config import PATHS
from index.data.index import Index
from index.indicators import (
    get_insufficient_food,
    get_inflation,
    get_wasting,
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


def run_index(dimensions: tuple):
    """Run the index analysis"""

    index = Index(dimensions=list(dimensions))

    with pd.ExcelWriter(PATHS.data + r"/index_results_detailed.xlsx") as writer:
            #result.to_excel(writer, sheet_name=result_n)
            pass

if __name__ == "__main__":
    dimensions_raw = get_dimensions()

    run_index(dimensions=dimensions_raw)
