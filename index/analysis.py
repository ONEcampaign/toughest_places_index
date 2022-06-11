""" THIS FILE EXISTS FOR TESTING PURPOSES ONLY


------- IT DOES NOT CONTAIN ACTUAL ANALYSIS YET -------

"""
import copy

from index.config import PATHS
from index.data.index import Index
from index.indicators import (
    get_insufficient_food,
    get_inflation,
    get_economist_index,
    get_wasting,
    get_fiscal_reserves,
    get_service_spending_ratio,
)

from index.data.indicator import Indicator
from index.data.dimension import Dimension

from index.common import STUDY_COUNTRIES

from index.common import add_short_names
import pandas as pd

countries = STUDY_COUNTRIES["lics_lmics"]
countries = [c for c in countries if c not in ["KIR", "FSM", "PRK", "WSM"]]

# Crete individual indicators. Filter to a set of countries by using countries_list


def get_dimensions() -> tuple:
    """Return a list of indicators"""

    insufficient_food = Indicator(
        get_insufficient_food(),
        "Insufficient Food Consumption",
        countries_list=countries,
    )
    # insufficient_food.raw_data = insufficient_food.raw_data.fillna(0)

    inflation = Indicator(
        get_inflation(), "Headline Inflation", countries_list=countries
    )

    economist_index = Indicator(
        get_economist_index(),
        "Economist Index",
        countries_list=countries,
        more_is_worse=False,
    )

    wasting = Indicator(get_wasting(), "Wasting", countries_list=countries)

    fiscal_reserves = Indicator(
        get_fiscal_reserves(),
        "Fiscal Reserves minus gold",
        countries_list=countries,
        more_is_worse=False,
    )

    service_spending_ratio = Indicator(
        get_service_spending_ratio(2022),
        "Service Spending Ratio",
        countries_list=countries,
    )

    dimension_1 = Dimension(indicators=[insufficient_food, inflation])
    dimension_2 = Dimension(indicators=[economist_index, wasting])
    dimension_3 = Dimension(indicators=[fiscal_reserves, service_spending_ratio])

    return dimension_1, dimension_2, dimension_3


def index_version(
    dimensions_tup: tuple, rescale_parameters: dict, impute_parameters: dict
) -> pd.DataFrame:

    dimension_1, dimension_2, dimension_3 = dimensions_tup
    index = Index(dimensions=[dimension_1, dimension_2, dimension_3])
    index.index_data(
        rescale_parameters=rescale_parameters, impute_parameters=impute_parameters
    )

    res = index.get_data().reset_index().pipe(add_short_names)
    res.index = res.index + 1

    return res


def main(dimensions: tuple):

    versions = {
        "robust and knn15": (
            {"scaler_name": "robust"},
            {"method": "knn", "n_neighbors": 15},
        ),
        "robust and knn10": (
            {"scaler_name": "robust"},
            {"method": "knn", "n_neighbors": 10},
        ),
        "robust and knn5": (
            {"scaler_name": "robust"},
            {"method": "knn", "n_neighbors": 5},
        ),
        "standard and knn15": (
            {"scaler_name": "standard"},
            {"method": "knn", "n_neighbors": 15},
        ),
        "standard and knn5": (
            {"scaler_name": "standard"},
            {"method": "knn", "n_neighbors": 5},
        ),
        "minmax and knn15": (
            {"scaler_name": "minmax"},
            {"method": "knn", "n_neighbors": 15},
        ),
        "minmax and knn5": (
            {"scaler_name": "minmax"},
            {"method": "knn", "n_neighbors": 5},
        ),
        "robust and iterative": (
            {"scaler_name": "robust"},
            {"method": "iterative"},
        ),
        "standard and iterative": (
            {"scaler_name": "standard"},
            {"method": "iterative"},
        ),
        "minmax and iterative": (
            {"scaler_name": "minmax"},
            {"method": "iterative"},
        ),
        "robust and region": (
            {"scaler_name": "robust"},
            {"method": "region"},
        ),
        "robust and income": (
            {"scaler_name": "robust"},
            {"method": "income"},
        ),
        "robust and continent": (
            {"scaler_name": "robust"},
            {"method": "continent"},
        ),
    }

    results = {
        version_n: index_version(copy.deepcopy(dimensions), *version)
        for version_n, version in versions.items()
    }

    with pd.ExcelWriter(PATHS.data + r"/results.xlsx") as writer:
        for result_n, result in results.items():
            result.to_excel(writer, sheet_name=result_n)


if __name__ == "__main__":
    dimensions_raw = get_dimensions()
    # main(dimensions=dimensions_raw)
    d1, d2, d3 = dimensions_raw
    index = Index(dimensions=[d1, d2, d3])
    md = index.check_missing_data()
