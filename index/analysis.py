""" THIS FILE EXISTS FOR TESTING PURPOSES ONLY


------- IT DOES NOT CONTAIN ACTUAL ANALYSIS YET -------

"""
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

countries = STUDY_COUNTRIES["no_hics"]

# Crete individual indicators. Filter to a set of countries by using countries_list
insufficient_food = Indicator(
    get_insufficient_food(), "Insufficient Food Consumption", countries_list=countries
)

inflation = Indicator(get_inflation(), "Headline Inflation", countries_list=countries)

economist_index = Indicator(
    get_economist_index(), "Economist Index", countries_list=countries
)

wasting = Indicator(get_wasting(), "Wasting", countries_list=countries)

fiscal_reserves = Indicator(
    get_fiscal_reserves(), "Fiscal Reserves minus gold", countries_list=countries
)

service_spending_ratio = Indicator(
    get_service_spending_ratio(2022), "Service Spending Ratio", countries_list=countries
)

# Create the dimensions
dimension_1 = Dimension(indicators=[insufficient_food, inflation])
dimension_2 = Dimension(indicators=[economist_index, wasting])
dimension_3 = Dimension(indicators=[fiscal_reserves, service_spending_ratio])

# to get a dataframe of all the indicators in a dimension
df_dimension_1 = dimension_1.get_data(orient="wide")
df_dimension_1_long = dimension_1.get_data(orient="long")

# to get a long dataframe with all the dimension indicators keeping the date
df_dimension_1_long_date = dimension_1.get_data(orient="long", with_date=True)

# Create the index object
index = Index(dimensions=[dimension_1, dimension_2, dimension_3])

# get the index data
df_index = index.get_data(orient="long", with_date=True)
df_index_wide = index.get_data(orient="wide")

# get index data long with date
df_index_long_date = index.get_data(orient="long", with_date=True)
