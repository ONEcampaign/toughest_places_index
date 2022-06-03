from index.data_io import ids_data, wb_data, wfp_inflation, wfp_food_sec
from index.data.indicator import Indicator
from index.data.dimension import Dimension
from index.data.index import Index

from index.common import get_latest

import pandas as pd

# 1.1 WFP hunger
hunger_1 = (
    wfp_food_sec.read_hunger_data()
    .filter(["iso_code", "date", "value"], axis=1)
    .pipe(get_latest, by="iso_code")
)

nutrition_1 = (
    wfp_food_sec.read_nutrition_data()
    .loc[lambda d: d.indicator == "wasting"]
    .reset_index(drop=True)
)

# debt
debt_ratio = ids_data.get_service_spending_ratio(2022).rename(
    columns={"ratio": "value"}
)

#reserves
reserves = wb_data.wb_reserves().pipe(get_latest, by='iso_code')

hunger_indicator = Indicator(hunger_1, "hunger")
nutrition_indicator = Indicator(nutrition_1, "nutrition")
debt_indicator = Indicator(debt_ratio, 'debt_ratio')
reserves_indicator = Indicator(reserves,'reserves')


# ----- Dimensions

dimension1 = Dimension(dimension_name='food')
dimension2 = Dimension(dimension_name='money')

dimension1.add_indicator(hunger_indicator)
dimension1.add_indicator(nutrition_indicator)

dimension2.add_indicator(debt_indicator)
dimension2.add_indicator(reserves_indicator)

d1 = dimension1.get_data(orient='long')
d2 = dimension2.get_data(orient='wide')

idx = Index()
idx.add_dimension(dimension1)
idx.add_dimension(dimension2)

test = idx.get_data(orient='wide')