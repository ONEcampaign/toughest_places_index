"""Summary Statistics functions"""

import pandas as pd
import country_converter as coco
from index import common


def __missing_prop(df: pd.DataFrame, target_col: str, grouping_col: str = None) -> dict:
    """returns percent of a column that is null, by specified groups"""

    if grouping_col is None:
        return {'overall':(df[target_col].isna().sum()/len(df))*100}
    else:
        return {group: (df.loc[df[grouping_col] == group, target_col].isna().sum()/len(df.loc[df[grouping_col] == group]))*100
                for group in df[grouping_col].unique()}


def summarize_missing(df: pd.DataFrame, target_col:str, iso_col: str = 'iso_code', by: str = 'overall') -> dict:
    """
    :param target_col:
    :param df:
    :param iso_col:
    :param by: ['overall', 'region', 'continent', 'income_level', 'country']
    :return:
    """

    match by:
        case 'overall':
            return __missing_prop(df, target_col)

        case 'region':
            return __missing_prop(df.assign(group = lambda d: coco.convert(d[iso_col], to = 'UNregion')),
                                  target_col,
                                  'group'
                                  )
        case 'continent':
            return __missing_prop(df.assign(group = lambda d: coco.convert(d[iso_col], to = 'continent')),
                           target_col,
                           'group'
                           )

        case 'income_level':
            return __missing_prop(common.add_income_levels(df), target_col, 'income_level')

        case 'country':
            return __missing_prop(df, target_col, iso_col)

        case _:
            raise ValueError(f'Invalid parameter: {by}')







