"""Summary Statistics functions"""

import pandas as pd
import country_converter as coco
from index import common
from typing import Callable


def __missing_prop(df: pd.DataFrame, target_col: str, grouping_col: str = None) -> dict:
    """returns percent of a column that is null, by specified groups"""

    if grouping_col is None:
        return {'overall':(df[target_col].isna().sum()/len(df))*100}
    else:
        return {group: (df.loc[df[grouping_col] == group, target_col].isna().sum()/len(df.loc[df[grouping_col] == group]))*100
                for group in df[grouping_col].unique()}

def missing_countries(df: pd.DataFrame, target_col:str, iso_col: str = 'iso_code') -> list:
    """ """

    return df.loc[df[target_col].isna(), iso_col].unique()


def summarize_missing(df: pd.DataFrame, target_col:str,by: str = 'overall', iso_col: str = 'iso_code') -> dict:
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




def __3sigma_test(series: pd.Series) -> list:
    """Uses the imperical rule to determine if a value is an outlier, returns a boolean list"""

    return abs(series - series.mean()) > 3*series.std()


def __iqr_test(series: pd.Series, factor: float = 2) -> list:
    """
    Uses Inter-quartile range test to determine if a value is an outlier
    factor - set the magnitude to test
    """

    q25, q75 = series.quantile(0.25), series.quantile(0.75)
    iqr = q75 - q25

    return (series < (q25 - (iqr * factor)))|(series> (q75 + (iqr * factor)))

AVAILABLE_METHODS = {'empirical': __3sigma_test, 'inter_quartile_range': __iqr_test}

def _get_outliers(df: pd.DataFrame, target_col: str, method: str = 'empirical', cluster_col: str = None):
    """ """

    if method not in AVAILABLE_METHODS.keys():
        raise ValueError(f'{method}: invalid method')
    method_func = AVAILABLE_METHODS[method]
    if cluster_col is None:
        df_outlier =  df[method_func(df[target_col])]

    else:
        df_outlier = pd.DataFrame()
        for cluster in df[cluster_col].unique():
            cluster_df = df[df[cluster_col] == cluster]
            df_outlier = pd.concat([df_outlier, cluster_df[method_func(cluster_df[target_col])]], ignore_index=True)

    return df_outlier



def outliers(df: pd.DataFrame, target_col:str, method: str = 'empirical', cluster: str = None, iso_col: str = 'iso_code'):
    """

    :param df:
    :param target_col:
    :param method:
    :param cluster:
    :param iso_col:
    :return:
    """

    match cluster:
        case None:
            return _get_outliers(df, target_col, method)

        case 'region':
            return _get_outliers(df.assign(cluster = lambda d: coco.convert(d[iso_col], to='UNregion')),
                          target_col, method, cluster_col= 'cluster'
                          )

        case 'continent':
            return _get_outliers(df.assign(cluster = lambda d: coco.convert(d[iso_col], to='continent')),
                                 target_col, method, cluster_col= 'cluster'
                                 )

        case 'income_group':
            return _get_outliers(common.add_income_levels(df), target_col, method, cluster_col='income_level')

        case 'country':
            return _get_outliers(df, target_col, method, cluster_col=iso_col)

        case _:
            raise ValueError(f'Invalid parameter: {cluster}')






