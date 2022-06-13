"""Summary Statistics functions"""

import pandas as pd
import numpy as np
import country_converter as coco
from index import common
from typing import Union


def add_grouping_col(df: pd.DataFrame, group_by:str, iso_col: str = 'iso_code'):
    """ """

    match group_by:
        case 'iso_code':
            return df.assign(grouping = lambda d: d[iso_col]).dropna(subset = 'grouping').reset_index(drop=True)

        case 'country':
            return df.assign(grouping = lambda d: coco.convert(d[iso_col], to = 'name_short', not_found = np.nan)).dropna(subset = 'grouping').reset_index(drop=True)

        case 'continent':
            return df.assign(grouping = lambda d: coco.convert(d[iso_col], to = 'continent', not_found = np.nan)).dropna(subset = 'grouping').reset_index(drop=True)

        case 'UNregion':
            return df.assign(grouping = lambda d: coco.convert(d[iso_col], to = 'UNregion', not_found = np.nan)).dropna(subset = 'grouping').reset_index(drop=True)

        case 'income_level':
            return common.add_income_levels(df, target_col = 'grouping').dropna(subset = 'grouping').reset_index(drop=True)
        case _:
            raise ValueError(f"Invalid parameter 'by': {group_by}")




def missing_by_column(df: pd.DataFrame, target_col: str, group_by: str = None, iso_col:str = 'iso_code') -> dict:
    """ """

    if group_by is None:
        return {'overall': round(df[target_col].isna().sum()/len(df), 2)}
    else:
        df = add_grouping_col(df, group_by, iso_col)
        return {group: round(df.loc[df['grouping'] == group, target_col].isna().sum()/len(df[df['grouping'] == group]), 2)
                for group in df['grouping'].unique()}



def missing_by_row(df: pd.DataFrame, index_cols: Union[str, list] = 'iso_code') -> dict:
    """ """

    if index_cols is not None:
        df = df.set_index(index_cols)

    return (df.isna().sum(axis = 1)/len(df.columns)).to_dict()


def missing_country_list(df: pd.DataFrame, target_col: str, group_by: str = None, iso_col:str = 'iso_code') -> dict:
    """ """

    if group_by is None:
        return {'overall': df.loc[df[target_col].isna(), iso_col].unique()}
    else:
        df = add_grouping_col(df, group_by, iso_col)
        return {group: list(df.loc[(df[target_col].isna())&(df['grouping'] == group), iso_col].unique()) for group in df['grouping'].unique()}











# ====================================================
# Missing data
# ====================================================


def __missing_prop(df: pd.DataFrame, target_col: str, grouping_col: str = None) -> dict:
    """returns percent of a column that is null, by specified groups"""

    if grouping_col is None:
        return {"overall": (df[target_col].isna().sum() / len(df)) * 100}
    else:
        return {
            group: (
                df.loc[df[grouping_col] == group, target_col].isna().sum()
                / len(df.loc[df[grouping_col] == group])
            )
            * 100
            for group in df[grouping_col].unique()
        }


def __missing_countries_subset(
    df: pd.DataFrame, target_col: str, subset_col: str, iso_col: str = "iso_code"
) -> dict:
    """returns dictionary with grouping categories as keys and lists of countries where values are null as values"""

    return {
        subset: df.loc[
            (df[subset_col] == subset) & (df[target_col].isna()), iso_col
        ].unique()
        for subset in df[subset_col].unique()
    }


def missing_countries(
    df: pd.DataFrame, target_col: str, iso_col: str = "iso_code", by=None
) -> dict:
    """
    Finds countries with null values, broken down by a specific grouping

    :param df: pd.Dataframe
        pandas dataframe with an iso3 column
    :param target_col: str
        name of column that stores data values
    :param iso_col: str
        name of column that stores iso3 codes
    :param by: Optional[str],
        country grouping [continent, region, income_level]

    :return: dict
        dictionary with grouping categories as keys and lists of countries as values
    """
    if by is None:
        return {"overall": df.loc[df[target_col].isna(), iso_col].unique()}

    elif by == "continent":
        df["subset"] = coco.convert(df[iso_col], to="continent")
        return __missing_countries_subset(df, target_col, subset_col="subset")

    elif by == "region":
        df["subset"] = coco.convert(df[iso_col], to="UNregion")
        return __missing_countries_subset(df, target_col, subset_col="subset")

    elif by == "income_level":
        return __missing_countries_subset(
            common.add_income_levels(df), target_col, subset_col="income_level"
        )

    else:
        raise ValueError(f"{by}: invalid parameter")


def summarize_missing(
    df: pd.DataFrame, target_col: str, by: str = None, iso_col: str = "iso_code"
) -> dict:
    """
    calculates proportions of missing values by specific country grouping

    :param df: pd.Dataframe
        pandas dataframe with an iso3 column
    :param target_col: str
        name of column that stores data values
    :param iso_col: str
        name of column that stores iso3 codes
    :param by:  Optional[str],
        country grouping [continent, region, income_level, country]

    :return: dict
         dictionary with grouping categories as keys and null percentages as values
    """

    match by:
        case None:
            return __missing_prop(df, target_col)

        case "region":
            return __missing_prop(
                df.assign(group=lambda d: coco.convert(d[iso_col], to="UNregion")),
                target_col,
                "group",
            )
        case "continent":
            return __missing_prop(
                df.assign(group=lambda d: coco.convert(d[iso_col], to="continent")),
                target_col,
                "group",
            )

        case "income_level":
            return __missing_prop(
                common.add_income_levels(df), target_col, "income_level"
            )

        case "country":
            return __missing_prop(df, target_col, iso_col)

        case _:
            raise ValueError(f"Invalid parameter: {by}")


def summarize_missing_full(df: pd.DataFrame, index_col: str = None) -> dict:
    """
    calculate proportion of null values across columns in a dataframe

    :param df: pd.Dataframe
        pandas dataframe with an iso3 column
    :param index_col: str
        name of index column, None if no index column exists

    :return: dict
        dictionary with index column values as keys and percent of values missing as values
    """

    if index_col is not None:
        df = df.set_index(index_col)

    return df.assign(missing=lambda d: (d.isna().sum(axis=1) / len(d.columns)) * 100)[
        "missing"
    ].to_dict()


# ====================================================
# outliers
# ====================================================


def __3sigma_test(series: pd.Series) -> list:
    """Uses the imperical rule to determine if a value is an outlier, returns a boolean list"""

    return abs(series - series.mean()) > 3 * series.std()


def __iqr_test(series: pd.Series) -> list:
    """
    Uses Inter-quartile range test to determine if a value is an outlier
    factor - set the magnitude to test
    """

    multiplier = 1.5  # iqr multiplier
    q25, q75 = series.quantile(0.25), series.quantile(0.75)
    iqr = q75 - q25

    return (series < (q25 - (iqr * multiplier))) | (series > (q75 + (iqr * multiplier)))


# outlier methods
AVAILABLE_METHODS = {"empirical": __3sigma_test, "inter_quartile_range": __iqr_test}


def _get_outliers(
    df: pd.DataFrame,
    target_col: str,
    method: str = "empirical",
    cluster_col: str = None,
) -> pd.DataFrame:
    """
    Applies an outlier calculation method to a dataframe
    Optionally apply method by looping through country clusters
    """

    if method not in AVAILABLE_METHODS.keys():
        raise ValueError(f"{method}: invalid method")
    method_func = AVAILABLE_METHODS[method]
    if cluster_col is None:
        df_outlier = df[method_func(df[target_col])]

    else:
        df_outlier = pd.DataFrame()
        for cluster in df[cluster_col].unique():
            cluster_df = df[df[cluster_col] == cluster]
            df_outlier = pd.concat(
                [df_outlier, cluster_df[method_func(cluster_df[target_col])]],
                ignore_index=True,
            )

    return df_outlier


def outliers(
    df: pd.DataFrame,
    target_col: str,
    method: str = "empirical",
    cluster: str = None,
    iso_col: str = "iso_code",
) -> pd.DataFrame:
    """
    Finds outliers in a dataset

    :param df: pd.Dataframe
        pandas dataframe with an iso3 column
    :param target_col: str
        name of column that stores data values
    :param method: str
        outlier calculation method, default = empirical, [empirical, inter_quartile_range]
    :param cluster: str
        country grouping on which to apply the method, default = None. applies method to the whole dataframe
        [region, continent, income_group, country]
    :param iso_col: str
        name of column that stores iso3 codes

    :return: pd.DataFrame
        dataframe with outliers
    """

    match cluster:
        case None:
            return _get_outliers(df, target_col, method)

        case "region":
            return _get_outliers(
                df.assign(cluster=lambda d: coco.convert(d[iso_col], to="UNregion")),
                target_col,
                method,
                cluster_col="cluster",
            )

        case "continent":
            return _get_outliers(
                df.assign(cluster=lambda d: coco.convert(d[iso_col], to="continent")),
                target_col,
                method,
                cluster_col="cluster",
            )

        case "income_group":
            return _get_outliers(
                common.add_income_levels(df),
                target_col,
                method,
                cluster_col="income_level",
            )

        case "country":
            return _get_outliers(df, target_col, method, cluster_col=iso_col)

        case _:
            raise ValueError(f"Invalid parameter: {cluster}")


# ====================================================
# Zero values
# ====================================================


def __zero_subset(df: pd.DataFrame, target_col: str, subset_col: str) -> dict:
    """calculates proportion of 0 values for subsets of a dataframe"""

    return {
        subset: (
            len(df[(df[subset_col] == subset) & (df[target_col] == 0)])
            / len(df[df[subset_col] == subset])
        )
        * 100
        for subset in df[subset_col].unique()
    }


def check_zeros(
    df: pd.DataFrame, target_col: str, by: str = None, iso_col: str = "iso_code"
) -> dict:
    """
    Calculated proportion of a dataframe (by specific country grouping) that has 0 values

    :param df: pd.Dataframe
        pandas dataframe with an iso3 column
    :param target_col: str
        name of column that stores data values
    :param by: Optional[str],
        country grouping [continent, region, income_level, country]
    :param iso_col: str
        name of column that stores iso3 codes

    :return: dict
        dictionary with proportions of 0 values as dictionary values
    """

    if by is None:
        return {"overall": (len(df[df[target_col] == 0]) / len(df)) * 100}

    elif by == "region":
        return __zero_subset(
            df.assign(group=lambda d: coco.convert(d[iso_col], to="UNregion")),
            target_col,
            "group",
        )
    elif by == "continent":
        return __zero_subset(
            df.assign(group=lambda d: coco.convert(d[iso_col], to="continent")),
            target_col,
            "group",
        )
    elif by == "income_level":
        return __zero_subset(
            common.add_income_levels(df).dropna(subset="income_level"),
            target_col,
            "income_level",
        )

    elif by == "country":
        return __zero_subset(df, target_col, iso_col)
    else:
        raise ValueError(f"{by}: Invalid parameter")


# ====================================================
# Correlation
# ====================================================


def collinearity(
    df: pd.DataFrame,
    bounds: tuple = (0.7, -0.7),
    index_col: str = None,
    column: str = None,
) -> dict:
    """
    Returns a dictionary with variables as keys and correlated variables in a list as values, using pearson correlation

    :param df: pd.Dataframe
        pandas dataframe with an iso3 column
    :param bounds: tuple
        upper bound and lower bound for pearson correlation values,  default =(0.7, -0.7)
    :param index_col: str
        name of index column, None if no index column exists
    :param column: Optional[str]
        column on which to run test collinearity with other columns

    :return: dict
        dictionary with variable names as keys and list of correlated variables as values
    """

    if index_col is not None:
        df = df.set_index(index_col)

    corr_df = df.corr()

    if column is None:
        return {
            var: list(
                corr_df[
                    ((corr_df[var] >= bounds[0]) | (corr_df[var] <= bounds[1]))
                    & (corr_df[var].index != var)
                ].index
            )
            for var in corr_df.columns
        }
    else:
        return {
            column: list(
                corr_df[
                    ((corr_df[column] >= bounds[0]) | (corr_df[column] <= bounds[1]))
                    & (corr_df[column].index != column)
                ].index
            )
        }
