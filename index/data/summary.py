"""Summary Statistics functions"""

import pandas as pd
import numpy as np
import country_converter as coco
from index import common
from typing import Union


def add_grouping_col(df: pd.DataFrame, group_by: str, iso_col: str = "iso_code"):
    """
    Adds a column for country groups to a dataframe

    :param df: pd.DataFrame
        pandas dataframe
    :param group_by: str
        category to group values by
    :param iso_col:
        column containting iso3 codes
    :return: pd.DataFrame
        pandas dataframe with new column 'grouping' with grouping category values
    """

    match group_by:
        case "iso_code":
            return (
                df.assign(grouping=lambda d: d[iso_col])
                .dropna(subset="grouping")
                .reset_index(drop=True)
            )

        case "country":
            return (
                df.assign(
                    grouping=lambda d: coco.convert(
                        d[iso_col], to="name_short", not_found=np.nan
                    )
                )
                .dropna(subset="grouping")
                .reset_index(drop=True)
            )

        case "continent":
            return (
                df.assign(
                    grouping=lambda d: coco.convert(
                        d[iso_col], to="continent", not_found=np.nan
                    )
                )
                .dropna(subset="grouping")
                .reset_index(drop=True)
            )

        case "UNregion":
            return (
                df.assign(
                    grouping=lambda d: coco.convert(
                        d[iso_col], to="UNregion", not_found=np.nan
                    )
                )
                .dropna(subset="grouping")
                .reset_index(drop=True)
            )

def __missing_countries_subset(
    df: pd.DataFrame, target_col: str, subset_col: str, iso_col: str = "iso_code"
) -> dict:
    """returns dictionary with grouping categories as keys and lists of COUNTRIES where values are null as values"""


        case "income_level":
            return (
                common.add_income_levels(df, target_col="grouping")
                .dropna(subset="grouping")
                .reset_index(drop=True)
            )
        case _:
            raise ValueError(f"Invalid parameter 'by': {group_by}")


def missing_by_column(
    df: pd.DataFrame, target_col: str, group_by: str = None, iso_col: str = "iso_code"
) -> dict:
    """
    Calculates proportions of missing in a column

    :param df: pd.Dataframe
        pandas dataframe with an iso3 column
    :param target_col: str
        name of column that stores data values
    :param group_by: str, optional
        country grouping
    :param iso_col: str, optional
        name of column that stores iso3 codes

    :return: dict
        dictionary with grouping categories as keys and null proportions as values

    """

    if group_by is None:
        return {"overall": round(df[target_col].isna().sum() / len(df), 2)}
    else:
        df = add_grouping_col(df, group_by, iso_col)
        return {
            group: round(
                df.loc[df["grouping"] == group, target_col].isna().sum()
                / len(df[df["grouping"] == group]),
                2,
            )
            for group in df["grouping"].unique()
        }


def missing_by_row(df: pd.DataFrame, index_cols: Union[str, list] = None) -> dict:
    """
    Calculate proportion of null values by row

    :param df: pd.DataFrame
        pandas dataframe
    :param index_cols: str or list, optional
        name of index column
        default = None

    :return: dict
        dictionary with index column values as keys
        and proportion of values missing as values
    """

    if index_cols is not None:
        df = df.set_index(index_cols)

    return (df.isna().sum(axis=1) / len(df.columns)).to_dict()


def missing_country_list(
    df: pd.DataFrame, target_col: str, group_by: str = None, iso_col: str = "iso_code"
) -> dict:
    """
    Finds countries with null values

    :param df: pd.Dataframe
        pandas dataframe
    :param target_col: str
        name of column that stores data values
    :param group_by: str, optional
        country grouping
    :param iso_col: str, optional
        name of column that stores iso3 codes

    :return: dict
        dictionary with grouping categories as keys
        and lists of countries with missing data as values
    """

    if group_by is None:
        return {"overall": df.loc[df[target_col].isna(), iso_col].unique()}
    else:
        df = add_grouping_col(df, group_by, iso_col)
        return {
            group: list(
                df.loc[
                    (df[target_col].isna()) & (df["grouping"] == group), iso_col
                ].unique()
            )
            for group in df["grouping"].unique()
        }


def __3sigma_test(series: pd.Series) -> list:
    """
    Uses the imperical rule to determine if a value is an outlier
    returns a boolean list
    """

    return abs(series - series.mean()) > 3 * series.std()


def __iqr_test(series: pd.Series) -> list:
    """
    Uses Inter-quartile range test to determine if a value is an outlier
    returns a boolean list
    """

    multiplier = 1.5  # iqr multiplier
    q25, q75 = series.quantile(0.25), series.quantile(0.75)
    iqr = q75 - q25

    return (series < (q25 - (iqr * multiplier))) | (series > (q75 + (iqr * multiplier)))


# outlier methods
AVAILABLE_METHODS = {"empirical": __3sigma_test, "inter_quartile_range": __iqr_test}


def get_outliers(
    df: pd.DataFrame,
    target_col: str,
    method: str = "empirical",
    outlier_grouping: str = None,
    iso_col: str = "iso_code",
) -> pd.DataFrame:
    """
    Finds outliers in a dataframe

    :param df: pd.Dataframe
        pandas dataframe with an iso3 column
    :param target_col: str
        name of column that stores data values
    :param method: str, optional
        outlier calculation method, default = empirical, [empirical, inter_quartile_range]
        default = empirical
    :param outlier_grouping: str, optional
        country grouping on which to apply the method, default = None.
    :param iso_col: str, optional
        name of column that stores iso3 codes

    :return: pd.DataFrame
        dataframe of outliers
    """

    if method not in AVAILABLE_METHODS.keys():
        raise ValueError(f"{method}: invalid method")
    else:
        method_func = AVAILABLE_METHODS[method]

    if outlier_grouping is None:
        df_outlier = df[method_func(df[target_col])]
    else:
        df_outlier = pd.DataFrame()
        df = add_grouping_col(df, outlier_grouping, iso_col)
        for group in df["grouping"].unique():
            group_df = df[df["grouping"] == group]

            df_outlier = pd.concat(
                [
                    df_outlier,
                    group_df[method_func(group_df[target_col])],
                ],
                ignore_index=True,
            )

    return df_outlier


def get_zeros(
    df: pd.DataFrame, target_col: str, group_by: str = None, iso_col: str = "iso_code"
) -> dict:
    """
    Calculates the proportion of a dataframe (by specific country grouping) that has 0 values

    :param df: pd.Dataframe
        pandas dataframe with an iso3 column
    :param target_col: str
        name of column that stores data values
    :param group_by: str, optional
        country grouping
    :param iso_col: str, optional
        name of column that stores iso3 codes

    :return: dict
        dictionary with proportions of 0 values as dictionary values
    """

    if group_by is None:
        return {"overall": round(len(df[df[target_col] == 0]) / len(df), 2)}

    else:
        df = add_grouping_col(df, group_by, iso_col)
        return {
            group: round(
                len(df[(df["grouping"] == group) & (df[target_col] == 0)])
                / len(df[df["grouping"] == group]),
                2,
            )
            for group in df["grouping"].unique()
        }


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
