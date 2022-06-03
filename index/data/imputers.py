from typing import Any

import pandas as pd
from country_converter import country_converter
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer

from index import common


def __skl_imputer(df: pd.DataFrame, *, imputer_obj: Any, **kwargs) -> pd.DataFrame:
    """Use a simple imputer applied to the whole column.
    Strategy can be "median", "mean", or "most_frequent"
    """

    if "iso_code" in df.columns:
        df = df.set_index("iso_code")

    # Columns which only contained missing values at fit are discarded upon transform
    # For the integrity of the resulting dataset, keep track of such columns
    all_missing = []
    for column in df.columns:
        if df[column].notna().sum() == 0:
            all_missing.append(column)

    # Create a DataFrame with columns where everything is missing
    df_missing = df[all_missing].copy()

    # Filter the original DataFrame where no columns are all missing
    df = df[[c for c in df.columns if c not in all_missing]]

    # Create the imputer object
    imputer = imputer_obj(**kwargs)

    # Impute and create a new dataframe which preserves the columns and index
    df = pd.DataFrame(
        imputer.fit_transform(df.values), columns=df.columns, index=df.index
    )

    return pd.concat([df_missing, df], axis=1)


def __one_imputer(
    df: pd.DataFrame, *, strategy: str = "median", **kwargs
) -> pd.DataFrame:
    new_df = pd.DataFrame()

    grouped = df.groupby("grouper")

    for group in grouped.groups:
        try:
            _ = (
                grouped.get_group(group)
                .drop("grouper", axis=1)
                .pipe(skl_simple_imputer, strategy=strategy)
            )
        except ValueError as ve:
            _ = None
            print(f"Problem with imputing {group}: {ve}")

        new_df = pd.concat([new_df, _], ignore_index=False)

    return new_df.reindex(df.index)


def skl_simple_imputer(
    df: pd.DataFrame, *, strategy: str = "median", **kwargs
) -> pd.DataFrame:
    """Use a simple imputer applied to the whole column.
    Strategy can be "median", "mean", or "most_frequent"
    """
    return __skl_imputer(df, imputer_obj=SimpleImputer, strategy=strategy)


def skl_iterative_imputer(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Use a simple imputer applied to the whole column.
    Strategy can be "median", "mean", or "most_frequent"
    """
    return __skl_imputer(df, imputer_obj=IterativeImputer, **kwargs)


def skl_knn_imputer(df: pd.DataFrame, n_neighbors: int = 5, **kwargs) -> pd.DataFrame:
    return __skl_imputer(df, imputer_obj=KNNImputer, n_neighbors=n_neighbors, **kwargs)


def one_region_imputer(
    df: pd.DataFrame, *, strategy: str = "median", **kwargs
) -> pd.DataFrame:
    """Use UN Regions to impute missing values.
    Strategy can be "median", "mean", or "most_frequent"""

    if "iso_code" in df.columns:
        df = df.set_index("iso_code")

    df = df.assign(grouper=lambda d: country_converter.convert(d.index, to="UNregion"))

    return __one_imputer(df, strategy=strategy)


def one_continent_imputer(
    df: pd.DataFrame, *, strategy: str = "median", **kwargs
) -> pd.DataFrame:
    """Use Continents to impute missing values.
    Strategy can be "median", "mean", or "most_frequent"""

    if "iso_code" in df.columns:
        df = df.set_index("iso_code")

    df = df.assign(grouper=lambda d: country_converter.convert(d.index, to="continent"))

    return __one_imputer(df, strategy=strategy)


def one_income_imputer(
    df: pd.DataFrame, *, strategy: str = "median", **kwargs
) -> pd.DataFrame:
    """Use WB Income levels to impute missing values.
    Strategy can be "median", "mean", or "most_frequent"""

    if "iso_code" in df.columns:
        df = df.set_index("iso_code")

    df = df.assign(grouper=lambda d: d.index.map(common.income_levels()))

    return __one_imputer(df, strategy=strategy)
