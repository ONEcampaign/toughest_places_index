from typing import Any

import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    QuantileTransformer,
    PowerTransformer,
)


def __skl_scaler(df: pd.DataFrame, *, scaler_obj: Any, **kwargs) -> pd.DataFrame:
    """wrapper around sklearn scalers, passed as Callables"""

    if "iso_code" in df.columns:
        df = df.set_index("iso_code")

    # Create the scaler object
    scaler = scaler_obj(**kwargs)

    # Impute and create a new dataframe which preserves the columns and index
    df = pd.DataFrame(
        scaler.fit_transform(df.values), columns=df.columns, index=df.index
    )

    return df.reset_index(drop=False)


def skl_standard_scaler(
    df: pd.DataFrame, *, with_mean: bool = True, with_std: bool = True
) -> pd.DataFrame:
    """Standardize features by removing the mean and scaling to unit variance"""

    return __skl_scaler(df, scaler_obj=StandardScaler)


def skl_minmax_scaler(
    df: pd.DataFrame, *, feature_range: tuple = (0, 1)
) -> pd.DataFrame:
    """Scale features to a given range."""

    return __skl_scaler(df, scaler_obj=MinMaxScaler, feature_range=feature_range)


def skl_maxabs_scaler(df: pd.DataFrame, **Kwargs) -> pd.DataFrame:
    """Scale each feature by its maximum absolute value."""

    return __skl_scaler(df, scaler_obj=MaxAbsScaler)


def skl_quantile_transformer(
    df: pd.DataFrame, *, n_quantiles: int = 200, output_distribution: str = "uniform"
) -> pd.DataFrame:
    return __skl_scaler(
        df,
        scaler_obj=QuantileTransformer,
        n_quantiles=n_quantiles,
        output_distribution=output_distribution,
    )


def skl_power_transformer(
    df: pd.DataFrame, *, method: str = "yeo-johnson"
) -> pd.DataFrame:
    return __skl_scaler(df, scaler_obj=PowerTransformer)


def slk_robust_scaler(
    df: pd.DataFrame,
    *,
    with_centering: bool = True,
    with_scaling: bool = True,
    unit_variance: bool = False
) -> pd.DataFrame:
    """Scale features using quantiles instead of percentiles."""

    return __skl_scaler(
        df,
        scaler_obj=RobustScaler,
        with_centering=with_centering,
        with_scaling=with_scaling,
        unit_variance=unit_variance,
    )


SCALERS: dict = {
    "standard": skl_standard_scaler,
    "minmax": skl_minmax_scaler,
    "maxabs": skl_maxabs_scaler,
    "robust": slk_robust_scaler,
    "quantile": skl_quantile_transformer,
    "power": skl_power_transformer,
}
