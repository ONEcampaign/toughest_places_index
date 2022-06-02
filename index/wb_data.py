import pandas as pd

from index.common import add_short_names, wb_data


def wb_stunting(refresh: bool = False) -> pd.DataFrame:
    """Get the World Bank data for stunting"""
    id_: str = "SH.STA.STNT.ZS"

    return wb_data(
        series=id_, series_name="Stunting (%)", years=15, download=refresh
    ).pipe(add_short_names)


def wb_wasting(refresh: bool = False) -> pd.DataFrame:
    """Get the World Bank data for wasting"""
    id_: str = "SH.STA.WAST.ZS"
    return wb_data(
        series=id_, series_name="Wasting (%)", years=15, download=refresh
    ).pipe(add_short_names)
