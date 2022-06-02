import pandas as pd
from index import config, common


def wb_stunting(refresh: bool = False) -> pd.DataFrame:
    """Get the World Bank data for stunting"""
    id_: str = "SH.STA.STNT.ZS"

    return common.wb_data(
        series=id_, series_name="Stunting (%)", years=15, download=refresh
    ).pipe(common.add_short_names)


def wb_wasting(refresh: bool = False) -> pd.DataFrame:
    """Get the World Bank data for wasting"""
    id_: str = "SH.STA.WAST.ZS"
    return common.wb_data(
        series=id_, series_name="Wasting (%)", years=15, download=refresh
    ).pipe(common.add_short_names)


if __name__ == "__main__":
    s = wb_stunting()
    w = wb_wasting()
