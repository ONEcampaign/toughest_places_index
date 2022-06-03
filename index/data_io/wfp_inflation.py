import pandas as pd
from index.common import add_short_names, read_and_append
from index.config import PATHS
import country_converter as cc


def _read_inflation(country_iso: str) -> pd.DataFrame:

    url = f"https://api.vam.wfp.org/dataviz/api/GetCsv?idx=71,116&iso3={country_iso}"
    try:
        return (
            pd.read_csv(
                url,
                usecols=[0, 1, 2],
                skipfooter=2,
                engine="python",
                parse_dates=["Time"],
            )
            .rename(
                columns={
                    "Time": "date",
                    "Value (percent)": "value",
                    "Indicator": "indicator",
                }
            )
            .assign(iso_code=country_iso)
            .sort_values(by=["indicator", "date"])
            .reset_index(drop=True)
        )
    except ConnectionError:
        print(f"Data not available for {country_iso}")


def refresh_inflation_data(iso_codes: list = None) -> None:

    # If no list of iso_codes provided, use all (from coco)
    if iso_codes is None:
        iso_codes = cc.CountryConverter().data["ISO3"].unique()

    # Save each country's data to a csv. Don't overwrite existing but append instead
    for iso_code in iso_codes:
        _ = (
            _read_inflation(iso_code)
            .astype({"date": "datetime64[ns]"})
            .pipe(add_short_names, id_col="iso_code", target_col="country_name")
            .pipe(
                read_and_append,
                file_path=rf"{PATHS.raw_wfp_inflation}/{iso_code}.csv",
                date_col="date",
                idx=["iso_code", "indicator"],
            )
            .to_csv(rf"{PATHS.raw_wfp_inflation}/{iso_code}.csv", index=False)
        )


if __name__ == "__main__":
    refresh_inflation_data()
