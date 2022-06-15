import json
import os
from dataclasses import dataclass

import pandas as pd
import requests

from index.common import add_short_names, read_and_append
from index.config import PATHS


def _get_country_codes() -> None:
    """Script to fetch the country codes used by WFP. Saved as a dataframe."""

    # Get the json file from WFP website
    url: str = "https://api.hungermapdata.org/covid/data"
    file = requests.get(url).content

    # WFP codes
    wfp = json.loads(file)["COUNTRIES"]

    # Create a dictionary with the iso_code: country code
    codes = {d["iso3"]: d["adm0_code"] for d in wfp}

    # Create a dataframe with the country codes and save
    df = pd.DataFrame(list(codes.items()), columns=["iso_code", "wfp_code"])
    df.to_csv(rf"{PATHS.raw_hunger}/wfp_country_codes.csv", index=False)

    print("WFP country codes successfully downloaded.")


def country_codes() -> dict:
    """Returns a dictionary with the country codes used by WFP."""

    if not os.path.exists(rf"{PATHS.raw_hunger}/wfp_country_codes.csv"):
        _get_country_codes()

    d = pd.read_csv(rf"{PATHS.raw_hunger}/wfp_country_codes.csv")
    return dict(zip(d["iso_code"], d["wfp_code"].astype(int)))


@dataclass
class FsDataWFP:
    iso_code: str
    admin_code: int = None
    raw_data_hunger: json = None
    raw_data_nutrition: json = None
    hunger_data: pd.DataFrame = None
    nutrition_data: pd.DataFrame = None

    def __post_init__(self):
        if self.admin_code is None:
            self.admin_code = country_codes()[self.iso_code]

    def __wfp_fetch(self, url) -> json:
        # Get the country json file from WFP website
        raw_data = requests.get(url)

        # If empty flag and return None
        if raw_data.status_code == 404:
            print(f"{self.iso_code} not found.")
            return None

        return raw_data.json()

    def __create_df(self, json_obj: json, key: str) -> pd.DataFrame:

        if json_obj is None:
            print(f"No json data passed for {self.iso_code}")
            return pd.DataFrame()

        # Try to create dataframe
        try:
            return pd.DataFrame(json_obj[key])

        except ValueError:
            return pd.DataFrame([json_obj[key]])

        except KeyError as error:
            print(f"{self.iso_code} returned a key error ({error})")
            return pd.DataFrame()

    def _download_hunger_data(self) -> None:
        """Downloads the data from WFP."""

        url = (
            "https://5763353767114258.eu-central-1.fc.aliyuncs.com/2016-08-15/"
            f"proxy/wfp-data-api.34/map-data/adm0/{self.admin_code}/countryData.json"
        )

        self.raw_data_hunger = self.__wfp_fetch(url)

    def _download_nutrition_data(self) -> None:

        url = (
            f"https://5763353767114258.eu-central-1.fc.aliyuncs.com/2016-08-15/"
            f"proxy/wfp-data-api.36/map-data/iso3/{self.iso_code}/countryIso3Data.json"
        )

        self.raw_data_nutrition = self.__wfp_fetch(url)

    def refresh_insufficient_food(self) -> None:

        if self.raw_data_hunger is None:
            self._download_hunger_data()

        # Create dataframe
        data = self.__create_df(self.raw_data_hunger, key="fcsGraph")

        # Code may be valid but data may be empty. If so, return None
        if len(data) == 0:
            print(f"No hunger data for {self.iso_code}")
            return None

        # If data is valid, clean it and save to munged
        data = (
            data.rename(
                columns=(
                    {
                        "x": "date",
                        "fcs": "value",
                        "fcsHigh": "value_high",
                        "fcsLow": "value_low",
                    }
                )
            )
            .assign(
                date=lambda d: pd.to_datetime(d.date, format="%Y-%m-%d"),
                iso_code=self.iso_code,
            )
            .pipe(add_short_names)
            .pipe(
                read_and_append,
                file_path=rf"{PATHS.raw_hunger}/{self.iso_code}_insufficient_food.csv",
                date_col="date",
                idx=["iso_code"],
            )
        )

        data.to_csv(
            rf"{PATHS.raw_hunger}/{self.iso_code}_insufficient_food.csv", index=False
        )
        self.hunger_data = data

    def refresh_nutrition(self) -> None:

        if self.raw_data_nutrition is None:
            self._download_nutrition_data()

        # Create dataframe
        data = self.__create_df(self.raw_data_nutrition, key="nutrition")

        # Code may be valid but data may be empty. If so, return None
        if len(data) == 0:
            print(f"No nutrition data for {self.iso_code}")
            return None

        # If data is valid, clean it and save to munged
        data = (
            data.assign(iso_code=self.iso_code)
            .drop(columns=["source"])
            .melt(id_vars=["iso_code"], var_name="indicator", value_name="value")
            .pipe(add_short_names)
        )

        data.to_csv(rf"{PATHS.raw_hunger}/{self.iso_code}_nutrition.csv", index=False)

        self.nutrition_data = data


def create_wfp_database() -> None:
    """Update and return data for Africa"""

    wfp_codes = country_codes()
    insufficient = pd.DataFrame()
    nutrition = pd.DataFrame()

    for iso_code in wfp_codes:
        data = FsDataWFP(iso_code)
        data.refresh_insufficient_food()
        data.refresh_nutrition()

        insufficient = pd.concat([insufficient, data.hunger_data], ignore_index=True)
        nutrition = pd.concat([nutrition, data.nutrition_data], ignore_index=True)

    insufficient.to_csv(rf"{PATHS.data}/wfp_insufficient_food.csv", index=False)
    nutrition.to_csv(rf"{PATHS.data}/wfp_nutrition.csv", index=False)


def read_hunger_data() -> pd.DataFrame:
    return pd.read_csv(rf"{PATHS.data}/wfp_insufficient_food.csv", parse_dates=["date"])


def read_nutrition_data() -> pd.DataFrame:
    return pd.read_csv(rf"{PATHS.data}/wfp_nutrition.csv")


if __name__ == "__main__":
    create_wfp_database()
