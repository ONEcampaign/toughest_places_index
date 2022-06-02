from dataclasses import dataclass

import pandas as pd
import requests
from index.common import add_iso_codes
from index.config import PATHS

import os

URL: str = (
    "https://impact.economist.com/sustainability/project/"
    "food-security-index/Home/DownloadIndex"
)

COLS: dict = {
    "OVERALL FOOD SECURITY ENVIRONMENT": "ranking_overall",
    "Unnamed: 13": "country_overall",
    "Unnamed: 14": "score_overall",
    "1) AFFORDABILITY": "ranking_affordability",
    "Unnamed: 24": "country_affordability",
    "Unnamed: 25": "score_affordability",
    "2) AVAILABILITY": "ranking_availability",
    "Unnamed: 35": "country_availability",
    "Unnamed: 36": "score_availability",
    "3) QUALITY AND SAFETY": "ranking_quality",
    "Unnamed: 46": "country_quality",
    "Unnamed: 47": "score_quality",
    "4) NATURAL RESOURCES & RESILIENCE": "ranking_natural",
    "Unnamed: 57": "country_natural",
    "Unnamed: 58": "score_natural",
}


def _rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda d: d.split("_")[0])


def _download_data():
    """Download the data from the Economist website"""

    file = requests.get(URL)

    df = (
        pd.read_excel(
            file.content,
            sheet_name="2021 Ranking",
            skiprows=20,
            skipfooter=3,
            usecols=COLS,
        )
        .loc[1:]
        .rename(columns=COLS)
    )

    ov = (
        df[["ranking_overall", "country_overall", "score_overall"]]
        .pipe(_rename_cols)
        .assign(indicator="overall")
    )
    aff = (
        df[
            [
                "ranking_affordability",
                "country_affordability",
                "score_affordability",
            ]
        ]
        .pipe(_rename_cols)
        .assign(indicator="affordability")
    )
    ava = (
        df[["ranking_availability", "country_availability", "score_availability"]]
        .pipe(_rename_cols)
        .assign(indicator="availability")
    )
    qua = (
        df[["ranking_quality", "country_quality", "score_quality"]]
        .pipe(_rename_cols)
        .assign(indicator="quality_and_safety")
    )
    nat = (
        df[["ranking_natural", "country_natural", "score_natural"]]
        .pipe(_rename_cols)
        .assign(indicator="natural_resources_and_resilience")
    )

    data = (
        pd.concat([ov, aff, ava, qua, nat], ignore_index=True)
        .astype({"score": float})
        .pipe(add_iso_codes, id_col="country")
    )

    data.to_csv(f"{PATHS.data}/full_economist_fs_index.csv", index=False)


@dataclass
class EconomistIndex:
    refresh: bool = False
    full_data: pd.DataFrame = None

    """A class to get and extract data from the Economist Food Security Index
    Attributes:
        refresh : bool
            Whether to refresh the data from the website
    
    Methods:
        get_overall_data() : pd.DataFrame
        Returns a dataframe with the overall index data
        
        get_affordability_data() : pd.DataFrame
        Returns a dataframe with the affordability index data
        
        get_availability_data() : pd.DataFrame
        Returns a dataframe with the availability index data
        
        get_quality_data() : pd.DataFrame
        Returns a dataframe with the quality and safety index data
        
        get_natural_resource_data() : pd.DataFrame
        Returns a dataframe with the natural resources and resilience index data
        
    """

    def __post_init__(self):
        if self.refresh:
            _download_data()

        if not os.path.exists(f"{PATHS.data}/full_economist_fs_index.csv"):
            _download_data()

        self.full_data = pd.read_csv(
            f"{PATHS.data}/full_economist_fs_index.csv", dtype={"score": float}
        )

    def get_overall_data(self) -> pd.DataFrame:
        return self.full_data.loc[self.full_data.indicator == "overall"]

    def get_affordability_data(self) -> pd.DataFrame:
        return self.full_data.loc[self.full_data.indicator == "affordability"]

    def get_availability_data(self) -> pd.DataFrame:
        return self.full_data.loc[self.full_data.indicator == "availability"]

    def get_quality_data(self) -> pd.DataFrame:
        return self.full_data.loc[self.full_data.indicator == "quality_and_safety"]

    def get_natural_resource_data(self) -> pd.DataFrame:
        return self.full_data.loc[
            self.full_data.indicator == "natural_resources_and_resilience"
        ]


if __name__ == "__main__":

    # Example use
    index = EconomistIndex()
    overall = index.get_overall_data()
