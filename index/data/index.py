import copy
from dataclasses import dataclass, field
from index.data.dimension import Dimension
import pandas as pd

from index.data.imputers import IMPUTERS, one_income_imputer
from index.data.summary import (
    collinearity,
    check_zeros,
    outliers,
    summarize_missing_full,
)
import numpy as np


@dataclass
class Index:
    dimensions: list = field(default_factory=list[Dimension])
    data: pd.DataFrame = None

    def add_dimension(self, indicator: Dimension):
        """Add an indicator to the dataframe"""
        self.dimensions.append(indicator)

    def rescale(self, scaler_name: str, **kwargs) -> None:
        """Rescale the data by calling on the dimension rescale method"""
        for dimension in self.dimensions:
            dimension.rescale(scaler_name, **kwargs)

    def impute_missing_data(self, method: str, **kwargs) -> None:
        """Impute missing data at the Index level"""
        if method not in IMPUTERS:
            raise ValueError(
                f"Method {method} not found in the available imputers. "
                f"Available imputers are {IMPUTERS.keys()}"
            )

        self.data = IMPUTERS[method](self.get_data(), **kwargs)

    def check_collinearity(self, **kwargs) -> dict:
        """Check for collinearity in the data"""
        return self.get_data().pipe(collinearity, **kwargs)

    def check_zeros(self, target_col: str) -> dict:
        """Check for zeros in the data"""
        return self.get_data().pipe(check_zeros, target_col=target_col)

    def check_outliers(self, method: str = "empirical", cluster: str = None) -> dict:
        """Check for outliers in the data"""

        out = {}

        for dimension in self.dimensions:
            for indicator in dimension.indicators:
                out[indicator.indicator_name] = indicator.get_data().pipe(
                    outliers, target_col="value", method=method, cluster=cluster
                )

        return out

    def check_missing_data(self) -> pd.DataFrame:
        """Check for missing data in the data"""
        return pd.DataFrame([self.get_data().pipe(summarize_missing_full)]).T

    def index_data(
        self,
        *,
        rescale_parameters: dict = None,
        impute_parameters: dict = None,
        summarised: bool = True,
    ) -> None:
        """Produce a basic index using the data and parameters"""
        if rescale_parameters is None:
            rescale_parameters = {"scaler_name": "quantile"}

        if impute_parameters is None:
            impute_parameters = {"method": "knn", "n_neighbors": 15}

        self.rescale(**rescale_parameters)
        self.impute_missing_data(**impute_parameters)

        # If less is better, invert the data for that indicator.
        # Given some rescaling strategies, it may be better to do 100-x instead of x*-1
        for dimension in self.dimensions:
            for indicator in dimension.indicators:
                if not indicator.more_is_worse:
                    self.data[indicator.indicator_name] *= -1

        # For testing, a simple equally weighted average
        if summarised:
            self.data = self.data.mean(axis=1).sort_values(ascending=False)

    def run_pca(self, components: int = 2) -> tuple:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=components)
        pca.fit_transform(X=self.data.values)

        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        correlations = pd.DataFrame(
            loadings,
            columns=[f"PC{i}" for i in range(1, components + 1)],
            index=self.data.columns,
        )

        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f"PC{i}" for i in range(1, components + 1)],
            index=self.data.columns,
        )

        return pca, loadings, correlations

    def test_stability(
        self,
        rescale_parameters: dict = None,
        impute_parameters: dict = None,
        countries: list = None,
    ):
        """Test stability of index by replacing small amounts of data"""
        if rescale_parameters is None:
            rescale_parameters = {"scaler_name": "quantile"}

        if impute_parameters is None:
            impute_parameters = {"method": "knn", "n_neighbors": 10}

        # Make a copy of the data
        test = copy.deepcopy(self)

        if countries is None:
            # Randomly select the COUNTRIES for which data will be changed
            countries = (
                test.dimensions[0]
                .indicators[0]
                .raw_data.iso_code.sample(frac=0.1)
                .values
            )

        print("Modified COUNTRIES:", countries)

        for dimension in test.dimensions:
            for indicator in dimension.indicators:
                indicator.raw_data.loc[
                    lambda d: d.iso_code.isin(countries), "value"
                ] = np.nan
                imputed = one_income_imputer(indicator.raw_data).reset_index()
                imputed = imputed.loc[lambda d: d.iso_code.isin(countries)]
                indicator.raw_data = indicator.raw_data.loc[
                    lambda d: ~d.iso_code.isin(countries)
                ]
                indicator.raw_data = pd.concat(
                    [indicator.raw_data, imputed], ignore_index=True
                )

        test.rescale(**rescale_parameters)
        test.impute_missing_data(**impute_parameters)

        for dimension in test.dimensions:
            for indicator in dimension.indicators:
                if not indicator.more_is_worse:
                    test.data[indicator.indicator_name] *= -1

        test.rescale(**rescale_parameters)
        test.impute_missing_data(**impute_parameters)

        self.rescale(**rescale_parameters)
        self.impute_missing_data(**impute_parameters)

        # If less is better, invert the data for that indicator.
        # Given some rescaling strategies, it may be better to do 100-x instead of x*-1
        for dimension in self.dimensions:
            for indicator in dimension.indicators:
                if not indicator.more_is_worse:
                    self.data[indicator.indicator_name] *= -1

        df_raw = (
            self.data.mean(axis=1)
            .sort_values(ascending=False)
            .reset_index()
            .reset_index()
            .rename(columns={"index": "rank"})
        )
        df_modified = (
            test.data.mean(axis=1)
            .sort_values(ascending=False)
            .reset_index()
            .reset_index()
            .rename(columns={"index": "rank"})
        )

        df = df_raw.merge(
            df_modified,
            on=["iso_code"],
            suffixes=("_raw", "_modified"),
        ).assign(difference=lambda d: d.rank_raw - d.rank_modified)

        return df

    def rescale_index(self) -> pd.DataFrame:
        """Rescale the index using the data"""
        data = self.data.copy()

        # find the minimum possible score
        min_scores = [data[c].min() for c in data.columns]
        min_score = sum(min_scores) / len(min_scores)
        max_scores = [data[c].max() for c in data.columns]
        max_score = sum(max_scores) / len(max_scores)

        range_ = max_score - min_score

        # rescale the data
        r_data = self.data.mean(axis=1).sort_values(ascending=False).reset_index()
        r_data.columns = ["iso_code", "score"]
        r_data.score += abs(min_score)
        r_data.score = round(100 * r_data.score / range_, 1)

        return r_data.set_index("iso_code")

    def get_data(self, orient="wide", with_date: bool = False) -> pd.DataFrame:
        """Return the stored data. An orientation can be passed ('wide' or 'long')"""

        if self.data is not None:
            return self.data

        if orient not in ["wide", "long"]:
            raise ValueError(f"Orientation must be 'wide' or 'long' but got {orient}")

        df = pd.DataFrame()

        for dimension in self.dimensions:
            _ = dimension.get_data(orient=orient, with_date=with_date)

            if orient == "wide":
                df = pd.concat([df, _.set_index("iso_code")], axis=1)
            else:
                df = pd.concat([df, _], axis=0)

        self.data = df
        return self.data
