import copy
from dataclasses import dataclass, field
from index.data.dimension import Dimension
import pandas as pd

from index.data.imputers import IMPUTERS
from index.data.summary import (
    collinearity,
    check_zeros,
    outliers,
    summarize_missing_full,
)


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

    def index_dev(
        self,
        rescale_parameters: dict = None,
        impute_parameters: dict = None,
    ) -> pd.DataFrame:

        original = copy.deepcopy(self)
        original = original.get_data()

        self.index_data(
            rescale_parameters=rescale_parameters,
            impute_parameters=impute_parameters,
            summarised=False,
        )
        indexed = self.get_data()

        cols = original.columns
        new_cols = []
        for col in cols:
            new_cols.append(col)
            new_cols.append(f"{col}_raw")

        df = indexed.merge(
            original,
            left_index=True,
            right_index=True,
            how="left",
            suffixes=("", "_raw"),
        )
        df = df.filter(new_cols, axis=1)
        return df

    def tune_neighbors(self, rescale_parameters: dict, n_neighbors: int) -> float:
        """Tune the number of neighbors for the imputation method"""
        import numpy as np

        # Rescale the dataset
        self.rescale(**rescale_parameters)

        def __neighbours_test(n: int) -> float:
            # Make a copy of the data
            new = copy.deepcopy(self)

            # Track the introduced missing data

            changed_data = {}
            # introduce a few errors
            for dimension in new.dimensions:
                for indicator in dimension.indicators:
                    to_change = indicator.data.sample(frac=0.05)
                    indicator.data.iloc[to_change.index, 1] = np.nan
                    changed_data[indicator.indicator_name] = to_change

            # impute the changed data
            new.impute_missing_data(method="knn", n_neighbors=n)

            df = pd.DataFrame()
            # check the errors
            for col in new.data.columns:
                changed = new.data.iloc[changed_data[col].index, 1]
                original = changed_data[col]
                _ = pd.DataFrame({"changed": changed}).reset_index()
                _ = original.merge(_, on="iso_code").rename(
                    columns={"value": "original"}
                )
                _["indicator"] = col
                df = pd.concat([df, _], ignore_index=True)

            df["difference"] = round(
                100 * abs(df.original - df.changed) / df.original, 1
            )

            return df.replace(np.inf, np.nan).difference.mean()

        results = []
        for version in range(500):
            results.append(__neighbours_test(n=n_neighbors))

        return round(sum(results) / len(results), 1)

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

    def rescale_index(self) -> None:
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
