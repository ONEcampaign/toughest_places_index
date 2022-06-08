"""Configuration scripts for the project"""

import os


class Paths:
    """A class which helps with managing access to the right file paths, independent
    of operating system, user or setup"""

    def __init__(self, project_dir):
        self.project_dir = project_dir

    @property
    def media(self) -> str:
        return os.path.join(self.project_dir, "media")

    @property
    def root(self) -> str:
        return os.path.join(self.project_dir)

    @property
    def data(self) -> str:
        return os.path.join(self.project_dir, "data")

    @property
    def raw_hunger(self) -> str:
        return os.path.join(self.data, "wfp_insufficient_food_raw")

    @property
    def raw_wfp_inflation(self) -> str:
        return os.path.join(self.data, "wfp_inflation_raw")


# Paths object with property access to the right places
PATHS: Paths = Paths(os.path.dirname(os.path.dirname(__file__)))


DEBT_SERVICE_IDS: dict = {
    "DT.AMT.BLAT.CD": "Bilateral",
    "DT.AMT.MLAT.CD": "Multilateral",
    "DT.AMT.PBND.CD": "Private",
    "DT.AMT.PCBK.CD": "Private",
    "DT.AMT.PROP.CD": "Private",
    "DT.INT.BLAT.CD": "Bilateral",
    "DT.INT.MLAT.CD": "Multilateral",
    "DT.INT.PBND.CD": "Private",
    "DT.INT.PCBK.CD": "Private",
    "DT.INT.PROP.CD": "Private",
}