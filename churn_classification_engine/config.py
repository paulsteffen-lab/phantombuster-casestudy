from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    data_url: str = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTDNEx8k8MRyu0WnI3b5saS2eVjpIo3rYwONcCFpUDlgVIUM4SzyP5UbHXWZl1wnZx-sRctwYZii7Fi/pub?output=csv"
    data_filename: str = "dataset.csv"
    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    models_dir: Path = Path("models")


class HyperparametersSearchArea(BaseSettings):
    n_components: dict = {"name": "n_components", "low": 4, "high": 64, "log": True}
    learning_rate: dict = {
        "name": "learning_rate",
        "low": 0.001,
        "high": 0.1,
        "log": True,
    }
    max_iter: dict = {"name": "max_iter", "low": 50, "high": 200}
    max_depth: dict = {"name": "max_depth", "low": 3, "high": 12}
    min_samples_leaf: dict = {"name": "min_samples_leaf", "low": 10, "high": 50}
    max_bins: dict = {"name": "max_bins", "low": 2, "high": 255}


settings = Settings()
hyperparameters_search_area = HyperparametersSearchArea()
