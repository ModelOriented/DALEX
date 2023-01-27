import pandas as pd

from evaluation.config import DATA_DIR, DATASETS


def load_dataset(dataset_name: str) -> pd.DataFrame:
    if dataset_name not in DATASETS:
        raise ValueError
    # TODO


def load_complete_results(name: str) -> pd.DataFrame:
    if name not in DATASETS:
        raise ValueError
    return pd.read_parquet(DATA_DIR / "estimates" / f"{name}_all.parquet")
