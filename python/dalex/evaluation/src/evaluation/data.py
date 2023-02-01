import pandas as pd
from typing import Tuple

from config import DATA_DIR, DATASETS


def load_dataset(dataset_name: str) -> Tuple[pd.DataFrame, pd.Series]:
    if dataset_name not in DATASETS:
        raise ValueError
    elif dataset_name == "housing":
        df = pd.read_csv(DATA_DIR / "input/kc_house_data.csv").dropna()
        X = df[["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", \
                            "grade", "sqft_above", "sqft_basement", "yr_built"]]
        y = df["price"]
    elif dataset_name == "cancer":
        df = pd.read_csv(DATA_DIR / "input/cancer_reg_org.csv", encoding='latin-1').dropna()
        X = df.drop(["TARGET_deathRate", "binnedInc", "Geography"], axis=1)
        y = df["TARGET_deathRate"]

    return X, y


def load_complete_results(name: str) -> pd.DataFrame:
    if name not in DATASETS:
        raise ValueError
    return pd.read_parquet(DATA_DIR / "estimates" / f"{name}_all.parquet")
