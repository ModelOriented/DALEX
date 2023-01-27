from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT_DIR / "data"

MODELS = ("svm", "xgboost")
DATASETS = ("housing",)  # TODO
METHODS = ("exact", "kernel", "unbiased")
