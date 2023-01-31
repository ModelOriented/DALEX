from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
DATA_DIR = ROOT_DIR / "data"

MODELS = ("svm", "xgboost")
DATASETS = ("housing","cancer")  # TODO
METHODS = ("exact", "kernel", "unbiased")
RANDOM_STATE = 446519
