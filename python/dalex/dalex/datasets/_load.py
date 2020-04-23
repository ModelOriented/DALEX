import pandas as pd
import os


def load_titanic():
    """
    Load the preprocessed titanic dataset.
    """

    abs_dir_path = os.path.dirname(os.path.abspath(__file__))
    abs_datasets_path = os.path.join(abs_dir_path, 'data', 'titanic.csv')

    dataset = pd.read_csv(abs_datasets_path)

    return dataset


def load_fifa():
    """
    Load 'fifa', the preprocessed 'players_20.csv' dataset which comes as
    a part of 'FIFA 20 complete player dataset' at 'Kaggle'.
    It contains 5000 'overall' best players and 43 variables. These are:
    - short_name (index)
    - nationality of the player (not used in modeling)
    - overall, potential, value_eur, wage_eur (4 potential target variables)
    - age, height, weight, attacking skills, defending skills, goalkeeping skills (37 variables)

    It is advised to leave only one target variable for modeling.

    Format: pd.DataFrame with 5000 rows, 42 columns and index
    Source: https://www.kaggle.com/stefanoleone992/fifa-20-complete-player-dataset#players_20.csv January 1, 2020
    """

    abs_dir_path = os.path.dirname(os.path.abspath(__file__))
    abs_datasets_path = os.path.join(abs_dir_path, 'data', 'fifa.csv')

    dataset = pd.read_csv(abs_datasets_path, index_col=0)  # use short_name as index

    return dataset
