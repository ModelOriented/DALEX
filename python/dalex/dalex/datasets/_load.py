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
    Load the preprocessed players_20 dataset.
    It contains 5000 'overall' best players and 43 columns. These are:
    - name and nationality of the player (not used in modeling)
    - overall, potential, value_eur, wage_eur (4 potential target variables)
    - age, height, weight, attacking skills, defending skills, goalkeeping skills (37 variables)

    Link: https://www.kaggle.com/stefanoleone992/fifa-20-complete-player-dataset
    """

    abs_dir_path = os.path.dirname(os.path.abspath(__file__))
    abs_datasets_path = os.path.join(abs_dir_path, 'data', 'fifa.csv')

    dataset = pd.read_csv(abs_datasets_path)

    return dataset
