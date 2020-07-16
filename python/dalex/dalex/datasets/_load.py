import os

import pandas as pd


def load_titanic():
    """
    Load the preprocessed 'titanic' dataset.

    :return: pandas.DataFrame
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

    :return: pandas.DataFrame
    """

    abs_dir_path = os.path.dirname(os.path.abspath(__file__))
    abs_datasets_path = os.path.join(abs_dir_path, 'data', 'fifa.csv')

    dataset = pd.read_csv(abs_datasets_path, index_col=0)  # use short_name as index

    return dataset


def load_apartments():
    """
    Loads the apartments data set.

    Datasets apartments and apartments_test are artificial,
    generated from the same model. Structure of the dataset is copied
    from real dataset from PBImisc package, but they were generated
    in a way to mimic effect of Anscombe quartet for complex black box models.

    :return: pandas.DataFrame
    """

    abs_dir_path = os.path.dirname(os.path.abspath(__file__))
    abs_datasets_path = os.path.join(abs_dir_path, 'data', 'apartments.csv')

    dataset = pd.read_csv(abs_datasets_path, index_col=0)  # use short_name as index

    return dataset


def load_apartments_test():
    """
    Loads the apartments test data set.

    Datasets apartments and apartments_test are artificial,
    generated from the same model. Structure of the dataset is copied
    from real dataset from PBImisc package, but they were generated
    in a way to mimic effect of Anscombe quartet for complex black box models.

    :return: pandas.DataFrame
    """

    abs_dir_path = os.path.dirname(os.path.abspath(__file__))
    abs_datasets_path = os.path.join(abs_dir_path, 'data', 'apartments_test.csv')

    dataset = pd.read_csv(abs_datasets_path, index_col=0)  # use short_name as index

    return dataset


def load_dragons():
    """
    Load dragons data set.

    Data sets dragons and dragons_test are artificial,
    generated from the same ground truth model,
    but with sometimes different data distridution.

    :return: pandas.DataFrame
    """

    abs_dir_path = os.path.dirname(os.path.abspath(__file__))
    abs_datasets_path = os.path.join(abs_dir_path, 'data', 'dragons.csv')

    dataset = pd.read_csv(abs_datasets_path, index_col=0)  # use short_name as index

    return dataset


def load_dragons_test():
    """
    Load dragons test data set.

    Data sets dragons and dragons_test are artificial,
    generated from the same ground truth model,
    but with sometimes different data distridution.

    :return: pandas.DataFrame
    """

    abs_dir_path = os.path.dirname(os.path.abspath(__file__))
    abs_datasets_path = os.path.join(abs_dir_path, 'data', 'dragons_test.csv')

    dataset = pd.read_csv(abs_datasets_path, index_col=0)  # use short_name as index

    return dataset


def load_hr():
    """
    Load HR data set.

    Datasets HR and HR_test are artificial, generated from the same model.
    Structure of the dataset is based on a real data, from Human Resources department
    with information which employees were promoted, which were fired.

    :return: pandas.DataFrame
    """

    abs_dir_path = os.path.dirname(os.path.abspath(__file__))
    abs_datasets_path = os.path.join(abs_dir_path, 'data', 'hr.csv')

    dataset = pd.read_csv(abs_datasets_path, index_col=0)  # use short_name as index

    return dataset


def load_hr_test():
    """
    Load HR test data set.

    Datasets HR and HR_test are artificial, generated from the same model.
    Structure of the dataset is based on a real data, from Human Resources department
    with information which employees were promoted, which were fired.

    :return: pandas.DataFrame
    """

    abs_dir_path = os.path.dirname(os.path.abspath(__file__))
    abs_datasets_path = os.path.join(abs_dir_path, 'data', 'hr_test.csv')

    dataset = pd.read_csv(abs_datasets_path, index_col=0)  # use short_name as index

    return dataset
