import os

import pandas as pd


def load_titanic():
    """Load the preprocessed 'titanic' dataset

    Returns
    -----------
    pd.DataFrame
    """

    abs_dir_path = os.path.dirname(os.path.abspath(__file__))
    abs_datasets_path = os.path.join(abs_dir_path, 'data', 'titanic.csv')

    dataset = pd.read_csv(abs_datasets_path)

    return dataset


def load_fifa():
    """Load the preprocessed 'players_20' dataset

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

    Returns
    -----------
    pd.DataFrame
    """

    abs_dir_path = os.path.dirname(os.path.abspath(__file__))
    abs_datasets_path = os.path.join(abs_dir_path, 'data', 'fifa.csv')

    dataset = pd.read_csv(abs_datasets_path, index_col=0)  # use short_name as index

    return dataset


def load_apartments():
    """Loads the apartments data set

    Datasets apartments and apartments_test are artificial,
    generated from the same model. Structure of the dataset is copied
    from real dataset from PBImisc package, but they were generated
    in a way to mimic effect of Anscombe quartet for complex black box models.

    Returns
    -----------
    pd.DataFrame
    """

    abs_dir_path = os.path.dirname(os.path.abspath(__file__))
    abs_datasets_path = os.path.join(abs_dir_path, 'data', 'apartments.csv')

    dataset = pd.read_csv(abs_datasets_path, index_col=0, sep=';')

    return dataset


def load_apartments_test():
    """Loads the apartments test data set

    Datasets apartments and apartments_test are artificial,
    generated from the same model. Structure of the dataset is copied
    from real dataset from PBImisc package, but they were generated
    in a way to mimic effect of Anscombe quartet for complex black box models.

    Returns
    -----------
    pd.DataFrame
    """

    abs_dir_path = os.path.dirname(os.path.abspath(__file__))
    abs_datasets_path = os.path.join(abs_dir_path, 'data', 'apartments_test.csv')

    dataset = pd.read_csv(abs_datasets_path, index_col=0, sep=';')

    return dataset


def load_dragons():
    """Load the dragons data set

    Data sets dragons and dragons_test are artificial,
    generated from the same ground truth model,
    but with sometimes different data distridution.

    Returns
    -----------
    pd.DataFrame
    """

    abs_dir_path = os.path.dirname(os.path.abspath(__file__))
    abs_datasets_path = os.path.join(abs_dir_path, 'data', 'dragons.csv')

    dataset = pd.read_csv(abs_datasets_path, index_col=0, sep=';')

    return dataset


def load_dragons_test():
    """Load the dragons test data set

    Data sets dragons and dragons_test are artificial,
    generated from the same ground truth model,
    but with sometimes different data distridution.

    Returns
    -----------
    pd.DataFrame
    """

    abs_dir_path = os.path.dirname(os.path.abspath(__file__))
    abs_datasets_path = os.path.join(abs_dir_path, 'data', 'dragons_test.csv')

    dataset = pd.read_csv(abs_datasets_path, index_col=0, sep=';')

    return dataset


def load_hr():
    """Load the HR data set

    Datasets HR and HR_test are artificial, generated from the same model.
    Structure of the dataset is based on a real data, from Human Resources department
    with information which employees were promoted, which were fired.

    Returns
    -----------
    pd.DataFrame
    """

    abs_dir_path = os.path.dirname(os.path.abspath(__file__))
    abs_datasets_path = os.path.join(abs_dir_path, 'data', 'hr.csv')

    dataset = pd.read_csv(abs_datasets_path, index_col=0, sep=';')

    return dataset


def load_hr_test():
    """Load the HR test data set

    Datasets HR and HR_test are artificial, generated from the same model.
    Structure of the dataset is based on a real data, from Human Resources department
    with information which employees were promoted, which were fired.

    Returns
    -----------
    pd.DataFrame
    """

    abs_dir_path = os.path.dirname(os.path.abspath(__file__))
    abs_datasets_path = os.path.join(abs_dir_path, 'data', 'hr_test.csv')

    dataset = pd.read_csv(abs_datasets_path, index_col=0, sep=';')

    return dataset

def load_german():
    """Load the German Credit data

    Dataset contains information about people and their credit risk. On the base of age, purpose, credit amount, job, sex, etc...
    model should predict target variable Risk. Risk tells if credit rate will be good (1) or bad (0).
    This data contains some bias and it can be discovered with fairness module.
    It has 1000 rows and 10 columns.
    The original source of this data is https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data).
    It can also be found on Kaggle https://www.kaggle.com/kabure/german-credit-data-with-risk/

    Returns
    -----------
    pd.DataFrame
    """
    abs_dir_path = os.path.dirname(os.path.abspath(__file__))
    abs_datasets_path = os.path.join(abs_dir_path, 'data', 'german.csv')

    dataset = pd.read_csv(abs_datasets_path, index_col= False)

    return dataset