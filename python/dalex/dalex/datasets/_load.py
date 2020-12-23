import os

import pandas as pd


def load_titanic():
    """Load the preprocessed 'titanic' dataset
    
    Details: https://modeloriented.github.io/DALEX/reference/titanic.html

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
    
    License: see file ./data/LICENSE-DATA.txt

    Returns
    -----------
    pd.DataFrame
    """

    abs_dir_path = os.path.dirname(os.path.abspath(__file__))
    abs_datasets_path = os.path.join(abs_dir_path, 'data', 'fifa.csv')

    dataset = pd.read_csv(abs_datasets_path, index_col=0)  # use short_name as index

    return dataset


def load_apartments():
    """Loads the artificial 'apartments' dataset

    Datasets 'apartments' and 'apartments_test' are artificial, generated
    from the same model. Structure of the dataset is copied from the real
    dataset from the PBImisc R package, but they were generated in a way
    to mimic the effect of Anscombe quartet for complex black-box models.

    Returns
    -----------
    pd.DataFrame
    """

    abs_dir_path = os.path.dirname(os.path.abspath(__file__))
    abs_datasets_path = os.path.join(abs_dir_path, 'data', 'apartments.csv')

    dataset = pd.read_csv(abs_datasets_path, index_col=0)  # use 1:1000 as index

    return dataset


def load_apartments_test():
    """Loads the artificial 'apartments_test' dataset

    Datasets 'apartments' and 'apartments_test' are artificial, generated
    from the same model. Structure of the dataset is copied from the real
    dataset from the PBImisc R package, but they were generated in a way
    to mimic the effect of Anscombe quartet for complex black-box models.

    Returns
    -----------
    pd.DataFrame
    """

    abs_dir_path = os.path.dirname(os.path.abspath(__file__))
    abs_datasets_path = os.path.join(abs_dir_path, 'data', 'apartments_test.csv')

    dataset = pd.read_csv(abs_datasets_path, index_col=0)  # use 1001:9000 as index

    return dataset


def load_dragons():
    """Load the artificial 'dragons' dataset

    Datasets 'dragons' and 'dragons_test' are artificial,
    generated from the same ground truth model,
    but with sometimes different data distridution.

    Returns
    -----------
    pd.DataFrame
    """

    abs_dir_path = os.path.dirname(os.path.abspath(__file__))
    abs_datasets_path = os.path.join(abs_dir_path, 'data', 'dragons.csv')

    dataset = pd.read_csv(abs_datasets_path, index_col=0)  # use 1:n as index

    return dataset


def load_dragons_test():
    """Load the artificial 'dragons_test' dataset

    Datasets 'dragons' and 'dragons_test' are artificial,
    generated from the same ground truth model,
    but with sometimes different data distridution.

    Returns
    -----------
    pd.DataFrame
    """

    abs_dir_path = os.path.dirname(os.path.abspath(__file__))
    abs_datasets_path = os.path.join(abs_dir_path, 'data', 'dragons_test.csv')

    dataset = pd.read_csv(abs_datasets_path, index_col=0)  # use 1:n as index

    return dataset


def load_hr():
    """Load the artificial 'HR' dataset

    Datasets 'HR' and 'HR_test' are artificial, generated from the same model.
    Structure of the dataset is based on the real data from the Human Resources
    department containing information about which employees were promoted or fired.

    Returns
    -----------
    pd.DataFrame
    """

    abs_dir_path = os.path.dirname(os.path.abspath(__file__))
    abs_datasets_path = os.path.join(abs_dir_path, 'data', 'hr.csv')

    dataset = pd.read_csv(abs_datasets_path, index_col=0)  # use 7847 numbers from 1:n as index

    return dataset


def load_hr_test():
    """Load the artificial 'HR_test' dataset

    Datasets 'HR' and 'HR_test' are artificial, generated from the same model.
    Structure of the dataset is based on the real data from the Human Resources
    department containing information about which employees were promoted or fired.

    Returns
    -----------
    pd.DataFrame
    """

    abs_dir_path = os.path.dirname(os.path.abspath(__file__))
    abs_datasets_path = os.path.join(abs_dir_path, 'data', 'hr_test.csv')

    dataset = pd.read_csv(abs_datasets_path, index_col=0)  # use 7847 numbers from 1:n as index

    return dataset


def load_german():
    """Load the preprocessed 'German Credit' dataset

    Dataset 'german' contains information about people and their credit risk.
    On the base of age, purpose, credit amount, job, sex, etc. the model should
    predict the target - risk. risk tells if the credit rate will be good (1) or bad (0).
    This data contains some bias and it can be detected using the dalex.fairness module.

    Format: pd.DataFrame with 1000 rows and 10 columns

    Source: https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data)
    
    Kaggle: https://www.kaggle.com/kabure/german-credit-data-with-risk/
    
    License: see file ./data/LICENSE-DATA.txt
    
    Returns
    -----------
    pd.DataFrame
    """

    abs_dir_path = os.path.dirname(os.path.abspath(__file__))
    abs_datasets_path = os.path.join(abs_dir_path, 'data', 'german.csv')

    dataset = pd.read_csv(abs_datasets_path, index_col=False)

    return dataset
