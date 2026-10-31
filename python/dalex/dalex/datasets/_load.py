import os
import narwhals.stable.v2 as nw


def _load(name, backend, include_index, *, indexed=False):
    if backend not in ('pandas', 'polars', 'pyarrow'):
        raise ValueError("backend must be 'pandas', 'polars' or 'pyarrow'")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', name + '.csv')
    if backend == 'pandas':
        import pandas as pd
        frame = pd.read_csv(path, index_col=0 if indexed else None)
        return frame.reset_index() if indexed and include_index else frame
    missing = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan',
               '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'None', 'n/a', 'nan', 'null']
    if backend == 'polars':
        frame = nw.read_csv(path, backend=backend, null_values=missing, infer_schema_length=None)
    else:
        from pyarrow.csv import ConvertOptions
        frame = nw.read_csv(path, backend=backend,
                            convert_options=ConvertOptions(null_values=missing, strings_can_be_null=True))
    if indexed:
        first = frame.columns[0]
        if not include_index:
            frame = frame.drop(first)
        elif not first:
            label = 'index'
            while label in frame.columns:
                label = '_' + label
            frame = frame.rename({first: label})
    return frame.to_native()


def load_titanic(*, backend='pandas', include_index=False):
    """Load the preprocessed 'titanic' dataset
    
    Details: https://modeloriented.github.io/DALEX/reference/titanic.html

    Parameters
    ----------
    backend : {'pandas', 'polars', 'pyarrow'}
        Dataframe backend; pandas retains the historical default.
    include_index : bool
        Include CSV row identifiers as an ordinary column. Native defaults
        contain the same feature columns as the pandas table.

    Returns
    -----------
    pandas.DataFrame, polars.DataFrame or pyarrow.Table
    """

    return _load('titanic', backend, include_index, indexed=False)


def load_fifa(*, backend='pandas', include_index=False):
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

    Parameters
    ----------
    backend : {'pandas', 'polars', 'pyarrow'}
        Dataframe backend; pandas retains the historical default.
    include_index : bool
        Include CSV row identifiers as an ordinary column. Native defaults
        contain the same feature columns as the pandas table.

    Returns
    -----------
    pandas.DataFrame, polars.DataFrame or pyarrow.Table
    """

    return _load('fifa', backend, include_index, indexed=True)


def load_apartments(*, backend='pandas', include_index=False):
    """Loads the artificial 'apartments' dataset

    Datasets 'apartments' and 'apartments_test' are artificial, generated
    from the same model. Structure of the dataset is copied from the real
    dataset from the PBImisc R package, but they were generated in a way
    to mimic the effect of Anscombe quartet for complex black-box models.

    Parameters
    ----------
    backend : {'pandas', 'polars', 'pyarrow'}
        Dataframe backend; pandas retains the historical default.
    include_index : bool
        Include CSV row identifiers as an ordinary column. Native defaults
        contain the same feature columns as the pandas table.

    Returns
    -----------
    pandas.DataFrame, polars.DataFrame or pyarrow.Table
    """

    return _load('apartments', backend, include_index, indexed=True)


def load_apartments_test(*, backend='pandas', include_index=False):
    """Loads the artificial 'apartments_test' dataset

    Datasets 'apartments' and 'apartments_test' are artificial, generated
    from the same model. Structure of the dataset is copied from the real
    dataset from the PBImisc R package, but they were generated in a way
    to mimic the effect of Anscombe quartet for complex black-box models.

    Parameters
    ----------
    backend : {'pandas', 'polars', 'pyarrow'}
        Dataframe backend; pandas retains the historical default.
    include_index : bool
        Include CSV row identifiers as an ordinary column. Native defaults
        contain the same feature columns as the pandas table.

    Returns
    -----------
    pandas.DataFrame, polars.DataFrame or pyarrow.Table
    """

    return _load('apartments_test', backend, include_index, indexed=True)


def load_dragons(*, backend='pandas', include_index=False):
    """Load the artificial 'dragons' dataset

    Datasets 'dragons' and 'dragons_test' are artificial,
    generated from the same ground truth model,
    but with sometimes different data distridution.

    Parameters
    ----------
    backend : {'pandas', 'polars', 'pyarrow'}
        Dataframe backend; pandas retains the historical default.
    include_index : bool
        Include CSV row identifiers as an ordinary column. Native defaults
        contain the same feature columns as the pandas table.

    Returns
    -----------
    pandas.DataFrame, polars.DataFrame or pyarrow.Table
    """

    return _load('dragons', backend, include_index, indexed=True)


def load_dragons_test(*, backend='pandas', include_index=False):
    """Load the artificial 'dragons_test' dataset

    Datasets 'dragons' and 'dragons_test' are artificial,
    generated from the same ground truth model,
    but with sometimes different data distridution.

    Parameters
    ----------
    backend : {'pandas', 'polars', 'pyarrow'}
        Dataframe backend; pandas retains the historical default.
    include_index : bool
        Include CSV row identifiers as an ordinary column. Native defaults
        contain the same feature columns as the pandas table.

    Returns
    -----------
    pandas.DataFrame, polars.DataFrame or pyarrow.Table
    """

    return _load('dragons_test', backend, include_index, indexed=True)


def load_hr(*, backend='pandas', include_index=False):
    """Load the artificial 'HR' dataset

    Datasets 'HR' and 'HR_test' are artificial, generated from the same model.
    Structure of the dataset is based on the real data from the Human Resources
    department containing information about which employees were promoted or fired.

    Parameters
    ----------
    backend : {'pandas', 'polars', 'pyarrow'}
        Dataframe backend; pandas retains the historical default.
    include_index : bool
        Include CSV row identifiers as an ordinary column. Native defaults
        contain the same feature columns as the pandas table.

    Returns
    -----------
    pandas.DataFrame, polars.DataFrame or pyarrow.Table
    """

    return _load('hr', backend, include_index, indexed=True)


def load_hr_test(*, backend='pandas', include_index=False):
    """Load the artificial 'HR_test' dataset

    Datasets 'HR' and 'HR_test' are artificial, generated from the same model.
    Structure of the dataset is based on the real data from the Human Resources
    department containing information about which employees were promoted or fired.

    Parameters
    ----------
    backend : {'pandas', 'polars', 'pyarrow'}
        Dataframe backend; pandas retains the historical default.
    include_index : bool
        Include CSV row identifiers as an ordinary column. Native defaults
        contain the same feature columns as the pandas table.

    Returns
    -----------
    pandas.DataFrame, polars.DataFrame or pyarrow.Table
    """

    return _load('hr_test', backend, include_index, indexed=True)


def load_german(*, backend='pandas', include_index=False):
    """Load the preprocessed 'German Credit' dataset

    Dataset 'german' contains information about people and their credit risk.
    On the base of age, purpose, credit amount, job, sex, etc. the model should
    predict the target - risk. risk tells if the credit rate will be good (1) or bad (0).
    This data contains some bias and it can be detected using the dalex.fairness module.

    Format: pd.DataFrame with 1000 rows and 10 columns

    Source: https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data)
    
    Kaggle: https://www.kaggle.com/kabure/german-credit-data-with-risk/
    
    License: see file ./data/LICENSE-DATA.txt
    
    Parameters
    ----------
    backend : {'pandas', 'polars', 'pyarrow'}
        Dataframe backend; pandas retains the historical default.
    include_index : bool
        Include CSV row identifiers as an ordinary column. Native defaults
        contain the same feature columns as the pandas table.

    Returns
    -----------
    pandas.DataFrame, polars.DataFrame or pyarrow.Table
    """

    return _load('german', backend, include_index, indexed=False)
