import pandas as pd
import numpy as np
from .._plot_container import PlotContainer

class VariableAgainstAnotherContainer(PlotContainer):
    info = {
        'name': 'Variable Against Another',
        'plotType': 'VariableAgainstAnother',
        'plotCategory': 'EDA',
        'plotComponent': 'VariableAgainstAnother',
        'requiredParams': ['dataset', 'variable']
    }
    options_category = 'VariableAgainstAnother'
    options = {
        'points_number': {
            'default': 150, 
            'desc': 'Maximum sample size to visualize in the variable against another scatter plot'
        }
    }

    def _fit(self, dataset, variable):
        variable_column_first = dataset.dataset.loc[:, variable.variable]
        data = {}
        for variable_another in dataset.dataset.columns:    
            if variable.variable == variable_another:
                continue
            variable_column_another = dataset.dataset.loc[:, variable_another]
            # scatter plot for two numeric variables
            if pd.api.types.is_numeric_dtype(variable_column_first) and\
                pd.api.types.is_numeric_dtype(variable_column_another):
                # sample a subset of points
                n_points = min(self.get_option('points_number'), variable_column_first.size)
                ids = np.random.choice(variable_column_first.size, size=n_points, replace=False) 
                data[variable_another] = {
                    'type': 'scatter', 
                    'first': variable_column_first.iloc[ids].tolist(), 
                    'secondary': variable_column_another.iloc[ids].tolist()
                }
            elif pd.api.types.is_numeric_dtype(variable_column_first):
                # boxplots of the first variable
                data_boxplot = []
                for value_another in variable_column_another.unique():
                    variable_column_first_filtered = variable_column_first[variable_column_another == value_another]
                    quantiles = np.nanquantile(variable_column_first_filtered, [0.25, 0.5, 0.75])
                    iqr = quantiles[2] - quantiles[0]
                    # lower, upper fences
                    lf = max(quantiles[0] - (1.5 * iqr), np.nanmin(variable_column_first_filtered))
                    uf = min(quantiles[2] + (1.5 * iqr), np.nanmax(variable_column_first_filtered))
                    data_boxplot += [{
                        'q1': quantiles[0],
                        'q3': quantiles[2],
                        'mean': np.nanmean(variable_column_first_filtered),
                        'median': quantiles[1],
                        'lf': lf,
                        'uf': uf,
                        'outliers': variable_column_first_filtered[(variable_column_first_filtered > uf) |\
                                                                   (variable_column_first_filtered < lf)].tolist()
                    }]
                data[variable_another] = {
                    'type': 'boxplots', 
                    'first': data_boxplot, 
                    'secondary': variable_column_another.unique().tolist(),
                    'numerical': 'first'
                }
            elif pd.api.types.is_numeric_dtype(variable_column_another):
                # boxplots of another variable
                data_boxplot = []
                for value_first in variable_column_first.unique():
                    variable_column_another_filtered = variable_column_another[variable_column_first == value_first]
                    quantiles = np.nanquantile(variable_column_another_filtered, [0.25, 0.5, 0.75])
                    iqr = quantiles[2] - quantiles[0]
                    # lower, upper fences
                    lf = max(quantiles[0] - (1.5 * iqr), np.nanmin(variable_column_another_filtered))
                    uf = min(quantiles[2] + (1.5 * iqr), np.nanmax(variable_column_another_filtered))
                    data_boxplot += [{
                        'q1': quantiles[0],
                        'q3': quantiles[2],
                        'mean': np.nanmean(variable_column_another_filtered),
                        'median': quantiles[1],
                        'lf': lf,
                        'uf': uf,
                        'outliers': variable_column_another_filtered[(variable_column_another_filtered > uf) |\
                                                                    (variable_column_another_filtered < lf)].tolist()
                    }]
                data[variable_another] = {
                    'type': 'boxplots', 
                    'first': variable_column_first.unique().tolist(), 
                    'secondary': data_boxplot,
                    'numerical': 'secondary'
                }
            else:
                # table for two categorical variables
                tab = pd.crosstab(variable_column_first, variable_column_another)
                data[variable_another] = {
                    'type': 'table',
                    'counts': tab.values.tolist(),
                    'first': tab.index.tolist(),
                    'secondary': tab.columns.tolist()
                }
        self.data = data
