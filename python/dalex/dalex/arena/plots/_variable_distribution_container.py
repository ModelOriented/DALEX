import pandas as pd
import numpy as np
from .._plot_container import PlotContainer

class VariableDistributionContainer(PlotContainer):
    info = {
        'name': 'Variable Distribution',
        'plotType': 'VariableDistribution',
        'plotCategory': 'EDA',
        'requiredParams': ['dataset', 'variable']
    }
    options_category = 'VariableDistribution'
    options = {
        'bins': {
            'default': list(range(5, 41, 5)),
            'desc': 'List of available bin counts for the variable distribution plot'
        }
    }

    def _fit(self, dataset, variable):
        variable_column = dataset.dataset.loc[:, variable.variable]
        if pd.api.types.is_numeric_dtype(variable_column):
            bins = self.get_option('bins')
            data = {}
            # get histogram for each bin number
            for bin in bins:
                # compute breaks first for the output
                breaks = np.histogram_bin_edges(variable_column, bins=bin)
                hist_data = np.histogram(variable_column, bins=breaks)
                counts = hist_data[0]
                data[bin] = {
                    'breaks': breaks.tolist(),
                    'mids': ((breaks[1:] + breaks[:-1]) / 2).tolist(),
                    'density': (counts / counts.sum()).tolist(),
                    'counts': counts.tolist()
                }
            self.data = data
            self.plot_component = "DistributionHistogram"
        else:
            counts = variable_column.value_counts()
            self.data = {
                'names': counts.index.tolist(),
                'count': counts.tolist(),
                'density': (counts / counts.sum()).tolist()
            }
            self.plot_component = "DistributionCounts"
