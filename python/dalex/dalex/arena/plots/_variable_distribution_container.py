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
        from .._native_eda import distribution
        self.plot_component, self.data = distribution(dataset.dataset, variable.variable, self.get_option('bins'))
