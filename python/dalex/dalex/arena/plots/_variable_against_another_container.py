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
        from .._native_eda import against
        self.data = against(dataset.dataset, variable.variable, self.get_option('points_number'))
