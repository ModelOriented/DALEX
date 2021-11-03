from .._plot_container import PlotContainer

class MetricsContainer(PlotContainer):
    info = {
        'name': "Metrics",
        'plotType': "Metrics",
        'plotCategory': "Model Performance",
        'requiredParams': ["model"]
    }
    options_category = 'Metrics'
    options = {}
    def _fit(self, model):
        perf = model.explainer.model_performance().result
        self.data = dict(perf.iloc[0])
