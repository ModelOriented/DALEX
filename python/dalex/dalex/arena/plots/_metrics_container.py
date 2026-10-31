import narwhals.stable.v2 as nw
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
        perf = nw.from_native(model.explainer.model_performance().result_as(), eager_only=True).drop("label")
        self.data = dict(zip(perf.columns, next(perf.iter_rows())))
