import numpy as np
import narwhals.stable.v2 as nw
from types import SimpleNamespace
from ... import _global_utils


def prepare_data_for_shap_plot(x, baseline, prediction, max_vars, rounding_function, digits):
    namespace = nw.get_native_namespace(x)
    x = x.select('variable', 'contribution', 'sign')
    order = np.argsort(-np.abs(x['contribution'].to_numpy()), kind='quicksort')
    x = x[order.tolist(), :]
    if len(x) > max_vars:
        last_row = max_vars - 1
        contribution_sum = np.nansum(x[last_row:, :]['contribution'].to_numpy())
        other = x[last_row:last_row + 1, :].with_columns(
            nw.lit('+ all other factors').alias('variable'),
            nw.lit(contribution_sum).alias('contribution'),
            nw.lit(float(np.sign(contribution_sum))).alias('sign'),
        )
        x = nw.concat([x.head(last_row), other], how='vertical')
    argument = x['contribution'].to_native() if namespace.__name__ == 'pandas' else x['contribution'].to_numpy()
    contributions = np.asarray(rounding_function(argument, digits))
    x = x.with_columns(nw.new_series('contribution', contributions, backend=namespace))
    baseline = rounding_function(baseline, digits)
    prediction = rounding_function(prediction, digits)
    tooltips = [tooltip_text(SimpleNamespace(**row), baseline, prediction) for row in x.iter_rows(named=True)]
    return x.with_columns(
        nw.new_series('tooltip_text', tooltips, backend=namespace),
        nw.new_series('label_text', _global_utils.convert_float_to_str(contributions, '+'), backend=namespace),
    )


def tooltip_text(row, baseline, prediction):
    if row.contribution > 0:
        key_word = "increases"
    else:
        key_word = "decreases"
    return "Average response: " + str(baseline) + "<br>Prediction: " + str(prediction) + "<br>" + \
           str(row.variable) + "<br>" + key_word + " average response <br>by " + str(np.abs(row.contribution))

