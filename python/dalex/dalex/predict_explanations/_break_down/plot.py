import numpy as np
import narwhals.stable.v2 as nw
from types import SimpleNamespace
from ... import _global_utils


def prepare_data_for_break_down_plot(x, baseline, max_vars, rounding_function, digits):
    namespace = nw.get_native_namespace(x)
    x = x.select('variable', 'contribution', 'cumulative')
    first, last = x.head(1), x.tail(1)
    middle = x[1:-1, :]
    if len(middle) > max_vars:
        remainder = middle[max_vars:, :]
        other = remainder.head(1).with_columns(
            nw.lit('+ all other factors').alias('variable'),
            nw.lit(np.nansum(remainder['contribution'].to_numpy())).alias('contribution'),
            nw.lit(middle['cumulative'][-1]).alias('cumulative'),
        )
        middle = nw.concat([middle.head(max_vars), other], how='vertical')
    x = nw.concat([first, middle, last], how='vertical')
    contribution = x['contribution'].to_numpy().copy()
    contribution[[0, -1]] -= baseline
    # Existing pandas callbacks receive a Series; native callbacks receive the
    # numerical vectors already needed to draw the bars.
    argument = nw.new_series('contribution', contribution, backend=namespace)
    argument = argument.to_native() if namespace.__name__ == 'pandas' else argument.to_numpy()
    cumulative_arg = x['cumulative'].to_native() if namespace.__name__ == 'pandas' else x['cumulative'].to_numpy()
    contribution = np.asarray(rounding_function(argument, digits))
    cumulative = np.asarray(rounding_function(cumulative_arg, digits))
    x = x.with_columns(nw.new_series('contribution', contribution, backend=namespace),
                       nw.new_series('cumulative', cumulative, backend=namespace))
    labels = _global_utils.convert_float_to_str(contribution, '+')
    tooltips = [tooltip_text(SimpleNamespace(**row)) for row in x.iter_rows(named=True)]
    endpoint = 'Average response: ' + str(cumulative[0]) + '<br>Prediction: ' + str(cumulative[-1])
    tooltips[0] = tooltips[-1] = endpoint
    labels[0], labels[-1] = str(cumulative[0]), str(cumulative[-1])
    return x.with_columns(nw.new_series('tooltip_text', tooltips, backend=namespace),
                          nw.new_series('label_text', labels, backend=namespace))


def tooltip_text(row):
    if row.contribution > 0:
        key_word = "increases"
    else:
        key_word = "decreases"
    return str(row.variable) + "<br>" + key_word + " average response by"

