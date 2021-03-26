import numpy as np
from ... import _global_utils


def prepare_data_for_shap_plot(x, baseline, prediction, max_vars, rounding_function, digits):
    variable_count = x.shape[0]
    # sort by absolute value of contribution
    x = x.iloc[(-x['contribution'].abs()).argsort()].reset_index(drop=True)

    if variable_count > max_vars:
        last_row = max_vars - 1
        new_x = x.iloc[0:(last_row + 1), :].copy()
        new_x.iloc[last_row, new_x.columns.get_loc('variable')] = "+ all other factors"
        contribution_sum = np.sum(x.iloc[last_row::, x.columns.get_loc('contribution')])
        new_x.iloc[last_row, new_x.columns.get_loc('contribution')] = contribution_sum
        new_x.iloc[last_row, new_x.columns.get_loc('sign')] = np.sign(contribution_sum)

        x = new_x.copy()

    # use for text label and tooltip
    x.loc[:, 'contribution'] = rounding_function(x.loc[:, 'contribution'], digits)
    baseline = rounding_function(baseline, digits)
    prediction = rounding_function(prediction, digits)

    x = x.assign(tooltip_text=x.apply(lambda row: tooltip_text(row, baseline, prediction), axis=1),
                 label_text=_global_utils.convert_float_to_str(x.contribution, "+"))

    return x


def tooltip_text(row, baseline, prediction):
    if row.contribution > 0:
        key_word = "increases"
    else:
        key_word = "decreases"
    return "Average response: " + str(baseline) + "<br>Prediction: " + str(prediction) + "<br>" + \
           str(row.variable) + "<br>" + key_word + " average response <br>by " + str(np.abs(row.contribution))

