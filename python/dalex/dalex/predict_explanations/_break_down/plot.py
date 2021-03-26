import numpy as np
import pandas as pd
from ... import _global_utils


def prepare_data_for_break_down_plot(x, baseline, max_vars, rounding_function, digits):
    x.loc[x["variable_name"] == "", "variable_name"] = "prediction"

    temp = x.iloc[[0, x.shape[0] - 1], :].copy()
    x = x.drop([0, x.shape[0] - 1])

    variable_count = x.shape[0]

    if variable_count > max_vars:
        new_x = x.iloc[0:(max_vars + 1), :].copy()
        new_x.iloc[max_vars, new_x.columns.get_loc('variable')] = "+ all other factors"
        new_x.iloc[max_vars, new_x.columns.get_loc('contribution')] = \
            np.sum(x.iloc[max_vars::, x.columns.get_loc('contribution')])
        new_x.iloc[max_vars, new_x.columns.get_loc('cumulative')] = \
            x.iloc[variable_count - 1, x.columns.get_loc('cumulative')]
        x = new_x

    x = pd.concat((temp.iloc[[0]], x, temp.iloc[[1]]))

    # fix contribution and sign
    x.iloc[[0, x.shape[0] - 1], x.columns.get_loc("contribution")] -= baseline

    # use for text label and tooltip
    x.loc[:, 'contribution'] = rounding_function(x.loc[:, 'contribution'], digits)
    x.loc[:, 'cumulative'] = rounding_function(x.loc[:, 'cumulative'], digits)

    x = x.assign(tooltip_text=x.apply(lambda row: tooltip_text(row), axis=1),
                 label_text=_global_utils.convert_float_to_str(x.contribution, "+"))

    x.iloc[[0, x.shape[0] - 1], x.columns.get_loc('tooltip_text')] = "Average response: " + str(
        x.iloc[0, x.columns.get_loc('cumulative')]) + "<br>Prediction: " + str(
        x.iloc[x.shape[0] - 1, x.columns.get_loc('cumulative')])

    x.iloc[0, x.columns.get_loc("label_text")] = x.iloc[0, x.columns.get_loc('cumulative')]
    x.iloc[x.shape[0] - 1, x.columns.get_loc("label_text")] = x.iloc[x.shape[0] - 1, x.columns.get_loc('cumulative')]

    return x


def tooltip_text(row):
    if row.contribution > 0:
        key_word = "increases"
    else:
        key_word = "decreases"
    return str(row.variable) + "<br>" + key_word + " average response by"

