import numpy as np


def prepare_data_for_shap_plot(x, baseline, prediction, max_vars, rounding_function, digits):

    variable_count = x.shape[0]

    if variable_count > max_vars:
        last_row = max_vars - 1
        new_x = x.iloc[0:(last_row + 1), :].copy()
        new_x.iloc[last_row, new_x.columns.get_loc('variable')] = "+ all other factors"
        new_x.iloc[last_row, new_x.columns.get_loc('contribution')] = np.sum(
            x.iloc[last_row:(variable_count - 1), x.columns.get_loc('contribution')])

        x = new_x.copy()

    # use for text label and tooltip
    x.loc[:, 'contribution'] = rounding_function(x.loc[:, 'contribution'], digits)
    baseline = rounding_function(baseline, digits)
    prediction = rounding_function(prediction, digits)

    tt = x.apply(lambda row: tooltip_text(row, baseline, prediction), axis=1)
    x = x.assign(tooltip_text=tt.values)

    lt = label_text(x.iloc[:, x.columns.get_loc("contribution")].tolist())
    x = x.assign(label_text=lt)

    return x


def tooltip_text(row, baseline, prediction):
    if row.contribution > 0:
        key_word = "increases"
    else:
        key_word = "decreases"
    return "Average response: " + str(baseline) + "<br>Prediction: " + str(prediction) + "<br>" +\
           row.variable + "<br>" + key_word + " average response <br>by " + str(np.abs(row.contribution))


def label_text(contribution):
    def to_text(x):
        if x > 0:
            return "+" + str(x)
        else:
            return str(x)

    return [to_text(c) for c in contribution]
