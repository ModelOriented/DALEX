# import numpy as np


# def tooltip_text(r, variable_name, y_mean):
#     return str(variable_name) + ": " + str(r['_x_']) + "<br>" + "average prediction: " + str(r['_yhat_']) + "<br>" + \
#            "label: " + str(r['_label_']) + "<br><br>" + "mean observation prediction: " + "<br>" + str(y_mean)


# def check_for_groups(ap):
#     if '_groups_' in ap.result.columns:
#         return [ap.mean_prediction for _ in np.unique(ap.result.loc[:, '_groups_'])]
#     else:
#         return [ap.mean_prediction]
