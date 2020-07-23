import numpy as np


def tooltip_text(obs):
    temp = "</br>" + 'id: ' + str(obs['_ids_']) + "</br>" + \
        'prediction: ' + str(np.around(obs['_yhat_'], 3)) + "</br>" + \
        str(obs['_vname_']) + ': ' + str(obs[obs['_vname_']]) + "</br></br>"

    for index, value in obs.items():
        if index not in ["_ids_", "_yhat_", obs['_vname_'], "_x_", "_label_", "_vname_", "_original_"]:
            temp += str(index) + ": " + str(value) + "</br>"
        if len(temp) > 500:
            temp += "... too many variables"
            break
    return temp
