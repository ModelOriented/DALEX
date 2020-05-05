def tooltip_text(obs, r=None):
    temp = ""
    if r is not None:
        for var in obs.index:
            if var == "_yhat_":
                temp += "prediction:<br>" + "- before: " + str(obs[var]) + "<br>" + "- after: " + \
                        str(r[var]) + "<br><br>"
            elif var == r['_vname_']:
                temp += str(var) + ": " + str(r['_xhat_']) + "</br>"
            else:
                temp += str(var) + ": " + str(obs[var]) + "</br>"
            if len(temp) > 500:
                temp += "... too many variables"
                break
    else:
        for var in obs.index:
            if var == "_yhat_":
                temp += "prediction:" + str(obs[var]) + "<br><br>"
            else:
                temp += str(var) + ": " + str(obs[var]) + "</br>"
            if len(temp) > 500:
                temp += "... too many variables"
                break
    return temp
