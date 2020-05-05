import numpy as np


def get_default_colors(n, type):
    default_colors = ["#8bdcbe", "#f05a71", "#371ea3", "#46bac2", "#ae2c87", "#ffa58c", "#4378bf"]

    if n > len(default_colors):
        ret = []
        for i in range(np.ceil(n / len(default_colors)).astype(int)):
            ret += default_colors
        return ret

    bar_colors = {
        1: ["#46bac2"],
        2: ["#8bdcbe", "#4378bf"],
        3: ["#8bdcbe", "#4378bf", "#46bac2"],
        4: ["#46bac2", "#371ea3", "#8bdcbe", "#4378bf"],
        5: ["#8bdcbe", "#f05a71", "#371ea3", "#46bac2", "#ffa58c"],
        6: ["#8bdcbe", "#f05a71", "#371ea3", "#46bac2", "#ae2c87", "#ffa58c"],
        7: default_colors
    }

    line_colors = {
        1: ["#46bac2"],
        2: ["#8bdcbe", "#4378bf"],
        3: ["#8bdcbe", "#f05a71", "#4378bf"],
        4: ["#8bdcbe", "#f05a71", "#4378bf", "#ffa58c"],
        5: ["#8bdcbe", "#f05a71", "#4378bf", "#ae2c87", "#ffa58c"],
        6: ["#8bdcbe", "#f05a71", "#46bac2", "#ae2c87", "#ffa58c", "#4378bf"],
        7: default_colors
    }

    if type == 'bar':
        return bar_colors[n]
    elif type == 'line':
        return line_colors[n]
    else:
        return bar_colors[n]


def get_break_down_colors():
    return ["#371ea3", "#8bdcbe", "#f05a71"]
