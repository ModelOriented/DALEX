import numpy as np
from .utils import roc_curve
from ..._dataframe import column_values
from ..._plotting import descending_order
import plotly.graph_objects as go

def ecdf(x):
    # https://community.plot.ly/t/plot-the-empirical-cdf/29045
    x = np.sort(x)

    def result(v):
        return np.searchsorted(x, v, side='right') / x.size

    return result


def plot_ecdf(df_list, colors, title):
    fig = go.Figure()

    for i, _df in enumerate(df_list):
        _abs_residuals = np.abs(_df['residuals'].to_numpy())
        _unique_abs_residuals = np.unique(_abs_residuals)

        fig.add_scatter(
            x=_unique_abs_residuals,
            y=1 - ecdf(_abs_residuals)(_unique_abs_residuals),
            line_shape='hv',
            name=_df['label'][0],
            marker=dict(color=colors[i])
        )

    fig.update_yaxes({'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True, 'ticks': 'outside',
                    'tickcolor': 'white', 'ticklen': 10, 'fixedrange': True, 'tickformat': ',.0%'})

    fig.update_xaxes({'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True, 'ticks': "outside",
                    'tickcolor': 'white', 'ticklen': 10, 'fixedrange': True, 'title_text': '|residual|'})
    
    title = "Reverse cumulative distribution of |residual|" if title is None else title
    fig.update_layout(title_text=title, title_x=0.15, font={'color': "#371ea3"}, template="none",
                    margin={'t': 78, 'b': 71, 'r': 30})
    
    return fig


def plot_roc(df_list, colors, title):
    fig = go.Figure()
    grid_points = 101
    for i, df in enumerate(df_list):
        # The same vector kernel supplies scalar AUC and plotted ROC, preserving
        # missing-score omission, ties and the original target denominators.
        fpr, tpr = roc_curve(column_values(df.to_native(), 'y_hat', preserve_nulls=True),
                             column_values(df.to_native(), 'y', preserve_nulls=True))
        if len(fpr) > grid_points:
            positions = np.random.choice(len(fpr), size=grid_points, replace=False)
            positions = positions[np.argsort(fpr[positions], kind='quicksort')]
            tpr, fpr = tpr[positions], fpr[positions]

        fig.add_scatter(
            x=[0] + fpr.tolist(),
            y=[0] + tpr.tolist(),
            line_shape='hv',
            name=df['label'][0],
            marker=dict(color=colors[i])
        )

    fig.update_yaxes({'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True, 'ticks': 'outside',
                    'tickcolor': 'white', 'ticklen': 10, 'fixedrange': True, 'title_text': 'True positive rate'})

    fig.update_xaxes({'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True, 'ticks': "outside",
                    'tickcolor': 'white', 'ticklen': 10, 'fixedrange': True, 'title_text': 'False positive rate'})
    title = "Receiver Operating Characteristic" if title is None else title
    fig.update_layout(title_text=title, title_x=0.15, font={'color': "#371ea3"}, template="none",
                      margin={'t': 78, 'b': 71, 'r': 30})
    
    return fig

def plot_lift(df_list, colors, title):
    fig = go.Figure()
    grid_points = 101
    max_lift = sum(np.nansum(df['y'].to_numpy()) for df in df_list) / sum(len(df) for df in df_list)

    for i, _df in enumerate(df_list):
        order = descending_order(_df['y_hat'].to_numpy())
        n = len(_df)
        y = _df['y'].to_numpy()[order]
        # pandas cumsum skips missing targets without turning all later values
        # into NaN; retain a missing value only at that target's own position.
        cumulative = np.nancumsum(y).astype(float)
        cumulative[np.isnan(y)] = np.nan
        pr = np.linspace(0, 1, n)
        with np.errstate(divide='ignore', invalid='ignore'):
            lift = (cumulative / n) / pr
        if n > grid_points:
            # Select each model's own rows: the previous shared index could be
            # out of range for shorter models, and had a zero step below 101.
            positions = np.arange(n, step=max(1, int(n / grid_points)))
            pr, lift = pr[positions], lift[positions]

        fig.add_scatter(
            x=pr,
            y=lift/max_lift,
            line_shape='hv',
            name=_df['label'][0],
            marker=dict(color=colors[i])
        )
        
    fig.update_yaxes({'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True, 'ticks': 'outside',
                    'tickcolor': 'white', 'ticklen': 10, 'fixedrange': True, 'title_text': 'Lift'})

    fig.update_xaxes({'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True, 'ticks': "outside",
                    'tickcolor': 'white', 'ticklen': 10, 'fixedrange': True, 'title_text': 'Positive rate'})
    title = "LIFT chart" if title is None else title
    fig.update_layout(title_text=title, title_x=0.15, font={'color': "#371ea3"}, template="none",
                      margin={'t': 78, 'b': 71, 'r': 30})
    
    return fig

