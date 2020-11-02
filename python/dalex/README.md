# dalex

[moDel Agnostic Language for Exploration and eXplanation](http://dalex.drwhy.ai/)

[![Python-check](https://github.com/ModelOriented/DALEX/workflows/Python-check/badge.svg)](https://github.com/ModelOriented/DALEX/actions?query=workflow%3APython-check)
[![Supported Python
versions](https://img.shields.io/pypi/pyversions/dalex.svg)](https://pypi.org/project/dalex/)
[![PyPI version](https://badge.fury.io/py/dalex.svg)](https://badge.fury.io/py/dalex)
[![Downloads](https://pepy.tech/badge/dalex)](https://pepy.tech/project/dalex)

## Overview

Unverified black box model is the path to the failure. Opaqueness leads to distrust. Distrust leads to ignoration. Ignoration leads to rejection.

The `dalex` package xrays any model and helps to explore and explain its behaviour, helps to understand how complex models are working.
The main `Explainer` object creates a wrapper around a predictive model. Wrapped models may then be explored and compared with a collection of local and global explainers.
Recent developents from the area of Interpretable Machine Learning/eXplainable Artificial Intelligence.

The philosophy behind `dalex` explanations is described in the [Explanatory Model Analysis](https://pbiecek.github.io/ema/) e-book.

![](https://raw.githubusercontent.com/ModelOriented/DALEX/master/misc/DALEXpiramide.png)

The `dalex` package is a part of [DrWhy.AI](http://DrWhy.AI) universe.

## Installation

The `dalex` package is available [on PyPI](https://pypi.org/project/dalex/)

```console
pip install dalex -U
```

## Resources

* Introduction to the `dalex` package: [Titanic: tutorial and examples](http://dalex.drwhy.ai/python-dalex-titanic.html)
* Important features explained: [FIFA20: explain default vs tuned model with dalex](http://dalex.drwhy.ai/python-dalex-fifa.html)
* [How to use dalex with xgboost](http://dalex.drwhy.ai/python-dalex-xgboost.html)
* [How to use dalex with tensorflow](http://dalex.drwhy.ai/python-dalex-tensorflow.html)
* [Interesting features in v0.2.1](http://dalex.drwhy.ai/python-dalex-new.html)
* [New fairness module](http://dalex.drwhy.ai/python-dalex-fairness.html)
* Code in the form of [jupyter notebook](https://github.com/pbiecek/DALEX_docs/tree/master/jupyter-notebooks)
* [YouTube video](https://www.youtube.com/watch?v=PuKF2GS4_3Y) showing how to do [Break Down](https://pbiecek.github.io/ema/breakDown.html) analysis
* Changelog: [NEWS](https://github.com/ModelOriented/DALEX/blob/master/python/dalex/NEWS.md)
* Theoretical introduction to the plots: [Explanatory Model Analysis: Explore, Explain and Examine Predictive Models](https://pbiecek.github.io/ema)

## Plots

This package uses [plotly](https://plotly.com/python/) to render the plots:

* Install extensions to use `plotly` in **JupyterLab**:&emsp;[Getting Started](https://plot.ly/python/getting-started/#jupyterlab-support-python-35)&emsp;[Troubleshooting](https://plot.ly/python/troubleshooting/#jupyterlab-problems)
* Use `show=False` parameter in `plot` method to return `plotly Figure` object
* It is possible to [edit the figures](https://plotly.com/python/#fundamentals) and [save them](https://plotly.com/python/static-image-export/)

## Learn more

Machine Learning models are widely used and have various applications in classification or regression tasks. Due to increasing computational power, availability of new data sources and new methods, ML models are more and more complex. Models created with techniques like boosting, bagging of neural networks are true black boxes. It is hard to trace the link between input variables and model outcomes. They are use because of high performance, but lack of interpretability is one of their weakest sides.

In many applications we need to know, understand or prove how input variables are used in the model and what impact do they have on final model prediction.
`dalex` is a set of tools that help to understand how complex models are working.

[Talk with your model! at USeR 2020](https://www.youtube.com/watch?v=9WWn5ew8D8o)

## Authors

Main authors of the `dalex` package are:

* [Hubert Baniecki](https://github.com/hbaniecki)
* [Wojciech Kretowicz](https://github.com/wojciechkretowicz)

Under the supervision of [Przemyslaw Biecek](https://github.com/pbiecek).

Other contributors:

* [Jakub Wisnewski](https://github.com/jakwisn) maintains the `fairness` module

-------------------------------------------


