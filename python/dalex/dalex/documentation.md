

[dalex: Responsible Machine Learning in Python](http://dalex.drwhy.ai/python)

[![Python-check](https://github.com/ModelOriented/DALEX/workflows/Python-check/badge.svg)](https://github.com/ModelOriented/DALEX/actions?query=workflow%3APython-check)
[![Supported Python
versions](https://img.shields.io/pypi/pyversions/dalex.svg)](https://pypi.org/project/dalex/)
[![PyPI version](https://badge.fury.io/py/dalex.svg)](https://badge.fury.io/py/dalex)
[![Downloads](https://pepy.tech/badge/dalex)](https://pepy.tech/project/dalex)

## Overview

Unverified black box model is the path to the failure. Opaqueness leads to distrust. Distrust leads to ignoration. Ignoration leads to rejection.

The `dalex` package xrays any model and helps to explore and explain its behaviour, helps to understand how complex models are working.
The main `Explainer` object creates a wrapper around a predictive model. Wrapped models may then be explored and compared with a collection of model-level and predict-level explanations. Moreover, there are fairness methods and interactive exploration dashboards available to the user.

The philosophy behind `dalex` explanations is described in the [Explanatory Model Analysis](https://pbiecek.github.io/ema/) e-book.

![](https://raw.githubusercontent.com/ModelOriented/DALEX-docs/master/dalex/dalex-diagram.png)

## Installation

The `dalex` package is available [on PyPI](https://pypi.org/project/dalex/)

```console
pip install dalex -U
```

## Examples

* Introduction to the `dalex` package: [Titanic: tutorial and examples](https://dalex.drwhy.ai/python-dalex-titanic.html)
* Key features explained: [FIFA20: explain default vs tuned model with dalex](https://dalex.drwhy.ai/python-dalex-fifa.html)
* How to use dalex with: [xgboost](https://dalex.drwhy.ai/python-dalex-xgboost.html), [tensorflow](https://dalex.drwhy.ai/python-dalex-tensorflow.html), [h2o (feat. autokeras, catboost, lightgbm)](https://dalex.drwhy.ai/python-dalex-h2o.html)
* More explanations: [residuals, shap, lime](https://dalex.drwhy.ai/python-dalex-new.html)
* Introduction to the [Fairness module in dalex](https://dalex.drwhy.ai/python-dalex-fairness.html)
* Tutorial [on bias detection with dalex](https://dalex.drwhy.ai/python-dalex-fairness2.html)
* Introduction to the [Arena module in dalex](https://dalex.drwhy.ai/python-dalex-arena.html)
* Arena documentation: [Getting Started & Demos](https://arena.drwhy.ai/docs/guide/basic-concepts/)
* Code in the form of [jupyter notebook](https://github.com/ModelOriented/DALEX-docs/tree/master/jupyter-notebooks)

## Plots

This package uses [plotly](https://plotly.com/python/) to render the plots:

* Install extensions to use `plotly` in **JupyterLab**:&emsp;[Getting Started](https://plot.ly/python/getting-started/#jupyterlab-support-python-35)&emsp;[Troubleshooting](https://plot.ly/python/troubleshooting/#jupyterlab-problems)
* Use `show=False` parameter in `plot` method to return `plotly Figure` object
* It is possible to [edit the figures](https://plotly.com/python/#fundamentals) and [save them](https://plotly.com/python/static-image-export/)

## Citation

If you use `dalex`, please cite [our paper](https://arxiv.org/abs/2012.14406):

```html
@article{dalex,
  title={{dalex: Responsible Machine Learning with Interactive
          Explainability and Fairness in Python}},
  author={Hubert Baniecki and Wojciech Kretowicz and Piotr Piatyszek
          and Jakub Wisniewski and Przemyslaw Biecek},
  year={2020},
  journal={arXiv:2012.14406},
  url={https://arxiv.org/abs/2012.14406}
}
```

## Developer

### Class diagram

![](https://raw.githubusercontent.com/ModelOriented/DALEX-docs/master/dalex/dalex-class.png)

### Folder structure

![](https://raw.githubusercontent.com/ModelOriented/DALEX-docs/master/dalex/dalex-tree.png){ width=70% }

-------------------------------------------


