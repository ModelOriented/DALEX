# dalex

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

The philosophy behind `dalex` explanations is described in the [Explanatory Model Analysis](https://pbiecek.github.io/ema/) book.

## Installation

The `dalex` package is available on [PyPI](https://pypi.org/project/dalex/) and [conda-forge](https://anaconda.org/conda-forge/dalex).

```console
pip install dalex -U

conda install -c conda-forge dalex
```

One can install optional dependencies for all additional features using `pip install dalex[full]`.

### Resources: https://dalex.drwhy.ai/python

### API reference: https://dalex.drwhy.ai/python/api

[![http://python.drwhy.ai/](https://raw.githubusercontent.com/ModelOriented/DALEX-docs/master/dalex/dalex-diagram.png)](http://python.drwhy.ai/)

## Authors

The authors of the `dalex` package are:

* [Hubert Baniecki](https://github.com/hbaniecki)
* [Wojciech Kretowicz](https://github.com/wojciechkretowicz)
* [Piotr Piatyszek](https://github.com/piotrpiatyszek) maintains the `arena` module
* [Jakub Wisniewski](https://github.com/jakwisn) maintains the `fairness` module
* [Mateusz Krzyzinski](https://github.com/krzyzinskim) maintains the `aspect` module
* [Artur Zolkowski](https://github.com/arturzolkowski) maintains the `aspect` module
* [Przemyslaw Biecek](https://github.com/pbiecek)

We welcome contributions: [start by opening an issue on GitHub](https://github.com/ModelOriented/DALEX/issues/new).

## Citation

If you use `dalex`, please cite our [JMLR paper](https://jmlr.org/papers/v22/20-1473.html):

```html
@article{JMLR:v22:20-1473,
  author  = {Hubert Baniecki and
             Wojciech Kretowicz and
             Piotr Piatyszek and 
             Jakub Wisniewski and 
             Przemyslaw Biecek},
  title   = {dalex: Responsible Machine Learning 
             with Interactive Explainability and Fairness in Python},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {214},
  pages   = {1-7},
  url     = {http://jmlr.org/papers/v22/20-1473.html}
}
```

-------------------------------------------
