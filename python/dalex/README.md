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

The philosophy behind `dalex` explanations is described in the [Explanatory Model Analysis](https://pbiecek.github.io/ema/) e-book.

## Installation

The `dalex` package is available [on PyPI](https://pypi.org/project/dalex/)

```console
pip install dalex -U
```

### Resources: https://dalex.drwhy.ai/python

![](https://raw.githubusercontent.com/ModelOriented/DALEX-docs/master/dalex/dalex-diagram.png)

### API reference: https://dalex.drwhy.ai/python/api

![](https://raw.githubusercontent.com/ModelOriented/DALEX-docs/master/dalex/dalex-class.png)

## Authors

Main authors of the `dalex` package are:

* [Hubert Baniecki](https://github.com/hbaniecki)
* [Wojciech Kretowicz](https://github.com/wojciechkretowicz)

Under the supervision of [Przemyslaw Biecek](https://github.com/pbiecek).

Other contributors:

* [Piotr Piatyszek](https://github.com/piotrpiatyszek) maintains the `arena` module
* [Jakub Wisnewski](https://github.com/jakwisn) maintains the `fairness` module

## Citation

If you use `dalex`, please cite our paper:

```
@article{dalex,
  title={dalex: Responsible Machine Learning with Interactive Explainability and Fairness in Python},
  author={Hubert Baniecki and Wojciech Kretowicz and Piotr Piatyszek and Jakub Wisniewski and Przemyslaw Biecek},
  year={2020},
  eprint={2012.14406},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2012.14406}
}
```

-------------------------------------------
