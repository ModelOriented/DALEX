# dalex

[moDel Agnostic Language for Exploration and eXplanation](http://dalex.drwhy.ai/)

[![Python-check](https://github.com/ModelOriented/DALEX/workflows/Python-check/badge.svg?branch=master)
[![PyPI version](https://badge.fury.io/py/dalex.svg)](https://badge.fury.io/py/dalex)
[![Downloads](https://pepy.tech/badge/dalex)](https://pepy.tech/project/dalex)

## Overview

Unverified black box model is the path to the failure. Opaqueness leads to distrust. Distrust leads to ignoration. Ignoration leads to rejection.

The `dalex` package xrays any model and helps to explore and explain its behaviour, helps to understand how complex models are working.
The main `Explainer` object creates a wrapper around a predictive model. Wrapped models may then be explored and compared with a collection of local and global explainers.
Recent developents from the area of Interpretable Machine Learning/eXplainable Artificial Intelligence.

The philosophy behind `dalex` explanations is described in the [Explanatory Model Analysis](https://pbiecek.github.io/ema/) e-book.

<p align="center">
    <a href="https://pbiecek.github.io/ema/introduction.html#bookstructure">
        <img src="https://github.com/ModelOriented/DALEX/blob/master/misc/DALEXpiramide.png" width="800"/>
    </a>
</center>

The `dalex` package is a part of [DrWhy.AI](http://DrWhy.AI) universe.

## Installation

```console
pip install dalex
```

## Resources

* Introduction to the `dalex` package: [Titanic: tutorial and examples](http://dalex.drwhy.ai/python-dalex-titanic.html)
* Important features explained: [FIFA20: explain default vs tuned model with dalex](http://dalex.drwhy.ai/python-dalex-fifa.html)
* Code in the form of [jupyter notebook](https://github.com/pbiecek/DALEX_docs/blob/master/jupyter-notebooks/dalex-titanic.ipynb) - more to come
* [YouTube video](https://www.youtube.com/watch?v=PuKF2GS4_3Y) showing how to do [Break Down](https://pbiecek.github.io/ema/breakDown.html) analysis
* Theoretical introduction to the plots: [Explanatory Model Analysis. Explore, Explain and Examine Predictive Models.](https://pbiecek.github.io/ema)
* Changelog: [NEWS](https://github.com/ModelOriented/DALEX/blob/master/python/dalex/NEWS.md)

## Plots

This package uses [plotly](https://plotly.com/python/) to render the plots:

* Install extentions to use `plotly` in **JupyterLab**:&emsp;[Getting Started](https://plot.ly/python/getting-started/#jupyterlab-support-python-35)&emsp;[Troubleshooting](https://plot.ly/python/troubleshooting/#jupyterlab-problems)
* Use `show=False` parameter in `plot` method to return `plotly Figure` object
* It is possible to [edit the figures](https://plotly.com/python/#fundamentals) and [save them](https://plotly.com/python/static-image-export/)

## Learn more

Machine Learning models are widely used and have various applications in classification or regression tasks. Due to increasing computational power, availability of new data sources and new methods, ML models are more and more complex. Models created with techniques like boosting, bagging of neural networks are true black boxes. It is hard to trace the link between input variables and model outcomes. They are use because of high performance, but lack of interpretability is one of their weakest sides.

In many applications we need to know, understand or prove how input variables are used in the model and what impact do they have on final model prediction.
`DALEX` is a set of tools that help to understand how complex models are working.

### Talks about DALEX

* [Talk about DALEX at Complexity Institute / NTU February 2018](https://github.com/pbiecek/pbiecek.github.io/blob/master/Presentations/DALEX_at_NTU_2018.pdf)
* [Talk about DALEX at SER / WTU April 2018](https://github.com/pbiecek/Talks/blob/master/2018/SER_DALEX.pdf)
* [Talk about DALEX at STWUR May 2018 (in Polish)](https://github.com/STWUR/eRementarz-29-05-2018)
* [Talk about DALEX at BayArea 2018](https://github.com/pbiecek/Talks/blob/master/2018/DALEX_BayArea.pdf)
* [Talk about DALEX at PyData Warsaw 2018](https://github.com/pbiecek/Talks/blob/master/2018/DALEX_PyDataWarsaw2018.pdf)


## Why

76 years ago Isaac Asimov devised [Three Laws of Robotics](https://en.wikipedia.org/wiki/Three_Laws_of_Robotics): 1) a robot may not injure a human being, 2) a robot must obey the orders given it by human beings and 3) A robot must protect its own existence. These laws impact discussion around [Ethics of AI](https://en.wikipedia.org/wiki/Ethics_of_artificial_intelligence). Today’s robots, like cleaning robots, robotic pets or autonomous cars are far from being conscious enough to be under Asimov’s ethics.

Today we are surrounded by complex predictive algorithms used for decision making. Machine learning models are used in health care, politics, education, judiciary and many other areas. Black box predictive models have far larger influence on our lives than physical robots. Yet, applications of such models are left unregulated despite many examples of their potential harmfulness. See *Weapons of Math Destruction* by Cathy O'Neil for an excellent overview of potential problems.

It's clear that we need to control algorithms that may affect us. Such control is in our civic rights. Here we propose three requirements that any predictive model should fulfill.

-	**Prediction's justifications**. For every prediction of a model one should be able to understand which variables affect the prediction and how strongly. Variable attribution to final prediction.
-	**Prediction's speculations**. For every prediction of a model one should be able to understand how the model prediction would change if input variables were changed. Hypothesizing about what-if scenarios.
-	**Prediction's validations** For every prediction of a model one should be able to verify how strong are evidences that confirm this particular prediction.

There are two ways to comply with these requirements.
One is to use only models that fulfill these conditions by design. White-box models like linear regression or decision trees. In many cases the price for transparency is lower performance.
The other way is to use approximated explainers – techniques that find only approximated answers, but work for any black box model. Here we present such techniques.
