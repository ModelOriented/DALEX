[![CRAN_Status_Badge](http://www.r-pkg.org/badges/version/DALEX)](https://cran.r-project.org/package=DALEX)
[![Downloads](http://cranlogs.r-pkg.org/badges/DALEX)](http://cran.rstudio.com/package=DALEX)
[![Total Downloads](http://cranlogs.r-pkg.org/badges/grand-total/DALEX?color=orange)](http://cranlogs.r-pkg.org/badges/grand-total/DALEX)
[![Build Status](https://api.travis-ci.org/pbiecek/DALEX.png)](https://travis-ci.org/pbiecek/DALEX)
[![Github Issues](http://githubbadges.herokuapp.com/pbiecek/DALEX/issues.svg)](https://github.com/pbiecek/DALEX/issues)

# DALEX

Descriptive mAchine Learning EXplanations

[A gentle introduction to DALEX with examples](https://pbiecek.github.io/DALEX_docs/)

## Install

From GitHub

```{r}
# dependencies
devtools::install_github("MI2DataLab/factorMerger")
devtools::install_github("pbiecek/breakDown")

# DALEX package
devtools::install_github("pbiecek/DALEX")
```

or from CRAN

```{r}
install.packages("DALEX")
```

## Intro

Machine Learning models are widely used and have various applications in classification or regression tasks. Due to increasing computational power, availability of new data sources and new methods, ML models are more and more complex. Models created with techniques like boosting, bagging of neural networks are true black boxes. It is hard to trace the link between input variables and model outcomes. They are use because of high performance, but lack of interpretability is one of their weakest sides.

In many applications we need to know, understand or prove how input variables are used in the model and what impact do they have on final model prediction. DALEX is a set of tools that help to understand how complex models are working.

![DALEX_intro](misc/DALEX_intro.png)


## Single Variable

![DALEX_single_variable](misc/DALEX_single_variable.png)

## Single Prediction

![DALEX_single_prediction](misc/DALEX_single_prediction.png)

## Variable Drop-out / Importance

![DALEX_variable_dropout](misc/DALEX_variable_dropout.png)


## DALEX Stories

* [DALEX at Complexity Institute / NTU February 2018](https://github.com/pbiecek/pbiecek.github.io/blob/master/Presentations/DALEX_at_NTU_2018.pdf)
* [DALEX at SER / WTU April 2018](https://github.com/pbiecek/Talks/blob/master/2018/SER_DALEX.pdf)

## Acknowledgments

Work on this package was financially supported by the 'NCN Opus grant 2016/21/B/ST6/02176'.
    
