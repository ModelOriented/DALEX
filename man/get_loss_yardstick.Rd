% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/misc_loss_functions.R
\name{get_loss_yardstick}
\alias{get_loss_yardstick}
\alias{loss_yardstick}
\title{Wrapper for Loss Functions from the yardstick Package}
\usage{
get_loss_yardstick(loss, reverse = FALSE, reference = 1)

loss_yardstick(loss, reverse = FALSE, reference = 1)
}
\arguments{
\item{loss}{loss function from the \code{yardstick} package}

\item{reverse}{shall the metric be reversed? for loss metrics lower values are better. \code{reverse = TRUE} is useful for accuracy-like metrics}

\item{reference}{if the metric is reverse then it is calculated as \code{reference - loss}. The default value is 1.}
}
\value{
loss function that can be used in the model_parts function
}
\description{
The yardstick package provides many auxiliary functions for calculating
the predictive performance of the model. However, they have an interface
that is consistent with the tidyverse philosophy. The loss_yardstick
function adapts loss functions from the yardstick package to functions
understood by DALEX. Type compatibility for y-values and for predictions
must be guaranteed by the user.
}
\examples{
 \donttest{
 titanic_glm_model <- glm(survived~., data = titanic_imputed, family = "binomial")
 explainer_glm <- DALEX::explain(titanic_glm_model,
                                 data = titanic_imputed[,-8],
                                 y = factor(titanic_imputed$survived))
 # See the 'How to use DALEX with the yardstick package' vignette
 # which explains this model with measures implemented in the 'yardstick' package
}

}
