#' Plot Variable Importance Explanations
#'
#' @param x an object of the class \code{model_parts}
#' @param ... other parameters described below
#'
#' @return An object of the class \code{ggplot}.
#'
#' @section Plot options:
#'
#' \subsection{variable_importance}{
#' \itemize{
#'  \item{max_vars}{maximal number of features to be included in the plot. default value is \code{10}}
#'  \item{show_boxplots}{logical if \code{TRUE} (default) boxplot will be plotted to show permutation data.}
#'  \item{bar_width}{width of bars. By default \code{10}}
#'  \item{desc_sorting}{logical. Should the bars be sorted descending? By default \code{TRUE}}
#'  \item{title}{the plot's title, by default \code{'Feature Importance'}}
#'  \item{subtitle}{a character. Plot subtitle. By default \code{NULL} - then subtitle is set to "created for the XXX, YYY model",
#'        where XXX, YYY are labels of given explainers.}
#' }
#' }
#'
#' @export
plot.model_parts <- function(x, ...) {
  class(x)[1] <- NULL
  plot(x)
}
