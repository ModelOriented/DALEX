#' Plot Variable Profile Explanations
#'
#' @param x an object of the class \code{predict_profile}
#' @param ... other parameters
#'
#' @return An object of the class \code{ggplot}.
#'
#' @section Plot options:
#'
#' \subsection{ceteris_paribus}{
#' \itemize{
#'  \item{color}{ a character. Either name of a color or name of a variable that should be used for coloring}
#'  \item{size}{ a numeric. Size of lines to be plotted}
#'  \item{alpha}{ a numeric between \code{0} and \code{1}. Opacity of lines}
#'  \item{facet_ncol}{ number of columns for the \code{\link[ggplot2]{facet_wrap}}}
#'  \item{variables}{ if not \code{NULL} then only \code{variables} will be presented}
#'  \item{variable_type}{ a character. If \code{numerical} then only numerical variables will be plotted.
#'        If \code{categorical} then only categorical variables will be plotted.}
#'  \item{title}{ a character. Plot title. By default \code{"Ceteris Paribus profile"}.}
#'  \item{subtitle}{ a character. Plot subtitle. By default \code{NULL} - then subtitle is set to "created for the XXX, YYY model",
#'        where XXX, YYY are labels of given explainers.}
#'  \item{categorical_type}{ a character. How categorical variables shall be plotted? Either \code{"lines"} (default) or \code{"bars"}.}
#' }
#' }
#'
#' @export
plot.predict_profile <- function(x, ...) {
  class(x) <- class(x)[-1]
  plot(x, ...) +
    ingredients::show_observations(x, size = 3, ...)
}
