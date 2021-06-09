#' Plot List of Explanations
#'
#' @param x a list of explanations of the same class
#' @param ... other parameters
#'
#' @return An object of the class \code{ggplot}.
#'
#' @export
#' @examples
#'  \donttest{
#' library("ranger")
#' titanic_ranger_model <- ranger(survived~., data = titanic_imputed, num.trees = 50,
#'                                probability = TRUE)
#' explainer_ranger  <- explain(titanic_ranger_model, data = titanic_imputed[,-8],
#'                              y = titanic_imputed$survived)
#' mp_ranger <- model_performance(explainer_ranger)
#'
#' titanic_ranger_model2 <- ranger(survived~gender + fare, data = titanic_imputed,
#'                                 num.trees = 50, probability = TRUE)
#' explainer_ranger2  <- explain(titanic_ranger_model2, data = titanic_imputed[,-8],
#'                               y = titanic_imputed$survived,
#'                               label = "ranger2")
#' mp_ranger2 <- model_performance(explainer_ranger2)
#'
#' plot(list(mp_ranger, mp_ranger2), geom = "prc")
#' plot(list(mp_ranger, mp_ranger2), geom = "roc")
#' tmp <- list(mp_ranger, mp_ranger2)
#' names(tmp) <- c("ranger", "ranger2")
#' plot(tmp)
#' }
#'
#
plot.list <- function(x, ...) {
  names(x) <- NULL
  args <- c(x, list(...))
  do.call(plot, args)
}


