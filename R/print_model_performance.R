#' Print Dataset Level Model Performance Summary
#'
#' @param x a model to be explained, object of the class 'model_performance_explainer'
#' @param ... other parameters
#'
#' @importFrom stats quantile
#' @export
#' @examples
#'  \dontrun{
#' library("ranger")
#' titanic_ranger_model <- ranger(survived~., data = titanic_imputed, num.trees = 100,
#'                                probability = TRUE)
#' # It's a good practice to pass data without target variable
#' explainer_ranger  <- explain(titanic_ranger_model, data = titanic_imputed[,-8],
#'                              y = titanic_imputed$survived)
#' # resulting dataframe has predicted values and residuals
#' mp_ex_rn <- model_performance(explainer_ranger)
#' mp_ex_rn
#' plot(mp_ex_rn)
#'  }
#'
print.model_performance <- function(x, ...) {
  cat("Measures for: ", x$type)
  cat(paste0("\n", substr(paste0(names(x$measures), "         "), 1, 11), ": ", sapply(x$measures, prettyNum)))
  cat("\n\nResiduals:\n")
  print(quantile(x$residuals$diff, seq(0, 1, 0.1)))
}


