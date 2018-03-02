#' Prints Explainer Summary
#'
#' @param x a model expaliner created with the `explain` function
#' @param ... other parameters
#'
#' @export
#' @import ggplot2
#'
#' @examples
#' library("breakDown")
#'
#' wine_lm_model4 <- lm(quality ~ pH + residual.sugar + sulphates + alcohol, data = wine)
#' wine_lm_explainer4 <- explain(wine_lm_model4, data = wine, label = "model_4v")
#' wine_lm_explainer4
#'
#' #\dontrun{
#' library("randomForest")
#' wine_rf_model4 <- randomForest(quality ~ pH + residual.sugar + sulphates + alcohol, data = wine)
#' wine_rf_explainer4 <- explain(wine_rf_model4, data = wine, label = "model_rf")
#' wine_rf_explainer4
#' #}
#'
print.explainer <- function(x, ...) {
  cat("Model label: ", x$label, "\n")
  cat("Model class: ", paste(x$class, collapse = ","), "\n")
  cat("Data head  :\n")
  print(head(x$data,2))
  return(invisible(NULL))
}

