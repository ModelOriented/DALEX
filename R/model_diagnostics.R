#' Dataset Level Model Diagnostics
#'
#' This function performs model diagnostic of residuals.
#' Residuals are calculated and ploted against predictions, true y values or selected variables.
#' Find information how to use this function here: \url{https://pbiecek.github.io/ema/residualDiagnostic.html}.
#'
#' @param explainer a model to be explained, preprocessed by the \code{explain} function
#' @param variables character - name of variables to be explained. Default \code{NULL} stands for all variables
#' @param ... other parameters
#'
#' @return An object of the class \code{model_diagnostics}.
#' It's a data frame with residuals and selected variables.
#'
#' @references Explanatory Model Analysis. Explore, Explain and Examine Predictive Models. \url{https://pbiecek.github.io/ema/}
#' @export
#' @examples
#' apartments_lm_model <- lm(m2.price ~ ., data = apartments)
#' explainer_lm <- explain(apartments_lm_model,
#'                          data = apartments,
#'                          y = apartments$m2.price)
#' diag_lm <- model_diagnostics(explainer_lm)
#' diag_lm
#' plot(diag_lm)
#' \dontrun{
#' library("ranger")
#' apartments_ranger_model <- ranger(m2.price ~ ., data = apartments)
#' explainer_ranger <- explain(apartments_ranger_model,
#'                          data = apartments,
#'                          y = apartments$m2.price)
#' diag_ranger <- model_diagnostics(explainer_ranger)
#' diag_ranger
#' plot(diag_ranger)
#' plot(diag_ranger, diag_lm)
#' plot(diag_ranger, diag_lm, variable = "y")
#' plot(diag_ranger, diag_lm, variable = "construction.year")
#' plot(diag_ranger, variable = "y", yvariable = "y_hat")
#' plot(diag_ranger, variable = "y", yvariable = "abs_residuals")
#' plot(diag_ranger, variable = "ids")
#'}
#' @name model_diagnostics
#' @export
model_diagnostics <-  function(explainer, variables = NULL, ...) {
  test_expaliner(explainer, has_data = TRUE, function_name = "model_diagnostics")

  # if variables = NULL then all variables are added
  # otherwise only selected
  if (is.null(variables)) {
    results <- explainer$data
  } else {
    results <- explainer$data[, intersect(variables, colnames(explainer$data)), drop = FALSE]
  }
  # is there target
  if (!is.null(explainer$y)) {
    results$y <- explainer$y
  }
  # are there predictions
  if (is.null(explainer$y_hat)) {
    explainer$y_hat <- explainer$predict_function(explainer$model, explainer$data)
  }
  if (is.null(dim(explainer$y_hat))) {
    results$y_hat <- explainer$y_hat
  } else {
    results$y_hat <- explainer$y_hat[, 1] # this will work only for first column
  }

  # are there residuals
  if (is.null(explainer$residuals)) {
    explainer$residuals <- explainer$residual_function(explainer$model, explainer$data)
  }
  if (is.null(dim(explainer$residuals))) {
    results$residuals <- explainer$residuals
  } else {
    results$residuals <- explainer$residuals[, 1] # this will work only for first column
  }
  results$abs_residuals <- abs(results$residuals)
  results$label <- explainer$label
  results$ids <- seq_along(results$label)

  class(results) <- c("model_diagnostics", "data.frame")
  results
}
