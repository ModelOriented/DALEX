#' Calculate Model Performance
#'
#' Prepare a data frame with model residuals.
#'
#' @param explainer a model to be explained, preprocessed by the \code{\link{explain}} function
#' @param ... other parameters
#'
#' @return An object of the class \code{model_performance_explainer}.
#' @references Predictive Models: Visual Exploration, Explanation and Debugging \url{https://pbiecek.github.io/PM_VEE/}
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
#'
#' titanic_glm_model <- glm(survived~., data = titanic_imputed, family = "binomial")
#' explainer_glm <- explain(titanic_glm_model, data = titanic_imputed[,-8],
#'                          y = titanic_imputed$survived,
#'                     predict_function = function(m,x) predict.glm(m,x,type = "response"),
#'                          label = "glm")
#' mp_ex_glm <- model_performance(explainer_glm)
#' mp_ex_glm
#' plot(mp_ex_glm)
#' plot(mp_ex_glm, mp_ex_rn)
#'
#' titanic_lm_model <- lm(survived~., data = titanic_imputed)
#' explainer_lm <- explain(titanic_lm_model, data = titanic_imputed[,-8], y = titanic_imputed$survived)
#' mp_ex_lm <- model_performance(explainer_lm)
#' plot(mp_ex_lm)
#' plot(mp_ex_glm, mp_ex_rn, mp_ex_lm)
#'  }
#'
model_performance <- function(explainer, ...) {
  if (!("explainer" %in% class(explainer))) stop("The model_performance() function requires an object created with explain() function.")
  if (is.null(explainer$data)) stop("The model_performance() function requires explainers created with specified 'data' parameter.")
  if (is.null(explainer$y)) stop("The model_performance() function requires explainers created with specified 'y' parameter.")
  # Check since explain could have been run with precalculate = FALSE
  if (is.null(explainer$y_hat)){
    predicted <- explainer$predict_function(explainer$model, explainer$data, ...)
  } else {
    predicted <- explainer$y_hat
  }
  observed <- explainer$y
  # Check since explain could have been run with precalculate = FALSE
  if (is.null(explainer$residuals)){
    diff <- observed - predicted
  } else {
    # changed according to #130
    diff <- explainer$residuals
  }

  residuals <- data.frame(predicted, observed, diff = diff)

  class(residuals) <- c("model_performance_explainer", "data.frame")
  residuals$label <- explainer$label
  residuals
}


#' Print Model Performance Summary
#'
#' @param x a model to be explained, object of the class 'model_performance_explainer'
#' @param ... other parameters
#'
#' @importFrom stats quantile
#' @export
#' @examples
#'  \dontrun{
#' library("randomForest")
#' HR_rf_model <- randomForest(status == "fired"~., data = HR, ntree = 100)
#' explainer_rf  <- explain(HR_rf_model, data = HR, y = HR$status == "fired")
#' mp_ex_rf <- model_performance(explainer_rf)
#' mp_ex_rf
#' plot(mp_ex_rf)
#'  }
#'
print.model_performance_explainer <- function(x, ...) {
  print(quantile(x$diff, seq(0, 1, 0.1)))
}


