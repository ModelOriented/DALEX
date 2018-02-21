#' Loss from Variable Dropout
#'
#' @param explainer a model to be explained, preprocessed by the 'explain' function
#' @param loss_function a function thet will be used to assess variable importance
#' @param ... other parameters
#' @param type character, type of transformation that should be applied for dropout loss. 'raw' results raw drop lossess, 'ratio' returns \code{drop_loss/drop_loss_full_model} while 'difference' returns \code{drop_loss - drop_loss_full_model}
#' @param n_sample number of observations that should be sampled for calculations of variable leverages
#'
#' @return An object of the class 'variable_leverage_explainer'.
#' It's a data frame with calculated average response.
#'
#' @export
#' @examples
#'
variable_dropout <- function(explainer,
                              loss_function = function(observed, predicted) sum((observed - predicted)^2),
                              ...,
                              type = "raw",
                              n_sample = 1000) {
  stopifnot(class(explainer) == "explainer")

  variables <- colnames(explainer$data)
  sampled_rows <- sample.int(nrow(explainer$data), n_sample, replace = TRUE)
  sampled_data <- explainer$data[sampled_rows,]

  observed <- explainer$y[sampled_rows]

  loss_0 <- loss_function(observed,
                          explainer$predict.function(explainer$model, sampled_data))
  loss_full <- loss_function(sample(observed),
                          explainer$predict.function(explainer$model, sampled_data))
  res <- sapply(variables, function(variable) {
    ndf <- sampled_data
    ndf[,variable] <- sample(ndf[,variable])
    predicted <- explainer$predict.function(explainer$model, ndf)
    loss_function(observed, predicted)
  })
  res <- sort(res)
  res <- data.frame(variable = c("_full_model_",names(res), "_baseline_"),
                    dropout_loss = c(loss_0, res, loss_full))
  if (type == "ratio") {
    res$dropout_loss = res$dropout_loss / loss_0
  }
  if (type == "difference") {
    res$dropout_loss = res$dropout_loss - loss_0
  }

  class(res) <- c("variable_dropout_explainer", "data.frame")
  res$label <- explainer$label
  res
}


