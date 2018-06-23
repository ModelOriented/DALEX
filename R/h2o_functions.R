#' @importFrom h2o h2o.cbind
model_performance_h2o <- function(explainer, ...) {
  observed <- explainer$y
  predicted <- explainer$predict_function(explainer$model, explainer$data, ...)
  residuals <- h2o.cbind(predicted, observed, predicted - observed)

  residuals$label <- explainer$label
  residuals <- as.data.frame(residuals)
  colnames(residuals) <- c("predicted", "observed","diff", "label")
  class(residuals) <- c("model_performance_explainer", "data.frame")

  residuals
}


#' @importFrom h2o as.h2o h2o.rbind h2o.nrow
variable_importance_h2o <- function(explainer,
                                loss_function = function(observed, predicted) sum((observed - predicted)^2),
                                ...,
                                type = "raw",
                                n_sample = 1000) {
  variables <- colnames(explainer$data)
  if (n_sample > 0) {
    sampled_rows <- sample.int(nrow(explainer$data), n_sample, replace = TRUE)
    sampled_rows <- sort(sampled_rows)
  } else {
    sampled_rows <- 1:nrow(explainer$data)
  }
  sampled_data <- as.data.frame(explainer$data)[sampled_rows,]
  observed <- as.data.frame(explainer$y)[sampled_rows,]
  sampled_observed <- as.h2o(sample(observed))
  sampled_data <- as.h2o(sampled_data)
  observed <- as.h2o(observed)

  loss_0 <- loss_function(observed,
                          explainer$predict_function(explainer$model, sampled_data))
  loss_full <- loss_function(sampled_observed,
                             explainer$predict_function(explainer$model, sampled_data))
  res <- sapply(variables, function(variable) {
    ndf <- sampled_data
    ndf[,variable] <- h2o.rbind(h2o.splitFrame(ndf[,variable]))
    predicted <- explainer$predict_function(explainer$model, ndf)
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

  class(res) <- c("variable_importance_explainer", "data.frame")
  res$label <- explainer$label
  res
}

#' @importFrom h2o h2o.cbind as.h2o h2o.splitFrame h2o.levels
variable_response_h2o <- function(explainer, variable, type = "pdp", trans = explainer$link, ...) {
  switch(type,
         factor = {
           lev <- h2o.levels(explainer$data[,variable])

           preds <- lapply(lev, function(cur_lev) {
             tmp <- explainer$data
             tmp[,variable] <- cur_lev
             levels <- as.h2o(rep(cur_lev, h2o.nrow(tmp)))
             res <- h2o.cbind(explainer$predict_function(explainer$model, tmp), levels)
             names(res) <- c("scores", "level")
             return(res)
           })
           preds_combined <- do.call(h2o.rbind, preds)
           preds_combined <- as.data.frame(preds_combined)

           res <- mergeFactors(preds_combined$scores, preds_combined$level, abbreviate = FALSE)
           res$label = explainer$label
           class(res) <-  c("variable_response_explainer", "factorMerger", "gaussianFactorMerger")
           res
         },
         pdp = {
           # pdp requires predict function with only two arguments
           predictor_pdp <- function(object, newdata) mean(explainer$predict_function(object, as.h2o(newdata)), na.rm = TRUE)
           traindata <- as.data.frame(explainer$data)
           part <- partial(explainer$model, pred.var = variable, train = traindata, ..., pred.fun = predictor_pdp, recursive = FALSE)
           res <- data.frame(x = part[,1], y = trans(part$yhat), var = variable, type = type, label = explainer$label)
           class(res) <- c("variable_response_explainer", "data.frame", "pdp")
           res
         },
         ale = {
           # pdp requires predict function with only two arguments
           predictor_ale <- function(X.model, newdata){
             pred <- explainer$predict_function(X.model, as.h2o(newdata))
             return(as.vector(pred))
           }

           # need to create a temporary file to stop ALEPlot function from plotting anytihing
           tmpfn <- tempfile()
           pdf(tmpfn)
           Xdata <- as.data.frame(explainer$data)
           part <- ALEPlot(X = Xdata, X.model = explainer$model, J = variable, pred.fun = predictor_ale)
           dev.off()
           unlink(tmpfn)
           res <- data.frame(x = part$x.values, y = trans(part$f.values), var = variable, type = type, label = explainer$label)
           class(res) <- c("variable_response_explainer", "data.frame", "ale")
           res
         },
         stop("Currently only 'pdp', 'ale' and 'factor' methods are implemented"))
}


prediction_breakdown_h2o <- function(explainer, observation, ...) {
  predictor_broken <- function(model, newdata){
    pred <- explainer$predict_function(model, as.h2o(newdata))
    return(as.vector(pred))
  }

  # breakDown
  res <- broken(explainer$model,
                new_observation = as.data.frame(observation),
                data = as.data.frame(explainer$data),
                predict.function = predictor_broken,
                baseline = "Intercept", ...)
  res$label <- rep(explainer$label, length(res$variable))

  class(res) <- c("prediction_breakdown_explainer", "data.frame")
  res

}
