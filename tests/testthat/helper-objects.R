# stops waring messages
assign("message_variable_importance", value = TRUE, envir = DALEX:::.DALEX.env)
assign("message_prediction_breakdown", value = TRUE, envir = DALEX:::.DALEX.env)
assign("message_partial_dependency", value = TRUE, envir = DALEX:::.DALEX.env)
assign("message_accumulated_dependency", value = TRUE, envir = DALEX:::.DALEX.env)
assign("message_variable_effect", value = TRUE, envir = DALEX:::.DALEX.env)


library("DALEX")
library("ranger")

# models
model_classif_glm <- glm(status == "fired"~., data = HR, family = "binomial")
model_classif_ranger <- ranger::ranger(survived~., data = titanic_imputed, num.trees = 50, probability = TRUE)
model_regr_ranger <- ranger::ranger(m2.price~., data = apartments, num.trees = 50)
model_regr_lm <- lm(m2.price~., data = apartments)
model_multiclassif_ranger <- ranger::ranger(status~., data = HR, num.trees = 50)
model_multiclassif_ranger_prob <- ranger::ranger(status~., data = HR, num.trees = 50, probability = TRUE)

# explain()
p_fun_ranger <- function(model, x) predict(model, x)$predictions
p_fun_glm <- function(model, x) predict(model, x, type = "response")

# predict.*** functions are masking the external model specific predict function for tests purpose.

predict.randomForest <- function(X.model, newdata, type) {
  if (X.model$type == "classification") {
    pred <- data.frame(rep(0.5, times = nrow(newdata)), rep(0.5, times = nrow(newdata)))
  } else {
    pred <- rep(1, times = nrow(newdata))
  }
  pred
}

predict.svm <- function(X.model, newdata, ...) {
  if (X.model$type == 0) {
    pred <- list(1)
    attr(pred, "probabilities") <- data.frame(rep(0.5, times = nrow(newdata)), rep(0.5, times = nrow(newdata)))
  } else {
    pred <- rep(1, times = nrow(newdata))
  }
  pred
}

predict.gbm <- function(X.model, newdata, ...) {
  rep(0.14, times = nrow(newdata))
}

predict.glmnet <- function(X.model, newdata, ...) {
  if (!is.null(X.model$classnames)) {
    if (length(X.model$classnames) == 2) {
      response <- matrix(rep(0.5, times = nrow(newdata)), ncol = 1)
    } else {
      response <- matrix(rep(0.5, times = length(X.model$classnames)*nrow(newdata)), ncol = length(X.model$classnames))
    }
  } else {
    response <- rep(0.14, times = nrow(newdata))
  }
  response
}

predict.cv.glmnet <- function(X.model, newdata, ...) {
  if (!is.null(X.model$glmnet.fit$classnames)) {
    response <- matrix(rep(0.5, times = length(X.model$glmnet.fit$classnames)*nrow(newdata)), ncol = length(X.model$glmnet.fit$classnames))
  } else {
    response <- rep(0.14, times = nrow(newdata))
  }
  response
}

predict.model_fit <- function(X.model, newdata, ...) {
  if (X.model$spec$mode == "classification") {
    response <- data.frame(rep(0.5, times = nrow(newdata)), rep(0.5, times = nrow(newdata)))
  }
  if (X.model$spec$mode == "regression") {
    response <- list(a = 1)
    response$.pred <- rep(0.5, times = nrow(newdata))
  }
  response
}

predict.train <- function(X.model, newdata, type, ...) {
  if (X.model$modelType == "Classification") {
    response <- data.frame(rep(0.5, times = nrow(newdata)), rep(0.5, times = nrow(newdata)))
  }
  if (X.model$modelType == "Regression") {
    response <- rep(0.5, times = nrow(newdata))

  }
  response
}

predict.rpart <- function(X.model, newdata, ...) {
  if (attr(X.model$terms, "dataClasses")[1] == "factor") {
    response <-   data.frame(rep(0.5, times = nrow(newdata)), rep(0.5, times = nrow(newdata)))
  } else {
    response <- rep(0.5, times = nrow(newdata))
  }
  response
}


explainer_classif_ranger  <- explain(model_classif_ranger, data = titanic_imputed, y = titanic_imputed$survived, verbose = FALSE)
explainer_classif_glm  <- explain(model_classif_glm, data = HR, predict_function = p_fun_glm, verbose = FALSE)
explainer_regr_ranger <- explain(model_regr_ranger, data = apartments_test[1:1000, ], y = apartments_test$m2.price[1:1000], verbose = FALSE)
explainer_regr_ranger_wo_y <- explain(model_regr_ranger, data = apartments_test[1:1000, ], verbose = FALSE)
explainer_regr_lm <- explain(model_regr_lm, data = apartments_test[1:1000, ], y = apartments_test$m2.price[1:1000], verbose = FALSE)
explainer_wo_data  <- explain(model_classif_ranger, data = NULL, verbose = FALSE)


