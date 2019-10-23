# stops waring messages
assign("message_variable_importance", value = TRUE, envir = DALEX:::.DALEX.env)
assign("message_prediction_breakdown", value = TRUE, envir = DALEX:::.DALEX.env)
assign("message_partial_dependency", value = TRUE, envir = DALEX:::.DALEX.env)
assign("message_accumulated_dependency", value = TRUE, envir = DALEX:::.DALEX.env)

library(randomForest)

# models
model_classif_glm <- glm(status == "fired"~., data = HR, family = "binomial")
model_classif_rf <- randomForest::randomForest(status == "fired"~., data = HR, ntree = 50)
model_regr_rf <- randomForest::randomForest(m2.price~., data = apartments, ntree = 50)
model_regr_lm <- lm(m2.price~., data = apartments)
library("gbm")
gbm1 <- gbm(m2.price ~ ., data = apartments, distribution = "gaussian")
gbm2 <- gbm(survived~., data = titanic_imputed, distribution = "bernoulli")

# explain()
p_fun_rf <- function(model, x) predict(model, x)
p_fun_glm <- function(model, x) predict(model, x, type = "response")
explainer_classif_rf  <- explain(model_classif_rf, data = HR, predict_function = p_fun_rf)
explainer_classif_glm  <- explain(model_classif_glm, data = HR, predict_function = p_fun_glm)
explainer_regr_rf <- explain(model_regr_rf, data = apartments_test[1:1000, ], y = apartments_test$m2.price[1:1000])
explainer_regr_rf_wo_y <- explain(model_regr_rf, data = apartments_test[1:1000, ])
explainer_regr_lm <- explain(model_regr_lm, data = apartments_test[1:1000, ], y = apartments_test$m2.price[1:1000])
explainer_wo_data  <- explain(model_classif_rf, data = NULL)



