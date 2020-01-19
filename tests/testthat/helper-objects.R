# stops waring messages
assign("message_variable_importance", value = TRUE, envir = DALEX:::.DALEX.env)
assign("message_prediction_breakdown", value = TRUE, envir = DALEX:::.DALEX.env)
assign("message_partial_dependency", value = TRUE, envir = DALEX:::.DALEX.env)
assign("message_accumulated_dependency", value = TRUE, envir = DALEX:::.DALEX.env)

library("ranger")
library("DALEX")

# models
model_classif_glm <- glm(status == "fired"~., data = HR, family = "binomial")
model_classif_ranger <- ranger(survived~., data = titanic_imputed, num.trees = 50, probability = TRUE)
model_regr_ranger <- ranger(m2.price~., data = apartments, num.trees = 50)
model_regr_lm <- lm(m2.price~., data = apartments)
model_multiclassif_ranger <- ranger(status~., data = HR, num.trees = 50)
model_multiclassif_ranger_prob <- ranger(status~., data = HR, num.trees = 50, probability = TRUE)

# explain()
p_fun_ranger <- function(model, x) predict(model, x)$predictions
p_fun_glm <- function(model, x) predict(model, x, type = "response")


explainer_classif_ranger  <- explain(model_classif_ranger, data = titanic_imputed, y = titanic_imputed$survived)
explainer_classif_glm  <- explain(model_classif_glm, data = HR, predict_function = p_fun_glm)
explainer_regr_ranger <- explain(model_regr_ranger, data = apartments_test[1:1000, ], y = apartments_test$m2.price[1:1000])
explainer_regr_ranger_wo_y <- explain(model_regr_ranger, data = apartments_test[1:1000, ])
explainer_regr_lm <- explain(model_regr_lm, data = apartments_test[1:1000, ], y = apartments_test$m2.price[1:1000])
explainer_wo_data  <- explain(model_classif_ranger, data = NULL)



