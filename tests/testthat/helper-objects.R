library(randomForest)
HR_data <- breakDown::HR_data

# models
model_classif_glm <- glm(factor(left)~., data = breakDown::HR_data, family = "binomial")
model_classif_rf <- randomForest::randomForest(factor(left)~., data = breakDown::HR_data, ntree = 50)
model_regr_rf <- randomForest::randomForest(m2.price~., data = apartments, ntree = 50)
model_regr_lm <- lm(m2.price~., data = apartments)

# explain()
p_fun_rf <- function(model, x) predict(model, x, type = "prob")[,2]
p_fun_glm <- function(model, x) predict(model, x, type = "response")
explainer_classif_rf  <- explain(model_classif_rf, data = HR_data, predict_function = p_fun_rf)
explainer_classif_glm  <- explain(model_classif_glm, data = HR_data, predict_function = p_fun_glm)
explainer_regr_rf <- explain(model_regr_rf, data = apartmentsTest[1:1000, ], y = apartmentsTest$m2.price[1:1000])
explainer_regr_rf_wo_y <- explain(model_regr_rf, data = apartmentsTest[1:1000, ])
explainer_regr_lm <- explain(model_regr_lm, data = apartmentsTest[1:1000, ], y = apartmentsTest$m2.price[1:1000])
explainer_wo_data  <- explain(model_classif_rf, data = NULL)



