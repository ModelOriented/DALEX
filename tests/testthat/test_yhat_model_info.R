##############################
#### ONLY FOR LOCAL USAGE ####
##############################


# context("Check yhat() and model_info() functions")
#
# source("helper-objects.R")
#
#
# test_that("Output is numeric or model_info", {
#   skip_if_no_codecov()
#   model_classif_rf <- randomForest(as.factor(survived)~., data = titanic_imputed, num.trees = 50, probability = TRUE)
#   model_regr_rf <- randomForest(m2.price~., data = apartments, num.trees = 50)
#
#   model_classif_svm <- svm(as.factor(survived)~., data = titanic_imputed, num.trees = 50, probability = TRUE)
#   model_regr_svm <- svm(m2.price~., data = apartments, num.trees = 50)
#
#   model_classif_gbm <- gbm(as.factor(survived)~., data = titanic_imputed, distribution = "bernoulli",
#                            n.trees = 50)
#   model_regr_gbm <- gbm(m2.price~., data = apartments, n.trees = 50, distribution = "gaussian")
#
#   model_regr_glm <- glmnet(matrix(rnorm(100 * 20), 100, 20), rnorm(100))
#
#   model_regr_cvglm <- cv.glmnet(matrix(rnorm(100 * 20), 100, 20), rnorm(100))
#
#   parsnip_regr <- svm_rbf(mode = "regression", rbf_sigma = 0.2)
#   parsnip_regr <- set_engine(parsnip_regr, "kernlab")
#   parsnip_regr <- fit(parsnip_regr, fare ~ ., data = titanic_imputed)
#   parsnip_classif <- svm_rbf(mode = "classification", rbf_sigma = 0.2)
#   parsnip_classif <- set_engine(parsnip_classif, "kernlab")
#   parsnip_classif <- fit(parsnip_classif, as.factor(survived) ~ ., data = titanic_imputed)
#
#   caret_regr <- train(m2.price~., data = apartments, method="rf", ntree = 50)
#   caret_regr_lm <- train(m2.price~., data = apartments, method="lm", ntree = 50)
#   caret_classif <- train(as.factor(survived)~., data = titanic_imputed, method="rf", ntree = 50)
#
#   explainer_classif_rf <- explain(model_classif_rf, data = titanic_imputed, y = titanic_imputed$survived)
#   explainer_regr_rf <- explain(model_regr_rf, data = apartments, y = apartments$m2.price)
#   expect_is(explainer_classif_rf$y_hat, "numeric")
#   expect_is(explainer_classif_rf$model_info, "model_info")
#   expect_is(explainer_regr_rf$y_hat, "numeric")
#   expect_is(explainer_regr_rf$model_info, "model_info")
#
#   explainer_classif_svm <- explain(model_classif_svm, data = titanic_imputed, y = titanic_imputed$survived)
#   explainer_regr_svm <- explain(model_regr_svm, data = apartments, y = apartments$m2.price)
#   expect_is(explainer_classif_svm$y_hat, "numeric")
#   expect_is(explainer_classif_svm$model_info, "model_info")
#   expect_is(explainer_regr_svm$y_hat, "numeric")
#   expect_is(explainer_regr_svm$model_info, "model_info")
#
#   explainer_classif_gbm <- explain(model_classif_gbm, data = titanic_imputed, y = titanic_imputed$survived)
#   explainer_regr_gbm <- explain(model_regr_gbm, data = apartments, y = apartments$m2.price)
#   expect_is(explainer_classif_gbm$y_hat, "numeric")
#   expect_is(explainer_classif_gbm$model_info, "model_info")
#   expect_is(explainer_regr_gbm$y_hat, "numeric")
#   expect_is(explainer_regr_gbm$model_info, "model_info")
#
#   explainer_regr_glm <- explain(model_regr_glm, matrix(rnorm(100 * 20), 100, 20), rnorm(100))
#   expect_is(explainer_regr_glm$y_hat, "matrix")
#   expect_is(explainer_regr_glm$model_info, "model_info")
#
#   explainer_regr_cvglm <- explain(model_regr_cvglm, matrix(rnorm(100 * 20), 100, 20), rnorm(100))
#   expect_is(explainer_regr_cvglm$y_hat, "matrix")
#   expect_is(explainer_regr_cvglm$model_info, "model_info")
#
#   explainer_classif_parsnip <- explain(parsnip_classif, data = titanic_imputed, y = titanic_imputed$survived)
#   explainer_regr_parsnip <- explain(parsnip_regr, data = titanic_imputed, y = titanic_imputed$fare)
#   expect_is(explainer_classif_parsnip$y_hat, "numeric")
#   expect_is(explainer_classif_parsnip$model_info, "model_info")
#   expect_is(explainer_regr_parsnip$y_hat, "numeric")
#   expect_is(explainer_regr_parsnip$model_info, "model_info")
#
#   explainer_classif_caret <- explain(caret_classif, data = titanic_imputed, y = titanic_imputed$survived)
#   explainer_regr_caret <- explain(caret_regr, data = apartments, y = apartments$m2.price)
#   explainer_regr_caret_lm <- explain(caret_regr_lm, data = apartments, y = apartments$m2.price)
#   expect_is(explainer_classif_caret$y_hat, "numeric")
#   expect_is(explainer_classif_caret$model_info, "model_info")
#   expect_is(explainer_regr_caret$y_hat, "numeric")
#   expect_is(explainer_regr_caret$model_info, "model_info")
#   expect_is(explainer_regr_caret_lm$y_hat, "numeric")
#   expect_is(explainer_regr_caret_lm$model_info, "model_info")
#
#   explainer_classif_glm <- explain(model_classif_glm, data = HR)
#   expect_is(explainer_classif_glm$y_hat, "numeric")
#   expect_is(explainer_classif_glm$model_info, "model_info")
#
#
#   expect_error(print(explainer_classif_caret$model_info), NA)
#
#
#
# })
