##############################
#### ONLY FOR LOCAL USAGE ####
##############################
library(testthat)

context("Check yhat() and model_info() functions")

##############################
#### HELPER OBJECTS ##########
##############################


# stops waring messages
assign("message_variable_importance", value = TRUE, envir = DALEX:::.DALEX.env)
assign("message_prediction_breakdown", value = TRUE, envir = DALEX:::.DALEX.env)
assign("message_partial_dependency", value = TRUE, envir = DALEX:::.DALEX.env)
assign("message_accumulated_dependency", value = TRUE, envir = DALEX:::.DALEX.env)


##############################
#### BEGIN TESTS #############
##############################

library(DALEX)


titanic_imputed_cut <- titanic_imputed[1:100,]
apartments_cut <- apartments[1:100,]

#:# removed due to the randomForest requirement for R v4.1

# test_that("randomForest", {
#   library(randomForest)
#
#   model_classif_rf <- randomForest(as.factor(survived)~., data = titanic_imputed_cut, num.trees = 50, probability = TRUE)
#   model_regr_rf <- randomForest(m2.price~., data = apartments_cut, num.trees = 50)
#
#   explainer_classif_rf <- explain(model_classif_rf, data = titanic_imputed_cut, y = titanic_imputed_cut$survived, verbose = FALSE)
#   explainer_classif_rf_pos <- explain(model_classif_rf, data = titanic_imputed_cut, y = titanic_imputed_cut$survived, verbose = FALSE, predict_function_target_column = 1)
#   explainer_regr_rf <- explain(model_regr_rf, data = apartments_cut, y = apartments_cut$m2.price, verbose = FALSE)
#   expect_is(explainer_classif_rf$y_hat, "numeric")
#   expect_is(explainer_classif_rf_pos$y_hat, "numeric")
#   expect_false(all(explainer_classif_rf_pos$y_hat == explainer_classif_rf$y_hat))
#   expect_is(explainer_classif_rf$model_info, "model_info")
#   expect_is(explainer_regr_rf$y_hat, "numeric")
#   expect_is(explainer_regr_rf$model_info, "model_info")
#   expect_length(DALEX:::yhat.randomForest(model_classif_rf, titanic_imputed_cut[1,]), 1)
#   expect_length(DALEX:::yhat.randomForest(model_regr_rf, apartments_cut[1,]), 1)
#
# })

test_that("svm", {
  library(e1071)

  model_classif_svm <- svm(as.factor(survived)~., data = titanic_imputed_cut, num.trees = 50, probability = TRUE)
  model_regr_svm <- svm(m2.price~., data = apartments_cut, num.trees = 50)
  explainer_classif_svm <- explain(model_classif_svm, data = titanic_imputed_cut, y = titanic_imputed_cut$survived, verbose = FALSE)
  explainer_classif_svm_pos <- explain(model_classif_svm, data = titanic_imputed_cut, y = titanic_imputed_cut$survived, verbose = FALSE, predict_function_target_column = 1)
  explainer_regr_svm <- explain(model_regr_svm, data = apartments_cut, y = apartments_cut$m2.price, verbose = FALSE)
  expect_is(explainer_classif_svm$y_hat, "numeric")
  expect_is(explainer_classif_svm_pos$y_hat, "numeric")
  expect_false(all(explainer_classif_svm_pos$y_hat == explainer_classif_svm$y_hat))
  expect_is(explainer_classif_svm$model_info, "model_info")
  expect_is(explainer_regr_svm$y_hat, "numeric")
  expect_is(explainer_regr_svm$model_info, "model_info")
  expect_length(DALEX:::yhat.svm(model_classif_svm, titanic_imputed_cut[1,]), 1)
  expect_length(DALEX:::yhat.svm(model_regr_svm, apartments_cut[1,]), 1)

})

test_that("gbm", {

  library(gbm)

  model_classif_gbm <- gbm(survived~., data = titanic_imputed_cut, distribution = "bernoulli",
                           n.trees = 50)
  model_regr_gbm <- gbm(m2.price~., data = apartments_cut, n.trees = 50, distribution = "gaussian")

  explainer_classif_gbm <- explain(model_classif_gbm, data = titanic_imputed_cut, y = titanic_imputed_cut$survived, verbose = FALSE)
  explainer_classif_gbm_pos <- explain(model_classif_gbm, data = titanic_imputed_cut, y = titanic_imputed_cut$survived, verbose = FALSE, predict_function_target_column = 0)
  explainer_regr_gbm <- explain(model_regr_gbm, data = apartments_cut, y = apartments_cut$m2.price, verbose = FALSE)
  expect_is(explainer_classif_gbm$y_hat, "numeric")
  expect_is(explainer_classif_gbm_pos$y_hat, "numeric")
  # gbm returns only one column for binary so positive class should not affect it
  expect_true(all(explainer_classif_gbm_pos$y_hat == explainer_classif_gbm$y_hat))
  expect_is(explainer_classif_gbm$model_info, "model_info")
  expect_is(explainer_regr_gbm$y_hat, "numeric")
  expect_is(explainer_regr_gbm$model_info, "model_info")
  expect_length(DALEX:::yhat.gbm(model_classif_gbm, titanic_imputed_cut[1,]), 1)
  expect_length(DALEX:::yhat.gbm(model_regr_gbm, apartments_cut[1,]), 1)


})

test_that("glmnet and cvglmnet", {

  library(glmnet)

  model_regr_glm <- glmnet(matrix(rnorm(100 * 20), 100, 20), rnorm(100))
  model_classif_glm_bin <- glmnet(matrix(rnorm(100 * 20), 100, 20), as.factor(round(runif(100))), family = "binomial")
  model_classif_glm_multi <- glmnet(matrix(rnorm(100 * 20), 100, 20), as.factor(round(runif(100, 0, 2))), family = "multinomial")

  model_regr_cvglm <- cv.glmnet(matrix(rnorm(100 * 20), 100, 20), rnorm(100))
  model_classif_cvglm_bin <- cv.glmnet(matrix(rnorm(100 * 20), 100, 20), as.factor(round(runif(100))), family = "binomial")

  set.seed(123)
  explainer_regr_glm <- explain(model_regr_glm, matrix(rnorm(100 * 20), 100, 20), rnorm(100), verbose = FALSE)
  expect_is(explainer_regr_glm$y_hat, "numeric")
  expect_is(explainer_regr_glm$model_info, "model_info")
  expect_length(DALEX:::yhat.glmnet(model_regr_glm, matrix(rnorm(1 * 20), 1, 20)), 1)


  explainer_regr_cvglm <- explain(model_regr_cvglm, matrix(rnorm(100 * 20), 100, 20), rnorm(100), verbose = FALSE)
  expect_is(explainer_regr_cvglm$y_hat, "numeric")
  expect_is(explainer_regr_cvglm$model_info, "model_info")
  expect_length(DALEX:::yhat.cv.glmnet(model_regr_cvglm, matrix(rnorm(1 * 20), 1, 20)), 1)

  explainer_classif_cvglm_bin <- explain(model_classif_cvglm_bin, matrix(rnorm(100 * 20), 100, 20), rnorm(100), verbose = FALSE)
  expect_is(explainer_regr_glm$y_hat, "numeric")
  expect_is(explainer_regr_glm$model_info, "model_info")
  expect_length(DALEX:::yhat.glmnet(model_regr_glm, matrix(rnorm(1 * 20), 1, 20)), 1)


  explainer_classif_glm_bin <- explain(model_classif_glm_bin, matrix(rnorm(100 * 20), 100, 20), rnorm(100), verbose = FALSE)
  expect_is(explainer_regr_cvglm$y_hat, "numeric")
  expect_is(explainer_regr_cvglm$model_info, "model_info")
  expect_length(DALEX:::yhat.cv.glmnet(model_regr_cvglm, matrix(rnorm(1 * 20), 1, 20)), 1)

  explainer_model_classif_glm_multi <- explain(model_classif_glm_multi, matrix(rnorm(100 * 20), 100, 20), rnorm(100), verbose = FALSE)
  expect_is(explainer_regr_glm$y_hat, "numeric")
  expect_is(explainer_regr_glm$model_info, "model_info")
  expect_length(DALEX:::yhat.glmnet(model_regr_glm, matrix(rnorm(1 * 20), 1, 20)), 1)

})

test_that("parsnip", {

  library(parsnip)

  parsnip_regr <- svm_rbf(mode = "regression", rbf_sigma = 0.2)
  parsnip_regr <- set_engine(parsnip_regr, "kernlab")
  parsnip_regr <- fit(parsnip_regr, fare ~ ., data = titanic_imputed_cut)
  parsnip_classif <- svm_rbf(mode = "classification", rbf_sigma = 0.2)
  parsnip_classif <- set_engine(parsnip_classif, "kernlab")
  parsnip_classif <- fit(parsnip_classif, as.factor(survived) ~ ., data = titanic_imputed_cut)



  explainer_classif_parsnip <- explain(parsnip_classif, data = titanic_imputed_cut, y = titanic_imputed_cut$survived, verbose = FALSE)
  explainer_classif_parsnip_pos <- explain(parsnip_classif, data = titanic_imputed_cut, y = titanic_imputed_cut$survived, verbose = FALSE, predict_function_target_column = 1)
  explainer_regr_parsnip <- explain(parsnip_regr, data = titanic_imputed_cut, y = titanic_imputed_cut$fare, verbose = FALSE)
  expect_is(explainer_classif_parsnip$y_hat, "numeric")
  expect_is(explainer_classif_parsnip_pos$y_hat, "numeric")
  expect_false(all(explainer_classif_parsnip_pos$y_hat == explainer_classif_parsnip$y_hat))
  expect_is(explainer_classif_parsnip$model_info, "model_info")
  expect_is(explainer_regr_parsnip$y_hat, "numeric")
  expect_is(explainer_regr_parsnip$model_info, "model_info")
  expect_length(DALEX:::yhat.model_fit(parsnip_classif, titanic_imputed_cut[1,]), 1)
  expect_length(DALEX:::yhat.model_fit(parsnip_regr, titanic_imputed_cut[1,]), 1)

})

test_that("caret", {

  library(caret)

  caret_regr <- train(m2.price~., data = apartments_cut, method="ranger", num.trees = 50)
  caret_regr_lm <- train(m2.price~., data = apartments_cut, method="lm")
  caret_classif <- train(as.factor(survived)~., data = titanic_imputed_cut, method="ranger", num.trees = 50)

  explainer_classif_caret <- explain(caret_classif, data = titanic_imputed_cut, y = titanic_imputed_cut$survived, verbose = FALSE)
  explainer_classif_caret_pos <- explain(caret_classif, data = titanic_imputed_cut, y = titanic_imputed_cut$survived, verbose = FALSE, predict_function_target_column = 1)
  explainer_regr_caret <- explain(caret_regr, data = apartments_cut, y = apartments_cut$m2.price, verbose = FALSE)
  explainer_regr_caret_lm <- explain(caret_regr_lm, data = apartments_cut, y = apartments_cut$m2.price, verbose = FALSE)
  expect_is(explainer_classif_caret$y_hat, "numeric")
  expect_is(explainer_classif_caret_pos$y_hat, "numeric")
  expect_false(all(explainer_classif_caret_pos$y_hat == explainer_classif_caret$y_hat))
  expect_is(explainer_classif_caret$model_info, "model_info")
  expect_is(explainer_regr_caret$y_hat, "numeric")
  expect_is(explainer_regr_caret$model_info, "model_info")
  expect_is(explainer_regr_caret_lm$y_hat, "numeric")
  expect_is(explainer_regr_caret_lm$model_info, "model_info")
  expect_error(print(explainer_classif_caret$model_info), NA)
  expect_length(DALEX:::yhat.train(caret_regr, apartments_cut[1,]), 1)

})


test_that("glm", {

  model_classif_glm <- glm(status == "fired"~., data = HR, family = "binomial")
  explainer_classif_glm <- explain(model_classif_glm, data = HR, verbose = FALSE)
  expect_is(explainer_classif_glm$y_hat, "numeric")
  expect_is(explainer_classif_glm$model_info, "model_info")
  expect_length(DALEX:::yhat(model_classif_glm, HR[1,]), 1)


})

test_that("rpart", {

  library(rpart)

  model_classif_rpart <- rpart(as.factor(survived)~., data = titanic_imputed_cut)
  model_regr_rpart <- rpart(m2.price~., data = apartments_cut)

  explainer_classif_rpart <- explain(model_classif_rpart, data = titanic_imputed_cut, y = titanic_imputed_cut$survived, verbose = FALSE)
  explainer_classif_rpart_pos <- explain(model_classif_rpart, data = titanic_imputed_cut, y = titanic_imputed_cut$survived, verbose = FALSE, predict_function_target_column = 1)
  explainer_regr_rpart <- explain(model_regr_rpart, data = apartments_cut, y = apartments_cut$m2.price, verbose = FALSE)
  expect_is(explainer_classif_rpart$y_hat, "numeric")
  expect_is(explainer_classif_rpart_pos$y_hat, "numeric")
  expect_false(all(explainer_classif_rpart_pos$y_hat == explainer_classif_rpart$y_hat))
  expect_is(explainer_classif_rpart$model_info, "model_info")
  expect_is(explainer_regr_rpart$y_hat, "numeric")
  expect_is(explainer_regr_rpart$model_info, "model_info")
  expect_length(DALEX:::yhat.rpart(model_classif_rpart, titanic_imputed_cut[1,]), 1)
  expect_length(DALEX:::yhat.rpart(model_regr_rpart, apartments_cut[1,]), 1)



})

test_that("yhat default", {
  model_classif_rpart <- rpart(as.factor(survived)~., data = titanic_imputed_cut)
  model_regr_rpart <- rpart(m2.price~., data = apartments_cut)

  expect_is(DALEX:::yhat.default(model_classif_rpart, titanic_imputed_cut), "numeric")
  expect_length(DALEX:::yhat.default(model_classif_rpart, titanic_imputed_cut[1,]), 1)

  expect_length(DALEX:::yhat.default(model_regr_rpart, apartments_cut[1,]), 1)
  expect_is(DALEX:::yhat.default(model_regr_rpart, apartments_cut), "numeric")


})

test_that("yhat ranger", {
  model_classif_ranger <- ranger::ranger(survived~., data = titanic_imputed, num.trees = 50, probability = TRUE)
  model_regr_ranger <- ranger::ranger(m2.price~., data = apartments, num.trees = 50)
  expect_length(DALEX:::yhat.ranger(model_classif_ranger, titanic_imputed[1,]), 1)
  expect_length(DALEX:::yhat.ranger(model_regr_ranger, apartments[1,]), 1)

})
