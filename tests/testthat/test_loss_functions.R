context("Check loss functions")

model_classif_multilabel_rf <-
  randomForest::randomForest(status ~ ., data = HR, ntree = 50)

test_that("Output length = 1", {
  expect_length(loss_cross_entropy(
    HR$status,
    predict(
      model_classif_multilabel_rf,
      newdata = HR,
      type = "prob"
    )
  ), 1)
  expect_length(loss_root_mean_square(
    explainer_regr_rf$y,
    predict(explainer_regr_rf, newdata = explainer_regr_rf$data)
  ), 1)
  expect_length(loss_sum_of_squares(
    explainer_regr_rf$y,
    predict(explainer_regr_rf, newdata = explainer_regr_rf$data)
  ), 1)
  expect_length(loss_accuracy(
    as.factor(HR$status == "fired"),
    predict(explainer_classif_rf, newdata = explainer_classif_rf$data)
  ), 1)
  expect_length(loss_one_minus_auc(
    as.numeric(HR$status == "fired"),
    predict.glm(
      explainer_classif_glm$model,
      newdata = explainer_classif_glm$data,
      type = "response"
    )
  ), 1)

})
