context("Check loss functions")

test_that("Output length = 1", {
  expect_length(loss_cross_entropy(
    HR$status,
    predict(
      model_multiclassif_ranger_prob,
      HR
    )$predictions
  ), 1)
  expect_length(loss_root_mean_square(
    explainer_regr_ranger$y,
    predict(explainer_regr_ranger, newdata = explainer_regr_ranger$data)
  ), 1)
  expect_length(loss_sum_of_squares(
    explainer_regr_ranger$y,
    predict(explainer_regr_ranger, newdata = explainer_regr_ranger$data)
  ), 1)
  expect_length(loss_accuracy(
    HR$status,
    predict(model_multiclassif_ranger, HR)$predictions
  ), 1)
  expect_length(loss_one_minus_auc(
    titanic_imputed$survived,
    predict(model_classif_ranger, titanic_imputed)$predictions[,1]
  ), 1)
})
