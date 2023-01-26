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

test_that("Generators", {
  expect_is(
    get_loss_default(explainer_classif_ranger),
    "function"
  )
  expect_equal(
    attr(get_loss_default(explainer_classif_ranger), "loss_name"),
    "One minus AUC"
  )
  expect_is(
    get_loss_default(explainer_regr_lm),
    "function"
  )
  expect_equal(
    attr(get_loss_default(explainer_regr_lm), "loss_name"),
    "Root mean square error (RMSE)"
  )
  expect_is(
    get_loss_one_minus_accuracy(0.123, FALSE),
    "function"
  )
  expect_equal(
    get_loss_one_minus_accuracy(0.5, FALSE)(c(1, 1, 1), c(0, 0.5, 1)),
    loss_one_minus_accuracy(c(1, 1, 1), c(0, 0.5, 1))
  )
  expect_equal(
    get_loss_one_minus_accuracy(0.3, FALSE)(c(1, 1, 1), c(0, 0.5, 1)),
    loss_one_minus_accuracy(c(1, 1, 1), c(0, 0.5, 1))
  )
  expect_false(
    get_loss_one_minus_accuracy(0.7, FALSE)(c(1, 1, 1), c(0, 0.5, 1)) ==
      loss_one_minus_accuracy(c(1, 1, 1), c(0, 0.5, 1))
  )
  expect_equal(
    get_loss_one_minus_accuracy(0.7, FALSE)(c(1, 1, 1, 0), c(0.9, 0.9, 0.9, 0.9)),
    0.25
  )
  expect_equal(
    loss_one_minus_accuracy(c(1, 1, 1, 0), c(0.1, 0.1, 0.1, 0.1)),
    0.75
  )
  ff <- function(x)
  expect_is(
    get_loss_yardstick(ff),
    "function"
  )
  expect_is(
    get_loss_yardstick(ff, reverse = TRUE, reference = 2),
    "function"
  )
  expect_is(
    attr(get_loss_yardstick(ff), "loss_name"),
    "character"
  )

  # warnings
  expect_warning(
    DALEX::loss_yardstick()
  )
  expect_warning(
    DALEX::loss_default(explainer_classif_ranger)
  )
})

