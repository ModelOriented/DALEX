context("Check explain() function")

source("helper-objects.R")

test_that("Type of data in the explainer and label",{
  explainer_ranger_1 <- explain(model_classif_ranger, label = "Test", verbose = FALSE)
  explainer_ranger_2 <- explain(model_classif_ranger, data = titanic_imputed, label = "Test 2", verbose = FALSE)
  explainer_lm_1 <- explain(model_regr_lm, verbose = FALSE)
  data_matrix <- as.matrix(titanic_imputed)
  rownames(data_matrix) <- NULL
  explainer_ranger_13 <- explain(model_classif_ranger, data = data_matrix, label = "rownames", verbose = FALSE)
  colnames(data_matrix) <- NULL
  explainer_ranger_14 <- explain(model_classif_ranger, data = data_matrix, label = "colnames", verbose = FALSE)


  expect_is(explainer_ranger_1, "explainer")
  expect_is(explainer_ranger_2, "explainer")
  expect_is(explainer_lm_1, "explainer")
  expect_is(explainer_ranger_1$label, "character")
  expect_is(explainer_ranger_2$label, "character")
  expect_is(explainer_lm_1$label, "character")
  expect_null(explainer_ranger_1$data)
  expect_is(explainer_ranger_2$data, "data.frame")
  expect_is(explainer_lm_1$data, "data.frame")
  expect_false(is.null(rownames(explainer_ranger_13$data)))
  expect_false(is.null(colnames(explainer_ranger_14$data)))
})

test_that("Checks for y",{


  explainer_ranger_3 <- explain(model_classif_ranger, data = titanic_imputed, verbose = FALSE)
  explainer_ranger_4 <- explain(model_classif_ranger, data = titanic_imputed, y = data.frame(titanic_imputed$survived), verbose = FALSE)
  explainer_ranger_5 <- explain(model_classif_ranger, data = titanic_imputed, y = c(1, 1, 1, 1, 1), verbose = FALSE)
  explainer_ranger_6 <- explain(model_classif_ranger, data = titanic_imputed, y = as.factor(titanic_imputed$survived), verbose = FALSE)
  explainer_ranger_7 <- explain(model_classif_ranger, data = titanic_imputed[,-8], y = titanic_imputed$survived, verbose = FALSE)

  expect_null(explainer_ranger_3$y)
  expect_is(explainer_ranger_4$y, "numeric")
  expect_is(explainer_ranger_5, "explainer")
  expect_is(explainer_ranger_6, "explainer")
  expect_is(explainer_ranger_7, "explainer")

})


test_that("Checks for model_info, weights and prints",{


  explainer_lm_2 <- explain(model_regr_lm, data = apartments, y = apartments$m2.price, weights = rep(1, times = nrow(apartments)), verbose = FALSE)
  explainer_lm_3 <- explain(model_regr_lm, data = apartments, y = apartments$m2.price,
                            weights = data.frame(rep(1, times = nrow(apartments))), verbose = FALSE)
  explainer_ranger_8 <- explain(model_regr_ranger, data = apartments[,-1], y = apartments$m2.price, weights = c(1,1,1,1), verbose = FALSE)

  expect_is(explainer_lm_2, "explainer")
  expect_is(explainer_lm_3$weights, "numeric")
  expect_is(explainer_ranger_8, "explainer")
  expect_is(explainer_lm_2$model_info, "model_info")
  expect_is(explainer_lm_3$model_info, "model_info")
  expect_is(explainer_ranger_8$model_info, "model_info")
  # Check print causes no error
  expect_error(print(explainer_ranger_8$model_info), NA)
  expect_error(print(explainer_ranger_8), NA)
  expect_is(explain(list(1), verbose = FALSE)$model_info, "model_info")


})

test_that("predict and residual functions", {

  explainer_ranger_9 <- explain(model_multiclassif_ranger, data = HR, y = HR$status, predict_function = p_fun_ranger, verbose = FALSE)
  explainer_ranger_10 <- explain(model_regr_ranger, data = apartments[,-1], y = as.data.frame(apartments$m2.price), predict_function = p_fun_ranger,
                                residual_function = function(model, data, y) {
                                  y - p_fun_ranger(model, data)
                                }, weights =  rep(1, times = nrow(apartments)), verbose = FALSE)
  explainer_ranger_11 <- explain(model_classif_ranger, data = titanic_imputed[,-8], y = titanic_imputed$survived, verbose = FALSE)
  explainer_ranger_12 <- explain(model_multiclassif_ranger, data = HR, y = HR$status, verbose = FALSE)

  expect_is(explainer_ranger_9$y_hat, "factor")
  expect_is(explainer_ranger_10$residuals, "numeric")
  expect_is(explainer_ranger_11$y_hat, "numeric")


})

test_that("Checks tests", {
  expect_error(explain(model_classif_ranger, y = titanic_imputed$survived, verbose = FALSE), NA)
  expect_error(explain(model_classif_ranger, weights = rep(1, times=nrow(titanic_imputed)), verbose = FALSE), NA)
  expect_error(explain(model_classif_ranger, data = titanic_imputed, y = titanic_imputed$survived, predict_function = "function", verbose = FALSE), NA)
  expect_error(explain(model_classif_ranger, data = titanic_imputed, y = titanic_imputed$survived, residual_function = "function", verbose = FALSE), NA)
  expect_error(explain(model_classif_ranger, data = titanic_imputed, y = titanic_imputed$survived, label = list(test = "label"), verbose = FALSE), NA)
})


test_that("Suppressing output does not cause erros", {
  expect_error(explain(model_regr_lm, colorize = FALSE, verbose = FALSE, precalculate = FALSE), NA)
  expect_error(explain(model_regr_lm, colorize = FALSE, verbose = FALSE), NA)
  expect_error(explain(model_regr_lm, verbose = FALSE, precalculate = FALSE), NA)
})


test_that("update_* work", {
  linear_model <- lm(m2.price ~ construction.year + surface + floor + no.rooms + district, data = apartments)
  explainer_lm0 <- explain(linear_model, colorize = FALSE, verbose = FALSE)
  expect_error(update_label(explainer_lm0, 1, verbose = FALSE))
  expect_error(update_label(list(1), "Label", verbose = FALSE))
  expect_error(update_data(explainer_lm0, 1, verbose = FALSE))
  expect_error(update_data(list(1), apartmentsTest, verbose = FALSE))
  expect_is(update_label(explainer_lm0, "new_label", verbose = FALSE), "explainer")
  expect_is(update_data(explainer_lm0, apartments, as.data.frame(apartments$m2.price), verbose = FALSE), "explainer")
  expect_is(update_data(explainer_lm0, apartments, as.factor(apartments$m2.price), verbose = FALSE), "explainer")
})
