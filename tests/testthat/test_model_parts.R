context("Check model_parts() function")

mp_ranger <- model_parts(explainer_regr_ranger, N = 100)
mp_lm <- model_parts(explainer_regr_lm, N = 100)
mp_ranger_ratio <- model_parts(explainer_regr_ranger, N = 100, type = "ratio")

test_that("Description prints properly", {
  des <- ingredients::describe(mp_ranger)
  expect_error(print(des), NA)
})

test_that("y not provided",{
  expect_error(model_parts(explainer_regr_ranger_wo_y, N = 100))
})

test_that("data not provided",{
  expect_error(model_parts(explainer_wo_data, N = 100))
})

test_that("wrong type value",{
  expect_error(model_parts(explainer_regr_ranger, type="anything"))
})

test_that("Wrong object class (not explainer)", {
  expect_error(model_parts(c(1,1)))
})

test_that("Output format",{
  expect_is(mp_ranger, c("model_parts", "feature_importance_explainer"))
  expect_is(mp_lm, c("model_parts", "feature_importance_explainer"))
  expect_is(mp_ranger_ratio, c("model_parts", "feature_importance_explainer"))
})

test_that("Output format - plot",{
  expect_is(plot(mp_ranger_ratio), "gg")
  expect_is(plot(mp_ranger, mp_lm), "gg")
  expect_is(plot(mp_lm), "gg")
})


test_that("Inverse sorting of bars",{
  expect_is(plot(mp_ranger_ratio, desc_sorting = FALSE), "gg")
  expect_is(plot(mp_ranger, mp_lm, desc_sorting = FALSE), "gg")
  expect_is(plot(mp_ranger, desc_sorting = FALSE), "gg")
})

#:# alias

fi_ranger <- feature_importance(explainer_regr_ranger, N = 100)
fi_lm <- feature_importance(explainer_regr_lm, N = 100)
fi_ranger_ratio <- feature_importance(explainer_regr_ranger, N = 100, type = "ratio")
vi_ranger <- variable_importance(explainer_regr_ranger, N = 100)
vi_lm <- variable_importance(explainer_regr_lm, N = 100)
vi_ranger_ratio <- variable_importance(explainer_regr_ranger, N = 100, type = "ratio")

test_that("Output format",{
  expect_is(fi_ranger, c("model_parts", "feature_importance_explainer"))
  expect_is(fi_lm, c("model_parts", "feature_importance_explainer"))
  expect_is(fi_ranger_ratio, c("model_parts", "feature_importance_explainer"))
  expect_is(vi_ranger, c("model_parts", "feature_importance_explainer"))
  expect_is(vi_lm, c("model_parts", "feature_importance_explainer"))
  expect_is(vi_ranger_ratio, c("model_parts", "feature_importance_explainer"))
})

test_that("Output format - plot",{
  expect_is(plot(fi_ranger_ratio), "gg")
  expect_is(plot(fi_ranger, fi_lm), "gg")
  expect_is(plot(fi_lm), "gg")
  expect_is(plot(vi_ranger_ratio), "gg")
  expect_is(plot(vi_ranger, vi_lm), "gg")
  expect_is(plot(vi_lm), "gg")
})
