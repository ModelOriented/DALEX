test_that("Type of data in the explainer",{
  apartmentsTest_tibble <- dplyr::as_tibble(apartmentsTest)
  model_liniowy <- lm(m2.price ~ construction.year + surface + floor + no.rooms + district, data = apartments)

  explainer_lm <- DALEX::explain(model_liniowy, data = apartmentsTest_tibble[,2:6], y = apartmentsTest_tibble$m2.price)

  expect_true(is.data.frame(explainer_lm$data))
})
