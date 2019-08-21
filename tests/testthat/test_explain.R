context("Check explain() function")

test_that("Type of data in the explainer",{
  apartmentsTest_tibble <- dplyr::as_tibble(apartments_test)
  linear_model <- lm(m2.price ~ construction.year + surface + floor + no.rooms + district, data = apartments)

  explainer_lm1 <- DALEX::explain(linear_model, data = apartmentsTest_tibble[,2:6], y = apartmentsTest_tibble$m2.price)
  explainer_lm2 <- DALEX::explain(linear_model, data = apartmentsTest_tibble[,2:6])

  expect_true(is.data.frame(explainer_lm1$data))
  expect_is(explainer_lm2, "explainer")
})
