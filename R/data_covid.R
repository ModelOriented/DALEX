#' Data for early COVID mortality
#'
#' Two datasets of characteristics of patients infected with COVID. It is important to note that these are not real patient data. This is simulated data, generated to have relationships consistent with real data (obtained from NIH), but the data itself is not real. Fortunately, they are sufficient for the purposes of our exercise.
#'
#' The data is divided into two sets covid_spring and covid_summer. The first is acquired in spring 2020 and will be used as training data while the second dataset is acquired in summer and will be used for validation. In machine learning, model validation is performed on a separate data set. This controls the risk of overfitting an elastic model to the data. If we do not have a separate set then it is generated using cross-validation, out of sample or out of time techniques.
#'
#' It contains 20 000 rows related fo COVID mortality. it contains 11 variables such as: Gender, Age, Cardiovascular.Diseases, Diabetes, Neurological.Diseases, Kidney.Diseases.
#'
#' Source: \url{https://github.com/BetaAndBit/RML}
#'
#' @docType data
#' @keywords covid_summer covid_spring
#' @aliases covid_summer covid_spring
#' @name covid
#'
#' @source \url{https://github.com/BetaAndBit/RML}
#' @usage
#' data(covid_summer)
#' data(covid_spring)
#' @format a data frame with 10 000 rows each and 12 columns
NULL
