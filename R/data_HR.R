#' Human Resources Data
#'
#' Datasets \code{HR} and \code{HR_test} are artificial, generated form the same model.
#' Structure of the dataset is based on a real data, from Human Resources department with
#' information which employees were promoted, which were fired.
#'
#' Values are generated in a way to:
#' - have interaction between age and gender for the 'fired' variable
#' - have non monotonic relation for the salary variable
#' - have linear effects for hours and evaluation.
#'
#' \itemize{
#' \item gender - gender of an employee.
#' \item age - age of an employee in the moment of evaluation.
#' \item hours - average number of working hours per week.
#' \item evaluation - evaluation in the scale 2 (bad) - 5 (very good).
#' \item salary - level of salary in the scale 0 (lowest) - 5 (highest).
#' \item status - target variable, either `fired` or `promoted` or `ok`.
#' }
#'
#' @aliases HRTest HR_test
#' @docType data
#' @keywords HR
#' @name HR
#' @usage data(HR)
#' @format a data frame with 10000 rows and 6 columns
NULL


# N <- 10000
# set.seed(1313)
#
# gender <- rbinom(N, size = 1, prob = 0.5)
# age    <- runif(N, 20, 60)
# hours  <- 35 + 45*runif(N, 0, 1)^2
# evaluation <- floor(runif(N, 0, 4)) + 2
# salary <- floor(runif(N, 0, 6))
#
#
# score1 <- 2*(gender - 0.5)*(age-40)/15 + 0.35*(salary - 2.5)^2 - 1.6*(hours > 45)
# score2 <- 2*(evaluation > 3.5) + (hours-50)/15
#
# y1 <- runif(N) < pnorm(score1 - mean(score1))
# y2 <- runif(N) < pnorm(score2 - mean(score2))
#
# HR <- data.frame(gender = factor(ifelse(gender == 0, "female", "male")),
#                  age, hours, evaluation, salary,
#                  status = factor(ifelse(y1 == 1, "fired",
#                                         ifelse(y2 == 1, "promoted",
#                                                "ok"))),
#                  y1 = factor(y1),
#                  y2 = factor(y2))
# HR <- HR[!(y1&y2),1:6]

