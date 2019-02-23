#' Dragon Data
#'
#' Datasets \code{dragons} and \code{dragons_test} are artificial, generated form the same ground truth model, 
#' but with sometimes different data distridution.
#' 
#' Values are generated in a way to:
#' - have nonlinearity in year_of_birth and height
#' - have concept drift in the test set
#'
#' \itemize{
#' \item year_of_birth - year in which the dragon was born. Negative year means year BC, eg: -1200 = 1201 BC
#' \item year_of_discovery - year in which the dragon was found.
#' \item height - height of the dragon in yards.
#' \item weight - weight of the dragon in tons.
#' \item scars - number of scars.
#' \item colour - colour of the dragon.
#' \item number_of_lost_teeth - number of teeth that the dragon lost.
#' \item life_length - life length of the dragon.
#' }
#'
#' @aliases dragons_test
#' @docType data
#' @keywords dragons
#' @name dragons
#' @usage data(dragons)
#' @format a data frame with 2000 rows and 8 columns
NULL


# true_model <- function(model, data){
#   return(abs(500 + 100 * (abs(data$year_of_birth - 1000) > 500) + 
#                0.02 * (data$height - 50)^2 + 40 * data$scars + 20 * data$number_of_lost_teeth))
# }
# 
# N <- 2000
# set.seed(756)
# 
# year_of_birth <- round(runif(N, -2000, 1800)) # year, for negative -n = n + 1 BC
# year_of_discovery <- sort(round(runif(N, 1700, 1800))) # year
# height <- rgamma(N, 50) # yards
# weight <- 1/4 * height + rexp(N, 1) # tons
# scars <- round(rexp(N, 0.1)) # number
# number_of_lost_teeth <- round(runif(N, 0, 40)) # number
# colour <- sample(c('red', 'blue', 'green', 'black'), N, replace=TRUE, p=c(0.5, 0.3, 0.18, 0.02))
# life_length <- true_model(NULL, data.frame(year_of_birth, height, weight, 
#                                            scars, colour, year_of_discovery, 
#                                            number_of_lost_teeth)) + rnorm(N, 0, 20)
# dragons <- data.frame(year_of_birth, height, weight, scars, colour, 
#                  year_of_discovery, number_of_lost_teeth, life_length)
# 
# N <- 1000
# set.seed(1)
# 
# year_of_birth <- round(runif(N, -2000, 2000)) # year, for negative -n = n + 1 BC
# year_of_discovery <- sort(round(runif(N, 1800, 2000))) # year
# colour <- sample(c('red', 'blue', 'green', 'black'), N, replace=TRUE, p=c(0.4, 0.3, 0.1, 0.2))
# height <- ifelse(
#   colour == 'black' & year_of_discovery > 1840,
#   rgamma(N, 200),
#   rgamma(N, 50)
# ) # yards
# weight <- 1/4 * height + rexp(N, 1) # tons
# scars <- round(rexp(N, 0.1)) # number
# number_of_lost_teeth <- round(runif(N, 0, 40)) # number
# life_length <- true_model(NULL, data.frame(year_of_birth, height, weight, 
#                                            scars, colour, year_of_discovery, 
#                                            number_of_lost_teeth)) + rnorm(N, 0, 20)
# dragons_test <- data.frame(year_of_birth, height, weight, scars, colour, 
#                       year_of_discovery, number_of_lost_teeth, life_length)
