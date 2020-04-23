#' FIFA 20 preprocessed data
#'
#' The \code{fifa} dataset is a preprocessed \code{players_20.csv} dataset which comes as
#' a part of "FIFA 20 complete player dataset" at Kaggle.
#'
#' It contains 5000 'overall' best players and 43 variables. These are:
#' \itemize{
#' \item short_name (rownames)
#' \item nationality of the player (not used in modeling)
#' \item overall, potential, value_eur, wage_eur (4 potential target variables)
#' \item age, height, weight, attacking skills, defending skills, goalkeeping skills (37 variables)
#' }
#' It is advised to leave only one target variable for modeling.
#'
#'
#' Source: \url{https://www.kaggle.com/stefanoleone992/fifa-20-complete-player-dataset}
#'
#' All transformations:
#' \enumerate{
#' \item take 43 columns: \code{[3, 5, 7:9, 11:14, 45:78]} (R indexing)
#' \item take rows with \code{value_eur > 0}
#' \item convert \code{short_name} to ASCII
#' \item remove rows with duplicated \code{short_name} (keep first)
#' \item sort rows on \code{overall} and take top \code{5000}
#' \item set \code{short_name} column as rownames
#' \item transform \code{nationality} to factor
#' \item reorder columns
#' }
#'
#' @docType data
#' @keywords fifa
#' @name fifa
#'
#' @source The \code{players_20.csv} dataset was downloaded from the Kaggle site and went through few transformations.
#' The complete dataset was obtained from
#' \url{https://www.kaggle.com/stefanoleone992/fifa-20-complete-player-dataset#players_20.csv} on January 1, 2020.
#' @usage
#' data(fifa)
#' @format a data frame with 5000 rows, 42 columns and rownames
NULL

