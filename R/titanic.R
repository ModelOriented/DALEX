#' Passengers and Crew on the RMS Titanic
#'
#' The \code{titanic} data is a complete list of passengers and crew members on  the RMS Titanic.
#' It includes a variable indicating whether a person did  survive the sinking of the RMS
#' Titanic on April 15, 1912.
#'
#' This dataset was copied from the \code{stablelearner} package and went through few variable
#' transformations. Levels in \code{embarked} was replaced with full names, \code{sibsp}, \code{parch} and \code{fare}
#' were converted to numerical variables and values for crew were replaced with 0.
#' If you use this dataset please cite the original package.
#'
#' From \code{stablelearner}: The website \url{http://www.encyclopedia-titanica.org/} offers detailed  information about passengers and crew
#' members on the RMS Titanic. According to the website 1317 passengers and 890 crew member were abord.
#' 8 musicians and 9 employees of the shipyard company are listed as passengers, but travelled with a
#' free ticket, which is why they have \code{NA} values in \code{fare}. In addition to that, \code{fare}
#' is truely missing for a few regular passengers.
#'
#' \itemize{
#' \item gender a factor with levels \code{male} and \code{female}.
#' \item age a numeric value with the persons age on the day of the sinking.
#' \item class a factor specifying the class for passengers or the type of service aboard for crew members.
#' \item embarked a factor with the persons place of of embarkment (Belfast/Cherbourg/Queenstown/Southampton).
#' \item country a factor with the persons home country.
#' \item fare a numeric value with the ticket price (\code{0} for crew members, musicians and employees of the shipyard company).
#' \item sibsp an ordered factor specifying the number if siblings/spouses aboard; adopted from Vanderbild data set (see below).
#' \item parch an ordered factor specifying the number of parents/children aboard; adopted from Vanderbild data set (see below).
#' \item survived a factor with two levels (\code{no} and \code{yes}) specifying whether the person has survived the sinking.
#' }
#'
#' @docType data
#' @keywords titanic
#' @name titanic
#' @references   \url{http://www.encyclopedia-titanica.org/}, \url{http://biostat.mc.vanderbilt.edu/DataSets} and \url{https://CRAN.R-project.org/package=stablelearner}
#' @source This dataset was copied from the \code{stablelearner} package and went through few variable
#' transformations. The complete list of persons on the RMS titanic was downloaded from
#' \url{http://www.encyclopedia-titanica.org/} on April 5, 2016. The  information given
#' in \code{sibsp} and \code{parch} was adopoted from a data set obtained from \url{http://biostat.mc.vanderbilt.edu/DataSets}.
#' @usage data(titanic)
#' @format a data frame with 2207 rows and 11 columns
NULL

