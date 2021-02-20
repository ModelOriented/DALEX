# installed.packages are slow, so let's find out what is installed quicker
quick.installed.packages <- function() {
  where_to_look <- .libPaths()
  packages <- sapply(where_to_look, function(dir) list.dirs(dir, recursive = FALSE, full.names = FALSE))
  unique(unlist(packages))
}

.onAttach <- function(...) {
  addons <- setdiff(c("ingredients", "iBreakDown", "ggpubr"),
                    quick.installed.packages())

  packageStartupMessage("Welcome to DALEX (version: ", utils::packageVersion("DALEX"), ").\n",
    "Find examples and detailed introduction at: http://ema.drwhy.ai/\n",
    ifelse(length(addons) == 0, "",
    paste0( "Additional features will be available after installation of: ",
            paste(addons, collapse = ", "),".\nUse 'install_dependencies()' to get all suggested dependencies"))
        )
}

.DALEX.env <- new.env()
