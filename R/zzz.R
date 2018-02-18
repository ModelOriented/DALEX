.onAttach <- function(...) {
  packageStartupMessage("Welcome to DALEX (version: ", utils::packageVersion("DALEX"), ").")
}

## no S4 methodology here; speedup :
.noGenerics <- TRUE


