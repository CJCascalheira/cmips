# COMBINE PSYCHOMETRIC SURVEYS AND DICHOTOMIZE ----------------------------

# Author = Cory Cascalheira
# Date = 03/23/2024

# Combine the psychometric surveys. Create classification labels.

# DEPENDENCIES AND IMPORT -------------------------------------------------

# Load dependencies
library(tidyverse)
library(rio)

# Import 
cmips_strain <- read_csv("data/participants/cleaned/cmips_strain_severity.csv")
cmips_qualitrics <- read_csv("data/participants/cleaned/cmips_qualtrics.csv")

# COMBINE SURVEYS ---------------------------------------------------------

# Combine the surveys
cmips_surveys <- left_join(cmips_qualitrics, cmips_strain, by = "email")

# DICHOTOMIZE OUTCOMES FOR CLASSIFICATION ---------------------------------

# Coming soon

# EXPORT ------------------------------------------------------------------

# Save the file
write_csv(cmips_surveys, "data/participants/for_analysis/cmips_surveys.csv")
