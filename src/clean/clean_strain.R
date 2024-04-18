# CLEAN ADULT STRAIN ------------------------------------------------------

# Author = Cory Cascalheira
# Date = 03/23/2024

# Preprocess the Adult STRAIN.

# DEPENDENCIES AND IMPORT -------------------------------------------------

# Load dependencies
library(tidyverse)
library(rio)

# Import Adult STRAIN
cmips_strain <- read_csv("data/raw/cmips_strain.csv")

# Import labels
strain_labels <- import("data/raw/cmips_strain_labels.sav")

# EXTRACT STRAIN LABELS ---------------------------------------------------

# Get the variable names
strain_variables <- names(strain_labels[, 7:127])

# Get the attributes
my_labels <- c()

for (i in 1:length(strain_variables)) {
  new_label <- attr(strain_labels[, 7:127][,i], "label")
  my_labels <- c(my_labels, new_label)
}

# Create a dataframe of labels
strain_labels_key <- data.frame(strain_variables, my_labels) %>%
  as_tibble()

# CLEAN OPERATIONS --------------------------------------------------------

# Remove tests runs - retain all stress variables
cmips_strain_full <- cmips_strain[-c(1:3),] %>%
  # Drop variables
  select(-proportionAttnChecksPassed, -Age, -Sex) %>%
  # Rename variables
  rename(email = ID, date = DateComp, duration = MinutesToComplete)

# Select the variables to retain for dissertation (i.e., total severity)
severity_vars <- strain_labels_key %>%
  mutate(severity = if_else(str_detect(my_labels, regex("total severity", ignore_case = TRUE)),
                            1, 0)) %>%
  filter(severity == 1) %>%
  pull(strain_variables)

# Add the total stressor count variable
severity_vars <- c(severity_vars, "StressCT")

# Retain only severity variables
cmips_strain_severity <- cmips_strain_full %>%
  select(email, date, duration, severity_vars)

# EXPORT ------------------------------------------------------------------

# Export all of the data
write_csv(strain_labels_key, "data/util/strain_labels_key.csv")
write_csv(cmips_strain_full, "data/participants/cleaned/cmips_strain_full.csv")
write_csv(cmips_strain_severity, "data/participants/cleaned/cmips_strain_severity.csv")
