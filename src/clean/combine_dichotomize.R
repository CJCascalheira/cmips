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
cmips_str8_folx <- read_csv("data/participants/util/cmips_str8_folx.csv")

# COMBINE SURVEYS ---------------------------------------------------------

# Combine the surveys
cmips_surveys_full <- left_join(cmips_qualitrics, cmips_strain, by = "email") %>%
  filter(ParticipantID != cmips_str8_folx$ParticipantID) %>%
  distinct(ParticipantID, .keep_all = TRUE) %>%
  # Remove some variables
  select(-starts_with("duration"), -date)

# Select only the variables for the dissertation, removing identifying data
cmips_surveys_anonymous_dissertation <- cmips_surveys_full %>%
  # Remove identifying data
  select(-name, -email, -phone, -IPAddress, -zipcode, -zipcode2) %>%
  # Remove qualitative responses
  select(-goal, -recruit, -feedback_gen, -feedback_sm) %>%
  # Remove fill-in-response demographics
  select(-ends_with("TEXT")) %>%
  # Remove clinical outcome variables
  select(-starts_with("AUDIT"), -starts_with("DUDIT"), -starts_with("PHQ9"),
         -starts_with("GAD7"))

# DICHOTOMIZE OUTCOMES FOR CLASSIFICATION ---------------------------------

# Coming soon

# EXPORT ------------------------------------------------------------------

# Save the file
write_csv(cmips_surveys_full, 
          "data/participants/for_analysis/cmips_surveys_full.csv")

write_csv(cmips_surveys_anonymous_dissertation, 
          "data/participants/for_analysis/cmips_surveys_anonymous_dissertation.csv")
