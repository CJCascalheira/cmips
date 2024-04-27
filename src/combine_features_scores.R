# COMBINE FEATURES AND SCORES ---------------------------------------------

# Author = Cory J. Cascalheira
# Date = 04/27/2024

# The purpose of this script is to combine all the features and scores to 
# prepare for the machine learning models. 

# DEPENDENCIES AND IMPORT -------------------------------------------------

# Libraries
library(tidyverse)

# Import data
cmips_features <- read_csv("data/participants/for_analysis/cmips_features.csv")

cmips_surveys <- read_csv("data/participants/for_analysis/cmips_surveys_full.csv") %>%
  rename(participant_id = ParticipantID) %>%
  filter(participant_id %in% cmips_features$participant_id)

# FOR REGRESSION ----------------------------------------------------------

# Select the variables for regression
cmips_regression <- cmips_surveys %>%
  select(
    # Identifier
    participant_id, missing_strain,
    # Covariates
    stress_posting, stress_n_content, high_freq_posting,
    BSMAS_total, SMBS_total, CLCS_total,
    # Outcomes
    StressTH, StressCT, PSS_total, LEC_total, DHEQ_mean, OI_mean, SOER_total,
    IHS_mean
  )

# ...1) Qualtrics Data ----------------------------------------------------

# For Qualtrics, retain all variables
cmips_reg_qualtrics <- cmips_regression %>%
  select(-missing_strain) %>%
  left_join(cmips_features) %>%
  select(participant_id, starts_with("covariate"), everything()) %>%
  select(-StressTH, -StressCT)

# ...2) Adult Strain Data -------------------------------------------------

# For Adult STRAIN, only retain people who completed the survey
cmips_reg_strain <- cmips_regression %>%
  filter(missing_strain == 0) %>%
  select(-missing_strain) %>%
  select(participant_id, stress_posting, stress_n_content, high_freq_posting,
         BSMAS_total, SMBS_total, CLCS_total, StressTH, StressCT) %>%
  left_join(cmips_features) %>%
  select(participant_id, starts_with("covariate"), everything())

# FOR CLASSIFICATION ------------------------------------------------------

# Select the variables for regression
cmips_classification <- cmips_surveys %>%
  select(
    # Identifier
    participant_id, missing_strain,
    # Covariates
    stress_posting, stress_n_content, high_freq_posting,
    BSMAS_total, SMBS_total, CLCS_total,
    # Outcomes
    starts_with("label_")
  )

# ...1) Qualtrics Data ----------------------------------------------------

# For Qualtrics, retain all variables
cmips_class_qualtrics <- cmips_classification %>%
  select(-missing_strain) %>%
  left_join(cmips_features) %>%
  select(participant_id, starts_with("covariate"), everything()) %>%
  select(-label_StressTH, -label_StressCT)

# ...2) Adult Strain Data -------------------------------------------------

# For Adult STRAIN, only retain people who completed the survey
cmips_class_strain <- cmips_classification %>%
  filter(missing_strain == 0) %>%
  select(-missing_strain) %>%
  select(participant_id, stress_posting, stress_n_content, high_freq_posting,
         BSMAS_total, SMBS_total, CLCS_total, label_StressTH, label_StressCT) %>%
  left_join(cmips_features) %>%
  select(participant_id, starts_with("covariate"), everything())

# EXPORT DATA -------------------------------------------------------------

write_csv(cmips_reg_qualtrics, "data/participants/for_analysis/for_models/cmips_reg_qualtrics.csv")
write_csv(cmips_reg_strain, "data/participants/for_analysis/for_models/cmips_reg_strain.csv")
write_csv(cmips_class_qualtrics, "data/participants/for_analysis/for_models/cmips_class_qualtrics.csv")
write_csv(cmips_class_strain, "data/participants/for_analysis/for_models/cmips_class_strain.csv")
