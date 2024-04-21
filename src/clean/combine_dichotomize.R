# COMBINE PSYCHOMETRIC SURVEYS AND DICHOTOMIZE ----------------------------

# Author = Cory Cascalheira
# Date = 03/23/2024

# Combine the psychometric surveys. Create classification labels. Create 
# demographic covariates for non-response bias modeling.

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
  select(-starts_with("duration"), -date) %>%
  # Add variable for missing Adult STRAIN
  mutate(missing_strain = if_else(is.na(StressCT), 1, 0))

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

# ...1) Select Vars to Dichotomize ----------------------------------------

# Names of variables
names(cmips_surveys_anonymous_dissertation)

# Select the variables
cmips_stress_outcomes <- cmips_surveys_anonymous_dissertation %>%
  select(ParticipantID, missing_strain, PSS_total, StressCT, StressTH, LEC_total, DHEQ_mean,
         OI_mean, SOER_total, IHS_mean)

# ...2) Dichotomize -------------------------------------------------------

# Stressful life events - total count
grand_mean_StressCT <- mean(cmips_stress_outcomes$StressCT, na.rm = TRUE)

cmips_stress_outcomes <- cmips_stress_outcomes %>%
  mutate(label_StressCT = if_else(StressCT >= grand_mean_StressCT, 1, 0))

# Stressful life events - severity
grand_mean_StressTH <- mean(cmips_stress_outcomes$StressTH, na.rm = TRUE)

cmips_stress_outcomes <- cmips_stress_outcomes %>%
  mutate(label_StressTH = if_else(StressTH >= grand_mean_StressTH, 1, 0))

# Perceived stress
grand_mean_PSS_total <- mean(cmips_stress_outcomes$PSS_total, na.rm = TRUE)

cmips_stress_outcomes <- cmips_stress_outcomes %>%
  mutate(label_PSS_total = if_else(PSS_total >= grand_mean_PSS_total, 1, 0))

# Potentially traumatic life events
grand_mean_LEC_total <- mean(cmips_stress_outcomes$LEC_total, na.rm = TRUE)

cmips_stress_outcomes <- cmips_stress_outcomes %>%
  mutate(label_LEC_total = if_else(LEC_total >= grand_mean_LEC_total, 1, 0))

# Prejudiced events
grand_mean_DHEQ_mean <- mean(cmips_stress_outcomes$DHEQ_mean, na.rm = TRUE)

cmips_stress_outcomes <- cmips_stress_outcomes %>%
  mutate(label_DHEQ_mean = if_else(DHEQ_mean >= grand_mean_DHEQ_mean, 1, 0))

# Identity concealment
grand_mean_OI_mean <- mean(cmips_stress_outcomes$OI_mean, na.rm = TRUE)

cmips_stress_outcomes <- cmips_stress_outcomes %>%
  mutate(label_OI_mean = if_else(OI_mean >= grand_mean_OI_mean, 1, 0))

# Expected rejection 
grand_mean_SOER_total <- mean(cmips_stress_outcomes$SOER_total, na.rm = TRUE)

cmips_stress_outcomes <- cmips_stress_outcomes %>%
  mutate(label_SOER_total = if_else(SOER_total >= grand_mean_SOER_total, 1, 0))

# Internalized homonegativity 
grand_mean_IHS_mean <- mean(cmips_stress_outcomes$IHS_mean, na.rm = TRUE)

cmips_stress_outcomes <- cmips_stress_outcomes %>%
  mutate(label_IHS_mean = if_else(IHS_mean >= grand_mean_IHS_mean, 1, 0))

# ...3) Add Binary Labels to Dataframes -----------------------------------

# Select the binary labels
cmips_stress_labels <- cmips_stress_outcomes %>%
  select(ParticipantID, starts_with("label"))

# Add the dataframes
cmips_surveys_full <- left_join(cmips_surveys_full, cmips_stress_labels)

cmips_surveys_anonymous_dissertation <- left_join(cmips_surveys_anonymous_dissertation, cmips_stress_labels)

# ...4) Create Demographic Covariates -------------------------------------

# Create covariates - full data
cmips_surveys_full <- cmips_surveys_full %>% 
  mutate(
    is_queer = if_else(sex_or == "Queer", 1, 0),
    is_trans = if_else(str_detect(gender, regex("Cis", ignore_case = TRUE)), 0, 1),
    is_bipoc = if_else(race != "White", 1, 0)
  )

# Create covariates - anon data
cmips_surveys_anonymous_dissertation <- cmips_surveys_anonymous_dissertation %>%
  mutate(
    is_queer = if_else(sex_or == "Queer", 1, 0),
    is_trans = if_else(str_detect(gender, regex("cisgender", ignore_case = TRUE)), 0, 1),
    is_bipoc = if_else(race != "White", 1, 0)
  )

# EXPORT ------------------------------------------------------------------

# Save the file
write_csv(cmips_surveys_full, 
          "data/participants/for_analysis/cmips_surveys_full.csv")

write_csv(cmips_surveys_anonymous_dissertation, 
          "data/participants/for_analysis/cmips_surveys_anonymous_dissertation.csv")
