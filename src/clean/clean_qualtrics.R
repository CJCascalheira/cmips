# CLEAN QUALTRICS SURVEY --------------------------------------------------

# Author: Cory J. Cascalheira
# Created: 03/13/2024

# The purpose of this script is to preprocess the Qualtrics data to prepare
# for analysis. 

# LIBRARIES AND IMPORT DATA -----------------------------------------------

# Load dependencies
library(tidyverse)
library(readxl)
library(Amelia)

# Import participants who passed BDTs
participant_tracker <- read_excel("data/participants/Participant_Tracking.xlsx",
                                  sheet = 1) %>%
  # Ensure email is lowercase
  mutate(Email = tolower(Email))

# Import participant raw data from Qualtrics (with text)
cmips <- read_csv("data/raw/cmips_qualtrics.csv") %>%
  # Ensure email is lowercase
  mutate(email = tolower(email))

# Important Qualtrics with numeric values
cmips_numeric <- read_csv("data/raw/cmips_qualtrics_numeric.csv") %>%
  # Ensure email is lowercase
  mutate(email = tolower(email))

# FILTER OUT BOTS / FRAUD / LOST TO FU ------------------------------------

# Participants retained 
participants_retained <- participant_tracker %>% 
  filter(Keep %in% c("SM_MISS", "YES")) %>%
  select(Keep, ParticipantID, ResponseId, Email)

# Filter Qualtrics survey to keep the retained participants
cmips_qualtrics <- cmips %>%
  filter(email %in% participants_retained$Email) %>%
  # Retain unique emails
  distinct(email, .keep_all = TRUE)

cmips_numeric <- cmips_numeric %>%
  filter(email %in% participants_retained$Email) %>%
  # Retain unique emails
  distinct(email, .keep_all = TRUE)

# SELECT VARIABLES AND CLEAN DEMOGRAPHICS ---------------------------------

# Variables to remove
full_vars <- names(cmips_qualtrics)

remove_vars <- c(full_vars[c(2:3, 5, 7:8, 10:35)], "appendix_a", "appendix_b", 
                 "eng_read", "eng_speak", "height1", "profile_fb", "profile_tw", 
                 "anagram1", "anagram2", "anagram3", "height2", "country", "SC0")

# Remove unnecessary variables
cmips_qualtrics <- cmips_qualtrics %>%
  select(-one_of(remove_vars)) %>% 
  # Remove attention check variables
  select(-BSMAS_4, -LEC1_9, -IHS_8) %>%
  # Rename duration
  rename(duration = `Duration (in seconds)`) 

cmips_numeric <- cmips_numeric %>%
  select(-one_of(remove_vars)) %>% 
  # Remove attention check variables
  select(-BSMAS_4, -LEC1_9, -IHS_8) %>%
  # Rename duration
  rename(duration = `Duration (in seconds)`) 

# Add participants ids
cmips_qualtrics <- participants_retained %>%
  select(ParticipantID, ResponseId) %>%
  left_join(cmips_qualtrics) %>%
  # Reorder the columns
  select(ParticipantID, ResponseId, name, email, phone, zipcode = zipcode1, IPAddress, everything())

# Recode demographics
cmips_qualtrics <- cmips_qualtrics %>%
  mutate(
    # Age as a number
    age = as.numeric(age),
    # Recode race
    race = if_else(str_detect(race, regex(",")), "Multiracial", race),
    race = recode(race, "White or European American" = "White", 
                  "You don’t have an option that describes my race or races (please specify):" = "Other",
                  "Latina/Latino/Latinx or Hispanic" = "Latinx",
                  "Black or African American" = "Black",
                  "Asian or Asian American" = "Asian",
                  "Biracial or Multiracial" = "Multiracial"),
    # Recode gender
    gender = recode(gender, "Man (cisgender)" = "Cis Man", "Male,Man (cisgender)" = "Cis Man",
                    "Male" = "Cis Man", "Woman (cisgender)" = "Cis Woman", 
                    "Female,Woman (cisgender)" = "Cis Woman",
                    "Female,Femme,Woman (cisgender)" = "Cis Woman",
                    "Dyke,Female,Femme,Queer,Woman (cisgender)" = "Cis Woman",
                    "Butch,Dyke,Female,Woman (cisgender)" = "Cis Woman",
                    "Femme,Woman (cisgender)" = "Cis Woman",
                    "Androgynous,Female" = "Other", "Female,Femme" = "Cis Woman",
                    "Androgynous,Changes a lot" = "Other",
                    "Unsure" = "Other",
                    "Male,Two-spirit" = "Other",
                    "Female" = "Cis Woman"),
    gender = if_else(str_detect(gender, regex("enby", ignore_case = TRUE)), 
                     "Nonbinary", gender),
    gender = if_else(str_detect(gender, regex("trans masculine|transgender man", ignore_case = TRUE)), 
                     "Trans Man", gender),
    gender = if_else(str_detect(gender, regex("trans feminine|trans woman", ignore_case = TRUE)), 
                     "Trans Woman", gender),
    gender = if_else(str_detect(gender, regex("You don’t have an option", ignore_case = TRUE)), 
                     "Other", gender),
    gender = if_else(str_detect(gender, regex("queer|fluid", ignore_case = TRUE)), 
                     "Queer / Fluid", gender),
    # Recode sexual orientation
    sex_or = if_else(str_detect(sex_or, regex("questioning", ignore_case = TRUE)), 
                     "Questioning", sex_or),
    sex_or = if_else(str_detect(sex_or, regex("You don’t have an option", ignore_case = TRUE)), 
                     "Other", sex_or),
    sex_or = if_else(str_detect(sex_or, regex("Queer", ignore_case = TRUE)), 
                     "Queer", sex_or),
    sex_or = if_else(str_detect(sex_or, regex(",", ignore_case = TRUE)), 
                     "Multiple Identities", sex_or)
  ) 

# Check work - demographics
table(cmips_qualtrics$race)
table(cmips_qualtrics$gender)
table(cmips_qualtrics$sex_or)

# Identify heterosexual people who got through screening
cmips_str8_folx <- cmips_qualtrics %>%
  filter(sex_or == "Straight or heterosexual") %>%
  select(ParticipantID, ResponseId)

# Remove heterosexual people due to minority stress measures
cmips_qualtrics <- cmips_qualtrics %>%
  filter(sex_or != "Straight or heterosexual")

# RECODE / REVERSE CODING -------------------------------------------------

# Prepare to substitute in the numeric values
cmips_numeric <- cmips_numeric %>%
  select(ResponseId, BSMAS_1:feedback_sm)

# Replace choice-text values with numeric values
cmips_qualtrics <- cmips_qualtrics %>%
  select(-c(BSMAS_1:feedback_sm)) %>%
  left_join(cmips_numeric)

# Fix all the reverse-scored items
cmips_qualtrics <- cmips_qualtrics %>%
  mutate(
    # Perceived Stress Scale
    PSS_2 = recode(PSS_2, `5` = 1, `4` = 2, `3` = 3, `2` = 4, `1` = 5),
    PSS_3 = recode(PSS_3, `5` = 1, `4` = 2, `3` = 3, `2` = 4, `1` = 5)
  )

# Covert scores of measures to numbers
cmips_qualtrics <- cmips_qualtrics %>%
  mutate(across(BSMAS_1:gad7_8, ~ as.numeric(.)))

# CHECK AND FIX SCORING  --------------------------------------------------

# * BSMAS -----------------------------------------------------------------

# Check range of values
temp_scales <- cmips_qualtrics %>%
  select(starts_with("BSMAS"))
temp_scales

for (i in 1:ncol(temp_scales)) {
  print(range(temp_scales[i]))
}

# * SMBS ------------------------------------------------------------------

# Check range of values
temp_scales <- cmips_qualtrics %>%
  select(starts_with("SMBS"))
temp_scales

for (i in 1:ncol(temp_scales)) {
  print(range(temp_scales[i], na.rm = TRUE))
}

# Fix the scoring
names(temp_scales)

smbs_corrected <- participants_retained %>%
  select(ParticipantID, ResponseId) %>%
  left_join(cmips) %>%
  select(ParticipantID, starts_with("SMBS")) %>%
  mutate(across(SMBS_fb_1:SMBS_tw_17, ~ recode(., "Yes" = 1, "No" = 0)))

cmips_qualtrics <- cmips_qualtrics %>%
  select(-starts_with("SMBS")) %>%
  left_join(smbs_corrected)

# * CLCS ------------------------------------------------------------------

# Check range of values
temp_scales <- cmips_qualtrics %>%
  select(starts_with("CLCS"))
temp_scales

# Fix the scoring
names(temp_scales)

clcs_corrected <- participants_retained %>%
  select(ParticipantID, ResponseId) %>%
  left_join(cmips) %>%
  select(ParticipantID, starts_with("CLCS")) %>%
  mutate(across(CLCS_1_1:CLCS_1_8, ~ recode(., "Disagree strongly 1" = "1",
                                            "Agree strongly 4</>" = "4"))) %>%
  mutate(across(CLCS_1_1:CLCS_1_8, ~ as.numeric(.)))

cmips_qualtrics <- cmips_qualtrics %>%
  select(-starts_with("CLCS")) %>%
  left_join(clcs_corrected)

# Now check range of values
temp_scales <- cmips_qualtrics %>%
  select(starts_with("CLCS"))
temp_scales

for (i in 1:ncol(temp_scales)) {
  print(range(temp_scales[i], na.rm = TRUE))
}

# * PSS -------------------------------------------------------------------

# Check range of values
temp_scales <- cmips_qualtrics %>%
  select(starts_with("PSS"))
temp_scales

for (i in 1:ncol(temp_scales)) {
  print(range(temp_scales[i], na.rm = TRUE))
}

# Correct scoring
cmips_qualtrics <- cmips_qualtrics %>%
  mutate(across(PSS_1:PSS_4, ~ recode(., `5` = 4, `4` = 3, `3` = 2, `2` = 1, `1` = 0)))

# Recheck range of values
temp_scales <- cmips_qualtrics %>%
  select(starts_with("PSS"))
temp_scales

for (i in 1:ncol(temp_scales)) {
  print(range(temp_scales[i], na.rm = TRUE))
}

# * LEC -------------------------------------------------------------------

# Check range of values
temp_scales <- cmips_qualtrics %>%
  select(starts_with("lec"))
temp_scales

# Fix the scoring
names(temp_scales)

lec_corrected <- participants_retained %>%
  select(ParticipantID, ResponseId) %>%
  left_join(cmips) %>%
  select(ParticipantID, starts_with("LEC")) %>%
  # Drop the repeated questions
  select(-LEC1_13, -LEC1_14, -LEC1_15, -LEC1_9) %>%
  mutate(across(LEC1_1:LEC2_2, ~ recode(., "Happened to me" = 3, "Witnessed it" = 2,
                                        "Learned about it" = 1, "Not sure" = 0, 
                                        "Doesn’t apply" = 0))) %>%
  mutate(across(LEC1_1:LEC2_2, ~ as.numeric(.)))

cmips_qualtrics <- cmips_qualtrics %>%
  select(-starts_with("LEC")) %>%
  left_join(lec_corrected)

# Recheck range of values
temp_scales <- cmips_qualtrics %>%
  select(starts_with("LEC"))
temp_scales

for (i in 1:ncol(temp_scales)) {
  print(range(temp_scales[i], na.rm = TRUE))
}

# * DHEQ ------------------------------------------------------------------

# Check range of values
temp_scales <- cmips_qualtrics %>%
  select(starts_with("DHEQ"))
temp_scales

for (i in 1:ncol(temp_scales)) {
  print(range(temp_scales[i], na.rm = TRUE))
}

# Fix the scoring
names(temp_scales)

cmips_qualtrics <- cmips_qualtrics %>%
  mutate(across(DHEQ1_1:DHEQ2_25, ~ recode(., `6` = 5, `5` = 4, `4` = 3, `3` = 2,
                                           `2` = 1, `1` = 1)))

# Recheck range of values
temp_scales <- cmips_qualtrics %>%
  select(starts_with("DHEQ"))
temp_scales

for (i in 1:ncol(temp_scales)) {
  print(range(temp_scales[i], na.rm = TRUE))
}

# * OI --------------------------------------------------------------------

# Check range of values
temp_scales <- cmips_qualtrics %>%
  select(starts_with("OI"))
temp_scales

for (i in 1:ncol(temp_scales)) {
  print(range(temp_scales[i], na.rm = TRUE))
}

# Fix scoring
names(temp_scales)

cmips_qualtrics <- cmips_qualtrics %>%
  mutate(across(OI_1:OI_10, ~ recode(., `8` = 7, `7` = 6, `6` = 5, `5` = 4, `4` = 3,
                                     `3` = 2, `2` = 1, `1` = 0)))

# Recheck range of values
temp_scales <- cmips_qualtrics %>%
  select(starts_with("OI"))
temp_scales

for (i in 1:ncol(temp_scales)) {
  print(range(temp_scales[i], na.rm = TRUE))
}

# * SOER ------------------------------------------------------------------

# Check range of values
temp_scales <- cmips_qualtrics %>%
  select(starts_with("SOER"))
temp_scales

for (i in 1:ncol(temp_scales)) {
  print(range(temp_scales[i], na.rm = TRUE))
}

# Fix scoring
names(temp_scales)

cmips_qualtrics <- cmips_qualtrics %>%
  mutate(across(SOER_1:SOER_9, ~ recode(., `5` = 4, `4` = 3, `3` = 2, `2` = 1, `1` = 0)))

# Recheck range of values
temp_scales <- cmips_qualtrics %>%
  select(starts_with("SOER"))
temp_scales

for (i in 1:ncol(temp_scales)) {
  print(range(temp_scales[i], na.rm = TRUE))
}

# * IHS -------------------------------------------------------------------

# Check range of values
temp_scales <- cmips_qualtrics %>%
  select(starts_with("IHS"))
temp_scales

for (i in 1:ncol(temp_scales)) {
  print(range(temp_scales[i], na.rm = TRUE))
}

# * DUDIT -----------------------------------------------------------------

# Check range of values
temp_scales <- cmips_qualtrics %>%
  select(starts_with("DUDIT"))
temp_scales

for (i in 1:ncol(temp_scales)) {
  print(range(temp_scales[i], na.rm = TRUE))
}

# Fix scoring
names(temp_scales)

dudit_corrected <- participants_retained %>%
  select(ParticipantID, ResponseId) %>%
  left_join(cmips) %>%
  select(ParticipantID, starts_with("DUDIT")) %>% 
  mutate(across(DUDIT_1_1:DUDIT_1_2, ~ recode(., "Never" = 0, 
                                              "Once a month or less often" = 1,
                                              "2-4 times a month" = 2,
                                              "2-3 times a week" = 3,
                                              "4 times a week or more often" = 4))) %>%
  mutate(DUDIT_3 = recode(DUDIT_3, "0" = "0", "1-2" = "1", "3-4" = "2", 
                          "5-6" = "3", "7 or more" = "4")) %>%
  mutate(across(DUDIT_4_1:DUDIT_4_6, ~ recode(., "Never" = 0,
                                              "Less often than once a month" = 1,
                                              "Every month" = 2,
                                              "Every week" = 3,
                                              "Daily or almost every day" = 4))) %>%
  mutate(across(DUDIT_10_1:DUDIT_10_2, ~ recode(., "Never" = 0, 
                                                "Yes, but not over the past year" = 2,
                                                "Yes, over the past year" = 4))) %>%
  mutate(across(DUDIT_1_1:DUDIT_10_2, ~ as.numeric(.)))

cmips_qualtrics <- cmips_qualtrics %>%
  select(-starts_with("DUDIT")) %>%
  left_join(dudit_corrected)

# Recheck range of values
temp_scales <- cmips_qualtrics %>%
  select(starts_with("DUDIT"))
temp_scales

for (i in 1:ncol(temp_scales)) {
  print(range(temp_scales[i], na.rm = TRUE))
}

# * AUDIT -----------------------------------------------------------------

# Check range of values
temp_scales <- cmips_qualtrics %>%
  select(starts_with("AUDIT"))
temp_scales

for (i in 1:ncol(temp_scales)) {
  print(range(temp_scales[i], na.rm = TRUE))
}

# Fix scoring
names(temp_scales)

audit_corrected <- participants_retained %>%
  select(ParticipantID, ResponseId) %>%
  left_join(cmips) %>%
  select(ParticipantID, starts_with("AUDIT")) %>% 
  mutate(AUDIT_1_1 = recode(AUDIT_1_1, "Never" = 0, 
                          "Monthly or less" = 1,
                          "2-4 times a month" = 2,
                          "2-3 times a week" = 3,
                          "4 or more times a week" = 4)) %>%
  mutate(AUDIT_2 = recode(AUDIT_2, "1 or 2" = "0", "3 or 4" = "1", "5 or 6" = "2", 
                          "7 to 9" = "3", "10 or more" = "4")) %>%
  mutate(across(AUDIT_3_1:AUDIT_3_6, ~ recode(., "Never" = 0,
                                              "Less than monthly" = 1,
                                              "Monthly" = 2,
                                              "Weekly" = 3,
                                              "Daily or almost daily" = 4))) %>%
  mutate(across(AUDIT_9:AUDIT_10, ~ recode(., "No" = 0, 
                                           "Yes, but not in the last year" = 2,
                                           "Yes, during the last year" = 4))) %>%
  mutate(across(AUDIT_1_1:AUDIT_10, ~ as.numeric(.)))

cmips_qualtrics <- cmips_qualtrics %>%
  select(-starts_with("AUDIT")) %>%
  left_join(audit_corrected)

# Recheck range of values
temp_scales <- cmips_qualtrics %>%
  select(starts_with("AUDIT"))
temp_scales

for (i in 1:ncol(temp_scales)) {
  print(range(temp_scales[i], na.rm = TRUE))
}

# * PHQ-9 -----------------------------------------------------------------

# Check range of values
temp_scales <- cmips_qualtrics %>%
  select(starts_with("phq"))
temp_scales

for (i in 1:ncol(temp_scales)) {
  print(range(temp_scales[i], na.rm = TRUE))
}

# Fix scoring
cmips_qualtrics <- cmips_qualtrics %>%
  mutate(across(phq9_1_1:phq9_1_9, ~ recode(., `4` = 3, `3` = 2, `2` = 1, `1` = 0)))

# Recheck range of values
temp_scales <- cmips_qualtrics %>%
  select(starts_with("phq"))
temp_scales

for (i in 1:ncol(temp_scales)) {
  print(range(temp_scales[i], na.rm = TRUE))
}

# * GAD-7 -----------------------------------------------------------------

# Check range of values
temp_scales <- cmips_qualtrics %>%
  select(starts_with("gad"))
temp_scales

for (i in 1:ncol(temp_scales)) {
  print(range(temp_scales[i], na.rm = TRUE))
}

# Fix scoring
cmips_qualtrics <- cmips_qualtrics %>%
  mutate(across(gad7_1_1:gad7_1_7, ~ recode(., `4` = 3, `3` = 2, `2` = 1, `1` = 0)))

# Recheck range of values
temp_scales <- cmips_qualtrics %>%
  select(starts_with("gad"))
temp_scales

for (i in 1:ncol(temp_scales)) {
  print(range(temp_scales[i], na.rm = TRUE))
}

# * SMBS ------------------------------------------------------------------

# Check range of values
temp_scales <- cmips_qualtrics %>%
  select(starts_with("SMBS"))
temp_scales

for (i in 1:ncol(temp_scales)) {
  print(range(temp_scales[i], na.rm = TRUE))
}

# Fix scoring
cmips_qualtrics <- cmips_qualtrics %>%
  mutate(across(SMBS_fb_1:SMBS_tw_17, ~ recode(., `3` = 0, `2` = 1)))

# Recheck range of values
temp_scales <- cmips_qualtrics %>%
  select(starts_with("SMBS"))
temp_scales

for (i in 1:ncol(temp_scales)) {
  print(range(temp_scales[i], na.rm = TRUE))
}

# * Stress-Related Posts --------------------------------------------------

# Recode
cmips_qualtrics <- cmips_qualtrics %>%
  mutate(sm_post = recode(sm_post, "Yes" = 1, "No" = 0))

# MISSING DATA ------------------------------------------------------------

# Select the scales
cmips_scales <- cmips_qualtrics %>%
  # Select just the measures
  select(ParticipantID, BSMAS_1:gad7_8, SMBS_fb_1:AUDIT_10) %>%
  # Remove variables with anticipated missing data
  select(-starts_with("SMBS_"))

# Check variables seleted
names(cmips_scales)

# Check for missing data in measures
complete.cases(cmips_scales)

# Number of participants with missing data
nrow(cmips_scales) - sum(complete.cases(cmips_scales))

# Find which variables have missing data
is_missing <- cmips_scales %>%
  pivot_longer(cols = BSMAS_1:AUDIT_10, names_to = "variables", values_to = "values") %>%
  mutate(is_missing = if_else(is.na(values), 1, 0)) %>%
  filter(is_missing == 1)
is_missing

# Percent of missing data for these participants
is_missing %>%
  count(ParticipantID) %>%
  mutate(total_items = ncol(cmips_scales) - 1) %>%
  mutate(percent = (n / total_items) * 100)

# Replace missing data with multiple imputation
cmips_imputed <- cmips_scales %>%
  select(ParticipantID, starts_with("AUDIT")) %>%
  as.data.frame() %>%
  amelia(idvars = "ParticipantID")

# Get the final imputed dataset
cmips_imputed_final <- cmips_imputed$imputations[[5]] %>%
  as_tibble()

# Merge the imputed values with the original data
cmips_qualtrics <- cmips_qualtrics %>%
  # Remove variables for which imputation was conducted
  select(-starts_with("AUDIT")) %>%
  # Add the imputed valyes
  left_join(cmips_imputed_final)

cmips_scales <- cmips_scales %>%
  # Remove variables for which imputation was conducted
  select(-starts_with("AUDIT")) %>%
  # Add the imputed valyes
  left_join(cmips_imputed_final)

# Check work - should be 100% complete cases
sum(complete.cases(cmips_scales)) / nrow(cmips_qualtrics)

# SCORE MEASURES ----------------------------------------------------------

# * BSMAS -----------------------------------------------------------------

# Calculate total score
bsmas_total <- cmips_qualtrics %>%
  select(ParticipantID, starts_with("BSMAS")) %>%
  pivot_longer(cols = BSMAS_1:BSMAS_7, names_to = "measure", values_to = "scores") %>%
  group_by(ParticipantID) %>%
  summarize(BSMAS_total = sum(scores))
bsmas_total

# * SMBS ------------------------------------------------------------------

# No total score calculated

# * CLCS ------------------------------------------------------------------

# Calculate total score
clcs_total <- cmips_qualtrics %>%
  select(ParticipantID, starts_with("CLCS")) %>%
  pivot_longer(cols = CLCS_1_1:CLCS_1_8, names_to = "measure", values_to = "scores") %>%
  group_by(ParticipantID) %>%
  summarize(CLCS_total = sum(scores))
clcs_total

# * PSS -------------------------------------------------------------------

# Calculate total score
pss_total <- cmips_qualtrics %>%
  select(ParticipantID, starts_with("PSS")) %>%
  pivot_longer(cols = PSS_1:PSS_4, names_to = "measure", values_to = "scores") %>%
  group_by(ParticipantID) %>%
  summarize(PSS_total = sum(scores))
pss_total

# * LEC -------------------------------------------------------------------

# Calculate total score
lec_total <- cmips_qualtrics %>%
  select(ParticipantID, starts_with("LEC")) %>%
  pivot_longer(cols = LEC1_1:LEC2_2, names_to = "measure", values_to = "scores") %>%
  group_by(ParticipantID) %>%
  summarize(LEC_total = sum(scores))
lec_total

# * DHEQ ------------------------------------------------------------------

# Calculate mean distress score
dheq_mean <- cmips_qualtrics %>%
  select(ParticipantID, starts_with("DHEQ")) %>%
  pivot_longer(cols = DHEQ1_1:DHEQ2_25, names_to = "measure", values_to = "scores") %>%
  group_by(ParticipantID) %>%
  summarize(DHEQ_mean = mean(scores))
dheq_mean

# * OI --------------------------------------------------------------------

# Calculate mean outness score
oi_mean <- cmips_qualtrics %>%
  select(ParticipantID, starts_with("OI")) %>%
  pivot_longer(cols = OI_1:OI_10, names_to = "measure", values_to = "scores") %>%
  group_by(ParticipantID) %>%
  summarize(OI_mean = mean(scores))
oi_mean

# * SOER ------------------------------------------------------------------

# Calculate total score
soer_total <- cmips_qualtrics %>%
  select(ParticipantID, starts_with("SOER")) %>%
  pivot_longer(cols = SOER_1:SOER_9, names_to = "measure", values_to = "scores") %>%
  group_by(ParticipantID) %>%
  summarize(SOER_total = sum(scores))
soer_total

# * IHS -------------------------------------------------------------------

# Calculate mean score
ihs_mean <- cmips_qualtrics %>%
  select(ParticipantID, starts_with("IHS")) %>%
  pivot_longer(cols = IHS_1:IHS_10, names_to = "measure", values_to = "scores") %>%
  group_by(ParticipantID) %>%
  summarize(IHS_mean = mean(scores))
ihs_mean

# * DUDIT -----------------------------------------------------------------

# Calculate total score
dudit_total <- cmips_qualtrics %>%
  select(ParticipantID, starts_with("DUDIT")) %>%
  pivot_longer(cols = DUDIT_1_1:DUDIT_10_2, names_to = "measure", values_to = "scores") %>%
  group_by(ParticipantID) %>%
  summarize(DUDIT_total = sum(scores))
dudit_total

# * AUDIT -----------------------------------------------------------------

# Calculate total score
audit_total <- cmips_qualtrics %>%
  select(ParticipantID, starts_with("AUDIT")) %>%
  pivot_longer(cols = AUDIT_1_1:AUDIT_10, names_to = "measure", values_to = "scores") %>%
  group_by(ParticipantID) %>%
  summarize(AUDIT_total = sum(scores))
audit_total

# * PHQ-9 -----------------------------------------------------------------

# Calculate total score
phq_total <- cmips_qualtrics %>%
  select(ParticipantID, starts_with("phq")) %>%
  pivot_longer(cols = phq9_1_1:phq9_1_9, names_to = "measure", values_to = "scores") %>%
  group_by(ParticipantID) %>%
  summarize(PHQ9_total = sum(scores))
phq_total

# * GAD-7 -----------------------------------------------------------------

# Calculate total score
gad_total <- cmips_qualtrics %>%
  select(ParticipantID, starts_with("gad")) %>%
  pivot_longer(cols = gad7_1_1:gad7_1_7, names_to = "measure", values_to = "scores") %>%
  group_by(ParticipantID) %>%
  summarize(GAD7_total = sum(scores))
gad_total

# * SMBS ------------------------------------------------------------------

# Extract the data for FB and TW separately
SMBS_fb <- cmips_qualtrics %>%
  select(ParticipantID, starts_with("SMBS_fb"))

SMBS_tw <- cmips_qualtrics %>%
  select(ParticipantID, starts_with("SMBS_tw"))

# Create new variable names
new_smbs_names = c("ParticipantID")

for (i in 1:17) {
  new_smbs_names = c(new_smbs_names, paste0("SMBS_", i))
}

# Rename the columns
names(SMBS_fb) <- new_smbs_names
names(SMBS_tw) <- new_smbs_names

# Score the SMBS variable
SMBS_total <- bind_rows(SMBS_fb, SMBS_tw) %>%
  pivot_longer(cols = SMBS_1:SMBS_17, names_to = "SMBS", values_to = "values") %>%
  group_by(ParticipantID, SMBS) %>%
  summarize(total = sum(values, na.rm = TRUE)) %>% 
  mutate(total = if_else(total == 2, 1, total)) %>%
  ungroup() %>%
  group_by(ParticipantID) %>%
  summarize(SMBS_total = sum(total))

# * Stress-Related Posts --------------------------------------------------

# Already scored, but change variable name
cmips_qualtrics <- cmips_qualtrics %>%
  mutate(stress_posting = sm_post)

# * Stress-Related Content ------------------------------------------------

# Score as a count
stress_n_content <- cmips_qualtrics %>%
  select(ParticipantID, sm_conten) %>%
  mutate(
    stress_n_content = str_count(sm_conten, ","),
    stress_n_content = stress_n_content + 1,
    stress_n_content = if_else(is.na(stress_n_content), 0, stress_n_content)
  ) %>%
  select(-sm_conten)

# * Frequency of Posting --------------------------------------------------

# Recode and then create the variable
high_freq_posting <- cmips_qualtrics %>%
  mutate(
    freq_fb = recode(sm_freq_fb, "I have Facebook, but never login" = 0,
                     "Yearly or less" = 0, "Every few months" = 0, 
                     "Monthly" = 0, "Weekly" = 0, "Once a day" = 0,
                     "Multiple times a day" = 1),
    freq_tw = recode(sm_freq_tw, "I have Facebook, but never login" = 0,
                     "Yearly or less" = 0, "Every few months" = 0, 
                     "Monthly" = 0, "Weekly" = 0, "Once a day" = 0,
                     "Multiple times a day" = 1)
  ) %>%
  select(ParticipantID, freq_fb, freq_tw) %>%
  mutate(
    freq_fb = if_else(is.na(freq_fb), 0, freq_fb),
    freq_tw = if_else(is.na(freq_tw), 0, freq_tw)
  ) %>%
  unite(col = "high_freq_posting", freq_fb:freq_tw, sep = "") %>%
  mutate(high_freq_posting = if_else(high_freq_posting == "00", 0, 1))

# * COMBINE ALL AGGREGATE SCORES ------------------------------------------

# Combine the scores
cmips_qualtrics <- left_join(cmips_qualtrics, bsmas_total) %>%
  left_join(clcs_total) %>%
  left_join(pss_total) %>%
  left_join(lec_total) %>%
  left_join(dheq_mean) %>%
  left_join(oi_mean) %>%
  left_join(soer_total) %>%
  left_join(ihs_mean) %>%
  left_join(dudit_total) %>%
  left_join(audit_total) %>%
  left_join(phq_total) %>%
  left_join(gad_total) %>%
  left_join(stress_n_content) %>%
  left_join(high_freq_posting) %>%
  left_join(SMBS_total)

# EXPORT DATA -------------------------------------------------------------

# Export the cleaned Qualtrics survey
write_csv(cmips_qualtrics, "data/participants/cleaned/cmips_qualtrics.csv")

# Export the heterosexual folx who we removed
write_csv(cmips_str8_folx, "data/participants/util/cmips_str8_folx.csv")
