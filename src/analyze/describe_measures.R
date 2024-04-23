# DESCRIBE MEASURES -------------------------------------------------------

# Author = Cory Cascalheira
# Date = 04/19/2024

# The purpose of this script is to calculate descriptive statistics and 
# reliability evidence of the psychometric surveys.

# References
# https://search.r-project.org/CRAN/refmans/sasLM/html/KurtosisSE.html
# https://search.r-project.org/CRAN/refmans/sur/html/se.skew.html

# LOAD DEPENDENCIES AND IMPORT --------------------------------------------

# Load libraries 
source("util/ci.reliability.R")
library(tidyverse)
library(psych)
library(readxl)
library(sur) # SE of skew
library(sasLM) # SE of kurtosis 

# Import data 
shared_wrong_data <- read_csv("data/participants/util/shared_wrong_data.csv")

participant_tracker <- read_excel("data/participants/Participant_Tracking.xlsx",
                                  sheet = 1) %>%
  filter(!is.na(ResponseId)) %>%
  filter(Keep == "YES") %>%
  filter(!(ParticipantID %in% shared_wrong_data$participant_id))

cmips_surveys <- read_csv("data/participants/for_analysis/cmips_surveys_anonymous_dissertation.csv") %>%
  filter(ParticipantID %in% participant_tracker$ParticipantID)

# INTERNAL CONSISTENCY EVIDENCE -------------------------------------------

# PSS 
cmips_int_con <- cmips_surveys %>%
  select(starts_with("PSS")) %>%
  select(-ends_with("total"), -ends_with("mean"))

ci.reliability(data = cmips_int_con, type = "omega", conf.level = 0.95, 
               interval.type = "bca", B = 1000)

alpha(cmips_int_con)

# LEC
cmips_int_con <- cmips_surveys %>%
  select(starts_with("LEC")) %>%
  select(-ends_with("total"), -ends_with("mean"))

ci.reliability(data = cmips_int_con, type = "omega", conf.level = 0.95, 
               interval.type = "bca", B = 1000)

alpha(cmips_int_con)
omega(cmips_int_con)

# DHEQ
cmips_int_con <- cmips_surveys %>%
  select(starts_with("DHEQ")) %>%
  select(-ends_with("total"), -ends_with("mean"))

ci.reliability(data = cmips_int_con, type = "omega", conf.level = 0.95, 
               interval.type = "bca", B = 1000)

alpha(cmips_int_con)
omega(cmips_int_con)

# OI
cmips_int_con <- cmips_surveys %>%
  select(starts_with("OI")) %>%
  select(-ends_with("total"), -ends_with("mean"))

ci.reliability(data = cmips_int_con, type = "omega", conf.level = 0.95, 
               interval.type = "bca", B = 1000)

alpha(cmips_int_con)
omega(cmips_int_con)

# SOER
cmips_int_con <- cmips_surveys %>%
  select(starts_with("SOER")) %>%
  select(-ends_with("total"), -ends_with("mean"))

ci.reliability(data = cmips_int_con, type = "omega", conf.level = 0.95, 
               interval.type = "bca", B = 1000)

alpha(cmips_int_con)
omega(cmips_int_con)

# IHS
cmips_int_con <- cmips_surveys %>%
  select(starts_with("IHS")) %>%
  select(-ends_with("total"), -ends_with("mean"))

ci.reliability(data = cmips_int_con, type = "omega", conf.level = 0.95, 
               interval.type = "bca", B = 1000)

alpha(cmips_int_con)
omega(cmips_int_con)

# BSMAS
cmips_int_con <- cmips_surveys %>%
  select(starts_with("BSMAS")) %>%
  select(-ends_with("total"), -ends_with("mean"))

ci.reliability(data = cmips_int_con, type = "omega", conf.level = 0.95, 
               interval.type = "bca", B = 1000)

alpha(cmips_int_con)
omega(cmips_int_con)

# SMBS
# Not calculated because only two items were used

# CLCS
cmips_int_con <- cmips_surveys %>%
  select(starts_with("CLCS")) %>%
  select(-ends_with("total"), -ends_with("mean"))

ci.reliability(data = cmips_int_con, type = "omega", conf.level = 0.95, 
               interval.type = "bca", B = 1000)

alpha(cmips_int_con)
omega(cmips_int_con)

# CORRELATION TABLES AND DESCRIPTIVE STATS --------------------------------

# How many participants said they post about stressful exps on social media?
cmips_surveys %>%
  count(stress_posting) %>%
  mutate(percent = (n / nrow(cmips_surveys)) * 100)

# Kinds of stressful experiences posted about
sm_content_df <- data.frame(sm_content = unlist(
  str_split(cmips_surveys$sm_conten, ","))
  ) %>%
  count(sm_content) %>%
  as_tibble() %>%
  arrange(desc(n)) %>%
  mutate(percent = (n / 51) * 100)

# Note that politics was in the survey twice, so only count it as one
sm_content_df

# Descriptive statistics
describe(cmips_surveys$StressTH)
describe(cmips_surveys$StressCT)
describe(cmips_surveys$PSS_total)
describe(cmips_surveys$LEC_total)
describe(cmips_surveys$DHEQ_mean)
describe(cmips_surveys$OI_mean)
describe(cmips_surveys$SOER_total)
describe(cmips_surveys$IHS_mean)
describe(cmips_surveys$stress_n_content)
describe(cmips_surveys$high_freq_posting)
describe(cmips_surveys$BSMAS_total)
describe(cmips_surveys$SMBS_total)
describe(cmips_surveys$CLCS_total)

# Skewness to SE of skewness ratios
skew(cmips_surveys$StressTH) / se.skew(cmips_surveys$StressTH)
skew(cmips_surveys$StressCT) / se.skew(cmips_surveys$StressCT)
skew(cmips_surveys$PSS_total) / se.skew(cmips_surveys$PSS_total)
skew(cmips_surveys$LEC_total) / se.skew(cmips_surveys$LEC_total)
skew(cmips_surveys$DHEQ_mean) / se.skew(cmips_surveys$DHEQ_mean)
skew(cmips_surveys$OI_mean) / se.skew(cmips_surveys$OI_mean)
skew(cmips_surveys$SOER_total) / se.skew(cmips_surveys$SOER_total)
skew(cmips_surveys$IHS_mean) / se.skew(cmips_surveys$IHS_mean)
skew(cmips_surveys$stress_n_content) / se.skew(cmips_surveys$stress_n_content)
skew(cmips_surveys$high_freq_posting) / se.skew(cmips_surveys$high_freq_posting)
skew(cmips_surveys$BSMAS_total) / se.skew(cmips_surveys$BSMAS_total)
skew(cmips_surveys$SMBS_total) / se.skew(cmips_surveys$SMBS_total)
skew(cmips_surveys$CLCS_total) / se.skew(cmips_surveys$CLCS_total)

# Kurtosis to SE of kurtosis ratios
kurtosi(cmips_surveys$StressTH) /  KurtosisSE(cmips_surveys$StressTH)
kurtosi(cmips_surveys$StressCT) /  KurtosisSE(cmips_surveys$StressCT)
kurtosi(cmips_surveys$PSS_total) /  KurtosisSE(cmips_surveys$PSS_total)
kurtosi(cmips_surveys$LEC_total) /  KurtosisSE(cmips_surveys$LEC_total)
kurtosi(cmips_surveys$DHEQ_mean) /  KurtosisSE(cmips_surveys$DHEQ_mean)
kurtosi(cmips_surveys$OI_mean) /  KurtosisSE(cmips_surveys$OI_mean)
kurtosi(cmips_surveys$SOER_total) /  KurtosisSE(cmips_surveys$SOER_total)
kurtosi(cmips_surveys$IHS_mean) /  KurtosisSE(cmips_surveys$IHS_mean)
kurtosi(cmips_surveys$stress_n_content) /  KurtosisSE(cmips_surveys$stress_n_content)
kurtosi(cmips_surveys$high_freq_posting) /  KurtosisSE(cmips_surveys$high_freq_posting)
kurtosi(cmips_surveys$BSMAS_total) /  KurtosisSE(cmips_surveys$BSMAS_total)
kurtosi(cmips_surveys$SMBS_total) /  KurtosisSE(cmips_surveys$SMBS_total)
kurtosi(cmips_surveys$CLCS_total) /  KurtosisSE(cmips_surveys$CLCS_total)

# CLASS BALANCE OF OUTCOMES -----------------------------------------------


# CONVERGENT VALIDITY SOER ------------------------------------------------

# Test whether scores on the Negative Expectations subscale are correlated with 
# scores on the DHEQ, OI, and IHS with a strength of ≥ .50 (Swank & Mullen, 2017). 
