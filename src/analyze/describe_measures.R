# DESCRIBE MEASURES -------------------------------------------------------

# Author = Cory Cascalheira
# Date = 04/19/2024

# The purpose of this script is to calculate descriptive statistics and 
# reliability evidence of the psychometric surveys.

# LOAD DEPENDENCIES AND IMPORT --------------------------------------------

# Load libraries 
source("util/ci.reliability.R")
library(tidyverse)
library(psych)
library(readxl)

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

# DHEQ
cmips_int_con <- cmips_surveys %>%
  select(starts_with("DHEQ")) %>%
  select(-ends_with("total"), -ends_with("mean"))

ci.reliability(data = cmips_int_con, type = "omega", conf.level = 0.95, 
               interval.type = "bca", B = 1000)

alpha(cmips_int_con)

# OI
cmips_int_con <- cmips_surveys %>%
  select(starts_with("OI")) %>%
  select(-ends_with("total"), -ends_with("mean"))

ci.reliability(data = cmips_int_con, type = "omega", conf.level = 0.95, 
               interval.type = "bca", B = 1000)

alpha(cmips_int_con)

# SOER
cmips_int_con <- cmips_surveys %>%
  select(starts_with("SOER")) %>%
  select(-ends_with("total"), -ends_with("mean"))

ci.reliability(data = cmips_int_con, type = "omega", conf.level = 0.95, 
               interval.type = "bca", B = 1000)

alpha(cmips_int_con)

# IHS
cmips_int_con <- cmips_surveys %>%
  select(starts_with("IHS")) %>%
  select(-ends_with("total"), -ends_with("mean"))

ci.reliability(data = cmips_int_con, type = "omega", conf.level = 0.95, 
               interval.type = "bca", B = 1000)

alpha(cmips_int_con)

# CLASS BALANCE OF OUTCOMES -----------------------------------------------


# CORRELATION TABLES AND DESCRIPTIVE STATS --------------------------------


# CONVERGENT VALIDITY SOER ------------------------------------------------

# Test whether scores on the Negative Expectations subscale are correlated with 
# scores on the DHEQ, OI, and IHS with a strength of ≥ .50 (Swank & Mullen, 2017). 
