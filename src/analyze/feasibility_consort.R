# FEASIBILITY ANALYSIS ----------------------------------------------------

# Author = Cory Cascalheira
# Date = 03/23/2023

# Calculate basic feasibility metrics and produce consort diagram.

# Resources
# - https://cran.r-project.org/web/packages/consort/vignettes/consort_diagram.html

# LIBRARIES AND DATA ------------------------------------------------------

# Load dependencies
library(tidyverse)
library(readxl)

# Import data
participant_tracker <- read_excel("data/participants/Participant_Tracking.xlsx",
                                  sheet = 1)

cimps_surveys <- read_csv("data/participants/for_analysis/cmips_surveys.csv")

# TRACK ENROLLMENT / COMPLETION -------------------------------------------

# Count number of participants retained + missing social media data
participant_tracker %>%
  count(Keep) %>%
  filter(Keep %in% c("YES", "SM_MISS")) %>%
  summarize(retained_sm_miss = sum(n))
