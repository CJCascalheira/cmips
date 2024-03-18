# FEASIBILITY ANALYSIS ----------------------------------------------------

# Author = Cory Cascalheira
# Date = 05/12/2023

# Calculate basic feasibility metrics.

# LIBRARIES AND DATA ------------------------------------------------------

# Load dependencies
library(tidyverse)
library(readxl)

# Import data
participant_tracker <- read_excel("data/participants/Participant_Tracking.xlsx",
                                  sheet = 1)

# ANALYSIS ----------------------------------------------------------------

# Count number of participants retained + missing social media data
participant_tracker %>%
  count(Keep) %>%
  filter(Keep %in% c("YES", "SM_MISS")) %>%
  summarize(retained_sm_miss = sum(n))
