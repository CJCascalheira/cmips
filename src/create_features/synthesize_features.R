# SYNTHESIZE FEATURES -----------------------------------------------------

# Author = Cory J. Cascalheira
# Date = 04/19/2024

# The purpose of this script is to calculate the time-based summary variables for
# all of the features. 


# LOAD DEPENDENCIES AND IMPORT --------------------------------------------

# Load libraries
library(tidyverse)

# Import feature sets
cmips_feature_set_00 <- read_csv("data/participants/features/cmips_feature_set_00.csv") %>%
  rename(participant_id = A, timestamp = B, posts_comments = C)

cmips_feature_set_01 <- read_csv("data/participants/features/cmips_feature_set_01.csv")

cmips_feature_set_02 <- read_csv("data/participants/features/cmips_feature_set_02.csv") %>%
  select(-text)

cmips_feature_set_04 <- read_csv("data/participants/features/cmips_feature_set_04.csv") 
