# SYNTHESIZE FEATURES -----------------------------------------------------

# Author = Cory J. Cascalheira
# Date = 04/19/2024

# The purpose of this script is to calculate the time-based summary variables for
# all of the features. 

# LOAD DEPENDENCIES AND IMPORT --------------------------------------------

# Load libraries
library(tidyverse)
library(rjson)

# Import feature sets

# LIWC
cmips_feature_set_00 <- read_csv("data/participants/features/cmips_feature_set_00.csv") %>%
  rename(participant_id = A, timestamp = B, posts_comments = C)

# Behavioral engagement
cmips_feature_set_01 <- read_csv("data/participants/features/cmips_feature_set_01.csv")

# Lexicons
cmips_feature_set_02 <- read_csv("data/participants/features/cmips_feature_set_02.csv") %>%
  select(-text)

# Stress classifier
cmips_feature_set_03 <- read_csv("data/participants/features/cmips_feature_set_03.csv")

# Relax n-grams
cmips_feature_set_03_relax_ngrams <- read_csv("data/util/tensi_strength_data/cmips_df_ngrams.csv")

# Word embeddings
cmips_feature_set_04 <- read_csv("data/participants/features/cmips_feature_set_04.csv") 

# Topic models
cmips_feature_set_05_lda <- read_csv("data/participants/features/cmips_feature_set_05_lda.csv")
cmips_feature_set_05_gsdmm <- read_csv("data/participants/features/cmips_feature_set_05_gsdmm.csv")

# Minority stress classifier
cmips_feature_set_06 <- read_csv("data/participants/features/cmips_feature_set_06.csv") 

# Import additional data
lda_topic_name_df <- read_csv("data/participants/features/lda_topic_name_df.csv")
gsdmm_log <- fromJSON(file="data/participants/features/gsdmm_log.json")



# For LDA topics, need to pivot into wide format and rename topics

# For GSDMM topics, need to organize the top words and assign names




