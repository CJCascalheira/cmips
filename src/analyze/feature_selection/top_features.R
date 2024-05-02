# ORGANIZE TOP FEATURES ---------------------------------------------------

# Author = Cory J. Cascalheira
# Date = 05/01/2024

# The purpose of this script is to organize the top features for each psycho-
# social stressor.

# LOAD DEPENDENCIES AND IMPORT --------------------------------------------

# Libraries
library(tidyverse)
library(janitor)

# Import data
feats_stressth <- read_csv("data/participants/for_analysis/for_models/reduced_features/feats_stressth.csv") %>%
  select(-...1) %>%
  names()

feats_stressct <- read_csv("data/participants/for_analysis/for_models/reduced_features/feats_stressct.csv") %>%
  select(-...1) %>%
  names()

feats_pss <- read_csv("data/participants/for_analysis/for_models/reduced_features/feats_pss.csv") %>%
  select(-...1) %>%
  names()

feats_lec <- read_csv("data/participants/for_analysis/for_models/reduced_features/feats_lec.csv") %>%
  select(-...1) %>%
  names()

feats_dheq <- read_csv("data/participants/for_analysis/for_models/reduced_features/feats_dheq.csv") %>%
  select(-...1) %>%
  names()

feats_oi <- read_csv("data/participants/for_analysis/for_models/reduced_features/feats_oi.csv") %>%
  select(-...1) %>%
  names()

feats_soer <- read_csv("data/participants/for_analysis/for_models/reduced_features/feats_soer.csv") %>%
  select(-...1) %>%
  names()

feats_ihs <- read_csv("data/participants/for_analysis/for_models/reduced_features/feats_ihs.csv") %>%
  select(-...1) %>%
  names()

# COMBINE AND EXPORT ------------------------------------------------------

# Combine into DF
features_df <- data.frame(
  "feats_stressth" = feats_stressth,
  "feats_stressct" = feats_stressct,
  "feats_pss" = feats_pss,
  "feats_lec" = feats_lec,
  "feats_dheq" = feats_dheq,
  "feats_oi" = feats_oi,
  "feats_soer" = feats_soer,
  "feats_ihs" = feats_ihs
) %>%
  as_tibble()

# Export
write_csv(features_df, "data/participants/for_analysis/for_models/reduced_features/features_df.csv")

# ANALYZE -----------------------------------------------------------------

# What features are shared across psychosocial stressors?
features_across_outcomes <- features_df %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value") %>%
  get_dupes(value) %>%
  # Feature is shared across all outcomes
  filter(dupe_count == 8) %>%
  distinct(value)

# Proportion of features shared across outcomes?
nrow(features_across_outcomes) / 100

# Which psychosocial stressors have the most in common with the other 
# psychosocial stressors?
features_df %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value") %>%
  get_dupes(value) %>%
  count(variable)

# Features with the most amount of overlap?
features_across_outcomes %>% 
  mutate(value = str_extract(value, regex("(?<=\\w{4,5}_)[A-z0-9]{3,6}(?=_)"))) %>%
  mutate(value = if_else(value %in% c("lda", "gsdmm"), "topic", value)) %>%
  count(value) %>%
  mutate(percent = n / nrow(features_across_outcomes)) %>%
  arrange(desc(n))

features_across_outcomes %>%
  View()

# What percentage of top features are mean, variance, entropy?
features_df %>%
  mutate(across(everything(), ~ str_extract(., regex("\\w{4,5}(?=_)")))) %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value") %>%
  group_by(variable) %>%
  count(value) %>%
  mutate(percent = (n / 100) * 100) %>%
  View()

# What is the "prej" feature
features_df %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value") %>%
  mutate(is_prej = if_else(str_detect(value, "prej"), 1, 0)) %>%
  filter(is_prej == 1)

# Percentage of feature types
features_df %>%
  mutate(across(everything(), ~ str_extract(., regex("(?<=\\w{4,5}_)[A-z0-9]{3,6}(?=_)")))) %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value") %>%
  mutate(value = if_else(value %in% c("lda", "gsdmm"), "topic", value)) %>%
  count(variable, value) %>% View()
