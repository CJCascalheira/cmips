# SYNTHESIZE FEATURES -----------------------------------------------------

# Author = Cory J. Cascalheira
# Date = 04/19/2024

# The purpose of this script is to combine the features and to calculate the 
# time-based summary variables for all of the features. 

# LOAD DEPENDENCIES AND IMPORT --------------------------------------------

# Load libraries
library(tidyverse)
library(rjson)
library(lubridate)
library(Amelia)

# Import the original data prior to feature creation 
cmips_df <- read_csv("data/participants/cleaned/social_media_posts_full.csv") %>%
  select(-posts_comments)
  
# Import feature sets

# LIWC
cmips_feature_set_00 <- read_csv("data/participants/features/cmips_feature_set_00.csv") %>%
  rename(participant_id = A, timestamp = B, posts_comments = C) %>%
  pivot_longer(cols = WC:OtherP, names_to = "liwc", values_to = "values") %>%
  mutate(liwc = paste0("liwc_", liwc)) %>%
  pivot_wider(names_from = "liwc", values_from = "values") %>%
  select(-posts_comments)

# Behavioral engagement
cmips_feature_set_01 <- read_csv("data/participants/features/cmips_feature_set_01.csv")

# Lexicons
cmips_feature_set_02 <- read_csv("data/participants/features/cmips_feature_set_02.csv") %>%
  select(-text)

# Stress classifier
cmips_feature_set_03 <- read_csv("data/participants/features/cmips_feature_set_03.csv") %>%
  select(-...1)

# Relax n-grams
cmips_feature_set_03_relax_ngrams <- read_csv("data/util/tensi_strength_data/cmips_df_ngrams.csv") %>%
  # For relaxation n-grams, filter out the stress n-grams
  select(participant_id, timestamp, beautiful:ó, `be happy`:`bless them`)

# Word embeddings
cmips_feature_set_04 <- read_csv("data/participants/features/cmips_feature_set_04.csv") %>%
  select(-...1, -text)

# Topic models
cmips_feature_set_05_lda <- read_csv("data/participants/features/cmips_feature_set_05_lda.csv") %>%
  select(-...1)

cmips_feature_set_05_gsdmm <- read_csv("data/participants/features/cmips_feature_set_05_gsdmm.csv") %>%
  select(-...1)

# Minority stress classifier
cmips_feature_set_06 <- read_csv("data/participants/features/cmips_feature_set_06.csv") %>%
  mutate(timestamp = mdy(timestamp))

# DASSP N-Grams
cmips_feature_set_07_dassp <- read_csv("data/participants/features/cmips_feature_set_07_dassp.csv") %>%
  select(-my_text)

# Minority Stress N-Grams
cmips_feature_set_08 <- read_csv("data/participants/features/cmips_feature_set_08.csv") %>%
  select(-my_text)

# Import additional data
lda_topic_name_df <- read_csv("data/participants/features/lda_topic_name_df.csv") %>%
  select(-...1)

gsdmm_log <- fromJSON(file="data/participants/features/gsdmm_log.json")

# PROCESS THE FEATURES ----------------------------------------------------

# ...1) Process the N-Grams -----------------------------------------------

# Printed cmips_feature_set_02 to console, which showed no overlapping 
# variables in lexicon df

# Combine all the n-gram dataframes
ngram_df <- cmips_df %>%
  select(-starts_with("covariate")) %>%
  left_join(cmips_feature_set_03_relax_ngrams, 
            by = c("participant_id", "timestamp")) %>%
  left_join(cmips_feature_set_07_dassp, 
            by = c("participant_id", "timestamp")) %>%
  left_join(cmips_feature_set_08, 
            by = c("participant_id", "timestamp")) %>%
  # Remove any duplicates
  select(-ends_with(".y"))
gc()

# Remove n-grams with no information
ngrams_to_keep <- ngram_df %>%
  pivot_longer(cols = beautiful:`straight cis`, names_to = "ngrams", 
               values_to = "presence") %>%
  group_by(ngrams) %>%
  summarize(
    sum_ngrams = sum(presence, na.rm = TRUE)
  ) %>%
  # Check the n-grams and remove any that are completely 0 all the way
  filter(sum_ngrams > 0) %>%
  distinct(ngrams, .keep_all = TRUE)
ngrams_to_keep
gc()

# Check distribution of the ngrams
ngrams_to_keep %>%
  summarize(
    median_ngrams = median(sum_ngrams),
    mean_ngrams = mean(sum_ngrams)
  )

# Remove more n-grams
ngrams_to_keep %>%
  filter(sum_ngrams < 25) %>%
  pull(ngrams)

# Do not remove nay more n-grams because the nuance of the signal may be in the
# ngrams below the median

# Filter the ngrams
ngram_df <- ngram_df %>%
  select(participant_id, timestamp, ngrams_to_keep$ngrams) %>%
  # Rename the n-grams
  pivot_longer(cols = `a amaze`:zaps, names_to = "ngram", values_to = "value") %>%
  mutate(
    ngram = str_replace(ngram, " ", "_"),
    ngram = paste0("ngram_", ngram)
  ) %>%
  pivot_wider(names_from = "ngram", values_from = "value")
gc()

# ...2) Process the Topic Models ------------------------------------------

# For LDA topics, need to pivot into wide format and rename topics
lda_df <- left_join(cmips_feature_set_05_lda, lda_topic_name_df, by = "topic_number") %>%
  select(-topic_number) %>%
  pivot_wider(names_from = "topic_names", values_from = "topic_probability")

# Count the number of topics
cmips_feature_set_05_gsdmm %>%
  count(gsdmm_predicted_topic)

# For every element in the GSDMM list
for (i in 1:length(gsdmm_log)) {
  # Create a dataframe of values
  assign(paste0("gsdmm", i), bind_rows(gsdmm_log[[i]]))
}

# Preprocess the GSDMM dataframes to select top three words associated w/ topics
gsdmm3 <- gsdmm3 %>%
  pivot_longer(cols = everything(), names_to = "words", values_to = "my_count") %>%
  arrange(desc(my_count)) %>%
  head(n = 3) %>%
  pull(words)

gsdmm7 <- gsdmm7 %>%
  pivot_longer(cols = everything(), names_to = "words", values_to = "my_count") %>%
  arrange(desc(my_count)) %>%
  head(n = 3) %>%
  pull(words)

gsdmm8 <- gsdmm8 %>%
  pivot_longer(cols = everything(), names_to = "words", values_to = "my_count") %>%
  arrange(desc(my_count)) %>%
  head(n = 3) %>%
  pull(words)

gsdmm9 <- gsdmm9 %>%
  pivot_longer(cols = everything(), names_to = "words", values_to = "my_count") %>%
  arrange(desc(my_count)) %>%
  head(n = 3) %>%
  pull(words)

gsdmm11 <- gsdmm11 %>%
  pivot_longer(cols = everything(), names_to = "words", values_to = "my_count") %>%
  arrange(desc(my_count)) %>%
  head(n = 3) %>%
  pull(words)

gsdmm13 <- gsdmm13 %>%
  pivot_longer(cols = everything(), names_to = "words", values_to = "my_count") %>%
  arrange(desc(my_count)) %>%
  head(n = 3) %>%
  pull(words)

gsdmm15 <- gsdmm15 %>%
  pivot_longer(cols = everything(), names_to = "words", values_to = "my_count") %>%
  arrange(desc(my_count)) %>%
  head(n = 3) %>%
  pull(words)

gsdmm17 <- gsdmm17 %>%
  pivot_longer(cols = everything(), names_to = "words", values_to = "my_count") %>%
  arrange(desc(my_count)) %>%
  head(n = 3) %>%
  pull(words)

gsdmm18 <- gsdmm18 %>%
  pivot_longer(cols = everything(), names_to = "words", values_to = "my_count") %>%
  arrange(desc(my_count)) %>%
  head(n = 3) %>%
  pull(words)

gsdmm19 <- gsdmm19 %>%
  pivot_longer(cols = everything(), names_to = "words", values_to = "my_count") %>%
  arrange(desc(my_count)) %>%
  head(n = 3) %>%
  pull(words)

gsdmm23 <- gsdmm23 %>%
  pivot_longer(cols = everything(), names_to = "words", values_to = "my_count") %>%
  arrange(desc(my_count)) %>%
  head(n = 3) %>%
  pull(words)

gsdmm24 <- gsdmm24 %>%
  pivot_longer(cols = everything(), names_to = "words", values_to = "my_count") %>%
  arrange(desc(my_count)) %>%
  head(n = 3) %>%
  pull(words)

gsdmm25 <- gsdmm25 %>%
  pivot_longer(cols = everything(), names_to = "words", values_to = "my_count") %>%
  arrange(desc(my_count)) %>%
  head(n = 3) %>%
  pull(words)

gsdmm26 <- gsdmm26 %>%
  pivot_longer(cols = everything(), names_to = "words", values_to = "my_count") %>%
  arrange(desc(my_count)) %>%
  head(n = 3) %>%
  pull(words)

gsdmm30 <- gsdmm30 %>%
  pivot_longer(cols = everything(), names_to = "words", values_to = "my_count") %>%
  arrange(desc(my_count)) %>%
  head(n = 3) %>%
  pull(words)

# Create a dataframe of GSDMM topic names
gsdmm_names <- c("gsdmm_none",
  paste0("gsdmm_", paste0(gsdmm3, collapse = "")),
  paste0("gsdmm_", paste0("asian", gsdmm7[c(2, 3)], collapse = "")),
  paste0("gsdmm_", paste0(gsdmm8, collapse = "")),
  paste0("gsdmm_", "all_asian"),
  paste0("gsdmm_", paste0(gsdmm11, collapse = "")),
  paste0("gsdmm_", paste0(gsdmm13, collapse = "")),
  paste0("gsdmm_", paste0(gsdmm15, collapse = "")),
  paste0("gsdmm_", paste0(gsdmm17, collapse = "")),
  paste0("gsdmm_", paste0(gsdmm18, collapse = "")),
  paste0("gsdmm_", paste0(gsdmm19, collapse = "")),
  paste0("gsdmm_", paste0(gsdmm23, collapse = "")),
  paste0("gsdmm_", paste0(gsdmm24, collapse = "")),
  paste0("gsdmm_", paste0(gsdmm25, collapse = "")),
  paste0("gsdmm_", paste0(gsdmm26, collapse = "")),
  paste0("gsdmm_", paste0(gsdmm30, collapse = "")))

gsdmm_nums <- c(0, 2, 6, 7, 8, 10, 12, 14, 16, 17, 18, 22, 23, 24, 25, 29)

gsdmm_names_df <- data.frame(gsdmm_predicted_topic = gsdmm_nums, 
                             gsdmm_names = gsdmm_names) %>%
  as_tibble()

# For GSDMM topics, need to organize the top words and assign names
gsdmm_df <- left_join(cmips_feature_set_05_gsdmm, gsdmm_names_df) %>%
  mutate(topic_presence = 1) %>%
  pivot_wider(names_from = "gsdmm_names", values_from = "topic_presence") %>%
  select(-gsdmm_predicted_topic, -gsdmm_topic_probability) %>%
  # Replace missing values in GSDMM df
  mutate(across(`gsdmm_peopletimeday`:`gsdmm_asianltasianpop`, ~ if_else(is.na(.), 0, .)))

# ...3) Process the Minority Stress Classifier ----------------------------

# Check the minority stress classifier for empty values
ms_class_keep <- cmips_feature_set_06 %>%
  pivot_longer(cols = label_minority_coping:label_minority_stress,
               names_to = "ms_classifier", values_to = "presence") %>%
  group_by(ms_classifier) %>%
  summarize(
    sum_ms = sum(presence, na.rm = TRUE)
  ) %>%
  filter(sum_ms > 0) %>%
  pull(ms_classifier)

# Filter the df
minoritystress_df <- cmips_feature_set_06 %>%
  select(participant_id, timestamp, all_of(ms_class_keep)) %>%
  # Rename the variables to not confuse with the outcomes
  pivot_longer(cols = label_dysphoria:label_prej_event,
               names_to = "ms", values_to = "vals") %>%
  mutate(ms = str_replace(ms, "label_", "ms_")) %>%
  pivot_wider(names_from = "ms", values_from = "vals")

# ...4) Double Check Other Variables --------------------------------------

# Check clinical keywords and lexicons
cmips_feature_set_02 %>%
  pivot_longer(cols = hate_lexicon_sexual_minority:pain_lexicon, 
               names_to = "feature", values_to = "value") %>%
  group_by(feature) %>%
  summarize(
    sum_feat = sum(value, na.rm = TRUE)
  )

# COUNT FINAL FEATURES ----------------------------------------------------

# Behavioral engagement
cmips_feature_set_01 %>%
  select(-participant_id) %>%
  ncol()

# Clinical keywords
cmips_feature_set_02 %>%
  select(starts_with("dsm5")) %>%
  ncol()

# Lexicons
cmips_feature_set_02 %>%
  select(-participant_id, -timestamp, -starts_with("dsm5")) %>%
  ncol()

# Topic models
left_join(lda_df, gsdmm_df) %>%
  select(-participant_id, -timestamp) %>%
  ncol()

# LIWC
cmips_feature_set_00 %>%
  select(-participant_id, -timestamp, -posts_comments) %>%
  ncol()

# Ngrams
ngram_df %>%
  select(-participant_id, -timestamp) %>%
  ncol()

# Classifiers
(ncol(cmips_feature_set_03) - 2) + (ncol(minoritystress_df) - 2)

# Word embeddings
cmips_feature_set_04 %>%
  select(-participant_id, -timestamp) %>%
  ncol()

# CONDENSE THE FEATURES ACROSS TIME ---------------------------------------

# Calculate total time points
cmips_total_days_posts <- cmips_df %>%
  count(participant_id) %>%
  rename(T_days = n)

# Combine the features
feature_df <- cmips_df %>%
  select(participant_id, timestamp) %>%
  # Clinical keywords and lexicons
  left_join(cmips_feature_set_02, by = c("participant_id", "timestamp")) %>%
  # Topic models
  left_join(lda_df, by = c("participant_id", "timestamp")) %>%
  left_join(gsdmm_df, by = c("participant_id", "timestamp")) %>%
  # LIWC
  left_join(cmips_feature_set_00, by = c("participant_id", "timestamp")) %>%
  # N-grams
  left_join(ngram_df, by = c("participant_id", "timestamp")) %>%
  # Classifiers
  left_join(cmips_feature_set_03, by = c("participant_id", "timestamp")) %>%
  left_join(minoritystress_df, by = c("participant_id", "timestamp")) %>%
  # Word embeddings
  left_join(cmips_feature_set_04, by = c("participant_id", "timestamp")) %>%
  # Remove the covariates - add them back at the end
  select(-starts_with("covariate"))
gc()

# Clean up the workspace to preserve memory
rm(cmips_feature_set_00, cmips_feature_set_02, cmips_feature_set_03,
   cmips_feature_set_03_relax_ngrams, cmips_feature_set_04, cmips_feature_set_05_gsdmm,
   cmips_feature_set_05_lda, cmips_feature_set_06, cmips_feature_set_07_dassp,
   cmips_feature_set_08, gsdmm1, gsdmm_df, gsdmm_names_df, gsdmm10, gsdmm12,
   gsdmm14, gsdmm16, gsdmm2, gsdmm20, gsdmm21, gsdmm22, gsdmm27, gsdmm28, gsdmm29,
   gsdmm4, gsdmm5, gsdmm6)

# ...1) Feature-Specific Mean ---------------------------------------------

# Start with the feature df
feature_df_mean <- feature_df %>%
  # Long format
  pivot_longer(cols = hate_lexicon_sexual_minority:w2v_299,
               names_to = "feature", values_to = "value") %>%
  # Remove missing values
  filter(!is.na(value)) %>%
  # For each feature for each participant
  group_by(participant_id, feature) %>%
  # Get the sum of the feature, removing missing values
  summarize(
    feat_sum = sum(value, na.rm = TRUE)
  ) %>%
  ungroup() %>%
  # Add the total number of days posted
  left_join(cmips_total_days_posts) %>%
  # For each feature for each participant
  group_by(participant_id, feature) %>%
  # Calculate the feature-specific mean
  summarize(
    feat_mean = feat_sum / T_days
  ) %>%
  ungroup()
gc()

# ...2) Feature-Specific Variance -----------------------------------------

# Start with the feature df
feature_df_variance <- feature_df %>%
  # Long format
  pivot_longer(cols = hate_lexicon_sexual_minority:w2v_299,
               names_to = "feature", values_to = "value") %>%
  # Remove missing values
  filter(!is.na(value)) %>%
  # Add the feature-specific mean
  left_join(feature_df_mean) %>%
  # For each feature for each participant
  group_by(participant_id, feature) %>%
  # Get the difference squared
  summarize(
    diff_sqd = (value - feat_mean)^2
  ) %>%
  ungroup() %>%
  # For each feature for each participant
  group_by(participant_id, feature) %>%
  # Get the sum of differences squared
  summarize(
    sum_diff_sqd = sum(diff_sqd, na.rm = TRUE)
  ) %>%
  ungroup() %>%
  # Add the total number of days posted
  left_join(cmips_total_days_posts) %>%
  # Minus 1 from the total number of days posted
  mutate(T_days = T_days - 1) %>%
  # For each feature for each participant
  group_by(participant_id, feature) %>%
  # Calculate the feature-specific variance
  summarize(
    feat_var = sum_diff_sqd / T_days
  ) %>%
  ungroup()
gc()

# ...3) Entropy -----------------------------------------------------------

# Define function for negative summation (-Sigma)
# https://math.stackexchange.com/questions/1587998/symbol-for-sequential-subtraction
neg_sum = function(my_vector) {
  
  # If the vector is really small
  if(length(my_vector) == 1) {
    
    return(my_vector[1])
    
  } else if(length(my_vector) == 2) {
    
    # Get the initial difference
    diff = my_vector[1] - my_vector[2]
    
    return(diff)
  } else {
    
    # Get the initial difference
    diff = my_vector[1] - my_vector[2]
    
    # Take sequential subtraction from the initial difference
    for(i in 3:length(my_vector)) {
      
      diff = diff - my_vector[i]
    }
    
    return(diff)
  }
}

# Start with the feature df
feature_df_entropy <- feature_df %>%
  # Long format
  pivot_longer(cols = hate_lexicon_sexual_minority:w2v_299,
               names_to = "feature", values_to = "value") %>%
  # Remove missing values
  filter(!is.na(value)) %>%
  # Add the constant to every value
  mutate(value = value + 0.01) %>%
  # For each feature for each participant
  group_by(participant_id, feature) %>%
  # Get the product of the value times the log of the value
  summarize(product_v_log = value * log(value)) %>%
  ungroup() %>%
  # For each feature for each participant
  group_by(participant_id, feature) %>%
  # Series summation for feature-specific entropy
  summarize(feat_entropy = neg_sum(product_v_log)) %>%
  ungroup()
gc()

# ...4) Prepare for Final Merge -------------------------------------------

# Prepare the feature-specific mean dataframe for merging
feature_df_mean <- feature_df_mean %>%
  # Rename features
  mutate(feature = paste0("fmean_", feature)) %>%
  pivot_wider(names_from = "feature", values_from = "feat_mean")

# Prepare the feature-specific variance dataframe for merging
feature_df_variance <- feature_df_variance %>%
  # Rename features
  mutate(feature = paste0("fvar_", feature)) %>%
  pivot_wider(names_from = "feature", values_from = "feat_var")

# Prepare the feature-specific entropy dataframe for merging
feature_df_entropy <- feature_df_entropy %>%
  # Rename features
  mutate(feature = paste0("fent_", feature)) %>%
  pivot_wider(names_from = "feature", values_from = "feat_entropy")

# Check for missing values
complete.cases(feature_df_mean)
complete.cases(feature_df_variance)
complete.cases(feature_df_entropy)

# Number of participants with missing data
nrow(feature_df_mean) - sum(complete.cases(feature_df_mean))
nrow(feature_df_variance) - sum(complete.cases(feature_df_variance))
nrow(feature_df_entropy) - sum(complete.cases(feature_df_entropy))

# Find which variables have missing data and calculate missingness
feature_df_mean %>%
  pivot_longer(cols = fmean_dsm5_anxiety:fmean_w2v_99, 
               names_to = "variables", values_to = "values") %>%
  mutate(is_missing = if_else(is.na(values), 1, 0)) %>%
  filter(is_missing == 1) %>%
  count(participant_id) %>%
  mutate(total_items = ncol(feature_df_mean) - 1) %>%
  mutate(percent = (n / total_items) * 100)

feature_df_variance %>%
  pivot_longer(cols = fvar_dsm5_anxiety:fvar_w2v_99, 
               names_to = "variables", values_to = "values") %>%
  mutate(is_missing = if_else(is.na(values), 1, 0)) %>%
  filter(is_missing == 1) %>%
  count(participant_id) %>%
  mutate(total_items = ncol(feature_df_variance) - 1) %>%
  mutate(percent = (n / total_items) * 100)

feature_df_entropy %>%
  pivot_longer(cols = fent_dsm5_anxiety:fent_w2v_99, 
               names_to = "variables", values_to = "values") %>%
  mutate(is_missing = if_else(is.na(values), 1, 0)) %>%
  filter(is_missing == 1) %>%
  count(participant_id) %>%
  mutate(total_items = ncol(feature_df_entropy) - 1) %>%
  mutate(percent = (n / total_items) * 100)

# Participant CMIPS_0072 has missing feature-specific variance due to only 
# having one day of data. Therefore, there is no variance, so reassign all values
# to zero for this participant
feature_df_variance_0072 <- feature_df_variance %>%
  filter(participant_id == "CMIPS_0072") %>%
  mutate(across(fvar_dsm5_anxiety:fvar_w2v_99, ~ 0))

feature_df_variance <- feature_df_variance %>%
  filter(participant_id != "CMIPS_0072") %>%
  bind_rows(feature_df_variance_0072)

# Get the names of the missing variables
missing_var_name <- feature_df_mean %>%
  pivot_longer(cols = fmean_dsm5_anxiety:fmean_w2v_99, 
               names_to = "variables", values_to = "values") %>%
  mutate(is_missing = if_else(is.na(values), 1, 0)) %>%
  filter(is_missing == 1) %>%
  distinct(variables) %>%
  pull(variables)
missing_var_name

feature_df_variance %>%
  pivot_longer(cols = fvar_dsm5_anxiety:fvar_w2v_99, 
               names_to = "variables", values_to = "values") %>%
  mutate(is_missing = if_else(is.na(values), 1, 0)) %>%
  filter(is_missing == 1) %>%
  distinct(variables) %>%
  pull(variables)

feature_df_entropy %>%
  pivot_longer(cols = fent_dsm5_anxiety:fent_w2v_99, 
               names_to = "variables", values_to = "values") %>%
  mutate(is_missing = if_else(is.na(values), 1, 0)) %>%
  filter(is_missing == 1) %>%
  distinct(variables) %>%
  pull(variables)

# Replace missing data with multiple imputation
feature_df_mean_imputed <- feature_df_mean %>%
  select(participant_id, contains("_lda_")) %>%
  as.data.frame() %>%
  amelia(idvars = "participant_id")

feature_df_variance_imputed <- feature_df_variance %>%
  select(participant_id, contains("_lda_")) %>%
  as.data.frame() %>%
  amelia(idvars = "participant_id")

feature_df_entropy_imputed <- feature_df_entropy %>%
  select(participant_id, contains("_lda_")) %>%
  as.data.frame() %>%
  amelia(idvars = "participant_id")

# Get the final imputed dataset
feature_df_mean_imputed_final <- feature_df_mean_imputed$imputations[[5]] %>%
  as_tibble()

feature_df_variance_imputed_final <- feature_df_variance_imputed$imputations[[5]] %>%
  as_tibble()

feature_df_entropy_imputed_final <- feature_df_entropy_imputed$imputations[[5]] %>%
  as_tibble()

# Merge the imputed values with the original data
feature_df_mean <- feature_df_mean %>%
  select(-contains("_lda_")) %>%
  left_join(feature_df_mean_imputed_final)

feature_df_variance <- feature_df_variance %>%
  select(-contains("_lda_")) %>%
  left_join(feature_df_variance_imputed_final)

feature_df_entropy <- feature_df_entropy %>%
  select(-contains("_lda_")) %>%
  left_join(feature_df_entropy_imputed_final)

# ...5) Final Merge -------------------------------------------------------

# Re-add the covariates
cmips_features <- cmips_df %>%
  select(-timestamp) %>%
  distinct(participant_id, .keep_all = TRUE) %>%
  # Merge the aggregated statistics
  left_join(feature_df_mean) %>%
  left_join(feature_df_variance) %>%
  left_join(feature_df_entropy) %>%
  # Merge with behavioral engagement
  left_join(cmips_feature_set_01)

# Export data
write_csv(cmips_features, "data/participants/for_analysis/cmips_features.csv")
