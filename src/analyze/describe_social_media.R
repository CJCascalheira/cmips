# DESCRIBE SOCIAL MEDIA DATA ----------------------------------------------

# Author = Cory Cascalheira
# Date = 04/14/2024

# The purpose of this script is to calculate descriptive statistics of the 
# social media data. 

# LOAD DEPENDENCIES AND IMPORT DATA ---------------------------------------

# Load libraries
library(tidyverse)
library(readxl)
library(tidytext)
library(janitor)
library(psych)
library(scales)
library(lsr) # Cohen's D

# Import data
social_media_posts <- read_csv("data/participants/combined_social_media/social_media_posts_full.csv")

social_media_reactions <- read_csv("data/participants/combined_social_media/social_media_reactions.csv")

participant_tracker <- read_excel("data/participants/Participant_Tracking.xlsx",
                                  sheet = 1) %>%
  # Keep only usable rows
  filter(!is.na(ResponseId))

# Check for social media extraction errors

# Get the IDs for participants who shared their social media data
participant_tracker_ids <- participant_tracker %>%
  filter(Keep == "YES") %>%
  select(participant_id = ParticipantID)

# Check the participant tracker against the data Santosh sent
extraction_posts_ids <- social_media_posts %>%
  distinct(participant_id)

extraction_reactions_ids <- social_media_reactions %>%
  distinct(participant_id)

# Find non matches
shared_wrong_data <- anti_join(participant_tracker_ids, extraction_posts_ids)
shared_wrong_data

anti_join(participant_tracker_ids, extraction_reactions_ids) %>% 
  pull(participant_id)

# Manually checked social media of participants still missing data. Fixed a bug
# in the extraction code. However, CMIPS_0312 and CMIPS_0374 are still missing
# data. They simply did not submit the requested data, so nothing we can do. 

# Import cleaned data
social_media_posts_cleaned <- read_csv("data/participants/cleaned/social_media_posts_cleaned.csv")

social_media_reactions_cleaned <- read_csv("data/participants/cleaned/social_media_reactions.csv")

# Import LIWC data
liwc <- read_csv("data/participants/features/cmips_feature_set_00.csv")

# Import features
cmips_features <- read_csv("data/participants/for_analysis/cmips_features.csv") %>%
  select(participant_id, starts_with("fmean"), starts_with("be_"))

# Import survey
cmips_surveys <- read_csv("data/participants/for_analysis/cmips_surveys_full.csv") %>%
  rename(participant_id = ParticipantID) %>%
  filter(participant_id %in% cmips_features$participant_id) %>%
  select(participant_id, starts_with("label"))

# ANALYZE RAW DATA ---------------------------------------------------------

# Count data shared from each social media platform
social_media_posts %>%
  count(participant_id, platform) %>%
  count(platform)

# People who shared data from both platforms - need covariate
participants_shared_both_fbtw <- social_media_posts %>%
  count(participant_id, platform) %>%
  get_dupes(participant_id) %>%
  distinct(participant_id)
nrow(participants_shared_both_fbtw)

# People who shared one year's worth of data - need covariate

# Get the max date and subtract one year to get the start date
start_end_dates <- social_media_posts %>%
  group_by(participant_id) %>%
  summarize(
    end_date = max(timestamp)
  ) %>%
  mutate(
    start_date = end_date - years(1)
  )

# Initialize and empty dataframe
sm_posts <- data.frame(
  participant_id = as.character(),
  timestamp = as_date(as.character()),
  posts_comments = as.character(),
  platform = as.character()
)

# Iterate over data
for (i in 1:nrow(start_end_dates)) {
  
  # Filter each participant
  filtered_participant <- social_media_posts %>%
    filter(participant_id == start_end_dates$participant_id[i]) %>%
    filter(timestamp >= start_end_dates$start_date[i], 
           timestamp <= start_end_dates$end_date[i])
  
  # Bind to the dataframe
  sm_posts <- bind_rows(sm_posts, filtered_participant) 
}

# Convert to tibble
sm_posts <- as_tibble(sm_posts)

# Find posts not included in the filtered dataset
participants_shared_more_one_year <- anti_join(social_media_posts, sm_posts, 
                                               by = c("participant_id", "timestamp")) %>%
  count(participant_id, platform) %>%
  # Code may have made some error, so remove any participant with fewer than
  # 20 posts
  filter(n > 20) %>%
  distinct(participant_id)

# How much data loss if use one-year request only
nrow(social_media_posts) - nrow(sm_posts)
1 - (nrow(sm_posts) / nrow(social_media_posts))

# EXPORT IDS TO MAKE COVARIATES -------------------------------------------

# Save file to make covariates
write_csv(participants_shared_both_fbtw, "data/participants/util/participants_shared_both_fbtw.csv")
write_csv(participants_shared_more_one_year, "data/participants/util/participants_shared_more_one_year.csv")

# Save file of participants who shared the wrong social media data
write_csv(shared_wrong_data, "data/participants/util/shared_wrong_data.csv")

# ANALYZE PREPROCESSED DATA -----------------------------------------------

# How many posts and reactions after preprocessing?
nrow(social_media_posts_cleaned)
nrow(social_media_reactions_cleaned)

# Descriptive statistics of word counts
describe(liwc$WC)

# DESCRIPTIVE ANALYSIS OF FEATURES ----------------------------------------

# ...1) Visualizations ----------------------------------------------------

# Bar chart of top 30 words
wc_bar_plot <- social_media_posts_cleaned %>%
  unnest_tokens(output = "word", input = "posts_comments") %>%
  # Remove stop words
  filter(!(word %in% stop_words$word)) %>%
  count(word) %>%
  arrange(desc(n)) %>%
  head(n = 20) %>%
  ggplot(aes(x = reorder(word, n), y = n)) +
  geom_bar(stat="identity", fill="steelblue") + 
  coord_flip() +
  scale_y_continuous(label = comma, breaks = pretty_breaks(n = 10)) +
  theme_bw() + 
  theme(text = element_text(family = "serif")) +
  labs(
    x = "Word in Social Media Posts",
    y = "Word Count"
  )
wc_bar_plot

# Save the plot
ggsave(filename = "results/plots/wc_bar_plot.png", plot = wc_bar_plot,
       width = 8, height = 5)

# ...2) Behavioral Engagement ---------------------------------------------

# Get the behavioral engagement metrics
be_engage <- cmips_features %>%
  select(starts_with("be_"))

# Descriptives of each BE feature
describe(be_engage$be_avg_12_6)
describe(be_engage$be_avg_n_urls)
describe(be_engage$be_avg_hashtags)
describe(be_engage$be_avg_daily_posts)
describe(be_engage$be_total_n_posts)
describe(be_engage$be_total_n_reactions)
describe(be_engage$be_max_posts_day)

# ...3) Group Comparisons -------------------------------------------------

# Select the data for group comparisons
cmips_groups <- cmips_features %>%
  select(participant_id, contains("_dsm5"), contains("sentiment"), 
         contains("_liwc"), contains("_lexicon"))

# Merge with labels
cmips_groups <- left_join(cmips_surveys, cmips_groups)

# ......3a) Lifetime Stressor Severity ------------------------------------

# Filter the data
cmips_groups_long <- cmips_groups %>%
  select(label_StressTH, starts_with("fmean")) %>%
  rename(stress_group = label_StressTH) %>%
  pivot_longer(cols = starts_with("fmean"), names_to = "variable", values_to = "value")

# Execute the independent samples t-tests w/o variance equality assumption
test_results <- lapply(split(cmips_groups_long, cmips_groups_long$variable), 
                       function(x) t.test(value~stress_group, x))
print(test_results)

# Bonferroni correction value
bonf_value <- 0.05 / length(names(test_results))

# Extract data from list into df, next few lines

# Prepare vectors
t_stat <- c()
p_val <- c()
deg_free <- c()

# For loop to extract data
for (i in 1:length(names(test_results))) {
  t_stat <- c(t_stat, test_results[[i]]$statistic)
  p_val <- c(p_val, test_results[[i]]$p.value)
  deg_free <- c(deg_free, test_results[[i]]$parameter)
}

# Effect size using Cohen's D
# https://rcompanion.org/handbook/I_03.html
effect_sizes <- lapply(split(cmips_groups_long, cmips_groups_long$variable), 
                       function(x) cohensD(value~stress_group, x)) %>%
  as_vector()

# Get LIWC vars names
variable_names <- names(test_results)

# Add all vectors into df
test_signf_results <- data.frame(variable_names, t_stat, deg_free, effect_sizes, p_val) %>%
  as_tibble() %>%
  # Set Bonferroni value and determine if p_val is beyond/more extreme
  mutate(bonf_value = bonf_value) %>%
  mutate(signf_at_bonf = p_val < bonf_value) %>%
  # Keep only significant differences
  filter(signf_at_bonf == TRUE)
test_signf_results

# Save data to csv
write_csv(test_signf_results, "results/group_comparisons/label_StressTH.csv")

# ......3a) Lifetime Stressor Count ------------------------------------

# Filter the data
cmips_groups_long <- cmips_groups %>%
  select(label_StressCT, starts_with("fmean")) %>%
  rename(stress_group = label_StressCT) %>%
  pivot_longer(cols = starts_with("fmean"), names_to = "variable", values_to = "value")

# Execute the independent samples t-tests w/o variance equality assumption
test_results <- lapply(split(cmips_groups_long, cmips_groups_long$variable), 
                       function(x) t.test(value~stress_group, x))
print(test_results)

# Bonferroni correction value
bonf_value <- 0.05 / length(names(test_results))

# Extract data from list into df, next few lines

# Prepare vectors
t_stat <- c()
p_val <- c()
deg_free <- c()

# For loop to extract data
for (i in 1:length(names(test_results))) {
  t_stat <- c(t_stat, test_results[[i]]$statistic)
  p_val <- c(p_val, test_results[[i]]$p.value)
  deg_free <- c(deg_free, test_results[[i]]$parameter)
}

# Effect size using Cohen's D
# https://rcompanion.org/handbook/I_03.html
effect_sizes <- lapply(split(cmips_groups_long, cmips_groups_long$variable), 
                       function(x) cohensD(value~stress_group, x)) %>%
  as_vector()

# Get LIWC vars names
variable_names <- names(test_results)

# Add all vectors into df
test_signf_results <- data.frame(variable_names, t_stat, deg_free, effect_sizes, p_val) %>%
  as_tibble() %>%
  # Set Bonferroni value and determine if p_val is beyond/more extreme
  mutate(bonf_value = bonf_value) %>%
  mutate(signf_at_bonf = p_val < bonf_value) %>%
  # Keep only significant differences
  filter(signf_at_bonf == TRUE)
test_signf_results

# Save data to csv
write_csv(test_signf_results, "results/group_comparisons/label_StressCT.csv")

# ......3a) Perceived Stress ------------------------------------

# Filter the data
cmips_groups_long <- cmips_groups %>%
  select(label_PSS_total, starts_with("fmean")) %>%
  rename(stress_group = label_PSS_total) %>%
  pivot_longer(cols = starts_with("fmean"), names_to = "variable", values_to = "value")

# Execute the independent samples t-tests w/o variance equality assumption
test_results <- lapply(split(cmips_groups_long, cmips_groups_long$variable), 
                       function(x) t.test(value~stress_group, x))
print(test_results)

# Bonferroni correction value
bonf_value <- 0.05 / length(names(test_results))

# Extract data from list into df, next few lines

# Prepare vectors
t_stat <- c()
p_val <- c()
deg_free <- c()

# For loop to extract data
for (i in 1:length(names(test_results))) {
  t_stat <- c(t_stat, test_results[[i]]$statistic)
  p_val <- c(p_val, test_results[[i]]$p.value)
  deg_free <- c(deg_free, test_results[[i]]$parameter)
}

# Effect size using Cohen's D
# https://rcompanion.org/handbook/I_03.html
effect_sizes <- lapply(split(cmips_groups_long, cmips_groups_long$variable), 
                       function(x) cohensD(value~stress_group, x)) %>%
  as_vector()

# Get LIWC vars names
variable_names <- names(test_results)

# Add all vectors into df
test_signf_results <- data.frame(variable_names, t_stat, deg_free, effect_sizes, p_val) %>%
  as_tibble() %>%
  # Set Bonferroni value and determine if p_val is beyond/more extreme
  mutate(bonf_value = bonf_value) %>%
  mutate(signf_at_bonf = p_val < bonf_value) %>%
  # Keep only significant differences
  filter(signf_at_bonf == TRUE)
test_signf_results

# Save data to csv
write_csv(test_signf_results, "results/group_comparisons/label_PSS_total.csv")

# ......3a) Potentially Traumatic Life Events -----------------------------

# Filter the data
cmips_groups_long <- cmips_groups %>%
  select(label_LEC_total, starts_with("fmean")) %>%
  rename(stress_group = label_LEC_total) %>%
  pivot_longer(cols = starts_with("fmean"), names_to = "variable", values_to = "value")

# Execute the independent samples t-tests w/o variance equality assumption
test_results <- lapply(split(cmips_groups_long, cmips_groups_long$variable), 
                       function(x) t.test(value~stress_group, x))
print(test_results)

# Bonferroni correction value
bonf_value <- 0.05 / length(names(test_results))

# Extract data from list into df, next few lines

# Prepare vectors
t_stat <- c()
p_val <- c()
deg_free <- c()

# For loop to extract data
for (i in 1:length(names(test_results))) {
  t_stat <- c(t_stat, test_results[[i]]$statistic)
  p_val <- c(p_val, test_results[[i]]$p.value)
  deg_free <- c(deg_free, test_results[[i]]$parameter)
}

# Effect size using Cohen's D
# https://rcompanion.org/handbook/I_03.html
effect_sizes <- lapply(split(cmips_groups_long, cmips_groups_long$variable), 
                       function(x) cohensD(value~stress_group, x)) %>%
  as_vector()

# Get LIWC vars names
variable_names <- names(test_results)

# Add all vectors into df
test_signf_results <- data.frame(variable_names, t_stat, deg_free, effect_sizes, p_val) %>%
  as_tibble() %>%
  # Set Bonferroni value and determine if p_val is beyond/more extreme
  mutate(bonf_value = bonf_value) %>%
  mutate(signf_at_bonf = p_val < bonf_value) %>%
  # Keep only significant differences
  filter(signf_at_bonf == TRUE)
test_signf_results

# Save data to csv
write_csv(test_signf_results, "results/group_comparisons/label_LEC_total.csv")

# ......3a) Prejudiced Events -----------------------------

# Filter the data
cmips_groups_long <- cmips_groups %>%
  select(label_DHEQ_mean, starts_with("fmean")) %>%
  rename(stress_group = label_DHEQ_mean) %>%
  pivot_longer(cols = starts_with("fmean"), names_to = "variable", values_to = "value")

# Execute the independent samples t-tests w/o variance equality assumption
test_results <- lapply(split(cmips_groups_long, cmips_groups_long$variable), 
                       function(x) t.test(value~stress_group, x))
print(test_results)

# Bonferroni correction value
bonf_value <- 0.05 / length(names(test_results))

# Extract data from list into df, next few lines

# Prepare vectors
t_stat <- c()
p_val <- c()
deg_free <- c()

# For loop to extract data
for (i in 1:length(names(test_results))) {
  t_stat <- c(t_stat, test_results[[i]]$statistic)
  p_val <- c(p_val, test_results[[i]]$p.value)
  deg_free <- c(deg_free, test_results[[i]]$parameter)
}

# Effect size using Cohen's D
# https://rcompanion.org/handbook/I_03.html
effect_sizes <- lapply(split(cmips_groups_long, cmips_groups_long$variable), 
                       function(x) cohensD(value~stress_group, x)) %>%
  as_vector()

# Get LIWC vars names
variable_names <- names(test_results)

# Add all vectors into df
test_signf_results <- data.frame(variable_names, t_stat, deg_free, effect_sizes, p_val) %>%
  as_tibble() %>%
  # Set Bonferroni value and determine if p_val is beyond/more extreme
  mutate(bonf_value = bonf_value) %>%
  mutate(signf_at_bonf = p_val < bonf_value) %>%
  # Keep only significant differences
  filter(signf_at_bonf == TRUE)
test_signf_results

# Save data to csv
write_csv(test_signf_results, "results/group_comparisons/label_DHEQ_mean.csv")

# ......3a) Identity Concealment -----------------------------

# Filter the data
cmips_groups_long <- cmips_groups %>%
  select(label_OI_mean, starts_with("fmean")) %>%
  rename(stress_group = label_OI_mean) %>%
  pivot_longer(cols = starts_with("fmean"), names_to = "variable", values_to = "value")

# Execute the independent samples t-tests w/o variance equality assumption
test_results <- lapply(split(cmips_groups_long, cmips_groups_long$variable), 
                       function(x) t.test(value~stress_group, x))
print(test_results)

# Bonferroni correction value
bonf_value <- 0.05 / length(names(test_results))

# Extract data from list into df, next few lines

# Prepare vectors
t_stat <- c()
p_val <- c()
deg_free <- c()

# For loop to extract data
for (i in 1:length(names(test_results))) {
  t_stat <- c(t_stat, test_results[[i]]$statistic)
  p_val <- c(p_val, test_results[[i]]$p.value)
  deg_free <- c(deg_free, test_results[[i]]$parameter)
}

# Effect size using Cohen's D
# https://rcompanion.org/handbook/I_03.html
effect_sizes <- lapply(split(cmips_groups_long, cmips_groups_long$variable), 
                       function(x) cohensD(value~stress_group, x)) %>%
  as_vector()

# Get LIWC vars names
variable_names <- names(test_results)

# Add all vectors into df
test_signf_results <- data.frame(variable_names, t_stat, deg_free, effect_sizes, p_val) %>%
  as_tibble() %>%
  # Set Bonferroni value and determine if p_val is beyond/more extreme
  mutate(bonf_value = bonf_value) %>%
  mutate(signf_at_bonf = p_val < bonf_value) %>%
  # Keep only significant differences
  filter(signf_at_bonf == TRUE)
test_signf_results

# Save data to csv
write_csv(test_signf_results, "results/group_comparisons/label_OI_mean.csv")

# ......3a) Expected Rejection -----------------------------

# Filter the data
cmips_groups_long <- cmips_groups %>%
  select(label_SOER_total, starts_with("fmean")) %>%
  rename(stress_group = label_SOER_total) %>%
  pivot_longer(cols = starts_with("fmean"), names_to = "variable", values_to = "value")

# Execute the independent samples t-tests w/o variance equality assumption
test_results <- lapply(split(cmips_groups_long, cmips_groups_long$variable), 
                       function(x) t.test(value~stress_group, x))
print(test_results)

# Bonferroni correction value
bonf_value <- 0.05 / length(names(test_results))

# Extract data from list into df, next few lines

# Prepare vectors
t_stat <- c()
p_val <- c()
deg_free <- c()

# For loop to extract data
for (i in 1:length(names(test_results))) {
  t_stat <- c(t_stat, test_results[[i]]$statistic)
  p_val <- c(p_val, test_results[[i]]$p.value)
  deg_free <- c(deg_free, test_results[[i]]$parameter)
}

# Effect size using Cohen's D
# https://rcompanion.org/handbook/I_03.html
effect_sizes <- lapply(split(cmips_groups_long, cmips_groups_long$variable), 
                       function(x) cohensD(value~stress_group, x)) %>%
  as_vector()

# Get LIWC vars names
variable_names <- names(test_results)

# Add all vectors into df
test_signf_results <- data.frame(variable_names, t_stat, deg_free, effect_sizes, p_val) %>%
  as_tibble() %>%
  # Set Bonferroni value and determine if p_val is beyond/more extreme
  mutate(bonf_value = bonf_value) %>%
  mutate(signf_at_bonf = p_val < bonf_value) %>%
  # Keep only significant differences
  filter(signf_at_bonf == TRUE)
test_signf_results

# Save data to csv
write_csv(test_signf_results, "results/group_comparisons/label_SOER_total.csv")

# ......3a) Internalized Stigma -----------------------------

# Filter the data
cmips_groups_long <- cmips_groups %>%
  select(label_IHS_mean, starts_with("fmean")) %>%
  rename(stress_group = label_IHS_mean) %>%
  pivot_longer(cols = starts_with("fmean"), names_to = "variable", values_to = "value")

# Execute the independent samples t-tests w/o variance equality assumption
test_results <- lapply(split(cmips_groups_long, cmips_groups_long$variable), 
                       function(x) t.test(value~stress_group, x))
print(test_results)

# Bonferroni correction value
bonf_value <- 0.05 / length(names(test_results))

# Extract data from list into df, next few lines

# Prepare vectors
t_stat <- c()
p_val <- c()
deg_free <- c()

# For loop to extract data
for (i in 1:length(names(test_results))) {
  t_stat <- c(t_stat, test_results[[i]]$statistic)
  p_val <- c(p_val, test_results[[i]]$p.value)
  deg_free <- c(deg_free, test_results[[i]]$parameter)
}

# Effect size using Cohen's D
# https://rcompanion.org/handbook/I_03.html
effect_sizes <- lapply(split(cmips_groups_long, cmips_groups_long$variable), 
                       function(x) cohensD(value~stress_group, x)) %>%
  as_vector()

# Get LIWC vars names
variable_names <- names(test_results)

# Add all vectors into df
test_signf_results <- data.frame(variable_names, t_stat, deg_free, effect_sizes, p_val) %>%
  as_tibble() %>%
  # Set Bonferroni value and determine if p_val is beyond/more extreme
  mutate(bonf_value = bonf_value) %>%
  mutate(signf_at_bonf = p_val < bonf_value) %>%
  # Keep only significant differences
  filter(signf_at_bonf == TRUE)
test_signf_results

# Save data to csv
write_csv(test_signf_results, "results/group_comparisons/label_IHS_mean.csv")
