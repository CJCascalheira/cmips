# CREATE FEATURES - FIRST SET ---------------------------------------------

# Author = Cory Cascalheira
# Date = 04/14/2024

# The purpose of this script is to create the first set of social media 
# features. Specifically, this script:
# - Creates the behavioral engagement features

# References
# - https://stackoverflow.com/questions/6038061/regular-expression-to-find-urls-within-a-string

# LOAD DEPENDENCIES AND IMPORT --------------------------------------------

# Libraries
library(tidyverse)
library(lubridate)
library(hms)

# Import data
social_media_posts_comments <- read_csv("data/participants/cleaned/social_media_posts_full.csv")

social_media_reactions <- read_csv("data/participants/cleaned/social_media_reactions.csv")

social_media_with_urls <- read_csv("data/participants/combined_social_media/social_media_posts_demojized.csv") %>%
  select(-...1)

# BEHAVIORAL ENGAGEMENT ---------------------------------------------------

# ...1) The average number of posts/comments between 12am and 6am ---------

# Set the start and end times
start_time <- as_hms(ymd_hms("2022-01-11 00:00:00 UTC"))
end_time <- as_hms(ymd_hms("2022-01-11 06:00:00 UTC"))

# Calculate the feature
feat_be_avg_12_6 <- social_media_posts_comments %>%
  mutate(just_time = as_hms(timestamp)) %>%
  mutate(btwn_12_6 = if_else(between(just_time, start_time, end_time), 1, 0)) %>%
  group_by(participant_id) %>%
  summarize(
    total_posts = n(),
    total_12_6 = sum(btwn_12_6, na.rm = TRUE),
    be_avg_12_6 = total_12_6 / total_posts
  ) %>%
  select(-starts_with("total")) %>%
  ungroup()

# ...2) The average number of URLs used -----------------------------------

# Create the URL feature
feat_be_avg_n_urls <- social_media_with_urls %>%
  mutate(n_urls = str_count(posts_comments, regex("(http|ftp|https):\\/\\/([\\w_-]+(?:(?:\\.[\\w_-]+)+))([\\w.,@?^=%&:\\/~+#-]*[\\w@?^=%&\\/~+#-])"))) %>%
  group_by(participant_id) %>%
  summarize(
    total_posts = n(),
    total_urls = sum(n_urls, na.rm = TRUE),
    be_avg_n_urls = total_urls / total_posts
  ) %>%
  select(-starts_with("total")) %>%
  ungroup()

# ...3) The average number of hashtags used -------------------------------

# Create the hashtag feature
feat_be_avg_hashtags <- social_media_posts_comments %>%
  mutate(n_hashtag = str_count(posts_comments, regex("#[:alnum:]+"))) %>%
  group_by(participant_id) %>%
  summarize(
    total_posts = n(),
    total_hashtags = sum(n_hashtag, na.rm = TRUE),
    be_avg_hashtags = total_hashtags / total_posts
  ) %>%
  select(-starts_with("total")) %>%
  ungroup()

# ...4) The average number of posts/comments per day ----------------------

# Calculate the feature
feat_be_avg_daily_posts <- social_media_posts_comments %>%
  # Extract date, removing time
  mutate(timestamp = as_date(timestamp)) %>%
  # For each user, count posts per day
  count(participant_id, timestamp) %>%
  rename(n_posts = n) %>%
  group_by(participant_id) %>%
  summarize(
    be_avg_daily_posts = mean(n_posts)
  ) %>%
  ungroup()

# ...5) Total number of posts/comments  -----------------------------------

# Create the feature
feat_be_total_n_posts <- social_media_posts_comments %>%
  group_by(participant_id) %>%
  summarize(
    be_total_n_posts = n()
  ) %>%
  ungroup()

# ...6) Total number likes/reactions --------------------------------------

# Create the feature
feat_be_total_n_reactions <- social_media_reactions %>%
  group_by(participant_id) %>%
  summarize(
    be_total_n_reactions = n()
  ) %>%
  ungroup()

# ...7) Maximum number posts/comments -------------------------------------

# Create feature
feat_be_max_posts_day <- social_media_posts_comments %>%
  # Extract date, removing time
  mutate(timestamp = as_date(timestamp)) %>%
  # For each user, count posts per day
  count(participant_id, timestamp) %>%
  rename(n_posts = n) %>%
  group_by(participant_id) %>%
  summarize(
    be_max_posts_day = max(n_posts)
  )

# COMBINE ALL FEATURES ----------------------------------------------------

feat_be_avg_12_6
feat_be_avg_n_urls
feat_be_avg_hashtags
feat_be_avg_daily_posts
feat_be_total_n_posts
feat_be_total_n_reactions
feat_be_max_posts_day

