# COMBINE SOCIAL MEDIA DATA -----------------------------------------------

# Author = Cory Cascalheira
# Date = 04/11/2024

# The purpose of this script is to combine all the social media data.

# LOAD DEPENDENCIES AND IMPORT --------------------------------------------

# Libraries
library(tidyverse)
library(lubridate)
library(hms)

# ...1) Facebook ----------------------------------------------------------

# Import data
facebook_reactions1 <- read_csv("data/raw/social_media/facebook_likes_and_reactions_html_data.csv") %>%
  mutate(timestamp = mdy_hms(timestamp)) %>%
  mutate(reaction = tolower(reaction)) %>%
  # Extract the type of reaction
  mutate(reaction = str_extract(reaction, regex("[A-z]+(?=\\.[A-z]+$)"))) %>%
  select(-actor)

facebook_reactions2 <- read_csv("data/raw/social_media/facebook_likes_and_reactions_json_data.csv") %>%
  mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(reaction = tolower(reaction)) %>%
  select(-actor)

facebook_posts1 <- read_csv("data/raw/social_media/facebook_posts_html_data.csv") %>%
  mutate(timestamp = mdy_hms(timestamp)) %>%
  # Put all text into one column
  unite(col = "posts_comments", description:post, sep = " ") %>%
  # Drop unnecessary columns
  select(-title)

facebook_posts2 <- read_csv("data/raw/social_media/facebook_posts_json_data.csv") %>%
  mutate(timestamp = as_datetime(timestamp)) %>% 
  # Unite the media text with posts
  unite(col = "posts_comments", media_description:post, sep = " ") %>%
  # Drop unnecessary columns
  select(participant_id, timestamp, posts_comments)

facebook_comments1 <- read_csv("data/raw/social_media/facebook_comments_html_data.csv") %>%
  select(-timestamp, -comment_author) %>%
  rename(timestamp = comment_timestamp) %>%
  mutate(timestamp = mdy_hms(timestamp)) %>%
  select(participant_id, timestamp, posts_comments = comment_content) 

facebook_comments2 <- read_csv("data/raw/social_media/facebook_comments_json_data.csv") %>%
  select(-timestamp, -comment_author) %>%
  rename(timestamp = comment_timestamp) %>%
  mutate(timestamp = as_datetime(timestamp)) %>%
  select(participant_id, timestamp, posts_comments = comment_content)

# Initial combine
facebook_reactions <- bind_rows(facebook_reactions1, facebook_reactions2) %>%
  mutate(platform = "facebook")
facebook_posts <- bind_rows(facebook_posts1, facebook_posts2)
facebook_comments <- bind_rows(facebook_comments1, facebook_comments2)

# Combine posts and comments
facebook_posts_comments <- bind_rows(facebook_comments, facebook_posts) %>%
  arrange(participant_id, timestamp) %>%
  mutate(platform = "facebook")

# Check timestamp work
facebook_reactions1 %>%
  filter(is.na(timestamp))

facebook_reactions2 %>%
  filter(is.na(timestamp))

facebook_posts1 %>%
  filter(is.na(timestamp))

facebook_posts2 %>%
  filter(is.na(timestamp))

facebook_comments1 %>%
  filter(is.na(timestamp))

facebook_comments2 %>%
  filter(is.na(timestamp))

# Take a closer look at Facebook posts with missing data
participants_with_missing_data <- facebook_posts2 %>%
  filter(is.na(timestamp)) %>%
  distinct(participant_id) %>%
  pull(participant_id)
participants_with_missing_data

# Check if all data is missing for these 21 participants 
facebook_posts2 %>%
  filter(participant_id %in% participants_with_missing_data) %>%
  # Is the timestamp missing?
  mutate(
    miss_time = if_else(is.na(timestamp), 1, 0)
  ) %>%
  count(participant_id, miss_time)

# No more missing data!

# ...2) Twitter/X ---------------------------------------------------------

# Import data
twitter_reactions <- read_csv("data/raw/social_media/twitter_like_js_data.csv") %>%
  select(-tweet_id, -expanded_url) %>%
  mutate(reaction = "like") %>%
  mutate(platform = "twitter/x")

twitter_posts_comments <- read_csv("data/raw/social_media/twitter_tweet_js_data.csv") %>%
  select(-tweet_id, -retweeted, -hashtags, -source) %>%
  rename(posts_comments = full_text) %>%
  mutate(
    timestamp = str_extract(created_at, regex("(?<=[A-z]{3} )[A-z]{3} [0-9]{2}")),
    timestamp = paste(timestamp, str_extract(created_at, regex("[0-9]{4}$"))),
    timestamp = paste(timestamp, str_extract(created_at, regex("(?<=[A-z]{3} [A-z]{3} [0-9]{2} )[0-9]{2}:[0-9]{2}:[0-9]{2}"))),
    timestamp = mdy_hms(timestamp)
  ) %>%
  select(-created_at) %>%
  select(participant_id, timestamp, posts_comments) %>%
  mutate(platform = "twitter/x")

# ...3) Combine Across Platforms ------------------------------------------

# Combine the posts and comments
social_media_posts_comments <- bind_rows(twitter_posts_comments, facebook_posts_comments)

# Select only variables needed to combine
facebook_reactions0 <- facebook_reactions %>%
  select(participant_id, reaction, platform)

twitter_reactions0 <- twitter_reactions %>%
  select(participant_id, reaction, platform)

# Combine the reactions
social_media_reactions <- bind_rows(facebook_reactions0, twitter_reactions0)

# Check work
social_media_posts_comments %>%
  filter(is.na(timestamp))

# EXPORT COMBINED DATA ----------------------------------------------------

# Export social media data - post level
write_csv(social_media_posts_comments, "data/participants/combined_social_media/social_media_posts_full.csv")

# Export reactions
write_csv(social_media_reactions, "data/participants/combined_social_media/social_media_reactions.csv")
