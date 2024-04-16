# CLEAN SOCIAL MEDIA DATA -------------------------------------------------

# Author = Cory Cascalheira
# Date = 04/14/2024

# The purpose of this script is to aggregate posts to the day level, remove 
# empty posts, and preprocess the text for use in most feature creation methods
# aside from LIWC and word embeddings.

# LOAD DEPENDENCIES AND IMPORT DATA ---------------------------------------

# Load libraries
library(tidyverse)
library(janitor)
library(lubridate)

# Import data
social_media_posts <- read_csv("data/participants/combined_social_media/social_media_posts_full.csv")
social_media_reactions <- read_csv("data/participants/combined_social_media/social_media_reactions.csv")

# Import covariates 
participants_shared_both_fbtw <- read_csv("data/participants/util/participants_shared_both_fbtw.csv") %>%
  mutate(covariate_both_fbtw = 1)
  
participants_shared_more_one_year <- read_csv("data/participants/util/participants_shared_more_one_year.csv") %>%
  mutate(covariate_more_one_year = 1)

# CLEAN & ADD COVARIATES --------------------------------------------------

# ...1) Basic Cleaning ----------------------------------------------------

# How many posts BEFORE missing data removal?
nrow(social_media_posts)

# How many reactions BEFORE missing data removal?
nrow(social_media_reactions)

# Any missing likes?
social_media_reactions <- social_media_reactions %>%
  filter(!is.na(reaction))

# How many reactions with missing data removed?
nrow(social_media_reactions)

# Remove participants with missing timestamps - are these deleted posts?
social_media_posts <- social_media_posts %>%
  filter(!is.na(timestamp)) %>%
  filter(!is.na(posts_comments))

# How many posts with missing data removed?
nrow(social_media_posts)

# Remove the NA characters that were converted to NA strings
social_media_posts <- social_media_posts %>%
  mutate(posts_comments = str_remove_all(posts_comments, "NA "))

# ...2) Add Covariates ----------------------------------------------------

# Add covariates to the data
social_media_posts <- social_media_posts %>%
  left_join(participants_shared_both_fbtw) %>%
  left_join(participants_shared_more_one_year) %>%
  # Convert NAs to zeros
  mutate(
    covariate_both_fbtw = if_else(is.na(covariate_both_fbtw), 0, covariate_both_fbtw),
    covariate_more_one_year = if_else(is.na(covariate_more_one_year), 0, covariate_more_one_year)
  )

# Replace emoji ":" and "_" with " "

# Check work
table(social_media_posts$covariate_both_fbtw)
table(social_media_posts$covariate_more_one_year)

# ...3) NLP Preprocessing for Most Features -------------------------------

# Compress dataset to day level
social_media_posts <- social_media_posts %>%
  # Extract date, removing time
  mutate(timestamp = as_date(timestamp)) %>%
  group_by(participant_id, timestamp) %>%
  summarize(
    posts_comments = paste(posts_comments, collapse = " ")
  ) %>%
  ungroup()

# Copy the full, unprocessed text for embeddings and LIWC
social_media_posts_full <- social_media_posts

# EXPORT DATA -------------------------------------------------------------

# Save the cleaned files
write_csv(social_media_posts_full, "data/participants/cleaned/social_media_posts_full.csv")
