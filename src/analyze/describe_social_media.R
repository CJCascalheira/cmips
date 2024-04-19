# DESCRIBE SOCIAL MEDIA DATA ----------------------------------------------

# Author = Cory Cascalheira
# Date = 04/14/2024

# The purpose of this script is to calculate descriptive statistics of the 
# social media data. 

# LOAD DEPENDENCIES AND IMPORT DATA ---------------------------------------

# Load libraries
library(tidyverse)
library(readxl)
library(janitor)

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

# ANALYZE PREPROCESSED DATA -----------------------------------------------


# EXPORT IDS TO MAKE COVARIATES -------------------------------------------

# Save file to make covariates
write_csv(participants_shared_both_fbtw, "data/participants/util/participants_shared_both_fbtw.csv")
write_csv(participants_shared_more_one_year, "data/participants/util/participants_shared_more_one_year.csv")

# Save file of participants who shared the wrong social media data
write_csv(shared_wrong_data, "data/participants/util/shared_wrong_data.csv")
