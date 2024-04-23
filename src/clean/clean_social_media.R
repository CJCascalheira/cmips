# CLEAN SOCIAL MEDIA DATA -------------------------------------------------

# Author = Cory Cascalheira
# Date = 04/14/2024

# The purpose of this script is to aggregate posts to the day level, remove 
# empty posts, and preprocess the text for use in most feature creation methods
# aside from behavioral engagement, LIWC, and word embeddings.

# LOAD DEPENDENCIES AND IMPORT DATA ---------------------------------------

# Load libraries
library(tidyverse)
library(janitor)
library(lubridate)
library(textclean)
library(tidytext)
library(textstem)

# Import data
social_media_posts <- read_csv("data/participants/combined_social_media/social_media_posts_demojized.csv") %>%
  select(-...1)

social_media_reactions <- read_csv("data/participants/combined_social_media/social_media_reactions.csv")

# Import covariates 
participants_shared_both_fbtw <- read_csv("data/participants/util/participants_shared_both_fbtw.csv") %>%
  mutate(covariate_both_fbtw = 1)
  
participants_shared_more_one_year <- read_csv("data/participants/util/participants_shared_more_one_year.csv") %>%
  mutate(covariate_more_one_year = 1)

# Import stop words
nltk_stopwords <- read_csv("data/util/NLTK_stopwords.csv")

# Add NLTK to tidytext stop words
stop_words <- stop_words %>%
  select(word) %>%
  bind_rows(nltk_stopwords) %>%
  distinct(word) %>%
  # Add words unique to this project
  bind_rows(data.frame(word = c("quot", "amp")))

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

# Check work
table(social_media_posts$covariate_both_fbtw)
table(social_media_posts$covariate_more_one_year)

# ...3) NLP Preprocessing ------------------------------------------------

# Do some basic cleaning up - shared across all features
social_media_posts <- social_media_posts %>%
  # Remove URLs
  mutate(posts_comments = str_remove_all(posts_comments, regex("(http|ftp|https):\\/\\/([\\w_-]+(?:(?:\\.[\\w_-]+)+))([\\w.,@?^=%&:\\/~+#-]*[\\w@?^=%&\\/~+#-])"))) %>%
  # Remove usernames / tags
  mutate(posts_comments = str_remove_all(posts_comments, regex("@[A-z0-9]+"))) %>%
  # Remove # symbol
  mutate(posts_comments = str_replace_all(posts_comments, "#", " ")) %>%
  # Replace emoji ":" and "_" with " "
  mutate(posts_comments = str_replace_all(posts_comments, regex(":|_+"), " "))

# Compress dataset to day level - shared across all features 
social_media_posts <- social_media_posts %>%
  # Extract date, removing time
  mutate(timestamp = as_date(timestamp)) %>%
  group_by(participant_id, timestamp) %>%
  summarize(
    posts_comments = paste(posts_comments, collapse = " ")
  ) %>%
  ungroup()

# Remove empty posts
social_media_posts <- social_media_posts %>%
  filter(posts_comments != "NA") %>%
  # Remove "NA" and "RT" from the text
  mutate(
    posts_comments = str_remove_all(posts_comments, regex("^NA | NA | NA$", ignore_case = TRUE)),
    posts_comments = str_remove_all(posts_comments, regex("^RT | RT | RT$", ignore_case = TRUE))
  ) %>%
  # Trim White space
  mutate(posts_comments = str_trim(posts_comments)) %>%
  # Remove empty posts
  filter(!is.na(posts_comments)) %>%
  filter(posts_comments != "")

# Copy the full, mostly unprocessed text for behavioral engagement, embeddings,
# and LIWC
social_media_posts_full <- social_media_posts

# Rename the social media posts object
social_media_posts_cleaned <- social_media_posts

# Tokenize and replace contractions
social_media_posts_cleaned <- social_media_posts_cleaned %>%
  # Convert to lowercase
  mutate(posts_comments = tolower(posts_comments)) %>%
  # Unnest the tokens
  unnest_tokens(output = "word", input = "posts_comments") %>%
  # Replace contractions
  mutate(posts_comments = replace_contraction(word)) %>% 
  select(-word) %>%
  rename(word = posts_comments)

# Execute remaining cleaning steps on the other variables
social_media_posts_cleaned <- social_media_posts_cleaned %>%
  # Trim white space
  mutate(word = str_trim(word)) %>%
  # Replace meaningful digit-related information with tokens
  mutate(
    # Year token
    word = if_else(str_detect(word, regex("^[:digit:]{4}$")), "tokenyear", word),
    # Date tokens
    word = if_else(str_detect(word, regex("^[:digit:]{1,2}st$|^[:digit:]{1,2}nd$|^[:digit:]{1,2}rd$|^[:digit:]{1,2}th$")), "tokendate", word),
    word = if_else(str_detect(word, regex("[:digit:]{2}\\.[:digit:]{2}\\.[:digit:]{2}")), "tokendate", word),
    word = if_else(str_detect(word, regex("[:digit:]{2}\\.[:digit:]{2}\\.[:digit:]{4}")), "tokendate", word),
    word = if_else(str_detect(word, regex("[:digit:]{2}s$|[:digit:]{4}s")), "tokendate", word),
    # Time token
    word = if_else(str_detect(word, regex("^[:digit:]{1}pm|^[:digit:]{1}am|^[:digit:]{2}pm|^[:digit:]{2}am")), "tokentime", word),
    # Money token
    word = if_else(str_detect(word, regex("^[:digit:]{1,4}k$")), "tokenmoney", word)
  ) %>%
  # Remove other digits without text associated with them
  mutate(only_digit = if_else(str_detect(word, regex("(?<![A-z])[:digit:](?![A-z])")), 1, 0)) %>%
  filter(only_digit == 0) %>%
  select(-only_digit) %>%
  # Manual clean up 
  mutate(
    # Change all apostrophes to be the same
    word = str_replace(word, "’", "'"),
    # Remove possessive
    word = str_remove(word, "'s"),
    # Rework common contractions not processed in the above code
    word = str_replace(word, "it's", "it is"),
    word = str_replace(word, "i'm", "i am"),
    word = str_replace(word, "don't", "do not"),
    word = str_replace(word, "haven't", "have not"),
    word = str_replace(word, "you're", "you are"),
    word = str_replace(word, "can't", "can not"),
    word = str_replace(word, "y'all", "you all"),
    word = str_replace(word, "i've", "i have"),
    word = str_replace(word, "that's", "that is"),
    word = str_replace(word, "didn't", "did not"),
    word = str_replace(word, "we're", "we are"),
    word = str_replace(word, "they're", "they are"),
    word = str_replace(word, "doesn't", "does not"),
    word = str_replace(word, "i'll", "i will"),
    word = str_replace(word, "isn't", "is not"),
    word = str_replace(word, "i'd", "i would"),
    word = str_replace(word, "won't", "will not"),
    word = str_replace(word, "aren't", "are not"),
    word = str_replace(word, "you've", "you have"),
    word = str_replace(word, "wasn't", "was not"),
    word = str_replace(word, "hadn't", "had not"),
    word = str_replace(word, "wouldn't", "would not"),
    word = str_replace(word, "we've", "we have"),
    word = str_replace(word, "shouldn't", "should not"),
    word = str_replace(word, "it'd", "it would"),
    word = str_replace(word, "you'll", "you will"),
    word = str_replace(word, "we'll", "we will"),
    word = str_replace(word, "weren't", "were not"),
    word = str_replace(word, "they're", "they are"),
    word = str_replace(word, "who've", "who have"),
    word = str_replace(word, "ain't", "ain't"),
    word = str_replace(word, "hasn't", "has not"),
    word = str_replace(word, "that'd", "that would"),
    word = str_replace(word, "they'll", "rhey will"),
    word = str_replace(word, "it'll", "it will"),
    word = str_replace(word, "couldn't", "could not"),
    word = str_replace(word, "ya'll", "you all")
  ) %>% 
  # Remove the stop words
  filter(!(word %in% stop_words$word)) %>%
  # Lemmatize the words (no stemming)
  # https://cran.r-project.org/web/packages/textstem/readme/README.html
  mutate(word = lemmatize_words(word)) %>%
  # Concatenate the words into a single string
  group_by(participant_id, timestamp) %>%
  mutate(posts_comments = paste(word, collapse = " ")) %>%
  ungroup() %>%
  # Keep one instance of the unique posts
  distinct(participant_id, timestamp, .keep_all = TRUE) %>%
  select(-word)
social_media_posts_cleaned

# EXPORT DATA -------------------------------------------------------------

# Save the cleaned files
write_csv(social_media_posts_full, "data/participants/cleaned/social_media_posts_full.csv")
write_csv(social_media_posts_cleaned, "data/participants/cleaned/social_media_posts_cleaned.csv")
write_csv(social_media_reactions, "data/participants/cleaned/social_media_reactions.csv")
