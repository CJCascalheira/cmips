# CREATE FEATURES - THIRD SET ---------------------------------------------

# Author = Cory J. Cascalheira
# Date = 04/19/2024

# The purpose of this script is to create features for CMIPS. Code is taken from
# the LGBTQ+ MiSSoM dataset.

# The following features are created:
# - Ngrams

# Decided to use caution with n-grams because the sample is so small--the n-grams being
# produced were person-dependent (i.e., unique names) that would likely not
# generalize to other models.

# LOAD DEPENDENCIES AND IMPORT --------------------------------------------

# Load libraries
library(tidyverse)
library(tidytext)

# Import data
cmips_social_media <- read_csv("data/participants/cleaned/social_media_posts_cleaned.csv")
cmips_surveys <- read_csv("data/participants/for_analysis/cmips_surveys_full.csv")

# PREPROCESS --------------------------------------------------------------

# Select just participants who shared their data
participant_ids <- cmips_social_media %>%
  distinct(participant_id)

# Filter the survey
cmips_surveys <- cmips_surveys %>%
  filter(ParticipantID %in% participant_ids$participant_id)

# Merge all social media data to participant level
cmips_social_media <- cmips_social_media %>%
  group_by(participant_id) %>%
  summarize(posts_comments = paste(posts_comments, collapse = " ")) %>%
  # Rename the posts variable
  rename(text = posts_comments)

# Distinguish between the most stressed vs. least stressed people
cmips_stress_aggregate <- cmips_surveys %>%
  select(ParticipantID, stress_posting, starts_with("label")) %>%
  # Unite the stress labels
  mutate(
    total_above_mean_stress = select(., label_StressCT:label_IHS_mean) %>% 
      rowSums(na.rm = TRUE)
  ) %>%
  # Drop the labels
  select(-starts_with("label"))

# What is the mean aggregate stress
mean_stressors <- mean(cmips_stress_aggregate$total_above_mean_stress)

# People with highest level of stress who are most likely to disclose stress 
# based on their self-report in survey
participants_disclose_high_stress <- cmips_stress_aggregate %>%
  # People who post about stress
  filter(stress_posting == 1) %>%
  # People above the mean of the total_above_mean_stress composite variable
  filter(total_above_mean_stress > mean_stressors) %>%
  pull(ParticipantID)
length(participants_disclose_high_stress)

# People with lowest level of stress who are least likely to disclose stress 
# based on their self-report in survey
participants_no_disclose_low_stress <- cmips_stress_aggregate %>%
  # People who DO NOT post about stress
  filter(stress_posting == 0) %>%
  # People below the mean of the total_above_mean_stress composite variable
  filter(total_above_mean_stress < mean_stressors) %>%
  pull(ParticipantID)
length(participants_no_disclose_low_stress)

# Create a stress variable to serve as n-grams label
cmips_social_media <- cmips_social_media %>%
  # Get the participants for whom the variable can be calculated
  filter(participant_id %in% c(participants_no_disclose_low_stress, 
                               participants_disclose_high_stress)) %>%
  # Create the variable
  mutate(
    composite_stress = if_else(participant_id %in% participants_disclose_high_stress, 
                               1, 0)
  )

# 1) N-GRAMS --------------------------------------------------------------

# Top unigrams
unigram_df <- cmips_social_media %>%
  # Generate unigrams
  unnest_tokens(word, text, drop = FALSE) %>%
  # Remove stop words
  count(composite_stress, word) %>%
  arrange(desc(n)) %>%
  # Remove stop words
  filter(!(word %in% stop_words$word)) %>%
  # Clean up based on remaining stop words
  mutate(
    stop_word = if_else(str_detect(word, regex("^na$|^im$|that's|i’m|it’s|you’re|don’t|dont|It|can’t|lt|he’s|she’s|i’ve|doesn’t|didn’t|isn’t|there’s|that'll|how’s|they’ll|it’ll|would've|we’ll|they’ve|shouldn’t|that’s|i’ll|they’re|aren’t|i’d|won’t|what’s|you’ve|we’re|wouldn’t|haven’t|wasn’t|y'all|let’s|here’s|who’s|you’ll|couldn’t|weren’t|hasn’t|we’ve|ain’t|you’d|y’all")), 1, 0) 
  ) %>%
  # Remove remaining stop words
  filter(stop_word == 0) %>%
  select(-stop_word)

# TF-IDF unigrams
unigram_vector <- unigram_df %>%
  # Calculate tf-idf
  bind_tf_idf(word, composite_stress, n) %>%
  # Get top tf-idf of unigrams for composite stress posts
  arrange(desc(tf_idf)) %>%
  filter(composite_stress == 1) %>%
  # Select the top 100 n-grams
  head(n = 100) %>%
  pull(word)

# Generate bigrams
bigram_df <- cmips_social_media %>%
  # Select key columns
  select(participant_id, text, composite_stress) %>%
  unnest_ngrams(bigram, text, n = 2, drop = FALSE) %>%
  # Separate the bigrams into two columns
  separate(bigram, c("word1", "word2")) %>%
  # Remove stop words
  filter(!(word1 %in% stop_words$word)) %>%
  filter(!(word2 %in% stop_words$word)) %>%
  # Clean up based on remaining stop words
  mutate(
    stop_word1 = if_else(str_detect(word1, regex("^im$|that's|i’m|it’s|you’re|don’t|dont|It|can’t|lt|he’s|she’s|i’ve|doesn’t|didn’t|isn’t|there’s|that'll|how’s|they’ll|it’ll|would've|we’ll|they’ve|shouldn’t|that’s|i’ll|they’re|aren’t|i’d|won’t|what’s|you’ve|we’re|wouldn’t|haven’t|wasn’t|y'all|let’s|here’s|who’s|you’ll|couldn’t|weren’t|hasn’t|we’ve|ain’t|you’d|y’all|amp")), 1, 0),
    stop_word2 = if_else(str_detect(word2, regex("^im$|that's|i’m|it’s|you’re|don’t|dont|It|can’t|lt|he’s|she’s|i’ve|doesn’t|didn’t|isn’t|there’s|that'll|how’s|they’ll|it’ll|would've|we’ll|they’ve|shouldn’t|that’s|i’ll|they’re|aren’t|i’d|won’t|what’s|you’ve|we’re|wouldn’t|haven’t|wasn’t|y'all|let’s|here’s|who’s|you’ll|couldn’t|weren’t|hasn’t|we’ve|ain’t|you’d|y’all")), 1, 0)
  ) %>%
  filter(stop_word1 == 0, stop_word2 == 0) %>%
  unite("bigram", c("word1", "word2"), sep = " ") %>%
  # Count top bigrams
  count(composite_stress, bigram) %>%
  arrange(desc(n))

# TF-IDF bigrams
bigram_vector <- bigram_df %>%
  # Calculate tf-idf
  bind_tf_idf(bigram, composite_stress, n) %>%
  # Get top tf-idf of unigrams for composite stress posts
  arrange(desc(tf_idf)) %>%
  filter(composite_stress == 1) %>%
  # Select the top 100 n-grams
  head(n = 100) %>%
  pull(bigram)

# Generate trigrams
trigram_df <- cmips_social_media %>%
  # Select key columns
  select(participant_id, text, composite_stress) %>%
  unnest_ngrams(trigram, text, n = 3, drop = FALSE) %>%
  # Separate into three columns
  separate(trigram, c("word1", "word2", "word3"), sep = " ") %>%
  # Remove stop words
  filter(!word1 %in% stop_words$word,
         !word2 %in% stop_words$word,
         !word3 %in% stop_words$word) %>%
  # Clean up based on remaining stop words
  mutate(
    stop_word1 = if_else(str_detect(word1, regex("^im$|that's|i’m|it’s|you’re|don’t|dont|It|can’t|lt|he’s|she’s|i’ve|doesn’t|didn’t|isn’t|there’s|that'll|how’s|they’ll|it’ll|would've|we’ll|they’ve|shouldn’t|that’s|i’ll|they’re|aren’t|i’d|won’t|what’s|you’ve|we’re|wouldn’t|haven’t|wasn’t|y'all|let’s|here’s|who’s|you’ll|couldn’t|weren’t|hasn’t|we’ve|ain’t|you’d|y’all")), 1, 0),
    stop_word2 = if_else(str_detect(word2, regex("^im$|that's|i’m|it’s|you’re|don’t|dont|It|can’t|lt|he’s|she’s|i’ve|doesn’t|didn’t|isn’t|there’s|that'll|how’s|they’ll|it’ll|would've|we’ll|they’ve|shouldn’t|that’s|i’ll|they’re|aren’t|i’d|won’t|what’s|you’ve|we’re|wouldn’t|haven’t|wasn’t|y'all|let’s|here’s|who’s|you’ll|couldn’t|weren’t|hasn’t|we’ve|ain’t|you’d|y’all")), 1, 0) ,
    stop_word3 = if_else(str_detect(word3, regex("^im$|that's|i’m|it’s|you’re|don’t|dont|It|can’t|lt|he’s|she’s|i’ve|doesn’t|didn’t|isn’t|there’s|that'll|how’s|they’ll|it’ll|would've|we’ll|they’ve|shouldn’t|that’s|i’ll|they’re|aren’t|i’d|won’t|what’s|you’ve|we’re|wouldn’t|haven’t|wasn’t|y'all|let’s|here’s|who’s|you’ll|couldn’t|weren’t|hasn’t|we’ve|ain’t|you’d|y’all")), 1, 0) 
  ) %>%
  # Remove contracted stop words
  filter(
    stop_word1 == 0,
    stop_word2 == 0,
    stop_word3 == 0
  ) %>%
  # Combine into trigrams
  unite("trigram", c("word1", "word2", "word3"), sep = " ") %>%
  count(composite_stress, trigram) %>%
  arrange(desc(n))

# TF-IDF Trigrams
trigram_vector <- trigram_df %>%
  # Manual remove of nonsense
  mutate(remove = if_else(str_detect(trigram, "\\d|ðÿ|^amp |amp | amp$|NA NA NA|poll$|jfe|_link|link_|playlist 3948ybuzmcysemitjmy9jg si|complete 3 surveys|gmail.com mailto:hellogoodbis42069 gmail.com|hellogoodbis42069 gmail.com mailto:hellogoodbis42069|comments 7n2i gay_marriage_debunked_in_2_minutes_obama_vs_alan|debatealtright comments 7n2i|gift card|amazon|action hirewheller csr|energy 106 fm|form sv_a3fnpplm8nszxfb width|â íœê í|âˆ âˆ âˆ"), 1, 0)) %>%
  filter(remove == 0) %>%
  # Calculate tf-idf
  bind_tf_idf(trigram, composite_stress, n) %>%
  # Get top tf-idf of unigrams for composite stress posts
  arrange(desc(tf_idf)) %>%
  filter(composite_stress == 1) %>%
  # Select the top 100 n-grams
  head(n = 100) %>%
  pull(trigram)

# 2) ASSIGN N-GRAMS -------------------------------------------------------

# Assign the unigrams as features
for (i in 1:length(unigram_vector)) {
  
  # Get the n-grams
  ngram <- unigram_vector[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(cmips_social_media$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  cmips_social_media[[ngram]] <- as.integer(x)  
}

# Assign the bigrams as features
for (i in 1:length(bigram_vector)) {
  
  # Get the n-grams
  ngram <- bigram_vector[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(cmips_social_media$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  cmips_social_media[[ngram]] <- as.integer(x)  
}

# Assign the trigrams as features
for (i in 1:length(trigram_vector)) {
  
  # Get the n-grams
  ngram <- trigram_vector[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(cmips_social_media$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  cmips_social_media[[ngram]] <- as.integer(x)  
}