# CREATE FEATURES - THIRD SET ---------------------------------------------

# Author = Cory J. Cascalheira
# Date = 04/23/2024

# The purpose of this script is to create features for CMIPS. This script prepares
# code to use in a Python script.

# The following features are prepared:
# - Stress classifier
# - Relaxation classifier 

# LOAD DEPENDENCIES AND IMPORT --------------------------------------------

# Load libraries
library(tidyverse)
library(textclean)
library(tidytext)
library(textstem)
library(lubridate)

# ...1) Transform to CSV --------------------------------------------------

# Transform the file to CSV and name columns for Python

# Import data
stress_relax_df <- import("data/util/tensi_strength_data/StressDev3084.txt") %>%
  as_tibble() %>%
  rename(relax = V1, stress = V2, text = V3)

# Export to Python to demojize text
write_csv(stress_relax_df, "data/util/tensi_strength_data/stress_relax_df.csv")

# ...2) For Cleaning ------------------------------------------------------

# Import stress & relax data
stress_relax_df <- read_csv("data/util/tensi_strength_data/stress_relax_df_demojized.csv") %>%
  select(-...1) %>%
  # Add a unique identifier
  mutate(
    post_id = 1:nrow(.),
    post_id = paste0("id_", post_id)
  )

# Import CMIPS data
cmips_df <- read_csv("data/participants/combined_social_media/social_media_posts_demojized.csv") %>%
  select(-...1) %>%
  rename(text = posts_comments)

# PREPROCESS THE TEXT -----------------------------------------------------

# ...1) Relax and Stress Data ---------------------------------------------

# Do some basic cleaning up 
stress_relax_df <- stress_relax_df %>%
  # Remove URLs
  mutate(text = str_remove_all(text, regex("(http|ftp|https):\\/\\/([\\w_-]+(?:(?:\\.[\\w_-]+)+))([\\w.,@?^=%&:\\/~+#-]*[\\w@?^=%&\\/~+#-])"))) %>%
  # Remove usernames / tags
  mutate(text = str_remove_all(text, regex("@[A-z0-9]+"))) %>%
  # Remove # symbol
  mutate(text = str_replace_all(text, "#", " ")) %>%
  # Replace emoji ":" and "_" with " "
  mutate(text = str_replace_all(text, regex(":|_+"), " "))

# Remove empty posts
stress_relax_df <- stress_relax_df %>%
  # Remove "RT" from the text
  mutate(
    text = str_remove_all(text, regex("^RT | RT | RT$", ignore_case = TRUE))
  ) %>%
  # Trim White space
  mutate(text = str_trim(text)) %>%
  # Remove empty posts
  filter(!is.na(text))

# Tokenize and replace contractions
stress_relax_df <- stress_relax_df %>%
  # Convert to lowercase
  mutate(text = tolower(text)) %>%
  # Unnest the tokens
  unnest_tokens(output = "word", input = "text") %>%
  # Replace contractions
  mutate(text = replace_contraction(word)) %>% 
  select(-word) %>%
  rename(word = text) %>%
  # Concatenate the words into a single string
  group_by(post_id) %>%
  mutate(text = paste(word, collapse = " ")) %>%
  ungroup() %>%
  distinct(text, .keep_all = TRUE) %>%
  select(-word)

# Execute remaining cleaning steps on the other variables
stress_relax_df <- stress_relax_df %>%
  # Unnest the tokens
  unnest_tokens(output = "word", input = "text") %>%
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
  # Lemmatize the words (no stemming)
  # https://cran.r-project.org/web/packages/textstem/readme/README.html
  mutate(word = lemmatize_words(word)) %>%
  # Concatenate the words into a single string
  group_by(post_id) %>%
  mutate(text = paste(word, collapse = " ")) %>%
  ungroup() %>%  # Trim white space
  mutate(text = str_trim(text)) %>%
  # Keep one instance of the unique posts
  distinct(post_id, .keep_all = TRUE) %>%
  select(-word)
stress_relax_df

# ...2) CMIPS Data --------------------------------------------------------

# Compress dataset to day level
cmips_df <- cmips_df %>%
  # Extract date, removing time
  mutate(timestamp = as_date(timestamp)) %>%
  group_by(participant_id, timestamp) %>%
  summarize(
    text = paste(text, collapse = " ")
  ) %>%
  ungroup()



cmips_df <- cmips_df %>%
  sample_n(size = 1000)



# Do some basic cleaning up 
cmips_df <- cmips_df %>%
  # Remove URLs
  mutate(text = str_remove_all(text, regex("(http|ftp|https):\\/\\/([\\w_-]+(?:(?:\\.[\\w_-]+)+))([\\w.,@?^=%&:\\/~+#-]*[\\w@?^=%&\\/~+#-])"))) %>%
  # Remove usernames / tags
  mutate(text = str_remove_all(text, regex("@[A-z0-9]+"))) %>%
  # Remove # symbol
  mutate(text = str_replace_all(text, "#", " ")) %>%
  # Replace emoji ":" and "_" with " "
  mutate(text = str_replace_all(text, regex(":|_+"), " "))

# Remove empty posts
cmips_df <- cmips_df %>%
  # Remove "RT" from the text
  mutate(
    text = str_remove_all(text, regex("^RT | RT | RT$", ignore_case = TRUE)),
    text = str_remove_all(text, regex("^NA | NA | NA$", ignore_case = TRUE))
  ) %>%
  # Trim White space
  mutate(text = str_trim(text)) %>%
  # Remove empty posts
  filter(!is.na(text))

# Tokenize and replace contractions
cmips_df <- cmips_df %>%
  # Convert to lowercase
  mutate(text = tolower(text)) %>%
  # Unnest the tokens
  unnest_tokens(output = "word", input = "text") %>%
  # Replace contractions
  mutate(text = replace_contraction(word)) %>% 
  select(-word) %>%
  rename(word = text) %>%
  # Concatenate the words into a single string
  group_by(participant_id, timestamp) %>%
  mutate(text = paste(word, collapse = " ")) %>%
  ungroup() %>%
  distinct(text, .keep_all = TRUE) %>%
  select(-word)

# Execute remaining cleaning steps on the other variables
cmips_df <- cmips_df %>%
  # Unnest the tokens
  unnest_tokens(output = "word", input = "text") %>%
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
  # Lemmatize the words (no stemming)
  # https://cran.r-project.org/web/packages/textstem/readme/README.html
  mutate(word = lemmatize_words(word)) %>%
  # Concatenate the words into a single string
  group_by(participant_id, timestamp) %>%
  mutate(text = paste(word, collapse = " ")) %>%
  ungroup() %>%  # Trim white space
  mutate(text = str_trim(text)) %>%
  # Keep one instance of the unique posts
  distinct(text, .keep_all = TRUE) %>%
  select(-word)
cmips_df

# CREATE N-GRAMS ----------------------------------------------------------

# Check distribution of codes
table(stress_relax_df$relax)
table(stress_relax_df$stress)

# Recode the relax and stress items
stress_relax_df <- stress_relax_df %>%
  mutate(
    relax = if_else(relax >= 2, 1, 0),
    stress = if_else(stress <= -2, 1, 0)
  )

# ...1) Unigrams ----------------------------------------------------------

# Top unigrams - relax
unigram_df_relax <- stress_relax_df %>%
  # Generate unigrams
  unnest_tokens(word, text, drop = FALSE) %>%
  count(relax, word) %>%
  arrange(desc(n))

# TF-IDF unigrams - relax
unigram_vector_relax <- unigram_df_relax %>%
  # Calculate tf-idf
  bind_tf_idf(word, relax, n) %>%
  # Get top tf-idf of unigrams for composite stress posts
  arrange(desc(tf_idf)) %>%
  filter(relax == 1) %>%
  # Select the top 100 n-grams
  head(n = 100) %>%
  pull(word)
unigram_vector_relax

# Top unigrams - stress
unigram_df_stress <- stress_relax_df %>%
  # Generate unigrams
  unnest_tokens(word, text, drop = FALSE) %>%
  # Remove stop words
  count(stress, word) %>%
  arrange(desc(n))

# TF-IDF unigrams - stress
unigram_vector_stress <- unigram_df_stress %>%
  # Calculate tf-idf
  bind_tf_idf(word, stress, n) %>%
  # Get top tf-idf of unigrams for composite stress posts
  arrange(desc(tf_idf)) %>%
  filter(stress == 1) %>%
  # Select the top 100 n-grams
  head(n = 100) %>%
  pull(word)
unigram_vector_stress

# ...2) Bigrams -----------------------------------------------------------

# Generate bigrams - relax
bigram_df_relax <- stress_relax_df %>%
  unnest_ngrams(bigram, text, n = 2, drop = FALSE) %>%
  # Separate the bigrams into two columns
  separate(bigram, c("word1", "word2")) %>%
  unite("bigram", c("word1", "word2"), sep = " ") %>%
  # Count top bigrams
  count(relax, bigram) %>%
  arrange(desc(n)) %>%
  # Remove whitespace
  mutate(
    whitespace = if_else(str_detect(bigram, regex("^ | $")), 1, 0)
  ) %>%
  filter(whitespace == 0)

# TF-IDF bigrams - relax
bigram_vector_relax <- bigram_df_relax %>%
  # Calculate tf-idf
  bind_tf_idf(bigram, relax, n) %>%
  # Get top tf-idf of unigrams for composite stress posts
  arrange(desc(tf_idf)) %>%
  filter(relax == 1) %>%
  # Select the top 100 n-grams
  head(n = 100) %>%
  pull(bigram)
bigram_vector_relax

# Generate bigrams - stress
bigram_df_stress <- stress_relax_df %>%
  unnest_ngrams(bigram, text, n = 2, drop = FALSE) %>%
  # Separate the bigrams into two columns
  separate(bigram, c("word1", "word2")) %>%
  unite("bigram", c("word1", "word2"), sep = " ") %>%
  # Count top bigrams
  count(stress, bigram) %>%
  arrange(desc(n)) %>%
  # Remove whitespace
  mutate(
    whitespace = if_else(str_detect(bigram, regex("^ | $")), 1, 0)
  ) %>%
  filter(whitespace == 0)

# TF-IDF bigrams - stress
bigram_vector_stress <- bigram_df_stress %>%
  # Calculate tf-idf
  bind_tf_idf(bigram, stress, n) %>%
  # Get top tf-idf of unigrams for composite stress posts
  arrange(desc(tf_idf)) %>%
  filter(stress == 1) %>%
  # Select the top 100 n-grams
  head(n = 100) %>%
  pull(bigram)
bigram_vector_stress

# ASSIGN N-GRAMS: STRESS & RELAX ------------------------------------------

# ...1) Unigrams ----------------------------------------------------------

# Assign the unigrams as features
for (i in 1:length(unigram_vector_relax)) {
  
  # Get the n-grams
  ngram <- unigram_vector_relax[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(stress_relax_df$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  stress_relax_df[[ngram]] <- as.integer(x)  
}

# Assign the unigrams as features
for (i in 1:length(unigram_vector_stress)) {
  
  # Get the n-grams
  ngram <- unigram_vector_stress[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(stress_relax_df$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  stress_relax_df[[ngram]] <- as.integer(x)  
}

# ...2) Bigrams -----------------------------------------------------------

# Assign the bigrams as features
for (i in 1:length(bigram_vector_relax)) {
  
  # Get the n-grams
  ngram <- bigram_vector_relax[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(stress_relax_df$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  stress_relax_df[[ngram]] <- as.integer(x)  
}

# Assign the bigrams as features
for (i in 1:length(bigram_vector_stress)) {
  
  # Get the n-grams
  ngram <- bigram_vector_stress[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(stress_relax_df$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  stress_relax_df[[ngram]] <- as.integer(x)  
}

# ASSIGN N-GRAMS: CMIPS ---------------------------------------------------

# ...1) Unigrams ----------------------------------------------------------

# Assign the unigrams as features
for (i in 1:length(unigram_vector_relax)) {
  
  # Get the n-grams
  ngram <- unigram_vector_relax[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(cmips_df$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  cmips_df[[ngram]] <- as.integer(x)  
}

# Assign the unigrams as features
for (i in 1:length(unigram_vector_stress)) {
  
  # Get the n-grams
  ngram <- unigram_vector_stress[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(cmips_df$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  cmips_df[[ngram]] <- as.integer(x)  
}

# ...2) Bigrams -----------------------------------------------------------

# Assign the bigrams as features
for (i in 1:length(bigram_vector_relax)) {
  
  # Get the n-grams
  ngram <- bigram_vector_relax[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(cmips_df$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  cmips_df[[ngram]] <- as.integer(x)  
}

# Assign the bigrams as features
for (i in 1:length(bigram_vector_stress)) {
  
  # Get the n-grams
  ngram <- bigram_vector_stress[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(cmips_df$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  cmips_df[[ngram]] <- as.integer(x)  
}

# EXPORT ------------------------------------------------------------------

# Save data
write_csv(stress_relax_df, "data/util/tensi_strength_data/stress_relax_df_ngrams.csv")
write_csv(cmips_df, "data/util/tensi_strength_data/cmips_df_ngrams.csv")
