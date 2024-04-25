# CREATE NGRAMS -----------------------------------------------------------

# Author = Cory J. Cascalheira
# Date = 04/25/2024

# The purpose of this script is to create DASSP n-grams.


# LOAD DEPENDENCIES AND IMPORT --------------------------------------------

# Libraries
library(tidyverse)
library(tidytext)
library(textclean)
library(lubridate)

# Import
depression_df <- read_csv("data/util/dassp/cleaned/depression_df_demojized.csv") %>%
  select(-...1) %>%
  rename(my_text = text)

anxiety_df <- read_csv("data/util/dassp/cleaned/anxiety_df_demojized.csv") %>%
  select(-...1) %>%
  rename(my_text = text)

stress_df <- read_csv("data/util/dassp/cleaned/stress_df_demojized.csv") %>%
  select(-...1) %>%
  rename(my_text = text)

suicide_df <- read_csv("data/util/dassp/cleaned/suicide_df_demojized.csv") %>%
  select(-...1) %>%
  rename(my_text = text)

ptsd_df <- read_csv("data/util/dassp/cleaned/ptsd_df_demojized.csv") %>%
  select(-...1) %>%
  rename(my_text = text)

social_media_posts <- read_csv("data/participants/combined_social_media/social_media_posts_demojized.csv") %>%
  select(-...1) %>%
  rename(my_text = posts_comments)

# CREATE N-GRAMS FOR DASSP ------------------------------------------------

# ...1) Depression --------------------------------------------------------

# Clean the data
depression_df <- depression_df %>%
  # Replace emoji ":" and "_" with " "
  mutate(my_text = str_replace_all(my_text, regex(":|_+"), " ")) %>%
  # Convert to lowercase
  mutate(my_text = tolower(my_text)) %>%
  # Replace all apostrophes
  mutate(my_text = str_replace_all(my_text, regex("’"), "'")) %>%
  # Get single words
  unnest_tokens(word, my_text, drop = TRUE) %>%
  # Remove stop words
  filter(!(word %in% stop_words$word)) %>%
  # Remove all punctuation 
  mutate(word = str_remove_all(word, regex("[:punct:]"))) %>%
  # Remove all digits
  mutate(word = str_remove_all(word, regex("[:digit:]"))) %>%
  # Remove empty words
  filter(word != "") %>%
  filter(word != " ") %>%
  filter(!is.na(word)) %>%
  # Manual cleaning 
  filter(
    !(word %in% c("^im$", "thats", "its", "youre", "dont", "it", "cant", "lt", "hes", "shes", "ive", "doesnt", "didnt", "isnt", "theres", "thatll", "hows", "theyll", "itll", "wouldve", "well", "theyve", "shouldnt", "thats", "ill", "theyre", "arent", "id", "wont", "whats", "youve", "were", "wouldnt", "havent", "wasnt", "yall", "lets", "heres", "whos", "youll", "couldnt", "werent", "hasnt", "weve", "aint", "youd"))
  ) %>%
  # Bind the data
  group_by(id, label) %>%
  summarize(
    my_text = paste(word, collapse = " ")
  ) %>%
  ungroup()

# Create unigrams
depression_df_unigrams <- depression_df %>%
  # Generate unigrams
  unnest_tokens(ngram, my_text, drop = FALSE) %>%
  # Remove stop words
  count(label, ngram) %>%
  arrange(desc(n))

# TF-IDF unigrams
depression_df_unigrams <- depression_df_unigrams %>%
  # Calculate tf-idf
  bind_tf_idf(ngram, label, n) %>%
  # Get top tf-idf of unigrams
  arrange(desc(tf_idf)) %>%
  filter(label == 1) %>%
  # Select the top 100 n-grams
  head(n = 100) %>%
  select(ngram)

# Generate bigrams 
depression_df_bigrams <- depression_df %>%
  unnest_ngrams(ngram, my_text, n = 2, drop = FALSE) %>%
  # Separate the bigrams into two columns
  separate(ngram, c("word1", "word2")) %>%
  unite("ngram", c("word1", "word2"), sep = " ") %>%
  # Count top bigrams
  count(label, ngram) %>%
  arrange(desc(n))

# TF-IDF bigrams
depression_df_bigrams <- depression_df_bigrams %>%
  # Calculate tf-idf
  bind_tf_idf(ngram, label, n) %>%
  # Get top tf-idf 
  arrange(desc(tf_idf)) %>%
  filter(label == 1) %>%
  # Select the top 100 n-grams
  head(n = 100) %>%
  select(ngram)

# ...2) Anxiety -----------------------------------------------------------

# Clean the data
anxiety_df <- anxiety_df %>%
  # Replace emoji ":" and "_" with " "
  mutate(my_text = str_replace_all(my_text, regex(":|_+"), " ")) %>%
  # Convert to lowercase
  mutate(my_text = tolower(my_text)) %>%
  # Replace all apostrophes
  mutate(my_text = str_replace_all(my_text, regex("’"), "'")) %>%
  # Get single words
  unnest_tokens(word, my_text, drop = TRUE) %>%
  # Remove stop words
  filter(!(word %in% stop_words$word)) %>%
  # Remove all punctuation 
  mutate(word = str_remove_all(word, regex("[:punct:]"))) %>%
  # Remove all digits
  mutate(word = str_remove_all(word, regex("[:digit:]"))) %>%
  # Remove empty words
  filter(word != "") %>%
  filter(word != " ") %>%
  filter(!is.na(word)) %>%
  # Manual cleaning 
  filter(
    !(word %in% c("^im$", "thats", "its", "youre", "dont", "it", "cant", "lt", "hes", "shes", "ive", "doesnt", "didnt", "isnt", "theres", "thatll", "hows", "theyll", "itll", "wouldve", "well", "theyve", "shouldnt", "thats", "ill", "theyre", "arent", "id", "wont", "whats", "youve", "were", "wouldnt", "havent", "wasnt", "yall", "lets", "heres", "whos", "youll", "couldnt", "werent", "hasnt", "weve", "aint", "youd"))
  ) %>%
  # Bind the data
  group_by(id, label) %>%
  summarize(
    my_text = paste(word, collapse = " ")
  ) %>%
  ungroup()

# Create unigrams
anxiety_df_unigrams <- anxiety_df %>%
  # Generate unigrams
  unnest_tokens(ngram, my_text, drop = FALSE) %>%
  # Remove stop words
  count(label, ngram) %>%
  arrange(desc(n))

# TF-IDF unigrams
anxiety_df_unigrams <- anxiety_df_unigrams %>%
  # Calculate tf-idf
  bind_tf_idf(ngram, label, n) %>%
  # Get top tf-idf of unigrams
  arrange(desc(tf_idf)) %>%
  filter(label == 1) %>%
  # Select the top 100 n-grams
  head(n = 100) %>%
  select(ngram)

# Generate bigrams 
anxiety_df_bigrams <- anxiety_df %>%
  unnest_ngrams(ngram, my_text, n = 2, drop = FALSE) %>%
  # Separate the bigrams into two columns
  separate(ngram, c("word1", "word2")) %>%
  unite("ngram", c("word1", "word2"), sep = " ") %>%
  # Count top bigrams
  count(label, ngram) %>%
  arrange(desc(n))

# TF-IDF bigrams
anxiety_df_bigrams <- anxiety_df_bigrams %>%
  # Calculate tf-idf
  bind_tf_idf(ngram, label, n) %>%
  # Get top tf-idf 
  arrange(desc(tf_idf)) %>%
  filter(label == 1) %>%
  # Select the top 100 n-grams
  head(n = 100) %>%
  select(ngram)

# ...3) Stress ------------------------------------------------------------

# Clean the data
stress_df <- stress_df %>%
  # Replace emoji ":" and "_" with " "
  mutate(my_text = str_replace_all(my_text, regex(":|_+"), " ")) %>%
  # Convert to lowercase
  mutate(my_text = tolower(my_text)) %>%
  # Replace all apostrophes
  mutate(my_text = str_replace_all(my_text, regex("’"), "'")) %>%
  # Get single words
  unnest_tokens(word, my_text, drop = TRUE) %>%
  # Remove stop words
  filter(!(word %in% stop_words$word)) %>%
  # Remove all punctuation 
  mutate(word = str_remove_all(word, regex("[:punct:]"))) %>%
  # Remove all digits
  mutate(word = str_remove_all(word, regex("[:digit:]"))) %>%
  # Remove empty words
  filter(word != "") %>%
  filter(word != " ") %>%
  filter(!is.na(word)) %>%
  # Manual cleaning 
  filter(
    !(word %in% c("^im$", "thats", "its", "youre", "dont", "it", "cant", "lt", "hes", "shes", "ive", "doesnt", "didnt", "isnt", "theres", "thatll", "hows", "theyll", "itll", "wouldve", "well", "theyve", "shouldnt", "thats", "ill", "theyre", "arent", "id", "wont", "whats", "youve", "were", "wouldnt", "havent", "wasnt", "yall", "lets", "heres", "whos", "youll", "couldnt", "werent", "hasnt", "weve", "aint", "youd"))
  ) %>%
  # Bind the data
  group_by(id, label) %>%
  summarize(
    my_text = paste(word, collapse = " ")
  ) %>%
  ungroup()

# Create unigrams
stress_df_unigrams <- stress_df %>%
  # Generate unigrams
  unnest_tokens(ngram, my_text, drop = FALSE) %>%
  # Remove stop words
  count(label, ngram) %>%
  arrange(desc(n))

# TF-IDF unigrams
stress_df_unigrams <- stress_df_unigrams %>%
  # Calculate tf-idf
  bind_tf_idf(ngram, label, n) %>%
  # Get top tf-idf of unigrams
  arrange(desc(tf_idf)) %>%
  filter(label == 1) %>%
  # Select the top 100 n-grams
  head(n = 100) %>%
  select(ngram)

# Generate bigrams 
stress_df_bigrams <- stress_df %>%
  unnest_ngrams(ngram, my_text, n = 2, drop = FALSE) %>%
  # Separate the bigrams into two columns
  separate(ngram, c("word1", "word2")) %>%
  unite("ngram", c("word1", "word2"), sep = " ") %>%
  # Count top bigrams
  count(label, ngram) %>%
  arrange(desc(n))

# TF-IDF bigrams
stress_df_bigrams <- stress_df_bigrams %>%
  # Calculate tf-idf
  bind_tf_idf(ngram, label, n) %>%
  # Get top tf-idf 
  arrange(desc(tf_idf)) %>%
  filter(label == 1) %>%
  # Select the top 100 n-grams
  head(n = 100) %>%
  select(ngram)

# ...4) Suicide -----------------------------------------------------------

# Clean the data
suicide_df <- suicide_df %>%
  # Replace emoji ":" and "_" with " "
  mutate(my_text = str_replace_all(my_text, regex(":|_+"), " ")) %>%
  # Convert to lowercase
  mutate(my_text = tolower(my_text)) %>%
  # Replace all apostrophes
  mutate(my_text = str_replace_all(my_text, regex("’"), "'")) %>%
  # Get single words
  unnest_tokens(word, my_text, drop = TRUE) %>%
  # Remove stop words
  filter(!(word %in% stop_words$word)) %>%
  # Remove all punctuation 
  mutate(word = str_remove_all(word, regex("[:punct:]"))) %>%
  # Remove all digits
  mutate(word = str_remove_all(word, regex("[:digit:]"))) %>%
  # Remove empty words
  filter(word != "") %>%
  filter(word != " ") %>%
  filter(!is.na(word)) %>%
  # Manual cleaning 
  filter(
    !(word %in% c("^im$", "thats", "its", "youre", "dont", "it", "cant", "lt", "hes", "shes", "ive", "doesnt", "didnt", "isnt", "theres", "thatll", "hows", "theyll", "itll", "wouldve", "well", "theyve", "shouldnt", "thats", "ill", "theyre", "arent", "id", "wont", "whats", "youve", "were", "wouldnt", "havent", "wasnt", "yall", "lets", "heres", "whos", "youll", "couldnt", "werent", "hasnt", "weve", "aint", "youd"))
  ) %>%
  # Bind the data
  group_by(id, label) %>%
  summarize(
    my_text = paste(word, collapse = " ")
  ) %>%
  ungroup()

# Create unigrams
suicide_df_unigrams <- suicide_df %>%
  # Generate unigrams
  unnest_tokens(ngram, my_text, drop = FALSE) %>%
  # Remove stop words
  count(label, ngram) %>%
  arrange(desc(n))

# TF-IDF unigrams
suicide_df_unigrams <- suicide_df_unigrams %>%
  # Calculate tf-idf
  bind_tf_idf(ngram, label, n) %>%
  # Get top tf-idf of unigrams
  arrange(desc(tf_idf)) %>%
  filter(label == 1) %>%
  # Select the top 100 n-grams
  head(n = 100) %>%
  select(ngram)

# Generate bigrams 
suicide_df_bigrams <- suicide_df %>%
  unnest_ngrams(ngram, my_text, n = 2, drop = FALSE) %>%
  # Separate the bigrams into two columns
  separate(ngram, c("word1", "word2")) %>%
  unite("ngram", c("word1", "word2"), sep = " ") %>%
  # Count top bigrams
  count(label, ngram) %>%
  arrange(desc(n))

# TF-IDF bigrams
suicide_df_bigrams <- suicide_df_bigrams %>%
  # Calculate tf-idf
  bind_tf_idf(ngram, label, n) %>%
  # Get top tf-idf 
  arrange(desc(tf_idf)) %>%
  filter(label == 1) %>%
  # Select the top 100 n-grams
  head(n = 100) %>%
  select(ngram)

# ...5) PTSD --------------------------------------------------------------

# Clean the data
ptsd_df <- ptsd_df %>%
  # Replace emoji ":" and "_" with " "
  mutate(my_text = str_replace_all(my_text, regex(":|_+"), " ")) %>%
  # Convert to lowercase
  mutate(my_text = tolower(my_text)) %>%
  # Replace all apostrophes
  mutate(my_text = str_replace_all(my_text, regex("’"), "'")) %>%
  # Get single words
  unnest_tokens(word, my_text, drop = TRUE) %>%
  # Remove stop words
  filter(!(word %in% stop_words$word)) %>%
  # Remove all punctuation 
  mutate(word = str_remove_all(word, regex("[:punct:]"))) %>%
  # Remove all digits
  mutate(word = str_remove_all(word, regex("[:digit:]"))) %>%
  # Remove empty words
  filter(word != "") %>%
  filter(word != " ") %>%
  filter(!is.na(word)) %>%
  # Manual cleaning 
  filter(
    !(word %in% c("^im$", "thats", "its", "youre", "dont", "it", "cant", "lt", "hes", "shes", "ive", "doesnt", "didnt", "isnt", "theres", "thatll", "hows", "theyll", "itll", "wouldve", "well", "theyve", "shouldnt", "thats", "ill", "theyre", "arent", "id", "wont", "whats", "youve", "were", "wouldnt", "havent", "wasnt", "yall", "lets", "heres", "whos", "youll", "couldnt", "werent", "hasnt", "weve", "aint", "youd"))
  ) %>%
  # Bind the data
  group_by(id, label) %>%
  summarize(
    my_text = paste(word, collapse = " ")
  ) %>%
  ungroup()

# Create unigrams
ptsd_df_unigrams <- ptsd_df %>%
  # Generate unigrams
  unnest_tokens(ngram, my_text, drop = FALSE) %>%
  # Remove stop words
  count(label, ngram) %>%
  arrange(desc(n))

# TF-IDF unigrams
ptsd_df_unigrams <- ptsd_df_unigrams %>%
  # Calculate tf-idf
  bind_tf_idf(ngram, label, n) %>%
  # Get top tf-idf of unigrams
  arrange(desc(tf_idf)) %>%
  filter(label == 1) %>%
  # Select the top 100 n-grams
  head(n = 100) %>%
  select(ngram)

# Generate bigrams 
ptsd_df_bigrams <- ptsd_df %>%
  unnest_ngrams(ngram, my_text, n = 2, drop = FALSE) %>%
  # Separate the bigrams into two columns
  separate(ngram, c("word1", "word2")) %>%
  unite("ngram", c("word1", "word2"), sep = " ") %>%
  # Count top bigrams
  count(label, ngram) %>%
  arrange(desc(n))

# TF-IDF bigrams
ptsd_df_bigrams <- ptsd_df_bigrams %>%
  # Calculate tf-idf
  bind_tf_idf(ngram, label, n) %>%
  # Get top tf-idf 
  arrange(desc(tf_idf)) %>%
  filter(label == 1) %>%
  # Select the top 100 n-grams
  head(n = 100) %>%
  select(ngram)

# ASSIGN N-GRAMS ----------------------------------------------------------

# ...1) Ensure Unique N-Grams ---------------------------------------------

# Combine and filter n-grams
ngrams_vector <- bind_rows(depression_df_unigrams, depression_df_bigrams) %>%
  bind_rows(anxiety_df_unigrams) %>%
  bind_rows(anxiety_df_bigrams) %>%
  bind_rows(stress_df_unigrams) %>%
  bind_rows(stress_df_bigrams) %>%
  bind_rows(suicide_df_unigrams) %>%
  bind_rows(suicide_df_bigrams) %>%
  bind_rows(ptsd_df_unigrams) %>%
  bind_rows(ptsd_df_bigrams) %>%
  # Keep distinct n-grams
  distinct(ngram) %>%
  pull(ngram)

# ...2) Clean CMIPS data --------------------------------------------------

# Compress dataset to day level
social_media_posts <- social_media_posts %>%
  # Extract date, removing time
  mutate(timestamp = as_date(timestamp)) %>%
  group_by(participant_id, timestamp) %>%
  summarize(
    my_text = paste(my_text, collapse = " ")
  ) %>%
  ungroup()

# Clean the data to make operations in n-gram cleaning
social_media_posts <- social_media_posts %>%
  # Replace emoji ":" and "_" with " "
  mutate(my_text = str_replace_all(my_text, regex(":|_+"), " ")) %>%
  # Convert to lowercase
  mutate(my_text = tolower(my_text)) %>%
  # Replace all apostrophes
  mutate(my_text = str_replace_all(my_text, regex("’"), "'")) %>%
  # Get single words
  unnest_tokens(word, my_text, drop = TRUE) %>%
  # Remove stop words
  filter(!(word %in% stop_words$word)) %>%
  # Remove all punctuation 
  mutate(word = str_remove_all(word, regex("[:punct:]"))) %>%
  # Remove all digits
  mutate(word = str_remove_all(word, regex("[:digit:]"))) %>%
  # Detect NA and remove
  mutate(word = str_remove(word, regex("^na$"))) %>%
  # Remove empty words
  filter(word != "") %>%
  filter(word != " ") %>%
  filter(!is.na(word)) %>%
  # Manual cleaning 
  filter(
    !(word %in% c("^im$", "thats", "its", "youre", "dont", "it", "cant", "lt", "hes", "shes", "ive", "doesnt", "didnt", "isnt", "theres", "thatll", "hows", "theyll", "itll", "wouldve", "well", "theyve", "shouldnt", "thats", "ill", "theyre", "arent", "id", "wont", "whats", "youve", "were", "wouldnt", "havent", "wasnt", "yall", "lets", "heres", "whos", "youll", "couldnt", "werent", "hasnt", "weve", "aint", "youd"))
  ) %>%
  # Bind the data
  group_by(participant_id, timestamp) %>%
  summarize(
    my_text = paste(word, collapse = " ")
  ) %>%
  ungroup()

# ...3) Detect & Assign N-Grams -------------------------------------------

# Assign the unigrams and bigrams
for (i in 1:length(ngrams_vector)) {
  
  # Get the n-grams
  ngram <- ngrams_vector[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(social_media_posts$my_text, regex(ngram))
  
  # Add the n-gram to the dataframe
  social_media_posts[[ngram]] <- as.integer(x)  
}

# Export data
write_csv(social_media_posts, "data/participants/features/cmips_feature_set_07_dassp.csv")
