# MINORITY STRESS N-GRAMS -------------------------------------------------

# Author = Cory J. Cascalheira
# Date = 04/25/2024

# The purpose of this script is to create minority stress n-grams using the 
# LGBTQ+ MiSSoM dataset.

# DEPENDENCIES AND IMPORT -------------------------------------------------

# Libraries
library(tidyverse)
library(tidytext)
library(lubridate)

# Import LGBTQ+ MiSSoM+
missom <- read_csv("data/util/missom_plus.csv") %>%
  select(post_id, my_text = text, label = label_minority_stress) %>%
  filter(label == 1) %>%
  mutate(post_id = as.character(post_id))

# Import DASSP
depression_df <- read_csv("data/util/dassp/cleaned/depression_df_demojized.csv") %>%
  select(-...1) %>%
  rename(my_text = text) %>%
  filter(label == 0)

anxiety_df <- read_csv("data/util/dassp/cleaned/anxiety_df_demojized.csv") %>%
  select(-...1) %>%
  rename(my_text = text) %>%
  filter(label == 0)

stress_df <- read_csv("data/util/dassp/cleaned/stress_df_demojized.csv") %>%
  select(-...1) %>%
  rename(my_text = text) %>%
  filter(label == 0)

suicide_df <- read_csv("data/util/dassp/cleaned/suicide_df_demojized.csv") %>%
  select(-...1) %>%
  rename(my_text = text) %>%
  filter(label == 0)

ptsd_df <- read_csv("data/util/dassp/cleaned/ptsd_df_demojized.csv") %>%
  select(-...1) %>%
  rename(my_text = text) %>%
  filter(label == 0)

# Import CMIPS
social_media_posts <- read_csv("data/participants/combined_social_media/social_media_posts_demojized.csv") %>%
  select(-...1) %>%
  rename(my_text = posts_comments)

# PREPROCESS THE DATA -----------------------------------------------------

# Create the main dataframe
missom_ngrams <- bind_rows(depression_df, anxiety_df) %>%
  bind_rows(stress_df) %>%
  bind_rows(suicide_df) %>%
  bind_rows(ptsd_df) %>%
  rename(post_id = id) %>%
  sample_n(size = nrow(missom)) %>%
  # Bind the two dataframes
  bind_rows(missom)

# Preprocess the text
missom_ngrams <- missom_ngrams %>%
  mutate(
    # Remove Reddit-specific language 
    my_text = str_remove_all(my_text, regex("\\[deleted\\]", ignore_case = TRUE)),
    my_text = str_remove_all(my_text, regex("\\[removed\\]", ignore_case = TRUE)),
    # Remove markdown links
    my_text = str_remove_all(my_text, regex("\\[.*\\]\\(.*\\)")),
    # Remove whitespace characters
    my_text = str_remove_all(my_text, regex("\\n|\\r")),
    # Remove URLs
    my_text = str_remove_all(my_text, regex("(http|ftp|https):\\/\\/([\\w_-]+(?:(?:\\.[\\w_-]+)+))([\\w.,@?^=%&:\\/~+#-]*[\\w@?^=%&\\/~+#-])")),
    # Remove NA
    my_text = str_remove_all(my_text, regex("^NA | NA$", ignore_case = TRUE)),
    # Trim spaces
    my_text = str_trim(my_text)
  ) %>%
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
    !(word %in% c("^im$", "quot", "thats", "its", "youre", "dont", "it", "cant", "lt", "hes", "shes", "ive", "doesnt", "didnt", "isnt", "theres", "thatll", "hows", "theyll", "itll", "wouldve", "well", "theyve", "shouldnt", "thats", "ill", "theyre", "arent", "id", "wont", "whats", "youve", "were", "wouldnt", "havent", "wasnt", "yall", "lets", "heres", "whos", "youll", "couldnt", "werent", "hasnt", "weve", "aint", "youd"))
  ) %>%
  # Bind the data
  group_by(post_id, label) %>%
  summarize(
    my_text = paste(word, collapse = " ")
  ) %>%
  ungroup()

# CREATE N-GRAMS ----------------------------------------------------------

# Create unigrams
missom_ngrams_unigrams <- missom_ngrams %>%
  # Generate unigrams
  unnest_tokens(ngram, my_text, drop = FALSE) %>%
  # Remove stop words
  count(label, ngram) %>%
  arrange(desc(n))

# TF-IDF unigrams
missom_ngrams_unigrams <- missom_ngrams_unigrams %>%
  # Calculate tf-idf
  bind_tf_idf(ngram, label, n) %>%
  # Get top tf-idf of unigrams
  arrange(desc(tf_idf)) %>%
  filter(label == 1) %>%
  # Select the top 100 n-grams
  head(n = 100) %>%
  select(ngram)

# Generate bigrams 
missom_ngrams_bigrams <- missom_ngrams %>%
  unnest_ngrams(ngram, my_text, n = 2, drop = FALSE) %>%
  # Separate the bigrams into two columns
  separate(ngram, c("word1", "word2")) %>%
  unite("ngram", c("word1", "word2"), sep = " ") %>%
  # Count top bigrams
  count(label, ngram) %>%
  arrange(desc(n))

# TF-IDF bigrams
missom_ngrams_bigrams <- missom_ngrams_bigrams %>%
  # Calculate tf-idf
  bind_tf_idf(ngram, label, n) %>%
  # Get top tf-idf 
  arrange(desc(tf_idf)) %>%
  filter(label == 1) %>%
  # Select the top 100 n-grams
  head(n = 100) %>%
  select(ngram)

# Combine n-grams
ngrams_vector <- bind_rows(missom_ngrams_unigrams, missom_ngrams_bigrams) %>%
  pull(ngram)

# ASSIGN N-GRAMS ----------------------------------------------------------

# ...1) Clean CMIPS data --------------------------------------------------

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
    !(word %in% c("^im$", "quot", "thats", "its", "youre", "dont", "it", "cant", "lt", "hes", "shes", "ive", "doesnt", "didnt", "isnt", "theres", "thatll", "hows", "theyll", "itll", "wouldve", "well", "theyve", "shouldnt", "thats", "ill", "theyre", "arent", "id", "wont", "whats", "youve", "were", "wouldnt", "havent", "wasnt", "yall", "lets", "heres", "whos", "youll", "couldnt", "werent", "hasnt", "weve", "aint", "youd"))
  ) %>%
  # Bind the data
  group_by(participant_id, timestamp) %>%
  summarize(
    my_text = paste(word, collapse = " ")
  ) %>%
  ungroup()

# ...2) Detect & Assign N-Grams -------------------------------------------

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
write_csv(social_media_posts, "data/participants/features/cmips_feature_set_08.csv")
