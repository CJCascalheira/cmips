# CREATE FEATURES - THIRD SET ---------------------------------------------

# Author = Cory J. Cascalheira
# Date = 04/19/2024

# The purpose of this script is to create features for CMIPS. Code is taken from
# the LGBTQ+ MiSSoM dataset.

# The following features are created:
# - Ngrams

# 1) OPEN VOCABULARY ------------------------------------------------------

# ...1a) N-GRAMS ----------------------------------------------------------

# Top unigrams
unigram_df <- cmips_social_media %>%
  # Select key columns
  select(participant_id, timestamp, text, label_minority_stress) %>%
  # Generate unigrams
  unnest_tokens(word, text, drop = FALSE) %>%
  # Remove stop words
  count(label_minority_stress, word) %>%
  arrange(desc(n)) %>%
  # Clean up based on remaining stop words
  mutate(
    stop_word = if_else(str_detect(word, regex("^im$|that's|i’m|it’s|you’re|don’t|dont|It|can’t|lt|he’s|she’s|i’ve|doesn’t|didn’t|isn’t|there’s|that'll|how’s|they’ll|it’ll|would've|we’ll|they’ve|shouldn’t|that’s|i’ll|they’re|aren’t|i’d|won’t|what’s|you’ve|we’re|wouldn’t|haven’t|wasn’t|y'all|let’s|here’s|who’s|you’ll|couldn’t|weren’t|hasn’t|we’ve|ain’t|you’d|y’all")), 1, 0) 
  ) %>%
  # Remove remaining stop words
  filter(stop_word == 0) %>%
  select(-stop_word)

# TF-IDF unigrams
unigram_vector <- unigram_df %>%
  # Calculate tf-idf
  bind_tf_idf(word, label_minority_stress, n) %>%
  # Get top tf-idf of unigrams for minority stress posts
  arrange(desc(tf_idf)) %>%
  filter(label_minority_stress == 1) %>%
  # Remove words based on close inspection of unigrams
  mutate(remove = if_else(str_detect(word, regex("’s|’d|'s|	
’ve|\\d|monday|tuesday|wednesday|thursday|friday|saturday|sunday|lockdown|covid|grammatical|film|eh|could’ve|december|vehicle|paint|ness|bout|brown|animals|âˆ|weather|bike|maria|albeit|amd|matt|minecraft|freind|have|𝙸|𝚠𝚒𝚕𝚕|𝚢𝚘𝚞|ᴍʏ|can‘t|causally")), 1, 0)) %>%
  filter(remove == 0) %>%
  # Select the top 100 n-grams
  head(n = 100) %>%
  pull(word)

# Generate bigrams
bigram_df <- cmips_social_media %>%
  # Select key columns
  select(participant_id, timestamp, text, label_minority_stress) %>%
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
  count(label_minority_stress, bigram) %>%
  arrange(desc(n))

# TF-IDF bigrams
bigram_vector <- bigram_df %>%
  # Calculate tf-idf
  bind_tf_idf(bigram, label_minority_stress, n) %>%
  # Get top tf-idf of unigrams for minority stress posts
  arrange(desc(tf_idf)) %>%
  filter(label_minority_stress == 1) %>%
  # Remove words based on close inspection of unigrams
  mutate(remove = if_else(str_detect(bigram, regex("’s|’d|'s|	
’ve|\\d|monday|tuesday|wednesday|thursday|friday|saturday|sunday|lockdown|covid|^ive |^lot |minutes ago|ame$")), 1, 0)) %>%
  filter(remove == 0) %>%
  # Select the top 100 n-grams
  head(n = 100) %>%
  pull(bigram)

# Generate trigrams
trigram_df <- cmips_social_media %>%
  # Select key columns
  select(participant_id, timestamp, text, label_minority_stress) %>%
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
  count(label_minority_stress, trigram) %>%
  arrange(desc(n))

# TF-IDF Trigrams
trigram_vector <- trigram_df %>%
  # Manual remove of nonsense
  mutate(remove = if_else(str_detect(trigram, "\\d|ðÿ|^amp |amp | amp$|NA NA NA|poll$|jfe|_link|link_|playlist 3948ybuzmcysemitjmy9jg si|complete 3 surveys|gmail.com mailto:hellogoodbis42069 gmail.com|hellogoodbis42069 gmail.com mailto:hellogoodbis42069|comments 7n2i gay_marriage_debunked_in_2_minutes_obama_vs_alan|debatealtright comments 7n2i|gift card|amazon|action hirewheller csr|energy 106 fm|form sv_a3fnpplm8nszxfb width|â íœê í|âˆ âˆ âˆ"), 1, 0)) %>%
  filter(remove == 0) %>%
  # Calculate tf-idf
  bind_tf_idf(trigram, label_minority_stress, n) %>%
  # Get top tf-idf of unigrams for minority stress posts
  arrange(desc(tf_idf)) %>%
  filter(label_minority_stress == 1) %>%
  # Select the top 100 n-grams
  head(n = 100) %>%
  pull(trigram)

# ...1b) ASSIGN N-GRAMS ---------------------------------------------------

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