# CREATE FEATURES - SECOND SET --------------------------------------------

# Author = Cory J. Cascalheira
# Date = 04/19/2024

# The purpose of this script is to create features for CMIPS. Code is taken from
# the LGBTQ+ MiSSoM dataset.

# The following features are created:
# - Clinical keywords
# - Sentiment lexicon
# - Hate speech lexicon
# - Theoretical lexicon
# - Pain lexicon

# LOAD LIBRARIES AND IMPORT DATA ------------------------------------------

# Load dependencies
library(textstem)
library(tidyverse)
library(tidytext)
library(pdftools)

# Import data files
cmips_social_media <- read_csv("data/participants/cleaned/social_media_posts_cleaned.csv") %>%
  # Recode posts and comments to text to work for code
  rename(text = posts_comments)

# Import the DSM-5 text
dsm5_anxiety <- read_file("data/util/dsm5_anxiety.txt")
dsm5_depression <- read_file("data/util/dsm5_depression.txt")
dsm5_ptsd <- read_file("data/util/dsm5_ptsd.txt")
dsm5_substance_use <- read_file("data/util/dsm5_substance_use.txt")
dsm5_gender_dysphoria <- read_csv("data/util/dsm5_gender_dysphoria.csv")

# Get AFINN sentiment lexicon
afinn <- get_sentiments("afinn")

# Get slangSD lexicon: https://github.com/airtonbjunior/opinionMining/blob/master/dictionaries/slangSD.txt
slangsd <- read_delim("data/util/slangSD.txt", delim = "\t", col_names = FALSE) %>%
  rename(word = X1, value = X2)

# Combine sentiment libraries
sentiment_df <- bind_rows(afinn, slangsd) %>%
  distinct(word, .keep_all = TRUE)

# Import hate speech lexicons
hate_lexicon_sexual_minority <- read_csv("data/util/hatespeech/hate_lexicon_sexual_minority.csv") %>%
  select(word) %>%
  mutate(hate_lexicon_sexual_minority = 1)

hate_lexicon_gender_minority <- read_csv("data/util/hatespeech/hate_lexicon_gender_minority.csv") %>%
  select(word) %>%
  mutate(hate_lexicon_gender_minority = 1)

hate_lexicon_woman_man <- read_csv("data/util/hatespeech/hatebase_woman_man.csv") %>%
  select(word) %>%
  mutate(hate_lexicon_woman_man = 1)

# Import theoretical lexicon of minority stress text
minority_stress_2003 <- read_file("data/util/minority_stress_text/minority_stress_2003.txt")

minority_stress_ethnicity <- read_file("data/util/minority_stress_text/minority_stress_ethnicity.txt")

minority_stress_1995 <- pdf_text("data/util/minority_stress_text/minority_stress_1995.pdf")

minority_stress_transgender <- pdf_text("data/util/minority_stress_text/minority_stress_transgender.pdf")

# Import pain lexicon
pain_lexicon <- read_csv("data/util/pain_lexicon.csv")

# 1) CLINICAL KEYWORDS ----------------------------------------------------

# ...1a) PREPARE THE DSM-5 TEXT -------------------------------------------

# Common DSM-5 words to remove
common_dsm5 <- c("attack", "social", "individual", "situation", "specific", "child", "substance", "occur", "adult", "include", "experience", "criterion", "onset", "generalize", "prevalence", "feature", "rate", "age", "due", "figure", "physical", "risk", "attachment", "month", "home", "event", "factor", "episode", "major", "meet", "persistent", "day", "period", "note", "excessive", "behavior", "adjustment", "response", "code", "mental", "effect", "significant", 'time', "develop", "unknown", "gender", "sex", "male", "female", "adolescent", "desire", "strong", "boy", "characteristic", "girl", "refer", "andor", "lateonset", "sexual", "express", "identity", "increase")

# Transdiagnostic clinical key words
dsm5_transdiagnostic <- data.frame(word = c("disorder", "symptom", "medical", "condition", "diagnosis", "diagnostic", "diagnose", "withdrawal", "impairment", "comorbid", "chronic", "acute", "disturbance", "severe", "treatment")) %>%
  tibble()

# DSM-5 anxiety disorders
dsm5_anxiety_df <- data.frame(dsm5 = dsm5_anxiety) %>% 
  as_tibble() %>%
  # Remove punctuation and digits
  mutate(dsm5 = str_remove_all(dsm5, regex("[:punct:]|[:digit:]", ignore_case = TRUE))) %>%
  # Convert to lowercase
  mutate(dsm5 = tolower(dsm5)) %>%
  unnest_tokens(output = "word", input = "dsm5") %>%
  # Remove stop words
  filter(!(word %in% stop_words$word)) %>%
  # Lemmatize the words
  mutate(word = lemmatize_words(word)) %>%
  # Top words in the DSM-5
  count(word) %>%
  arrange(desc(n)) %>% 
  # Remove common words
  filter(!(word %in% common_dsm5)) %>%
  # Remove DSM-5 transdiagnostic clinical key words
  filter(!(word %in% dsm5_transdiagnostic$word)) %>%
  filter(!(word %in% c("avoid", "object"))) %>%
  head(n = 10) %>%
  select(-n) %>%
  mutate(word = stem_words(word)) %>%
  # Add other permutations of the top 10 key words for the NLP search
  bind_rows(data.frame(word = c("anxiety", "worry", "phobic"))) %>%
  rename(dsm5_anxiety = word) 
dsm5_anxiety_df

# DSM-5 depressive disorders
dsm5_depression_df <- data.frame(dsm5 = dsm5_depression) %>% 
  as_tibble() %>%
  # Remove punctuation and digits
  mutate(dsm5 = str_remove_all(dsm5, regex("[:punct:]|[:digit:]", ignore_case = TRUE))) %>%
  # Convert to lowercase
  mutate(dsm5 = tolower(dsm5)) %>%
  unnest_tokens(output = "word", input = "dsm5") %>%
  # Remove stop words
  filter(!(word %in% stop_words$word)) %>%
  # Lemmatize the words
  mutate(word = lemmatize_words(word)) %>%
  # Top words in the DSM-5
  count(word) %>%
  arrange(desc(n)) %>% 
  # Remove common words
  filter(!(word %in% common_dsm5)) %>%
  # Remove DSM-5 transdiagnostic clinical key words
  filter(!(word %in% dsm5_transdiagnostic$word)) %>%
  filter(!(word %in% c("depression", "dysphoric", "depress"))) %>%
  head(n = 10) %>%
  select(-n) %>%
  mutate(word = stem_words(word)) %>%
  # Add other permutations of the top 10 key words for the NLP search
  bind_rows(data.frame(word = c("anxiety"))) %>%
  rename(dsm5_depression = word)
dsm5_depression_df

# DSM-5 PTSD and stress disorders
dsm5_ptsd_df <- data.frame(dsm5 = dsm5_ptsd) %>% 
  as_tibble() %>%
  # Remove punctuation and digits
  mutate(dsm5 = str_remove_all(dsm5, regex("[:punct:]|[:digit:]", ignore_case = TRUE))) %>%
  # Convert to lowercase
  mutate(dsm5 = tolower(dsm5)) %>%
  unnest_tokens(output = "word", input = "dsm5") %>%
  # Remove stop words
  filter(!(word %in% stop_words$word)) %>%
  # Lemmatize the words
  mutate(word = lemmatize_words(word)) %>%
  # Top words in the DSM-5
  count(word) %>%
  arrange(desc(n)) %>% 
  # Remove common words
  filter(!(word %in% common_dsm5)) %>%
  # Remove DSM-5 transdiagnostic clinical key words
  filter(!(word %in% dsm5_transdiagnostic$word)) %>%
  head(n = 10) %>%
  select(-n) %>%
  mutate(word = stem_words(word)) %>%
  rename(dsm5_ptsd = word)
dsm5_ptsd_df

# DSM-5 substance use
dsm5_substance_use_df <- data.frame(dsm5 = dsm5_substance_use) %>% 
  as_tibble() %>%
  # Remove punctuation and digits
  mutate(dsm5 = str_remove_all(dsm5, regex("[:punct:]|[:digit:]", ignore_case = TRUE))) %>%
  # Convert to lowercase
  mutate(dsm5 = tolower(dsm5)) %>%
  unnest_tokens(output = "word", input = "dsm5") %>%
  # Remove stop words
  filter(!(word %in% stop_words$word)) %>%
  # Lemmatize the words
  mutate(word = lemmatize_words(word)) %>%
  # Top words in the DSM-5
  count(word) %>%
  arrange(desc(n)) %>% 
  # Remove common words
  filter(!(word %in% common_dsm5)) %>%
  # Remove DSM-5 transdiagnostic clinical key words
  filter(!(word %in% dsm5_transdiagnostic$word)) %>%
  head(n = 10) %>%
  select(-n) %>%
  # Add other permutations of the top 10 key words for the NLP search
  bind_rows(data.frame(word = c("intoxicat"))) %>%
  rename(dsm5_substance_use = word)
dsm5_substance_use_df

# DSM-5 gender dysphoria
dsm5_gender_dysphoria_df <- data.frame(dsm5 = dsm5_gender_dysphoria) %>% 
  as_tibble() %>%
  # Remove punctuation and digits
  mutate(dsm5 = str_remove_all(dsm5, regex("[:punct:]|[:digit:]", ignore_case = TRUE))) %>%
  # Convert to lowercase
  mutate(dsm5 = tolower(dsm5)) %>%
  unnest_tokens(output = "word", input = "dsm5") %>%
  # Remove stop words
  filter(!(word %in% stop_words$word)) %>%
  # Lemmatize the words
  mutate(word = lemmatize_words(word)) %>%
  # Top words in the DSM-5
  count(word) %>%
  arrange(desc(n)) %>% 
  # Remove common words
  filter(!(word %in% common_dsm5)) %>%
  # Remove DSM-5 transdiagnostic clinical key words
  filter(!(word %in% dsm5_transdiagnostic$word)) %>%
  head(n = 10) %>%
  select(-n) %>%
  mutate(word = stem_words(word)) %>%
  # Add other permutations of the top 10 key words for the NLP search
  bind_rows(data.frame(word = c("surgery", "dysphoric"))) %>%
  rename(dsm5_gender_dysphoria = word)
dsm5_gender_dysphoria_df

# ...1b) GENERATE THE FEATURES --------------------------------------------

# Features 
cmips_social_media <- cmips_social_media %>%
  mutate(
    dsm5_anxiety = if_else(str_detect(text, regex(paste(dsm5_anxiety_df$dsm5_anxiety, collapse = "|"), 
                                                  ignore_case = TRUE)), 1, 0),
    dsm5_depression = if_else(str_detect(text, regex(paste(dsm5_depression_df$dsm5_depression, collapse = "|"), 
                                                     ignore_case = TRUE)), 1, 0),
    dsm5_ptsd = if_else(str_detect(text, regex(paste(dsm5_ptsd_df$dsm5_ptsd, collapse = "|"),
                                               ignore_case = TRUE)), 1, 0),
    dsm5_substance_use = if_else(str_detect(text, regex(paste(dsm5_substance_use_df$dsm5_substance_use, collapse = "|"), 
                                                        ignore_case = TRUE)), 1, 0),
    dsm5_gender_dysphoria = if_else(str_detect(text, regex(paste(dsm5_gender_dysphoria_df$dsm5_gender_dysphoria, collapse = "|"), 
                                                           ignore_case = TRUE)), 1, 0),
    dsm5_transdiagnostic = if_else(str_detect(text, regex(paste(dsm5_transdiagnostic$word, collapse = "|"), 
                                                          ignore_case = TRUE)), 1, 0)
  )

# 2) LEXICONS -------------------------------------------------------------

# ...2a) SENTIMENT LEXICON ------------------------------------------------

# Generate the features
cmips_social_media <- cmips_social_media %>%
  # Tokenize Reddit post
  unnest_tokens(word, text) %>%
  # Get sentiment of words
  left_join(sentiment_df) %>%
  # Recode missing to 0 sentiment
  mutate(value = if_else(is.na(value), 0, value)) %>%
  # Group by post
  group_by(participant_id, timestamp) %>%
  # Calculate total sentiment of post
  summarize(sentiment_lexicon = sum(value)) %>%
  # Split the sentiment into two column
  mutate(
    sentiment_overall_positive = if_else(sentiment_lexicon > 0, 1, 0),
    sentiment_overall_negative = if_else(sentiment_lexicon < 0, 1, 0)
  ) %>%
  select(-sentiment_lexicon) %>%
  # Join to main dataframe
  left_join(cmips_social_media) %>%
  ungroup()

# ...2b) HATE SPEECH LEXICONS ----------------------------------------------

# Generate the features
cmips_social_media <- cmips_social_media %>%
  # Reduce df size
  select(participant_id, timestamp, text) %>%
  # Tokenize Reddit post
  unnest_tokens(word, text) %>%
  # Join the hate speech data frames
  left_join(hate_lexicon_sexual_minority) %>%
  left_join(hate_lexicon_gender_minority) %>%
  left_join(hate_lexicon_woman_man) %>%
  # Recode missing to 0
  mutate(
    hate_lexicon_sexual_minority = if_else(is.na(hate_lexicon_sexual_minority), 0,
                                           hate_lexicon_sexual_minority),
    hate_lexicon_gender_minority = if_else(is.na(hate_lexicon_gender_minority), 0,
                                           hate_lexicon_gender_minority),
    hate_lexicon_woman_man = if_else(is.na(hate_lexicon_woman_man), 0,
                                     hate_lexicon_woman_man)
  ) %>%
  # Group by post
  group_by(participant_id, timestamp) %>%
  # Calculate presence of hate speech term
  summarize(
    hate_lexicon_sexual_minority = sum(hate_lexicon_sexual_minority),
    hate_lexicon_gender_minority = sum(hate_lexicon_gender_minority),
    hate_lexicon_woman_man = sum(hate_lexicon_woman_man)
  ) %>%
  # Join to main dataframe
  left_join(cmips_social_media) %>%
  ungroup()

# ...2c) THEORETICAL MINORITY STRESS LEXICON ------------------------------

# ......2c1) PREPARE THE DATA ---------------------------------------------

# Common terms to filter
ms_common_terms <- c("measure", "event", "association", "gay", "journal", "york", "study", "sample", "relate", "table", "aid", "subject", "american", "effect", "social", "google", "scholar", "lgb", "pubmed", "lesbian", "bisexual", "prevalence", "research", "al", "individual", "people", "process", "person", "pp", "editor", "suicide", "lgbt", "lgbtpoc", "participant", "item", "sexual", "experience", "scale", "subscale", "white", "trans", "gender", "transgender")

# Meyer (1995) - early version of minority stress
minority_stress_1995_df <- data.frame(article = minority_stress_1995) %>% 
  as_tibble() %>%
  mutate(
    # Remove punctuation and digits
    article = str_remove_all(article, regex("[:punct:]|[:digit:]", ignore_case = TRUE)),
    # Remove white space character
    article = str_replace_all(article, regex("\n"), " "),
    # Remove padding
    article = str_trim(article),
    article = str_squish(article),
    # Covert to lowercase
    article = tolower(article)
  ) %>%
  # Extract single words
  unnest_tokens(output = "word", input = "article") %>%
  # Remove stop words
  filter(!(word %in% stop_words$word)) %>%
  # Lemmatize the words
  mutate(word = lemmatize_words(word)) %>%
  # Top words 
  count(word) %>%
  arrange(desc(n)) %>%
  # Filter out common words
  filter(!(word %in% ms_common_terms)) %>%
  # Get the top unique words related to minority stress
  head(n = 13) %>%
  select(-n) %>%
  # Add other permutations and combinations of the top key words for the NLP search
  bind_rows(data.frame(word = c("minority stress", "homophobic", "violent", 
                                "mental health")))
minority_stress_1995_df

# Meyer (2003) - most popular version of minority stress
minority_stress_2003_df <- data.frame(article = minority_stress_2003) %>% 
  as_tibble() %>%
  mutate(
    # Remove punctuation and digits
    article = str_remove_all(article, regex("[:punct:]|[:digit:]", ignore_case = TRUE)),
    # Remove white space character
    article = str_replace_all(article, regex("\n"), " "),
    # Remove padding
    article = str_trim(article),
    article = str_squish(article),
    # Covert to lowercase
    article = tolower(article)
  ) %>%
  # Extract single words
  unnest_tokens(output = "word", input = "article") %>%
  # Remove stop words
  filter(!(word %in% stop_words$word)) %>%
  # Lemmatize the words
  mutate(word = lemmatize_words(word)) %>%
  # Top words
  count(word) %>%
  arrange(desc(n)) %>%
  # Filter out common words
  filter(!(word %in% ms_common_terms)) %>%
  # Get the top unique words related to minority stress
  head(n = 13) %>%
  select(-n) %>%
  # Add other permutations and combinations of the top key words for the NLP search
  bind_rows(data.frame(word = c("mental disorder")))
minority_stress_2003_df

# Balsam et al (2011) - minority stress adapted for ethnic minority LGBTQ+ folx
minority_stress_ethnicity_df <- data.frame(article = minority_stress_ethnicity) %>% 
  as_tibble() %>%
  mutate(
    # Remove punctuation and digits
    article = str_remove_all(article, regex("[:punct:]|[:digit:]", ignore_case = TRUE)),
    # Remove white space character
    article = str_replace_all(article, regex("\n"), " "),
    # Remove padding
    article = str_trim(article),
    article = str_squish(article),
    # Covert to lowercase
    article = tolower(article)
  ) %>%
  # Extract single words
  unnest_tokens(output = "word", input = "article") %>%
  # Remove stop words
  filter(!(word %in% stop_words$word)) %>%
  # Lemmatize the words
  mutate(word = lemmatize_words(word)) %>%
  # Top words 
  count(word) %>%
  arrange(desc(n)) %>%
  # Filter out common words
  filter(!(word %in% ms_common_terms)) %>%
  # Get the top unique words related to minority stress
  head(n = 11) %>%
  select(-n) %>%
  # Add other permutations and combinations of the top key words for the NLP search
  bind_rows(data.frame(word = c("racial", "ethnic")))
minority_stress_ethnicity_df

# Hendricks & Testa (2012) - minority stress adaoted for transgender folx
minority_stress_transgender_df <- data.frame(article = minority_stress_transgender) %>% 
  as_tibble() %>%
  mutate(
    # Remove punctuation and digits
    article = str_remove_all(article, regex("[:punct:]|[:digit:]", ignore_case = TRUE)),
    # Remove white space character
    article = str_replace_all(article, regex("\n"), " "),
    # Remove padding
    article = str_trim(article),
    article = str_squish(article),
    # Covert to lowercase
    article = tolower(article)
  ) %>%
  # Extract single words
  unnest_tokens(output = "word", input = "article") %>%
  # Remove stop words
  filter(!(word %in% stop_words$word)) %>%
  # Lemmatize the words
  mutate(word = lemmatize_words(word)) %>%
  # Top words 
  count(word) %>%
  arrange(desc(n)) %>%
  # Filter out common words
  filter(!(word %in% ms_common_terms)) %>%
  # Get the top unique words related to minority stress
  head(n = 4) %>%
  select(-n)
minority_stress_transgender_df

# Bind all dfs
minority_stress_df <- bind_rows(minority_stress_1995_df, minority_stress_2003_df) %>%
  bind_rows(minority_stress_ethnicity_df) %>%
  bind_rows(minority_stress_transgender_df) %>%
  # Remove repeats
  distinct(word)

# ......2c2) GENERATE THE FEATURES ----------------------------------------

# Execute code
cmips_social_media <- cmips_social_media %>%
  mutate(
    theoretical_ms_lexicon = if_else(str_detect(text, regex(paste(minority_stress_df$word, collapse = "|"), ignore_case = TRUE)), 1, 0)
  )

# ...2d) PAIN LEXICON -----------------------------------------------------

# Add pain terms 
cmips_social_media <- cmips_social_media %>%
  mutate(
    pain_lexicon = if_else(str_detect(text, regex(paste(pain_lexicon$keywords, collapse = "|"),
                                                  ignore_case = TRUE)), 1, 0)
  )

# EXPORT ------------------------------------------------------------------

# Export files
write_csv(cmips_social_media, "data/participants/features/cmips_feature_set_02.csv")
