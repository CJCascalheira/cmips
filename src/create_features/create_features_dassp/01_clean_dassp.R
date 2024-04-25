# CLEAN THE D.A.S.S.P. N-GRAMS ---------------------------------------------

# Author = Cory J. Cascalheira
# Date = 04/24/2024

# The purpose of this script is to prepare the depression, anxiety, stress,
# suicide, and PTSD n-grams for importing into the CMIPS project.

# LOAD DEPENDENCIES AND IMPORT --------------------------------------------

# Set seed
set.seed(1234567)

# Dependencies
library(tidyverse)

# Import BigQuery positive DASSP files
my_csvs <- list.files("data/util/dassp/pos_examples/")
my_files1 <- paste0("data/util/dassp/pos_examples/", my_csvs)
files_list <- lapply(my_files1, read_csv)

# Add subreddit names 
for (i in 1:length(my_files1)) {
  files_list[[i]] <- files_list[[i]] %>%
    mutate(subreddit = rep(my_files1[i], nrow(.)))
}

# Combine data frames
df_pos <- bind_rows(files_list, .id = "column_label") %>%
  select(-column_label) %>%
  select(id, subreddit, everything()) %>%
  mutate(subreddit = str_extract(subreddit, regex("(?<=data/util/dassp/pos_examples/bigquery_)\\w*(?=_\\d*.csv)")))

# Import BigQuery negative DASSP 9 subreddits
my_csvs <- list.files("data/util/dassp/neg_examples/")
my_files1 <- paste0("data/util/dassp/neg_examples/", my_csvs)
files_list <- lapply(my_files1, read_csv)

# Combine data frames
df_neg <- bind_rows(files_list, .id = "column_label") %>%
  select(-column_label)

# PREPROCESS THE DATA -----------------------------------------------------

# ...1) Positive Examples -------------------------------------------------

# Divide positive examples based on subreddits
depression <- df_pos %>%
  filter(subreddit == "depression") %>%
  select(id, title, selftext) %>%
  # Sample to preserve memory
  slice_sample(n = 100000) %>%
  unite(col = "text", title:selftext, sep = " ") %>%
  mutate(
    # Remove Reddit-specific language 
    text = str_remove_all(text, regex("\\[deleted\\]", ignore_case = TRUE)),
    text = str_remove_all(text, regex("\\[removed\\]", ignore_case = TRUE)),
    # Remove markdown links
    text = str_remove_all(text, regex("\\[.*\\]\\(.*\\)")),
    # Remove whitespace characters
    text = str_remove_all(text, regex("\\n|\\r")),
    # Remove URLs
    text = str_remove_all(text, regex("(http|ftp|https):\\/\\/([\\w_-]+(?:(?:\\.[\\w_-]+)+))([\\w.,@?^=%&:\\/~+#-]*[\\w@?^=%&\\/~+#-])")),
    # Remove NA
    text = str_remove_all(text, regex("^NA | NA$", ignore_case = TRUE)),
    # Trim spaces
    text = str_trim(text)
  ) %>%
  # Add label
  mutate(label = 1)

anxiety <- df_pos %>%
  filter(subreddit == "anxiety") %>%
  select(id, title, selftext) %>%
  # Sample to preserve memory
  slice_sample(n = 100000) %>%
  unite(col = "text", title:selftext, sep = " ") %>%
  mutate(
    # Remove Reddit-specific language 
    text = str_remove_all(text, regex("\\[deleted\\]", ignore_case = TRUE)),
    text = str_remove_all(text, regex("\\[removed\\]", ignore_case = TRUE)),
    # Remove markdown links
    text = str_remove_all(text, regex("\\[.*\\]\\(.*\\)")),
    # Remove whitespace characters
    text = str_remove_all(text, regex("\\n|\\r")),
    # Remove URLs
    text = str_remove_all(text, regex("(http|ftp|https):\\/\\/([\\w_-]+(?:(?:\\.[\\w_-]+)+))([\\w.,@?^=%&:\\/~+#-]*[\\w@?^=%&\\/~+#-])")),
    # Remove NA
    text = str_remove_all(text, regex("^NA | NA$", ignore_case = TRUE)),
    # Trim spaces
    text = str_trim(text)
  ) %>%
  # Add label
  mutate(label = 1)

stress <- df_pos %>%
  filter(subreddit == "stress") %>%
  select(id, title, selftext) %>%
  unite(col = "text", title:selftext, sep = " ") %>%
  mutate(
    # Remove Reddit-specific language 
    text = str_remove_all(text, regex("\\[deleted\\]", ignore_case = TRUE)),
    text = str_remove_all(text, regex("\\[removed\\]", ignore_case = TRUE)),
    # Remove markdown links
    text = str_remove_all(text, regex("\\[.*\\]\\(.*\\)")),
    # Remove whitespace characters
    text = str_remove_all(text, regex("\\n|\\r")),
    # Remove URLs
    text = str_remove_all(text, regex("(http|ftp|https):\\/\\/([\\w_-]+(?:(?:\\.[\\w_-]+)+))([\\w.,@?^=%&:\\/~+#-]*[\\w@?^=%&\\/~+#-])")),
    # Remove NA
    text = str_remove_all(text, regex("^NA | NA$", ignore_case = TRUE)),
    # Trim spaces
    text = str_trim(text)
  ) %>%
  # Add label
  mutate(label = 1)

suicide <- df_pos %>%
  filter(subreddit == "suicide") %>%
  select(id, title, selftext) %>%
  # Sample to preserve memory
  slice_sample(n = 100000) %>%
  unite(col = "text", title:selftext, sep = " ") %>%
  mutate(
    # Remove Reddit-specific language 
    text = str_remove_all(text, regex("\\[deleted\\]", ignore_case = TRUE)),
    text = str_remove_all(text, regex("\\[removed\\]", ignore_case = TRUE)),
    # Remove markdown links
    text = str_remove_all(text, regex("\\[.*\\]\\(.*\\)")),
    # Remove whitespace characters
    text = str_remove_all(text, regex("\\n|\\r")),
    # Remove URLs
    text = str_remove_all(text, regex("(http|ftp|https):\\/\\/([\\w_-]+(?:(?:\\.[\\w_-]+)+))([\\w.,@?^=%&:\\/~+#-]*[\\w@?^=%&\\/~+#-])")),
    # Remove NA
    text = str_remove_all(text, regex("^NA | NA$", ignore_case = TRUE)),
    # Trim spaces
    text = str_trim(text)
  ) %>%
  # Add label
  mutate(label = 1)

ptsd <- df_pos %>%
  filter(subreddit == "ptsd") %>%
  select(id, title, selftext) %>%
  unite(col = "text", title:selftext, sep = " ") %>%
  mutate(
    # Remove Reddit-specific language 
    text = str_remove_all(text, regex("\\[deleted\\]", ignore_case = TRUE)),
    text = str_remove_all(text, regex("\\[removed\\]", ignore_case = TRUE)),
    # Remove markdown links
    text = str_remove_all(text, regex("\\[.*\\]\\(.*\\)")),
    # Remove whitespace characters
    text = str_remove_all(text, regex("\\n|\\r")),
    # Remove URLs
    text = str_remove_all(text, regex("(http|ftp|https):\\/\\/([\\w_-]+(?:(?:\\.[\\w_-]+)+))([\\w.,@?^=%&:\\/~+#-]*[\\w@?^=%&\\/~+#-])")),
    # Remove NA
    text = str_remove_all(text, regex("^NA | NA$", ignore_case = TRUE)),
    # Trim spaces
    text = str_trim(text)
  ) %>%
  # Add label
  mutate(label = 1)

# ...2) Negative Examples -------------------------------------------------

# Combined negative files
df_neg1 <- df_neg %>%
  sample_n(size = 1000000) %>%
  select(id, title, selftext) %>%
  unite(col = "text", title:selftext, sep = " ") %>%
  mutate(
    # Remove Reddit-specific language 
    text = str_remove_all(text, regex("\\[deleted\\]", ignore_case = TRUE)),
    text = str_remove_all(text, regex("\\[removed\\]", ignore_case = TRUE)),
    # Remove markdown links
    text = str_remove_all(text, regex("\\[.*\\]\\(.*\\)")),
    # Remove whitespace characters
    text = str_remove_all(text, regex("\\n|\\r")),
    # Remove URLs
    text = str_remove_all(text, regex("(http|ftp|https):\\/\\/([\\w_-]+(?:(?:\\.[\\w_-]+)+))([\\w.,@?^=%&:\\/~+#-]*[\\w@?^=%&\\/~+#-])")),
    # Remove NA
    text = str_remove_all(text, regex("^NA | NA$", ignore_case = TRUE)),
    # Trim spaces
    text = str_trim(text)
  ) %>%
  # Add label
  mutate(label = 0)

# SPLIT INTO DATASETS -----------------------------------------------------

# Combine positive and negative examples
depression_df <- df_neg1 %>%
  # Take a sample of the negative examples
  filter(id %in% sample(df_neg1$id, size = nrow(depression))) %>%
  bind_rows(depression)

anxiety_df <- df_neg1 %>%
  # Take a sample of the negative examples
  filter(id %in% sample(df_neg1$id, size = nrow(anxiety))) %>%
  bind_rows(anxiety)

stress_df <- df_neg1 %>%
  # Take a sample of the negative examples
  filter(id %in% sample(df_neg1$id, size = nrow(stress))) %>%
  bind_rows(stress)

suicide_df <- df_neg1 %>%
  # Take a sample of the negative examples
  filter(id %in% sample(df_neg1$id, size = nrow(suicide))) %>%
  bind_rows(suicide)

ptsd_df <- df_neg1 %>%
  # Take a sample of the negative examples
  filter(id %in% sample(df_neg1$id, size = nrow(ptsd))) %>%
  bind_rows(ptsd)

# Shuffle the dataframes
depression_df <- depression_df[sample(nrow(depression_df)), ]
anxiety_df <- anxiety_df[sample(nrow(anxiety_df)), ]
stress_df <- stress_df[sample(nrow(stress_df)), ]
suicide_df <- suicide_df[sample(nrow(suicide_df)), ]
ptsd_df <- ptsd_df[sample(nrow(ptsd_df)), ]

# Check balance of class labels
count(depression_df, label)

# WRITE TO FILE -----------------------------------------------------------

write_csv(depression_df, "data/util/dassp/cleaned/depression_df.csv")
write_csv(anxiety_df, "data/util/dassp/cleaned/anxiety_df.csv")
write_csv(stress_df, "data/util/dassp/cleaned/stress_df.csv")
write_csv(suicide_df, "data/util/dassp/cleaned/suicide_df.csv")
write_csv(ptsd_df, "data/util/dassp/cleaned/ptsd_df.csv")
