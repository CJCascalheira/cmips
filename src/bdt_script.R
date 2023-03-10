# BOT DETECTION TACTICS (BDTS) FOR CMIPS ----------------------------------

# Author: Cory J. Cascalheira
# Created: 03/06/2023

# The purpose of this script is to execute several bot detection tactics (BDTs)
# on the data from the Qualtrics survey, which is the survey posted on the 
# Internet as an anonymous link.

# Resources:
# - https://psyarxiv.com/gtp6z/
# - https://cran.r-project.org/web/packages/httr/vignettes/quickstart.html
# - https://stackoverflow.com/questions/34045738/how-can-i-calculate-cosine-similarity-between-two-strings-vectors
# - https://github.com/yusuzech/r-web-scraping-cheat-sheet/blob/master/README.md#rvest4
# - https://evoldyn.gitlab.io/evomics-2018/ref-sheets/R_strings.pdf

# LIBRARIES AND IMPORT DATA -----------------------------------------------

# Load dependencies
library(tidyverse)
library(lubridate)
library(janitor)
library(zipcodeR)
library(rIP)
library(httr)
library(iptools)
library(gplots)
library(lsa)
library(rvest)

# Load botifyR functions
source("util/botifyr/missing_data.R")
source("util/botifyr/attention_checks.R")

# Import data
cmips <- read_csv("data/raw/cmips_qualtrics.csv")

# Remove the test responses
cmips <- cmips[-c(1:6), ] %>%
  # Rename the duration variable
  rename(duration = `Duration (in seconds)`) %>%
  # Convert all emails to lowercase
  mutate(email = tolower(email)) %>%
  mutate(
    # Convert date to ymd format
    StartDate = ymd_hms(StartDate),
    EndDate = ymd_hms(EndDate),
    # Convert to numeric
    duration = as.numeric(duration),
    # Change duration to minutes
    duration = duration / 60
  ) %>%
  # Organize by starting date
  arrange(StartDate)

# Set biweekly start
time_start <- ymd_hms("2023-03-06 00:00:00")

# Set biweekly end
time_end <- ymd_hms("2023-03-10 23:59:59")

# Select respondents who took the survey over the last BDT check period
cmips <- cmips %>%
  filter(StartDate > time_start & StartDate < time_end)

# Check count
nrow(cmips)

# Import previous cohort
passed_bdts_03.06.23 <- read_csv("data/participants/passed_bdts/passed_bdts_03.06.23.csv")

# Combined the previously enrolled participants
previously_enrolled <- c(passed_bdts_03.06.23$email)

# Remove any previously enrolled participants
cmips <- cmips %>% 
  filter(!(email %in% previously_enrolled))

# Check count
nrow(cmips)

# MISSING DATA ------------------------------------------------------------

# Remove respondents with >= 75% missing data
# Keep people with missing data? NO
cmips <- missing_data(cmips, "ResponseId", missing = .75, keep = FALSE)

# Check count
nrow(cmips)

# UNREASONABLE TIME AND DURATION ------------------------------------------

# Remove respondents with a duration <= 5 mins
cmips <- cmips %>%
  filter(duration > 5)

# Check count
nrow(cmips)

# INELIGIBLE OR IMPROBABLE DEMOGRAPHICS -----------------------------------

# Check for cishet folx
cishet_folx <- cmips %>%
  select(ResponseId, sex, gender, sex_or) %>%
  # Cishet women
  mutate(cishet_w = if_else(str_detect(sex, regex("female", ignore_case = TRUE)) &
                              str_detect(gender, regex("cisgender|female", ignore_case = TRUE)) &
                              str_detect(sex_or, regex("hetero", ignore_case = TRUE)), 1, 0)) %>%
  # Cishet men
  mutate(cishet_m = if_else(str_detect(sex, regex("^male", ignore_case = TRUE)) &
                              str_detect(gender, regex("cisgender|male", ignore_case = TRUE)) &
                              str_detect(sex_or, regex("hetero", ignore_case = TRUE)), 1, 0)) %>%
  # Identify cishet people
  filter(cishet_m == 1 | cishet_w == 1) %>%
  pull(ResponseId)

# Remove cishet folx
cmips <- cmips %>% 
  filter(!(ResponseId %in% cishet_folx))

# Check count
nrow(cmips)

# ATTENTION CHECKS --------------------------------------------------------

# Ensure participant are missing no more than 2 attention checks
cmips <- attention_checks(cmips,
                 vars = c("BSMAS_4", "LEC1_9", "IHS_8"),
                 correct_answers = c("Sometimes", "Learned about it", "Often"),
                 threshold = 2)

# Check count
nrow(cmips)

# INCONSISTENT ITEM RESPONSE ----------------------------------------------

# Check for inconsistency in key survey items
cmips <- cmips %>%
  # Keep respondents if their ZIP codes match
  filter(zipcode1 == zipcode2)

# Check count
nrow(cmips)

# ANAGRAMS ----------------------------------------------------------------

# Check for correct anagram response
cmips <- cmips %>%
  # Convert all responses to lowercase
  mutate(
    anagram1 = tolower(anagram1),
    anagram2 = tolower(anagram2),
    anagram3 = tolower(anagram3)
  ) %>%
  # Check for correct answer
  filter(
    anagram1 == "world",
    anagram2 == "peace",
    anagram3 == "happy"
  )

# Check count
nrow(cmips)

# SUSPICIOUS QUALITATIVE DATA ---------------------------------------------

# ...1) Duplicated Qual Responses -----------------------------------------

# Get duplicated qualitative responses 
dupes_goal <- cmips %>% 
  # Select item with qualitative data
  select(ResponseId, goal) %>%
  # Remove NA values
  filter(!is.na(goal)) %>%
  # Covert to lower
  mutate(goal = tolower(goal)) %>%
  # Remove common words
  filter(!goal %in% c("no", "none", "good", "n/a", "nothing", "thank you", "thanks")) %>%
  get_dupes(goal) %>%
  pull(ResponseId)

# Get duplicated qualitative responses 
dupes_fdbk1 <- cmips %>% 
  # Select item with qualitative data
  select(ResponseId, feedback_gen) %>%
  # Remove NA values
  filter(!is.na(feedback_gen)) %>%
  # Covert to lower
  mutate(feedback_gen = tolower(feedback_gen)) %>%
  # Remove common words
  filter(!feedback_gen %in% c("no", "none", "good", "n/a", "nothing", "thank you", "thanks")) %>%
  get_dupes(feedback_gen) %>%
  pull(ResponseId)

# Get duplicated qualitative responses 
dupes_fdbk2 <- cmips %>% 
  # Select item with qualitative data
  select(ResponseId, feedback_sm) %>%
  # Remove NA values
  filter(!is.na(feedback_sm)) %>%
  # Covert to lower
  mutate(feedback_sm = tolower(feedback_sm)) %>%
  # Remove common words
  filter(!feedback_sm %in% c("no", "none", "good", "n/a", "nothing", "thank you", "thanks")) %>%
  get_dupes(feedback_sm) %>%
  pull(ResponseId)

# Combine all duplicated responses
all_dupes <- c(dupes_goal, dupes_fdbk1, dupes_fdbk2)

# Remove respondents with duplicated responses
cmips <- cmips %>%
  filter(!(ResponseId %in% all_dupes))

# Check count
nrow(cmips)

# ...2) Cosine Similarity -------------------------------------------------

# Prepare document for cosine similarity
goals <- cmips %>% 
  # Select item with qualitative data
  select(ResponseId, goal) %>%
  # Remove NA
  filter(!is.na(goal))

# Create temp files
tdm_goals = tempfile()
dir.create(tdm_goals)

# Loop over each response
for (i in 1:nrow(goals)) {
  write(goals$goal[i], 
        file = paste(tdm_goals, goals$ResponseId[i], sep="/")) 
}

# Create a document-term matrix
tdm_goals <- textmatrix(tdm_goals, minWordLength=1)

# Prepare document for cosine similarity
fdbk1 <- cmips %>% 
  # Select item with qualitative data
  select(ResponseId, feedback_gen) %>%
  # Remove NA
  filter(!is.na(feedback_gen))

# Create temp files
tdm_fdbk1 = tempfile()
dir.create(tdm_fdbk1)

# Loop over each response
for (i in 1:nrow(fdbk1)) {
  write(fdbk1$feedback_gen[i], 
        file = paste(tdm_fdbk1, fdbk1$ResponseId[i], sep="/")) 
}

# Create a document-term matrix
tdm_fdbk1 <- textmatrix(tdm_fdbk1, minWordLength=1)

# Prepare document for cosine similarity
fdbk2 <- cmips %>% 
  # Select item with qualitative data
  select(ResponseId, feedback_sm) %>%
  # Remove NA
  filter(!is.na(feedback_sm))

# Create temp files
tdm_fdbk2 = tempfile()
dir.create(tdm_fdbk2)

# Loop over each response
for (i in 1:nrow(fdbk2)) {
  write(fdbk2$feedback_sm[i], 
        file = paste(tdm_fdbk2, fdbk2$ResponseId[i], sep="/")) 
}

# Create a document-term matrix
tdm_fdbk2 <- textmatrix(tdm_fdbk2, minWordLength=1)

# Calculate cosine similarity
cosine_goals <- cosine(tdm_goals)
cosine_fdbk1 <- cosine(tdm_fdbk1)
cosine_fdbk2 <- cosine(tdm_fdbk2) 

# Find respondents with high cosine similarity

# Goals
high_cos1 <- as.data.frame(cosine_goals) %>%
  # Convert to long format
  pivot_longer(cols = everything(), 
               names_to = "ResponseId", 
               values_to = "cos") %>%
  # Remove perfect similarity
  filter(cos != 1) %>%
  # Detect respondents with cos_similarity >= .80
  filter(cos >= .80) %>%
  distinct(ResponseId) %>%
  pull(ResponseId)

# Feedback general
high_cos2 <- as.data.frame(cosine_fdbk1) %>%
  # Convert to long format
  pivot_longer(cols = everything(), 
               names_to = "ResponseId", 
               values_to = "cos") %>%
  # Remove perfect similarity
  filter(cos != 1) %>%
  # Detect respondents with cos_similarity >= .80
  filter(cos >= .80) %>%
  distinct(ResponseId) %>%
  pull(ResponseId)

# Feedback social media
high_cos3 <- as.data.frame(cosine_fdbk2) %>%
  # Convert to long format
  pivot_longer(cols = everything(), 
               names_to = "ResponseId", 
               values_to = "cos") %>%
  # Remove perfect similarity
  filter(cos != 1) %>%
  # Detect respondents with cos_similarity >= .80
  filter(cos >= .80) %>%
  distinct(ResponseId) %>%
  pull(ResponseId)

# Combine
high_cosine_sim <- c(high_cos1, high_cos2, high_cos3)

# Remove respondents with high cosine similarity
cmips <- cmips %>%
  filter(!(ResponseId %in% high_cosine_sim))

# Check count
nrow(cmips)

# IP ADDRESS FRAUD CHECK --------------------------------------------------

# ...1) IP Hub ------------------------------------------------------------

# Select just IP address
cmips_ip_address <- cmips %>%
  select(email, IPAddress) %>%
  # Must be a data frame, not a tibble, to work
  as.data.frame()

# Get the IP address info from IP Hub
iphub_info <- getIPinfo(cmips_ip_address, "IPAddress", iphub_key = Sys.getenv("IPHUB_KEY"))

# Export the IP info
write_csv(iphub_info, file = "data/iphub_info/iphub_info_03.10.23.csv")

# Import the IP info
iphub_info <- read_csv("data/iphub_info/iphub_info_03.10.23.csv")

# Keep respondents not recommended to block
iphub_keep <- iphub_info %>%
  filter(IP_Hub_recommend_block != 1) %>%
  pull(IPAddress)

# Filter respondents to keep
cmips <- cmips %>%
  filter(IPAddress %in% iphub_keep)

# Check count
nrow(cmips)

# ...2) Scamalytics -------------------------------------------------------

# Initialize an empty vector
risk_level_vector <- c()

# For each respondent in the dataframe
for (i in 1:nrow(cmips)) {
  
  # Get the Scamalytics page for their URL
  my_url <- paste0("https://scamalytics.com/ip/", cmips$IPAddress[i])
  
  # Extract the HTML element corresponding to their risk category
  response <- read_html(my_url)
  div_header <- html_elements(response, xpath = '/html/body/div[3]/div[1]')
  
  # Get the risk category from the HTML element
  risk_level <- str_extract(as.character(div_header), regex("\\w+ Risk", ignore_case = TRUE))
  
  # Save the risk category to a vector
  risk_level_vector <- c(risk_level_vector, risk_level) 
}

# Find risky respondents
risky_scamalytics_ids <- cmips %>%
  # Add the risk levels to the response IDs
  select(ResponseId) %>%
  mutate(scamalytics_risk = risk_level_vector) %>%
  # Find respondents with high or very high risk
  filter(scamalytics_risk %in% c("High Risk", "Very High Risk")) %>%
  pull(ResponseId)

# Remove the risky Scamalytics IDs from the data
cmips <- cmips %>%
  filter(!(ResponseId %in% risky_scamalytics_ids))

# Check count
nrow(cmips)

# SAVE CLEANED DATA -------------------------------------------------------

# Filter the data needed for initial enrollment data set
passed_bdts <- cmips %>%
  select(StartDate, ResponseId, name, email, phone, has_fbtw) %>%
  mutate(
    SM_Account = str_extract(has_fbtw, 
                             regex("Twitter and a Facebook|Facebook|Twitter")),
    SM_Account = if_else(str_detect(SM_Account, "Twitter and a Facebook"), 
                         "Both", SM_Account)
  ) %>%
  select(-has_fbtw)
print(passed_bdts)

# Save the data
write_csv(passed_bdts, "data/participants/passed_bdts/passed_bdts_03.10.23.csv")
