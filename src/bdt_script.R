# BOT DETECTION TACTICS (BDTS) FOR CMIPS ----------------------------------

# Author: Cory J. Cascalheira
# Created: 03/06/2023

# The purpose of this script is to execute several bot detection tactics (BDTs)
# on the data from the Qualtrics survey, which is the survey posted on the 
# Internet as an anonymous link.

# If you use any of this code in your project, please remember to cite this 
# script; please use this paper for citation purposes: https://psyarxiv.com/gtp6z/

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

# This script is tailored to a data collection project, CMIPS, where we assess
# for the presence of bots on a weekly basis. Every few days, we run this script
# to see which respondents are reasonably human and, thus, and are ready for the 
# next phase of the project. If you wish to keep this format, you will need
# to change the dates each time you run the script. 

# Check count
nrow(cmips)

# Import previous cohort
# These are the cohorts of respondents who passed all the bot detection tactics
# (BDTs) during the previous sessions in which we ran this script. Each week,
# you will need to add the new cohort here.
passed_bdts_03.06.23 <- read_csv("data/participants/passed_bdts/passed_bdts_03.06.23.csv")
passed_bdts_03.10.23 <- read_csv("data/participants/passed_bdts/passed_bdts_03.10.23.csv")
passed_bdts_03.14.23 <- read_csv("data/participants/passed_bdts/passed_bdts_03.14.23.csv")
passed_bdts_03.17.23 <- read_csv("data/participants/passed_bdts/passed_bdts_03.17.23.csv")
passed_bdts_03.21.23 <- read_csv("data/participants/passed_bdts/passed_bdts_03.21.23.csv")
passed_bdts_03.24.23 <- read_csv("data/participants/passed_bdts/passed_bdts_03.24.23.csv")
passed_bdts_03.27.23 <- read_csv("data/participants/passed_bdts/passed_bdts_03.27.23.csv")
passed_bdts_03.31.23 <- read_csv("data/participants/passed_bdts/passed_bdts_03.31.23.csv")
passed_bdts_04.02.23 <- read_csv("data/participants/passed_bdts/passed_bdts_04.02.23.csv")
passed_bdts_04.07.23 <- read_csv("data/participants/passed_bdts/passed_bdts_04.07.23.csv")
passed_bdts_04.11.23 <- read_csv("data/participants/passed_bdts/passed_bdts_04.11.23.csv")
passed_bdts_04.18.23 <- read_csv("data/participants/passed_bdts/passed_bdts_04.18.23.csv")
passed_bdts_04.21.23 <- read_csv("data/participants/passed_bdts/passed_bdts_04.21.23.csv")
passed_bdts_04.25.23 <- read_csv("data/participants/passed_bdts/passed_bdts_04.25.23.csv")
passed_bdts_04.28.23 <- read_csv("data/participants/passed_bdts/passed_bdts_04.28.23.csv")
passed_bdts_05.02.23 <- read_csv("data/participants/passed_bdts/passed_bdts_05.02.23.csv")
passed_bdts_05.05.23 <- read_csv("data/participants/passed_bdts/passed_bdts_05.05.23.csv")
passed_bdts_05.09.23 <- read_csv("data/participants/passed_bdts/passed_bdts_05.09.23.csv")
# None enrolled on 05.12.23
passed_bdts_05.16.23 <- read_csv("data/participants/passed_bdts/passed_bdts_05.16.23.csv")
passed_bdts_05.19.23 <- read_csv("data/participants/passed_bdts/passed_bdts_05.19.23.csv")
passed_bdts_05.31.23 <- read_csv("data/participants/passed_bdts/passed_bdts_05.31.23.csv")
passed_bdts_06.05.23 <- read_csv("data/participants/passed_bdts/passed_bdts_06.05.23.csv")
passed_bdts_06.11.23 <- read_csv("data/participants/passed_bdts/passed_bdts_06.11.23.csv")
# None enrolled on 06.16.23
passed_bdts_06.20.23 <- read_csv("data/participants/passed_bdts/passed_bdts_06.20.23.csv")
passed_bdts_06.23.23 <- read_csv("data/participants/passed_bdts/passed_bdts_06.23.23.csv")
passed_bdts_06.26.23 <- read_csv("data/participants/passed_bdts/passed_bdts_06.26.23.csv")
passed_bdts_06.30.23 <- read_csv("data/participants/passed_bdts/passed_bdts_06.30.23.csv")
passed_bdts_07.04.23 <- read_csv("data/participants/passed_bdts/passed_bdts_07.04.23.csv")
passed_bdts_07.07.23 <- read_csv("data/participants/passed_bdts/passed_bdts_07.07.23.csv")
passed_bdts_07.11.23 <- read_csv("data/participants/passed_bdts/passed_bdts_07.11.23.csv")
passed_bdts_07.19.23 <- read_csv("data/participants/passed_bdts/passed_bdts_07.19.23.csv")
passed_bdts_07.26.23 <- read_csv("data/participants/passed_bdts/passed_bdts_07.26.23.csv")
passed_bdts_07.30.23 <- read_csv("data/participants/passed_bdts/passed_bdts_07.30.23.csv")
passed_bdts_08.02.23 <- read_csv("data/participants/passed_bdts/passed_bdts_08.02.23.csv")
passed_bdts_08.08.23 <- read_csv("data/participants/passed_bdts/passed_bdts_08.08.23.csv")
# Data collection paused due to IRB oversight 08.10.23
passed_bdts_09.02.23 <- read_csv("data/participants/passed_bdts/passed_bdts_09.02.23.csv")
passed_bdts_09.10.23 <- read_csv("data/participants/passed_bdts/passed_bdts_09.10.23.csv")
passed_bdts_09.17.23 <- read_csv("data/participants/passed_bdts/passed_bdts_09.17.23.csv")
passed_bdts_10.01.23 <- read_csv("data/participants/passed_bdts/passed_bdts_10.01.23.csv")
passed_bdts_10.09.23 <- read_csv("data/participants/passed_bdts/passed_bdts_10.09.23.csv")
passed_bdts_10.14.23 <- read_csv("data/participants/passed_bdts/passed_bdts_10.14.23.csv")
# None enrolled on 10.22.23
passed_bdts_10.31.23 <- read_csv("data/participants/passed_bdts/passed_bdts_10.31.23.csv")
passed_bdts_11.18.23 <- read_csv("data/participants/passed_bdts/passed_bdts_11.18.23.csv")
passed_bdts_12.01.23 <- read_csv("data/participants/passed_bdts/passed_bdts_12.01.23.csv")
# None enrolled on 12.13.23
passed_bdts_12.20.23 <- read_csv("data/participants/passed_bdts/passed_bdts_12.20.23.csv")
passed_bdts_01.02.24 <- read_csv("data/participants/passed_bdts/passed_bdts_01.02.24.csv")
passed_bdts_01.11.24 <- read_csv("data/participants/passed_bdts/passed_bdts_01.11.24.csv")
passed_bdts_01.27.24 <- read_csv("data/participants/passed_bdts/passed_bdts_01.27.24.csv")
passed_bdts_02.12.24 <- read_csv("data/participants/passed_bdts/passed_bdts_02.12.24.csv")
passed_bdts_03.04.24 <- read_csv("data/participants/passed_bdts/passed_bdts_03.04.24.csv")
passed_bdts_03.23.24 <- read_csv("data/participants/passed_bdts/passed_bdts_03.23.24.csv")

# Combined the previously enrolled participants
# You also need to add the cohort here.
previously_enrolled <- c(passed_bdts_03.06.23$email, passed_bdts_03.10.23$email,
                         passed_bdts_03.14.23$email, passed_bdts_03.17.23$email,
                         passed_bdts_03.21.23$email, passed_bdts_03.24.23$email,
                         passed_bdts_03.27.23$email, passed_bdts_03.31.23$email,
                         passed_bdts_04.02.23$email, passed_bdts_04.07.23$email,
                         passed_bdts_04.11.23$email, passed_bdts_04.18.23$email,
                         passed_bdts_04.21.23$email, passed_bdts_04.25.23$email,
                         passed_bdts_04.28.23$email, passed_bdts_05.02.23$email,
                         passed_bdts_05.05.23$email, passed_bdts_05.09.23$email,
                         passed_bdts_05.16.23$email, passed_bdts_05.19.23$email,
                         passed_bdts_05.31.23$email, passed_bdts_06.05.23$email,
                         passed_bdts_06.11.23$email, passed_bdts_06.20.23$email,
                         passed_bdts_06.23.23$email, passed_bdts_06.26.23$email, 
                         passed_bdts_06.30.23$email, passed_bdts_07.04.23$email,
                         passed_bdts_07.07.23$email, passed_bdts_07.11.23$email,
                         passed_bdts_07.19.23$email, passed_bdts_07.26.23$email,
                         passed_bdts_08.02.23$email, passed_bdts_08.08.23$email,
                         passed_bdts_09.02.23$email, passed_bdts_09.10.23$email,
                         passed_bdts_09.17.23$email, passed_bdts_10.01.23$email,
                         passed_bdts_10.09.23$email, passed_bdts_10.14.23$email,
                         passed_bdts_10.31.23$email, passed_bdts_11.18.23$email,
                         passed_bdts_12.01.23$email, passed_bdts_12.20.23$email,
                         passed_bdts_01.02.24$email, passed_bdts_01.11.24$email,
                         passed_bdts_01.27.24$email, passed_bdts_02.12.24$email,
                         passed_bdts_03.04.24$email, passed_bdts_03.23.24$email)

# Remove any previously enrolled participants
cmips <- cmips %>% 
  filter(!(email %in% previously_enrolled))

# Check count - 2947
nrow(cmips)

# MISSING DATA ------------------------------------------------------------

# This code will remove respondents who have >= 75% missing data. You need to
# replace "ResponseId" with whatever the unique identifier is for each
# respondent in your survey. If using Qualtrics, there is no need to change
# anything.

# Remove respondents with >= 75% missing data
# Keep people with missing data? NO
cmips <- missing_data(cmips, "ResponseId", missing = .75, keep = FALSE)

# Check count
nrow(cmips)

# UNREASONABLE TIME AND DURATION ------------------------------------------

# Change the duration that fits your study. For example, if you think it is
# impossible for real participants to take your 60-min survey in 30 mins, then
# change `duration > 5` to `duration > 35`

# Remove respondents with a duration <= 5 mins
cmips <- cmips %>%
  filter(duration > 5)

# Check count
nrow(cmips)

# INELIGIBLE OR IMPROBABLE DEMOGRAPHICS -----------------------------------

# You will need to customize this code to fit your demographic criteria.
# You can also delete this code without jeopardizing the rest of the script.

# Check for cishet folx so we can remove them (i.e., ineligible)
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
                 # Specify the variables that serve as attention checks
                 vars = c("BSMAS_4", "LEC1_9", "IHS_8"), 
                 # Match each variable to the correct answer
                 correct_answers = c("Sometimes", "Learned about it", "Often"),
                 # How many attention checks do respondents need to fail in
                 # order to be removed?
                 threshold = 2)

# Check count
nrow(cmips)

# INCONSISTENT ITEM RESPONSE ----------------------------------------------

# If you have identical items in your survey (e.g., at the beginning and at the 
# end of the survey), use this code to check if the items match.

# Check for inconsistency in key survey items
cmips <- cmips %>%
  # Keep respondents if their ZIP codes match
  filter(zipcode1 == zipcode2)

# Check count
nrow(cmips)

# ANAGRAMS ----------------------------------------------------------------

# Use this code if you added anagrams to your survey. Melissa Simone and I
# have a paper coming out that shows how anagrams are highly effective at
# removing suspected bots.

# Alternatively, you can safely delete this code without affecting the rest of
# the script.

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

# Check for duplicated qualitative responses and remove them. You will need to
# change the qualitative variables to match the qualitative data in your survey.

# Alternatively, you can safely delete this code without affecting the rest of
# the script.

# Get duplicated qualitative responses 
dupes_goal <- cmips %>% 
  # Select item with qualitative data
  # Change these variables to match your unique identifier and qualitative item
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
  # Change these variables to match your unique identifier and qualitative item
  select(ResponseId, feedback_gen) %>%
  # Remove NA values
  filter(!is.na(feedback_gen)) %>%
  # Covert to lower
  mutate(feedback_gen = tolower(feedback_gen)) %>%
  # Remove common words
  filter(!feedback_gen %in% c("no", "none", "good", "n/a", "nothing", "thank you", "thanks", "No suggestions",
                              "Na")) %>%
  get_dupes(feedback_gen) %>%
  pull(ResponseId)

# Get duplicated qualitative responses 
dupes_fdbk2 <- cmips %>% 
  # Select item with qualitative data
  # Change these variables to match your unique identifier and qualitative item
  select(ResponseId, feedback_sm) %>%
  # Remove NA values
  filter(!is.na(feedback_sm)) %>%
  # Covert to lower
  mutate(feedback_sm = tolower(feedback_sm)) %>%
  # Remove common words
  filter(!feedback_sm %in% c("no", "none", "good", "n/a", "nothing", "thank you", "thanks", "No suggestions", 
                             "Na")) %>%
  get_dupes(feedback_sm) %>%
  pull(ResponseId)

# Combine all duplicated responses
all_dupes <- c(dupes_goal, dupes_fdbk1, dupes_fdbk2)

# Keep respondents with reasonable responses
all_dupes <- str_remove(all_dupes, "R_10ugXC5xer4Gyn1")
all_dupes <- str_remove(all_dupes, "R_bE4odaZNISbUre1")

# Show the dupes before removing; n = 9
cmips %>%
  filter((ResponseId %in% all_dupes)) %>%
  select(StartDate, ResponseId, goal, feedback_gen, feedback_sm)

# Remove respondents with duplicated responses
cmips <- cmips %>%
  filter(!(ResponseId %in% all_dupes))

# Check count
nrow(cmips)

# ...2) Cosine Similarity -------------------------------------------------

# Check for qualitative responses that are very similar

# NOTE: this code requires personalization. For example, if you have a variable
# named `qual_1`, you would need to replace all instances of the word `goal` in
# this script with `qual_1`. Repeat this process with `feedback_gen` and
# `feedback_sm`

# Prepare document for cosine similarity
goals <- cmips %>% 
  # Select item with qualitative data
  select(ResponseId, goal) %>%
  # Remove people who may be writing the correct goals
  mutate(correct_goal = if_else(str_detect(goal, regex("artificial intelligence", ignore_case = TRUE)), 1, 0)) %>%
  filter(correct_goal == 0) %>%
  select(-correct_goal) %>%
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
high_cosine_sim

# Show people with high cosine similarity; n = 8
cmips %>%
  filter((ResponseId %in% high_cosine_sim)) %>%
  select(StartDate, goal, feedback_gen, feedback_sm)

# Remove respondents with high cosine similarity
cmips <- cmips %>%
  filter(!(ResponseId %in% high_cosine_sim))

# Check count
nrow(cmips)

# IP ADDRESS FRAUD CHECK --------------------------------------------------

# ...1) IP Hub ------------------------------------------------------------

# To use this code to check for international IP addresses and IP addresses
# in a fraud database, you need to sign up for: https://iphub.info/

# After you sign up, replace IPHUB_KEY with your unique key

# Select just IP address
cmips_ip_address <- cmips %>%
  select(email, IPAddress) %>%
  # Must be a data frame, not a tibble, to work
  as.data.frame()

# Get the IP address info from IP Hub
iphub_info <- getIPinfo(cmips_ip_address, "IPAddress", iphub_key = Sys.getenv("IPHUB_KEY"))

# CHOOSE ONE OR THE OTHER BELOW FILTER OPERATION, NOT BOTH

# Keep respondents not recommended to block
#iphub_keep <- iphub_info %>%
#  filter(IP_Hub_recommend_block != 1) %>%
#  pull(IPAddress)
#cmips <- cmips %>%
#  filter(IPAddress %in% iphub_keep)

# Filter out non-USA
iphub_keep <- iphub_info %>%
  filter(IP_Hub_nonUSIP != 1) %>%
  pull(IPAddress)
cmips <- cmips %>%
  filter(IPAddress %in% iphub_keep)

# Check count
nrow(cmips)

# ...2) Scamalytics -------------------------------------------------------

# Check respondent IP addresses with a free-to-use fraud service.

# Initialize an empty vector
risk_level_vector <- c()
scam_done <- c()

# For each respondent in the dataframe
for (i in 1:nrow(cmips)) {
  
  # Count the iteration
  scam_done <- c(scam_done, 1)
  
  # Get the Scamalytics page for their URL
  my_url <- paste0("https://scamalytics.com/ip/", cmips$IPAddress[i])
  
  # Extract the HTML element corresponding to their risk category
  response <- read_html(my_url)
  div_header <- html_elements(response, xpath = '/html/body/div[3]/div[1]')
  
  # Get the risk category from the HTML element
  risk_level <- str_extract(as.character(div_header), regex("\\w+ Risk", ignore_case = TRUE))
  
  # Save the risk category to a vector
  risk_level_vector <- c(risk_level_vector, risk_level) 
  
  # Get progress
  print(paste0("Scamalytics progress: ", round((sum(scam_done) / nrow(cmips)) * 100), "% ", "done"))
}

# Find risky respondents
risky_scamalytics_ids <- cmips %>%
  # Add the risk levels to the response IDs
  select(ResponseId) %>%
  mutate(scamalytics_risk = risk_level_vector) %>%
  # Find respondents with very high risk
  filter(scamalytics_risk %in% c("High Risk", "Very High Risk")) %>%
  pull(ResponseId)
length(risky_scamalytics_ids)

# Remove the risky Scamalytics IDs from the data
cmips <- cmips %>%
  filter(!(ResponseId %in% risky_scamalytics_ids))

# Check count
nrow(cmips)

# SAVE CLEANED DATA -------------------------------------------------------

# Export your data into a CSV file. These respondents are reasonably human.
# However, we recommend that you now contact each of them by phone to ensure
# they are not fraudulent (i.e., humans that do not qualify for your study,
# but snuck in and were not detected by our BDTs)

# Filter the data needed for initial enrollment data set
passed_bdts <- cmips %>%
  select(StartDate, ResponseId, name, email, phone, has_fbtw, profile_fb, profile_tw) %>%
  mutate(
    SM_Account = str_extract(has_fbtw, 
                             regex("Twitter and a Facebook|Facebook|Twitter")),
    SM_Account = if_else(str_detect(SM_Account, "Twitter and a Facebook"), 
                         "Both", SM_Account)
  ) %>%
  select(-has_fbtw) %>%
  select(StartDate, ResponseId, name, email, phone, SM_Account, everything())
print(passed_bdts)

# Save the data
write_csv(passed_bdts, "data/participants/passed_bdts/passed_bdts_04.10.24.csv")
