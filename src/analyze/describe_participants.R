# FEASIBILITY ANALYSIS ----------------------------------------------------

# Author = Cory Cascalheira
# Date = 03/23/2024

# Calculate basic feasibility metrics and produce consort diagram.

# Resources
# - https://cran.r-project.org/web/packages/consort/vignettes/consort_diagram.html

# LIBRARIES AND DATA ------------------------------------------------------

# Load dependencies
library(tidyverse)
library(readxl)
library(consort)
library(zipcodeR)

# Import data
participant_tracker <- read_excel("data/participants/Participant_Tracking.xlsx",
                                  sheet = 1) %>%
  # Keep only usable rows
  filter(!is.na(ResponseId))

cimps_surveys <- read_csv("data/participants/for_analysis/cmips_surveys.csv")

cmips_raw <- read_csv("data/raw/cmips_qualtrics.csv") %>%
  select(ResponseId, email) %>%
  mutate(email = tolower(email))

cmips_str8_folx <- read_csv("data/participants/util/cmips_str8_folx.csv")

cmips_strain <- read_csv("data/participants/cleaned/cmips_strain_full.csv")

# Remove the test trials from the raw qualtrics survey
cmips_raw <- cmips_raw[-c(1:6),]

# TRACK ENROLLMENT / COMPLETION -------------------------------------------

# Count number of participants retained + missing social media data
participant_tracker %>%
  count(Keep) %>%
  filter(Keep %in% c("YES", "SM_MISS")) %>%
  summarize(retained_sm_miss = sum(n))

# Final sample with participants who completed social media sharing 
participants_smsent <- participant_tracker %>%
  filter(Keep == "YES") %>%
  pull(ResponseId)

# For demographic analysis
demographics <- cimps_surveys %>%
  filter(ResponseId %in% participants_smsent) %>%
  filter(ResponseId != cmips_str8_folx$ResponseId)

# CONSORT DIAGRAM ---------------------------------------------------------

# Filter the participant tracking spreadsheet
participant_tracker_consort <- participant_tracker %>%
  select(ResponseId, Keep, Dropout)

# Get participants who dropped out at ID verification
dropout_smlink <- participant_tracker_consort %>%
  filter(Dropout == "VerifySM_Link")

# Get participants who failed Zoom/phone screen
dropout_idverify <- participant_tracker_consort %>%
  filter(Dropout == "ID_verify")

# Get participants who never sent social media data
dropout_smsent <- participant_tracker_consort %>%
  filter(Keep == "SM_MISS")

# Initialize the disposition dataframe 
consort_df <- cmips_raw %>%
  mutate(trialno = 1:nrow(.)) %>%
  # Add exclusion = failed BDT
  mutate(exc1 = if_else(!(ResponseId %in% participant_tracker_consort$ResponseId), 
                        "Failed bot dectection tactics", NA_character_)) %>%
  # Add exclusion = suspicious social media link
  mutate(exc1 = if_else(ResponseId %in% dropout_smlink$ResponseId, 
                        "Suspicious social media profile", exc1)) %>%
  # Add follow-up exclusion = ID verification
  mutate(fow1 = if_else(ResponseId %in% dropout_idverify$ResponseId, 
                        "Failed Zoom/phone screen", NA_character_)) %>%
  # Add follow-up exclusion = removed for being heterosexual, not LGBQ+
  mutate(fow1 = if_else(ResponseId %in% cmips_str8_folx$ResponseId, 
                        "Heterosexual", fow1)) %>%
  # Add dropout exclusion = never sent social media data
  mutate(fow2 = if_else(ResponseId %in% dropout_smsent$ResponseId, 
                      "Never sent social media data", NA_character_)) 

# Construct consort plot
cmips_consort <- consort_plot(data = consort_df,
                    orders = c(trialno = "Reached via Recruitment Efforts",
                               exc1    = "Bot and Fraud Detection",
                               trialno = "Plausibly Eligible Humans",
                               fow1    = "Human Verification",
                               trialno = "Received Social Media and 2nd Survey Emails",
                               fow2    = "Lost to Follow Up",
                               trialno = "Final Analysis"),
                    side_box = c("exc1", "fow1", "fow2"),
                    cex = 0.9)

plot(cmips_consort)

# TYPE OF SOCIAL MEDIA DATA -----------------------------------------------

# Finish this with Santosh CSVs

# PARTICIPANT DEMOGRAPHICS ------------------------------------------------

# Get geospatial data - https://rpubs.com/lokigao/maps
us_map <- map_data("state")

# Add region to demographics
demographics <- demographics %>%
  select(ParticipantID, zipcode) %>%
  mutate(zipcode = as.character(zipcode)) %>%
  left_join(zip_code_db, by = "zipcode") %>%
  select(ParticipantID, zipcode, state) %>%
  # Create region
  mutate(
    region = if_else(state %in% c("WA", "OR", "CA", "ID", "MT", "WY", "NV", 
                                  "UT", "CO", "AZ", "NM"), "West", NA_character_),
    region = if_else(state %in% c("ND", "SD", "NE", "KS", "MN", "IA", "MO", "WI",
                                  "IL", "IN", "MI", "OH"), "Midwest", region),
    region = if_else(state %in% c("NY", "PA", "NJ", "CT", "RI", "MA", "VT", "NH", 
                                  "ME"), "Northeast", region),
    region = if_else(state %in% c("DE", "DC", "MD", "WV", "VA", "NC", "SC", "GA",
                                  "FL", "AL", "TN", "KY", "MS", "LA", "AR", "OK",
                                  "TX"), "South", region)
  ) %>%
  select(-zipcode, -state) %>%
  left_join(demographics) %>%
  distinct(ParticipantID, .keep_all = TRUE)

# Age
demographics %>%
  summarize(
    age_m = format(mean(age), nsmall = 2),
    age_sd = format(sd(age), nsmall = 2)
  )

# Sexual identity
demographics %>%
  count(sex_or) %>%
  mutate(percent = format((n / nrow(demographics)) * 100, nsmall = 2)) %>%
  arrange(desc(n))

# Gender identity
demographics %>%
  count(gender) %>%
  mutate(percent = format((n / nrow(demographics)) * 100, nsmall = 2)) %>%
  arrange(desc(n))

# Race/ethnicity
demographics %>%
  count(race) %>%
  mutate(percent = format((n / nrow(demographics)) * 100, nsmall = 2)) %>%
  arrange(desc(n))

# Income
demographics %>%
  count(income) %>%
  mutate(percent = format((n / nrow(demographics)) * 100, nsmall = 2)) %>%
  arrange(desc(n))

# Education
demographics %>%
  count(education) %>%
  mutate(percent = format((n / nrow(demographics)) * 100, nsmall = 2)) %>%
  arrange(desc(n))

# Region
demographics %>%
  count(region) %>%
  mutate(percent = format((n / nrow(demographics)) * 100, nsmall = 2)) %>%
  arrange(desc(n))

# Total completed Qualtrics and Adult STRAIN
nrow(cmips_surveys)
nrow(cmips_strain)