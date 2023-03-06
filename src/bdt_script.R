# BOT DETECTION TACTICS (BDTS) FOR CMIPS ----------------------------------

# Author: Cory J. Cascalheira
# Created: 03/06/2023

# The purpose of this script is to execute several bot detection tactics (BDTs)
# on the data from the Qualtrics survey, which is the survey posted on the 
# Internet as an anonymous link.

# Resources:
# - https://psyarxiv.com/gtp6z/

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
  )

# UNREASONABLE TIME AND DURATION ------------------------------------------

# Remove respondents with a duration <= 5 mins
cmips <- cmips %>%
  filter(duration > 5)

# INELIGIBLE OR IMPROBABLE DEMOGRAPHICS -----------------------------------

# ATTENTION CHECKS --------------------------------------------------------

# Ensure participant followed all attention checks
cmips <- cmips %>%
  filter(
    BSMAS_4 == "Sometimes",
    LEC1_9 == "Learned about it",
    IHS_8 == "Often"
  )

# INCONSISTENT ITEM RESPONSE ----------------------------------------------

# Check for inconsistency in key survey items
cmips <- cmips %>%
  # Keep respondents if their ZIP codes match
  filter(zipcode1 == zipcode2)

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

# IP ADDRESS FRAUD CHECK --------------------------------------------------

# Select just IP address
cmips_ip_address <- cmips %>%
  select(email, IPAddress) %>%
  # Must be a data frame, not a tibble, to work
  as.data.frame()

# Get the IP address info from IP Hub
iphub_info <- getIPinfo(cmips_ip_address, "IPAddress", iphub_key = Sys.getenv("IPHUB_KEY"))

# Export the IP info
write_csv(iphub_info, file = "data/iphub_info/iphub_info_03.06.23.csv")

# Import the IP info
iphub_info <- read_csv("data/iphub_info/iphub_info_03.06.23.csv")

# Keep respondents not recommended to block
iphub_keep <- iphub_info %>%
  filter(IP_Hub_recommend_block != 1) %>%
  pull(IPAddress)

# Filter respondents to keep
cmips <- cmips %>%
  filter(IPAddress %in% iphub_keep)
