# AUTOMATICALLY CHECK SCAMALYTICS -----------------------------------------

# Author = Cory J. Cascalheira (cjcascalheira@gmail.com)
# Date = 02/02/2023

# If you use this script in your work, please cite the paper that
# inspired it: https://psyarxiv.com/gtp6z/

# DO NOT post the script publicly online, but you may share it with other
# qualified researchers as long as the author's name and citation remains
# in the comments. Thanks, and good luck!

# -------------------------------------------------------------------------

# Load dependencies
library(tidyverse)
library(httr)
library(rvest)

# Import your Qualtrics-style survey from CSV
# If you use another survey service, you will need to adjust the code slightly.
# If you use Qualtrics, just change the file path in the read_csv() function
qualtrics_survey <- read_csv("data/raw/cmips_qualtrics.csv")

# IMPORTANT = I recommend running this code AFTER you have completed other bot
# checks because, if the survey is long with multiple IP addresses, the code
# will take a longer time to run. 

# Also, if you are importing as CSV file from Qualtrics, make sure the IP address 
# variable only has IP addresses of missing values. You can check that quickly:
qualtrics_survey$IPAddress

# Check count
nrow(qualtrics_survey)

# Check respondent IP addresses with a free-to-use fraud service.

# Initialize an empty vector
risk_level_vector <- c()

# For each respondent in the dataframe
for (i in 1:nrow(qualtrics_survey)) {
  
  # Set the status
  print("Pulling the Scamalytics information...")
  
  # Get the Scamalytics page for their URL
  my_url <- paste0("https://scamalytics.com/ip/", qualtrics_survey$IPAddress[i])
  
  # Extract the HTML element corresponding to their risk category
  response <- read_html(my_url)
  div_header <- html_elements(response, xpath = '/html/body/div[3]/div[1]')
  
  # Get the risk category from the HTML element
  risk_level <- str_extract(as.character(div_header), regex("\\w+ Risk", ignore_case = TRUE))
  
  # Save the risk category to a vector
  risk_level_vector <- c(risk_level_vector, risk_level) 
  
  # Print status
  print("Risk status updated!")
}

# Find risky respondents
risky_scamalytics_ids <- qualtrics_survey %>%
  # Add the risk levels to the response IDs
  select(ResponseId) %>%
  mutate(scamalytics_risk = risk_level_vector) %>%
  # Find respondents with high or very high risk
  filter(scamalytics_risk %in% c("High Risk", "Very High Risk")) %>%
  pull(ResponseId)

# Remove the risky Scamalytics IDs from the data
qualtrics_survey <- qualtrics_survey %>%
  filter(!(ResponseId %in% risky_scamalytics_ids))

# Check count
nrow(qualtrics_survey)
