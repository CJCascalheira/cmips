# RECRUITMENT SOURCES -----------------------------------------------------

# Author: Cory J. Cascalheira
# Created: 04/02/2024

# The purpose of this script is to count the recruitment sources.

# LOAD DEPENDENCIES AND IMPORT --------------------------------------------

# Dependencies
library(tidyverse)
library(readxl)

# Import
df_list <- lapply(excel_sheets("data/util/CMIPS Recruitment Database.xlsx"), function(x)
  read_excel("data/util/CMIPS Recruitment Database.xlsx", sheet = x)
)

# COUNT SOURCE TYPES ------------------------------------------------------

# Count the type of contacts across lists and condense
recruitment_df <- map(df_list[1:36], ~ count(., `Type of Contact`)) %>%
  bind_rows()

# Clean up and combine
recruitment_df <- recruitment_df %>%
  filter(!is.na(`Type of Contact`)) %>%
  group_by(`Type of Contact`) %>%
  summarize(N = sum(n)) %>%
  arrange(`Type of Contact`) %>%
  ungroup()

# Clean up the type of contacts
recruitment_df <- recruitment_df %>%
  rename(source = `Type of Contact`) %>%
  mutate(
    source = if_else(str_detect(source, regex("bar|club|nightlife", ignore_case = TRUE)), "bar or club", source),
    source = if_else(str_detect(source, regex("advocacy|civil|politic|activist|rights|justice|movement|union", ignore_case = TRUE)), "advocacy", source),
    source = if_else(str_detect(source, regex("book", ignore_case = TRUE)), "bookstore", source),
    source = if_else(str_detect(source, regex("college|alumni|academic|campus|student|university|women", ignore_case = TRUE)), "college group", source),
    source = if_else(str_detect(source, regex("12-step|therapy|councel|counsel|HIV|health|in home care|therap|clinic|doctor|drug|treatment|surgery|medical", ignore_case = TRUE)), "health services", source),
    source = if_else(str_detect(source, regex("DV supp|IPV |DV shelt|domestic|crisis", ignore_case = TRUE)), "DV or IPV services", source),
    source = if_else(str_detect(source, regex("religi|church|Congregation|spiritual", ignore_case = TRUE)), "religious group", source),
    source = if_else(str_detect(source, regex("youth|alliance|family|GSA", ignore_case = TRUE)), "youth and family org", source),
    source = if_else(str_detect(source, regex("community|Commmunity|Commuity|safe space|PFLAG", ignore_case = TRUE)), "community org", source),
    source = if_else(str_detect(source, regex("accounting|business|accountant|commerce", ignore_case = TRUE)), "business services", source),
    source = if_else(str_detect(source, regex("support|social", ignore_case = TRUE)), "support group", source),
    source = if_else(str_detect(source, regex("education", ignore_case = TRUE)), "education", source),
    source = if_else(str_detect(source, regex("fundraising|fund rai|nonprofit|non-|non profit", ignore_case = TRUE)), "fundraising and nonprofit", source),
    source = if_else(str_detect(source, regex("film|chorus|museum|theatre|travel|gym|sports|spa|sex shop", ignore_case = TRUE)), "leisure", source),
    source = if_else(str_detect(source, regex("pride|lgbt events", ignore_case = TRUE)), "pride org", source),
    source = if_else(str_detect(source, regex("govt|govern|development", ignore_case = TRUE)), "government org", source),
    source = if_else(str_detect(source, regex("homeless|house|pantry|shelter", ignore_case = TRUE)), "homeless services", source),
    source = if_else(str_detect(source, regex("legal|law", ignore_case = TRUE)), "legal services", source),
    source = if_else(str_detect(source, regex("media|magazine|journal", ignore_case = TRUE)), "print media", source),
    source = if_else(str_detect(source, regex("bakery|store|coffee|restaurant|printers|shop|saloon|salon", ignore_case = TRUE)), "retail and restaurant", source),
    source = if_else(str_detect(source, regex("meetup|forum", ignore_case = TRUE)), "other online forum", source),
    source = if_else(str_detect(source, regex("Arizona|association|hotel|Organizarion|organization|yellow|research|scholarship", ignore_case = TRUE)), "other org", source)
  ) %>%
  group_by(source) %>%
  summarize(N = sum(N)) 

# Export
write_csv(recruitment_df, "data/util/recruitment_df.csv")
