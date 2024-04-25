"""
Author = Cory J. Cascalheira
Date = 04/25/2024

The purpose of this script is download r/ptsd data using PushShift.
"""

# Import packages - crawl functions
import os

import requests
import time

# Import packages - data manipulation
import datetime as dt
import pandas as pd

# Set working directory
my_path = os.getcwd()

# Set the URL
url = "https://api.pushshift.io/reddit/search/submission"

# region DEFINE FUNCTIONS

# Define crawl page function - Alex Patry
def crawl_page(subreddit: str, last_page = None):
  """
  Crawl a page of results from a given subreddit.

  :param subreddit: The subreddit to crawl.
  :param last_page: The last downloaded page.

  :return: A page or results.
  """
  params = {"subreddit": subreddit, "size": 500, "sort": "desc", "sort_type": "created_utc"}
  if last_page is not None:
    if len(last_page) > 0:
      # resume from where we left at the last page
      params["before"] = last_page[-1]["created_utc"]
    else:
      # the last page was empty, we are past the last page
      return []
  results = requests.get(url, params)
  if not results.ok:
    # something wrong happened
    raise Exception("Server returned status code {}".format(results.status_code))
  return results.json()["data"]

# Define crawl reddit function - Alex Patry
def crawl_subreddit(subreddit, max_submissions = 10000):
  """
  Crawl submissions from a subreddit.

  :param subreddit: The subreddit to crawl.
  :param max_submissions: The maximum number of submissions to download.

  :return: A list of submissions.
  """
  submissions = []
  last_page = None
  while last_page != [] and len(submissions) < max_submissions:
    last_page = crawl_page(subreddit, last_page)
    submissions += last_page
    time.sleep(5)
  return submissions[:max_submissions]

# endregion

# region DOWNLOAD FROM SUBREDDIT - PTSD

# Pull all the submissions from r/ptsd
submissions_ptsd = crawl_subreddit("ptsd")

# Initialize empty lists
record_id = []
sub_reddit = []
text = []
post_time = []
n_comments = []

# Loop to assign values to lists
for submission in submissions_ptsd:
    record_id.append(submission['id'])
    post_time.append(submission['created_utc'])
    sub_reddit.append(submission['subreddit'])
    # Avoid the key error for a missing selftext key
    try:
        text.append(submission['selftext'])
    except KeyError:
        text.append('')
    n_comments.append(submission['num_comments'])

# Initialize empty data frame
df_ptsd = pd.DataFrame()

# Create data frame from lists
df_ptsd['record_id'] = record_id
df_ptsd['post_time'] = post_time
df_ptsd['subreddit'] = sub_reddit
df_ptsd['text'] = text
df_ptsd['n_comments'] = n_comments

# Empty list initialization
new_time = []

# Create a datetime object
for i in range(0, len(df_ptsd)):
     new_time.append(dt.datetime.fromtimestamp(df_ptsd['post_time'][i]))

# Overwrite the timestamp with the datetime object
df_ptsd['post_time'] = new_time

# Write to file
df_ptsd.to_csv(my_path + '/data/util/dassp/pos_examples/pushshift_ptsd.csv')

# endregion