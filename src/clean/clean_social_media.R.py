"""
Author = Cory J. Cascalheira
Date = 04/15/2024

The purpose of this script is to clean up the emojis in the social media text so we do not lose those useful
signals of information.

# References
- https://medium.com/@sarahisdevs/convert-emoji-into-text-in-python-c2afdfd94ab4
- https://towardsdatascience.com/emojis-aid-social-media-sentiment-analysis-stop-cleaning-them-out-bb32a1e5fc8e
"""

#region LOAD AND IMPORT

# Dependencies
import os
import pandas as pd
import emoji

# Set file path
my_path = os.getcwd()

# Import data
social_media_posts = pd.read_csv(my_path + '/data/participants/combined_social_media/social_media_posts_full.csv')

#endregion

#region COVERT EMOJIS TO TEXTS

# Check example
emoji.demojize('\U0001f49c\U0001f60d Amber Palmer')

# Transform emojis in all columns
social_media_posts['posts_comments'] = social_media_posts['posts_comments'].astype(str)
social_media_posts['posts_comments'] = social_media_posts['posts_comments'].map(lambda x: emoji.demojize(x))

# Save file
social_media_posts.to_csv(my_path + '/data/participants/combined_social_media/social_media_posts_demojized.csv')

#endregion