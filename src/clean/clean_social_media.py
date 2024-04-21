"""
Author = Cory J. Cascalheira
Date = 04/15/2024

The purpose of this script is to clean up the emojis in the social media text so we do not lose those useful
signals of information. The script also translate any non-English text to English.

# References
- https://medium.com/@sarahisdevs/convert-emoji-into-text-in-python-c2afdfd94ab4
- https://towardsdatascience.com/emojis-aid-social-media-sentiment-analysis-stop-cleaning-them-out-bb32a1e5fc8e
"""

#region LOAD AND IMPORT

# Dependencies
import os
import re
import pandas as pd
import emoji
from deep_translator import GoogleTranslator

# Set file path
my_path = os.getcwd()

# Import data
social_media_posts = pd.read_csv(my_path + '/data/participants/combined_social_media/social_media_posts_full.csv')

#endregion

#region TRANSLATE INTO ENGLISH

# Participants most likely to write in Spanish
spanish_speakers = ['CMIPS_0214', 'CMIPS_0217', 'CMIPS_0218', 'CMIPS_0233']

# Split the dataframe
spanish_df = social_media_posts[social_media_posts['participant_id'].isin(spanish_speakers)]
social_media_posts = social_media_posts[~social_media_posts['participant_id'].isin(spanish_speakers)]

# Get a list of posts and comments
test_list = spanish_df['posts_comments'].to_list()

# Initialize an empty list
translated_list = []

# Loop through each post and translate
for my_sentence in range(len(test_list)):

    # If the string only has special characters
    if re.match("^\W+$", test_list[my_sentence]):

        # Just keep the posts as is
        print('Post is just special characters, skipping')
        translated_list.append(test_list[my_sentence])

    # If the string only has special characters
    elif re.match("^\d+$", test_list[my_sentence]):

        # Just keep the posts as is
        print('Post is just numbers, skipping')
        translated_list.append(test_list[my_sentence])

    # If the post is too long for Google Translator
    elif len(test_list[my_sentence]) >= 5000:

        # Just keep the posts as is
        print('Post too long, skipping')
        translated_list.append(test_list[my_sentence])

    else:

        # Translate the words
        translated_string = GoogleTranslator(source='spanish', target='english').translate(test_list[my_sentence])

        # Save to list
        translated_list.append(translated_string)

    # Print the progress
    print("Progress: %s" % (len(translated_list) / len(test_list)))

# Manual adjustment due to TypeError - some n = 108 posts not translated, which is a small amount of noise
# in the overall project, so retained for now
my_list = test_list[14483:14590]
full_list = translated_list + my_list

# Replace the column with the list
spanish_df['posts_comments'] = full_list

# Merge the dataframes back together
social_media_posts = pd.concat([social_media_posts, spanish_df], axis=0)

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
