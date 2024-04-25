"""
Author = Cory J. Cascalheira
Date = 04/25/2024

The purpose of this script is to clean up the emojis in the DASSP dataset.
"""

#region LOAD AND IMPORT

# General dependencies
import os
import pandas as pd
import emoji

# Set file path
my_path = os.getcwd()

# Import data
depression_df = pd.read_csv(my_path + '/data/util/dassp/cleaned/depression_df.csv')
anxiety_df = pd.read_csv(my_path + '/data/util/dassp/cleaned/anxiety_df.csv')
stress_df = pd.read_csv(my_path + '/data/util/dassp/cleaned/stress_df.csv')
suicide_df = pd.read_csv(my_path + '/data/util/dassp/cleaned/suicide_df.csv')
ptsd_df = pd.read_csv(my_path + '/data/util/dassp/cleaned/ptsd_df.csv')

#endregion

#region COVERT EMOJIS TO TEXTS

# Transform emojis and export
depression_df['text'] = depression_df['text'].astype(str)
depression_df['text'] = depression_df['text'].map(lambda x: emoji.demojize(x))
depression_df.to_csv(my_path + '/data/util/dassp/cleaned/depression_df_demojized.csv')

anxiety_df['text'] = anxiety_df['text'].astype(str)
anxiety_df['text'] = anxiety_df['text'].map(lambda x: emoji.demojize(x))
anxiety_df.to_csv(my_path + '/data/util/dassp/cleaned/anxiety_df_demojized.csv')

stress_df['text'] = stress_df['text'].astype(str)
stress_df['text'] = stress_df['text'].map(lambda x: emoji.demojize(x))
stress_df.to_csv(my_path + '/data/util/dassp/cleaned/stress_df_demojized.csv')

suicide_df['text'] = suicide_df['text'].astype(str)
suicide_df['text'] = suicide_df['text'].map(lambda x: emoji.demojize(x))
suicide_df.to_csv(my_path + '/data/util/dassp/cleaned/suicide_df_demojized.csv')

ptsd_df['text'] = ptsd_df['text'].astype(str)
ptsd_df['text'] = ptsd_df['text'].map(lambda x: emoji.demojize(x))
ptsd_df.to_csv(my_path + '/data/util/dassp/cleaned/ptsd_df_demojized.csv')

#endregion
