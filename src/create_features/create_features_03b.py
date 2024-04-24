"""
Author = Cory J. Cascalheira
Date = 04/23/2024

The purpose of this script is twofold: (1) to clean up the emojis in the stress & relaxation dataset and (2) to train
generic ML classifiers to predict stress and relaxation.

# References
- https://medium.com/@sarahisdevs/convert-emoji-into-text-in-python-c2afdfd94ab4
- https://towardsdatascience.com/emojis-aid-social-media-sentiment-analysis-stop-cleaning-them-out-bb32a1e5fc8e
- https://www.sciencedirect.com/science/article/abs/pii/S0306457316302321
"""

#region LOAD AND IMPORT

# Dependencies
import os
import pandas as pd
import emoji

# Set file path
my_path = os.getcwd()

# Import data
stress_relax_df = pd.read_csv(my_path + '/data/util/tensi_strength_data/stress_relax_df.csv')

#endregion

#region COVERT EMOJIS TO TEXTS

# Transform emojis in all columns
stress_relax_df['text'] = stress_relax_df['text'].astype(str)
stress_relax_df['text'] = stress_relax_df['text'].map(lambda x: emoji.demojize(x))

# Save file
stress_relax_df.to_csv(my_path + '/data/util/tensi_strength_data/stress_relax_df_demojized.csv')

#endregion

#region MACHINE LEARNING CLASSIFIER TRAINING
#endregion

#region MACHINE LABELING
#endregion
