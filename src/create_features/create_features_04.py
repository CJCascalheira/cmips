"""
Author = Cory J. Cascalheira
Date = 04/20/2024

The purpose of this script is to create features for the CMIPS project using word embeddings.
"""

#region LOAD AND IMPORT

# Load core dependencies
import os
import pandas as pd
import numpy as np

# Import NLTK
import nltk
from nltk.tokenize import RegexpTokenizer
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.append('amp')

# Load other libraries
# May need to install module manually with pip install importlib-metadata
import gensim.downloader as api
import spacy

# Run in terminal: python -m spacy download en_core_web_sm
# May need to restart IDE before loading the spaCy pipeline
nlp = spacy.load('en_core_web_sm')

# Set file path
my_path = os.getcwd()

# Import data
cmips = pd.read_csv(my_path + '/data/participants/cleaned/social_media_posts_full.csv')

# Rename column
cmips = cmips.rename(columns={'posts_comments': 'text'})

#endregion

#region WORD2VEC MODEL ------------------------------------------------------------------

# Create empty list
corpus_coded = []

# Set the stop words from NLTK
stop_words = set(stopwords.words('english'))

# Create a custom tokenizer to remove punctuation
tokenizer = RegexpTokenizer(r'\w+')

# Create corpus
for string in cmips['text'].astype(str).tolist():

    # Remove strange characters
    string = string.replace('\r', '')
    string = string.replace('*', '')

    # Get tokens (i.e., individual words)
    tokens = tokenizer.tokenize(string)

    # Set a list holder
    filtered_sentence = []

    # For each token, remove the stop words
    for w in tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    # Save list of tokens (i.e., sentences) to preprocessed corpus
    corpus_coded.append(filtered_sentence)

# Load the Word2vec model
wv = api.load('word2vec-google-news-300')

# List embeddings for each post
post_embeddings = []

# For every word in every sentence within the corpus
for sentence in corpus_coded:

    # List of word embeddings
    w2v_embeddings = []

    # Get the word embeddings for each word
    for word in sentence:

        # See if there is a pretrained word embedding
        try:
            vector_representation = wv[word]
            w2v_embeddings.append(vector_representation)

        # If there is no pretrained word embedding
        except KeyError:
            vector_representation = np.repeat(0, 300)
            w2v_embeddings.append(vector_representation)

    # Save the word embeddings at the post level
    post_embeddings.append(w2v_embeddings)

# Set a holder variable
avg_post_embeddings = []

# Aggregate word embeddings
for post in post_embeddings:

    # Transform embedding into data frame where each row is a word and each column is the embedding dimension
    df = pd.DataFrame(post)

    # Square each element in the data frame to remove negatives
    df = df.apply(np.square)

    # Get the mean of each embedding dimension
    df = df.apply(np.mean, axis=0)

    # The average word embedding for the entire Reddit post
    avg_embedding = df.tolist()

    # Append to list
    avg_post_embeddings.append(avg_embedding)

# Create a dataframe with the average word embeddings of each post
embedding_df = pd.DataFrame(avg_post_embeddings)

# Rename the columns
embedding_df = embedding_df.add_prefix('w2v_')

# Add average word embeddings
cmips1 = pd.concat([cmips, embedding_df], axis=1)

# Export files
cmips1.to_csv(my_path + '/data/participants/features/cmips_feature_set_04.csv')

#endregion
