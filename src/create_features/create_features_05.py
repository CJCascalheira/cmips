"""
Author = Cory J. Cascalheira
Date = 04/22/2024

The purpose of this script is to create features for the CMIPS project using topic models.

The core code is heavily inspired by the following resources:
- https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
- https://radimrehurek.com/gensim/

# Regular expressions in Python
- https://docs.python.org/3/howto/regex.html
"""

#region LIBRARIES AND IMPORT

# Core libraries
import os
import sys
import pandas as pd
import numpy as np
import time

# Import tool for regular expressions
import re

# Load plotting tools
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import seaborn as sns

# Load Gensim libraries
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Initialize spaCy language model
# Must download the spaCy model first in terminal with command: python -m spacy download en_core_web_sm
# May need to restart IDE before loading the spaCy pipeline
import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Load NLTK stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# Improve NLTK stopwords
new_stop_words = [re.sub("\'", "", sent) for sent in stop_words]
stop_words.extend(new_stop_words)
stop_words.extend(['ish', 'lol', 'non', 'im', 'like', 'ive', 'cant', 'amp', 'ok', 'gt'])

# Load GSDMM - topic modeling for short texts (i.e., social media)
from gsdmm import MovieGroupProcess

# Set file path
my_path = os.getcwd()

# Import preprocessed data
cmips = pd.read_csv(my_path + '/data/participants/cleaned/social_media_posts_cleaned.csv')

#endregion

#region HELPER FUNCTIONS

def transform_to_words(sentences):

    """
    A function that uses Gensim's simple_preprocess(), transforming sentences into tokens of word unit size = 1 and removing
    punctuation in a for loop.

    Parameters
    -----------
    sentences: a list
        A list of text strings to preprocess
    """

    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


def remove_stopwords(word_list):

    """
    A function to remove stop words with the NLTK stopword data set. Relies on NLTK.

    Parameters
    ----------
    word_list: a list
        A list of words that represent tokens from a list of sentences.
    """
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in word_list]


def make_bigrams(word_list):
    """
    A function to transform a list of words into bigrams if bigrams are detected by gensim. Relies on a bigram model
    created separately (see below). Relies on Gensim.

    Parameters
    ----------
    word_list: a list
        A list of words that represent tokens from a list of sentences.
    """
    return [bigram_mod[doc] for doc in word_list]


def make_trigrams(word_list):
    """
    A function to transform a list of words into trigrams if trigrams are detected by gensim. Relies on a trigram model
    created separately (see below). Relies on Gensim.

    Parameters
    ----------
    word_list: a list
        A list of words that represent tokens from a list of sentences.
    """
    return [trigram_mod[bigram_mod[doc]] for doc in word_list]


def lemmatization(word_list, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV', 'PROPN']):
    """
    A function to lemmatize words in a list. Relies on spaCy functionality.

    Parameters
    ----------
    word_list: a list
        A list of words that represent tokens from a list of sentences.
    allowed_postags: a list
        A list of language units to process.
    """
    # Initialize an empty list
    texts_out = []

    # For everyone word in the word list
    for word in word_list:

        # Process with spaCy to lemmarize
        doc = nlp(" ".join(word))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

    # Returns a list of lemmas
    return texts_out


def get_optimal_lda(dictionary, corpus, limit=30, start=2, step=2):
    """
    Execute multiple LDA topic models and computer the perplexity and coherence scores to choose the LDA model with
    the optimal number of topics. Relies on Gensim.

    Parameters
    ----------
    dictionary: Gensim dictionary
    corpus: Gensim corpus
    limit: an integer
        max num of topics
    start: an integer
        number of topics with which to start
    step: an integer
        number of topics by which to increase during each model training iteration

    Returns
    -------
    model_list: a list of LDA topic models
    coherence_values: a list
        coherence values corresponding to the LDA model with respective number of topics
    perplexity_values: a list
        perplexity values corresponding to the LDA model with respective number of topics
    """
    # Initialize empty lists
    model_list = []
    coherence_values = []
    perplexity_values = []

    # For each number of topics
    for num_topics in range(start, limit, step):

        # Train an LDA model with Gensim
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=100,
                                                update_every=1, chunksize=2000, passes=10, alpha='auto',
                                                per_word_topics=True)

        # Add the trained LDA model to the list
        model_list.append(model)

        # Compute UMass coherence score and add to list  - lower is better
        # https://radimrehurek.com/gensim/models/coherencemodel.html
        # https://www.os3.nl/_media/2017-2018/courses/rp2/p76_report.pdf
        cm = CoherenceModel(model=model, corpus=corpus, coherence='u_mass')
        coherence = cm.get_coherence()
        coherence_values.append(coherence)

        # Compute Perplexity and add to list - lower is better
        perplex = model.log_perplexity(corpus)
        perplexity_values.append(perplex)

    return model_list, coherence_values, perplexity_values


def top_words(cluster_word_distribution, top_cluster, values):
    """
    Print the top words associated with the GSDMM topic modeling algorithm.

    Parameters
    ----------
    cluster_word_distribution: a GSDMM word distribution
    top_cluster: a list of indices
    values: an integer
    """

    # For each cluster
    for cluster in top_cluster:

        # Sort the words associated with each topic
        sort_dicts = sorted(cluster_word_distribution[cluster].items(), key=lambda k: k[1], reverse=True)[:values]

        # Print the results to the screen
        print('Cluster %s : %s' % (cluster, sort_dicts))
        print('-' * 120)

#endregion

#region PREPROCESS THE TEXT

# Convert text to list
cmips_text = cmips['posts_comments'].values.tolist()

# Transform sentences into words, convert to list
cmips_words = list(transform_to_words(cmips_text))

# Build the bigram and trigram models
bigram = gensim.models.Phrases(cmips_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[cmips_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# Remove stop words
cmips_words_nostops = remove_stopwords(cmips_words)

# Form bigrams
cmips_words_bigrams = make_bigrams(cmips_words_nostops)

# Lemmatize the words, keeping nouns, adjectives, verbs, adverbs, and proper nouns
cmips_words_lemma = lemmatization(cmips_words_bigrams)

# Remove any stop words created in lemmatization
cmips_words_cleaned = remove_stopwords(cmips_words_lemma)

#endregion

#region CREATE DICTIONARY AND CORPUS

# Create Dictionary
id2word = corpora.Dictionary(cmips_words_cleaned)

# Create Corpus
texts = cmips_words_cleaned

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

#endregion

#region EXECUTE THE TOPIC MODELS WITH VANILLA LDA

# Get the LDA topic model with the optimal number of topics
start_time = time.time()
model_list, coherence_values, perplexity_values = get_optimal_lda(dictionary=id2word, corpus=corpus,
                                                                  limit=30, start=2, step=2)
end_time = time.time()
processing_time = end_time - start_time
print(processing_time / 60)
print((processing_time / 60) / 15)

# Plot the coherence scores
# Set the x-axis valyes
limit = 30
start = 2
step = 2
x = range(start, limit, step)

# Create the plot
plt.figure(figsize=(6, 4), dpi=200)
plt.plot(x, coherence_values)
plt.xlabel("Number of Topics")
plt.ylabel("UMass Coherence Score")
plt.xticks(np.arange(min(x), max(x)+1, 2.0))
plt.savefig('results/plots/lda_coherence_plot.png')

# From the plot, the best LDA model is when num_topics == 22
optimal_lda_model = model_list[10]

# Visualize best LDA topic model
# https://stackoverflow.com/questions/41936775/export-pyldavis-graphs-as-standalone-webpage
vis = pyLDAvis.gensim.prepare(optimal_lda_model, corpus, id2word)
pyLDAvis.save_html(vis, 'results/plots/lda.html')

# Get the Reddit post that best represents each topic
# https://radimrehurek.com/gensim/models/ldamodel.html

# Initialize empty lists
lda_output = []
topic_distributions = []

# For each post, get the LDA estimation output
for i in range(len(cmips_text)):
    lda_output.append(optimal_lda_model[corpus[i]])

# For each output, select just the topic distribution
for i in range(len(cmips_text)):
    topic_distributions.append(lda_output[i][0])

# Initialize empty lists
my_participant = []
my_timestamp = []
my_topic = []
my_probability = []

# For each participant
for kth_participant in range(len(cmips['participant_id'])):

    # Get each topic generated in the participant's post
    for ith_topic in range(len(topic_distributions[kth_participant])):
        # Save the participant ID and timestamp
        my_participant.append(cmips['participant_id'].iloc[kth_participant])
        my_timestamp.append(cmips['timestamp'].iloc[kth_participant])

        # Get the topic number
        my_topic.append(topic_distributions[kth_participant][ith_topic][0])

        # Get the topic probability
        my_probability.append(topic_distributions[kth_participant][ith_topic][1])

# Create a dictionary
topic_dict = {
    "participant_id": my_participant,
    "timestamp": my_timestamp,
    "topic_number": my_topic,
    "topic_probability": my_probability
}

# Create dataframe
cmips_lda_df = pd.DataFrame(topic_dict)

# Extract top words
# https://stackoverflow.com/questions/46536132/how-to-access-topic-words-only-in-gensim
lda_top_words = optimal_lda_model.show_topics(num_topics=22, num_words=3)
lda_tup_words = [lda_tup_words[1] for lda_tup_words in lda_top_words]

# Initialize empty list
lad_topic_names = []

# For each topic
for topic in range(len(lda_tup_words)):

    # Extract the top 3 words
    my_words = re.findall("\\w+", lda_tup_words[topic])
    my_elements = [2, 5, 8]

    # Concatenate the top 3 words together and save to list
    my_name = ''.join([my_words[i] for i in my_elements])
    my_name1 = 'lda_' + my_name
    lad_topic_names.append(my_name1)

# Save the topic names
topic_name_dict = {
    "topic_number": list(range(0, 22)),
    "topic_names": lad_topic_names
}

lda_topic_name_df = pd.DataFrame(topic_name_dict)

# Save the dataframe
cmips_lda_df.to_csv("data/participants/features/cmips_feature_set_05_lda.csv")
lda_topic_name_df.to_csv("data/participants/features/lda_topic_name_df.csv")

#endregion

#region EXECUTE THE TOPIC MODELS WITH GSDMM

# Create the vocabulary
vocab = set(x for doc in cmips_words_cleaned for x in doc)

# The number of terms in the vocabulary
n_terms = len(vocab)

# Train the GSDMM models, changing the value of beta given its meaning (i.e., how similar topics need to be to cluster
# together). K is 30, the same number of topic to consider as the above vanilla LDA. Alpha remains 0.1, which reduces
# the probability that a post will join an empty cluster

# Train the GSDMM model, beta = 1.0
mgp_10 = MovieGroupProcess(K=30, alpha=0.1, beta=1.0, n_iters=40)
gsdmm_b10 = mgp_10.fit(docs=cmips_words_cleaned, vocab_size=n_terms)
post_count_10 = np.array(mgp_10.cluster_doc_count)
print('Beta = 1.0. The number of posts per topic: ', post_count_10)

# Train the GSDMM model, beta = 0.9
mgp_09 = MovieGroupProcess(K=30, alpha=0.1, beta=0.9, n_iters=40)
gsdmm_b09 = mgp_09.fit(docs=cmips_words_cleaned, vocab_size=n_terms)
post_count_09 = np.array(mgp_09.cluster_doc_count)
print('Beta = 0.9. The number of posts per topic: ', post_count_09)

# Train the GSDMM model, beta = 0.8
mgp_08 = MovieGroupProcess(K=30, alpha=0.1, beta=0.8, n_iters=40)
gsdmm_b08 = mgp_08.fit(docs=cmips_words_cleaned, vocab_size=n_terms)
post_count_08 = np.array(mgp_08.cluster_doc_count)
print('Beta = 0.8. The number of posts per topic: ', post_count_08)

# Train the GSDMM model, beta = 0.7
mgp_07 = MovieGroupProcess(K=30, alpha=0.1, beta=0.7, n_iters=40)
gsdmm_b07 = mgp_07.fit(docs=cmips_words_cleaned, vocab_size=n_terms)
post_count_07 = np.array(mgp_07.cluster_doc_count)
print('Beta = 0.7. The number of posts per topic: ', post_count_07)

# Train the GSDMM model, beta = 0.6
mgp_06 = MovieGroupProcess(K=30, alpha=0.1, beta=0.6, n_iters=40)
gsdmm_b06 = mgp_06.fit(docs=cmips_words_cleaned, vocab_size=n_terms)
post_count_06 = np.array(mgp_06.cluster_doc_count)
print('Beta = 0.6. The number of posts per topic: ', post_count_06)

# Train the GSDMM model, beta = 0.5
mgp_05 = MovieGroupProcess(K=30, alpha=0.1, beta=0.5, n_iters=40)
gsdmm_b05 = mgp_05.fit(docs=cmips_words_cleaned, vocab_size=n_terms)
post_count_05 = np.array(mgp_05.cluster_doc_count)
print('Beta = 0.5. The number of posts per topic: ', post_count_05)

# Train the GSDMM model, beta = 0.4
mgp_04 = MovieGroupProcess(K=30, alpha=0.1, beta=0.4, n_iters=40)
gsdmm_b04 = mgp_04.fit(docs=cmips_words_cleaned, vocab_size=n_terms)
post_count_04 = np.array(mgp_04.cluster_doc_count)
print('Beta = 0.4. The number of posts per topic: ', post_count_04)

# Train the GSDMM model, beta = 0.3
mgp_03 = MovieGroupProcess(K=30, alpha=0.1, beta=0.3, n_iters=40)
gsdmm_b03 = mgp_03.fit(docs=cmips_words_cleaned, vocab_size=n_terms)
post_count_03 = np.array(mgp_03.cluster_doc_count)
print('Beta = 0.3. The number of posts per topic: ', post_count_03)

# Train the GSDMM model, beta = 0.2
mgp_02 = MovieGroupProcess(K=30, alpha=0.1, beta=0.2, n_iters=40)
gsdmm_b02 = mgp_02.fit(docs=cmips_words_cleaned, vocab_size=n_terms)
post_count_02 = np.array(mgp_02.cluster_doc_count)
print('Beta = 0.2. The number of posts per topic: ', post_count_02)

# Train the GSDMM model, beta = 0.1
mgp_01 = MovieGroupProcess(K=30, alpha=0.1, beta=0.1, n_iters=40)
gsdmm_b01 = mgp_01.fit(docs=cmips_words_cleaned, vocab_size=n_terms)
post_count_01 = np.array(mgp_01.cluster_doc_count)
print('Beta = 0.1. The number of posts per topic: ', post_count_01)

# Remove topics with 0 posts assigned
beta_01 = [x for x in post_count_01 if x > 0]
beta_02 = [x for x in post_count_02 if x > 0]
beta_03 = [x for x in post_count_03 if x > 0]
beta_04 = [x for x in post_count_04 if x > 0]
beta_05 = [x for x in post_count_05 if x > 0]
beta_06 = [x for x in post_count_06 if x > 0]
beta_07 = [x for x in post_count_07 if x > 0]
beta_08 = [x for x in post_count_08 if x > 0]
beta_09 = [x for x in post_count_09 if x > 0]
beta_10 = [x for x in post_count_10 if x > 0]

# Optimal number of topics
num_topic_sum = len(beta_01) + len(beta_02) + len(beta_03) + len(beta_04) + len(beta_05) + len(beta_06) + len(beta_07) + len(beta_08) + len(beta_09) + len(beta_10)
gsdmm_topic_average = num_topic_sum / 10

# Since optimal number of plots is GSDMM is 7.5---use model where beta = 0.3 (~7 topics)

# Rearrange the topics in order of importance
top_index = post_count_03.argsort()[-30:][::-1]

# Get the top 15 words per topic
stdoutOrigin=sys.stdout
sys.stdout = open("data/participants/features/gsdmm_log.txt", "w")
top_words(mgp_03.cluster_word_distribution, top_cluster=top_index, values=15)
sys.stdout.close()
sys.stdout=stdoutOrigin

# Code above is throwing type error, so use this workaround
import json
with open('data/participants/features/gsdmm_log.txt', 'w') as file:
    file.write(json.dumps(mgp_03.cluster_word_distribution))

# Initialize empty list
gsdmm_topics = []

# Predict the topic for each set of words
for i in range(len(cmips_words_cleaned)):
    gsdmm_topics.append(mgp_03.choose_best_label(cmips_words_cleaned[i]))

# Initialize empty lists
topic_classes = []
topic_probs = []

# For each post, extract the dominant topic from the topic distribution
for i in range(len(cmips_text)):

    # Extract the dominant topic
    topic_class = gsdmm_topics[i][0]
    topic_classes.append(topic_class)

    # Extract the probability of the dominant topic
    topic_prob = gsdmm_topics[i][1]
    topic_probs.append(topic_prob)

# Prepare to merge with original dataframe
gsdmm_cmips_df = cmips.loc[:, ['participant_id', 'timestamp']]

# Add the dominant topics and strengths
gsdmm_cmips_df['gsdmm_predicted_topic'] = topic_classes
gsdmm_cmips_df['gsdmm_topic_probability'] = topic_probs

# Save file
gsdmm_cmips_df.to_csv("data/participants/features/cmips_feature_set_05_gsdmm.csv")

#endregion
