# http://brandonrose.org/clustering

import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
import string

# load nltk's English stopwords as variable called 'stopwords'
stopwords = nltk.corpus.stopwords.words('english')

# load nltk's SnowballStemmer as variabled 'stemmer'
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

# here I define a tokenizer and stemmer which returns the set of stems in the text that it is passed
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

#use extend so it's a big flat list of vocab
totalvocab_stemmed = []
totalvocab_tokenized = []

path = "cleaned-up-category-txt-files"
reviews = []
filenames = []

for subdir, dirs, files in os.walk(path):
    for file in files:
        file_path = subdir + os.path.sep + file
        filenames.append(file[:len(file) - 4])
        with open(file_path, "r", encoding = "utf-8") as f:
            text = f.read()         
            map = str.maketrans('', '', string.punctuation)
            text = text.translate(map)   
            reviews.append(text)
            allwords_stemmed = tokenize_and_stem(text) #for each item in 'synopses', tokenize/stem
            totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
            allwords_tokenized = tokenize_only(text)
            totalvocab_tokenized.extend(allwords_tokenized)

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
# print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

from sklearn.feature_extraction.text import TfidfVectorizer

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(
                        max_df = 0.8, 
                        max_features = 200000,
                        min_df = 0.2, 
                        stop_words = 'english',
                        use_idf = True, 
                        tokenizer = tokenize_and_stem, 
                        ngram_range = (1,3)
                   )

tfidf_matrix = tfidf_vectorizer.fit_transform(reviews) #fit the vectorizer to synopses
# print(tfidf_matrix.shape)

terms = tfidf_vectorizer.get_feature_names()

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)

from scipy.cluster.hierarchy import ward, dendrogram
import matplotlib.pyplot as plt

linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize = (15, 20)) # set size
ax = dendrogram(linkage_matrix, orientation = "right", labels = filenames);

plt.tick_params(
	    axis = 'x',          # changes apply to the x-axis
	    which = 'both',      # both major and minor ticks are affected
	    bottom = 'off',      # ticks along the bottom edge are off
	    top = 'off',         # ticks along the top edge are off
	    labelbottom = 'off'
    )

plt.tight_layout() #show plot with tight layout
# plt.show()
plt.savefig('hierarchical-cluster.png', dpi = 200) #save figure as ward_clusters