# https://stackoverflow.com/questions/42740010/dynamically-assign-similarity-matrices-per-document-to-array-for-export-to-json

import pandas as pd
import numpy as np
import nltk
import string
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

path = "cleaned-up-category-txt-files/"
token_dict = {}
stemmer = PorterStemmer()
xticklabels = []

def tokenize(text):
   tokens = nltk.word_tokenize(text)
   stems = stem_tokens(tokens, stemmer)
   return stems

def stem_tokens(tokens, stemmer):
    stemmed_words = []
    for token in tokens:
        stemmed_words.append(stemmer.stem(token))
    return stemmed_words

for subdir, dirs, files in os.walk(path):
    for file in files:
        file_path = subdir + os.path.sep + file
        xticklabels.append(file[:len(file) - 4])
        with open(file_path, "r", encoding = "utf-8") as file:
            story = file
            text = story.read()
            lowers = text.lower()
            map = str.maketrans('', '', string.punctuation)
            no_punctuation = lowers.translate(map)
            token_dict[file.name.split("\\", 1)[1]] = no_punctuation

tfidf = TfidfVectorizer(tokenizer = tokenize, stop_words = 'english')
tfs = tfidf.fit_transform(token_dict.values())
sim = cosine_similarity(tfs)

import seaborn as sns
import matplotlib.pylab as plt

sns.set(font_scale=0.6)
yticklabels = xticklabels

ax = sns.heatmap(
		sim, 				 
		linewidth = 0.002, 				 
		square = True, 				 
		vmax = 1.0, 				 
		vmin = 0.0, 				 
		cmap = "YlGnBu", 				 
		xticklabels = xticklabels, 				 
		yticklabels = yticklabels	
	 )

plt.tight_layout() #show plot with tight layout
# plt.show()
plt.savefig('similarity-matrix.png', dpi=200)

# corr = np.corrcoef(sim)
# mask = np.zeros_like(corr)
# mask[np.triu_indices_from(mask)] = True
# with sns.axes_style("white"):
#     ax = sns.heatmap(corr, mask=mask, linewidth=0.002, square=True, vmax=1.0, vmin=0.0, cmap="YlGnBu", xticklabels=["asadfsafsdafsadfx","bsadfsafsdx","csadfsadfx"], yticklabels=["ayfsdfsf","bsadfsadfy","csafdsafsy"])
#     plt.show()