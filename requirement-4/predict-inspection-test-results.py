# https://www.nltk.org/book/ch06.html

import nltk 

def word_features(word):
    return {word: True}

# positive_words = [word.rstrip('\n') for word in open('yelp_data_set/positive-words-custom.txt')]
# # positive_vocab = np.random.choice(positive_vocab, 10, replace=False)

# negative_words = [word.rstrip('\n') for word in open('yelp_data_set/negative-words-custom.txt')]
# # negative_vocab = np.random.choice(negative_vocab, 2500, replace=False)

labeled_words = ([(word, 'pos') for word in [word.rstrip('\n') for word in open('yelp_data_set/positive-words-custom.txt')]] + \
    [(word, 'neg') for word in [word.rstrip('\n') for word in open('yelp_data_set/negative-words-custom.txt')]])

# labeled_words = ([(word, 'pos') for word in [word.rstrip('\n') for word in open('yelp_data_set/positive-words.txt')]] + \
    # [(word, 'neg') for word in [word.rstrip('\n') for word in open('yelp_data_set/negative-words.txt')]])

import random
random.shuffle(labeled_words)

featuresets = [(word_features(word), label) for (word, label) in labeled_words]
train_set, test_set = featuresets[:90], featuresets[90:] # 70% training, 30% testing, otal of 130
classifier = nltk.NaiveBayesClassifier.train(train_set)

# print(classifier.classify(word_features('dirty')))
# print(classifier.classify(word_features('filthy')))
# print(nltk.classify.accuracy(classifier, test_set))

# https://pythonspot.com/python-sentiment-analysis/
import numpy as np
import pandas as pd
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names
from nltk.corpus import words
import string

df = pd.read_csv("yelp_data_set/Hygiene/hygiene-data-merged.csv")
y_true = []
y_pred = []
positive_scores = []
negative_scores = []

for row in df.itertuples():
    y_true.append(row.passed_inspection)
    # Predict
    neg = 0
    pos = 0
    review = row.text
    review = review.lower()
    map = str.maketrans('', '', string.punctuation)
    review = review.translate(map)    
    review = review.replace("\n", "") 
    review = review.replace("\r", "")
    words = review.split(" ")

    for word in words:
        classResult = classifier.classify(word_features(word))
        if classResult == 'neg':
            neg = neg + 1
        if classResult == 'pos':
            pos = pos + 1
     
    percent_positive = float(pos)/len(words)
    percent_negative = float(neg)/len(words)

    negative_scores.append(percent_negative)
    positive_scores.append(percent_positive)

    # if percent_positive >= percent_negative:
    #     y_pred.append(0)
    # else:
    #     y_pred.append(1)

    if (percent_positive - percent_negative) >= -0.875:
        y_pred.append(0)
    else:
        y_pred.append(1)

accuracy = len([y_pred[i] for i in range(0, len(y_pred)) if y_pred[i] == y_true[i]]) / len(y_pred)
print(accuracy)

from sklearn.metrics import precision_recall_fscore_support
import numpy as np

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
# The recall is intuitively the ability of the classifier to find all the positive samples.
# The F-beta score can be interpreted as a weighted harmonic mean of the precision and recall, where an 
#    F-beta score reaches its best value at 1 and worst score at 0. The F-beta score weights recall more 
#    than precision by a factor of beta. beta == 1.0 means recall and precision are equally important.
# The support is the number of occurrences of each class in y_true.

print(precision_recall_fscore_support(y_true, y_pred, average = 'macro'))
print(precision_recall_fscore_support(y_true, y_pred, average = 'micro'))
print(precision_recall_fscore_support(y_true, y_pred, average = 'weighted'))