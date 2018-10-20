# http://www.nltk.org/howto/sentiment.html

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import string

import time
start_time = time.time()

dish = input("Enter the name of a dish or food item to rank restaurants for: ").lower()
print("Mining the data set for: " + dish)

# shape of df: (5261668, 9)
# n = 100  # every 100th row, or about 52,616 rows, rather than 5,260,000
# n = 50  # every 50th row, or about 105,233 rows, rather than 5,260,000
# n = 10  # every 10th row, or about 526,166 rows, rather than 5,260,000
# df = pd.read_csv('yelp_data_set/yelp_review_business_joined.csv', skiprows=lambda i: i % n != 0)
# df = pd.read_csv('yelp_data_set/yelp_review_business_joined.csv')
df = pd.read_csv("yelp_data_set/Hygiene/hygiene-data-merged.csv")
# df = pd.read_csv('yelp_data_set/joined_test.csv')

results = []

for row in df.itertuples():
    review = ""
    review = str(row.text)
    review = review.lower()
    map = str.maketrans('', '', string.punctuation)
    review = review.translate(map)
    review = " ".join(review.split())

    if review.find(dish) != -1 or review.find(dish + "s") != -1 or review.find(dish + "es") != -1:
        sid = SentimentIntensityAnalyzer()
        ss = sid.polarity_scores(review)
        results.append((row.name[1:len(row.name) - 1] + "<br>in " + row.city + ", " + row.state + "<br>Average rating: <b>" + str(row.stars_y) + "</b>/5 stars", 
            row.stars_x, row.stars_y, ss))