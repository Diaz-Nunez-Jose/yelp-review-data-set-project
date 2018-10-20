# Python program to generate WordCloud
# https://www.geeksforgeeks.org/generating-word-cloud-python/
 
# importing all necessery modules
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
import string

import time
start_time = time.time()

# shape of df: (5261668, 9)
# n = 500  # every 500th row, or about 10,500 rows, rather than 5,260,000 = ~54 min
n = 250  # every 500th row, or about 21,046 rows, rather than 5,260,000 =
df = pd.read_csv("yelp_data_set/yelp_review.csv", skiprows=lambda i: i % n != 0)

comment_words = ' '
stopwords = set(STOPWORDS)
 
# iterate through the csv file
for val in df.text:     
    # typecaste each val to string
    val = str(val)
    val = val.lower() # make lower case
    map = str.maketrans('', '', string.punctuation) # remove punctuation
    val = val.translate(map)

    # split the value
    tokens = val.split()
         
    for word in tokens:
    	comment_words = comment_words + word + ' ' 
 
wordcloud = WordCloud(width = 800, 
                      height = 800,
                      background_color ='white',
                      stopwords = stopwords,
                      min_font_size = 10
                    ).generate(comment_words)
 
# plot the WordCloud image                       
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")

# plt.tight_layout(pad = 0) 
plt.tight_layout() #show plot with tight layout
# plt.show() #show the plot
plt.savefig('word-cloud.png', dpi=200)

print("--- %s seconds ---" % (time.time() - start_time))