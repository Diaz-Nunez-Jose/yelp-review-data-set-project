import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("yelp_data_set/yelp_review.csv")
data = list(df.stars.values)
 
objects = ('1', '2', '3', '4', '5')
y_pos = np.arange(len(objects))
performance = [data.count(1), data.count(2), data.count(3), data.count(4), data.count(5)]
 
plt.bar(y_pos, performance, align = 'center', alpha = 0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Counts')
plt.title('Distribution of Star Ratings')
 
plt.tight_layout() #show plot with tight layout
# plt.show() #show the plot
plt.savefig('star-ratings-distribution.png', dpi = 200)