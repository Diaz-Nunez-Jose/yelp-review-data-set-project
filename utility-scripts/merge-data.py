import pandas as pd

csv1 = pd.read_csv('yelp_data_set/yelp_review.csv')
csv2 = pd.read_csv('yelp_data_set/yelp_business.csv')

merged = csv1.merge(csv2, on='business_id')
merged.to_csv("yelp_data_set/yelp_review_business_joined.csv", index=False)\