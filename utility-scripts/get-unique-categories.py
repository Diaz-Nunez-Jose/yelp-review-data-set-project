filename = "yelp_data_set/unique-categories.txt"
categories = set()

with open(filename) as f:
    for line in f:
       [categories.add(cat) for cat in [cats for cats in line.strip().split(';')]]