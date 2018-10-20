# http://www.nltk.org/howto/sentiment.html

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import string

import time
start_time = time.time()

dish = input("Enter the name of a dish or food item to rank restaurants for: ").lower()
print("Mining the data set for: " + dish)

# shape of df: (5261668, 9)
# n = 500  # every 100th row, or about 10,520 rows, rather than 5,260,000 = ~4 min
n = 100  # every 100th row, or about 52,616 rows, rather than 5,260,000 = ~4 min
# n = 50  # every 50th row, or about 105,233 rows, rather than 5,260,000
# n = 10  # every 10th row, or about 526,166 rows, rather than 5,260,000
df = pd.read_csv('yelp_data_set/yelp_review_business_joined.csv', skiprows=lambda i: i % n != 0)
# df = pd.read_csv('yelp_data_set/yelp_review_business_joined.csv')

results = []

for row in df.itertuples():
    review_for_search = ""
    review_for_sentiment_analysis = ""

    review_for_search = str(row.text)
    review_for_sentiment_analysis = str(row.text)

    review_for_search = review_for_search.lower()
    map = str.maketrans('', '', string.punctuation)
    review_for_search = review_for_search.translate(map)
    review_for_search = " ".join(review_for_search.split())

    review_for_sentiment_analysis = review_for_sentiment_analysis.replace("\n", "")
    review_for_sentiment_analysis = review_for_sentiment_analysis.replace("\r", "")

    if review_for_search.find(dish) != -1 or review_for_search.find(dish + "s") != -1 or review_for_search.find(dish + "es") != -1:
        sid = SentimentIntensityAnalyzer()
        ss = sid.polarity_scores(review_for_sentiment_analysis)
        restaurant_info = row.name[1:len(row.name) - 1] + "<br>in " + row.city + ", " + row.state + "<br>Average rating: <b>" + str(row.stars_y) + "</b>/5 stars"
        results.append((restaurant_info, row.stars_x, row.stars_y, ss))

results_sorted = sorted(results, key=lambda tup: tup[3]['pos'])
results_sorted = results_sorted[-10:]

x_data = []
for result in results_sorted:
    x_entry = []
    x_entry.append(result[3]['neg'])
    x_entry.append(result[3]['neu'])
    x_entry.append(result[3]['pos'])
    x_data.append(x_entry)

y_data = []
for result in results_sorted:
    y_data.append(result[0])

import plotly.plotly as py
import plotly.graph_objs as go

py.sign_in("jdiaznun", "kG4nbGLR5xuxkNtu3mC7")

top_labels = ['Negative', 'Neutral', 'Positive']
colors = ['rgba(38, 24, 74, 0.8)', 'rgba(71, 58, 131, 0.8)', 'rgba(122, 120, 168, 0.8)']
traces = []

for i in range(0, len(x_data[0])):
    for xd, yd in zip(x_data, y_data):
        traces.append(go.Bar(
            x=[xd[i]],
            y=[yd],
            orientation='h',
            marker=dict(
                color=colors[i],
                line=dict(
                        color='rgb(248, 248, 249)',
                        width=1)
            )
        ))

layout = go.Layout(
    xaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=False,
        zeroline=False,
        domain=[0.15, 1]
    ),
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=False,
        zeroline=False,
    ),
    barmode='stack',
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
    margin=dict(
        l=120,
        r=10,
        t=140,
        b=80
    ),
    showlegend=False,
)

annotations = []

for yd, xd in zip(y_data, x_data):
    # labeling the y-axis
    annotations.append(dict(xref='paper', yref='y',
                            x=0.14, y=yd,
                            xanchor='right',
                            text=str(yd),
                            font=dict(family='Arial', size=14,
                                      color='rgb(67, 67, 67)'),
                            showarrow=False, align='right'))
    # labeling the first percentage of each bar (x_axis)
    annotations.append(dict(xref='x', yref='y',
                            x=xd[0] / 2, y=yd,
                            text=str(xd[0]) + '%',
                            font=dict(family='Arial', size=14,
                                      color='rgb(248, 248, 255)'),
                            showarrow=False))
    # labeling the first Likert scale (on the top)
    if yd == y_data[-1]:
        annotations.append(dict(xref='x', yref='paper',
                                x=xd[0] / 2, y=1.1,
                                text=top_labels[0],
                                font=dict(family='Arial', size=14,
                                          color='rgb(67, 67, 67)'),
                                showarrow=False))
    space = xd[0]
    for i in range(1, len(xd)):
            # labeling the rest of percentages for each bar (x_axis)
            annotations.append(dict(xref='x', yref='y',
                                    x=space + (xd[i]/2), y=yd, 
                                    text=str(xd[i]) + '%',
                                    font=dict(family='Arial', size=14,
                                              color='rgb(248, 248, 255)'),
                                    showarrow=False))
            # labeling the Likert scale
            if yd == y_data[-1]:
                annotations.append(dict(xref='x', yref='paper',
                                        x=space + (xd[i]/2), y=1.1,
                                        text=top_labels[i],
                                        font=dict(family='Arial', size=14,
                                                  color='rgb(67, 67, 67)'),
                                        showarrow=False))
            space += xd[i]

layout['annotations'] = annotations

fig = go.Figure(data=traces, layout=layout)
py.iplot(fig, filename='restaurant-rankings-dish')

print("--- %s seconds ---" % (time.time() - start_time))