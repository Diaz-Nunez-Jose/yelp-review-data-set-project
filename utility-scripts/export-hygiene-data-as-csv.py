import pandas as pd

df1 = pd.read_table("yelp_data_set/Hygiene/test/hygiene3.dat", header=None)
df2 = pd.read_csv("yelp_data_set/Hygiene/test/hygiene.dat.additional", header=None)
df3 = pd.read_csv("yelp_data_set/Hygiene/test/hygiene.dat.labels", header=None)

df1['tmp'] = 1
df2['tmp'] = 1
df3['tmp'] = 1

df12 = pd.merge(df1, df2, on=['tmp'])
df = pd.merge(df12, df3, on=['tmp'])
df = df.drop('tmp', axis=1)

df.to_csv("yelp_data_set/Hygiene/test/hygiene_merged.csv", encoding='utf-8', index=False)

# import winsound
# fname = "C:\\Users\\josed\\Anaconda3\\Lib\\site-packages\\IPython\\lib\\tests\\test.wav"
# for i in range(20):
#   winsound.PlaySound(fname, winsound.SND_FILENAME)