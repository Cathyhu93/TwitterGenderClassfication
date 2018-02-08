import numpy as np
import pandas as pd
import re

from sklearn.preprocessing import LabelEncoder
def cleanColor(s):
	pattern = re.compile("\w+")
	match = pattern.match(s)
	if (not match):
		return np.NaN
	else:
		s = match.group(0)
		return int(s.zfill(6),16)

def process(data):
	data['link_color'] = [cleanColor(s) for s in data['link_color']]
	data['sidebar_color'] = [cleanColor(s) for s in data['sidebar_color']]
	data['fav_number'] = data.fav_number.astype(np.int16)
	data['tweet_count'] = data.tweet_count.astype(np.uint16)

	link_m = data['link_color'].mode().iloc[0]
	sidebar_m = data['sidebar_color'].mode().iloc[0]

	data['link_color'].fillna(link_m, inplace=True)
	data['sidebar_color'].fillna(sidebar_m, inplace=True)

	encoder = LabelEncoder()
	data['gender'] = encoder.fit_transform(data['gender'])
	return data


data = pd.read_csv("gender-classifier-DFE-791531.csv",
				   encoding='latin1')
data = data.loc[data["gender"].isin(["female", "male"])]

data = process(data)

# X = data[['fav_number', 'link_color', 'sidebar_color', 'tweet_count']]