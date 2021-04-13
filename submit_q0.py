"""
EXTRACTION & OUTPUT
--------------------
Extracting the two fields from the json json
reviewText/overall

The output is stored in the .csv file
Change the filename as per requirement or pass an <arg> in the python command

OUTPUT of df.head()
-------------------
 reviewText  rating
0  They look good and stick good! I just don't li...     4.0
1  These stickers work like the review says they ...     5.0
2  These are awesome and make my phone look so st...     5.0
3  Item arrived in great time and was in perfect ...     4.0
4  awesome! stays on, and looks great. can be use...     5.0
"""

import json
import pandas as pd

# change the file path as per system/user
file_path = "/Users/sbanik/PycharmProjects/textAnalytics/assignemnt/data_cell.json"

data_list = []
res = {}
reviewText = []
rating = []

for line in open(file_path, 'r'):
    data_list.append(json.loads(line))

for dict_val in data_list:
    res.update(dict_val)
    reviewText.append((res['reviewText']))
    rating.append(res['overall'])

data_df = {'reviewText': reviewText, 'rating': rating}

# creating dataframe
df = pd.DataFrame.from_dict(data_df)

print(df.head())

# saving reviewText/overall(rating)to a csv
df.to_csv('output_data_cleaned.csv', sep=",")
