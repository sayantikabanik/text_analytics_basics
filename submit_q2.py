"""
Take the first 10 review texts. Perform sentence detection using Spacy.
Each line should have review ID (i.e., line number from the file) and the sentence itself.

IMPROVE:
-------
,/and sentence detection

OUTPUT
-------------------
Dataframe with review_id and sentence
- printed
- .csv created
,review_id,sentence
0,1,They look good and stick good!
1,1,I just don't like the rounded shape because I was always bumping it and Siri kept popping up
2,1,and it was irritating.
3,1,I just won't buy a product like this again
4,2,These stickers work like the review says they do.
5,2,They stick on great
6,2,and they stay on the phone.
7,2,They are super stylish and I can share them with my sister. :)
8,3,These are awesome and make my phone look so stylish!
9,3,I have only used one so far and have had it on for almost a year!
10,3,CAN YOU BELIEVE THAT!
11,3,ONE YEAR!!
12,3,Great quality!
13,4,Item arrived in great time and was in perfect condition.
14,4,"However, I ordered these buttons because they were a great deal and included a FREE screen protector."
15,4,I never received one.
16,4,"Though its not a big deal, it would've been nice to get it since they claim it comes with one."
17,5,awesome!
18,5,"stays on, and looks great."
19,5,can be used on multiple apple products.
20,5," especially having nails, it helps to have an elevated key."
21,6,These make using the home button easy.
22,6,My daughter and I both like them.
23,6,
24,6,I would purchase them again.
25,6,Well worth the price.
26,7,Came just as described..
27,7,It doesn't come unstuck and its cute!
28,7,People ask where I got them from & it's great when driving.
29,8,it worked for the first week then it only charge my phone to 20%.
30,8,it is a waste of money.
31,9,"Good case, solid build."
32,9,Protects phone all around with good access to buttons.
33,9,Battery charges with full battery lasts me a full day.
34,9,I usually leave my house around 7am and return at 10pm.
35,9,I'm glad that it lasts from start to end.
36,9,5/5
37,10,This is a fantastic case.
38,10,Very stylish and protects my phone.
39,10,"Easy access to all buttons and features, without any loss of phone reception."
40,10,"But most importantly, it double power, just as promised."
41,10,Great buy

"""

import pandas as pd
import spacy

"""
Loading the language model
"""
nlp = spacy.load('en_core_web_sm')


def reading_10_review_text(file_name) -> object:
    """
    :param file_name: inout the csv file
    :return: first 10 records
    """
    df_10 = pd.read_csv(file_name, nrows=10)
    return df_10


def sentence_detection(df_data):
    cnt = 1
    res_cnt, res_sent_each = [], []
    for val in df_data['reviewText']:
        doc = nlp(val)
        sentences = list(doc.sents)
        for each_sent in sentences:
            res_cnt.append(cnt)
            res_sent_each.append(each_sent)
        cnt += 1
    result = {'review_id': res_cnt, 'sentence': res_sent_each}
    return pd.DataFrame.from_dict(result)


if __name__ == '__main__':
    df_records = reading_10_review_text('output_data_cleaned.csv')
    sentence_detection_df = sentence_detection(df_records)
    sentence_detection_df.to_csv('sentence_detection.csv', sep=",")
    print(sentence_detection_df)
