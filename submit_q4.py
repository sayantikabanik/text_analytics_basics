"""
Take the first 1000 review texts. Using gensim,
create an LDA model with 10 topics.
Report the top 50 words with probs for each of the ten topics. Each line has topic number, word, prob in that topic

References
-----------------
[0] https://towardsdatascience.com/
[1] Topic model — Wikipedia. https://en.wikipedia.org/wiki/Topic_model
[2] Distributed Strategies for Topic Modeling. https://www.ideals.illinois.edu/bitstream/handle/2142/46405/ParallelTopicModels.pdf?sequence=2&isAllowed=y
[3] Topic Mapping — Software — Resources — Amaral Lab. https://amaral.northwestern.edu/resources/software/topic-mapping
[4] A Survey of Topic Modeling in Text Mining. https://thesai.org/Downloads/Volume6No1/Paper_21-A_Survey_of_Topic_Modeling_in_Text_Mining.pdf

OUTPUT (cnt - topic_number)
--------------------------
   cnt   prop             word
0    7  0.051        "easily"
0    8  0.045         "phone"
0    4  0.031       "headset"
0    3  0.030          "pair"
0    9  0.030   "plantronics"
1    3  0.030        "laptop"
0    0  0.028         "feels"
0    6  0.026           "one"
1    0  0.025     "protector"
0    1  0.025        "sounds"
1    4  0.024          "case"
1    1  0.024        "sturdy"
0    5  0.023      "customer"
1    7  0.022      "sidekick"
1    5  0.022        "buying"
1    9  0.022        "almost"
2    7  0.020       "include"
2    4  0.020         "phone"
2    3  0.019   "electronics"
3    0  0.019          "name"
2    0  0.019        "theres"
2    1  0.018         "loves"
2    9  0.018          "read"
3    9  0.018        "inside"
3    4  0.017         "sound"
4    4  0.016           "ear"
2    5  0.016  "disappointed"
4    9  0.015          "true"
2    6  0.015          "like"
1    6  0.015         "great"
3    7  0.015      "complain"
3    5  0.015       "clearly"
3    6  0.014     "bluetooth"
4    0  0.014         "bezel"
3    3  0.014        "covers"
0    2  0.014       "limited"
5    9  0.014        "wanted"
1    8  0.014           "use"
4    5  0.014            "us"
8    9  0.013      "together"
7    9  0.013         "broke"
6    0  0.013    "protectors"
5    5  0.013        "within"
5    0  0.013         "twice"
6    9  0.013         "ports"
4    3  0.013           "cut"
1    2  0.013          "thus"
5    7  0.013       "happens"
4    7  0.013         "party"
9    9  0.013          "cause"
"""

import pandas as pd
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = stopwords.words('english')


def reading_1000_review_text(file_name) -> object:
    """
    :param file_name: inout the csv file
    :return: first 1000 records
    """
    df_1000 = pd.read_csv(file_name, nrows=1000)
    return df_1000


def lower_casing_removing_punctuation(df_1000, df_lower=pd.DataFrame(), df_remove_punctuation=pd.DataFrame()):
    """
    :param df_lower:
    :param df_1000: first 10,00 records
    :return: 1000 lower cased + no punctuation records
    """
    df_lower['reviewText_lower'] = df_1000['reviewText'] = df_1000['reviewText'].str.lower()
    df_remove_punctuation['final_text'] = df_lower['reviewText_lower'].str.replace('[^\w\s]', '')
    return df_remove_punctuation


def sent_to_words(df_remove_punctuation):
    for sentence in df_remove_punctuation['final_text']:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]


if __name__ == '__main__':
    df_records = reading_1000_review_text('output_data_cleaned.csv')
    df_cleaned = lower_casing_removing_punctuation(df_records)
    data_words = list(sent_to_words(df_cleaned))
    data_words_final = remove_stopwords(data_words)
    # Create Dictionary
    id2word = corpora.Dictionary(data_words_final)
    # Create Corpus
    texts = data_words_final
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    topics_cnt = 10
    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=150,
        alpha='auto',
        eta='auto',
        iterations=10,
        num_topics=topics_cnt,
        passes=10,
        eval_every=None
    )
    result = lda_model.print_topics()
    L1 = dict(result)
    words_prop_df =[]
    for k, v in L1.items():
        tmp1 = []
        tmp2 = []
        prop, word = [], []
        tmp2.append(v.split('+'))
        for m in tmp2[0]:
            prop.append(float(m.split('*')[0].replace(' ', '')))
            word.append(m.split('*')[1])
            tmp1.append(k)
        words_prop_df.append(pd.DataFrame.from_dict({'cnt': tmp1, 'prop': prop, 'word': word}))
    df_row = pd.concat([words_prop_df[0], words_prop_df[1], words_prop_df[2], words_prop_df[3], words_prop_df[4], words_prop_df[5],
                        words_prop_df[6], words_prop_df[7], words_prop_df[8], words_prop_df[9]])
    print(df_row.sort_values('prop', ascending=False).head(50))
