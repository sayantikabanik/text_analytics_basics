"""
picking "FIRST" 10,000 review texts from the file "output_data_cleaned.csv"
(run the script <submit_q0.py> to generate the .csv file in your local system)

PRE-PROCESSING
---------------
- Performing lowercasing
- Removing punctuation

EVALUATION
---------------
- Computing IDF of all words in these 10,000 reviews
- Report top 20 words, based on IDF, with their IDF scores
- Report bottom 20 words, based on IDF, with their IDF scores

OUTPUT
-------------------------------------
Top 20 =       feature_name       idf
20994         zx80  9.517293
7281        facade  9.517293
14684    quickdial  9.517293
14680        queue  9.517293
7285      facedown  9.517293
14678  questioning  9.517293
14673       querty  9.517293
7288      facetime  9.517293
14672      queries  9.517293
7290         facil  9.517293
7291        facile  9.517293
14671       queing  9.517293
7293      facility  9.517293
7294    facilmente  9.517293
7295      facinate  9.517293
14670       queens  9.517293
14668      quatity  9.517293
14667        quasi  9.517293
14664      quarrel  9.517293
14687    quickfire  9.517293
---------------------------------------
Bottom 20 =       feature_name       idf
16991           so  2.090149
20247          was  2.089257
20902          you  1.990306
13631        phone  1.918142
12595          not  1.869030
8889          have  1.860956
12882           on  1.804626
3366           but  1.790199
18466         that  1.744751
20611         with  1.729289
9658            in  1.715084
12804           of  1.671681
12202           my  1.547244
7840           for  1.500151
10152           is  1.482662
18610         this  1.400130
18810           to  1.330968
10188           it  1.239627
1794           and  1.232284
18474          the  1.169466

"""
import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
nlp = spacy.load('en_core_web_sm')


def reading_10000_review_text(file_name) -> object:
    """
    :param file_name: inout the csv file
    :return: first 10,000 records
    """
    df_10000 = pd.read_csv(file_name, nrows=10000)
    return df_10000


def lower_casing(df_10000, df_lower=pd.DataFrame()):
    """
    :param df_lower:
    :param df_10000: first 10,000 records
    :return: 10,000 lower cased records
    """
    df_lower['reviewText_lower'] = df_10000['reviewText'] = df_10000['reviewText'].str.lower()
    return df_lower


def remove_punctuation(df_lower, df_remove_punctuation=pd.DataFrame()):
    """
    :param df_remove_punctuation:
    :param df_lower: inout the lower cased processed data
    :return: punctuation free data in df format
    """
    df_remove_punctuation['final_text'] = df_lower['reviewText_lower'].str.replace('[^\w\s]', '')
    return df_remove_punctuation


def idf_values(df_records_10000) -> object:
    tf = TfidfVectorizer(use_idf=True)
    tf.fit_transform(df_records_10000['reviewText'].apply(lambda x: np.str_(x)))
    idf = tf.idf_
    feature_name = tf.get_feature_names()
    idf_df = pd.DataFrame.from_dict({'feature_name': feature_name,  'idf': idf})
    sorted_idf_info = pd.DataFrame(idf_df.sort_values('idf', ascending=False))
    return sorted_idf_info.head(20), sorted_idf_info.tail(20)


if __name__ == '__main__':
    df_records = reading_10000_review_text('output_data_cleaned.csv')
    df_lower_casing = lower_casing(df_records)
    df_no_punctuation = remove_punctuation(df_lower_casing)
    idf_values(df_records)
    idf_info_sorted_top, idf_info_sorted_bottom = idf_values(df_records)
    print("Top 20 =", idf_info_sorted_top)
    print("Bottom 20 =", idf_info_sorted_bottom)

    # <uncomment> to create a .csv with all features and IDF scores
    # idf_info.to_csv('idf_info.csv', sep=",")

    # <uncomment> to print last 5 rows of records, lower_info, punctuation data
    # print(df_records.tail())
    # print(df_lower_casing.tail())
    # print(df_no_punctuation.tail())
