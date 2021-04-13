"""
Take the first 10 reviews texts. Perform word tokenization, lemmatization, part-of-speech tagging. Use Spacy.
Each line should have review ID (i.e., line number from the file), token (i.e. word), lemma, and POS tag.

OUTPUT
-------------------
Dataframe with review_id , tokenization, lemmatization and part-of-speech
- printed
- .csv created

OUTPUT FORMAT
--------------
   review_id     token    lemma    POS
0            1      They     they   PRON
1            1      look     look   VERB
2            1      good     good    ADJ
3            1       and      and  CCONJ
4            1     stick    stick   VERB
..         ...       ...      ...    ...
325         10      just     just    ADV
326         10        as       as    ADP
327         10  promised  promise   VERB
328         10     Great    great    ADJ
329         10       buy      buy   NOUN

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


def processing_text(df_data, df_remove_punctuation=pd.DataFrame()):
    """
    :param df_data
    :param df_remove_punctuation
    :return: df of result containing token, lemma, POS
    """
    res_token, res_lemma, res_pos, res_cnt = [], [], [], []
    cnt = 1
    df_remove_punctuation['final_text'] = df_data['reviewText'].str.replace('[^\w\s]', '')
    for val in df_remove_punctuation['final_text']:
        doc = nlp(val)
        sentences = list(doc.sents)
        for sent in sentences:
            for word in sent:
                res_token.append(word.text)
                res_lemma.append(word.lemma_)
                res_pos.append(word.pos_)
                res_cnt.append(cnt)
        cnt += 1
    result = {'review_id': res_cnt,'token': res_token, 'lemma': res_lemma, 'POS': res_pos}
    return pd.DataFrame.from_dict(result)


if __name__ == '__main__':
    df_records = reading_10_review_text('output_data_cleaned.csv')
    df_res = processing_text(df_records)
    df_res.to_csv('pos_lemma_token.csv', sep=",")
    print(df_res)

