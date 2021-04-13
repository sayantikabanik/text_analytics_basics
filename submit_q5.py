"""
Use the entire dataset. Take the first 80% dataset for train and remaining 20% for test. On the train set,
obtain TFIDF features (with 50K vocabulary) and learn a multinomial Naïve Bayes model.
Report the accuracy on the test set for this five-class classification problem.
Accuracy should be reported as class-wise precision, recall and F1

OUTPUT
--------------
                        precision    recall  f1-score   support

           1       0.60      0.01      0.02      2663
           2       0.00      0.00      0.00      2241
           3       0.50      0.00      0.00      4296
           4       0.40      0.01      0.01      8050
           5       0.56      1.00      0.72     21618

    accuracy                           0.56     38868
   macro avg       0.41      0.20      0.15     38868
weighted avg       0.49      0.56      0.40     38868
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

df = pd.read_csv('output_data_cleaned.csv', sep=',')

col = ['rating', 'reviewText']
df = df[col]
df = df[pd.notnull(df['reviewText'])]
df.columns = ['rating', 'reviewText']
df['category_id'] = df['rating'].factorize()[0]
category_id_df = df[['rating', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'rating']].values)
# print(df.head(20))

fig = plt.figure(figsize=(8, 6))
df.groupby('rating').reviewText.count().plot.bar(ylim=0)
# plt.show()
# uncomment the above line to view the plot

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                        stop_words='english')
features = tfidf.fit_transform(df.reviewText[0:50000]).toarray()
labels = df.category_id
# print(features.shape) #(50000, 67576)
# uncomment the above line to get the shape of features

"""Now, each of 50000 consumer complaint narratives is represented by 67576 features, 
representing the tf-idf score for different unigrams and bigrams."""

X_train, X_test, y_train, y_test = train_test_split(df['reviewText'], df['rating'], random_state=0, test_size=0.2)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)
predicted_result = clf.predict(count_vect.transform(X_test))
print(classification_report(y_test, predicted_result))
