import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from preprocessing import preprocess

data_path = "../data/"
database = "final_merged_dataset.csv"
df = pd.read_csv(data_path + database)
df['Unnamed: 0'] = [i for i in range(len(df))]
#print(df.tail())

test = preprocess(df)

from sklearn.feature_extraction.text import CountVectorizer
corpus = df['tweet']
vectorizer = CountVectorizer(max_features = 100)
vectorizer.fit(corpus)
X = vectorizer.transform(corpus).toarray()
Y = df['label']

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

clf = RandomForestClassifier(n_estimators=50)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(Y_pred, Y_test))

errors = []
for i in range(len(Y_pred)):
    if Y_test.values[i] - Y_pred[i] != 0:
        errors.append({
                'text': df['tweet'][Y_test.index[i]], 
                'value': Y_pred[i], 
                'predicted': Y_test.values[i]
                })
 
words_error = []
for error in errors:
    error_text = re.sub(r'[^\w\s]', '', error['text'])
    words_error += word_tokenize(error_text)

stopwords = set(stopwords.words('english')) 
n = 2
words_error = [word for word in words_error if not word in stopwords]
ngram_all=(pd.Series(nltk.ngrams(words_error, n)).value_counts())[:30]
ngram_all=pd.DataFrame(ngram_all)
print(ngram_all)