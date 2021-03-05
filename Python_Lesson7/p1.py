from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.svm import SVC
from nltk.util import ngrams

'''
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
tfidf_Vect = TfidfVectorizer()
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)
predicted = clf.predict(X_test_tfidf)
score = metrics.accuracy_score(twenty_test.target, predicted)
print(score)
'''

#Part a - using svm instead of clf

'''
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
tfidf_Vect = TfidfVectorizer()
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
svc = SVC()
svc.fit(X_train_tfidf, twenty_train.target)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)
predicted = svc.predict(X_test_tfidf)
score = metrics.accuracy_score(twenty_test.target, predicted)
print(score)

#Output takes forever to appear. In the interest of time, Score is 0.81864
'''

#Part b - setting vectorizer param to use bigram

'''
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
tfidf_Vect = TfidfVectorizer(ngram_range=(1,2))
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)
predicted = clf.predict(X_test_tfidf)
score = metrics.accuracy_score(twenty_test.target, predicted)
print(score)
'''

#Part c - setting vertorizer arg to use stop_words = 'english

#'''
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
tfidf_Vect = TfidfVectorizer(stop_words= 'english')
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)
predicted = clf.predict(X_test_tfidf)
score = metrics.accuracy_score(twenty_test.target, predicted)
print(score)
#'''