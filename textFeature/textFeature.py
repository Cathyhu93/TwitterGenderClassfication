import pandas as pd
import numpy as np
import re
import time
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing, svm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, pairwise
from sklearn.metrics import recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

def cleantext(s):
    s = re.sub(URL_PATTERN, ' ', s)
    s = re.sub(HASHTAG_PATTERN, ' ', s)
    s = re.sub(MENTION_PATTERN, ' ', s)
    s = re.sub('\s+', ' ', s)

    return s

def cleanname(s):
    s = re.sub(NAME_PATTERN,' ',s)
    s = re.sub('\s+',' ',s)
    return s

metadata = pd.read_csv("gender-classifier-DFE-791531.csv",encoding='latin1',usecols=[0,5,6,10,14,19])
data = metadata.loc[metadata["gender"].isin(["female","male"]) & metadata["gender:confidence"]>=0.6]

##clean Nan
data.description = data.description.fillna(' ')
data.text = data.text.fillna(' ')

##clean tweets
URL_PATTERN = '(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?$'
HASHTAG_PATTERN = '#'
MENTION_PATTERN = '@\w*'
data['text_clean'] = [ cleantext(s) for s in data['text']]
data['desc_clean'] = [s.lower() for s in data['description']]
NAME_PATTERN = '[^a-zA-Z]'
data['name_clean'] = [cleanname(s) for s in data['name']]

## countVectorizer
vectorizer = CountVectorizer(stop_words='english',lowercase=True,ngram_range=(1,2))
encoder = LabelEncoder()

# ##only use tweet data
# x = vectorizer.fit_transform(data['text_clean'])
# y = encoder.fit_transform(data['gender'])
# nb = MultinomialNB()
# scores = cross_val_score(nb, x, y, cv=10)
# print(scores)
# print(np.mean(scores))
# ##0.613467984917

# ##only use description data
# x = vectorizer.fit_transform(data['desc_clean'])
# y = encoder.fit_transform(data['gender'])
# nb = MultinomialNB()
# scores = cross_val_score(nb, x, y, cv=10)
# print(scores)
# print(np.mean(scores))


# ##use both tweet and desc
# ##NB
# data["all_text"] = data[['text_clean', 'desc_clean']].apply(lambda x: ' '.join(x), axis=1)
# x = vectorizer.fit_transform(data['all_text'])
# nb = MultinomialNB()
# acc = []
# pre = []
# recall = []
# duration =[]
# for i in range(10):
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
#     start =time.time()
#     nb.fit(x_train, y_train)
#     end = time.time()
#     duration.append(end-start)
#     ypred = nb.predict(x_test)
#     acc.append(nb.score(x_test, y_test))
#     pre.append(precision_score(y_test,ypred))
#     recall.append(recall_score(y_test,ypred))

# print(acc)
# print(np.mean(acc))
# print(np.mean(pre))
# print(np.mean(recall))
# print(duration[0])


# ##SVM(cos)
# clf = svm.SVC(kernel = pairwise.cosine_similarity,probability = True)
# acc = []
# pre = []
# recall = []
# duration =[]
# for i in range(10):
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
#     start =time.time()
#     clf.fit(x_train, y_train)
#     end = time.time()
#     duration.append(end-start)
#     ypred = clf.predict(x_test)
#     acc.append(clf.score(x_test, y_test))
#     pre.append(precision_score(y_test,ypred))
#     recall.append(recall_score(y_test,ypred))

# print(acc)
# print(np.mean(acc))
# print(np.mean(pre))
# print(np.mean(recall))
# print(duration[0])


# ##logreg
# clf = LogisticRegression()
# acc = []
# pre = []
# recall = []
# duration =[]
# for i in range(10):
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
#     start =time.time()
#     clf.fit(x_train, y_train)
#     end = time.time()
#     duration.append(end-start)
#     ypred = clf.predict(x_test)
#     acc.append(clf.score(x_test, y_test))
#     pre.append(precision_score(y_test,ypred))
#     recall.append(recall_score(y_test,ypred))

# print(acc)
# print(np.mean(acc))
# print(np.mean(pre))
# print(np.mean(recall))
# print(duration[0])
# ##0.66154726036
# from sklearn.model_selection import GridSearchCV
# param = {"C":(0.1,0.3,0.4,0.5,0.6,0.8,1),"penalty":("l1","l2"),"class_weight":(None,"balanced")}
# model = LogisticRegression()
# clf = GridSearchCV(model,param)
# clf.fit(x,y)
# bestparam = clf.best_params_ 
# bestparam
# ##{'C': 0.1, 'class_weight': None, 'penalty': 'l2'}
# clf = LogisticRegression(C=0.1,penalty='l2')
# acc = []
# pre = []
# recall = []
# duration =[]
# for i in range(10):
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
#     start =time.time()
#     clf.fit(x_train, y_train)
#     end = time.time()
#     duration.append(end-start)
#     ypred = clf.predict(x_test)
#     acc.append(clf.score(x_test, y_test))
#     pre.append(precision_score(y_test,ypred))
#     recall.append(recall_score(y_test,ypred))

# print(acc)
# print(np.mean(acc))
# print(np.mean(pre))
# print(np.mean(recall))
# print(duration[0])

## charVectorizer
vectorizer_char = CountVectorizer(analyzer='char_wb',lowercase=True,ngram_range=(1,5),binary=True)
x = vectorizer_char.fit_transform(data['name_clean'])
y = encoder.fit_transform(data['gender'])
nb = MultinomialNB()