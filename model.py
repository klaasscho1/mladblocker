import pickle
import random
import re
from nltk import ngrams
from itertools import chain, imap
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes, metrics
from math import log
import numpy as np


def url_to_tokens(url):
    return re.findall(r"[\w']+", url)


def url_to_ngram(url, n):
    tokens = re.findall(r"[\w']+", url)
    return ngrams(tokens, n)


def flatmap(f, items):
    return chain.from_iterable(imap(f, items))


def create_mat(urls, features):
    # Create feature matrix
    fMat = np.zeros((sum([len(v) for v in urls.values()]), len(features)))
    labelsVec = np.zeros((sum([len(v) for v in urls.values()])))

    urlId = 0

    for type in ["block", "pass"]:
        for url in urls[type]:
            if type == "block":
                labelsVec[urlId] = 1

            for token in url_to_tokens(url):
                if token in features:
                    fMat[urlId, features[token]] += 1

            urlId += 1

    return fMat, labelsVec


def tfidf_td(urls, url, token):
    # Count no. of docs containing word
    no_urls_cont = 0
    for u in urls:
        if token in url_to_tokens(u):
            no_urls_cont += 1

    # Count no. of occurrences in document
    no_in_url = url.count(token)

    if no_urls_cont == 0:
        return 0

    return no_in_url * log(len(urls) / no_urls_cont)


# Load data from pickle

block_url_map = pickle.load( open( "url_block_map.pkl", "rb"))

to_block = []
to_pass = []

for (url, block) in block_url_map.most_common():
    if block:
        to_block.append(url)
    else:
        to_pass.append(url)

# Pick 200 random samples from not block list to
# size down the sample size

random.shuffle(to_pass)

to_pass = to_pass[0:10000]


# Train/test split

block_train, block_test = train_test_split(to_block, test_size=0.20)
pass_train, pass_test = train_test_split(to_pass, test_size=0.20)


train_docs = {
    "block": block_train,
    "pass": pass_train
}

test_docs = {
    "block": block_test,
    "pass": pass_test
}


# Create features using TF-IDF

candidate_features = Counter()

for type in ["block", "pass"]:
    for url in train_docs[type]:
        tokens = url_to_tokens(url)
        for token in tokens:
            candidate_features[token] = tfidf_td(train_docs[type], tokens, token)


print(candidate_features.most_common())

# Select features

features = dict()

for idx, (f, v) in enumerate(candidate_features.most_common()):
    if v == 9:
        break

    features[f] = idx

# Create matrices

(trainMat, trainLabels) = create_mat(train_docs, features)
(testMat, goldStandard) = create_mat(test_docs, features)

clf = naive_bayes.MultinomialNB()
clf.fit(trainMat, trainLabels)

# Analysis

predicted = clf.predict(testMat)

clf_acc = metrics.accuracy_score(predicted, goldStandard)
clf_pre = metrics.precision_score(predicted, goldStandard)
clf_rec = metrics.recall_score(predicted, goldStandard)
clf_f1 = metrics.f1_score(predicted, goldStandard)

print clf_acc, clf_pre, clf_rec, clf_f1