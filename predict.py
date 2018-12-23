import pickle
import re
import numpy as np
import sys

CLF_PKL_PATH = "data/trained_model.pkl"
FEAT_PKL_PATH = "data/features.pkl"


def url_to_tokens(url):
    return re.findall(r"[\w']+", url)


def url_to_mat(url, features):
    tokens = url_to_tokens(url)
    fMat = np.zeros(len(features))

    for token in tokens:
        if token in features:
            fMat[features[token]] += 1

    return [fMat]


clf = pickle.load(open(CLF_PKL_PATH, "rb"))
features = pickle.load(open(FEAT_PKL_PATH, "rb"))

url = sys.argv[0]

tm = url_to_mat(url, features)

print "Ad?:", clf.predict(tm)[0] > 0