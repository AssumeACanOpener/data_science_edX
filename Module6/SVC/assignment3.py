#!/usr/bin/python3

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing

X = pd.read_csv('Datasets/parkinsons.data')
X = X.drop('name', axis=1)
y = pd.DataFrame(X.status)
X = X.drop('status', axis=1)

#X = preprocessing.Normalizer(norm='l1').fit_transform(X)
X = preprocessing.MaxAbsScaler().fit_transform(X)
#X = preprocessing.MinMaxScaler().fit_transform(X)
#X = preprocessing.StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

#C = 1.65
#gamma = 0.005
#model = SVC(C=C, gamma=gamma)
#model.fit(X_train, y_train.values.ravel())
#print(model.score(X_test, y_test.values.ravel()))

best = 0
best_C = 0
best_gamma = 0

for C in np.arange(0.05, 2, 0.05):
    for gamma in np.arange(0.001, 0.01, 0.001):
        model = SVC(C=C, gamma=gamma)
        model.fit(X_train, y_train.values.ravel())
        s = model.score(X_test, y_test.values.ravel())
        if s > best:
            best = s
            best_C = C
            best_gamma = gamma

print('best C:', best_C)
print('best gamma:', best_gamma)
print('best score:', best)
