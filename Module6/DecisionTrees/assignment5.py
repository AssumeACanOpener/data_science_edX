#!/usr/bin/python3

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz

#https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.names


# 
# TODO: Load up the mushroom dataset into dataframe 'X'
# Verify you did it properly.
# Indices shouldn't be doubled.
# Header information is on the dataset's website at the UCI ML Repo
# Check NA Encoding
#
# .. your code here ..
X = pd.read_csv('Datasets/agaricus-lepiota.data', names=['label', 'capShape', 'capSurface', 'capColor', 'bruises', 'odor', 'gillAttachment', 'gillSpacing', 'gillSize', 'gillColor', 'stalkShape', 'stalkRoot', 'stalkSurfaceAboveRing', 'stalkSurfaceBelowRing', 'stalkColorAboveRing', 'stalkColorBelowRing', 'veilType', 'veilColor', 'ringNumber', 'ringType', 'sporePrintColor', 'population', 'habitat'])

# INFO: An easy way to show which rows have nans in them
#print(X[pd.isnull(X).any(axis=1)])
#exit()


# 
# TODO: Go ahead and drop any row with a nan
#
# .. your code here ..
X = X.dropna(axis=0)
#print(X.shape)


#
# TODO: Copy the labels out of the dset into variable 'y' then Remove
# them from X. Encode the labels, using the .map() trick we showed
# you in Module 5 -- canadian:0, kama:1, and rosa:2
#
# .. your code here ..
y = pd.DataFrame(X.label.map({'e': 0, 'p': 1}))
X = X.drop('label', axis=1)
#print(X.shape)
#print(X.head())
#print(y.shape)
#print(y.head())
#exit()

#
# TODO: Encode the entire dataset using dummies
#
# .. your code here ..
X = pd.get_dummies(X, columns=['capShape', 'capSurface', 'capColor', 'bruises', 'odor', 'gillAttachment', 'gillSpacing', 'gillSize', 'gillColor', 'stalkShape', 'stalkRoot', 'stalkSurfaceAboveRing', 'stalkSurfaceBelowRing', 'stalkColorAboveRing', 'stalkColorBelowRing', 'veilType', 'veilColor', 'ringNumber', 'ringType', 'sporePrintColor', 'population', 'habitat'])
#print(X.shape)
#print(X.head())
#exit()

# 
# TODO: Split your data into test / train sets
# Your test size can be 30% with random_state 7
# Use variable names: X_train, X_test, y_train, y_test
#
# .. your code here ..
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)


#
# TODO: Create an DT classifier. No need to specify any parameters
#
# .. your code here ..
model = DecisionTreeClassifier()
 
#
# TODO: train the classifier on the training data / labels:
# TODO: score the classifier on the testing data / labels:
#
# .. your code here ..
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print("High-Dimensionality Score: ", round((score*100), 3))
#print("High-Dimensionality Score: ", score)

print(X.iloc[:, 63].head())
#
# TODO: Use the code on the courses SciKit-Learn page to output a .DOT file
# Then render the .DOT to .PNGs. Ensure you have graphviz installed.
# If not, `brew install graphviz`.
#
# .. your code here ..
with open("mushrooms.dot", 'w') as f:
    f = export_graphviz(model, out_file=f)
