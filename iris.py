#!/usr/bin/env python

# Load libraries

import matplotlib.pyplot as plt

import pandas
from pandas.plotting import scatter_matrix

import seaborn as sb

from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ---- Load the iris dataset ----

local_file = "iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(local_file, names=names)

# ---- Summarize the iris dataset ----

# Show the shape (rows & columns) of the dataset
print("Rows, columns = " + str(dataset.shape))
print

# Show the first 20 rows
print(dataset.head(20))
print

# Show the stats of the dataset
print(dataset.describe())
print

# Show the class distribution
print(dataset.groupby('class').size())
print

# ---- Visualize the iris dataset ----

# 1) - Univariate plots

# Plot box-and-whisker (univariate) plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# Plot histograms
dataset.hist()
plt.show()

# 2) - Multivariate plots

# Plot scatter-plot matrix
scatter_matrix(dataset)
plt.show()

# Plot 'seaborn' scatter-plot matrix, broken down by class
sb.pairplot(dataset, hue = 'class', diag_kind = "kde", kind = "scatter")
plt.show()

# ---- Evaluate algorithms ----

# Segment our dataset into training and validation sets
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20 # 20 percent => Implies training data is 80 percent
validation_seed = 7    # Using a seed allows us to compare subsequent runs
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = validation_size, random_state = validation_seed)

# Test option and evaluation metric
test_seed = 7 # Can be different from the previous seed
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits = 10, random_state = test_seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv = kfold, scoring = scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

"""
According to the tutorial, results should be as follows:

LR: 0.966667 (0.040825)
LDA: 0.975000 (0.038188)
KNN: 0.983333 (0.033333)
CART: 0.975000 (0.038188)
NB: 0.975000 (0.053359)
SVM: 0.981667 (0.025000)

[Along with some deprecation warnings, these are what I got.]
"""
