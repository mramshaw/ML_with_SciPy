#!/usr/bin/env python

# Load libraries

import matplotlib.pyplot as plt

import pandas
from pandas.plotting import scatter_matrix

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

