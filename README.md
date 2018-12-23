# Machine Learning with SciPy

[![Known Vulnerabilities](http://snyk.io/test/github/mramshaw/ML_with_SciPy/badge.svg?style=plastic&targetFile=requirements.txt)](http://snyk.io/test/github/mramshaw/ML_with_SciPy?style=plastic&targetFile=requirements.txt)

A quick end-to-end exploration of a simple Machine Learning project.

## Motivation

While I had previously investigated
[Machine Learning](http://github.com/mramshaw/Intro-to-ML)
and [Data Cleaning](http://github.com/mramshaw/Data-Cleaning),
the opportunity to follow along and experience an ML project
from end to end seemed like a great way to gain perspective
on what actually happens in a simple project.

The sequence of events is as follows:

1. Install the Python and SciPy platform
2. Load the dataset
3. Summarize the dataset
4. Visualize the dataset
5. Evaluate some algorithms
6. Make some predictions

I had previously done most of these things - with the exception of
__5__ (evaluating different algorithms), which was thus of particular interest.

## Prerequisites

Either __Python 2__ or __Python 3__ is required, as well as
a copy of `pip` (either `pip` for Python 2 or `pip3` for
Python 3).

Install the required libraries as follows:

    $ pip install --user -r requirements.txt

[I never recommend Global installation. Replace with `pip3` for Python 3.]

Find the installed versions by running <kbd>python versions.py</kbd> as shown:

```bash
$ python versions.py
Python: 2.7.12 (default, Nov 12 2018, 14:36:49) 
[GCC 5.4.0 20160609]
scipy: 0.17.0
numpy: 1.14.0
matplotlib: 2.0.2
pandas: 0.20.3
sklearn: 0.20.0
$
```

## Data

We will use the well-known
[Iris data set](http://en.wikipedia.org/wiki/Iris_flower_data_set),
which I previously used in my
[Iris](http://github.com/mramshaw/Intro-to-ML/tree/master/Iris) exercise.

[There we accessed the data set via sklearn's `load_iris` convenience method.]

This data set should be available at:

    http://archive.ics.uci.edu/ml/datasets/Iris

My experience has been that data sets, as well as software libraries, tend
to experience ___drift___ over time. Accordingly, in order to try to replicate
the published results as closely as possible, I downloaded the data set from
the [author's published version](http://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv).

As this is a well-known data set, we will not need to do any data cleaning
(which would generally be a considerable time-sink in any ML exercise).

## Versions

* Python __2.7.12__

[The tutorial also covered Python 3, but I used Python 2]

* matplotlib __2.0.2__
* numpy __1.14.0__
* pandas __0.20.3__
* scipy __0.17.0__
* sklearn __0.20.0__

## Reference

read_csv

    http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html

## To Do

- [ ] Add __Snyk.io__ vulnerability scanning
- [ ] Verify code conforms to `pylint`, `pycodestyle` and `pydocstyle`
- [ ] Investigate [populating missing data](http://machinelearningmastery.com/handle-missing-data-python/) / [Dora](http://github.com/NathanEpstein/Dora)

## Credits

I followed this excellent tutorial:

    http://machinelearningmastery.com/machine-learning-in-python-step-by-step/

While I already had experience with all of the individual steps, it was nice to see it in an end-to-end format.
