Analyze interactively predictions using different probabilistic machine learning methods.

The use case is as follows. We start by modeling a data set. This module models the data using different probabilitisic classifiers.  A probabilistic machine classifer associates a probability with each prediction.

Once the model is built, we use it to analyze test data interactively, particularly seeking answers to these questions:

1. How probable are predicted outcomes by different classifiers?
2. How agreeable are the classifiers for the test data set?

## Features

- Visualize probability of prediction of different ML algorithms for each (test) data point.
This sheds light on how probable predictions are, and how much different algorithms agree.

- Show pairwise correlation of and similarity of ML algorithms.

- Examine each data point.


## Usage

```
import pandas
from sklearn.model_selection import KFold
from probably import Model

model = Model('data/iris.csv')
model.define(
	features = ['SepalWidth','SepalLength','PetalWidth','PetalLength'],
	target = 'Species',
	cv = KFold(10,True),
)
data = pandas.read_csv('data/iris.csv')
test_data = data.sample(15)[['SepalWidth','SepalLength','PetalWidth','PetalLength']]
model.predict(test_data)
```

The output is an html file (built using Python Bokeh), from which you can
interactively analyze predicted data.

## Required Python 3 packages

- bokeh
- numpy
- pandas
- scipy
- scikit-learn

## Supported classification algorithms

- Gaussian process
- Gradient boosting
- Logistic regression
- Random forrest
- Support vector machine

## Snapshots of visualizations

##### Iris data set
<img src="Figs/probably_iris.png">

##### Graduate school admission data set
<img src="Figs/probably_admission.png">