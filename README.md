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

Model building
```
import pandas
from sklearn.model_selection import KFold
from probably import Model
from sklearn.ensemble import GradientBoostingClassifier

model = Model(
	data = 'data/admission.csv',
	features = ['gre', 'gpa', 'rank'],
	target = 'admit',
	cv = KFold(10,True),
	classifiers = [ 'rf', 'logit', ('gb', GradientBoostingClassifier()) ],
)
model.save('admission.model')

# data = pandas.read_csv('data/admission.csv')
# test_data = data.sample(50, random_state=42)[['gre', 'gpa', 'rank']]
# model.predict(test_data)
```

Load model and predict new data
```
import pandas
data = pandas.read_csv('data/admission.csv')
test_data = data.sample(20)[['gre', 'gpa', 'rank']]

from probably import Model
model = Model(saved_model = 'admission.model')
model.predict(test_data, 'prediction_saved.html')
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
- Random forest
- Support vector machine

## Snapshots of visualizations

##### Iris data set
<img src="Figs/probably_iris.png">

##### Graduate school admission data set
<img src="Figs/probably_admission.png">