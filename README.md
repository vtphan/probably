Machine learning methods often provide predictions without justification.  Fortunately, a few methods provide estimates of probabilities of their predictions being correct.  While probabilistic information can be helpful, it must be used carefully.

This module helps users make sense of probabilistic estimates provided by different machine learning methods.  Naturally, we want to know two basic questions:

1. How probable are predicted outcomes by different classifiers?
2. How agreeable are the classifiers for the test data set?

The goal of this module is to get answers to these and other questions to facilitate the decision making process.

## Usage

Model building
```
import pandas
from probably import Model
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier

# We start by building a model.  We specify a dataset, features, the target variable,
# and the classifiers that we be used.  There are 5 built-in classifiers; users can
# also create and provide their own classifiers.

model = Model(
	data = 'data/admission.csv',
	features = ['gre', 'gpa', 'rank'],
	target = 'admit',
	cv = KFold(10,True),
	classifiers = [ 'rf', 'logit', ('gb', GradientBoostingClassifier()) ],
)

# A model can be saved for later usage.
model.save('admission.model')

# Start the analysis
# model.predict(test_data)
```

Load model and predict new data
```
import pandas
data = pandas.read_csv('data/admission.csv')
test_data = data.sample(20)[['gre', 'gpa', 'rank']]

from probably import Model
model = Model(saved_model = 'admission.model')         # Load a previously saved model.
model.predict(test_data, 'prediction_saved.html')
```

## Execution and visualization

Use the bokeh server to serve your app from the command line:
```
bokeh serve --show yourapp.py
```

This will open a web browser, where the data and predictions can be analyzed and visualized.
When visualizing the data and predictions, each data point can be examined.  Associated with each data point are many attributes.

It is important to know not just the probability of each prediction, but also the contexts of such prediction.  The most important 3 attributes associated with each data point are:

- Probability p of the data point being predicted for a specific label/class.
- The percentage of true positives (with respective to that label/class) with predicted probabilities less than the probability p.
- The percentage of false positives (with respective to that label/class) with predicted probabilities greater than the probability p.

For example, a 0.8 prediction by one method can be very different from a 0.8 prediction by another method.

## Required Python 3 packages

- bokeh
- numpy
- pandas
- scipy
- scikit-learn

These packages can be easily installed if you use the Anaconda distribution.

## Supported classification algorithms

- Gaussian process (gauss)
- Gradient boosting (gboost)
- Logistic regression (logit)
- Random forest (rf)
- Support vector classification (svc)

## Example


##### Graduate school admission data set

Prediction of [admission into UCLA's graduate school](https://stats.idre.ucla.edu/r/dae/logit-regression/) based on GRE score, GPA, and undergraduate school ranking.

As an example, the figure shows a data point with GRE=760, GPA=3.35, undergrad school ranking = 2.  The data point is predicted to be admited (class 1), with probability 0.83.  Further, 85% of true positives have probabilities less than or equal to 0.83; and 16% of false positives have probabilities greater than 0.83.

In this figure, we filtered and selected only data points whose predictions are agreed by 3 or more methods.

By hiding and unhiding predictions of different methods, more information can be gained.


<img src="Figs/probably_admission.png">