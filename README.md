Analyze interactively predictions using different probabilistic machine learning methods.

### Usage

```
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

##### Iris data set
<img src="Figs/probably_iris.png">

##### Graduate school admission data set
<img src="Figs/probably_admission.png">