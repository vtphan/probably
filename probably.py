import pandas
import numpy
from scipy.spatial.distance import cosine
from sklearn.model_selection import StratifiedKFold, KFold, ShuffleSplit
# from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from bokeh.plotting import figure, output_file, show
from bokeh.models import FuncTickFormatter, FixedTicker
from bokeh.palettes import Category10, Category20
from bokeh.models import ColumnDataSource, HoverTool, Legend, Span, Range1d
from bokeh.layouts import row, column, widgetbox
from bokeh.models.widgets import Div, PreText, DataTable, TableColumn
from sklearn.metrics import jaccard_similarity_score

#----------------------------------------------------------------
class Model(object):
	def __init__(self, input_file, title=None):
		self.df = pandas.read_csv(input_file)

		# self.models[0] built on whole dataset to predict new data.
		# self.models[1] built on partial dataset to evaluate.
		self.models = {
			'svc' : [SVC(probability=True),SVC(probability=True)],
			'logit' : [LogisticRegression(),LogisticRegression()],
			'gauss' : [GaussianProcessClassifier(),GaussianProcessClassifier()],
			'gboost' : [GradientBoostingClassifier(), GradientBoostingClassifier()],
			'rf' : [RandomForestClassifier(n_estimators=50), RandomForestClassifier(n_estimators=50)],
			# 'dt' : [DecisionTreeClassifier(), DecisionTreeClassifier()],
		}
		self.title = title

	#----------------------------------------------------------------
	def define(self, features, target, cv=ShuffleSplit(10, test_size=0.1)):
		self.features = features
		self.target = target
		self.cv = cv
		self.X = self.df[features]
		self.y = self.df[target]
		for name, model in self.models.items():
			model[0].fit(self.X, self.y)

	#----------------------------------------------------------------
	def _splits(self):
		if type(self.cv) == StratifiedKFold:
			return self.cv.split(self.X, self.y)
		else:
			return self.cv.split(self.X)

	#----------------------------------------------------------------
	# return X_train, X_test, y_train, y_test
	def _train_test_data(self, train, test):
		return self.X.iloc[train], self.X.iloc[test], self.y.iloc[train], self.y.iloc[test]

	#----------------------------------------------------------------
	def evaluate(self, target):
		thresholds = numpy.arange(0.4,0.7,0.05)
		scores_test = {}
		scores_train = {}
		for t in thresholds:
			print('Threshold = {}'.format(t))
			scores_test[t] = {'svc':[0,0,0,0], 'logit':[0,0,0,0], 'gaussian':[0,0,0,0]}
			scores_train[t] = {'svc':[0,0,0,0], 'logit':[0,0,0,0], 'gaussian':[0,0,0,0]}
			for train, test in self._splits():
				X_train, X_test, y_train, y_test = self._train_test_data(train,test)
				# Train with model[1]
				for name, (_, model) in self.models.items():
					model.fit(X_train, y_train)

					y_prob_train = model.predict_proba(X_train)
					y_pred_train = self._predict_with_prob(y_prob_train, model.classes_, target, t)
					score = self.score(y_train, y_pred_train, target)
					for i in range(4):
						scores_train[t][name][i] += score[i]

					y_prob_test = model.predict_proba(X_test)
					y_pred_test = self._predict_with_prob(y_prob_test, model.classes_, target, t)
					score = self.score(y_test, y_pred_test, target)
					for i in range(4):
						scores_test[t][name][i] += score[i]

					# print(y_test)
					# print(y_pred_test)
					# classification_report(y_test,y_pred_test)

			for name in scores_train[t]:
				for i in range(4):
					scores_train[t][name][i] = scores_train[t][name][i] / self.cv.n_splits
			for name in scores_test[t]:
				for i in range(4):
					scores_test[t][name][i] = scores_test[t][name][i] / self.cv.n_splits
		for t in scores_test:
			for name in scores_test[t]:
				print('{} {} {}'.format(t,name,[ round(v,2) for v in scores_test[t][name]]))

	#----------------------------------------------------------------
	# Need to fix this!!!!  have to start from model's predictions
	# then filter by threshold.
	def _predict_with_prob(self, y_prob, labels, target, threshold):
		target_idx = list(labels).index(target)
		for lab in labels:
			# THIS DOESN'T WORK FOR MULTICLASS PREDICTION
			if lab != target:
				nlab = lab
		pred = [ target if p[target_idx]>=threshold else nlab for p in y_prob ]
		return pred

	#----------------------------------------------------------------
	# return accuracy, precision, recall (sensitivity), specificity
	#----------------------------------------------------------------
	def score(self, y_true, y_pred, target):
		P = [ i for i in range(len(y_pred)) if y_pred[i]==target ]
		N = [ i for i in range(len(y_pred)) if y_pred[i]!=target ]
		y_true_val = y_true.values
		TP = [ i for i in P if y_pred[i]==y_true_val[i] ]
		FP = [ i for i in P if y_pred[i]!=y_true_val[i] ]
		TN = [ i for i in N if y_true_val[i]!=target ]
		FN = [ i for i in N if y_true_val[i]==target ]
		tp, fp, tn, fn = len(TP), len(FP), len(TN), len(FN)
		pre = tp/(tp+fp) if tp+fp > 0 else 0
		rec = tp/(tp+fn) if tp+fn > 0 else 0
		spe = tn/(tn+fp) if tp+fp > 0 else 0
		return (tp+tn)/(tp+fp+tn+fn), pre, rec, spe

	#----------------------------------------------------------------
	def predict(self, X):
		P = {}
		output_file('prediction.html')
		# predict with model[0]
		for name, (model,_) in self.models.items():
			probs = model.predict_proba(X)
			P[name] = dict(
				prob = { t:probs[:,i] for i,t in enumerate(model.classes_) },
				pred = model.predict(X),
			)

		figs = []
		for i, target in enumerate(model.classes_):
			fig = self.plot(X, P, target)
			figs.append(fig)

		for i in range(len(figs)):
			if i > 0:
				figs[i].y_range = figs[0].y_range
				figs[i].yaxis.visible = False
		figs[-1].toolbar_location = 'right'
		figs[-1].width = 430

		# Plot pairwise correlation and similarity
		classes = {}
		for name in self.models:
			for c in model.classes_:
				classes[c] = { name : P[name]['prob'][c] for name in P }
			break
		df_prob = { c : pandas.DataFrame(p) for c,p in classes.items() }
		df_prob_corr = { c: df_prob[c].corr().round(2) for c in classes }
		corr_text = [
			Div(
				text="<h3>Pairwise correlation class {}</h3><pre>{}</pre>".format(
					c, str(df_prob_corr[c])),
				width=400
			) for c in classes
		]

		jaccard = {}
		for n1 in self.models:
			jaccard[n1] = []
			for n2 in self.models:
				jaccard[n1].append(jaccard_similarity_score(P[n1]['pred'], P[n2]['pred']))

		df_jaccard = pandas.DataFrame(jaccard,
			columns=self.models.keys(), index=self.models.keys()).round(2)
		jaccard_text = [
			Div(text="<h3>Jaccard prediction similarity</h3><pre>{}</pre>".format(
					str(df_jaccard)), width=400)
		]

		# Layout figures
		layout = column(row(figs), row(*corr_text), row(jaccard_text))
		show(layout)

	#----------------------------------------------------------------
	def plot(self, X, prediction, target):
		COLOR = Category10[10]
		plot_width, plot_height, num_points = 400, 400, 10
		tooltips = [
			('Prediction', 'Class @predicted (@classifier)'),
			('Probability, Entropy', '@prob, @entropy'),
		]
		tooltips += [ (name, '@{}'.format(name)) for name in X.columns  ]
		hover = HoverTool(
			tooltips = tooltips,
			names = [str(target)],
		)
		fig = figure(
			# title=self.title,
			x_axis_label='Probability of class {}'.format(target),
			y_axis_label='Samples',
			x_range = (0,1),
			tools = ['ypan','ywheel_zoom','reset',hover],
			active_scroll="ywheel_zoom",
			toolbar_location=None,
			logo = None,
			plot_width = plot_width,
			plot_height = plot_height,
		)
		if len(X) > num_points:
			fig.y_range = Range1d(len(X)/2 - (num_points*0.5+0.5), len(X)/2 + (num_points*0.5+0.5))
		fig.xaxis.ticker = FixedTicker(ticks=numpy.arange(0,1.1,0.1))
		fig.yaxis.ticker = FixedTicker(ticks=numpy.arange(0,len(X)))
		fig.xgrid.grid_line_color = None
		fig.ygrid.grid_line_color = None
		i = 0
		plots = []
		for k,v in prediction.items():
			data_source = dict(
				sample = numpy.arange(0, len(X)),
				classifier = [ k for j in range(len(X)) ],
				prob = v['prob'][target],
				predicted = v['pred'],
				color = [ COLOR[i] for j in range(len(X)) ],
				alpha = [ 1 if v['pred'][j]==target else 0.3 for j in range(len(X)) ],
			)
			data_source['entropy'] = [
				0 if (p<0.0001 or p>0.9999) else -(p*numpy.log2(p) + (1-p)*numpy.log2(1-p)) for p in data_source['prob']
			]
			for c in X.columns:
				data_source[c] = X[c]
			source = ColumnDataSource(data=data_source, name=k)
			plot = fig.circle(
				x='prob',
				y='sample',
				color='color',
				alpha='alpha',
				muted_color='color',
				muted_alpha=0.1,
				size=10,
				source=source,
				# legend=k,
				name = str(target),
			)
			plots.append((k,[plot]))
			i += 1

		for i in range(len(X)):
			sample_line = Span(location=i,
				dimension='width',
				line_color='grey',
				line_dash='dashed',
				line_width=1)
			fig.add_layout(sample_line)
		dline = Span(location=0.5,dimension='height',line_dash='dashed',line_width=2)
		fig.add_layout(dline)
		legend = Legend(
			items=plots,
			location=(0,2),
			orientation='horizontal',
			click_policy='mute',
		)
		fig.add_layout(legend, 'above')
		return fig

#----------------------------------------------------------------
if __name__ == '__main__':
	model = Model('data/admission.csv')
	model.define(
		features = ['gre', 'gpa', 'rank'],
		target = 'admit',
		cv = KFold(10,True),
	)
	data = pandas.read_csv('data/admission.csv')
	test_data = data.sample(50)[['gre', 'gpa', 'rank']]
	# model.evaluate(0)
	model.predict(test_data)

	model = Model('data/iris.csv')
	model.define(
		features = ['SepalWidth','SepalLength','PetalWidth','PetalLength'],
		target = 'Species',
		cv = KFold(10,True),
	)
	data = pandas.read_csv('data/iris.csv')
	test_data = data.sample(15)[['SepalWidth','SepalLength','PetalWidth','PetalLength']]
	# model.evaluate('versicolor')
	model.predict(test_data)

