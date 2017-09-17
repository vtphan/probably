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
from sklearn.externals import joblib

#----------------------------------------------------------------
class Model(object):
	def __init__(self):
		self.info = {}
		self.info['model_test'] = {
			'gauss' : GaussianProcessClassifier(),
			'gboost' : GradientBoostingClassifier(),
			'logit' : LogisticRegression(),
			'rf' : RandomForestClassifier(n_estimators=100),
			'svc' : SVC(probability=True),
		}
		self.info['model_evaluate'] = {
			'gauss' : GaussianProcessClassifier(),
			'gboost' : GradientBoostingClassifier(),
			'logit' : LogisticRegression(),
			'rf' : RandomForestClassifier(n_estimators=100),
			'svc' : SVC(probability=True),
		}
		self.info['model_names'] = sorted(self.info['model_test'].keys())
		self.info['classes'] = []
		self.info['accuracy'] = {}
		self.info['precision'] = {}
		self.info['recall'] = {}
		self.info['specificity'] = {}
		self.info['tp_probs'] = {}
		self.info['fp_probs'] = {}

	#----------------------------------------------------------------
	def get_model(self, name, version):
		if name not in self.info['model_names']:
			raise Exception('Unknonw model name:', name)
		if version not in ['test', 'evaluate']:
			raise Exception('Unknown model version:', version)
		return self.info['model_' + version][name]

	def model_names(self):
		return self.info['model_names']

	def classes(self):
		return self.info['classes']

	#----------------------------------------------------------------
	def define(self, data, features, target, cv=ShuffleSplit(10, test_size=0.1)):
		if type(data) == pandas.DataFrame:
			df = data
		elif type(data) == str:
			df = pandas.read_csv(data)
		else:
			raise Exception('data must be either a string (csv file) or Pandas.DataFrame.')
		self.features = features
		self.target = target
		self.cv = cv
		self.X = df[features]
		self.y = df[target]
		self.info['classes'] = sorted(self.y.unique())
		for name in self.model_names():
			print('Building', name, 'model.')
			model = self.get_model(name, 'test')
			model.fit(self.X, self.y)

		for name in self.model_names():
			count = 0
			acc, pre, rec, spe = 0, 0, 0, 0
			self.info['tp_probs'][name] = { c:[] for c in self.classes() }
			self.info['fp_probs'][name] = { c:[] for c in self.classes() }
			for train, test in self._splits():
				count += 1
				X_train, X_test, y_train, y_test = self._train_test_data(train,test)
				model = self.get_model(name, 'evaluate')
				model.fit(X_train, y_train)
				# y_prob_train = model.predict_proba(X_train)
				# y_pred_train = model.predict(X_train)
				# print(name, model.score(X_test, y_test))
				y_prob_test = model.predict_proba(X_test)
				y_pred_test = model.predict(X_test)
				for i,c in enumerate(model.classes_):
					scores = self.score(y_test, y_pred_test, c)
					acc += scores[0]
					pre += scores[0]
					rec += scores[0]
					spe += scores[0]
					tp_probs, fp_probs = self.tp_fp_probs(y_test, y_pred_test, y_prob_test[:,i], c)
					self.info['tp_probs'][name][c] += tp_probs
					self.info['fp_probs'][name][c] += fp_probs
			self.info['accuracy'][name] = acc/count
			self.info['precision'][name] = pre/count
			self.info['recall'][name] = rec/count
			self.info['specificity'][name] = spe/count

	#----------------------------------------------------------------
	# return the percentage of tp's whose probabilites are <= p
	# SLOW NEEDS TO BE FASTER
	#----------------------------------------------------------------
	def tp_le(self, name, c, probs):
		tp_probs = self.info['tp_probs'][name][c]
		if len(tp_probs) == 0:
			return [ 0 ] * len(probs)
		n = len(tp_probs)
		return [ round(100*len([q for q in tp_probs if q<=p])/n,0) for p in probs ]

	#----------------------------------------------------------------
	# return the percentage of fp's whose probabilites are > p
	# SLOW NEEDS TO BE FASTER
	#----------------------------------------------------------------
	def fp_gt(self, name, c, probs):
		fp_probs = self.info['fp_probs'][name][c]
		if len(fp_probs) == 0:
			return [ 0 ] * len(probs)
		n = len(fp_probs)
		return [ round(100*len([q for q in fp_probs if q>p])/n,0) for p in probs ]

	#----------------------------------------------------------------
	#----------------------------------------------------------------
	def tp_fp_probs(self, y_true, y_pred, y_prob, target):
		P = [ i for i in range(len(y_pred)) if y_pred[i]==target ]
		N = [ i for i in range(len(y_pred)) if y_pred[i]!=target ]
		y_true_val = y_true.values
		TPprob = [ y_prob[i] for i in P if y_pred[i]==y_true_val[i] ]
		FPprob = [ y_prob[i] for i in P if y_pred[i]!=y_true_val[i] ]
		return TPprob, FPprob

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
			scores_test[t] = {'svc':[0,0,0,0], 'logit':[0,0,0,0], 'gauss':[0,0,0,0], 'gboost':[0,0,0,0,0], 'rf':[0,0,0,0,0]}
			scores_train[t] = scores_test[t].copy()
			for train, test in self._splits():
				X_train, X_test, y_train, y_test = self._train_test_data(train,test)
				for name in self.model_names():
					model = self.get_model(name, 'evaluate')
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
	def corr_tables(self, P):
		probs = {
			c : { name : P[name]['prob'][c] for name in self.model_names() }
			for c in self.classes()
		}
		df_prob = { c : pandas.DataFrame(p) for c,p in probs.items() }
		df_prob_corr = { c: df_prob[c].corr().round(2) for c in probs }
		corr_text = [
			Div(
				text="<h3>Model correlation (class {})</h3><pre>{}</pre>".format(
					c, str(df_prob_corr[c])),
				width=420
			) for c in probs
		]
		return corr_text

	#----------------------------------------------------------------
	def sim_table(self, P):
		jaccard = {
			n : [ jaccard_similarity_score(P[n]['pred'], P[m]['pred'])
				for m in self.model_names() ] for n in self.model_names()
		}
		df_jaccard = pandas.DataFrame(jaccard,
			columns=self.model_names(), index=self.model_names()).round(2)
		jaccard_text = [
			Div(text="<h3>Model prediction (Jaccard) similarity</h3><pre>{}</pre>".format(
					str(df_jaccard)), width=420)
		]
		return jaccard_text

	#----------------------------------------------------------------
	def predict(self, X, out='prediction.html'):
		if type(X) != pandas.DataFrame:
			raise Exception('input must be of type pandas.DataFrame.')

		P = {}
		for name in self.model_names():
			model = self.get_model(name, 'test')
			probs = model.predict_proba(X)
			P[name] = dict(
				pred = model.predict(X),
				prob = { c : probs[:,i] for i,c in enumerate(model.classes_) },
				tp = { c : self.tp_le(name,c,probs[:,i]) for i,c in enumerate(model.classes_) },
				fp = { c : self.fp_gt(name,c,probs[:,i]) for i,c in enumerate(model.classes_) },
			)

		#--------------------------------------------------
		figs1 = [ self.plot_fig1(X, P, c) for c in self.classes() ]
		for i in range(len(figs1)):
			if i > 0:
				figs1[i].y_range = figs1[0].y_range
				# figs1[i].yaxis.visible = False
		figs1[-1].toolbar_location = 'right'
		figs1[-1].width = 420

		#--------------------------------------------------
		figs2 = [ self.plot_fig2(X, P, c) for c in self.classes() ]
		# for i in range(len(figs2)):
		# 	if i > 0:
		# 		figs2[i].x_range = figs2[0].x_range
		# 		figs2[i].y_range = figs2[0].y_range
		# 		# figs2[i].yaxis.visible = False
		# figs2[-1].toolbar_location = 'right'
		# figs2[-1].width = 420

		#--------------------------------------------------
		# Plot pairwise correlation and similarity
		#--------------------------------------------------
		corr_tables = self.corr_tables(P)
		sim_table = self.sim_table(P)

		#--------------------------------------------------
		# Layout figures
		#--------------------------------------------------
		layout = column(row(figs1), row(figs2), row(*corr_tables), row(sim_table))
		output_file(out)
		show(layout)

	#----------------------------------------------------------------
	def plot_fig2(self, X, prediction, target):
		COLOR = Category10[10]
		plot_width, plot_height = 420, 420
		tooltips = [
			('Sample', '@sample'),
			('Prediction', 'Class @predicted (@classifier)'),
			('Probability', '@prob{1.11}'),
			('%TP ≤ p', '@tp%'),
			('%FP >  p', '@fp%'),
		]
		tooltips += [ (name, '@{}'.format(name)) for name in X.columns  ]
		hover = HoverTool(
			tooltips = tooltips,
			names = [str(target)],
		)
		fig = figure(
			x_axis_label='% of TP w. prob ≤ p',
			y_axis_label='% of FP w. prob > p',
			# x_range = (0,100),
			# y_range = (0,100),
			tools = ['pan','box_zoom','reset',hover],
			# active_scroll="ywheel_zoom",
			toolbar_location='right',
			logo = None,
			plot_width = plot_width,
			plot_height = plot_height,
		)
		# fig.xaxis.ticker = FixedTicker(ticks=numpy.arange(0,1.1,0.1))
		# fig.yaxis.ticker = FixedTicker(ticks=numpy.arange(0,1.1,0.1))
		fig.xgrid.grid_line_color = None
		fig.ygrid.grid_line_color = None
		i = 0
		plots = []
		for k,v in prediction.items():
			data_source = dict(
				sample = numpy.arange(0, len(X)),
				classifier = [ k for j in range(len(X)) ],
				prob = v['prob'][target],
				tp = v['tp'][target],
				fp = v['fp'][target],
				predicted = v['pred'],
				color = [ COLOR[i] for j in range(len(X)) ],
				alpha = [ p for p in v['prob'][target] ],
			)
			for c in X.columns:
				data_source[c] = X[c]
			source = ColumnDataSource(data=data_source, name=k)
			plot = fig.circle(
				x='tp',
				y='fp',
				color='color',
				alpha='alpha',
				size=10,
				source=source,
				name = str(target),
			)
			plots.append((k,[plot]))
			i += 1
		legend = Legend(
			items=plots,
			location=(0,2),
			orientation='horizontal',
			click_policy='hide',
		)
		fig.add_layout(legend, 'above')
		return fig

	#----------------------------------------------------------------
	def plot_fig1(self, X, prediction, target):
		COLOR = Category10[10]
		plot_width, plot_height, num_points = 410, 410, 10
		tooltips = [
			('Prediction', 'Class @predicted (@classifier)'),
			('Probability', '@prob{1.11}'),
			('%TP ≤ p', '@tp%'),
			('%FP >  p', '@fp%'),
		]
		tooltips += [ (name, '@{}'.format(name)) for name in X.columns  ]
		hover = HoverTool(
			tooltips = tooltips,
			names = [str(target)],
		)
		fig = figure(
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
				tp = v['tp'][target],
				fp = v['fp'][target],
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
		# dline = Span(location=0.5,dimension='height',line_dash='dashed',line_width=2)
		# fig.add_layout(dline)
		legend = Legend(
			items=plots,
			location=(0,2),
			orientation='horizontal',
			click_policy='hide',
		)
		fig.add_layout(legend, 'above')
		return fig

	#----------------------------------------------------------------
	def save(self, output_file='output.pkl'):
		print('Saving model to', output_file)
		joblib.dump(self.info, output_file)

	def load(self, input_file):
		print('Loading model from', input_file)
		self.info = joblib.load(input_file)

	#----------------------------------------------------------------


