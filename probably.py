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
	def __init__(self, data=None, classifiers=None, features=None, target=None, cv=None, saved_model=None):
		if saved_model is not None:
			print('Loading model from', saved_model)
			self.info = joblib.load(saved_model)
		else:
			self.info = {}
			if classifiers is None:
				classifiers = ['rf', 'logit']
			self._select_classifiers(classifiers)
			if cv is None:
				cv = ShuffleSplit(10, test_size=0.1)
			self.info['cv'] = cv
			self.info['features'] = features
			self.info['target'] = target
			self.info['accuracy'] = {}
			self.info['precision'] = {}
			self.info['recall'] = {}
			self.info['specificity'] = {}
			self.info['tp_probs'] = {}
			self.info['fp_probs'] = {}
			if type(data) == pandas.DataFrame:
				df = data
			elif type(data) == str:
				df = pandas.read_csv(data)
			else:
				raise Exception('data must be either a string (csv file) or Pandas.DataFrame.')
			self.X = df[features]
			self.y = df[target]
			self.info['labels'] = sorted(self.y.unique())
			self.eval_all()
			self.fit_all()

	#----------------------------------------------------------------
	def _select_classifiers(self, classifiers):
		self.info['classifier'] = {}
		for c in classifiers:
			if type(c) == str:
				if c not in ['gauss', 'gboost', 'logit', 'rf', 'svf']:
					raise Exception('Unknown classifier: ', c)
				if c == 'gauss':
					self.info['classifier']['gauss'] = GaussianProcessClassifier()
				elif c == 'gboost':
					self.info['classifier']['gboost'] = GradientBoostingClassifier()
				elif c == 'logit':
					self.info['classifier']['logit'] = LogisticRegression()
				elif c == 'rf':
					self.info['classifier']['rf'] = RandomForestClassifier(n_estimators=100)
				elif c == 'svc':
					self.info['classifier']['svc'] = SVC(probability=True)
			elif type(c)==tuple and len(c)==2:
				self.info['classifier'][c[0]] = c[1]
		self.info['classifier_names'] = sorted(self.info['classifier'].keys())

	#----------------------------------------------------------------
	def fit_all(self):
		for c in self.info['classifier_names']:
			print('Building', c, 'model.')
			model = self.info['classifier'][c]
			model.fit(self.X, self.y)

	#----------------------------------------------------------------
	def eval_all(self):
		for c in self.info['classifier_names']:
			count = 0
			acc, pre, rec, spe = 0, 0, 0, 0
			self.info['tp_probs'][c] = { l:[] for l in self.info['labels'] }
			self.info['fp_probs'][c] = { l:[] for l in self.info['labels'] }
			for train, test in self._splits():
				count += 1
				X_train, X_test, y_train, y_test = self._train_test_data(train,test)
				cls = self.info['classifier'][c]
				cls.fit(X_train, y_train)
				y_prob_test = cls.predict_proba(X_test)
				y_pred_test = cls.predict(X_test)
				for i,label in enumerate(cls.classes_):
					scores = self.score(y_test, y_pred_test, label)
					acc += scores[0]
					pre += scores[0]
					rec += scores[0]
					spe += scores[0]
					tp_probs, fp_probs = self.tp_fp_probs(y_test,y_pred_test,y_prob_test[:,i],label)
					self.info['tp_probs'][c][label] += tp_probs
					self.info['fp_probs'][c][label] += fp_probs
			self.info['accuracy'][c] = acc/count
			self.info['precision'][c] = pre/count
			self.info['recall'][c] = rec/count
			self.info['specificity'][c] = spe/count

	#----------------------------------------------------------------
	# return the percentage of tp's whose probabilites are <= p
	# SLOW NEEDS TO BE FASTER
	#----------------------------------------------------------------
	def tp_le(self, c, lab, probs):
		tp_probs = self.info['tp_probs'][c][lab]
		if len(tp_probs) == 0:
			return [ 0 ] * len(probs)
		n = len(tp_probs)
		return [ round(100*len([q for q in tp_probs if q<=p])/n,0) for p in probs ]

	#----------------------------------------------------------------
	# return the percentage of fp's whose probabilites are > p
	# SLOW NEEDS TO BE FASTER
	#----------------------------------------------------------------
	def fp_gt(self, c, lab, probs):
		fp_probs = self.info['fp_probs'][c][lab]
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
		if type(self.info['cv']) == StratifiedKFold:
			return self.info['cv'].split(self.X, self.y)
		else:
			return self.info['cv'].split(self.X)

	#----------------------------------------------------------------
	# return X_train, X_test, y_train, y_test
	#----------------------------------------------------------------
	def _train_test_data(self, train, test):
		return self.X.iloc[train], self.X.iloc[test], self.y.iloc[train], self.y.iloc[test]

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
			l : { c : P[c]['prob'][l] for c in self.info['classifier_names'] }
			for l in self.info['labels']
		}
		df_prob = { l : pandas.DataFrame(p) for l,p in probs.items() }
		df_prob_corr = { l: df_prob[l].corr().round(2) for l in probs }
		corr_text = [
			Div(
				text="<h3>Model correlation (class {})</h3><pre>{}</pre>".format(
					l, str(df_prob_corr[l])),
				width=420
			) for l in probs
		]
		return corr_text

	#----------------------------------------------------------------
	def sim_table(self, P):
		jaccard = {
			n : [ jaccard_similarity_score(P[n]['pred'], P[m]['pred'])
				for m in self.info['classifier_names'] ]
				for n in self.info['classifier_names']
		}
		df_jaccard = pandas.DataFrame(jaccard,
			columns=self.info['classifier_names'],
			index = self.info['classifier_names']
		).round(2)
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
		for c in self.info['classifier_names']:
			cls = self.info['classifier'][c]
			probs = cls.predict_proba(X)
			P[c] = dict(
				pred = cls.predict(X),
				prob = { lab : probs[:,i] for i,lab in enumerate(cls.classes_) },
				tp = { lab : self.tp_le(c,lab,probs[:,i]) for i,lab in enumerate(cls.classes_) },
				fp = { lab : self.fp_gt(c,lab,probs[:,i]) for i,lab in enumerate(cls.classes_) },
			)

		#--------------------------------------------------
		figs1 = [ self.plot_fig1(X, P, l) for l in self.info['labels'] ]
		for i in range(len(figs1)):
			if i > 0:
				figs1[i].y_range = figs1[0].y_range
		figs1[-1].toolbar_location = 'right'
		figs1[-1].width = 420

		#--------------------------------------------------
		figs2 = [ self.plot_fig2(X, P, l) for l in self.info['labels'] ]

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
			tools = ['pan','box_zoom','reset',hover],
			toolbar_location='right',
			logo = None,
			plot_width = plot_width,
			plot_height = plot_height,
		)
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
	def save(self, output_file):
		print('Saving model to', output_file)
		joblib.dump(self.info, output_file)


	#----------------------------------------------------------------


