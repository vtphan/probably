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
from sklearn.metrics import jaccard_similarity_score
from sklearn.externals import joblib
from bokeh.plotting import figure, output_file, show
from bokeh.models import FuncTickFormatter, FixedTicker
from bokeh.palettes import Category10, Category20
from bokeh.models import ColumnDataSource, HoverTool, Legend, Span, Range1d
from bokeh.layouts import row, column, widgetbox
from bokeh.models.widgets import Div, Slider
from bokeh.models.callbacks import CustomJS
from bokeh.io import curdoc

import time
#----------------------------------------------------------------
LAST_TIME = time.time()
def timer():
	global LAST_TIME
	print('Time: ', round((time.time() - LAST_TIME), 3))
	LAST_TIME = time.time()

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
			self.info['scores'] = {}
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
				else:
					raise Exception('Unknown classifier: ', c)
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
			self.info['tp_probs'][c] = { l:[] for l in self.info['labels'] }
			self.info['fp_probs'][c] = { l:[] for l in self.info['labels'] }
			self.info['scores'][c] = { l:numpy.array([0.0,0.0,0.0,0.0])
										for l in self.info['labels'] }
			count = 0
			print('Training', c)
			for train, test in self._splits():
				count += 1
				X_train, X_test, y_train, y_test = self._train_test_data(train,test)
				cls = self.info['classifier'][c]
				cls.fit(X_train, y_train)
				y_prob_test = cls.predict_proba(X_test)
				y_pred_test = cls.predict(X_test)
				for i,label in enumerate(cls.classes_):
					scores = self.score(y_test, y_pred_test, label)
					self.info['scores'][c][label] += scores
					tp, fp = self.tp_fp_probs(y_test,y_pred_test,y_prob_test[:,i],label)
					self.info['tp_probs'][c][label] += tp
					self.info['fp_probs'][c][label] += fp
			for l in self.info['labels']:
				self.info['scores'][c][l] /= count
				self.info['tp_probs'][c][l] = sorted(self.info['tp_probs'][c][l])
				self.info['fp_probs'][c][l] = sorted(self.info['fp_probs'][c][l])

	#----------------------------------------------------------------
	# return the percentage of tp's whose probabilites are <= p
	#----------------------------------------------------------------
	def tp_le(self, c, lab, probs):
		tp_probs = self.info['tp_probs'][c][lab]
		if len(tp_probs) == 0:
			return [ 0 ] * len(probs)
		n = len(tp_probs)
		return [ round(100*len([q for q in tp_probs if q<=p])/n,0) for p in probs ]
		# perc = []
		# for p in probs:
		# 	count = 0
		# 	for q in tp_probs:
		# 		if q <= p:
		# 			count += 1
		# 		else:
		# 			break
		# 	perc.append( round(100*count/n,0 ) )
		# return perc

	#----------------------------------------------------------------
	# return the percentage of fp's whose probabilites are > p
	#----------------------------------------------------------------
	def fp_gt(self, c, lab, probs):
		fp_probs = self.info['fp_probs'][c][lab]
		if len(fp_probs) == 0:
			return [ 0 ] * len(probs)
		n = len(fp_probs)
		perc = []
		return [ round(100*len([q for q in fp_probs if q>p])/n,0) for p in probs ]
		# for p in probs:
		# 	count = 0
		# 	for q in fp_probs:
		# 		if q <= p:
		# 			count += 1
		# 		else:
		# 			break
		# 	perc.append( round(100*(n-count)/n,0 ) )
		# return perc

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
		return numpy.array([(tp+tn)/(tp+fp+tn+fn), pre, rec, spe])

	#----------------------------------------------------------------
	def corr_tables(self):
		probs = {
			l : { c : self.Data[c]['prob'][l] for c in self.info['classifier_names'] }
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
	def sim_table(self):
		jaccard = {
			n : [ jaccard_similarity_score(self.Data[n]['pred'], self.Data[m]['pred'])
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
	def predict(self, X_test, target=None, out='prediction.html'):
		if type(X_test) != pandas.DataFrame:
			raise Exception('input must be of type pandas.DataFrame.')

		self.Data = {}
		self.Votes = {}
		self.X_test = X_test
		for c in self.info['classifier_names']:
			cls = self.info['classifier'][c]
			probs = cls.predict_proba(X_test)
			pred = cls.predict(X_test)
			self.Data[c] = dict(
				pred = pred,
				prob = { lab : probs[:,i] for i,lab in enumerate(cls.classes_) },
				tp = { lab : self.tp_le(c,lab,probs[:,i]) for i,lab in enumerate(cls.classes_) },
				fp = { lab : self.fp_gt(c,lab,probs[:,i]) for i,lab in enumerate(cls.classes_) },
			)

		#--------------------------------------------------
		# Count votes
		#--------------------------------------------------
		for l in self.info['labels']:
			self.Votes[l] = [0] * len(self.X_test)
			for c in self.info['classifier_names']:
				for i,vote in enumerate(self.Data[c]['pred']):
					if vote == l:
						self.Votes[l][i] += 1

		#--------------------------------------------------
		self.init_graphics()
		self.data_source = self.make_data_source()

		votes_filter = self.votes_slider()
		votes_filter.on_change('value', self.votes_filter_callback)

		#--------------------------------------------------
		figs1 = [ self.plot_fig1(l,d,target) for (l, d) in self.data_source.items() ]
		for i in range(1, len(figs1)):
			figs1[i].y_range = figs1[0].y_range

		#--------------------------------------------------
		# Plot %tp ≤ p versus %fp > p
		#--------------------------------------------------
		figs2 = [ self.plot_fig2(l,d) for (l, d) in self.data_source.items() ]

		#--------------------------------------------------
		# Plot pairwise correlation and similarity
		#--------------------------------------------------
		corr_tables = self.corr_tables()
		sim_table = self.sim_table()

		#--------------------------------------------------
		# Layout figures
		#--------------------------------------------------
		layout = column(
			row(votes_filter),
			row(figs1),
			row(figs2),
			row(*corr_tables),
			row(sim_table)
		)
		# output_file(out)
		# show(layout)
		curdoc().add_root(layout)
		curdoc().title = "Probably"

	#----------------------------------------------------------------
	def votes_filter_callback(self, attrname, old, threshold):
		def filter(seq):
			return [ seq[i] for i in range(len(seq)) if votes[i]>=threshold ]

		N = len(self.X_test)
		for label in self.data_source:
			votes = self.Votes[label]
			for cls in self.data_source[label]:
				# data = self.data_source[label][cls].data
				data = {}
				data['sample'] = [i for i in range(N) if votes[i]>=threshold]
				data['classifier'] = [cls for i in range(N) if votes[i]>=threshold]
				data['color'] = [ self.get_classifier_color(cls) for i in range(N) if votes[i]>=threshold ]
				data['alpha'] = [ 0.8 if self.Data[cls]['pred'][i]==label else 0.15 for i in range(N) if votes[i]>=threshold ]
				data['size'] = [int(13*self.Data[cls]['prob'][label][i]) for i in range(N) if votes[i]>=threshold ]
				data['prob'] = filter(self.Data[cls]['prob'][label])
				data['tp'] = filter(self.Data[cls]['tp'][label])
				data['fp'] = filter(self.Data[cls]['fp'][label])
				data['predicted'] = filter(self.Data[cls]['pred'])
				for c in self.X_test.columns:
					data[c] = filter(list(self.X_test[c]))

				self.data_source[label][cls].data.update(data)

	#----------------------------------------------------------------
	def votes_slider(self):
		slider = Slider(
						start=0,
						end=len(self.info['classifier_names']),
						value=0,
						step=1,
						title="Consensus",
						width = 200,
		)
		return slider

	#----------------------------------------------------------------
	def make_data_source(self):
		N = len(self.X_test)
		data_source = { l : {} for l in self.info['labels'] }
		for label in self.info['labels']:
			for cls in self.info['classifier_names']:
				data = dict(
					sample = numpy.arange(0, N),
					classifier = [ cls for j in range(N) ],
					prob = self.Data[cls]['prob'][label],
					tp = self.Data[cls]['tp'][label],
					fp = self.Data[cls]['fp'][label],
					predicted = self.Data[cls]['pred'],
					color = [ self.get_classifier_color(cls) for j in range(N) ],
					alpha = [ 0.8 if self.Data[cls]['pred'][j]==label else 0.15 for j in range(N) ],
					size = [ int(p*13) for p in self.Data[cls]['prob'][label] ],
				)
				for c in self.X_test.columns:
					data[c] = self.X_test[c]
				data_source[label][cls] = ColumnDataSource(data=data)
		return data_source

	#----------------------------------------------------------------
	def plot_fig2(self, label, data_source):
		plot_width, plot_height = 380, 380
		tooltips = [
			('Sample', '@sample'),
			('Prediction', 'Class @predicted (@classifier)'),
			('Probability', '@prob{1.11}'),
			('%TP ≤ p', '@tp%'),
			('%FP >  p', '@fp%'),
		]
		tooltips += [ (name, '@{}'.format(name)) for name in self.X_test.columns  ]
		hover = HoverTool(
			tooltips = tooltips,
			names = [str(label)],
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
		# fig.xgrid.grid_line_color = None
		# fig.ygrid.grid_line_color = None
		plots = []
		for cls in self.Data:
			plot = fig.circle(
				x='tp',
				y='fp',
				color='color',
				alpha='alpha',
				size='size',
				source=data_source[cls],
				name = str(label),
			)
			plots.append((cls,[plot]))

		legend = Legend(
			items=plots,
			location=(0,2),
			orientation='horizontal',
			click_policy='hide',
		)
		fig.add_layout(legend, 'above')
		return fig

	#----------------------------------------------------------------
	def plot_fig1(self, label, data_source, target):
		plot_width, plot_height, num_points = 380, 380, 10
		tooltips = [
			('Prediction', 'Class @predicted (@classifier)'),
			('Probability', '@prob{1.11}'),
			('%TP ≤ p', '@tp%'),
			('%FP >  p', '@fp%'),
		]
		tooltips += [ (name, '@{}'.format(name)) for name in self.X_test.columns  ]
		hover = HoverTool(
			tooltips = tooltips,
			names = [str(label)],
		)
		fig = figure(
			x_axis_label='Probability of class {}'.format(label),
			y_axis_label='Samples',
			x_range = (0,1),
			tools = ['ypan','ywheel_zoom','reset',hover],
			active_scroll="ywheel_zoom",
			toolbar_location='right',
			logo = None,
			plot_width = plot_width,
			plot_height = plot_height,
		)
		N = len(self.X_test)
		if N > num_points:
			fig.y_range = Range1d(N/2 - (num_points*0.5+0.5), N/2 + (num_points*0.5+0.5))
		fig.xaxis.ticker = FixedTicker(ticks=numpy.arange(0,1.1,0.1))
		fig.yaxis.ticker = FixedTicker(ticks=numpy.arange(0,N))
		fig.xgrid.grid_line_color = None
		fig.ygrid.grid_line_color = None
		plots = []
		for cls in self.Data:
			plot = fig.circle(
				x='prob',
				y='sample',
				color='color',
				alpha='alpha',
				size=10,
				source = data_source[cls],
				name = str(label),
			)
			plots.append((cls,[plot]))

		for i in range(N):
			if target is not None and label==target.iloc[i]:
				# positive sample
				sample_line = Span(location=i,dimension='width',
					line_color='DarkSlateGrey',line_dash='dashed',line_width=1)
			else:
				# negative sample
				sample_line = Span(location=i,dimension='width',
					line_color='LightSlateGrey',line_dash='dotted',line_width=1)

			fig.add_layout(sample_line)
		legend = Legend(
			items=plots,
			location=(0,2),
			orientation='horizontal',
			click_policy='hide',
		)
		fig.add_layout(legend, 'above')
		return fig

	#----------------------------------------------------------------
	def init_graphics(self):
		COLOR = Category10[10]
		self.classifier_color = {}
		for i, c in enumerate(self.info['classifier_names']):
			self.classifier_color[c] = COLOR[i]
	#----------------------------------------------------------------
	def get_classifier_color(self, c):
		return self.classifier_color[c]

	#----------------------------------------------------------------
	def available_classifiers(self):
		return [ 'rf', 'logit', 'gboost', 'svc', 'gauss' ]

	#----------------------------------------------------------------
	def save(self, output_file):
		print('Saving model to', output_file)
		joblib.dump(self.info, output_file)


	#----------------------------------------------------------------


