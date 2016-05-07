### Machine Learning for Public Policy
### Pipeline: Build, select, and evaluate classifiers
### Héctor Salvador López

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
import time

plt.style.use('ggplot')

# Classifiers to test
classifiers = {'LR': LogisticRegression(),
				'KNN': KNeighborsClassifier(),
				'DT': DecisionTreeClassifier(),
				'SVM': LinearSVC(),
				'RF': RandomForestClassifier(),
				'GB': GradientBoostingClassifier()}

grid = {#'LR': {'penalty': ['l1', 'l2'], 'C': [0.1, 1]}, 
		'LR': {'penalty': ['l1', 'l2'], 'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]}, 
        'KNN': {'n_neighbors': [1, 5, 10, 25, 50, 100], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree']},
        #'KNN': {'n_neighbors': [5, 10], 'weights': ['uniform', 'distance'], 'algorithm': ['auto']},
        'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1, 5, 10, 20, 50, 100], 'max_features': ['sqrt', 'log2'], 'min_samples_split': [2, 5, 10]},
        'SVM' : {'C' : [0.1, 1]},
        'RF': {'n_estimators': [1, 10], 'max_depth': [1, 5, 10], 'max_features': ['sqrt', 'log2'], 'min_samples_split': [2, 5, 10]},
        'GB': {'n_estimators': [1, 10], 'learning_rate' : [0.1, 0.5], 'subsample' : [0.5, 1.0], 'max_depth': [1, 3, 5]},
        }

def classify(X, y, models, iters, threshold, decision_metric = 'auc', verbose =False):
	'''
	Takes:
		X, a dataframe of features 
		y, a dataframe of the label
		models, a list of strings indicating models to run (e.g. ['LR', 'DT'])

	Returns:
		A new dataframe comparing each classifier's performace on the given
		evaluation metrics.
	'''
	columns = ['time', 'avg_{}'.format(decision_metric), 'params']
	rv = pd.DataFrame(index = models, columns = columns)
	
	best_metric = 0
	best_model = ''
	best_params = ''
	best_models = {}
	mtr = ['precision', 'recall', 'f1', 'auc']

	# Construct train and test splits
	xtrain, xtest, ytrain, ytest = \
						train_test_split(X, y, test_size=0.2, random_state=0)
	
	# for every classifier, try any possible combination of parameters on grid
	for index, clf in enumerate([classifiers[x] for x in models]):
		name = models[index]
		print(name)
		parameter_values = grid[name]
		top_intra_metric = 0 	# so that we can get a best set of parameters
		

		for p in ParameterGrid(parameter_values):
			precision_per_iter = []
			recall_per_iter = []
			f1_per_iter = []
			auc_per_iter = []
			time_per_iter = []
			avg_metrics = {}
			clf.set_params(**p)
			if verbose:
				print(clf)

			for i in range(iters):
				try:
					# run the model with the combinations of the above parameters
					if verbose:
						print('Iteration {}.'.format(i))
					start_time = time.time() # to calculate running time
					
					# get the predicted results from the model
					if hasattr(clf, 'predict_proba'):
						yscores = clf.fit(xtrain, ytrain).predict_proba(xtest)[:,1]
					else:
						yscores = clf.fit(xtrain, ytrain).decision_function(xtest)

					yhat = np.asarray([1 if i >= threshold else 0 for i in yscores])
					end_time = time.time()

					# obtain metrics
					metrics = evaluate_classifier(ytest, yhat)
					for met, value in metrics.items():
						eval('{}_per_iter'.format(met)).append(value)
						if verbose:
							print('  Adding {}: {}.'.format(met, value))
					time_per_iter.append(end_time - start_time)
					if verbose:
						print('  Adding time: {}.'.format(end_time - start_time))

				except IndexError:
					print('Error')
					continue
			
			avg_metrics['time'] = np.mean(time_per_iter)

			# print average metrics of model p
			if verbose:
				print('Got average time: {}.'.format(avg_metrics['time']))
			for met in mtr:
				avg_metrics[met] = np.mean(eval('{}_per_iter'.format(met)))
				if verbose:
					print('Got average {}: {}.'.format(met, avg_metrics[met]))

			# compare if metrics of model p are better than best model so far
			if verbose:
				print('\nIs {} greater than {}?'.format(avg_metrics[decision_metric],\
					top_intra_metric))
			if avg_metrics[decision_metric] > top_intra_metric:
				top_intra_metric = avg_metrics[decision_metric]
				if verbose:
					print('	New best model.')
				top_avg_metrics = avg_metrics
				best_models[name] = clf

			if verbose:
				print('Finished: {} model.\n'.format(p))

		print('Finished running {}'.format(name))
		print('Best model was: {}, with {} = {}.\n'.format(best_models[name],\
			decision_metric, top_intra_metric))

		to_append = [top_avg_metrics['time'], top_avg_metrics[decision_metric],\
		 str(p)]
		if verbose:
			print('Appending: {}'.format(to_append))
		rv.loc[name] = to_append
		
		if verbose:
			print('Is {} greater that {}?'.format(top_intra_metric, best_metric))
		if top_intra_metric > best_metric:
			if verbose:
				print('{} with parameters {} is the new best model.\n'.format(name, p))
			best_metric = top_intra_metric
			best_model = name
			best_params = clf

	return rv, best_models

def gen_precision_recall_plots(X, y, best_models):
	'''
	'''
	xtrain, xtest, ytrain, ytest = \
						train_test_split(X, y, test_size=0.2, random_state=0)

	for name, clf in best_models.items():
		y_true = ytest
		y_prob = clf.fit(xtrain, ytrain).predict_proba(xtest)[:,1]
		plot_precision_recall_n(y_true, y_prob, name)

def evaluate_classifier(ytest, yhat):
	'''
	For an index of a given classifier, evaluate it by various metrics
	'''
	# Metrics to evaluate
	metrics = {'precision': precision_score(ytest, yhat),
				'recall': recall_score(ytest, yhat),
				'f1': f1_score(ytest, yhat),
				'auc': roc_auc_score(ytest, yhat)}
	
	return metrics

def plot_precision_recall_n(y_true, y_prob, model_name):
    '''
    '''
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    
    name = model_name
    plt.title(name)
    #plt.savefig(name)
    plt.show()
