### Machine Learning for Public Policy
### Pipeline: Build classifier
### Héctor Salvador López

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
import numpy as np

# Classifiers to test
classifiers = {'LR': LogisticRegression(),
				'KNN': KNeighborsClassifier(),
				'DT': DecisionTreeClassifier(),
				'SVM': LinearSVC(),
				'RF': RandomForestClassifier(),
				'BOO': GradientBoostingClassifier(),
				'BAG': BaggingClassifier()}

grid = {'LR': {'penalty': ['l1', 'l2'], 'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]}, 
        'KNN': {'n_neighbors': [1, 5, 10, 25, 50, 100], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree']},
        'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1, 5, 10, 20, 50, 100], 'max_features': ['sqrt', 'log2'], 'min_samples_split': [2, 5, 10]},
        'SVM' : {'C' : [0.1, 1]},
        'RF': {'n_estimators': [1, 10], 'max_depth': [1, 5, 10], 'max_features': ['sqrt', 'log2'], 'min_samples_split': [2, 5, 10]},
        'GB': {'n_estimators': [1, 10], 'learning_rate' : [0.1, 0.5], 'subsample' : [0.5, 1.0], 'max_depth': [1, 3, 5]},
        }

def classify(X, y, models, iters, threshold, decision_metric = 'auc'):
	'''
	Takes:
		X, a dataframe of features 
		y, a dataframe of the label
		models, a list of strings indicating models to run (e.g. ['LR', 'DT'])

	Returns:
		A new dataframe comparing each classifier's performace on the given
		evaluation metrics.
	'''
	rv = pd.DataFrame()
	
	# for every classifier, try any possible combination of parameters on grid
	for index, clf in enumerate([classifiers[x] for x in models]):
		name = models[index]
		print(name)
		parameter_values = grid[name]
		top_intra_model_dec_metric = 0 	# so that we can get a best set of parameters

		for p in ParameterGrid(parameter_values):
			dec_metric_per_iter = []

			for i in range(iters):
					# Construct train and test splits
				xtrain, xtest, ytrain, ytest = \
						train_test_split(X, y, test_size=0.2, random_state=0)
				try:
					# run the model with the combinations of the above parameters
					clf.set_params(**p)
					print(clf)
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
					dec_metric_per_iter.append(metrics[decision_metric])
 				
				except IndexError:
					print('Error')
					continue

			avg_dec_metric = np.mean(dec_metric_per_iter)
			if avg_dec_metric > top_intra_model_dec_metric:
				top_intra_model_dec_metric = avg_dec_metric
				best_models[name] = p
	return rv


def evaluate_classifier(ytest, yhat):
	'''
	For an index of a given classifier, evaluate it by various metrics
	'''
	# Metrics to evaluate
	metrics = {'precision': precision_score(ytest, yhat),
				'recall': recall_score(ytest, yhat),
				'f1': f1_score(ytest, yhat),
				'auc': roc_auc_score(ytest, yhat)}

	for name, m in metrics:
		print('{}: {}'.format(name, m))
	
	return metrics
'''		
		# Iterate through folds
		# for train_index, test_index in kf:
		# 	X_train, X_test = X[train_index], X[test_index]
		# 	y_train = y[train_index]

		# 	#Initialize classifier
		# 	model = clf
		# 	model.fit(X_train, y_train)
		# 	y_pred[test_index] = model.predict(X_test)
		
		# end_time = time.time() 
		# rv.loc[index, 'time'] = end_time - start_time

		# evaluate_classifier(rv, index, y, y_pred)
		# index += 1

	return rv
'''
