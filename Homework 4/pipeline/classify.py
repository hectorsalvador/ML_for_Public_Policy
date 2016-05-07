### Machine Learning for Public Policy
### Pipeline: Build classifier
### Héctor Salvador López

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
import time

def classify(X, y):
	'''
	Takes:
		X, a dataframe of features 
		y, a dataframe of the label

	Returns:
		A new dataframe comparing each classifier's performace on the given
		evaluation metrics.
	'''
	rv = pd.DataFrame()

	# Classifiers to test
	classifiers = {'LogR': LogisticRegression(),
					('KNN': KNeighborsClassifier(),
					('DT': DecisionTreeClassifier(),
					('SVM': LinearSVC(),
					('RF': RandomForestClassifier(),
					('BOO': GradientBoostingClassifier(),
					('BAG': BaggingClassifier()}

	# Construct train and test splits
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
	
	for name, clf in enumerate([classifiers[x] for :
		print(name)
		rv.loc[index,'classifier'] = name
		start_time = time.time() # to calculate running time
		
		# Iterate through folds
		for train_index, test_index in kf:
			X_train, X_test = X[train_index], X[test_index]
			y_train = y[train_index]

			#Initialize classifier
			model = clf
			model.fit(X_train, y_train)
			y_pred[test_index] = model.predict(X_test)
		
		end_time = time.time() 
		rv.loc[index, 'time'] = end_time - start_time

		evaluate_classifier(rv, index, y, y_pred)
		index += 1

	return rv

def evaluate_classifier(df, index, y_real, y_predict):
	'''
	For an index of a given classifier, evaluate it by various metrics
	'''
	# Metrics to evaluate
	metrics = [('accuracy', accuracy_score(y_real, y_predict)),
				('precision', precision_score(y_real, y_predict)),
				('recall', recall_score(y_real, y_predict)),
				('f1', f1_score(y_real, y_predict)),
				('area_under_curve', roc_auc_score(y_real, y_predict))]

	for name, m in metrics:
		df.loc[index, name] = m


