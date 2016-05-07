### Machine Learning for Public Policy
### Homework 2
### Héctor Salvador López
### Code for the pipeline was significantly inspired on:
### 	/rayidghani/magicloops/blob/master/magicloops.py
### 	/BridgitD/Machine-Learning-Pipeline/blob/master/pipeline.py
### 	/danilito19/CAPP-ML-dla/blob/master/pa3/workflow.py
###		/ladyson/ml-for-public-policy/blob/master/PA3/pipeline.py
### 	/demunger/CAPP30254/blob/master/HW3/hw3.py
### 	/aldengolab/ML-basics/blob/master/pipeline/model.py


import math
import pandas as pd
from pipeline import reading, explore, preprocess, features, classify
from sklearn.cross_validation import train_test_split


def go(filename):
	# define features and label for this dataset
	fts = ['RevolvingUtilizationOfUnsecuredLines', 
            'age', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 
            'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 
            'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines', 
            'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']
	label = 'SeriousDlqin2yrs'

	# read dataset
	df = reading.read(filename)

	# divide dataset to train and test
	xtrain, xtest, ytrain, ytest = train_test_split(df[fts], df[label])
	train = xtrain.copy()
	train[label] = ytrain
	test = xtest.copy()
	test[label] = ytest
	df = train

	# generate statistics and generic exploration histograms
	explore.statistics_csv(df)
	# explore.plots_csv(df)
	# explore.crosstabs_csv(df, label, fts)

	# dive deeper into histograms

	# impute null values with mean value and transform income to log(income)
	preprocess.impute_csv(df)
	preprocess.transform_feature(df, 'MonthlyIncome', lambda x: math.log(x + 1))

	# create a feature of income quartile
	features.binning(df, 'f(MonthlyIncome)', 'quantiles', [0, 0.25, 0.5, 0.75, 1])

	# deploy classifiers
	models = ['LR', 'KNN', 'DT', 'SVM', 'RF', 'GB']
	results, models = classify.classify(df[fts], df[label], models, 3, 0.05)

