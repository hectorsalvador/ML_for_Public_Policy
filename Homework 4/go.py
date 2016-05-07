### Machine Learning for Public Policy
### Homework 2
### Héctor Salvador López

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
	models = ['LR', 'KNN', 'DT', 'SVM', 'RF', 'BOO', 'BAG']
	classify.classify(df[fts], df[label], models, 0.05)

