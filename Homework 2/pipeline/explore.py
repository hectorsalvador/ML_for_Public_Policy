### Machine Learning for Public Policy
### Pipeline: Explore
### Héctor Salvador López

import pandas as pd
import matplotlib.pyplot as plt 
plt.style.use('ggplot')

def explore(data, data_type, plots):
	'''
	Takes:
		filename, a string with the path to the data file
		data_type, a string with the type of file in which the data is saved
		(for example: "csv", "json")
	
	Returns:
		a data structure
	'''
	data_types_all = ['csv', 'json']
	assert data_type in data_types_all

	if data_type == 'csv':
		explore_csv(data, plots)
	elif data_type == 'json':
		pass

def explore_csv(data, plots):
	'''
	Takes:
		data, a pd.dataframe 

	Prints:
		keys of the df
		first five observations
		number of observations 
		descriptive statistics
	
	Generates histograms in a separate folder

	'''
	print('Keys:\n' + '{}'.format(data.keys()) + '\n')
	print('Sample observations:\n' + '{}'.format(data.head()) + '\n')
	print('Observations:\n' + '{}'.format(data.size) + '\n')
	print('Descriptive statistics:\n' + '{}'.format(data.describe()) + '\n')
	print('Var-Cov matrix:\n' + '{}'.format(data.cov()) + '\n')

	if plots:
		print('Check the current folder for default histograms of these variables.')
		for variable in data.describe().keys():
			figure = plt.figure()
			data[variable].plot.hist()
			plt.ylabel('Frequency')
			plt.title('{}'.format(variable))
			plt.savefig('histograms/{}'.format(variable) + '_hist')
			plt.close()


def gen_crosstabs(data, categorical, covariates):
	'''
	Takes:
		data, a pd.dataframe
		categorical, an int indicating a column with a categorical variable
		covariates, a list of ints with the column numbers of covariates

	Prints crosstabs of desired variables.
	'''
	keys = [i for i in data.keys()]
	for col in covariates:
		print('Crosstab table for {} and {}:'.format(keys[categorical],\
			keys[col]))
		print('{}'.format(pd.crosstab(data[keys[categorical]], \
			data[keys[col]])) + '\n')

def explore_json(data):
	pass
		