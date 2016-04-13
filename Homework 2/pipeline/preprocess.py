### Machine Learning for Public Policy
### Pipeline: Pre-process data
### Héctor Salvador López

def preprocess(data, data_type, fill_nulls, fill_type, conditional_var = None):
	'''
	Takes:
		data, a data structure
		data_type, a string indicating the type of file from which the data
		structure was created
		fill_nulls, a boolean indicating if the nulls should be filled 
		conditional_var, when selecting a conditional mean, this is an int
			indicating the column of the conditional variable 

	Changes nulls in site
	'''
	fill_types = ['mean', 'median', 'cmean']
	assert fill_type in fill_types

	if data_type == 'csv':
		if fill_nulls:
			for key in [i for i in data.keys()]:
				has_nulls = data[key].isnull().values.any()
				print('{} has null values: {}.'.format(key, has_nulls))
				if has_nulls:
					print('	Changing null values for {} value.'.format(fill_type))
					fill_col = fill_csv_nulls(data[key], fill_type, \
						conditional_var)
					data[key] = fill_col
				print('\n')

	elif data_type == 'json':
		pass


def fill_csv_nulls(data, fill_type, conditional_var = None):
	'''
	Assumes that a pandas dataframe is passed.

	Takes,
		data, a pandas dataframe
		fill_type, a string indicating a method to fill missing values
		conditional_var, when selecting a conditional mean, this is an int
			indicating the column of the conditional variable  

	'''
	if fill_type == 'mean':
		fill = data.mean()

	elif fill_type == 'median':
		fill = data.median()

	elif fill_type == 'cmean':
		pass

	return data.fillna(fill)

