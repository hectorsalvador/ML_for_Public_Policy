### Machine Learning for Public Policy
### Pipeline: Generate features
### Héctor Salvador López

### Functions for dataframes


def binning_feature(data, feature, type_cut, num_bins):
	'''
	Assumes a that a pandas dataframe is passed.

	Takes:
		data, a pandas dataframe that should be discretized, this procedure 
			will throw observations into "bins"
		feature
		num_bins
	'''
	valid_cuts = ['q', 'n']
	assert type_cut in valid_cuts

	if type_cut == 'q':
		bins = pd.qcut(data[feature], num_bins, labels=False)
	elif type_cut == 'n':
		bins = pd.cut(data[feature], num_bins, labels=False)

	df = pd.concat([data, bins], axis = 1)
	
	return df

def binarize_feature(data, feature):
	'''
	Assumes that a pandas dataframe is passed.

	Takes:
		data, a pandas dataframe 
		feature,
	
	'''
	for value in [i for i in set(data[feature].values)][:-1]:
		data[feature + "=" + str(value)] = data[feature] == value

	# dfnp = pd.DataFrame(np.random.rand(df11.size, len(values)))
	# dfnp.columns = [i for i in values]

