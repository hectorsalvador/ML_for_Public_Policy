### Machine Learning for Public Policy
### Homework 2
### Héctor Salvador López

import pipeline.reading
import pipeline.explore
import pipeline.preprocess
import pipeline.classify
import pipeline.evaluate
import pandas as pd
import matplotlib.pyplot as plt 

PROB_THRESH1 = 0.5
PROB_THRESH2 = 0.9

plt.style.use('ggplot')
pd.set_option('precision', 3)

def go(filename, data_type, plots = True):

	df = pipeline.reading.read(filename, data_type)
	print('Keys:\n' + '{}'.format(df.keys()) + '\n')
	df.columns = ['ID', 'SD2Y', 'RUUL', 'Age', 'LP30_59', 'DR', 'MI', 'OCLL',\
		'LP90_', 'LP60_90', 'MREL', 'Deps']

	# explore this dataset
	keys = [i for i in df.keys()]
	pipeline.explore.explore(df, data_type, plots)
	print('\n')
	pipeline.explore.gen_crosstabs(df, 1, [2,3,4,5,6,7,8,9,10,11])

	if plots:
		print('Check the current folder for scatters of these variables.')
		for key in keys:
			figure = plt.figure()
			df.plot(kind = 'scatter', x = keys[0], y = key)
			plt.xlabel('Observation')
			plt.title('{}'.format(key))
			plt.savefig('scatters/{}'.format(key) + '_scat')
			plt.close()

	# check for nulls and fill in
	print('Checking for nulls and replacing them.')
	pipeline.preprocess.preprocess(df,data_type, True, 'median')

	# no discretization or categorization of variables
	# logistic regression classification
	y = df[keys[1]]
	x = df[df.columns[2:]]
	yhat = pipeline.classify.logistic_regression(y, x, PROB_THRESH1)

	x2 = df[df.columns[3:]]
	yhat2 = pipeline.classify.logistic_regression(y, x2, PROB_THRESH1)

	yhat3 = pipeline.classify.logistic_regression(y, x2, PROB_THRESH2)

	# check how good the models are
	ac1 = pipeline.evaluate.accuracy(y, yhat)
	ac2 = pipeline.evaluate.accuracy(y, yhat2)
	ac3 = pipeline.evaluate.accuracy(y, yhat3)

	print('Accuracy model 1: {}'.format(ac1))
	print('Accuracy model 2: {}'.format(ac2))
	print('Accuracy model 3: {}'.format(ac3))

	return df

'''
if __name__ == "__main__":
	num_args = len(sys.argv)

	if num_args != 3:
		print('usage: python3 ' + sys.argv[0] + '<path of data file>' + \
			'<type of data file (e.g. "csv")>')
		sys.exit[0]
	else:
		go(sys.argv[1], sys.argv[2])
'''