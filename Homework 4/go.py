### Machine Learning for Public Policy
### Homework 2
### Héctor Salvador López

import pandas as pd
import pipeline.reading as rdn
import pipeline.explore as exp
import pipeline.preprocess as prp
import pipeline.classify as cls
import pipeline.evaluate as evl
import matplotlib.pyplot as plt 

PROB_THRESH1 = 0.5
PROB_THRESH2 = 0.9

plt.style.use('ggplot')
pd.set_option('precision', 3)

def go(filename, dtype):
	df = rdn.read(filename, dtype)
	






def train(filename, data_type, plots = True):

	
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
	reg1 = pipeline.classify.logistic_regression(y, x, PROB_THRESH1)
	yhat1 = reg1[0]
	r1 = reg1[1]

	x2 = df[df.columns[3:]]
	reg2 = pipeline.classify.logistic_regression(y, x2, PROB_THRESH1)
	yhat2 = reg2[0]
	r2 = reg2[1]

	reg3 = pipeline.classify.logistic_regression(y, x2, PROB_THRESH2)
	yhat3 = reg3[0]
	r3 = reg3[1]

	# check how good the models are
	ac1 = pipeline.evaluate.accuracy(y, yhat1)
	ac2 = pipeline.evaluate.accuracy(y, yhat2)
	ac3 = pipeline.evaluate.accuracy(y, yhat3)

	print('Accuracy model 1: {}'.format(ac1))
	print('Accuracy model 2: {}'.format(ac2))
	print('Accuracy model 3: {}'.format(ac3))

	models = [r1, r2, r3]

	return r2

def test(filename, model):
	df = pipeline.reading.read(filename, 'csv')
	keys = [i for i in df.keys()]
	x = df[df.columns[3:]]
	predicted = model.predict(x)
	txt = open('Fitted.txt', 'w')

	for j in predicted:
		if j > PROB_THRESH2:
			txt.write('{} \n'.format(1))
		else:
			txt.write('{} \n'.format(0))


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