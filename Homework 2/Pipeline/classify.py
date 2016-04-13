### Machine Learning for Public Policy
### Pipeline: Build classifier
### Héctor Salvador López

import statsmodels.api as sm

def logistic_regression(y, x, threshold):
	'''
	Takes:
		y, an array with dependent variable observations
		x, an array with independent variable(s) observations
		Note that y and x must be the same size

	Returns:
		a statsmodels.discrete.discrete_model.Logit object corresponding to
			the logit regression of x on y
	'''
	logit = sm.Logit(y, x)
	result = logit.fit()
	print(result.summary2())
	predicted = result.predict(x)
	yhat = []

	for j in predicted:
		if j > threshold:
			yhat.append(1)
		else:
			yhat.append(0)

	return yhat

def linear_regression(data):
	pass