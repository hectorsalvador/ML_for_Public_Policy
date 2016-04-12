### Machine Learning for Public Policy
### Pipeline: Reading
### Héctor Salvador López

import csv
import json
import pandas as pd 

def read(filename, data_type = 'csv'):
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
		df = read_csv(filename)
	elif data_type == 'json':
		df = read_json(filename)

	return df


def read_csv(filename):
	df = pd.read_csv(filename)
	return df

def read_json(filename):
	with open(filename) as data:
		df = json.load(data) 
	return df