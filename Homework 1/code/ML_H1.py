### Machine Learning for Public Policy
### Assignment 1
### Héctor Salvador López

import csv
import pandas as pd
import requests
import matplotlib.pyplot as plt 

STUDENT_DATA = 'mock_student_data.csv'

def go(filename):
	# Gets descriptive statistics and histograms
	data = pd.read_csv(filename)
	data.describe()

	figure = plt.figure()
	data['Days_missed'].hist()
	plt.ylabel('Frequency')
	plt.title('Number of days missed')
	plt.savefig('days_hist')

	figure = plt.figure()
	data['GPA'].hist()
	plt.ylabel('Frequency')
	plt.title('GPA')
	plt.savefig('gpa_hist')

	figure = plt.figure()
	data['Age'].hist()
	plt.ylabel('Frequency')
	plt.title('Age')
	plt.savefig('age_hist')

	# Saves a csv with predicted gender from genderize API
	for i in range(len(data['Gender'])):
		if data['Gender'][i] not in ['Male', 'Female']:
			fname = data['First_name'][i]
			address = "https://api.genderize.io/?name=" + fname
			req = requests.get(address)
			pred_gender = req.json()['gender']
			data['Gender'][i] = pred_gender
	data.to_csv('filled_gender.csv')

	# For approaches A and B
	gpa_mean = data['GPA'].mean()
	gpa_grad_mean = data.groupby('Graduated').GPA.mean()['Yes']
	gpa_nograd_mean = data.groupby('Graduated').GPA.mean()['No']

	age_mean = data['Age'].mean()
	age_grad_mean = data.groupby('Graduated').Age.mean()['Yes']
	age_nograd_mean = data.groupby('Graduated').Age.mean()['No']

	days_mean = data['Days_missed'].mean()
	days_grad_mean = data.groupby('Graduated').Days_missed.mean()['Yes']
	days_nograd_mean = data.groupby('Graduated').Days_missed.mean()['No']

	dataA = data.copy()
	dataB = data.copy()

	for i in range(len(data['GPA'])):
		if not data['GPA'][i].is_integer():
			dataA['GPA'][i] = gpa_mean
			if data['Graduated'][i] == 'Yes':
				dataB['GPA'][i] = gpa_grad_mean
			elif data['Graduated'][i] == 'No':
				dataB['GPA'][i] = gpa_nograd_mean

	for i in range(len(data['Age'])):
		if not data['Age'][i].is_integer():
			dataA['Age'][i] = age_mean
			if data['Graduated'][i] == 'Yes':
				dataB['Age'][i] = age_grad_mean
			elif data['Graduated'][i] == 'No':
				dataB['Age'][i] = age_nograd_mean

	for i in range(len(data['Days_missed'])):
		if not data['Days_missed'][i].is_integer():
			dataA['Days_missed'][i] = days_mean
			if data['Graduated'][i] == 'Yes':
				dataB['Days_missed'][i] = days_grad_mean
			elif data['Graduated'][i] == 'No':
				dataB['Days_missed'][i] = days_nograd_mean

	dataA.to_csv('approach_A.csv')
	dataB.to_csv('approach_B.csv')

if __name__ == "__main__":
	go(STUDENT_DATA)
