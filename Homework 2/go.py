### Machine Learning for Public Policy
### Pipeline
### Héctor Salvador López

def go(filename, data_type):

if __name__ == "__main__":
	num_args = len(sys.argv)

	if num_args != 3:
		print('usage: python3 ' + sys.argv[0] + '<path of data file>' + \
			'<type of data file (e.g. "csv")>')
		sys.exit[0]
	else:
		go(sys.argv[1], sys.argv[2])