import sys
import numpy as np

def loadData():
	data = np.loadtxt('./data/letterSet/letter-recognition.data', dtype='float32', delimiter=',',converters={0:lambda ch:ord(ch)-ord('A')})

	#split dataset into train_set and test_set
	train_set, test_set = np.vsplit(data, 2)
	#split train_set and test_set into label and features
	train_labels, train_features = np.hsplit(train_set, [1])
	test_labels, test_features = np.hsplit(test_set, [1])

	return train_labels, train_features, test_labels, test_features
