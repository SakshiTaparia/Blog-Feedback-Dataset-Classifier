import sys
import numpy as np
import scipy

import time

####################################################################

def function_H(X, W):

	arr = np.exp(np.dot(X, W))

	arr_sum = np.sum(arr, axis=1)
	arr = arr/arr_sum.reshape((arr_sum.shape[0],1))

	return (arr)

def function_L(W, X, Y):

	arr = np.multiply( Y, np.log(function_H(X,W)) ) 
	sum = np.sum(arr)

	return (sum/X.shape[0])

####################################################################

train_data_raw = open(sys.argv[1], 'rt')
train_data = np.loadtxt(train_data_raw, dtype = 'str', delimiter=",")

feature_labels = ['usual', 'pretentious', 'great_pret', 
'proper', 'less_proper', 'improper', 'critical', 'very_crit', 
'complete', 'completed', 'incomplete', 'foster',
'1', '2', '3', 'more',
'convenient', 'less_conv', 'critical',
'convenient', 'inconv',
'nonprob', 'slightly_prob', 'problematic',
'recommended', 'priority', 'not_recom']

answer_labels = ['not_recom', 'recommend', 'very_recom', 'priority', 'spec_prior']

X = np.zeros((train_data.shape[0], 27))
Y = np.zeros((train_data.shape[0], 5))

for i in range (0, train_data.shape[0]):

	j = 0
	index = 0

	while (j < 8):

		if (train_data[i][j] == feature_labels[index]):

			X[i][index] = 1
			j = j+1

		index = index+1

for i in range (0, train_data.shape[0]):

	for j in range (0,5):

		if (train_data[i][8] == answer_labels[j]):

			Y[i][j] = 1

X1 = np.ones((train_data.shape[0],1))
X_f = np.hstack((X1,X))

####################################################################

rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

batch_size = [6000, 3000, 2000, 1000, 500, 100, 50, 10]

max_itr = 10000

print("Constant Learning Rate")
print("__________________")

####################################################################

for r in (rate):

	for b in (batch_size):

		W = np.zeros((28, 5))

		itr = 0

		start = time.time()

		while(itr <= max_itr):

			batches = int(train_data.shape[0]/b)

			for i in range (0, batches):

				mini_X = X_f[i*b : (i+1)*b, :]
				mini_Y = Y[i*b : (i+1)*b, :]

				derivative = np.dot( mini_X.transpose() , (mini_Y - function_H(mini_X, W)) )
		
				W = W + np.multiply(derivative, r/b) 
		
			itr = itr+1

		print(r)
		print(b)
		print(time.time() - start)
		print(function_L(W, X_f, Y))
		print("__________________")


####################################################################

