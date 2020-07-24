import sys
import numpy as np
import scipy

####################################################################

def function_H(X, W):

	arr = np.exp(np.dot(X, W))

	arr_sum = np.sum(arr, axis=1)
	arr = arr/arr_sum.reshape((arr_sum.shape[0],1))

	return (arr)

def function_L(W, X, Y):

	arr = np.multiply( Y, np.log(function_H(X,W)) ) 
	sum = np.sum(arr)

	return ((-1)*sum/X.shpae[0])

####################################################################

train_data_raw = open(sys.argv[1], 'rt')
train_data = np.loadtxt(train_data_raw, dtype = 'str', delimiter=",")

test_data_raw = open(sys.argv[2], 'rt')
test_data = np.loadtxt(test_data_raw, dtype = 'str', delimiter=",")

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

test = np.zeros((test_data.shape[0], 27))

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

for i in range (0, test_data.shape[0]):

	j = 0
	index = 0

	while (j < 8):

		if (test_data[i][j] == feature_labels[index]):

			test[i][index] = 1
			j = j+1

		index = index+1

test1 = np.ones((test_data.shape[0],1))
test_f = np.hstack((test1, test))

####################################################################

param_raw = open(sys.argv[3],'rt')
param = np.loadtxt(param_raw, dtype = 'str', delimiter="\n")

W = np.zeros((28, 5))

####################################################################

if (float(param[0]) == 1.0):

	# Constant Learning Rate

	batch_size = int(param[3])

	batches = int(train_data.shape[0]/batch_size)
	
	rate = float(param[1])

	max_itr = float(param[2])

	itr = 0

	while(itr <= max_itr):

		for i in range (0, batches):

			mini_X = X_f[i*batch_size : (i+1)*batch_size, :]
			mini_Y = Y[i*batch_size : (i+1)*batch_size, :]

			derivative = np.dot( mini_X.transpose() , (function_H(mini_X, W) - mini_Y) )
		
			W = W - np.multiply(derivative, rate/batch_size) 
		
		itr = itr+1

####################################################################

elif (float(param[0]) == 2.0):

	# Adaptive Learning Rate

	batch_size = float(param[3])

	batches = int(train_data.shape[0]/batch_size)
	
	seed_rate = float(param[1])

	max_itr = float(param[2])

	itr = 0

	while(itr <= max_itr):

		itr = itr+1

		rate = seed_rate/np.sqrt(itr)

		for i in range (0, batches):
			
			mini_X = X_f[i*batch_size : (i+1)*batch_size, :]
			mini_Y = Y[i*batch_size : (i+1)*batch_size, :]

			derivative = np.dot( mini_X.transpose() , (function_H(mini_X, W) - mini_Y) )
		
			W = W - np.multiply(derivative, rate/batch_size) 


####################################################################

else:

	# Alpha - Beta Backtracking

	batch_size = float(param[3])

	batches = int(train_data.shape[0]/batch_size)

	values = param[1].split(',')

	alpha = float(values[0])
	beta = float(values[1])

	max_itr = float(param[2])

	itr = 0

	while(itr <= max_itr):

		itr = itr + 1

		for i in range (0, batches):

			mini_X = X_f[i*batch_size : (i+1)*batch_size, :]
			mini_Y = Y[i*batch_size : (i+1)*batch_size, :]

			rate = 1

			derivative = np.dot( mini_X.transpose() , (function_H(mini_X, W) - mini_Y) )
			function = function_L(W, mini_X, mini_Y)

			while( function_L( W - np.multiply(rate, derivative), mini_X, mini_Y) > function - rate*alpha*(np.linalg.norm(derivative,2))):
				rate = rate * beta
				
			W = W - np.multiply(derivative, rate/(train_data.shape[0])) 


####################################################################

print(max_itr)
np.savetxt(sys.argv[5], W, delimiter = ',')

o = open(sys.argv[4], 'w')
sys.stdout = o

answer = np.dot(test_f, W)

for i in range (0,test_data.shape[0]):

	print (answer_labels[ np.argmax(answer[i,:]) ])

####################################################################
